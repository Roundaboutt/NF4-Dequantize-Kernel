#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at line " << __LINE__ << std::endl;           \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

void read_file(
    const std::string file_path, std::vector<uint8_t>& packed_weights, 
    std::vector<uint8_t>& absmax_q, 
    std::vector<uint16_t>& absmax2, 
    std::vector<uint16_t>& code2, float& offset, 
    int64_t& rows, int64_t& cols, int32_t& blocksize
) 
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file) { std::cerr << "无法打开文件" << std::endl; exit(1); }

    file.read(reinterpret_cast<char*>(&rows), 8);
    file.read(reinterpret_cast<char*>(&cols), 8);
    file.read(reinterpret_cast<char*>(&blocksize), 4);

    int64_t total_original = rows * cols;
    int32_t groupsize = 256; 
    int32_t alignment = blocksize * groupsize;
    int64_t total_aligned = ((total_original + alignment - 1) / alignment) * alignment;
    
    size_t num_blocks = total_aligned / blocksize;
    size_t num_groups = total_aligned / (blocksize * groupsize);

    std::cout << "原始元素: " << total_original << std::endl;
    std::cout << "对齐元素: " << total_aligned << std::endl;

    packed_weights.resize(total_aligned / 2);
    absmax_q.resize(num_blocks);
    absmax2.resize(num_groups);
    code2.resize(256);

    file.read(reinterpret_cast<char*>(packed_weights.data()), packed_weights.size());
    file.read(reinterpret_cast<char*>(absmax_q.data()), absmax_q.size());
    file.read(reinterpret_cast<char*>(absmax2.data()), absmax2.size() * 2);
    file.read(reinterpret_cast<char*>(code2.data()), 256 * 2);
    file.read(reinterpret_cast<char*>(&offset), 4);
}

float half2float(uint16_t h) 
{
    uint8_t h_exp = (h >> 10) & 0x1F;
    uint8_t h_sig = h & 0x3FF;
    uint8_t h_sgn = (h >> 15) & 0x1;
    uint32_t f_sgn = (uint32_t)h_sgn << 31;
    uint32_t f_exp, f_sig;

    if (h_exp == 0) 
    {
        if (h_sig == 0) return (h_sgn ? -0.0f : 0.0f);
        f_exp = 0; h_sig <<= 1;
        while ((h_sig & 0x400) == 0) { h_sig <<= 1; f_exp++; }
        f_exp = 127 - 15 - f_exp;
        f_sig = (h_sig & 0x3FF) << 13;
    } 
    else if (h_exp == 0x1F) 
    {
        f_exp = 0xFF; f_sig = (uint32_t)h_sig << 13;
    } 
    else 
    {
        f_exp = h_exp + (127 - 15);
        f_sig = (uint32_t)h_sig << 13;
    }
    uint32_t f_bits = f_sgn | (f_exp << 23) | f_sig;
    return *reinterpret_cast<float*>(&f_bits);
}

__constant__ float c_nf4_table[16] = {
-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, 
-0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 
0.07958029955625534, 0.16093020141124725,0.24611230194568634, 0.33791524171829224, 
0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
};

__constant__ float c_code2[256];

__global__ void dequantize_nf4_kernel
(
    const uint8_t*  __restrict__ packed_weights, 
    const uint8_t*  __restrict__ absmax_q,       
    const half*     __restrict__ absmax2,
    uint32_t*       __restrict__ output_packed,       
    int num_bytes,
    int block_size,
    int group_size,
    float offset
) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_bytes) return;

    int element_idx = tid * 2;
    int block_idx = element_idx / block_size;
    int group_idx = block_idx / group_size; 

    float scale_1 = c_code2[absmax_q[block_idx]] + offset; 
    float scale_2 = __half2float(absmax2[group_idx]); 
    float final_scale = scale_1 * scale_2;

    uint8_t byte_val = packed_weights[tid]; 
    uint8_t idx_0 = byte_val >> 4;          
    uint8_t idx_1 = byte_val & 0x0F;        

    float v0_fp32 = c_nf4_table[idx_0] * final_scale;
    float v1_fp32 = c_nf4_table[idx_1] * final_scale;
    
    __nv_bfloat16 v0_bf16 = __float2bfloat16(v0_fp32);
    __nv_bfloat16 v1_bf16 = __float2bfloat16(v1_fp32);

    uint16_t bits_0 = *reinterpret_cast<unsigned short*>(&v0_bf16);
    uint16_t bits_1 = *reinterpret_cast<unsigned short*>(&v1_bf16);
    
    output_packed[tid] = ((uint32_t)bits_1 << 16) | (uint32_t)bits_0;
}

void nf4_dequantize_cuda
(
    std::vector<uint8_t>& h_packed_weights, 
    std::vector<uint8_t>& h_absmax_q,
    std::vector<uint16_t>& h_absmax2, 
    std::vector<uint16_t>& h_code2,
    int64_t rows, int64_t cols, int32_t blocksize, int32_t goupsize,float offset
)
{
    size_t num_bytes = h_packed_weights.size();
    size_t out_size = num_bytes * sizeof(uint32_t);

    float h_code2_f32[256];
    for(int i = 0; i < 256; ++i) h_code2_f32[i] = half2float(h_code2[i]);
    CHECK_CUDA(cudaMemcpyToSymbol(c_code2, h_code2_f32, sizeof(h_code2_f32)));

    uint8_t *d_packed, *d_absmax_q;
    half *d_absmax2;
    uint32_t *d_output;

    CHECK_CUDA(cudaMalloc(&d_packed, h_packed_weights.size()));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, h_absmax_q.size()));
    CHECK_CUDA(cudaMalloc(&d_absmax2, h_absmax2.size() * 2));
    CHECK_CUDA(cudaMalloc(&d_output, out_size));

    CHECK_CUDA(cudaMemcpy(d_packed, h_packed_weights.data(), h_packed_weights.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), h_absmax_q.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2, h_absmax2.data(), h_absmax2.size() * 2, cudaMemcpyHostToDevice));

    int threadsPerBlock = 1024;
    int blocksPerGrid = (num_bytes + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    
    cudaEventRecord(start); // 开始记录
    dequantize_nf4_kernel<<<blocksPerGrid, threadsPerBlock>>>
    (d_packed, d_absmax_q, d_absmax2, d_output, num_bytes, blocksize, goupsize, offset);
    cudaEventRecord(stop);  // 结束记录
    
    CHECK_CUDA(cudaEventSynchronize(stop)); // 等待 Event 完成
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // 计算带宽
    // 读取: Packed(1) + Indices(1) + Scales(2) + Code2(忽略不计)
    size_t total_read = h_packed_weights.size() + h_absmax_q.size() + h_absmax2.size() * 2;
    // 写入: Output(4) (因为每个packed byte生成一个uint32)
    size_t total_write = num_bytes * 4;
    
    double total_bytes = (double)(total_read + total_write);
    double gb_per_sec = (total_bytes / 1e9) / (milliseconds / 1000.0);

    std::cout << "Kernel 耗时: " << milliseconds << " ms" << std::endl;
    std::cout << "有效带宽: " << gb_per_sec << " GB/s" << std::endl;

    // 保存时间与带宽
    std::ofstream timefile("./data/log/log_cpp.txt");
    timefile << milliseconds << "," << gb_per_sec;
    timefile.close();

    std::vector<uint32_t> h_output(num_bytes);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, out_size, cudaMemcpyDeviceToHost));

    const std::string output_path = "./data/cpp_output.bin";
    std::ofstream outfile(output_path, std::ios::binary);
    outfile.write(reinterpret_cast<char*>(h_output.data()), out_size);
    outfile.close();

    cudaFree(d_packed); cudaFree(d_absmax_q); cudaFree(d_absmax2); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() 
{
    const std::string file_path = "./data/qlora_test.bin";
    std::vector<uint8_t> packed_weights, absmax_q;
    std::vector<uint16_t> absmax2, code2;
    float offset;
    int64_t rows, cols;
    int32_t blocksize;
    int32_t groupsize = 256;
    read_file(file_path, packed_weights, absmax_q, absmax2, code2, offset, rows, cols, blocksize);
    nf4_dequantize_cuda(packed_weights, absmax_q, absmax2, code2, rows, cols, blocksize, groupsize, offset);

    return 0;
}