import torch
import struct
import numpy as np
import os
from bitsandbytes import functional as F

class MockQuantState:
    def __init__(self, absmax, code, blocksize=256):
        self.absmax = absmax      # 二级缩放因子
        self.code = code          # 二级码表
        self.blocksize = blocksize
        self.nested = None
        self.offset = 0.0
        self.dtype = torch.float32

def dequantize(input_bin="./data/qlora_test.bin", output_bin="./data/py_output.bin", 
               time_file="./data/log/log_py.txt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 读取二进制文件与对齐计算
    with open(input_bin, "rb") as f:
        # 读取 Header
        header = f.read(20)
        rows, cols, blocksize = struct.unpack("<qqi", header)
        
        total_original = rows * cols
        groupsize = 256
        alignment = blocksize * groupsize
        
        # 向上取整计算对齐后的总元素量
        total_aligned = ((total_original + alignment - 1) // alignment) * alignment
        
        num_blocks = total_aligned // blocksize
        num_groups = total_aligned // (blocksize * groupsize)
        
        print(f"原始维度: {rows}x{cols} ({total_original})")
        print(f"对齐维度: {total_aligned} (Padding: {total_aligned - total_original})")

        # 顺序读取对齐后的数据
        # 权重 (4-bit packed): 长度为总元素数的一半
        packed_data = np.frombuffer(f.read(total_aligned // 2), dtype=np.uint8)
        packed_tensor = torch.from_numpy(packed_data).to(device)

        # 一级索引 (uint8): 长度为 Block 数
        absmax_q_data = np.frombuffer(f.read(num_blocks), dtype=np.uint8)
        absmax_q_tensor = torch.from_numpy(absmax_q_data).to(device)

        # 二级缩放因子 (FP16): 长度为 Group 数
        absmax2_data = np.frombuffer(f.read(num_groups * 2), dtype=np.float16)
        absmax2_tensor = torch.from_numpy(absmax2_data).to(torch.float32).to(device)

        # 二级码表 (FP16): 固定 256 个
        code2_data = np.frombuffer(f.read(512), dtype=np.float16)
        code2_tensor = torch.from_numpy(code2_data).to(torch.float32).to(device)

        offset = struct.unpack("<f", f.read(4))[0]


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()

    # 二级反量化
    nested_state = MockQuantState(absmax2_tensor, code2_tensor, blocksize=groupsize)
    absmax_1 = F.dequantize_blockwise(absmax_q_tensor, quant_state=nested_state)
    print(absmax_1)
    if offset != 0:
        absmax_1 += offset

    # 一级反量化
    _, state = F.quantize_4bit(
        torch.zeros(1, 1, device=device), 
        quant_type='nf4', 
        compress_statistics=False 
    )
    
    state.absmax = absmax_1
    state.shape = torch.Size([1, total_aligned]) 
    state.dtype = torch.bfloat16
    state.blocksize = blocksize
    
    dequantized_padded = F.dequantize_4bit(packed_tensor, quant_state=state)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)


    # 读取量: Packed(int4) + Indices(int8) + Scales(fp16) + Code(fp16)
    read_bytes = packed_tensor.numel() * 1 + absmax_q_tensor.numel() * 1 + absmax2_tensor.numel() * 2 + 512
    # 写入量: Output(bf16)
    write_bytes = dequantized_padded.numel() * 2 
    
    total_bytes = read_bytes + write_bytes
    gb_per_sec = (total_bytes / 1e9) / (elapsed_ms / 1000.0)

    print(f"耗时: {elapsed_ms:.4f} ms")
    print(f"吞吐: {gb_per_sec:.2f} GB/s")
    
    # 保存时间与带宽: "耗时(ms),带宽(GB/s)"
    with open(time_file, "w") as f:
        f.write(f"{elapsed_ms},{gb_per_sec}")

    # 结果裁剪与保存
    out_tensor = dequantized_padded.view(-1)[:total_original]
    out_tensor = out_tensor.view(rows, cols).to(torch.bfloat16).cpu()
    raw_bytes = out_tensor.view(torch.int16).numpy().tobytes()
    
    with open(output_bin, "wb") as f:
        f.write(raw_bytes)
        
    print(f"数据已保存: {output_bin}")

if __name__ == "__main__":
    dequantize()