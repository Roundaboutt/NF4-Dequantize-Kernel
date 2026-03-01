import torch
from bitsandbytes import functional as F
import struct
import numpy as np
import os
import math
from get_param import parse_params

def generate_data(config_file="./data/param.txt"):
    if not os.path.exists("./data"): os.makedirs("./data")
    
    # 获取动态配置
    config = parse_params(config_file)
    blocksize = config["blocksize"]
    groupsize = config["groupsize"]
    rows = config["rows"]
    cols = config["cols"]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if config["compute_type"] == "bf16" else torch.float16
    
    print(f"原始维度: {rows} x {cols} | Block: {blocksize} | Group: {groupsize}")
    
    original_weights = torch.randn(rows, cols, dtype=dtype, device=device) * 0.5
    total_elements = rows * cols
    alignment = blocksize * groupsize
    
    # 计算需要填充的数量
    padded_total = math.ceil(total_elements / alignment) * alignment
    padding_needed = padded_total - total_elements
    
    if padding_needed > 0:
        print(f"填充 {padding_needed} 个元素以对齐到 {padded_total}")
        # 展平并填充零
        flat_weights = original_weights.reshape(-1)
        padding = torch.zeros(padding_needed, dtype=dtype, device=device)
        aligned_weights = torch.cat([flat_weights, padding], dim=0)
    else:
        aligned_weights = original_weights.reshape(-1)
        padded_total = total_elements

    # 3. 对齐后的量化过程
    # 注意：我们将对齐后的 1D 张量视为 (1, padded_total) 的 2D 张量送入量化
    # 这样可以确保 bitsandbytes 内部的分块逻辑与我们的对齐完全匹配
    aligned_weights = aligned_weights.reshape(1, -1)
    
    # 一级量化 (NF4)
    q_weight, state_1 = F.quantize_4bit(
        aligned_weights,
        blocksize=blocksize,
        quant_type='nf4'
    )
    packed_np = q_weight.cpu().numpy()
    
    # 二级量化 (FP8)
    absmax_1 = state_1.absmax
    absmax_q, state_2 = F.quantize_blockwise(absmax_1, blocksize=groupsize, nested=False)
    
    absmax_q_np = absmax_q.cpu().numpy()
    absmax2_np = state_2.absmax.to(torch.float16).cpu().numpy()

    # 构造线性码表
    code2_final = np.zeros(256, dtype=np.float16)
    for i in range(256):
        val_int8 = i if i < 128 else i - 256
        code2_final[i] = val_int8 / 127.0

    offset_val = 0.0

    # 写入二进制文件
    bin_path = "./data/qlora_test.bin"
    with open(bin_path, "wb") as f:
        # Header 依然保存原始的 rows 和 cols，用于 test.py 最终裁剪
        # 但 C++ Kernel 将会根据这个 rows*cols 自动推算出对齐后的读取长度
        f.write(struct.pack("<qqi", rows, cols, blocksize)) 
        
        f.write(packed_np.tobytes())
        f.write(absmax_q_np.tobytes()) 
        f.write(absmax2_np.tobytes())
        f.write(code2_final.tobytes())
        f.write(struct.pack("<f", offset_val))
        
    print(f"对齐数据生成完毕: {bin_path}")
    print(f"实际写入元素量: {padded_total}")
    print(f"一级块数 (Blocks): {absmax_q_np.size}")
    print(f"二级组数 (Groups): {absmax2_np.size}")

if __name__ == "__main__":
    generate_data()