import subprocess
import os
import sys

def run_command(command, description):
    print(f"--- 正在执行: {description} ---")
    try:
        # 使用 check=True 确保命令失败时抛出异常
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"错误: {description} 失败！")
        sys.exit(1)

def main():
    # 0. 确保数据目录存在
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # 1. 编译 CUDA Kernel
    # 使用 -O3 开启最大性能优化，-lcudart 链接 CUDA 运行时库
    compile_cmd = "nvcc -O3 ./src/nf4_kernel.cu -o nf4_kernel -lcudart"
    run_command(compile_cmd, "编译 C++ CUDA 内核")

    # 2. 生成测试数据
    run_command("uv run ./src/generate_data.py", "生成随机量化权重")

    # 3. 运行 Python 反量化 (Baseline)
    run_command("uv run ./src/dq_nf4.py", "运行 Python (bitsandbytes) 基准测试")

    # 4. 运行 C++ 反量化 (Custom Kernel)
    # 注意：根据你代码里的相对路径调整执行位置
    run_command("./nf4_kernel", "运行自定义 CUDA 内核")

    # 5. 运行对比分析脚本
    run_command("uv run ./src/benchmark.py", "对比精度与性能表现")

if __name__ == "__main__":
    main()