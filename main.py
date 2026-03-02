import subprocess
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

    # 编译 CUDA Kernel
    compile_cmd = "nvcc -O3 ./src/nf4_kernel.cu -o nf4_kernel"
    run_command(compile_cmd, "编译 C++ CUDA 内核")

    # 生成测试数据
    run_command("uv run ./src/generate_data.py", "生成随机量化权重")

    # 运行 C++ 反量化
    run_command("./nf4_kernel", "运行自定义 CUDA 内核")

    # 运行对比分析脚本
    run_command("uv run ./src/benchmark.py", "对比精度与性能表现")

if __name__ == "__main__":
    main()