#!/bin/bash

set -e

echo "CUDA Transfer Benchmark Build Script"
echo "====================================="

# 检查CUDA安装
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not found. Please install CUDA toolkit."
    exit 1
fi

# 显示CUDA版本
echo "CUDA Version:"
nvcc --version | grep "release"

# 检查GPU
if ! nvidia-smi &> /dev/null; then
    echo "Warning: No NVIDIA GPU detected or driver not installed."
else
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
fi

echo ""

# 创建构建目录
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# 选择构建系统
BUILD_SYSTEM=${1:-"cmake"}

if [ "$BUILD_SYSTEM" = "cmake" ] && command -v cmake &> /dev/null; then
    echo "Building with CMake..."
    cmake ..
    make -j$(nproc)
    echo ""
    echo "Build completed: ./cuda_transfer_benchmark"
elif [ "$BUILD_SYSTEM" = "make" ] || [ ! command -v cmake &> /dev/null ]; then
    echo "Building with Makefile..."
    cd ..
    make clean
    make -j$(nproc)
    echo ""
    echo "Build completed: ./bin/cuda_transfer_benchmark"
else
    echo "Error: Unknown build system: $BUILD_SYSTEM"
    echo "Usage: $0 [cmake|make]"
    exit 1
fi

echo ""
echo "To run the benchmark:"
if [ -f "cuda_transfer_benchmark" ]; then
    echo "  ./cuda_transfer_benchmark"
elif [ -f "../bin/cuda_transfer_benchmark" ]; then
    echo "  ./bin/cuda_transfer_benchmark"
fi
echo ""
echo "For help: --help"
echo "Quick test: --iterations 3 --warmup 1 --test-types h2d,d2d"