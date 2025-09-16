cmake_minimum_required(VERSION 3.18)
project(GPU_Transfer_Benchmark LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xcompiler -Wall")

# 设置CUDA架构 (简化版本，避免引号问题)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# 包含目录
include_directories(include)

# 添加源文件
set(SOURCES
    src/common.cpp
    src/dma_transfer.cu
    src/thread_transfer.cu
    src/transfer_benchmark.cpp
    src/main.cu
)

# 创建可执行文件
add_executable(cuda_transfer_benchmark ${SOURCES})

# 链接CUDA库
target_link_libraries(cuda_transfer_benchmark ${CUDA_LIBRARIES})

# 设置CUDA分离编译
set_property(TARGET cuda_transfer_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 设置输出目录
set_target_properties(cuda_transfer_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 安装规则
install(TARGETS cuda_transfer_benchmark
    RUNTIME DESTINATION bin
)

# 打印配置信息
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")