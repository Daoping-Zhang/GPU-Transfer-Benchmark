#include "thread_transfer.h"
#include <algorithm>

// H2D Kernels - 从主机内存读取并写入设备内存
__global__ void h2d_basic_kernel(int* d_dst, const int* h_src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        d_dst[i] = h_src[i];
    }
}

__global__ void h2d_vectorized_kernel(uint4* d_dst, const uint4* h_src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        d_dst[i] = h_src[i];
    }
}

// D2H Kernels - 从设备内存读取并写入主机内存
__global__ void d2h_basic_kernel(int* h_dst, const int* d_src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        h_dst[i] = d_src[i];
    }
}

__global__ void d2h_vectorized_kernel(uint4* h_dst, const uint4* d_src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        h_dst[i] = d_src[i];
    }
}

// D2D Kernels - 设备到设备拷贝
__global__ void d2d_basic_kernel(int* d_dst, const int* d_src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        d_dst[i] = d_src[i];
    }
}

__global__ void d2d_vectorized_kernel(uint4* d_dst, const uint4* d_src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        d_dst[i] = d_src[i];
    }
}

// 共享内存优化的D2D kernel
__global__ void d2d_shared_memory_kernel(int* d_dst, const int* d_src, size_t count) {
    __shared__ int shared_buffer[256]; // 与block size匹配
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t base = blockIdx.x * blockDim.x; base < count; base += stride) {
        size_t local_idx = threadIdx.x;
        size_t global_idx = base + local_idx;
        
        // 协作加载到共享内存
        if (global_idx < count) {
            shared_buffer[local_idx] = d_src[global_idx];
        }
        
        __syncthreads();
        
        // 协作写入到目标内存
        if (global_idx < count) {
            d_dst[global_idx] = shared_buffer[local_idx];
        }
        
        __syncthreads();
    }
}

// 多流优化的D2D kernel（基础版本，实际多流在外部管理）
__global__ void d2d_multi_stream_kernel(uint4* d_dst, const uint4* d_src, size_t count, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    if (idx < count) {
        d_dst[idx] = d_src[idx];
    }
}

extern "C" {
    void launch_h2d_basic_kernel(void* d_dst, void* h_src, size_t count, 
                                int grid_size, int block_size, cudaStream_t stream) {
        h2d_basic_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<int*>(d_dst), static_cast<const int*>(h_src), count);
    }
    
    void launch_h2d_vectorized_kernel(void* d_dst, void* h_src, size_t count,
                                     int grid_size, int block_size, cudaStream_t stream) {
        h2d_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<uint4*>(d_dst), static_cast<const uint4*>(h_src), count);
    }
    
    void launch_d2h_basic_kernel(void* h_dst, void* d_src, size_t count,
                                int grid_size, int block_size, cudaStream_t stream) {
        d2h_basic_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<int*>(h_dst), static_cast<const int*>(d_src), count);
    }
    
    void launch_d2h_vectorized_kernel(void* h_dst, void* d_src, size_t count,
                                     int grid_size, int block_size, cudaStream_t stream) {
        d2h_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<uint4*>(h_dst), static_cast<const uint4*>(d_src), count);
    }
    
    void launch_d2d_basic_kernel(void* d_dst, void* d_src, size_t count,
                                int grid_size, int block_size, cudaStream_t stream) {
        d2d_basic_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<int*>(d_dst), static_cast<const int*>(d_src), count);
    }
    
    void launch_d2d_vectorized_kernel(void* d_dst, void* d_src, size_t count,
                                     int grid_size, int block_size, cudaStream_t stream) {
        d2d_vectorized_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<uint4*>(d_dst), static_cast<const uint4*>(d_src), count);
    }
    
    void launch_d2d_shared_memory_kernel(void* d_dst, void* d_src, size_t count,
                                        int grid_size, int block_size, cudaStream_t stream) {
        d2d_shared_memory_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<int*>(d_dst), static_cast<const int*>(d_src), count);
    }
    
    void launch_d2d_multi_stream_kernel(void* d_dst, void* d_src, size_t count, size_t offset,
                                       int grid_size, int block_size, cudaStream_t stream) {
        d2d_multi_stream_kernel<<<grid_size, block_size, 0, stream>>>(
            static_cast<uint4*>(d_dst), static_cast<const uint4*>(d_src), count, offset);
    }
}

ThreadTransfer::ThreadTransfer() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    setup_timing_events();
}

ThreadTransfer::~ThreadTransfer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    cudaStreamDestroy(stream_);
}

void ThreadTransfer::setup_timing_events() {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

double ThreadTransfer::get_elapsed_time() {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
    return static_cast<double>(milliseconds);
}

int ThreadTransfer::calculate_grid_size(size_t size, size_t element_size) {
    size_t elements = size / element_size;
    int grid_size = (elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return std::min(grid_size, MAX_GRID_SIZE);
}

// H2D Thread传输实现
double ThreadTransfer::benchmark_h2d_basic_pinned(void* d_dst, void* h_pinned_src, size_t size) {
    size_t count = size / sizeof(int);
    int grid_size = calculate_grid_size(size, sizeof(int));
    
    // 预热
    launch_h2d_basic_kernel(d_dst, h_pinned_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_h2d_basic_kernel(d_dst, h_pinned_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_h2d_vectorized_pinned(void* d_dst, void* h_pinned_src, size_t size) {
    size_t count = size / sizeof(uint4);
    int grid_size = calculate_grid_size(size, sizeof(uint4));
    
    // 预热
    launch_h2d_vectorized_kernel(d_dst, h_pinned_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_h2d_vectorized_kernel(d_dst, h_pinned_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_h2d_mapped(void* d_dst, void* h_mapped_src, size_t size) {
    size_t count = size / sizeof(uint4);
    int grid_size = calculate_grid_size(size, sizeof(uint4));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_h2d_vectorized_kernel(d_dst, h_mapped_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

// D2H Thread传输实现
double ThreadTransfer::benchmark_d2h_basic_pinned(void* h_pinned_dst, void* d_src, size_t size) {
    size_t count = size / sizeof(int);
    int grid_size = calculate_grid_size(size, sizeof(int));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_d2h_basic_kernel(h_pinned_dst, d_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_d2h_vectorized_pinned(void* h_pinned_dst, void* d_src, size_t size) {
    size_t count = size / sizeof(uint4);
    int grid_size = calculate_grid_size(size, sizeof(uint4));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_d2h_vectorized_kernel(h_pinned_dst, d_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_d2h_mapped(void* h_mapped_dst, void* d_src, size_t size) {
    size_t count = size / sizeof(uint4);
    int grid_size = calculate_grid_size(size, sizeof(uint4));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_d2h_vectorized_kernel(h_mapped_dst, d_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

// D2D传输实现（GPU内部）
double ThreadTransfer::benchmark_d2d_basic(void* d_dst, void* d_src, size_t size) {
    size_t count = size / sizeof(int);
    int grid_size = calculate_grid_size(size, sizeof(int));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_d2d_basic_kernel(d_dst, d_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_d2d_vectorized(void* d_dst, void* d_src, size_t size) {
    size_t count = size / sizeof(uint4);
    int grid_size = calculate_grid_size(size, sizeof(uint4));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_d2d_vectorized_kernel(d_dst, d_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_d2d_shared_memory(void* d_dst, void* d_src, size_t size) {
    size_t count = size / sizeof(int);
    int grid_size = calculate_grid_size(size, sizeof(int));
    
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    launch_d2d_shared_memory_kernel(d_dst, d_src, count, grid_size, BLOCK_SIZE, stream_);
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double ThreadTransfer::benchmark_d2d_multi_stream(void* d_dst, void* d_src, size_t size) {
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    
    // 创建多个流
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    size_t count = size / sizeof(uint4);
    size_t chunk_size = (count + num_streams - 1) / num_streams;
    
    CUDA_CHECK(cudaEventRecord(start_event_));
    
    // 在多个流中并行执行
    for (int i = 0; i < num_streams; ++i) {
        size_t offset = i * chunk_size;
        size_t current_chunk_size = std::min(chunk_size, count - offset);
        
        if (current_chunk_size > 0) {
            int grid_size = (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            grid_size = std::min(grid_size, MAX_GRID_SIZE);
            
            launch_d2d_multi_stream_kernel(d_dst, d_src, count, offset, 
                                          grid_size, BLOCK_SIZE, streams[i]);
        }
    }
    
    // 同步所有流
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    // 清理流
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    
    return get_elapsed_time();
}

// P2P函数简化实现
double ThreadTransfer::benchmark_p2p_basic(void* d_dst, void* d_src, size_t size, int dst_device, int src_device) {
    return 0.0; // 简化跳过
}

double ThreadTransfer::benchmark_p2p_vectorized(void* d_dst, void* d_src, size_t size, int dst_device, int src_device) {
    return 0.0; // 简化跳过
}