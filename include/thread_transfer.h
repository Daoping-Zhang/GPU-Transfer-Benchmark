#ifndef THREAD_TRANSFER_H
#define THREAD_TRANSFER_H

#include "common.h"

class ThreadTransfer {
public:
    ThreadTransfer();
    ~ThreadTransfer();

    // H2D Thread传输
    double benchmark_h2d_basic_pinned(void* d_dst, void* h_pinned_src, size_t size);
    double benchmark_h2d_vectorized_pinned(void* d_dst, void* h_pinned_src, size_t size);
    double benchmark_h2d_mapped(void* d_dst, void* h_mapped_src, size_t size);

    // D2H Thread传输
    double benchmark_d2h_basic_pinned(void* h_pinned_dst, void* d_src, size_t size);
    double benchmark_d2h_vectorized_pinned(void* h_pinned_dst, void* d_src, size_t size);
    double benchmark_d2h_mapped(void* h_mapped_dst, void* d_src, size_t size);

    // D2D Thread传输
    double benchmark_d2d_basic(void* d_dst, void* d_src, size_t size);
    double benchmark_d2d_vectorized(void* d_dst, void* d_src, size_t size);
    double benchmark_d2d_shared_memory(void* d_dst, void* d_src, size_t size);
    double benchmark_d2d_multi_stream(void* d_dst, void* d_src, size_t size);

    // P2P Thread传输
    double benchmark_p2p_basic(void* d_dst, void* d_src, size_t size, int dst_device, int src_device);
    double benchmark_p2p_vectorized(void* d_dst, void* d_src, size_t size, int dst_device, int src_device);

private:
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    static const int BLOCK_SIZE = 256;
    static const int MAX_GRID_SIZE = 2048;
    
    void setup_timing_events();
    double get_elapsed_time();
    int calculate_grid_size(size_t size, size_t element_size);
};

// CUDA kernel声明
extern "C" {
    // H2D kernels (通过UVA访问)
    void launch_h2d_basic_kernel(void* d_dst, void* h_src, size_t count, 
                                int grid_size, int block_size, cudaStream_t stream);
    void launch_h2d_vectorized_kernel(void* d_dst, void* h_src, size_t count,
                                     int grid_size, int block_size, cudaStream_t stream);
    
    // D2H kernels (通过UVA访问)
    void launch_d2h_basic_kernel(void* h_dst, void* d_src, size_t count,
                                int grid_size, int block_size, cudaStream_t stream);
    void launch_d2h_vectorized_kernel(void* h_dst, void* d_src, size_t count,
                                     int grid_size, int block_size, cudaStream_t stream);
    
    // D2D kernels
    void launch_d2d_basic_kernel(void* d_dst, void* d_src, size_t count,
                                int grid_size, int block_size, cudaStream_t stream);
    void launch_d2d_vectorized_kernel(void* d_dst, void* d_src, size_t count,
                                     int grid_size, int block_size, cudaStream_t stream);
}

#endif // THREAD_TRANSFER_H