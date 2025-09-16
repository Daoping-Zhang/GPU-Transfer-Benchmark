#ifndef DMA_TRANSFER_H
#define DMA_TRANSFER_H

#include "common.h"

class DMATransfer {
public:
    DMATransfer();
    ~DMATransfer();

    // H2D DMA传输
    double benchmark_h2d_sync_pageable(void* d_dst, void* h_src, size_t size);
    double benchmark_h2d_async_pageable(void* d_dst, void* h_src, size_t size);
    double benchmark_h2d_sync_pinned(void* d_dst, void* h_pinned_src, size_t size);
    double benchmark_h2d_async_pinned(void* d_dst, void* h_pinned_src, size_t size);
    double benchmark_h2d_mapped(void* d_dst, void* h_mapped_src, size_t size);

    // D2H DMA传输
    double benchmark_d2h_sync_pageable(void* h_dst, void* d_src, size_t size);
    double benchmark_d2h_async_pageable(void* h_dst, void* d_src, size_t size);
    double benchmark_d2h_sync_pinned(void* h_pinned_dst, void* d_src, size_t size);
    double benchmark_d2h_async_pinned(void* h_pinned_dst, void* d_src, size_t size);
    double benchmark_d2h_mapped(void* h_mapped_dst, void* d_src, size_t size);

    // D2D DMA传输
    double benchmark_d2d_sync(void* d_dst, void* d_src, size_t size);
    double benchmark_d2d_async(void* d_dst, void* d_src, size_t size);

    // P2P DMA传输
    double benchmark_p2p_sync(void* d_dst, void* d_src, size_t size, int dst_device, int src_device);
    double benchmark_p2p_async(void* d_dst, void* d_src, size_t size, int dst_device, int src_device);

private:
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    
    void setup_timing_events();
    double get_elapsed_time();
};

#endif // DMA_TRANSFER_H