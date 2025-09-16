#include "dma_transfer.h"
#include <stdexcept>

DMATransfer::DMATransfer() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    setup_timing_events();
}

DMATransfer::~DMATransfer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    cudaStreamDestroy(stream_);
}

void DMATransfer::setup_timing_events() {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

double DMATransfer::get_elapsed_time() {
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event_, stop_event_));
    return static_cast<double>(milliseconds);
}

// H2D DMA传输实现
double DMATransfer::benchmark_h2d_sync_pageable(void* d_dst, void* h_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_));
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_h2d_async_pageable(void* d_dst, void* h_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src, size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_h2d_sync_pinned(void* d_dst, void* h_pinned_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpy(d_dst, h_pinned_src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_));
    CUDA_CHECK(cudaMemcpy(d_dst, h_pinned_src, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_h2d_async_pinned(void* d_dst, void* h_pinned_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_pinned_src, size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_pinned_src, size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_h2d_mapped(void* d_dst, void* h_mapped_src, size_t size) {
    // 对于mapped memory，实际上是zero-copy，不需要DMA传输
    // 但这里我们测试显式拷贝的性能
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_dst, h_mapped_src, size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

// D2H DMA传输实现
double DMATransfer::benchmark_d2h_sync_pageable(void* h_dst, void* d_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_));
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_d2h_async_pageable(void* h_dst, void* d_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_src, size, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_d2h_sync_pinned(void* h_pinned_dst, void* d_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpy(h_pinned_dst, d_src, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_));
    CUDA_CHECK(cudaMemcpy(h_pinned_dst, d_src, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_d2h_async_pinned(void* h_pinned_dst, void* d_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_src, size, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_dst, d_src, size, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_d2h_mapped(void* h_mapped_dst, void* d_src, size_t size) {
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_mapped_dst, d_src, size, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

// D2D DMA传输实现
double DMATransfer::benchmark_d2d_sync(void* d_dst, void* d_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_));
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    
    return get_elapsed_time();
}

double DMATransfer::benchmark_d2d_async(void* d_dst, void* d_src, size_t size) {
    // 预热
    CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    // 计时
    CUDA_CHECK(cudaEventRecord(start_event_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    
    return get_elapsed_time();
}

// P2P DMA传输实现（简化版）
double DMATransfer::benchmark_p2p_sync(void* d_dst, void* d_src, size_t size, int dst_device, int src_device) {
    // 暂时返回0，避免复杂的P2P设置
    return 0.0;
}

double DMATransfer::benchmark_p2p_async(void* d_dst, void* d_src, size_t size, int dst_device, int src_device) {
    // 暂时返回0，避免复杂的P2P设置
    return 0.0;
}