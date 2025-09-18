#include "dma_transfer.h"
#include <stdexcept>
#include <iostream>  // 添加这行


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

// 修复DMA P2P同步传输
double DMATransfer::benchmark_p2p_sync(void* d_dst, void* d_src, size_t size, int dst_device, int src_device) {
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));
    
    try {
        // 检查P2P访问能力
        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, dst_device, src_device));
        if (!can_access_peer) {
            std::cout << "P2P access not supported between devices " 
                      << src_device << " and " << dst_device << std::endl;
            CUDA_CHECK(cudaSetDevice(original_device));
            return 0.0;
        }
        
        // 启用P2P访问（双向）
        CUDA_CHECK(cudaSetDevice(dst_device));
        cudaError_t p2p_result = cudaDeviceEnablePeerAccess(src_device, 0);
        if (p2p_result != cudaSuccess && p2p_result != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(cudaSetDevice(original_device));
            return 0.0;
        }
        
        CUDA_CHECK(cudaSetDevice(src_device));
        p2p_result = cudaDeviceEnablePeerAccess(dst_device, 0);
        if (p2p_result != cudaSuccess && p2p_result != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(cudaSetDevice(original_device));
            return 0.0;
        }
        
        // 使用目标设备执行P2P传输
        CUDA_CHECK(cudaSetDevice(dst_device));
        
        // 预热
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 创建专用的事件用于计时
        cudaEvent_t p2p_start, p2p_stop;
        CUDA_CHECK(cudaEventCreate(&p2p_start));
        CUDA_CHECK(cudaEventCreate(&p2p_stop));
        
        // 计时
        CUDA_CHECK(cudaEventRecord(p2p_start));
        CUDA_CHECK(cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(p2p_stop));
        CUDA_CHECK(cudaEventSynchronize(p2p_stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, p2p_start, p2p_stop));
        
        // 清理
        CUDA_CHECK(cudaEventDestroy(p2p_start));
        CUDA_CHECK(cudaEventDestroy(p2p_stop));
        CUDA_CHECK(cudaSetDevice(original_device));
        
        return static_cast<double>(milliseconds);
        
    } catch (const std::exception& e) {
        CUDA_CHECK(cudaSetDevice(original_device));
        std::cerr << "P2P sync transfer error: " << e.what() << std::endl;
        return 0.0;
    }
}

// 修复DMA P2P异步传输
double DMATransfer::benchmark_p2p_async(void* d_dst, void* d_src, size_t size, int dst_device, int src_device) {
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));
    
    try {
        // 检查P2P访问能力
        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, dst_device, src_device));
        if (!can_access_peer) {
            CUDA_CHECK(cudaSetDevice(original_device));
            return 0.0;
        }
        
        // 启用P2P访问
        CUDA_CHECK(cudaSetDevice(dst_device));
        cudaError_t p2p_result = cudaDeviceEnablePeerAccess(src_device, 0);
        if (p2p_result != cudaSuccess && p2p_result != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(cudaSetDevice(original_device));
            return 0.0;
        }
        
        CUDA_CHECK(cudaSetDevice(src_device));
        p2p_result = cudaDeviceEnablePeerAccess(dst_device, 0);
        if (p2p_result != cudaSuccess && p2p_result != cudaErrorPeerAccessAlreadyEnabled) {
            CUDA_CHECK(cudaSetDevice(original_device));
            return 0.0;
        }
        
        // 在目标设备上创建流和事件
        CUDA_CHECK(cudaSetDevice(dst_device));
        
        cudaStream_t p2p_stream;
        cudaEvent_t p2p_start, p2p_stop;
        CUDA_CHECK(cudaStreamCreate(&p2p_stream));
        CUDA_CHECK(cudaEventCreate(&p2p_start));
        CUDA_CHECK(cudaEventCreate(&p2p_stop));
        
        // 预热
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, p2p_stream));
        CUDA_CHECK(cudaStreamSynchronize(p2p_stream));
        
        // 计时
        CUDA_CHECK(cudaEventRecord(p2p_start, p2p_stream));
        CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, size, cudaMemcpyDeviceToDevice, p2p_stream));
        CUDA_CHECK(cudaEventRecord(p2p_stop, p2p_stream));
        CUDA_CHECK(cudaStreamSynchronize(p2p_stream));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, p2p_start, p2p_stop));
        
        // 清理
        CUDA_CHECK(cudaEventDestroy(p2p_start));
        CUDA_CHECK(cudaEventDestroy(p2p_stop));
        CUDA_CHECK(cudaStreamDestroy(p2p_stream));
        CUDA_CHECK(cudaSetDevice(original_device));
        
        return static_cast<double>(milliseconds);
        
    } catch (const std::exception& e) {
        CUDA_CHECK(cudaSetDevice(original_device));
        std::cerr << "P2P async transfer error: " << e.what() << std::endl;
        return 0.0;
    }
}