#include "transfer_benchmark.h"
#include "dma_transfer.h"
#include "thread_transfer.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>



// 在TransferBenchmark类中添加P2P辅助函数的实现
TransferBenchmark::P2PBuffers TransferBenchmark::allocate_p2p_buffers(size_t size, int src_device, int dst_device) {
    P2PBuffers buffers;
    buffers.size = size;
    
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));
    
    try {
        // 在源设备分配内存
        CUDA_CHECK(cudaSetDevice(src_device));
        CUDA_CHECK(cudaMalloc(&buffers.d_src, size));
        CUDA_CHECK(cudaMemset(buffers.d_src, 0xAB, size));
        
        // 在目标设备分配内存
        CUDA_CHECK(cudaSetDevice(dst_device));
        CUDA_CHECK(cudaMalloc(&buffers.d_dst, size));
        CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
        
        CUDA_CHECK(cudaSetDevice(original_device));
        return buffers;
        
    } catch (const std::exception& e) {
        CUDA_CHECK(cudaSetDevice(original_device));
        throw;
    }
}

void TransferBenchmark::free_p2p_buffers(const P2PBuffers& buffers, int src_device, int dst_device) {
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));
    
    try {
        if (buffers.d_src) {
            CUDA_CHECK(cudaSetDevice(src_device));
            CUDA_CHECK(cudaFree(buffers.d_src));
        }
        
        if (buffers.d_dst) {
            CUDA_CHECK(cudaSetDevice(dst_device));
            CUDA_CHECK(cudaFree(buffers.d_dst));
        }
        
        CUDA_CHECK(cudaSetDevice(original_device));
    } catch (const std::exception& e) {
        CUDA_CHECK(cudaSetDevice(original_device));
        // 静默处理清理错误
    }
}

void TransferBenchmark::reset_p2p_buffers(const P2PBuffers& buffers, size_t size, int dst_device) {
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));
    
    try {
        CUDA_CHECK(cudaSetDevice(dst_device));
        CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
        CUDA_CHECK(cudaSetDevice(original_device));
    } catch (const std::exception& e) {
        CUDA_CHECK(cudaSetDevice(original_device));
        throw;
    }
}

bool TransferBenchmark::verify_p2p_transfer_result(const P2PBuffers& buffers, size_t size) {
    const size_t sample_size = std::min(size, static_cast<size_t>(1024));
    std::vector<char> sample_src(sample_size);
    std::vector<char> sample_dst(sample_size);
    
    int original_device;
    CUDA_CHECK(cudaGetDevice(&original_device));
    
    try {
        // 读取源数据
        CUDA_CHECK(cudaMemcpy(sample_src.data(), buffers.d_src, sample_size, cudaMemcpyDeviceToHost));
        // 读取目标数据  
        CUDA_CHECK(cudaMemcpy(sample_dst.data(), buffers.d_dst, sample_size, cudaMemcpyDeviceToHost));
        
        bool result = (memcmp(sample_src.data(), sample_dst.data(), sample_size) == 0);
        
        CUDA_CHECK(cudaSetDevice(original_device));
        return result;
        
    } catch (const std::exception& e) {
        CUDA_CHECK(cudaSetDevice(original_device));
        return false;
    }
}


TransferBenchmark::TransferBenchmark() 
    : iterations_(10), warmup_iterations_(3), verify_results_(true), gpu_count_(0) {
    
    initialize_test_sizes();
    initialize_multi_gpu();
    
    dma_tester_ = std::make_unique<DMATransfer>();
    thread_tester_ = std::make_unique<ThreadTransfer>();
    
    // 默认测试所有类型
    test_types_ = {TransferType::H2D, TransferType::D2H, TransferType::D2D, TransferType::P2P};
}

TransferBenchmark::~TransferBenchmark() = default;

void TransferBenchmark::initialize_test_sizes() {
    // 从1KB到1GB的测试大小
    test_sizes_ = {
        1024,           // 1KB
        4096,           // 4KB  
        16384,          // 16KB
        65536,          // 64KB
        131072,         // 128KB
        262144,         // 256KB
        524288,         // 512KB
        1048576,        // 1MB
        4194304,        // 4MB
        16777216,       // 16MB
        67108864,       // 64MB
        268435456,      // 256MB
        1073741824      // 1GB
    };
}

void TransferBenchmark::initialize_multi_gpu() {
    CUDA_CHECK(cudaGetDeviceCount(&gpu_count_));
    
    for (int i = 0; i < gpu_count_; ++i) {
        available_gpus_.push_back(i);
    }
}

TransferBenchmark::MemoryBuffers TransferBenchmark::allocate_buffers(size_t size) {
    MemoryBuffers buffers;
    buffers.size = size;
    
    // 分配各种类型的主机内存
    buffers.h_pageable = malloc(size);
    if (!buffers.h_pageable) {
        throw std::runtime_error("Failed to allocate pageable host memory");
    }
    
    CUDA_CHECK(cudaMallocHost(&buffers.h_pinned, size));
    
    // 分配mapped memory (zero-copy)
    CUDA_CHECK(cudaHostAlloc(&buffers.h_mapped, size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&buffers.d_ptr, buffers.h_mapped, 0));
    
    // 分配GPU内存
    CUDA_CHECK(cudaMalloc(&buffers.d_src, size));
    CUDA_CHECK(cudaMalloc(&buffers.d_dst, size));
    
    // 初始化数据
    memset(buffers.h_pageable, 0xAB, size);
    memset(buffers.h_pinned, 0xAB, size);
    memset(buffers.h_mapped, 0xAB, size);
    CUDA_CHECK(cudaMemset(buffers.d_src, 0xAB, size));
    CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
    
    return buffers;
}

void TransferBenchmark::free_buffers(const MemoryBuffers& buffers) {
    if (buffers.h_pageable) free(buffers.h_pageable);
    if (buffers.h_pinned) cudaFreeHost(buffers.h_pinned);
    if (buffers.h_mapped) cudaFreeHost(buffers.h_mapped);
    if (buffers.d_src) cudaFree(buffers.d_src);
    if (buffers.d_dst) cudaFree(buffers.d_dst);
}

BenchmarkResult TransferBenchmark::create_result(const std::string& method_name, TransferType type, 
                                                size_t size, double time_ms, bool success, 
                                                const std::string& memory_type) {
    BenchmarkResult result;
    result.method_name = method_name;
    result.transfer_type = type;
    result.data_size = size;
    result.time_ms = time_ms;
    result.bandwidth_gbps = (size / 1e9) / (time_ms / 1000.0);
    result.latency_us = time_ms * 1000.0;
    result.success = success;
    result.memory_type = memory_type;
    return result;
}

std::vector<BenchmarkResult> TransferBenchmark::benchmark_h2d(size_t size) {
    std::vector<BenchmarkResult> results;
    auto buffers = allocate_buffers(size);
    
    try {
        // DMA H2D测试 - Sync Pinned
        {
            CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, buffers.size));

            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_h2d_sync_pinned(buffers.d_dst, buffers.h_pinned, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = dma_tester_->benchmark_h2d_sync_pinned(buffers.d_dst, buffers.h_pinned, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::H2D) : true;
            results.push_back(create_result("DMA_Sync", TransferType::H2D, size, 
                                           total_time / iterations_, success, "Pinned"));
        }
        
        // DMA H2D测试 - Async Pinned
        {
            CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, buffers.size));
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_h2d_async_pinned(buffers.d_dst, buffers.h_pinned, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = dma_tester_->benchmark_h2d_async_pinned(buffers.d_dst, buffers.h_pinned, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::H2D) : true;
            results.push_back(create_result("DMA_Async", TransferType::H2D, size, 
                                           total_time / iterations_, success, "Pinned"));
        }
        
        // Thread H2D测试 - Basic Pinned
        {
            CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, buffers.size));
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                thread_tester_->benchmark_h2d_basic_pinned(buffers.d_dst, buffers.h_pinned, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = thread_tester_->benchmark_h2d_basic_pinned(buffers.d_dst, buffers.h_pinned, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::H2D) : true;
            results.push_back(create_result("Thread_Basic", TransferType::H2D, size, 
                                           total_time / iterations_, success, "Pinned"));
        }
        
        // Thread H2D测试 - Vectorized Pinned
        {
            CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, buffers.size));
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                thread_tester_->benchmark_h2d_vectorized_pinned(buffers.d_dst, buffers.h_pinned, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = thread_tester_->benchmark_h2d_vectorized_pinned(buffers.d_dst, buffers.h_pinned, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::H2D) : true;
            results.push_back(create_result("Thread_Vectorized", TransferType::H2D, size, 
                                           total_time / iterations_, success, "Pinned"));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in H2D benchmark: " << e.what() << std::endl;
    }
    
    free_buffers(buffers);
    return results;
}

std::vector<BenchmarkResult> TransferBenchmark::benchmark_d2h(size_t size) {
    std::vector<BenchmarkResult> results;
    auto buffers = allocate_buffers(size);
    
    try {
        // DMA D2H测试 - Async Pinned
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_d2h_async_pinned(buffers.h_pinned, buffers.d_src, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                memset(buffers.h_pinned, 0x00, size);
                double time = dma_tester_->benchmark_d2h_async_pinned(buffers.h_pinned, buffers.d_src, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::D2H) : true;
            results.push_back(create_result("DMA_Async", TransferType::D2H, size, 
                                           total_time / iterations_, success, "Pinned"));
        }
        
        // Thread D2H测试 - Basic Pinned
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                thread_tester_->benchmark_d2h_basic_pinned(buffers.h_pinned, buffers.d_src, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                memset(buffers.h_pinned, 0x00, size);
                double time = thread_tester_->benchmark_d2h_basic_pinned(buffers.h_pinned, buffers.d_src, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::D2H) : true;
            results.push_back(create_result("Thread_Basic", TransferType::D2H, size, 
                                           total_time / iterations_, success, "Pinned"));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in D2H benchmark: " << e.what() << std::endl;
    }
    
    free_buffers(buffers);
    return results;
}

std::vector<BenchmarkResult> TransferBenchmark::benchmark_d2d(size_t size) {
    std::vector<BenchmarkResult> results;
    auto buffers = allocate_buffers(size);
    
    try {
        // DMA D2D测试 - Sync
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_d2d_sync(buffers.d_dst, buffers.d_src, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = dma_tester_->benchmark_d2d_sync(buffers.d_dst, buffers.d_src, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::D2D) : true;
            results.push_back(create_result("DMA_Sync", TransferType::D2D, size, 
                                           total_time / iterations_, success, "Device"));
        }
        
        // DMA D2D测试 - Async
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_d2d_async(buffers.d_dst, buffers.d_src, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = dma_tester_->benchmark_d2d_async(buffers.d_dst, buffers.d_src, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::D2D) : true;
            results.push_back(create_result("DMA_Async", TransferType::D2D, size, 
                                           total_time / iterations_, success, "Device"));
        }
        
        // Thread D2D测试 - Vectorized
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                thread_tester_->benchmark_d2d_vectorized(buffers.d_dst, buffers.d_src, size);
            }
            for (int i = 0; i < iterations_; ++i) {
                CUDA_CHECK(cudaMemset(buffers.d_dst, 0x00, size));
                double time = thread_tester_->benchmark_d2d_vectorized(buffers.d_dst, buffers.d_src, size);
                total_time += time;
            }
            bool success = verify_results_ ? verify_transfer_result(buffers, TransferType::D2D) : true;
            results.push_back(create_result("Thread_Vectorized", TransferType::D2D, size, 
                                           total_time / iterations_, success, "Device"));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in D2D benchmark: " << e.what() << std::endl;
    }
    
    free_buffers(buffers);
    return results;
}

std::vector<BenchmarkResult> TransferBenchmark::benchmark_p2p(size_t size) {
    std::vector<BenchmarkResult> results;
    
    if (gpu_count_ < 2) {
        std::cout << "P2P benchmark skipped: requires at least 2 GPUs" << std::endl;
        return results;
    }
    
    std::cout << "Running P2P benchmarks between GPU devices..." << std::endl;
    
    // 测试第一对GPU (0 -> 1)
    int src_device = 0;
    int dst_device = 1;
    
    try {
        // 分配P2P内存缓冲区
        auto p2p_buffers = allocate_p2p_buffers(size, src_device, dst_device);
        
        std::cout << "  Testing P2P transfer: GPU " << src_device << " -> GPU " << dst_device << std::endl;
        
        // DMA P2P测试 - Sync
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_p2p_sync(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
            }
            for (int i = 0; i < iterations_; ++i) {
                // 重置目标数据
                reset_p2p_buffers(p2p_buffers, size, dst_device);
                double time = dma_tester_->benchmark_p2p_sync(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
                if (time > 0) total_time += time;
            }
            
            if (total_time > 0) {
                bool success = verify_results_ ? verify_p2p_transfer_result(p2p_buffers, size) : true;
                results.push_back(create_result("DMA_P2P_Sync", TransferType::P2P, size, 
                                               total_time / iterations_, success, "Device"));
            }
        }
        
        // DMA P2P测试 - Async
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                dma_tester_->benchmark_p2p_async(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
            }
            for (int i = 0; i < iterations_; ++i) {
                reset_p2p_buffers(p2p_buffers, size, dst_device);
                double time = dma_tester_->benchmark_p2p_async(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
                if (time > 0) total_time += time;
            }
            
            if (total_time > 0) {
                bool success = verify_results_ ? verify_p2p_transfer_result(p2p_buffers, size) : true;
                results.push_back(create_result("DMA_P2P_Async", TransferType::P2P, size, 
                                               total_time / iterations_, success, "Device"));
            }
        }
        
        // Thread P2P测试 - Basic
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                thread_tester_->benchmark_p2p_basic(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
            }
            for (int i = 0; i < iterations_; ++i) {
                reset_p2p_buffers(p2p_buffers, size, dst_device);
                double time = thread_tester_->benchmark_p2p_basic(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
                if (time > 0) total_time += time;
            }
            
            if (total_time > 0) {
                bool success = verify_results_ ? verify_p2p_transfer_result(p2p_buffers, size) : true;
                results.push_back(create_result("Thread_P2P_Basic", TransferType::P2P, size, 
                                               total_time / iterations_, success, "Device"));
            }
        }
        
        // Thread P2P测试 - Vectorized
        {
            double total_time = 0;
            for (int i = 0; i < warmup_iterations_; ++i) {
                thread_tester_->benchmark_p2p_vectorized(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
            }
            for (int i = 0; i < iterations_; ++i) {
                reset_p2p_buffers(p2p_buffers, size, dst_device);
                double time = thread_tester_->benchmark_p2p_vectorized(p2p_buffers.d_dst, p2p_buffers.d_src, size, dst_device, src_device);
                if (time > 0) total_time += time;
            }
            
            if (total_time > 0) {
                bool success = verify_results_ ? verify_p2p_transfer_result(p2p_buffers, size) : true;
                results.push_back(create_result("Thread_P2P_Vectorized", TransferType::P2P, size, 
                                               total_time / iterations_, success, "Device"));
            }
        }
        
        free_p2p_buffers(p2p_buffers, src_device, dst_device);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in P2P benchmark: " << e.what() << std::endl;
    }
    
    return results;
}
bool TransferBenchmark::verify_transfer_result(const MemoryBuffers& buffers, TransferType type) {
    // 简化验证：检查目标是否包含源数据的模式
    const size_t sample_size = std::min(buffers.size, static_cast<size_t>(1024));
    char* sample_src = new char[sample_size];
    char* sample_dst = new char[sample_size];
    
    bool result = false;
    
    try {
        switch (type) {
            case TransferType::H2D:
                memcpy(sample_src, buffers.h_pinned, sample_size);
                CUDA_CHECK(cudaMemcpy(sample_dst, buffers.d_dst, sample_size, cudaMemcpyDeviceToHost));
                result = (memcmp(sample_src, sample_dst, sample_size) == 0);
                break;
                
            case TransferType::D2H:
                CUDA_CHECK(cudaMemcpy(sample_src, buffers.d_src, sample_size, cudaMemcpyDeviceToHost));
                memcpy(sample_dst, buffers.h_pinned, sample_size);
                result = (memcmp(sample_src, sample_dst, sample_size) == 0);
                break;
                
            case TransferType::D2D:
                CUDA_CHECK(cudaMemcpy(sample_src, buffers.d_src, sample_size, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(sample_dst, buffers.d_dst, sample_size, cudaMemcpyDeviceToHost));
                result = (memcmp(sample_src, sample_dst, sample_size) == 0);
                break;
                
            case TransferType::P2P:
                result = true;  // P2P验证较复杂，暂时跳过
                break;
        }
    } catch (const std::exception& e) {
        result = false;
    }

    
    
    delete[] sample_src;
    delete[] sample_dst;
    
    return result;
}

void TransferBenchmark::run_full_benchmark() {
    std::cout << "Starting comprehensive transfer benchmark..." << std::endl;
    std::cout << std::endl;
    
    std::vector<BenchmarkResult> all_results;
    
    for (size_t size : test_sizes_) {
        std::cout << "Testing size: " << (size / 1024) << " KB";
        if (size >= 1024 * 1024) {
            std::cout << " (" << (size / (1024 * 1024)) << " MB)";
        }
        std::cout << std::endl;
        
        for (TransferType type : test_types_) {
            std::cout << "  " << transfer_type_to_string(type) << " transfers..." << std::endl;
            
            std::vector<BenchmarkResult> results;
            switch (type) {
                case TransferType::H2D:
                    results = benchmark_h2d(size);
                    break;
                case TransferType::D2H:
                    results = benchmark_d2h(size);
                    break;
                case TransferType::D2D:
                    results = benchmark_d2d(size);
                    break;
                case TransferType::P2P:
                    results = benchmark_p2p(size);
                    break;
            }
            
            all_results.insert(all_results.end(), results.begin(), results.end());
            print_results(results);
        }
        
        std::cout << std::string(100, '-') << std::endl;
    }
    
    // 输出总结
    print_summary(all_results);
}

void TransferBenchmark::print_results(const std::vector<BenchmarkResult>& results) {
    if (results.empty()) return;
    
    std::cout << std::setw(25) << "Method" 
              << std::setw(12) << "Memory Type"
              << std::setw(12) << "Time(ms)"
              << std::setw(12) << "BW(GB/s)"
              << std::setw(12) << "Latency(μs)"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(83, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(25) << result.method_name
                  << std::setw(12) << result.memory_type
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.bandwidth_gbps
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.latency_us
                  << std::setw(10) << (result.success ? "PASS" : "FAIL") << std::endl;
    }
    std::cout << std::endl;
}

void TransferBenchmark::print_summary(const std::vector<BenchmarkResult>& all_results) {
    std::cout << "\n=== BENCHMARK SUMMARY ===" << std::endl;
    
    // 按传输类型和方法分组
    std::map<std::string, std::vector<BenchmarkResult>> grouped_results;
    for (const auto& result : all_results) {
        std::string key = transfer_type_to_string(result.transfer_type) + "_" + result.method_name + "_" + result.memory_type;
        grouped_results[key].push_back(result);
    }
    
    std::cout << std::setw(35) << "Method" 
              << std::setw(15) << "Avg BW(GB/s)"
              << std::setw(15) << "Min Latency(μs)"
              << std::setw(15) << "Max BW(GB/s)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& pair : grouped_results) {
        const auto& key = pair.first;
        const auto& results = pair.second;
        
        double avg_bandwidth = 0;
        double min_latency = 1e9;
        double max_bandwidth = 0;
        
        for (const auto& result : results) {
            avg_bandwidth += result.bandwidth_gbps;
            min_latency = std::min(min_latency, result.latency_us);
            max_bandwidth = std::max(max_bandwidth, result.bandwidth_gbps);
        }
        avg_bandwidth /= results.size();
        
        std::cout << std::setw(35) << key
                  << std::setw(15) << std::fixed << std::setprecision(1) << avg_bandwidth
                  << std::setw(15) << std::fixed << std::setprecision(1) << min_latency
                  << std::setw(15) << std::fixed << std::setprecision(1) << max_bandwidth << std::endl;
    }
}

void TransferBenchmark::save_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 写入CSV头
    file << "TransferType,Method,MemoryType,DataSize(MB),Time(ms),Bandwidth(GB/s),Latency(μs),Status\n";
    
    // 写入数据
    for (const auto& result : results) {
        file << transfer_type_to_string(result.transfer_type) << ","
             << result.method_name << ","
             << result.memory_type << ","
             << (result.data_size / (1024.0 * 1024.0)) << ","
             << result.time_ms << ","
             << result.bandwidth_gbps << ","
             << result.latency_us << ","
             << (result.success ? "PASS" : "FAIL") << "\n";
    }
    
    file.close();
}

