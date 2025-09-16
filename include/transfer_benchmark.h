#ifndef TRANSFER_BENCHMARK_H
#define TRANSFER_BENCHMARK_H

#include "common.h"
#include <vector>
#include <map>
#include <memory>

// 前向声明
class DMATransfer;
class ThreadTransfer;

class TransferBenchmark {
public:
    TransferBenchmark();
    ~TransferBenchmark();

    // 运行完整benchmark
    void run_full_benchmark();
    
    // 分类型运行benchmark
    std::vector<BenchmarkResult> benchmark_h2d(size_t size);
    std::vector<BenchmarkResult> benchmark_d2h(size_t size);
    std::vector<BenchmarkResult> benchmark_d2d(size_t size);
    std::vector<BenchmarkResult> benchmark_p2p(size_t size);
    
    // 设置参数
    void set_iterations(int iterations) { iterations_ = iterations; }
    void set_warmup_iterations(int warmup) { warmup_iterations_ = warmup; }
    void enable_verification(bool enable) { verify_results_ = enable; }
    void set_test_types(const std::vector<TransferType>& types) { test_types_ = types; }
    
    // 输出结果
    void print_results(const std::vector<BenchmarkResult>& results);
    void print_summary(const std::vector<BenchmarkResult>& all_results);
    void save_csv(const std::vector<BenchmarkResult>& results, const std::string& filename);

private:
    int iterations_;
    int warmup_iterations_;
    bool verify_results_;
    std::vector<TransferType> test_types_;
    std::vector<size_t> test_sizes_;
    
    // 测试对象
    std::unique_ptr<DMATransfer> dma_tester_;
    std::unique_ptr<ThreadTransfer> thread_tester_;
    
    // 内存管理结构
    struct MemoryBuffers {
        void* h_pageable;
        void* h_pinned;
        void* h_mapped;
        void* d_ptr;
        void* d_src;
        void* d_dst;
        size_t size;
    };
    
    MemoryBuffers allocate_buffers(size_t size);
    void free_buffers(const MemoryBuffers& buffers);
    bool verify_transfer_result(const MemoryBuffers& buffers, TransferType type);
    
    void initialize_test_sizes();
    void initialize_multi_gpu();
    
    // 多GPU支持
    int gpu_count_;
    std::vector<int> available_gpus_;
    
    // 辅助函数
    BenchmarkResult create_result(const std::string& method_name, TransferType type, size_t size, 
                                 double time_ms, bool success, const std::string& memory_type);
};

#endif // TRANSFER_BENCHMARK_H