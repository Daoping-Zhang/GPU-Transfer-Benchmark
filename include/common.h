#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <stdexcept>

enum class TransferType {
    H2D,    // Host to Device
    D2H,    // Device to Host  
    D2D,    // Device to Device (same GPU)
    P2P     // Peer to Peer (different GPUs)
};

enum class CopyMethod {
    // DMA方法
    DMA_SYNC_PAGEABLE,      // 同步DMA - 普通主机内存
    DMA_ASYNC_PAGEABLE,     // 异步DMA - 普通主机内存
    DMA_SYNC_PINNED,        // 同步DMA - 锁页内存
    DMA_ASYNC_PINNED,       // 异步DMA - 锁页内存
    DMA_MAPPED,             // DMA - Zero-copy映射内存
    
    // Thread方法
    THREAD_BASIC_PAGEABLE,  // 基础线程拷贝 - 普通内存
    THREAD_BASIC_PINNED,    // 基础线程拷贝 - 锁页内存
    THREAD_VECTORIZED_PINNED,   // 向量化线程拷贝 - 锁页内存
    THREAD_MAPPED,          // 线程拷贝 - Zero-copy映射内存
    THREAD_SHARED_MEMORY,   // 共享内存优化线程拷贝（仅D2D）
    THREAD_MULTI_STREAM     // 多流线程拷贝
};

struct BenchmarkResult {
    std::string method_name;
    TransferType transfer_type;
    size_t data_size;
    double time_ms;
    double bandwidth_gbps;
    double latency_us;
    bool success;
    std::string memory_type;  // "Pageable", "Pinned", "Mapped"
};

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + \
                                   std::to_string(__LINE__) + " - " + cudaGetErrorString(error)); \
        } \
    } while(0)

// 转换函数
std::string transfer_type_to_string(TransferType type);
std::string copy_method_to_string(CopyMethod method);
std::string get_memory_type_from_method(CopyMethod method);

// 内存类型枚举
enum class MemoryType {
    PAGEABLE,
    PINNED,
    MAPPED
};

#endif // COMMON_H