#include "common.h"

std::string transfer_type_to_string(TransferType type) {
    switch (type) {
        case TransferType::H2D: return "H2D";
        case TransferType::D2H: return "D2H";
        case TransferType::D2D: return "D2D";
        case TransferType::P2P: return "P2P";
        default: return "Unknown";
    }
}

std::string copy_method_to_string(CopyMethod method) {
    switch (method) {
        case CopyMethod::DMA_SYNC_PAGEABLE: return "DMA_Sync_Pageable";
        case CopyMethod::DMA_ASYNC_PAGEABLE: return "DMA_Async_Pageable";
        case CopyMethod::DMA_SYNC_PINNED: return "DMA_Sync_Pinned";
        case CopyMethod::DMA_ASYNC_PINNED: return "DMA_Async_Pinned";
        case CopyMethod::DMA_MAPPED: return "DMA_Mapped";
        case CopyMethod::THREAD_BASIC_PAGEABLE: return "Thread_Basic_Pageable";
        case CopyMethod::THREAD_BASIC_PINNED: return "Thread_Basic_Pinned";
        case CopyMethod::THREAD_VECTORIZED_PINNED: return "Thread_Vectorized_Pinned";
        case CopyMethod::THREAD_MAPPED: return "Thread_Mapped";
        case CopyMethod::THREAD_SHARED_MEMORY: return "Thread_SharedMem";
        case CopyMethod::THREAD_MULTI_STREAM: return "Thread_MultiStream";
        default: return "Unknown";
    }
}

std::string get_memory_type_from_method(CopyMethod method) {
    switch (method) {
        case CopyMethod::DMA_SYNC_PAGEABLE:
        case CopyMethod::DMA_ASYNC_PAGEABLE:
        case CopyMethod::THREAD_BASIC_PAGEABLE:
            return "Pageable";
        
        case CopyMethod::DMA_SYNC_PINNED:
        case CopyMethod::DMA_ASYNC_PINNED:
        case CopyMethod::THREAD_BASIC_PINNED:
        case CopyMethod::THREAD_VECTORIZED_PINNED:
        case CopyMethod::THREAD_SHARED_MEMORY:
        case CopyMethod::THREAD_MULTI_STREAM:
            return "Pinned";
        
        case CopyMethod::DMA_MAPPED:
        case CopyMethod::THREAD_MAPPED:
            return "Mapped";
        
        default:
            return "Unknown";
    }
}