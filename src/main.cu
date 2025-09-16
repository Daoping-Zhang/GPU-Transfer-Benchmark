#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include "transfer_benchmark.h"
#include "common.h"

void print_gpu_info() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Number of CUDA devices: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " 
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
                  << " GB/s" << std::endl;
        
        // 检查特殊功能支持
        std::cout << "  Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "  Managed Memory: " << (prop.managedMemory ? "Yes" : "No") << std::endl;
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;
        
        // 检查P2P支持
        if (device_count > 1) {
            for (int j = 0; j < device_count; ++j) {
                if (i != j) {
                    int can_access;
                    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
                    std::cout << "  P2P to Device " << j << ": " << (can_access ? "Yes" : "No") << std::endl;
                }
            }
        }
        std::cout << std::endl;
    }
}

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --iterations N      Number of benchmark iterations (default: 10)" << std::endl;
    std::cout << "  --warmup N          Number of warmup iterations (default: 3)" << std::endl;
    std::cout << "  --no-verify         Disable result verification" << std::endl;
    std::cout << "  --save-csv FILE     Save results to CSV file" << std::endl;
    std::cout << "  --test-types TYPE   Comma-separated list: h2d,d2h,d2d,p2p (default: h2d,d2h,d2d)" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
}

std::vector<TransferType> parse_test_types(const std::string& types_str) {
    std::vector<TransferType> types;
    std::stringstream ss(types_str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        if (item == "h2d") types.push_back(TransferType::H2D);
        else if (item == "d2h") types.push_back(TransferType::D2H);
        else if (item == "d2d") types.push_back(TransferType::D2D);
        else if (item == "p2p") types.push_back(TransferType::P2P);
        else {
            std::cerr << "Warning: Unknown transfer type: " << item << std::endl;
        }
    }
    
    return types;
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Transfer Benchmark (H2D/D2H/D2D/P2P)" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Testing DMA vs Thread Copy Performance" << std::endl;
    std::cout << "Memory Types: Pageable, Pinned, Mapped" << std::endl;
    std::cout << std::endl;
    
    try {
        // 检查CUDA设备
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        
        print_gpu_info();
        
        // 解析命令行参数
        int iterations = 10;
        int warmup = 3;
        bool verify = true;
        bool save_csv = false;
        std::string csv_filename = "transfer_benchmark_results.csv";
        std::vector<TransferType> test_types = {
            TransferType::H2D, TransferType::D2H, TransferType::D2D
        };
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--iterations" && i + 1 < argc) {
                iterations = std::stoi(argv[++i]);
            } else if (arg == "--warmup" && i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
            } else if (arg == "--no-verify") {
                verify = false;
            } else if (arg == "--save-csv" && i + 1 < argc) {
                save_csv = true;
                csv_filename = argv[++i];
            } else if (arg == "--test-types" && i + 1 < argc) {
                test_types = parse_test_types(argv[++i]);
                if (test_types.empty()) {
                    std::cerr << "Error: No valid transfer types specified" << std::endl;
                    return 1;
                }
            } else if (arg == "--help") {
                print_help(argv[0]);
                return 0;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_help(argv[0]);
                return 1;
            }
        }
        
        // 创建benchmark对象
        TransferBenchmark benchmark;
        benchmark.set_iterations(iterations);
        benchmark.set_warmup_iterations(warmup);
        benchmark.enable_verification(verify);
        benchmark.set_test_types(test_types);
        
        std::cout << "Benchmark Configuration:" << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Warmup: " << warmup << std::endl;
        std::cout << "  Verification: " << (verify ? "enabled" : "disabled") << std::endl;
        std::cout << "  Test types: ";
        for (size_t i = 0; i < test_types.size(); ++i) {
            std::cout << transfer_type_to_string(test_types[i]);
            if (i < test_types.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << "  CSV output: " << (save_csv ? csv_filename : "disabled") << std::endl;
        std::cout << std::endl;
        
        // 运行benchmark
        auto start_time = std::chrono::high_resolution_clock::now();
        
        benchmark.run_full_benchmark();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "Benchmark completed successfully!" << std::endl;
        std::cout << "Total execution time: " << duration.count() << " seconds" << std::endl;
        
        if (save_csv) {
            std::cout << "Note: CSV saving functionality needs to be implemented" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}