### Set Environment
``` bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

### Cmake and Run
``` bash
./scripts/build.sh cmake
./build/cuda_transfer_benchmark --iterations 3 --warmup 1 --test-types h2d,d2h,p2p,d2d
```