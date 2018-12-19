GPU Integers Compression
==========================

GPU-Accelerated Faster Decoding of Integer Lists

## Usage 

### Build
```
git clone git@github.com:amallia/gpu-integers-compression.git
cd gpu-integers-compression
mkdir build 
cd build 
cmake .. 
make -j
```

### External libraries
    - Google Test
    - Google benchmark
    - NVlabs CUB
    - FastPFor

### Benchmark

Benchmarks tasks can be found in `bench` folder.

```
./bench/cuda_bp_bench
```

## Codecs

### GPU Binary-Packing
```cpp
#include "gpu_ic/cuda_bp.cuh"

// Values to encode
std::vector<uint32_t> values = {\* ... *\};

// Encode
std::vector<uint8_t> encoded_values;
encoded_values.resize(values.size() * 8);
auto compressedsize = cuda_bp::encode(encoded_values.data(), values.data(), values.size());
encoded_values.resize(compressedsize);
encoded_values.shrink_to_fit();

// Decode
std::vector<uint32_t> decoded_values;
decoded_values.resize(values.size());
CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, encoded_values.size() * sizeof(uint8_t)));
CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, encoded_values.data(), encoded_values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, values.size() * sizeof(uint32_t)));
cuda_bp::decode(d_decoded, d_encoded, decoded_values.size());
CUDA_CHECK_ERROR(cudaMemcpy(decoded_values.data(), d_decoded, values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
```

### GPU Vbyte

```cpp
#include "gpu_ic/cuda_vbyte.cuh"

// Values to encode
std::vector<uint32_t> values = {\* ... *\};

// Encode
std::vector<uint8_t> encoded_values;
encoded_values.resize(values.size() * 8);
auto compressedsize = cuda_vbyte::encode(encoded_values.data(), values.data(), values.size());
encoded_values.resize(compressedsize);
encoded_values.shrink_to_fit();

// Decode
std::vector<uint32_t> decoded_values;
decoded_values.resize(values.size());
CUDA_CHECK_ERROR(cudaMalloc((void **)&d_encoded, encoded_values.size() * sizeof(uint8_t)));
CUDA_CHECK_ERROR(cudaMemcpy(d_encoded, encoded_values.data(), encoded_values.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

CUDA_CHECK_ERROR(cudaMalloc((void **)&d_decoded, values.size() * sizeof(uint32_t)));
cuda_vbyte::decode(d_decoded, d_encoded, decoded_values.size());
CUDA_CHECK_ERROR(cudaMemcpy(decoded_values.data(), d_decoded, values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

```


## Benchmarks
<p align="center">
<img src="plot.png" width="75%">
</p>
    
```
Running ./bench/cuda_bp_bench
Run on (28 X 3500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x28)
  L1 Instruction 32K (x28)
  L2 Unified 256K (x28)
  L3 Unified 35840K (x2)
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
---------------------------------------------------------------------------------------------------
Benchmark                                            Time           CPU Iterations UserCounters...
---------------------------------------------------------------------------------------------------
UniformValuesFixture/decode/32768                 4014 ns       4014 ns     169300 bpi=17.4648
UniformValuesFixture/decode/65536                 5906 ns       5906 ns     130976 bpi=16.458
UniformValuesFixture/decode/131072                9227 ns       9214 ns      83445 bpi=15.4509
UniformValuesFixture/decode/262144               15944 ns      15944 ns      47850 bpi=14.451
UniformValuesFixture/decode/524288               28886 ns      28886 ns      25926 bpi=13.4471
UniformValuesFixture/decode/1048576              53098 ns      53097 ns      10000 bpi=12.4435
UniformValuesFixture/decode/2097152             101586 ns     101586 ns      10000 bpi=11.4459
UniformValuesFixture/decode/4194304             200550 ns     200552 ns      10000 bpi=10.4459
UniformValuesFixture/decode/8388608             399127 ns     399128 ns      10000 bpi=9.44149
UniformValuesFixture/decode/16777216            795107 ns     795117 ns      10000 bpi=8.43567
UniformValuesFixture/decode/33554432           1586458 ns    1586433 ns      10000 bpi=7.42388

Running ./bench/cuda_vbyte_bench
Run on (28 X 3500 MHz CPU s)
CPU Caches:
  L1 Data 32K (x28)
  L1 Instruction 32K (x28)
  L2 Unified 256K (x28)
  L3 Unified 35840K (x2)
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
----------------------------------------------------------------------------------------------
Benchmark                                       Time           CPU Iterations UserCounters...
----------------------------------------------------------------------------------------------
UniformValuesFixture/decode/32768            4390 ns       4390 ns     154046 bpi=19.7529
UniformValuesFixture/decode/65536            6071 ns       6071 ns     127392 bpi=19.3794
UniformValuesFixture/decode/131072           9469 ns       9468 ns      81539 bpi=19.0952
UniformValuesFixture/decode/262144          16427 ns      16427 ns      46396 bpi=18.6882
UniformValuesFixture/decode/524288          29391 ns      29391 ns      25463 bpi=17.8624
UniformValuesFixture/decode/1048576         55090 ns      55090 ns      10000 bpi=16.4886
UniformValuesFixture/decode/2097152        103277 ns     103254 ns      10000 bpi=14.5715
UniformValuesFixture/decode/4194304        203438 ns     203441 ns      10000 bpi=12.7031
UniformValuesFixture/decode/8388608        405796 ns     405806 ns      10000 bpi=12.0025
UniformValuesFixture/decode/16777216       807620 ns     807505 ns      10000 bpi=12
UniformValuesFixture/decode/33554432      1609049 ns    1608882 ns      10000 bpi=12
```
