# Optimising matrix multiplication on CPUs
## INTRODUCTION
This project focuses on implementing various kernels for matrix multiplication on the CPU. Each kernel is designed to perform a specific optimization to improve the overall performance of matrix multiplication operations. For simplicity, we assumed the matrix be a square matrix of size 4096 x 4096.

## KERNELS AND RUNTIME

| PROCESSOR | PLATFORM |
|--|--|
Intel Core i7-12650H | WINDOWS 11 |


NUM FLOP(s) = 2 * N^3 = 2 * (4096 * 4096 * 4096) = 137.44 GFLOPS


| KERNEL | GFLOPs | relative speedup | absolute speedup
|--------|---------|--| -- |
| LVL1. NAIVE MATMUL| 0.14 | 1 | 1 |
| LVL2. CACHE AWARE LOOP REARRANGEMENT | 1.21 | 8.2x | 8.2x |
| LVL3. COMPILER CHANGE TO ICPX & -O3 | 10.08 | 8.29x | 68.12x |
| LVL4. PARALLEL FOR LOOP USING TBB | 60.14 | 5.96x | 406.1x |
| LVL5. TILING | 86.82 | 1.443x | 586.2x |
| LVL6. PARALLEL DIVIDE AND CONQUER | 106.29 | 1.2x | 717.7x |
| LVL7. COMPILER FLAGS fastmath, arch=native | 274.88 | 2.58x | 1856x |
| LVL8. HARD CODED AVX2 INTRINSICS | 335.21 |  1.21x |  2263x |
| numpy openBLAS | 549.76 | - | - |


### CODE STRUCTURE

All the kernels are in the form of template functions in src/kernelsCPU.h file.

Benchmarking code in src/matMulCPU.cpp file.

### USAGE
Since we've used tbb, environment variables for Intel OneAPI will have to be set, and tbb will have to linked.

First 2 kernels have been compiled with MSVC.

<code>cl.exe matMulCPU.cpp -ltbb </code> 

Next kernels (till LVL 6) are compiled with Intel C++ compiler.

<code>icpx matMulCPU.cpp -o main.exe -ltbb</code>

LVL 7,8 kernels are compiled with additional compiler flags.

<code>icpx matMulCPU.cpp -o main.exe -ltbb -march=native -ffast-math</code>


### ACKNOWLEDGEMENTS
We acknowledge the following resources and references that helped in developing and optimizing the matrix multiplication kernels:

- Charles Leiserson's lecture on MIT OCW performance engineering. 
- https://www.youtube.com/watch?v=eweD5_mV7h4 - for divide and conquer matMul
