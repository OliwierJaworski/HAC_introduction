#include "accelerated.h"

__global__ void sayhi_kernel() {
    printf("Hi from CUDA thread %d\n", threadIdx.x);
}

void sayhi() {
    sayhi_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
}