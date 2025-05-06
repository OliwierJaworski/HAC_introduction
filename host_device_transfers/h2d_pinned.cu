#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h" 
#define ARRAY_SIZE 256
#define NUM_BLOCKS  8
#define THREADS_PER_BLOCK 32

/* results 1 block */
/*
elapsed time (milliseconds): 1.781760
Effective Bandwidth (GB/s): 0.001149
*/

/* results multiblock */
/*
elapsed time (milliseconds): 1.654784
Effective Bandwidth (GB/s): 0.001238k
*/

__global__ void negate(int *d_a){
    int idx = threadIdx.x;
    d_a[idx] = -1 * d_a[idx];
}
 
__global__ void negate_multiblock(int *d_a){
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    d_a[idx] = -1 * d_a[idx];
}
 
int main(int argc, char *argv[]){
    int *h_a, *h_out;
    int *d_a;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int i;
    size_t siz_b = ARRAY_SIZE * sizeof(int);
 
    cudaMallocHost((int**)&h_a, siz_b);
    cudaMallocHost((int**)&h_out, siz_b);

    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_out[i] = 0;
    }

    cudaMalloc((void**)&d_a, siz_b);

    cudaMemcpy(d_a, h_a, siz_b, cudaMemcpyHostToDevice);

    //dim3 blocksPerGrid( ); 
    //dim3 threadsPerBlock( );
    cudaEventRecord(start);
    negate_multiblock<<< NUM_BLOCKS , THREADS_PER_BLOCK >>>( d_a );
    cudaEventRecord(stop);
    //negate_multiblock<<<,>>>();
    //cudaDeviceSynchronize();
 
    cudaMemcpy(h_out, d_a, siz_b, cudaMemcpyDeviceToHost);
 
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("elapsed time (milliseconds): %f\n",milliseconds);
    double bytes = ARRAY_SIZE * sizeof(int) * 2;  // Read + Write
    printf("Effective Bandwidth (GB/s): %f\n", bytes/milliseconds/1e6);

    printf("Results: ");
    for (i = 0; i < ARRAY_SIZE; i++) {
      printf("%d, ", h_out[i]);
    }
    printf("\n\n");
 
    cudaFreeHost(h_a);
    cudaFreeHost(h_out);
    cudaFree(d_a);
 
    return 0;
}