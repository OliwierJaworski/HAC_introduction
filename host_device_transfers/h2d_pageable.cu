#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include "cuda_runtime.h" 
#define ARRAY_SIZE 256
#define NUM_BLOCKS  1
#define THREADS_PER_BLOCK 256

/* results */
/*
elapsed time (milliseconds): 1.662976
Effective Bandwidth (GB/s): 0.001232
*/
__global__ void negate(int *d_a){
    int idx = threadIdx.x;
    d_a[idx] = -1 * d_a[idx];
}
 
__global__ void negate_multiblock(int *d_a){
 // CODE_2
}
 
int main(int argc, char *argv[]){
    int *h_a, *h_out;
    int *d_a;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int i;
    size_t siz_b = ARRAY_SIZE * sizeof(int);
    h_a = (int *) malloc(siz_b);
    h_out = (int *) malloc(siz_b);
 
    cudaMalloc(&d_a, siz_b);
 
    for (i = 0; i < ARRAY_SIZE; i++) {
        h_a[i] = i;
        h_out[i] = 0;
    }
 
    cudaMemcpy(d_a, h_a, siz_b, cudaMemcpyHostToDevice);
 
    //dim3 blocksPerGrid( ); 
    //dim3 threadsPerBlock( );
    cudaEventRecord(start);
    negate<<< NUM_BLOCKS , THREADS_PER_BLOCK >>>( d_a );
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
 
    cudaFree(d_a);
 
    free(h_a);
    free(h_out);
 
    return 0;
}