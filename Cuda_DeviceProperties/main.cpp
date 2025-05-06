#include "iostream"
#include "cuda.h"
#include "cuda_runtime.h"

void printprops(cudaDeviceProp devProp){
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
}


int main(){
    int Devcount;
    cudaGetDeviceCount(&Devcount);
    std::cout << "Cuda devices recognised" << Devcount << std::endl;

    for (int i = 0; i < Devcount; i++){
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp DevProp;
        cudaGetDeviceProperties(&DevProp, i);
        printprops(DevProp);
    }
}



