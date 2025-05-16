/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include "accelerated.h"

#include "stb_image.h"
#include "stb_image_write.h"


#define WIDTH 640
#define HEIGHT 480

#define TILE_W 32
#define TILE_H 16

#define KERNEL_W 3
#define KERNEL_H 3
#define KERNEL_R 1

#define IMG_PIXELS 307200 
#define IMG_COMPONENTS (IMG_PIXELS * 4)

__device__ __constant__ float d_KernelW[KERNEL_W];
__device__ __constant__ float d_KernelH[KERNEL_H];


dim3 threadPerBlockRow(TILE_W + 2 * KERNEL_R, TILE_H);
dim3 threadPerBlock(TILE_W, TILE_H);
dim3 blockGrids((WIDTH + TILE_W - 1) / TILE_W, (HEIGHT + TILE_H - 1) / TILE_H);


__global__ void 
Cuda_ConvCalcRow(Pixel_t* in, Pixel_t* out){    
    __shared__ Pixel_t data[TILE_H][KERNEL_R + TILE_W + KERNEL_R];

    const int tileStart     = blockIdx.x * TILE_W;
    const int tileEnd       = tileStart + TILE_W -1;
    const int apronStart    = tileStart - KERNEL_R;
    const int apronEnd      = tileEnd + KERNEL_R;

    const int tileEndClamped    = min(tileEnd, WIDTH -1);
    const int apronStartClamped = max(apronStart, 0);
    const int apronEndClamped   = min(apronEnd, WIDTH -1);

    const int y = blockIdx.y * TILE_H + threadIdx.y;
    if (y >= HEIGHT) return;

    const int rowStart = y * WIDTH;
    const int unaligned = apronStart;
    const int apronStartAligned = unaligned & ~15;

    for (int lx = threadIdx.x; lx < (TILE_W + 2 * KERNEL_R); lx += blockDim.x) {
        int loadPos = apronStart + lx;
        Pixel_t val = Pixel_t{0,0,0,0};

        if (loadPos >= 0 && loadPos < WIDTH) {
            val = in[y * WIDTH + loadPos];
        }
        data[threadIdx.y][lx] = val;
    }

    __syncthreads();

    const int writePos = tileStart + threadIdx.x;
    if (writePos <= tileEndClamped){
        int memPos = writePos - apronStart;
        float sum[3]{0, 0, 0};

        for(int c = -1; c <= 1; ++c){
            int k = c + 1;
            sum[0] += data[threadIdx.y][memPos + c].r * d_KernelW[k];
            sum[1] += data[threadIdx.y][memPos + c].g * d_KernelW[k];
            sum[2] += data[threadIdx.y][memPos + c].b * d_KernelW[k];
        }

        auto clamp = [] __device__ (float val) {
            return val < 0 ? 0 : (val > 255 ? 255 : val);
        };

        out[rowStart + writePos] = Pixel_t{
            static_cast<unsigned char>(clamp(sum[0])),
            static_cast<unsigned char>(clamp(sum[1])),
            static_cast<unsigned char>(clamp(sum[2])),
            255
        };
    }
}

__global__ void 
Cuda_ConvCalcColumn(Pixel_t* in, Pixel_t* out){    
     __shared__ Pixel_t data[TILE_W * (KERNEL_R + TILE_H + KERNEL_R)];

    const int         tileStart = blockIdx.y * TILE_H;
    const int           tileEnd = tileStart + TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_R;
    const int          apronEnd = tileEnd   + KERNEL_R;

    const int    tileEndClamped = min(tileEnd, HEIGHT - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, HEIGHT - 1);

    const int       columnStart = (blockIdx.x * TILE_W) + threadIdx.x;

    int smemPos = (threadIdx.y * TILE_W) + threadIdx.x;
    int gmemPos = ((apronStart + threadIdx.y) * WIDTH) + columnStart;

    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] = 
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ? 
        in[gmemPos] : Pixel_t{0,0,0,0};
        smemPos += TILE_W * blockDim.y;
        gmemPos += WIDTH * blockDim.y;
    }

    __syncthreads();

    smemPos = ((threadIdx.y + KERNEL_R) * TILE_W) + threadIdx.x;
    gmemPos = ((tileStart + threadIdx.y) * WIDTH) + columnStart;

    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum[3]{0,0,0};

        for(int c{-1}; c < 2; c++){
            int k = c + 1;
            sum[0] += data[smemPos+c*TILE_W].r * d_KernelH[k];
            sum[1] += data[smemPos+c*TILE_W].g * d_KernelH[k];
            sum[2] += data[smemPos+c*TILE_W].b * d_KernelH[k];
        }

        float greyscaled = sum[0] * 0.2126f + sum[1] * 0.7152f + sum[2] * 0.0722f;  

        auto clamp = [] __device__ (float val) {
            return val < 0 ? 0 : (val > 255 ? 255 : val);
        };

        unsigned char clamped_GS = clamp(greyscaled);

        out[gmemPos] = Pixel_t{ clamped_GS, clamped_GS, clamped_GS, 255 };

        smemPos += TILE_W * blockDim.y;
        gmemPos += WIDTH * blockDim.y;
    }
} 

void 
Convolution::DeviceConvCalc(){
    std::cout << __func__ << " being performed\r\n";
    
    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->Getheight();
    size_t width = *image->GetWidth();

    newImage->SetHeight( height );
    newImage->SetWidth( width );
    newImage->SetcomponentCount( height * width * 4 );

    Pixel_t* HIN_pixels;
    Pixel_t* HOUT_pixels;

    Pixel_t* DIN_pixels;
    Pixel_t* DTMP_RSLT;
    Pixel_t* DOUT_pixels;

    float MMH[3] = {1.0f, 1.0f, 1.0f};
    float MMW[3] = {-1.0f, 0.0f, 1.0f};
    cudaMemcpyToSymbol(d_KernelH,&MMH , KERNEL_H * sizeof(float));
    cudaMemcpyToSymbol(d_KernelW, &MMW, KERNEL_W * sizeof(float));

    cudaMallocHost((void**)&HIN_pixels, sizeof(Pixel_t) * IMG_PIXELS ); 
    cudaMallocHost((void**)&HOUT_pixels, sizeof(Pixel_t) * IMG_PIXELS );
    
    Pixel_t** pixels = image->GetPixelARR();

    for(size_t pxl{0}; pxl < IMG_PIXELS; ++pxl){
        HIN_pixels[pxl] = *pixels[pxl];
    }
    memset(HOUT_pixels, 0, IMG_COMPONENTS );

    cudaMalloc((void**)&DIN_pixels, sizeof(Pixel_t) * IMG_PIXELS );
    cudaMalloc((void**)&DTMP_RSLT, sizeof(Pixel_t) * IMG_PIXELS);
    cudaMalloc((void**)&DOUT_pixels, sizeof(Pixel_t) * IMG_PIXELS );

    cudaMemcpy(DIN_pixels, HIN_pixels, IMG_COMPONENTS, cudaMemcpyHostToDevice);

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    float totalTimeRow = 0.0f;
    float totalTimeColumn = 0.0f;

    float warmupTimeRow = 0.0f;
    float warmupTimeCol = 0.0f;

    for (int i = 0; i < 11; ++i) {
        float time = 0.0f;

        // Row kernel timing
        cudaEventRecord(startEvent, 0);
        Cuda_ConvCalcRow<<<blockGrids, threadPerBlockRow>>>(DIN_pixels, DTMP_RSLT);
        cudaDeviceSynchronize();
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        if (i == 0) {
        warmupTimeRow = time;
        } else {
            totalTimeRow += time;
        }

        // Column kernel timing
        time = 0.0f;
        cudaEventRecord(startEvent, 0);
        Cuda_ConvCalcColumn<<<blockGrids, threadPerBlock>>>(DTMP_RSLT, DOUT_pixels);
        cudaDeviceSynchronize();
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&time, startEvent, stopEvent);
        if (i == 0) {
            warmupTimeCol = time;
        } else {
            totalTimeColumn += time;
        }
    }

    float avgRow = totalTimeRow / 10.0f;
    float avgCol = totalTimeColumn / 10.0f;
    float avgTotal = avgRow + avgCol;
    float warmupTotal = warmupTimeRow + warmupTimeCol;

    printf("Warmup Row Kernel Time: %.4f ms\n", warmupTimeRow);
    printf("Warmup Column Kernel Time: %.4f ms\n", warmupTimeCol);
    printf("Warmup Total Time: %.4f ms\n", warmupTotal);
    printf("Avg Row Kernel Time: %.4f ms\n", avgRow);
    printf("Avg Column Kernel Time: %.4f ms\n", avgCol);
    printf("Avg Total Time: %.4f ms\n", avgTotal);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaMemcpy(HOUT_pixels, DOUT_pixels, IMG_COMPONENTS, cudaMemcpyDeviceToHost);

    stbi_write_png(newImage->GetFPath(), width, height, 4, HOUT_pixels, width * 4);

    cudaFreeHost(HIN_pixels);
    cudaFreeHost(HOUT_pixels);

    cudaFree(DIN_pixels);
    cudaFree(DTMP_RSLT);
    cudaFree(DOUT_pixels);

    printf("DONE\r\n");
}





void 
Convolution::DeviceMaxP(){

}

void 
Convolution::DeviceMinP(){

}



__global__ void 
Cuda_MaxP(Pixel_t* pixels){

}

__global__ void 
Cuda_MinP(){

}