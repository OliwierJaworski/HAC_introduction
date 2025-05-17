
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




/**
 * @file accelerated_kernels.cu
 * @brief Device-side CUDA kernels and implementations for convolution and pooling.
 *
 * @details Defines separable convolution kernels (horizontal and vertical passes),
 *          2×2 pooling kernels, and the `Convolution::Device*` methods that manage
 *          device memory, kernel launches, timing, and output image writing.
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

/**
 * @var d_KernelW
 * @brief Device constant memory for horizontal convolution kernel weights.
 *
 * @details Stored in __constant__ memory for fast broadcast to all threads.
 */
__device__ __constant__ float d_KernelW[KERNEL_W];

/**
 * @var d_KernelH
 * @brief Device constant memory for vertical convolution kernel weights.
 *
 * @details Stored in __constant__ memory for fast broadcast to all threads.
 */
__device__ __constant__ float d_KernelH[KERNEL_H];

/**
 * @var threadPerBlockRow
 * @brief Block dimensions for the horizontal convolution kernel.
 *
 * @details Grid dimensions: TILE_W + 2*KERNEL_R columns × TILE_H rows per block.
 */
dim3 threadPerBlockRow(TILE_W + 2 * KERNEL_R, TILE_H);

/**
 * @var threadPerBlock
 * @brief Block dimensions for the vertical convolution kernel.
 *
 * @details Grid dimensions: TILE_W columns × TILE_H rows per block.
 */
dim3 threadPerBlock(TILE_W, TILE_H);

/**
 * @var blockGrids
 * @brief Grid dimensions for both convolution kernels.
 *
 * @details Computed as ceil(WIDTH / TILE_W) × ceil(HEIGHT / TILE_H).
 */
dim3 blockGrids((WIDTH + TILE_W - 1) / TILE_W, (HEIGHT + TILE_H - 1) / TILE_H);

/**
 * @kernel Cuda_ConvCalcRow
 * @brief Horizontal pass of separable convolution using shared memory tiling.
 *
 * @param in  Input image pixel buffer on device.
 * @param out Intermediate output buffer before vertical pass.
 *
 * @details Loads a TILE_H×(TILE_W+2*KERNEL_R) tile into shared memory with apron,
 *          applies 1D horizontal kernel d_KernelW, clamps results, and writes to out.
 */
__global__ void 
Cuda_ConvCalcRow(Pixel_t* in, Pixel_t* out){    
    __shared__ Pixel_t data[TILE_H][KERNEL_R + TILE_W + KERNEL_R];

    const int tileStart     = blockIdx.x * TILE_W;
    const int tileEnd       = tileStart + TILE_W - 1;
    const int apronStart    = tileStart - KERNEL_R;
    const int apronEnd      = tileEnd   + KERNEL_R;

    const int tileEndClamped    = min(tileEnd, WIDTH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int apronEndClamped   = min(apronEnd, WIDTH - 1);

    const int y = blockIdx.y * TILE_H + threadIdx.y;
    if (y >= HEIGHT) return;

    const int rowStart = y * WIDTH;

    // Load tile + apron into shared memory
    for (int lx = threadIdx.x; lx < TILE_W + 2 * KERNEL_R; lx += blockDim.x) {
        int loadPos = apronStart + lx;
        Pixel_t val = Pixel_t{0,0,0,0};
        if (loadPos >= apronStartClamped && loadPos <= apronEndClamped) {
            val = in[rowStart + loadPos];
        }
        data[threadIdx.y][lx] = val;
    }

    __syncthreads();

    // Apply horizontal kernel
    const int writePos = tileStart + threadIdx.x;
    if (writePos <= tileEndClamped) {
        int memPos = writePos - apronStart;
        float sum[3]{0, 0, 0};
        for (int c = -1; c <= 1; ++c) {
            int k = c + 1;
            sum[0] += data[threadIdx.y][memPos + c].r * d_KernelW[k];
            sum[1] += data[threadIdx.y][memPos + c].g * d_KernelW[k];
            sum[2] += data[threadIdx.y][memPos + c].b * d_KernelW[k];
        }
        auto clamp = [] __device__ (float v) {
            return v < 0 ? 0 : (v > 255 ? 255 : v);
        };
        out[rowStart + writePos] = Pixel_t{
            static_cast<unsigned char>(clamp(sum[0])),
            static_cast<unsigned char>(clamp(sum[1])),
            static_cast<unsigned char>(clamp(sum[2])),
            255
        };
    }
}

/**
 * @kernel Cuda_ConvCalcColumn
 * @brief Vertical pass of separable convolution using shared memory tiling.
 *
 * @param in  Intermediate buffer from horizontal pass.
 * @param out Final output pixel buffer (greyscale) on device.
 *
 * @details Loads a (TILE_W×TILE_H+2*KERNEL_R) vertical tile into shared memory with apron,
 *          applies 1D vertical kernel d_KernelH, converts to greyscale, clamps, and writes to out.
 */
__global__ void 
Cuda_ConvCalcColumn(Pixel_t* in, Pixel_t* out){    
    __shared__ Pixel_t data[TILE_W * (KERNEL_R + TILE_H + KERNEL_R)];

    const int tileStart       = blockIdx.y * TILE_H;
    const int tileEnd         = tileStart + TILE_H - 1;
    const int apronStart      = tileStart - KERNEL_R;
    const int apronEnd        = tileEnd   + KERNEL_R;
    const int tileEndClamp    = min(tileEnd, HEIGHT - 1);
    const int apronStartClamp = max(apronStart, 0);
    const int apronEndClamp   = min(apronEnd, HEIGHT - 1);

    const int columnStart = blockIdx.x * TILE_W + threadIdx.x;

    // Load vertical tile + apron
    int smemPos = threadIdx.y * TILE_W + threadIdx.x;
    int gmemPos = (apronStart + threadIdx.y) * WIDTH + columnStart;
    for (int y = apronStart; y <= apronEnd; y += blockDim.y) {
        data[smemPos] = ((y >= apronStartClamp && y <= apronEndClamp)
                         ? in[gmemPos]
                         : Pixel_t{0,0,0,0});
        smemPos += TILE_W * blockDim.y;
        gmemPos += WIDTH * blockDim.y;
    }

    __syncthreads();

    // Apply vertical kernel and greyscale
    smemPos = (threadIdx.y + KERNEL_R) * TILE_W + threadIdx.x;
    gmemPos = (tileStart + threadIdx.y) * WIDTH + columnStart;
    for (int y = tileStart; y <= tileEndClamp; y += blockDim.y) {
        float sum[3]{0,0,0};
        for (int c = -1; c <= 1; ++c) {
            int k = c + 1;
            sum[0] += data[smemPos + c * TILE_W].r * d_KernelH[k];
            sum[1] += data[smemPos + c * TILE_W].g * d_KernelH[k];
            sum[2] += data[smemPos + c * TILE_W].b * d_KernelH[k];
        }
        float grey = sum[0] * 0.2126f + sum[1] * 0.7152f + sum[2] * 0.0722f;
        auto clamp = [] __device__ (float v) {
            return v < 0 ? 0 : (v > 255 ? 255 : v);
        };
        unsigned char gs = clamp(grey);
        out[gmemPos] = Pixel_t{gs, gs, gs, 255};
        smemPos += TILE_W * blockDim.y;
        gmemPos += WIDTH * blockDim.y;
    }
}

/**
 * @brief Device implementation of full separable convolution.
 *
 * @details Copies image data to device, copies kernels to constant memory,
 *          launches horizontal and vertical passes with timing measurement,
 *          retrieves results from device, and writes output PNG.
 */
void 
Convolution::DeviceConvCalc(){
    std::cout << __func__ << " being performed\r\n";

    std::stringstream s; s << __func__ << ".png";
    std::string outPath = s.str();
    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->GetHeight();
    size_t width  = *image->GetWidth();
    newImage->SetHeight(height);
    newImage->SetWidth(width);
    newImage->SetComponentCount(height * width * 4);

    Pixel_t *HIN, *HOUT, *DIN, *DTMP, *DOUT;
    float MMH[3] = {1.0f, 1.0f, 1.0f};
    float MMW[3] = {-1.0f, 0.0f, 1.0f};
    cudaMemcpyToSymbol(d_KernelH, &MMH, KERNEL_H * sizeof(float));
    cudaMemcpyToSymbol(d_KernelW, &MMW, KERNEL_W * sizeof(float));

    cudaMallocHost((void**)&HIN,  sizeof(Pixel_t) * IMG_PIXELS);
    cudaMallocHost((void**)&HOUT, sizeof(Pixel_t) * IMG_PIXELS);
    Pixel_t** pix = image->GetPixelARR();
    for (size_t i = 0; i < IMG_PIXELS; ++i) {
        HIN[i] = *pix[i];
    }
    memset(HOUT, 0, IMG_COMPONENTS);

    cudaMalloc((void**)&DIN,  sizeof(Pixel_t) * IMG_PIXELS);
    cudaMalloc((void**)&DTMP, sizeof(Pixel_t) * IMG_PIXELS);
    cudaMalloc((void**)&DOUT, sizeof(Pixel_t) * IMG_PIXELS);

    cudaMemcpy(DIN, HIN, IMG_COMPONENTS, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float totalRow = 0, totalCol = 0, warmRow = 0, warmCol = 0;
    for (int i = 0; i < 11; ++i) {
        float t = 0;
        cudaEventRecord(start);
        Cuda_ConvCalcRow<<<blockGrids, threadPerBlockRow>>>(DIN, DTMP);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        (i == 0 ? warmRow : totalRow) += t;

        t = 0;
        cudaEventRecord(start);
        Cuda_ConvCalcColumn<<<blockGrids, threadPerBlock>>>(DTMP, DOUT);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        (i == 0 ? warmCol : totalCol) += t;
    }

    printf("Warmup Row: %.4f ms, Col: %.4f ms, Total: %.4f ms\n", warmRow, warmCol, warmRow + warmCol);
    printf("Avg Row: %.4f ms, Col: %.4f ms, Total: %.4f ms\n", totalRow/10, totalCol/10, (totalRow+totalCol)/10);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(HOUT, DOUT, IMG_COMPONENTS, cudaMemcpyDeviceToHost);
    stbi_write_png(newImage->GetFPath(), width, height, 4, HOUT, width * 4);

    cudaFreeHost(HIN);
    cudaFreeHost(HOUT);
    cudaFree(DIN);
    cudaFree(DTMP);
    cudaFree(DOUT);

    printf("DONE\r\n");
}

/**
 * @kernel Cuda_MaxP
 * @brief Device kernel for 2×2 max pooling.
 *
 * @param input  Input image pixel buffer.
 * @param output Downsampled output buffer.
 * @param width  Input image width.
 * @param height Input image height.
 */
__global__ void 
Cuda_MaxP(const Pixel_t* input, Pixel_t* output, int width, int height) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = (width + 1) / 2;
    if (out_x >= out_w || out_y >= (height + 1) / 2) return;

    Pixel_t max_p = {0,0,0,0};
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int x = out_x*2 + dx;
            int y = out_y*2 + dy;
            if (x < width && y < height) {
                Pixel_t p = input[y*width + x];
                max_p.r = max(max_p.r, p.r);
                max_p.g = max(max_p.g, p.g);
                max_p.b = max(max_p.b, p.b);
                max_p.a = max(max_p.a, p.a);
            }
        }
    }
    output[out_y*out_w + out_x] = max_p;
}

/**
 * @brief Device implementation of 2×2 max pooling.
 *
 * @details Allocates host/device buffers, launches `Cuda_MaxP`, retrieves results,
 *          and writes the output PNG.
 */
void 
Convolution::DeviceMaxP() {
    std::cout << __func__ << " being performed\r\n";
    
    size_t h = *image->GetHeight(), w = *image->GetWidth();
    size_t ow = (w + 1)/2, oh = (h + 1)/2;

    newImage = std::make_unique<Image_T>("DeviceMaxP.png");
    newImage->SetWidth(ow);
    newImage->SetHeight(oh);
    newImage->SetComponentCount(ow * oh * 4);

    Pixel_t *HIN, *HOUT, *DIN, *DOUT;
    size_t in_sz = w*h*sizeof(Pixel_t), out_sz = ow*oh*sizeof(Pixel_t);

    cudaMallocHost((void**)&HIN,  in_sz);
    cudaMallocHost((void**)&HOUT, out_sz);
    Pixel_t** pix = image->GetPixelARR();
    for (size_t i = 0; i < w*h; ++i) HIN[i] = *pix[i];

    cudaMalloc((void**)&DIN, in_sz);
    cudaMalloc((void**)&DOUT,out_sz);
    cudaMemcpy(DIN, HIN, in_sz, cudaMemcpyHostToDevice);

    dim3 bd(16,16), gd((ow+15)/16,(oh+15)/16);
    Cuda_MaxP<<<gd,bd>>>(DIN, DOUT, w, h);
    cudaDeviceSynchronize();

    cudaMemcpy(HOUT, DOUT, out_sz, cudaMemcpyDeviceToHost);
    stbi_write_png(newImage->GetFPath(), ow, oh, 4, HOUT, ow * 4);

    cudaFreeHost(HIN);
    cudaFreeHost(HOUT);
    cudaFree(DIN);
    cudaFree(DOUT);

    printf("DONE\n");
}

/**
 * @kernel Cuda_MinP
 * @brief Device kernel for 2×2 min pooling.
 *
 * @param input  Input image pixel buffer.
 * @param output Downsampled output buffer.
 * @param width  Input image width.
 * @param height Input image height.
 */
__global__ void 
Cuda_MinP(const Pixel_t* input, Pixel_t* output, int width, int height) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_w = (width + 1) / 2;
    if (out_x >= out_w || out_y >= (height + 1) / 2) return;

    Pixel_t min_p = {255,255,255,255};
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int x = out_x*2 + dx;
            int y = out_y*2 + dy;
            if (x < width && y < height) {
                Pixel_t p = input[y*width + x];
                min_p.r = min(min_p.r, p.r);
                min_p.g = min(min_p.g, p.g);
                min_p.b = min(min_p.b, p.b);
                min_p.a = min(min_p.a, p.a);
            }
        }
    }
    output[out_y*out_w + out_x] = min_p;
}

/**
 * @brief Device implementation of 2×2 min pooling.
 *
 * @details Allocates host/device buffers, launches `Cuda_MinP`, retrieves results,
 *          and writes the output PNG.
 */
void 
Convolution::DeviceMinP() {
    std::cout << __func__ << " being performed\r\n";

    size_t h = *image->GetHeight(), w = *image->GetWidth();
    size_t ow = (w + 1)/2, oh = (h + 1)/2;

    newImage = std::make_unique<Image_T>("DeviceMinP.png");
    newImage->SetWidth(ow);
    newImage->SetHeight(oh);
    newImage->SetComponentCount(ow * oh * 4);

    Pixel_t *HIN, *HOUT, *DIN, *DOUT;
    size_t in_sz = w*h*sizeof(Pixel_t), out_sz = ow*oh*sizeof(Pixel_t);

    cudaMallocHost((void**)&HIN,  in_sz);
    cudaMallocHost((void**)&HOUT, out_sz);
    Pixel_t** pix = image->GetPixelARR();
    for (size_t i = 0; i < w*h; ++i) HIN[i] = *pix[i];

    cudaMalloc((void**)&DIN, in_sz);
    cudaMalloc((void**)&DOUT,out_sz);
    cudaMemcpy(DIN, HIN, in_sz, cudaMemcpyHostToDevice);

    dim3 bd(16,16), gd((ow+15)/16,(oh+15)/16);
    Cuda_MinP<<<gd,bd>>>(DIN, DOUT, w, h);
    cudaDeviceSynchronize();

    cudaMemcpy(HOUT, DOUT, out_sz, cudaMemcpyDeviceToHost);
    stbi_write_png(newImage->GetFPath(), ow, oh, 4, HOUT, ow * 4);

    cudaFreeHost(HIN);
    cudaFreeHost(HOUT);
    cudaFree(DIN);
    cudaFree(DOUT);

    printf("DONE\n");
}
