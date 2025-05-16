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


dim3 threadPerBlock(TILE_W, TILE_H); // 32x16 = 512 threads per block
dim3 blockGrids((WIDTH + TILE_W - 1) / TILE_W, (HEIGHT + TILE_H - 1) / TILE_H);


__global__ void 
Cuda_ConvCalcRow(Pixel_t* in, Pixel_t* out){    
    __shared__ Pixel_t data[KERNEL_R + TILE_W + KERNEL_R];

    const int tileStart     = blockIdx.x * TILE_W;
    const int tileEnd       = tileStart + TILE_W -1;
    const int apronStart    = tileStart - KERNEL_R;
    const int apronEnd      = tileEnd + KERNEL_R;

    const int tileEndClamped    = min(tileEnd, WIDTH -1);
    const int apronStartClamped = max(apronStart, 0);
    const int apronEndClamped   = min(apronEnd, WIDTH -1);
    
    const int rowStart = blockIdx.y * WIDTH;

    const int unaligned = tileStart - KERNEL_R; //= -1 for 0-1;
    const int apronStartAligned = unaligned & ~15; // -1 = 0xFFFFFFFF | -1 & ~15 = 0xFFFFFFFF & 0xFFFFFFF0 = 0xFFFFFFF0 = -16

    const int loadPos = apronStartAligned + threadIdx.x;

    if(loadPos >= apronStart){
        const int MemPos = loadPos - apronStart;

        data[MemPos] = ( (loadPos >= apronStartClamped) && (loadPos <= apronEndClamped) ) ? in[rowStart + loadPos]: Pixel_t{0,0,0,0};
    }

    __syncthreads();

    const int writePos = tileStart + threadIdx.x;

    if(writePos <= tileEndClamped){
        const int MemPos = writePos - apronStart;
        float sum[3]{0,0,0};

        for(int c{-1}; c < 2; c++){
            int k = c + 1;
            sum[0] += data[MemPos+c].r * d_KernelW[k];
            sum[1] += data[MemPos+c].g * d_KernelW[k];
            sum[2] += data[MemPos+c].b * d_KernelW[k];
        }

        auto clamp = [] __device__ (float val) {
            return val < 0 ? 0 : (val > 255 ? 255 : val);
        };

        Pixel_t result;
        result.r = clamp(sum[0]);
        result.g = clamp(sum[1]);
        result.b = clamp(sum[2]);
        result.a = 255;

        out[rowStart + writePos] = result;
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

    Cuda_ConvCalcRow<<<blockGrids, threadPerBlock>>>(DIN_pixels,DTMP_RSLT);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    Cuda_ConvCalcColumn<<<blockGrids, threadPerBlock>>>(DTMP_RSLT, DOUT_pixels);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(HOUT_pixels, DOUT_pixels, IMG_COMPONENTS, cudaMemcpyDeviceToHost);

    stbi_write_png(newImage->GetFPath(), width, height, 4, HOUT_pixels, width * 4);

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