#include "accelerated.h"

#include "stb_image.h"
#include "stb_image_write.h"


#define WIDTH 640
#define HEIGHT 480

#define TILE_W 32
#define TILE_H 16

#define KERNEL_W 3
#define KERNEL_H 3

#define IMG_PIXELS 307200 
#define IMG_COMPONENTS (IMG_PIXELS * 4)

__device__ __constant__ float d_KernelW[KERNEL_W];
__device__ __constant__ float d_KernelH[KERNEL_H];


dim3 threadPerBlock(TILE_W, TILE_H); // 32x16 = 512 threads per block
dim3 blockGrids((WIDTH + TILE_W - 1) / TILE_W, (HEIGHT + TILE_H - 1) / TILE_H);

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
    Pixel_t* DOUT_pixels;

    int MMH[3] = {-1,0,1};
    int MMW[3] = {-1,-1,-1};
    cudaMemcpyToSymbol(d_KernelH,&MMH , KERNEL_H * sizeof(int));
    cudaMemcpyToSymbol(d_KernelW, &MMW, KERNEL_W * sizeof(int));

    cudaMallocHost((void**)&HIN_pixels, sizeof(Pixel_t) * IMG_COMPONENTS ); 
    cudaMallocHost((void**)&HOUT_pixels, sizeof(Pixel_t) * IMG_COMPONENTS );
    
    Pixel_t** pixels = image->GetPixelARR();

    for(size_t pxl{0}; pxl < IMG_COMPONENTS +1; pxl++){
        HIN_pixels[pxl] = *pixels[pxl];
    }
    memset(HOUT_pixels, 0, *newImage->GetcomponentCount() );

    cudaMalloc((void**)&DIN_pixels, sizeof(Pixel_t) * IMG_COMPONENTS );
    cudaMalloc((void**)&DOUT_pixels, sizeof(Pixel_t) * IMG_COMPONENTS );

    cudaMemcpy(DIN_pixels, HIN_pixels, IMG_COMPONENTS, cudaMemcpyHostToDevice);
    Cuda_ConvCalcRow<<<blockGrids, threadPerBlock>>>(HIN_pixels,DOUT_pixels);

    printf("DONE\r\n");
}

__global__ void 
Cuda_ConvCalcRow(Pixel_t* in, Pixel_t* out){    
    
}

__global__ void 
Cuda_ConvCalcColumn(Pixel_t* in, Pixel_t* out){    

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