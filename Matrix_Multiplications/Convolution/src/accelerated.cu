#include "accelerated.h"

#include "stb_image.h"
#include "stb_image_write.h"


__constant__ int MM_CUDA[3][3];
__constant__ int N; //item amount

void 
Convolution::DeviceConvCalc(){
    std::cout << __func__ << " being performed\r\n";
    
    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->Getheight();
    size_t width = *image->GetWidth();

    newImage->SetHeight( height-2 ); //because convolution does not calculate using border-pixels
    newImage->SetWidth( width-2 ); //because convolution does not calculate using border-pixels
    newImage->SetcomponentCount( *newImage->GetWidth() * *newImage->Getheight() * 4 );

    size_t num_pixels = *newImage->GetcomponentCount();

    Pixel_t* HIN_pixels;
    Pixel_t* HOUT_pixels;

    Pixel_t* DIN_pixels;
    Pixel_t* DOUT_pixels;

    cudaMemcpyToSymbol(MM_CUDA, &MM, 9 * sizeof(int));
    cudaMemcpyToSymbol(N, &num_pixels, sizeof(size_t));

    cudaMallocHost((void**)&HIN_pixels, ( sizeof(Pixel_t) * *image->GetcomponentCount()) ); //.input pixels
    cudaMallocHost((void**)&HOUT_pixels, sizeof(Pixel_t) * num_pixels ); //output pixels
    
    Pixel_t** pixels = image->GetPixelARR();
    size_t NI_CC = (*newImage->GetcomponentCount());
    for(size_t pxl{0}; pxl < NI_CC+1; pxl++){
        HIN_pixels[pxl] = *pixels[pxl];
    }
    memset(HOUT_pixels, 0, *newImage->GetcomponentCount() );

    cudaMalloc((void**)&DIN_pixels, ( sizeof(Pixel_t) * *image->GetcomponentCount()) );
    cudaMalloc((void**)&DOUT_pixels, ( sizeof(Pixel_t) * *newImage->GetcomponentCount()) );

    cudaMemcpy(DIN_pixels, HIN_pixels, NI_CC, cudaMemcpyHostToDevice);

    //original 307200 pixels -> 300 blocks * 1024 threads
    //output 304964 -> 1191.265 * 256 (1 extra block for remaining pixels)
    Cuda_ConvCalc<<< (*newImage->GetcomponentCount()+255)/ 256, 256 >>>(DIN_pixels);
    //stbi_write_png(newImage->GetFPath(), *newImage->GetWidth(), *newImage->Getheight(), 4, newImage->GetData(), 4 * (*newImage->GetWidth()) );
    printf("DONE\r\n");
}

__global__ void 
Cuda_ConvCalc(Pixel_t* pixels){    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    for (int ky = -1; ky <= 1; ky++){
        for (int kx = -1; kx <= 1; kx++){
            
        }
    }    
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