#include "accelerated.h"

#include "stb_image.h"
#include "stb_image_write.h"

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

    newImage->AllocDataSize( *newImage->GetcomponentCount() );

    stbi_write_png(newImage->GetFPath(), *newImage->GetWidth(), *newImage->Getheight(), 4, newImage->GetData(), 4 * (*newImage->GetWidth()) );
    printf("DONE\r\n");
}

void 
Convolution::DeviceMaxP(){

}

void 
Convolution::DeviceMinP(){

}

__global__ void 
Cuda_ConvCalc(){

}

__global__ void 
Cuda_MaxP(){

}

__global__ void 
Cuda_MinP(){

}