#include "accelerated.h"

#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void 
Convolution::initialize(const char* image_path_){
    image = std::make_unique<Image_T>(image_path_);
    image ->loadimage();
    image->loadPixels();
}

void 
Image_T::loadimage(){
    printf("Loading image file...\r\n");
    SetData( stbi_load( image_path, &width, &height, &channels, 4 ) ); 
    assert(GetData() && "[USER ERROR] : Provided image path does not exist.");  

    if(*Getheight() != 640 && *GetWidth() != 480 )
        assert("[USER ERROR] : Provided image resolution is not supported.");
}

void 
Convolution::HostConvCalc(){
    printf("HostConvCalc being performed...\r\n");
    newImage = std::make_unique<Image_T>("Hostconvolution.png");

    size_t height = *image->Getheight();
    size_t width = *image->GetWidth();

    newImage->SetHeight( height-2 ); //because convolution does not calculate using border-pixels
    newImage->SetWidth( width-2 ); //because convolution does not calculate using border-pixels
    newImage->SetcomponentCount( *newImage->GetWidth() * *newImage->Getheight() * 4 );

    newImage->AllocDataSize( *newImage->GetcomponentCount() );

    for (int y = 1; y < height-1; y++){
        
        for (int x = 1; x < width-1; x++){
            float acc_rgb[3]{0,0,0};

            for (int ky = -1; ky <= 1; ky++){

                for (int kx = -1; kx <= 1; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);
                    int weight = kernel[ky + 1][kx + 1];

                    acc_rgb[0] += pixel.r * weight;
                    acc_rgb[1] += pixel.g * weight;
                    acc_rgb[2] += pixel.b * weight;                    
                }
            }
            acc_rgb[0] = std::abs(acc_rgb[0]);
            acc_rgb[1] = std::abs(acc_rgb[1]);
            acc_rgb[2] = std::abs(acc_rgb[2]);

            float greyscaled = acc_rgb[0]* 0.2126f + acc_rgb[1]* 0.7152f + acc_rgb[2]* 0.0722f;
            checkbounds(greyscaled);

            int dst_x = x - 1;
            int dst_y = y - 1;

            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[ (dst_y * (*newImage->GetWidth()) + dst_x) * 4 ];

            dst->r = greyscaled;
            dst->g = greyscaled; 
            dst->b = greyscaled;
            dst->a = 255;
        }
    }

    stbi_write_png(newImage->GetFPath(), *newImage->GetWidth(), *newImage->Getheight(), 4, newImage->GetData(), 4 * (*newImage->GetWidth()) );
    printf("DONE\r\n");
}

unsigned char 
Convolution::checkbounds(float val) {
    return val < 0 ? 0 : (val > 255 ? 255 : val); //cool effect
}

void 
Convolution::perform(Actors actor){
    switch(actor){
        case Host:
            HostConvCalc();
            break;
        case Device:

            break;
        default:
            std::cout << "user provided unsupported perform() option\n";
            exit(1);
            break;
    }
}

void 
Image_T::loadPixels(){
    for (int y = 0; y < height; y++){
        
        for (int x = 0; x < width; x++){
          Pixel_t* ptrPixel = (Pixel_t*)&imageData[y * width * 4 + 4 * x];
          pixels.push_back(ptrPixel);    
        }
    }
}

