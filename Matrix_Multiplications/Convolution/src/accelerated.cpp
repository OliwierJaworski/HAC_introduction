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
    SetData( stbi_load(  GetFPath(),
                                GetWidth(),
                                Getheight(),
                                GetcomponentCount(), Getchannels()) ); 
    assert(GetData() && "[USER ERROR] : Provided image path does not exist.");  

    if(*Getheight() != 640 && *GetWidth() != 480 )
        assert("[USER ERROR] : Provided image resolution is not supported.");
}

void 
Convolution::HostConvCalc(){
    printf("HostConvCalc being performed...\r\n");
    newImage = std::make_unique<Image_T>("Hostconvolution.png");

    size_t height = *image->Getheight()/4;
    size_t width = *image->GetWidth()/4;

    newImage->SetHeight( height-2 );
    newImage->SetWidth( width-2 );
    newImage->SetcomponentCount( 4*(width*height) );
    newImage->AllocDataSize( *newImage->GetcomponentCount() );
    for (int y = 1; y < height-1; y++){
        
        for (int x =1; x < width-1; x++){
            float acc_rgb[4]{0,0,0,0};

            for (int ky = -1; ky <= 1; ky++){

                for (int kx = -1; kx <= 1; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);
                    int weight = convolution[ky + 1][kx + 1];
                    acc_rgb[0] += pixel.r * weight;
                    acc_rgb[1] += pixel.g * weight;
                    acc_rgb[2] += pixel.b * weight;
                    acc_rgb[3] += pixel.a * weight;
                }
            }

            int dst_x = x - 1;
            int dst_y = y - 1;

            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[(dst_y * (width - 2) + dst_x) * 4];
            dst->r = checkbounds(acc_rgb[0]);
            dst->g = checkbounds(acc_rgb[1]); 
            dst->b = checkbounds(acc_rgb[2]);
            dst->a = checkbounds(acc_rgb[3]);
        }
    }

    stbi_write_png(newImage->GetFPath(), width, height, 4, newImage->GetData(), 4 * width);
    printf("DONE\r\n");
}

unsigned char 
Convolution::checkbounds(int val) {
    return val < 0 ? 0 : (val > 255 ? 255 : val);
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