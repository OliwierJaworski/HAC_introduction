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
Convolution::ConvCalc(Actors actor){

    switch(actor){
        case Host:
                Convolution::instance().HostConvCalc();
            break;

        case Device:
                Convolution::instance().DeviceConvCalc();
            break;
        default:
            std::cout << "user provided unsupported perform() option\n";
            exit(1);
            break;
    }
}
void 
Convolution::MaxP(Actors actor){

    switch(actor){
        case Host:
               Convolution::instance().HostMaxP();
            break;

        case Device:
                Convolution::instance().DeviceMaxP();
            break;
        default:
            std::cout << "user provided unsupported perform() option\n";
            exit(1);
            break;
    } 
}

void 
Convolution::MinP(Actors actor){

    switch(actor){
        case Host:
                Convolution::instance().HostMinP();
            break;

        case Device:
                Convolution::instance().DeviceMinP();
            break;
        default:
            std::cout << "user provided unsupported perform() option\n";
            exit(1);
            break;
    }
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

void 
Convolution::HostMaxP(){
    //160x120
    printf("HostMaxP being performed...\r\n");
    newImage = std::make_unique<Image_T>("HostMaxP.png");

    size_t height = *image->Getheight();
    size_t width = *image->GetWidth();

    newImage->SetHeight( height / 2); 
    newImage->SetWidth( width / 2); 
    newImage->SetcomponentCount( *newImage->GetWidth() * *newImage->Getheight() * 4 );

    newImage->AllocDataSize( *newImage->GetcomponentCount() );
    
    for (int y = 0; y < height; y+=2){
                    
        for (int x = 0; x < width; x+=2){
            unsigned char max_rgb[3]{0,0,0};

            for (int ky = 0; ky < 2; ky++){

                for (int kx = 0; kx < 2; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);

                    max_rgb[0] = (max_rgb[0] < pixel.r)? pixel.r : max_rgb[0];
                    max_rgb[1] = (max_rgb[1] < pixel.g)? pixel.g : max_rgb[1];
                    max_rgb[2] = (max_rgb[2] < pixel.b)? pixel.b : max_rgb[2];
                }
            }; 
            
            int dst_x = x /2;
            int dst_y = y /2;
            
            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[ (dst_y * (*newImage->GetWidth()) + dst_x) * 4 ];
            dst->r = max_rgb[0];
            dst->g = max_rgb[1]; 
            dst->b = max_rgb[2];
            dst->a = 255;
        }
    }

    stbi_write_png(newImage->GetFPath(), *newImage->GetWidth(), *newImage->Getheight(), 4, newImage->GetData(), 4 * (*newImage->GetWidth()) );
    printf("DONE\r\n");
}

void 
Convolution::HostMinP(){
    //160x120
    printf("HostMinP being performed...\r\n");
    newImage = std::make_unique<Image_T>("HostMinP.png");

    size_t height = *image->Getheight();
    size_t width = *image->GetWidth();

    newImage->SetHeight( height / 2); 
    newImage->SetWidth( width / 2); 
    newImage->SetcomponentCount( *newImage->GetWidth() * *newImage->Getheight() * 4 );

    newImage->AllocDataSize( *newImage->GetcomponentCount() );
    
    for (int y = 0; y < height; y+=2){
                    
        for (int x = 0; x < width; x+=2){
            unsigned char min_rgb[3]{255,255,255};

            for (int ky = 0; ky < 2; ky++){

                for (int kx = 0; kx < 2; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);

                    min_rgb[0] = (min_rgb[0] > pixel.r)? pixel.r : min_rgb[0];
                    min_rgb[1] = (min_rgb[1] > pixel.g)? pixel.g : min_rgb[1];
                    min_rgb[2] = (min_rgb[2] > pixel.b)? pixel.b : min_rgb[2];
                }
            }; 
            
            int dst_x = x /2;
            int dst_y = y /2;

            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[ (dst_y * (*newImage->GetWidth()) + dst_x) * 4 ];
            dst->r = min_rgb[0];
            dst->g = min_rgb[1]; 
            dst->b = min_rgb[2];
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
Image_T::loadPixels(){
    for (int y = 0; y < height; y++){
        
        for (int x = 0; x < width; x++){
          Pixel_t* ptrPixel = (Pixel_t*)&imageData[y * width * 4 + 4 * x];
          pixels.push_back(ptrPixel);    
        }
    }
}

Convolution& 
Convolution::instance(std::optional<char*> path){
    static Convolution* conv = nullptr;

    if (!conv && path.has_value()) {
        conv = new Convolution(path.value());
    } else if (!conv && !path.has_value()) {
        throw std::runtime_error("Convolution::instance() needs a path on first call");
    }
    return *conv;
}