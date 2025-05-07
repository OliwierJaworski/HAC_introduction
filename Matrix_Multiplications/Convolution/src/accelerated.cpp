#include "accelerated.h"

#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void 
Convolution::initialize(){

}

void 
Convolution::loadimage(){
    printf("Loading image file...\r\n");
    image->SetData( stbi_load(  image->GetFPath(),
                                image->GetWidth(),
                                image->Getheight(),
                                image->GetcomponentCount(), image->Getchannels()) ); 
    assert(image->GetData() && "[USER ERROR] : Provided image path does not exist.");  

    if(*image->Getheight() != 640 && *image->GetWidth() != 480 )
        assert("[USER ERROR] : Provided image resolution is not supported.");
}
void 
Convolution::HostConvCalc(){
    int& height = *image->Getheight();
    int& width = *image->GetWidth();

    for (int y = 0; y < height; y++){
        
        for (int x = 0; x < width; x++){

        }
    }
}

void 
Convolution::perform(Actors actor){
    switch(actor){
        case Host:

            break;
        case Device:

            break;
        default:
            std::cout << "user provided unsupported perform() option\n";
            exit(1);
            break;
    }
}