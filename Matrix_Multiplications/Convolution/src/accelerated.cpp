#include "accelerated.h"

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
}

