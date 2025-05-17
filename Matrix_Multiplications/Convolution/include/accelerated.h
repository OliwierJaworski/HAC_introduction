#pragma once 

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
 * @file accelerated.cpp
 * @class Convolution
 * @brief class for performing convolution and pooling operations on images
 *
 * @details - manages image pipeline initialization from file paths
 * @details - dispatches operations to CPU or GPU based on Actors enum
 * @details - implements Host and Device methods for convolution and pooling
 */

#include "accelerated.h"

#define STB_IMAGE_IMPLEMENTATION 
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

/**
 * @brief Initializes the convolution pipeline with an image path.
 *
 * @param image_path_ Path to the input image file.
 *
 * @details Creates a new `Image_T` instance, loads image data into memory,
 *          and populates the pixel pointer array.
 */
void 
Convolution::initialize(const char* image_path_){
    image = std::make_unique<Image_T>(image_path_);
    image->loadimage();
    image->loadPixels();
}

/**
 * @brief Loads image data from file into memory.
 *
 * @details Uses stb_image to read raw pixels into `imageData`.
 *          Asserts if the file does not exist or resolution is unsupported.
 */
void 
Image_T::loadimage(){
    printf("Loading image file...\r\n");
    SetData(stbi_load(image_path, &width, &height, &channels, 4)); 
    assert(GetData() && "[USER ERROR] : Provided image path does not exist.");  

    if(*Getheight() != 640 && *GetWidth() != 480 )
        assert("[USER ERROR] : Provided image resolution is not supported.");
}

/**
 * @brief Dispatches the convolution operation based on execution context.
 *
 * @param actor  `Actors::Host` for CPU or `Actors::Device` for GPU.
 */
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
    }
}

/**
 * @brief Dispatches the max-pooling operation based on execution context.
 *
 * @param actor  `Actors::Host` for CPU or `Actors::Device` for GPU.
 */
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
    } 
}

/**
 * @brief Dispatches the min-pooling operation based on execution context.
 *
 * @param actor  `Actors::Host` for CPU or `Actors::Device` for GPU.
 */
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
    }
}

/**
 * @brief Performs 2D convolution on the host (CPU).
 *
 * @details Applies the 3×3 kernel to each pixel (excluding borders),
 *          computes greyscale result, allocates output image, and writes PNG.
 */
void 
Convolution::HostConvCalc(){
    std::cout << __func__ << " being performed\r\n";

    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->Getheight();
    size_t width = *image->GetWidth();

    newImage->SetHeight( height-2 ); //because convolution does not calculate using border-pixels
    newImage->SetWidth( width-2 );   //because convolution does not calculate using border-pixels
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

/**
 * @brief Performs 2×2 max-pooling on the host (CPU).
 *
 * @details Divides into 2×2 blocks, picks maximum RGB, writes downsampled PNG.
 */
void 
Convolution::HostMaxP(){
    //160x120
    std::cout << __func__ << " being performed\r\n";

    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

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

/**
 * @brief Performs 2×2 min-pooling on the host (CPU).
 *
 * @details Divides into 2×2 blocks, picks minimum RGB, writes downsampled PNG.
 */
void 
Convolution::HostMinP(){
    //160x120
    std::cout << __func__ << " being performed\r\n";
    
    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

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

/**
 * @brief Clamps a float to the [0,255] range and returns as unsigned char.
 *
 * @param val  Input float value.
 * @return     0 if val<0, 255 if val>255, else static_cast<unsigned char>(val).
 */
unsigned char 
Convolution::checkbounds(float val) {
    return val < 0 ? 0 : (val > 255 ? 255 : val); //cool effect
}

/**
 * @brief Populates pixel pointers from raw image data.
 *
 * @details Iterates each (x,y) and stores pointer to Pixel_t in `pixels`.
 */
void 
Image_T::loadPixels(){
    for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
            Pixel_t* ptrPixel = (Pixel_t*)&imageData[y * width * 4 + 4 * x];
            pixels.push_back(ptrPixel);    
        }
    }
}

/**
 * @brief Returns the singleton Convolution instance.
 *
 * @param path  Optional path on first call to create instance.
 * @return      Reference to the global `Convolution` object.
 *
 * @details Constructs on first call with path, throws if no path provided.
 */
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