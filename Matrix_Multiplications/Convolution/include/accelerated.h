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

#include <iostream>
#include <memory>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "vector"
#include <cstdarg>
#include <unordered_map>
#include <functional>
#include <optional>
#include <sstream>

enum Actors{
    Host,
    Device
};

struct Pixel_t
{
 unsigned char r, g, b, a;
};

struct Image_T{
public:
    unsigned char* GetData()            { return imageData; };
    const char* GetFPath()              { return image_path; };
    int* GetWidth()                     { return &width; }
    int* Getheight()                    { return &height; }
    int& Getchannels()                  { return channels; }
    int* GetcomponentCount()            { return &componentCount; }
    Pixel_t& GetPixel(size_t index)     { return *pixels[index];}
    Pixel_t** GetPixelARR()     { return pixels.data();}

    void SetData(unsigned char* data_)  { imageData = data_; };
    void SetWidth(int width_)  { width = width_; };
    void SetHeight(int height_)  { height = height_; };
    void SetcomponentCount(int componentCount_)  { componentCount = componentCount_; };
    void AllocDataSize(size_t size){ imageData = (unsigned char*) malloc(size); };

    void loadimage();
    void loadPixels();

    Image_T(const char* image_path_): image_path{ image_path_ }{}
    ~Image_T(){ if(imageData != nullptr) free(imageData); }
private:
    std::vector<Pixel_t*> pixels;
    int width;
    int height;
    int channels{4};
    int componentCount;
    unsigned char* imageData;
    const char* image_path;
};

class Convolution{
public:
    void ConvCalc(Actors actor);
    void MaxP(Actors actor);
    void MinP(Actors actor);

    static Convolution& instance(std::optional<char*> path = std::nullopt );

    Convolution(const char* image_path_){ initialize( image_path_ ); }
    ~Convolution(){}
private:
    void initialize(const char* image_path_);
    unsigned char checkbounds(float val);

    void HostConvCalc();
    void HostMaxP();
    void HostMinP();

    void DeviceConvCalc();
    void DeviceMaxP();
    void DeviceMinP();

    std::unique_ptr<Image_T> newImage;
    std::unique_ptr<Image_T> image;
    int kernel[3][3]{1,0,-1,
                          1,0,-1,
                          1,0,-1};

    int MM[3][3]{-1, 0, 1,
                 -1, 0, 1,
                 -1, 0, 1};
};

