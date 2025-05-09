#pragma once 

#include <iostream>
#include <memory>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "vector"

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

    void SetData(unsigned char* data_)  { imageData = data_; };
    void SetWidth(int width_)  { width = width_; };
    void SetHeight(int height_)  { height = height_; };
    void SetcomponentCount(int componentCount_)  { componentCount = componentCount_; };
    void AllocDataSize(size_t size){ imageData = (unsigned char*) malloc(componentCount); };

    void loadimage();
    void loadPixels();

    Image_T(const char* image_path_): image_path{ image_path_ }{}
    ~Image_T(){ free(imageData); }
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
    void perform(Actors actor);

    Convolution(const char* image_path_){ initialize( image_path_ ); }
    ~Convolution(){}
private:
    void initialize(const char* image_path_);
    unsigned char checkbounds(float val);
    void HostConvCalc();

    std::unique_ptr<Image_T> newImage;
    std::unique_ptr<Image_T> image;
    int kernel[3][3]{1,0,-1,
                          1,0,-1,
                          1,0,-1};

    int MM[3][3]{-1, 0, 1,
                 -1, 0, 1,
                 -1, 0, 1};
};

