#pragma once 

#include <iostream>
#include <memory>
#include <assert.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "stb_image.h"
#include "stb_image_write.h"


struct Pixel
{
 unsigned char r, g, b, a;
};

struct Image_T{
public:
    unsigned char* GetData()    { return imageData; };
    const char* GetFPath()      { return image_path; };
    int* GetWidth()       { return &width; }
    int* Getheight()      { return &height; }
    int& Getchannels()      { return channels; }
    int* GetcomponentCount()      { return &componentCount; }

    void SetData(unsigned char* data_){ imageData = data_; };

    Image_T(const char* image_path_): image_path{ image_path_ }{}
private:
    int width;
    int height;
    int channels{4};
    int componentCount;
    unsigned char* imageData;
    const char* image_path;
};

class Convolution{
public:

    void initialize();
    Convolution(const char* image_path_){image = std::make_unique<Image_T>(image_path_);}
    ~Convolution(){}
private:
    void loadimage();
    std::unique_ptr<Image_T> image;
};

