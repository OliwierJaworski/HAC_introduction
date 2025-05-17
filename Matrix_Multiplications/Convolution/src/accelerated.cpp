/**
 * @brief Host-side implementations of image pipeline initialization, convolution, and pooling.
 *
 * @details Defines methods for:
 *          - initializing the `Convolution` singleton with an image
 *          - loading image data and pixel pointers (`Image_T::loadimage`, `loadPixels`)
 *          - dispatching operations to host or device (`ConvCalc`, `MaxP`, `MinP`)
 *          - CPU-based convolution (`HostConvCalc`) and pooling (`HostMaxP`, `HostMinP`)
 *          - utility functions for clamping and singleton access
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
 * @details Creates a new `Image_T` instance for the given file path,
 *          loads the image data into memory, and populates the pixel pointer array.
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
 * @details Uses `stb_image` to read raw pixel data into `imageData`.
 *          Asserts if the provided path is invalid or if the resolution is unsupported.
 */
void 
Image_T::loadimage(){
    printf("Loading image file...\r\n");
    SetData(stbi_load(image_path, &width, &height, &channels, 4)); 
    assert(GetData() && "[USER ERROR] : Provided image path does not exist.");  

    if (*GetHeight() != 640 && *GetWidth() != 480)
        assert("[USER ERROR] : Provided image resolution is not supported.");
}

/**
 * @brief Dispatches the convolution operation based on the execution context.
 *
 * @param actor  Execution context (`Actors::Host` or `Actors::Device`).
 *
 * @details Calls `HostConvCalc()` for CPU or `DeviceConvCalc()` for GPU processing.
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
 * @brief Dispatches the max-pooling operation based on the execution context.
 *
 * @param actor  Execution context (`Actors::Host` or `Actors::Device`).
 *
 * @details Calls `HostMaxP()` for CPU or `DeviceMaxP()` for GPU processing.
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
 * @brief Dispatches the min-pooling operation based on the execution context.
 *
 * @param actor  Execution context (`Actors::Host` or `Actors::Device`).
 *
 * @details Calls `HostMinP()` for CPU or `DeviceMinP()` for GPU processing.
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
 * @details Applies the 3×3 convolution kernel to each pixel (excluding borders),
 *          computes greyscale output, allocates the result image, and writes it as a PNG.
 */
void 
Convolution::HostConvCalc(){
    std::cout << __func__ << " being performed\r\n";

    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->GetHeight();
    size_t width  = *image->GetWidth();

    // Convolution reduces dimensions by kernel radius
    newImage->SetHeight(height - 2);
    newImage->SetWidth(width - 2);
    newImage->SetComponentCount(*newImage->GetWidth() * *newImage->GetHeight() * 4);
    newImage->AllocDataSize(*newImage->GetComponentCount());

    for (int y = 1; y < height - 1; y++){
        for (int x = 1; x < width - 1; x++){
            float acc_rgb[3]{0,0,0};

            // Apply 3×3 kernel
            for (int ky = -1; ky <= 1; ky++){
                for (int kx = -1; kx <= 1; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);
                    int weight = kernel[ky + 1][kx + 1];
                    acc_rgb[0] += pixel.r * weight;
                    acc_rgb[1] += pixel.g * weight;
                    acc_rgb[2] += pixel.b * weight;
                }
            }

            // Absolute value and greyscale conversion
            acc_rgb[0] = std::abs(acc_rgb[0]);
            acc_rgb[1] = std::abs(acc_rgb[1]);
            acc_rgb[2] = std::abs(acc_rgb[2]);
            float greyscaled = acc_rgb[0]*0.2126f + acc_rgb[1]*0.7152f + acc_rgb[2]*0.0722f;
            checkbounds(greyscaled);

            int dst_x = x - 1;
            int dst_y = y - 1;
            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[(dst_y * (*newImage->GetWidth()) + dst_x) * 4];
            dst->r = greyscaled;
            dst->g = greyscaled;
            dst->b = greyscaled;
            dst->a = 255;
        }
    }

    stbi_write_png(newImage->GetFPath(),
                   *newImage->GetWidth(),
                   *newImage->GetHeight(),
                   4,
                   newImage->GetData(),
                   4 * (*newImage->GetWidth()));
    printf("DONE\r\n");
}

/**
 * @brief Performs 2×2 max pooling on the host (CPU).
 *
 * @details Divides the image into 2×2 blocks, picks the maximum RGB values,
 *          allocates the result image, and writes it as a PNG.
 */
void 
Convolution::HostMaxP(){
    std::cout << __func__ << " being performed\r\n";

    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->GetHeight();
    size_t width  = *image->GetWidth();

    newImage->SetHeight(height / 2);
    newImage->SetWidth(width / 2);
    newImage->SetComponentCount(*newImage->GetWidth() * *newImage->GetHeight() * 4);
    newImage->AllocDataSize(*newImage->GetComponentCount());
    
    for (int y = 0; y < height; y += 2){
        for (int x = 0; x < width; x += 2){
            unsigned char max_rgb[3]{0,0,0};
            for (int ky = 0; ky < 2; ky++){
                for (int kx = 0; kx < 2; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);
                    max_rgb[0] = std::max(max_rgb[0], pixel.r);
                    max_rgb[1] = std::max(max_rgb[1], pixel.g);
                    max_rgb[2] = std::max(max_rgb[2], pixel.b);
                }
            }
            int dst_x = x / 2;
            int dst_y = y / 2;
            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[(dst_y * (*newImage->GetWidth()) + dst_x) * 4];
            dst->r = max_rgb[0];
            dst->g = max_rgb[1];
            dst->b = max_rgb[2];
            dst->a = 255;
        }
    }

    stbi_write_png(newImage->GetFPath(),
                   *newImage->GetWidth(),
                   *newImage->GetHeight(),
                   4,
                   newImage->GetData(),
                   4 * (*newImage->GetWidth()));
    printf("DONE\r\n");
}

/**
 * @brief Performs 2×2 min pooling on the host (CPU).
 *
 * @details Divides the image into 2×2 blocks, picks the minimum RGB values,
 *          allocates the result image, and writes it as a PNG.
 */
void 
Convolution::HostMinP(){
    std::cout << __func__ << " being performed\r\n";
    
    std::stringstream s;
    s << __func__ << ".png";
    std::string outPath = s.str();

    newImage = std::make_unique<Image_T>(outPath.c_str());

    size_t height = *image->GetHeight();
    size_t width  = *image->GetWidth();
    newImage->SetHeight(height / 2);
    newImage->SetWidth(width / 2);
    newImage->SetComponentCount(*newImage->GetWidth() * *newImage->GetHeight() * 4);
    newImage->AllocDataSize(*newImage->GetComponentCount());
    
    for (int y = 0; y < height; y += 2){
        for (int x = 0; x < width; x += 2){
            unsigned char min_rgb[3]{255,255,255};
            for (int ky = 0; ky < 2; ky++){
                for (int kx = 0; kx < 2; kx++){
                    Pixel_t& pixel = image->GetPixel((x + kx) + (y + ky) * width);
                    min_rgb[0] = std::min(min_rgb[0], pixel.r);
                    min_rgb[1] = std::min(min_rgb[1], pixel.g);
                    min_rgb[2] = std::min(min_rgb[2], pixel.b);
                }
            }
            int dst_x = x / 2;
            int dst_y = y / 2;
            Pixel_t* dst = (Pixel_t*)&newImage->GetData()[(dst_y * (*newImage->GetWidth()) + dst_x) * 4];
            dst->r = min_rgb[0];
            dst->g = min_rgb[1];
            dst->b = min_rgb[2];
            dst->a = 255;
        }
    }

    stbi_write_png(newImage->GetFPath(),
                   *newImage->GetWidth(),
                   *newImage->GetHeight(),
                   4,
                   newImage->GetData(),
                   4 * (*newImage->GetWidth()));
    printf("DONE\r\n");
}

/**
 * @brief Clamps a floating-point value to the [0,255] range.
 *
 * @param val  The input floating-point value.
 * @return     The clamped value as `unsigned char`.
 *
 * @details Returns 0 if `val < 0`, 255 if `val > 255`, else `val`.
 */
unsigned char 
Convolution::checkbounds(float val) {
    return val < 0 ? 0 : (val > 255 ? 255 : static_cast<unsigned char>(val));
}

/**
 * @brief Populates the pixel pointer array from raw image data.
 *
 * @details Iterates over each pixel position in `imageData` and
 *          stores pointers to `Pixel_t` in the `pixels` vector.
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
 * @brief Returns the singleton instance of `Convolution`.
 *
 * @param path  Optional image path on first call to create the instance.
 * @return      Reference to the global `Convolution` instance.
 *
 * @details Creates a new `Convolution` instance on first call if `path` is provided,
 *          throws `std::runtime_error` if called first without a path.
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
