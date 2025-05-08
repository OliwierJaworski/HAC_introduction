#include <iostream>
#include <accelerated.h>

int main(int argc, char** argv){ 
    std::cout << "Hello, from Convolution_pro!\n";
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
    return 1;
    }

    Convolution conv{ argv[1] };
    conv.perform(Actors::Host);
    return 0;
}
