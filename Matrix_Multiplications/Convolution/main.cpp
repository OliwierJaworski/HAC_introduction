#include <iostream>
#include <accelerated.h>

void select(int& argc, char** argv);

int main(int argc, char** argv){ 
    

    Convolution conv{ argv[1] };
    conv.perform(Actors::Host);
    return 0;
}
//appname -Host -convolution -Imagepath
//appname -medium -type -filepath

void select(int& argc, char** argv){
    if (argc != 4 ) {
        std::cerr << "Usage: " << argv[0]   << " < medium > [ the device to perform the process on ]\n"
                                            << " < type > [ the process to be performed ] \n" 
                                            << " < filepath > [path with name of file to save to ]" << std::endl;
    }   
}