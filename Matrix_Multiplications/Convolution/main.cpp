#include <iostream>
#include <accelerated.h>

void select(int& argc, char** argv);

int main(int argc, char** argv){ 

    select(argc,argv);

    return 0;
}

void select(int& argc, char** argv){
    if (argc != 4 ) {
        std::cerr << "Usage: " << argv[0]   << " < medium > [ the device to perform the process on ]\n"
                                            << " < type > [ the process to be performed ] \n" 
                                            << " < filepath > [path with name of file to save to ]" << std::endl;
    }  

    Convolution& conv = Convolution::instance( argv[3] );

    std::unordered_map <std::string, Actors> actortype{
        {"HOST", Actors::Host},
        {"DEVICE", Actors::Device}
    };

    std::unordered_map <std::string, std::function<void(Actors)>> convtype{
        {"Prewitt_Edge_detection", [&](Actors actor){ conv.ConvCalc(actor); }},
        {"Max_pooling", [&](Actors actor){ conv.MaxP(actor); }},
        {"Min_pooling", [&](Actors actor){ conv.MinP(actor); }}
    };
   
    convtype[argv[2]]( actortype[argv[1]] );
}