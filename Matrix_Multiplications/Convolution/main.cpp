#include <iostream>
#include <accelerated.h>

void select(int& argc, char** argv);

int main(int argc, char** argv){ 
    int argcc = 4;
    char* args[] = { (char*)"program", (char*)"HOST", (char*)"Min_pooling", (char*)"./wall-e.png" };
    select(argcc,args);

    return 0;
}

void select(int& argc, char** argv){
    if (argc != 4 ) {
        std::cerr << "Usage: " << argv[0]   << "\n < medium > [ the device to perform the process on ]\n"
                                            << " < type > [ the process to be performed ] \n" 
                                            << " < filepath > [path with name of file to save to ]" << std::endl;
        return;
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