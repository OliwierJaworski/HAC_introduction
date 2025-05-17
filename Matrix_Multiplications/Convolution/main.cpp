#include <iostream>
#include <accelerated.h>

/**
 * @file main.cpp
 * @brief Entry point and command-line argument parsing for the convolution application.
 *
 * @details - Defines the `main` function which sets up sample arguments and calls `select()`.
 * @details - `select()` parses the arguments, initializes the `Convolution` singleton, and
 *            dispatches the chosen operation (convolution or pooling) on the specified device.
 */

/**
 * @brief Parses command-line arguments and dispatches the requested operation.
 *
 * @param argc  Number of arguments (should be 4).
 * @param argv  Array of argument strings:
 *              argv[0] = program name,
 *              argv[1] = "HOST" or "DEVICE",
 *              argv[2] = operation type ("Prewitt_Edge_detection", "Max_pooling", "Min_pooling"),
 *              argv[3] = path to input/output image file.
 *
 * @details - Checks for exactly 4 arguments and prints usage instructions on error.
 * @details - Obtains the `Convolution` singleton instance for the given image path.
 * @details - Maps the actor string to the `Actors` enum and the operation string to the
 *            corresponding member function, then invokes it.
 */
void select(int& argc, char** argv);

/**
 * @brief Application entry point.
 *
 * @param argc  Command-line argument count (ignored in this example; overridden to 4).
 * @param argv  Command-line argument values (ignored in this example; replaced by sample args).
 * @return      Exit status code (0 on success).
 *
 * @details - Constructs a sample argument list:
 *            program name, device ("DEVICE"), operation ("Min_pooling"), image path ("./wall-e.png").
 * @details - Calls `select()` to perform the specified operation.
 */
int main(int argc, char** argv){ 
    int argcc = 4;
    // Optional example for edge detection:
    // char* args[] = { (char*)"program", (char*)"DEVICE", (char*)"Prewitt_Edge_detection", (char*)"./black_white.png" };
    char* args[] = { (char*)"program", (char*)"DEVICE", (char*)"Min_pooling", (char*)"./wall-e.png" };
    select(argcc, args);

    return 0;
}

void select(int& argc, char** argv){
    if (argc != 4 ) {
        std::cerr << "Usage: " << argv[0]   << "\n <medium> [HOST or DEVICE]\n"
                                          << " <type>   [Prewitt_Edge_detection | Max_pooling | Min_pooling]\n" 
                                          << " <path>   [input image file path]" << std::endl;
        return;
    }  

    // Initialize the singleton with the provided image path
    Convolution& conv = Convolution::instance(argv[3]);

    // Map string identifiers to Actors enum values
    std::unordered_map<std::string, Actors> actortype{
        {"HOST",   Actors::Host},
        {"DEVICE", Actors::Device}
    };

    // Map operation names to corresponding Convolution methods
    std::unordered_map<std::string, std::function<void(Actors)>> convtype{
        {"Prewitt_Edge_detection", [&](Actors actor){ conv.ConvCalc(actor); }},
        {"Max_pooling",            [&](Actors actor){ conv.MaxP(actor);   }},
        {"Min_pooling",            [&](Actors actor){ conv.MinP(actor);   }}
    };
   
    // Execute the chosen operation on the chosen actor
    convtype[argv[2]]( actortype[argv[1]] );
}
