#include "data_manager.h"
#include "cxxopts.hpp"

using namespace cxxopts;

int main(int argc, char *argv[]) {
    Options options("single-turn-conversation", "single turn conversation");
    options.add_options();
    
    return 0;
}
