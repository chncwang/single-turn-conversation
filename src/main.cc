#include "data_manager.h"
#include "cxxopts.hpp"

using namespace std;
using namespace cxxopts;

int main(int argc, char *argv[]) {
    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
        ("p,pair", "pair file name", cxxopts::value<string>());
    auto args = options.parse(argc, argv);
    string pair_filename = args["pair"].as<string>();

    readPostAndResponses(pair_filename);

    return 0;
}
