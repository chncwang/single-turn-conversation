#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H

#include <string>
#include <fstream>
#include <iostream>
#include "conversation_structure.h"

std::vector<PostAndResponses> readPostAndResponses(
        const std::string &filename) {
    std::vector<PostAndResponses> results;
    std::string line;
    while (std::getline(std::ifstream(filename), line)) {
        std::cout << line << std::endl;
    }

    return results;
}

#endif
