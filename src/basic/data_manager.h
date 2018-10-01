#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H

#include <string>
#include <fstream>
#include <iostream>
#include "conversation_structure.h"
#include <boost/algorithm/string.hpp>

std::vector<PostAndResponses> readPostAndResponses(
        const std::string &filename) {
    std::vector<PostAndResponses> results;
    std::string line;
    std::ifstream ifs(filename);
    while (std::getline(ifs, line)) {
        vector<string> strs;
        boost::split(strs, line, boost::is_any_of(":"));
        if (strs.size() != 2) {
            abort();
        }
        std::cout << strs.at(0) << " " << strs.at(1) << std::endl;
    }

    return results;
}

#endif
