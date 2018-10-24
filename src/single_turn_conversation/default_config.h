#ifndef SINGLE_TURN_CONVERSATION_DEFAULT_CONFIG_H
#define SINGLE_TURN_CONVERSATION_DEFAULT_CONFIG_H

#include <iostream>

struct DefaultConfig {
    bool check_grad;
    bool one_response;
    bool learn_test;
    int max_sample_count;
    int dev_size;
    int test_size;

    void print() const {
        std::cout << "check_grad:" << check_grad << std::endl
            << "one_response:" << one_response << std::endl
            << "learn_test:" << learn_test << std::endl
            << "max_sample_count:" << max_sample_count << std::endl
            << "dev_size:" << dev_size << std::endl
            << "test_size:" << test_size << std::endl;
    }
};

DefaultConfig &GetDefaultConfig() {
    static DefaultConfig default_config;
    return default_config;
}

#endif
