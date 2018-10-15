#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>

struct HyperParams {
    int word_dim;
    int hidden_dim;
    float dropout;
    int batchsize;
    float learning_rate;

    void print() const {
        std::cout << "word_dim:" << word_dim << std::endl
            << "hidden_dim:" << hidden_dim << std::endl
            << "dropout:" << dropout << std::endl
            << "batchsize:" << batchsize << std::endl
            << "learning_rate:" << learning_rate << std::endl;
    }
};

#endif
