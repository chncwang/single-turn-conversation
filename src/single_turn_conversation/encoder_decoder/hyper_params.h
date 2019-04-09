#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>

enum Optimizer {
    ADAM = 0,
    ADAGRAD = 1
};

struct HyperParams : public N3LDGSerializable {
    int word_dim;
    int encoding_hidden_dim;
    int decoding_hidden_dim;
    float dropout;
    int batch_size;
    int beam_size;
    float learning_rate;
    int word_cutoff;
    string word_file;
    bool word_finetune;
    float l2_reg;
    Optimizer optimizer;

    Json::Value toJson() const override {
        Json::Value json;
        json["word_dim"] = word_dim;
        json["encoding_hidden_dim"] = encoding_hidden_dim;
        json["decoding_hidden_dim"] = decoding_hidden_dim;
        json["dropout"] = dropout;
        json["batch_size"] = batch_size;
        json["beam_size"] = beam_size;
        json["learning_rate"] = learning_rate;
        json["word_cutoff"] = word_cutoff;
        json["word_file"] = word_file;
        json["word_finetune"] = word_finetune;
        json["l2_reg"] = l2_reg;
        json["optimizer"] = static_cast<int>(optimizer);
        return json;
    }

    void fromJson(const Json::Value &json) override {
        word_dim = json["word_dim"].asInt();
        encoding_hidden_dim = json["encoding_hidden_dim"].asInt();
        decoding_hidden_dim = json["decoding_hidden_dim"].asInt();
        dropout = json["dropout"].asFloat();
        batch_size = json["batch_size"].asInt();
        beam_size = json["beam_size"].asInt();
        learning_rate = json["learning_rate"].asFloat();
        word_cutoff = json["word_cutoff"].asInt();
        word_file = json["word_file"].asString();
        word_finetune = json["word_finetune"].asBool();
        l2_reg = json["l2_reg"].asFloat();
        optimizer = static_cast<Optimizer>(json["optimizer"].asInt());
    }

    void print() const {
        std::cout << "word_dim:" << word_dim << std::endl
            << "encoding_hidden_dim:" << encoding_hidden_dim << std::endl
            << "decoding_hidden_dim:" << decoding_hidden_dim << std::endl
            << "dropout:" << dropout << std::endl
            << "batch_size:" << batch_size << std::endl
            << "beam_size:" << beam_size << std::endl
            << "learning_rate:" << learning_rate << std::endl
	    << "word_cutoff:" << word_cutoff << std::endl
	    << "word_file:" << word_file << std::endl
    	    << "word_finetune:" << word_finetune << std::endl
    	    << "l2_reg:" << l2_reg << std::endl
    	    << "optimizer:" << optimizer << std::endl; 
    }
};

#endif
