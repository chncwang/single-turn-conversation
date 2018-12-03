#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_HYPER_PARAMS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <boost/format.hpp>

struct HyperParams {
    int word_dim;
    int hidden_dim;
    float dropout;
    int batch_size;
    int beam_size;
    float learning_rate;
	int word_cutoff;
	bool wordemb_finetune;
	string word_file;

    float flag() const {
        return word_dim + hidden_dim + dropout + batch_size + beam_size + learning_rate + word_cutoff;
    }

    void save(std::ofstream &os) const {
        os << word_dim << std::endl
            << hidden_dim << std::endl
            << dropout << std::endl
            << batch_size << std::endl
            << beam_size << std::endl
            << learning_rate << std::endl
			<< word_cutoff << std::endl
			<< word_file << std::endl
			<< wordemb_finetune << std::endl
            << flag() << std::endl;
    }

    void load(std::ifstream &is) {
        float f;
        is >> word_dim >> hidden_dim >> dropout >> batch_size >> beam_size >> learning_rate >> word_cutoff >>f;
        if (abs(f - flag()) > 0.001) {
            std::cerr << boost::format(
                    "loading hyper params error, s is %1%, but computed flag is %2%") % f % flag()
                << std::endl;
            abort();
        }
    }

    void print() const {
        std::cout << "word_dim:" << word_dim << std::endl
            << "hidden_dim:" << hidden_dim << std::endl
            << "dropout:" << dropout << std::endl
            << "batch_size:" << batch_size << std::endl
            << "beam_size:" << beam_size << std::endl
            << "learning_rate:" << learning_rate << std::endl
			<< "word_file:" << word_file << std::endl
			<< "wordemb_finetune:" << wordemb_finetune << std::endl
		    << "word_cutoff:" << word_cutoff << std::endl;
    }
};

#endif
