#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "N3LDG.h"

struct ModelParams
#if USE_GPU
: public TransferableComponents
#endif
{
    LookupTable lookup_table;
    UniParams hidden_to_wordvector_params;
    LSTM1Params encoder_params;
    LSTM1Params decoder_params;

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() {
        return {&lookup_table, &hidden_to_wordvector_params, &encoder_params, &decoder_params};
    }

    virtual std::string name() const {
        return "ModelParams";
    }
#endif

    void save(ofstream &os) const {
        lookup_table.save(os);
        encoder_params.save(os);
        decoder_params.save(os);
        hidden_to_wordvector_params.save(os);
    }

    void load(ifstream &is) {
        lookup_table.load(is, *lookup_table.elems);
        encoder_params.load(is);
        decoder_params.load(is);
        hidden_to_wordvector_params.load(is);
    }
};

#endif
