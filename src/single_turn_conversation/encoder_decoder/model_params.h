#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_MODEL_PARAMS_H

#include <fstream>
#include <iostream>

#include "N3LDG.h"

struct ModelParams : public N3LDGSerializable
#if USE_GPU
, public TransferableComponents
#endif
{
    LookupTable lookup_table;
    UniParams hidden_to_wordvector_params;
    LSTM1Params encoder_params;
    LSTM1Params decoder_params;

    Json::Value toJson() const {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["encoder_params"] = encoder_params.toJson();
        json["decoder_params"] = decoder_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &) {
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() {
        return {&lookup_table, &hidden_to_wordvector_params, &encoder_params, &decoder_params};
    }

    virtual std::string name() const {
        return "ModelParams";
    }
#endif
};

#endif
