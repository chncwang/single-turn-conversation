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
    UniParams transformed_h0_params;
    UniParams transformed_c0_params;

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["encoder_params"] = encoder_params.toJson();
        json["decoder_params"] = decoder_params.toJson();
        json["transformed_h0_params"] = transformed_h0_params.toJson();
        json["transformed_c0_params"] = transformed_c0_params.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        encoder_params.fromJson(json["encoder_params"]);
        decoder_params.fromJson(json["decoder_params"]);
        transformed_h0_params.fromJson(json["transformed_h0_params"]);
        transformed_c0_params.fromJson(json["transformed_c0_params"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params, &encoder_params, &decoder_params,
        &transformed_h0_params, &transformed_c0_params};
    }
#endif
};

#endif
