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
    UniParams hidden_to_keyword_params;
    LSTM1Params left_to_right_encoder_params;
    DotAttentionParams normal_attention_parrams;

    Json::Value toJson() const override {
        Json::Value json;
        json["lookup_table"] = lookup_table.toJson();
        json["hidden_to_wordvector_params"] = hidden_to_wordvector_params.toJson();
        json["hidden_to_keyword_params"] = hidden_to_keyword_params.toJson();
        json["left_to_right_encoder_params"] = left_to_right_encoder_params.toJson();
        json["normal_attention_parrams"] = normal_attention_parrams.toJson();
        return json;
    }

    void fromJson(const Json::Value &json) override {
        lookup_table.fromJson(json["lookup_table"]);
        hidden_to_wordvector_params.fromJson(json["hidden_to_wordvector_params"]);
        hidden_to_keyword_params.fromJson(json["hidden_to_keyword_params"]);
        left_to_right_encoder_params.fromJson(json["left_to_right_encoder_params"]);
        normal_attention_parrams.fromJson(json["normal_attention_parrams"]);
    }

#if USE_GPU
    std::vector<n3ldg_cuda::Transferable *> transferablePtrs() override {
        return {&lookup_table, &hidden_to_wordvector_params, &hidden_to_keyword_params,
            &left_to_right_encoder_params, &normal_attention_parrams};
    }
#endif
};

#endif
