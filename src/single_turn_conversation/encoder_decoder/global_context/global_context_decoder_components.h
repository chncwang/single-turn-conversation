#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GLOBAL_MODEL_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GLOBAL_MODEL_DECODER_COMPONENTS_H

#include <iostream>
#include <memory>
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

struct GlobalContextDecoderComponents : DecoderComponents {
    LinearNode transformed_h0;
    LinearNode transformed_c0;
    bool initiated = false;

    void init(const HyperParams &hyper_params, ModelParams &model_params) {
        transformed_h0.init(hyper_params.hidden_dim);
        transformed_h0.setParam(model_params.transformed_h0_params);
        transformed_c0.init(hyper_params.hidden_dim);
        transformed_c0.setParam(model_params.transformed_c0_params);
        initiated = true;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input, Node &h0, Node &c0, const std::vector<Node *> &encoder_hiddens) override {
        if (!initiated) {
            init(hyper_params, model_params);
            transformed_h0.forward(graph, h0);
            transformed_c0.forward(graph, c0);
        }
        decoder.forward(graph, model_params.encoder_params, input, transformed_h0, transformed_c0);
    }
};

#endif
