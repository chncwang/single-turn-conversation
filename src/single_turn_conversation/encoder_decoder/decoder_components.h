#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    std::vector<std::shared_ptr<LookupNode>> decoder_lookups_before_dropout;
    std::vector<std::shared_ptr<DropoutNode>> decoder_lookups;
    std::vector<std::shared_ptr<LinearNode>> decoder_to_wordvectors;
    std::vector<std::shared_ptr<LinearWordVectorNode>> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;

    virtual void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input, const std::vector<Node *> &encoder_hiddens) = 0;
};

#endif
