#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct DecoderComponents {
    std::vector<LookupNode *> decoder_lookups_before_dropout;
    std::vector<DropoutNode *> decoder_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<LinearWordVectorNode *> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;

    virtual void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            std::vector<Node *> &encoder_hiddens,
            bool is_training) = 0;

    virtual Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) = 0;
};

#endif
