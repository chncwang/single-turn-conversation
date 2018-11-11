#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"

struct DecoderComponents {
    std::vector<std::shared_ptr<LookupNode>> decoder_lookups;
    std::vector<std::shared_ptr<LinearNode>> decoder_to_wordvectors;
    std::vector<std::shared_ptr<LinearWordVectorNode>> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;

    virtual void forward(Graph &graph, LSTM1Params &lstm_params, Node &input, Node &h0, Node &c0,
            const std::vector<Node *> &encoder_hiddens) = 0;
};

#endif
