#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GLOBAL_MODEL_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GLOBAL_MODEL_DECODER_COMPONENTS_H

#include "single_turn_conversation/encoder_decoder/decoder_components.h"

struct GlobalContextDecoderComponents : DecoderComponents {
    std::vector<std::shared_ptr<LookupNode>> decoder_lookups;
    std::vector<std::shared_ptr<LinearNode>> decoder_to_wordvectors;
    std::vector<std::shared_ptr<LinearWordVectorNode>> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;

    void forward(Graph &graph, LSTM1Params &lstm_params, Node &input, Node &h0, Node &c0,
            const std::vector<Node *> &encoder_hiddens) override {
        decoder.forward(graph, lstm_params, input, h0, c0);
    }
};

#endif
