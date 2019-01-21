#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_ATTENTION_MODEL_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_ATTENTION_MODEL_DECODER_COMPONENTS_H

#include <iostream>
#include <memory>
#include <vector>
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

struct AttentionContextDecoderComponents : DecoderComponents {
    vector<shared_ptr<ConcatNode>> concat_nodes;

    shared_ptr<BucketNode> bucket(int dim, Graph &graph) {
        static shared_ptr<BucketNode> node(new BucketNode);
        static bool init;
        if (!init) {
            init = true;
            node->init(dim);
            node->forward(graph, 0);
        }
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input, const vector<Node *> &encoder_hiddens) override {


        shared_ptr<ConcatNode> concat(new ConcatNode);
        concat->init(hyper_params.word_dim + hyper_params.hidden_dim * 2);
        vector<Node *> ins = {&input, encoder_hiddens.at(encoder_hiddens.size() - 1)};
        concat->forward(graph, ins);
        concat_nodes.push_back(concat);

        decoder.forward(graph, model_params.encoder_params, *concat,
                *bucket(hyper_params.hidden_dim, graph), *bucket(hyper_params.hidden_dim, graph),
                hyper_params.dropout);
    }
};

#endif
