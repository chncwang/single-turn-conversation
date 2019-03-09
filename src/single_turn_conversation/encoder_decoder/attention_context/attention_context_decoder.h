#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_ATTENTION_MODEL_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_ATTENTION_MODEL_DECODER_COMPONENTS_H

#include <iostream>
#include <memory>
#include <vector>
#include <boost/format.hpp>
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

struct AttentionContextDecoderComponents : DecoderComponents {
    BucketNode *bucket(int dim, Graph &graph) {
        BucketNode *node(new BucketNode);
        node->init(dim);
        node->forward(graph, 0);
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_training) override {
        shared_ptr<AttentionBuilder> attention_builder(new AttentionBuilder);
        attention_builder->init(model_params.attention_parrams);
        Node *guide = decoder.size() == 0 ? static_cast<Node*>(bucket(hyper_params.hidden_dim,
                    graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, encoder_hiddens, *guide);

        ConcatNode* concat = new ConcatNode;
        concat->init(hyper_params.word_dim + hyper_params.hidden_dim * 2);
        vector<Node *> ins = {&input, attention_builder->_hidden};
        concat->forward(graph, ins);

        decoder.forward(graph, model_params.decoder_params, *concat,
                *bucket(hyper_params.hidden_dim, graph), *bucket(hyper_params.hidden_dim, graph),
                hyper_params.dropout, is_training);
    }
};

#endif
