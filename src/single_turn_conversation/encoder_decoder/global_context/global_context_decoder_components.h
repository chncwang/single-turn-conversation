#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GLOBAL_MODEL_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GLOBAL_MODEL_DECODER_COMPONENTS_H

#include <iostream>
#include <memory>
#include <vector>
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

// decrecated
struct GlobalContextDecoderComponents : DecoderComponents {
//    vector<shared_ptr<ConcatNode>> concat_nodes;

    BucketNode* bucket(int dim, Graph &graph) {
        static BucketNode* node(new BucketNode);
        static bool init;
        if (!init) {
            init = true;
            node->init(dim);
            node->forward(graph, 0);
        }
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_training) override {
        ConcatNode *concat(new ConcatNode);
        concat->init(hyper_params.word_dim + hyper_params.encoding_hidden_dim * 2);
        vector<Node *> ins = {&input, encoder_hiddens.at(encoder_hiddens.size() - 1)};
        concat->forward(graph, ins);

        decoder.forward(graph, model_params.decoder_params, *concat,
                *bucket(hyper_params.decoding_hidden_dim, graph),
                *bucket(hyper_params.decoding_hidden_dim, graph), hyper_params.dropout,
                is_training);
    }

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) override {
        LinearNode *decoder_to_wordvector(new LinearNode);
        decoder_to_wordvector->init(hyper_params.word_dim);
        decoder_to_wordvector->setParam(model_params.hidden_to_wordvector_params);
        decoder_to_wordvector->forward(graph, *decoder._hiddens.at(i));
        return decoder_to_wordvector;
    }
};

#endif
