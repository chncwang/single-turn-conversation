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
    vector<Node*> contexts;

    BucketNode *bucket(int dim, Graph &graph) {
        BucketNode *node(new BucketNode);
        node->init(dim);
        node->forward(graph, 0);
        return node;
    }

    void forward(Graph &graph, const HyperParams &hyper_params, ModelParams &model_params,
            Node &input,
            vector<Node *> &encoder_hiddens,
            bool is_training) {
        shared_ptr<AttentionBuilder> attention_builder(new AttentionBuilder);
        attention_builder->init(model_params.attention_parrams);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(bucket(hyper_params.decoding_hidden_dim,
                        graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, encoder_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        ConcatNode* concat = new ConcatNode;
        concat->init(hyper_params.word_dim + hyper_params.encoding_hidden_dim * 2);
        vector<Node *> ins = {&input, attention_builder->_hidden};
        concat->forward(graph, ins);

        decoder.forward(graph, model_params.decoder_params, *concat,
                *bucket(hyper_params.decoding_hidden_dim, graph),
                *bucket(hyper_params.decoding_hidden_dim, graph),
                hyper_params.dropout, is_training);
    }

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) {
        ConcatNode *concat_node = new ConcatNode();
        int context_dim = contexts.at(0)->getDim();
        concat_node->init(context_dim + hyper_params.decoding_hidden_dim + hyper_params.word_dim);
        vector<Node *> concat_inputs = {contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? bucket(hyper_params.word_dim, graph) : decoder_to_wordvectors.at(i - 1)};
        concat_node->forward(graph, concat_inputs);

        LinearNode *decoder_to_wordvector(new LinearNode);
        decoder_to_wordvector->init(hyper_params.word_dim);
        decoder_to_wordvector->setParam(model_params.hidden_to_wordvector_params);
        decoder_to_wordvector->forward(graph, *concat_node);
        return decoder_to_wordvector;
    }
};

#endif
