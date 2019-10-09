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
            bool is_training,
            int max_sentence_len_in_batch) {
        shared_ptr<DotAttentionBuilder> attention_builder(new DotAttentionBuilder);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(bucket(hyper_params.hidden_dim,
                        graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));

        int left_len = max_sentence_len_in_batch - encoder_hiddens.size();
        vector<Node*> len_fixed_hiddens = encoder_hiddens;
        if (is_training) {
            for (int i = 0; i < left_len; ++i) {
                Node *bucket = n3ldg_plus::bucket(graph, hyper_params.hidden_dim, -1000.0f);
                len_fixed_hiddens.push_back(bucket);
            }
        }

        attention_builder->forward(graph, len_fixed_hiddens, *guide);
        contexts.push_back(attention_builder->_hidden);

        ConcatNode* concat = new ConcatNode;
        concat->init(hyper_params.word_dim + hyper_params.hidden_dim);
        vector<Node *> ins = {&input, attention_builder->_hidden};
        concat->forward(graph, ins);

        decoder.forward(graph, model_params.left_to_right_encoder_params, *concat,
                *bucket(hyper_params.hidden_dim, graph),
                *bucket(hyper_params.hidden_dim, graph),
                hyper_params.dropout, is_training);
    }

    Node* decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i) {
        ConcatNode *concat_node = new ConcatNode();
        int context_dim = contexts.at(0)->getDim();
        concat_node->init(context_dim + hyper_params.hidden_dim + hyper_params.word_dim);
        vector<Node *> concat_inputs = {contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? bucket(hyper_params.word_dim, graph) :
                static_cast<Node*>(decoder_lookups.at(i - 1))};
        concat_node->forward(graph, concat_inputs);

        Node *decoder_to_wordvector = n3ldg_plus::uni(graph,
                model_params.hidden_to_wordvector_params, *concat_node);
        return decoder_to_wordvector;
    }
};

#endif
