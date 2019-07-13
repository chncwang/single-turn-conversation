#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_DECODER_COMPONENTS_H

#include <memory>
#include "N3LDG.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"

struct ResultAndKeywordVectors {
    Node *result;
    Node *keyword;
};

struct DecoderComponents {
    std::vector<LookupNode *> decoder_lookups_before_dropout;
    std::vector<DropoutNode *> decoder_lookups;
    std::vector<LookupNode *> decoder_keyword_lookups;
    std::vector<Node *> decoder_to_wordvectors;
    std::vector<Node *> decoder_to_keyword_vectors;
    std::vector<LinearWordVectorNode *> wordvector_to_onehots;
    std::vector<LinearWordVectorNode *> keyword_vector_to_onehots;
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
        shared_ptr<AttentionVBuilder> attention_builder(new AttentionVBuilder);
        attention_builder->init(model_params.attention_parrams);
        Node *guide = decoder.size() == 0 ?
            static_cast<Node*>(bucket(hyper_params.hidden_dim,
                        graph)) : static_cast<Node*>(decoder._hiddens.at(decoder.size() - 1));
        attention_builder->forward(graph, encoder_hiddens, *guide);
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

    ResultAndKeywordVectors decoderToWordVectors(Graph &graph, const HyperParams &hyper_params,
            ModelParams &model_params,
            int i,
            bool return_keyword) {
        ConcatNode *concat_node = new ConcatNode();
        int context_dim = contexts.at(0)->getDim();
        concat_node->init(context_dim + hyper_params.hidden_dim + 2 * hyper_params.word_dim);
        vector<Node *> concat_inputs = {
            contexts.at(i), decoder._hiddens.at(i),
            i == 0 ? bucket(hyper_params.word_dim, graph) :
                static_cast<Node*>(this->decoder_lookups.at(i - 1)),
            i == 0 ? bucket(hyper_params.word_dim, graph) :
                static_cast<Node*>(decoder_keyword_lookups.at(i - 1))
        };
        concat_node->forward(graph, concat_inputs);

        UniNode *keyword;
        if (return_keyword) {
            keyword = new UniNode;
            keyword->init(hyper_params.word_dim);
            keyword->setParam(model_params.hidden_to_keyword_params);
            keyword->forward(graph, *concat_node);
        } else {
            keyword = nullptr;
        }

        ConcatNode *keyword_concated = new ConcatNode();
        keyword_concated->init(concat_node->getDim() + hyper_params.word_dim);
        keyword_concated->forward(graph, {concat_node, decoder_keyword_lookups.at(i)});

        UniNode *decoder_to_wordvector(new UniNode);
        decoder_to_wordvector->init(hyper_params.word_dim);
        decoder_to_wordvector->setParam(model_params.hidden_to_wordvector_params);
        decoder_to_wordvector->forward(graph, *keyword_concated);

        return {decoder_to_wordvector, keyword};
    }
};

#endif
