#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <vector>
#include <string>
#include "N3LDG.h"
#include <memory>
#include "model_params.h"
#include "hyper_params.h"

template<typename T>
std::vector<Node*> toNodePointers(std::vector<std::shared_ptr<T>> &vec) {
    std::vector<Node *> results;
    for (std::shared_ptr<T> &p : vec) {
        results.push_back(p.get());
    }
    return results;
}

struct GraphBuilder {
    std::vector<std::shared_ptr<LookupNode>> encoder_lookups;
    std::vector<std::shared_ptr<LookupNode>> decoder_lookups;
    std::vector<std::shared_ptr<LinearNode>> decoder_to_wordvectors;
    std::vector<std::shared_ptr<LinearNode>> wordvector_to_onehots;
    DynamicLSTMBuilder encoder;
    DynamicLSTMBuilder decoder;
    BucketNode hidden_bucket;
    BucketNode word_bucket;

    void init(const HyperParams &hyper_params) {
        hidden_bucket.init(hyper_params.hidden_dim, -1);
        word_bucket.init(hyper_params.word_dim, -1);
    }

    void forward(Graph &graph, const std::vector<std::string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        hidden_bucket.forward(graph);
        word_bucket.forward(graph);

        for (const std::string &word : sentence) {
            std::shared_ptr<LookupNode> input_lookup(new LookupNode);
            input_lookup->init(hyper_params.word_dim, hyper_params.dropout);
            input_lookup->setParam(model_params.lookup_table);
            input_lookup->forward(graph, word);
            encoder_lookups.push_back(input_lookup);
        }

        for (std::shared_ptr<LookupNode> &node : encoder_lookups) {
            encoder.forward(graph, model_params.encoder_params, *node, hidden_bucket,
                    hidden_bucket);
        }
    }

    void forwardDecoder(Graph &graph, int answer_size, const HyperParams &hyper_params,
            ModelParams &model_params) {
        for (int i = 0; i < answer_size; ++i) {
            Node *last_input;
            if (i > 0) {
                last_input = decoder_to_wordvectors.at(i - 1).get();
            } else {
                last_input = &word_bucket;
            }
            decoder.forward(graph, model_params.decoder_params, *last_input, 
                *encoder._hiddens.at(encoder._hiddens.size() - 1),
                *encoder._cells.at(encoder._hiddens.size() - 1));

            std::shared_ptr<LinearNode> decoder_to_wordvector(new LinearNode);
            decoder_to_wordvector->init(hyper_params.word_dim, -1);
            decoder_to_wordvector->setParam(model_params.hidden_to_wordvector_params);
            decoder_to_wordvector->forward(graph, *decoder._hiddens.at(0));
            decoder_to_wordvectors.push_back(decoder_to_wordvector);

            std::shared_ptr<LinearNode> wordvector_to_onehot(new LinearNode);
            wordvector_to_onehot->init(model_params.lookup_table.nVSize, -1);
            wordvector_to_onehot->setParam(model_params.wordvector_to_onehot_params);
            wordvector_to_onehot->forward(graph, *decoder_to_wordvector);
            wordvector_to_onehots.push_back(wordvector_to_onehot);
        }
    }
};

#endif
