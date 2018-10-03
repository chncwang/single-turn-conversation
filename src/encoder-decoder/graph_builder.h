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
    std::vector<std::shared_ptr<LookupNode>> input_lookups;
    std::vector<std::shared_ptr<LookupNode>> output_lookups;
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
        for (const std::string &word : sentence) {
            std::shared_ptr<LookupNode> input_lookup(new LookupNode);
            input_lookup->init(hyper_params.word_dim, hyper_params.dropout);
            input_lookup->setParam(model_params.lookup_table);
            input_lookups.push_back(input_lookup);
        }

        for (std::shared_ptr<LookupNode> &node : input_lookups) {
            encoder.forward(graph, model_params.encoder_params, *node, hidden_bucket,
                    hidden_bucket);
        }
    }

    void forwardDecoder(Graph &graph, const std::vector<std::string> &answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        decoder.forward(graph, model_params.decoder_params, word_bucket,
                *encoder._hiddens.at(encoder._hiddens.size() - 1),
                *encoder._cells.at(encoder._hiddens.size() - 1));

        for (const std::string &word : answer) {
            std::shared_ptr<LookupNode> output_lookup(new LookupNode);
            output_lookup->init(hyper_params.word_dim, hyper_params.dropout);
            output_lookup->setParam(model_params.lookup_table);
            output_lookups.push_back(output_lookup);
            decoder.forward(graph, model_params.decoder_params, *output_lookup, 
                *encoder._hiddens.at(encoder._hiddens.size() - 1),
                *encoder._cells.at(encoder._hiddens.size() - 1));
        }
    }
};

#endif
