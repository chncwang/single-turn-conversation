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
    std::vector<std::shared_ptr<LookupNode>> lookups;
    DynamicLSTMBuilder encoder;
    DynamicLSTMBuilder decoder;
    BucketNode bucket_node;

    void init(const HyperParams &hyper_params) {
        bucket_node.init(hyper_params.hidden_dim, -1);
    }

    void forward(Graph &graph, const std::vector<std::string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        for (const std::string &word : sentence) {
            std::shared_ptr<LookupNode> lookup(new LookupNode);
            lookup->init(hyper_params.word_dim, hyper_params.dropout);
            lookup->setParam(model_params.lookup_table);
            lookups.push_back(lookup);
        }

        encoder.init(sentence.size(), model_params.encoder_params);
        std::vector<Node *> lookup_pointers = toNodePointers(lookups);
        encoder.forward(graph, lookup_pointers, bucket_node, bucket_node);
    }

    void forwardDecoder(Graph &graph, int answer_len, const HyperParams &hyper_params,
            ModelParams &model_params) {
        std::vector<Node *> inputs;
        inputs.push_back(&bucket_node);
        for (int i = 1; i < answer_len + 1; ++i) {
        }
    }
};

#endif
