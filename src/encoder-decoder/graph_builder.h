#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <vector>
#include <string>
#include "N3LDG.h"
#include <memory>
#include "model_params.h"
#include "hyper_params.h"

struct GraphBuilder {
    void forward(Graph &graph, const std::vector<std::string> &sentence,
            const HyperParams &hyper_params,
            const ModelParams &model_params) {
        std::vector<std::shared_ptr<LookupNode>> lookups;
        for (const std::string &word : sentence) {
            std::shared_ptr<LookupNode> lookup(new LookupNode);
            lookup->init(hyper_params.word_dim, hyper_params.dropout);
            lookup->setParam(model_params.lookup_table);
            lookups.push_back(lookup);
        }
    }
};

#endif
