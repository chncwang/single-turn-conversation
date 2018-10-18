#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include "N3LDG.h"
#include "model_params.h"
#include "hyper_params.h"

/* *
 * return vector<tuple<beam_i, index in val, log probability>>
 * */
std::vector<std::tuple<int, int, dtype>> mostLikeResults(const std::vector<Node *> &nodes,
        const std::vector<std::tuple<int, int, dtype>> &last_results,
        int k) {
    std::priority_queue<std::pair<dtype, std::pair<int, int>>> queue;
    for (int i = 0; i < nodes.size(); ++i) {
        const Node &node = *nodes.at(i);
        auto tuple = toExp(node);

        for (int j = 0; j < nodes.at(j)->dim; ++j) {
            dtype value = node.val[j];
            dtype log_probability = value - log(std::get<2>(tuple));

            if (queue.size() < k) {
                queue.push(std::make_pair(value, std::make_pair(i, j)));
            } else if (queue.top().first < value) {
                queue.pop();
                queue.push(std::make_pair(value, std::make_pair(i, j)));
            }
        }
    }

    if (k != queue.size()) {
        std::cerr << "k is not equal to queue.size()" << std::endl;
        abort();
    }

    std::vector<std::tuple<int, int, dtype>> results;

    while (!queue.empty()) {
        auto &e = queue.top();
        results.push_back(std::make_tuple(e.second.first, e.second.second, e.first));
        queue.pop();
    }

    return results;
}

struct DecoderComponents {
    std::vector<std::shared_ptr<LookupNode>> decoder_lookups;
    std::vector<std::shared_ptr<LinearNode>> decoder_to_wordvectors;
    std::vector<std::shared_ptr<LinearWordVectorNode>> wordvector_to_onehots;
    DynamicLSTMBuilder decoder;
};

struct GraphBuilder {
    std::vector<std::shared_ptr<LookupNode>> encoder_lookups;
    DynamicLSTMBuilder encoder;
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

    void forwardDecoder(Graph &graph, DecoderComponents &decoder_components,
            const std::vector<std::string> &answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        if (!graph.train) {
            std::cerr << "train should be true" << std::endl;
            abort();
        }

        for (int i = 0; i < answer.size(); ++i) {
            forwardDecoderByOneStep(graph, decoder_components, i,
                    i == 0 ? nullptr : &answer.at(i - 1),
                    hyper_params,
                    model_params);
        }
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        Node *last_input;
        if (i > 0) {
            std::shared_ptr<LookupNode> decoder_lookup(new LookupNode);
            decoder_lookup->init(hyper_params.word_dim, -1);
            decoder_lookup->setParam(model_params.lookup_table);
            decoder_lookup->forward(graph, *answer);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            last_input = decoder_components.decoder_lookups.at(i - 1).get();
        } else {
            last_input = &word_bucket;
        }

        decoder_components.decoder.forward(graph, model_params.decoder_params, *last_input, 
                *encoder._hiddens.at(encoder._hiddens.size() - 1),
                *encoder._cells.at(encoder._hiddens.size() - 1));

        std::shared_ptr<LinearNode> decoder_to_wordvector(new LinearNode);
        decoder_to_wordvector->init(hyper_params.word_dim, -1);
        decoder_to_wordvector->setParam(model_params.hidden_to_wordvector_params);
        decoder_to_wordvector->forward(graph, *decoder_components.decoder._hiddens.at(i));
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        std::shared_ptr<LinearWordVectorNode> wordvector_to_onehot(new LinearWordVectorNode);
        wordvector_to_onehot->init(model_params.lookup_table.nVSize, -1);
        wordvector_to_onehot->setParam(model_params.lookup_table.E);
        wordvector_to_onehot->forward(graph, *decoder_to_wordvector);
        decoder_components.wordvector_to_onehots.push_back(wordvector_to_onehot);
    }

    void forwardDecoderUsingBeamSearch(Graph &graph,
            const std::vector<DecoderComponents> &decoder_components_beam,
            int beam_size,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        if (graph.train) {
            std::cerr << "train should be false" << std::endl;
            abort();
        }
        std::vector<DecoderComponents> beam = decoder_components_beam;

        for (int i = 0;; ++i) {
            std::vector<std::string> last_answers;
            std::vector<std::tuple<int, int, dtype>> most_like_results;
            if (i > 0) {
                std::vector<Node *> last_outputs;
                for (DecoderComponents &decoder_components : beam) {
                    last_outputs.push_back(
                            decoder_components.wordvector_to_onehots.at(i - 1).get());
                }
                most_like_results = mostLikeResults(last_outputs, most_like_results, beam_size);
                std::vector<DecoderComponents> last_beam = beam;
                int j = 0;
                for (std::tuple<int, int, dtype> &tuple : most_like_results) {
                    int last_beam_i = std::get<0>(tuple);
                    beam.at(j++) = last_beam.at(last_beam_i);
                    int last_word_id = std::get<1>(tuple);
                    last_answers.push_back(model_params.lookup_table.elems->from_id(last_word_id));
                }
            }

            for (int beam_i = 0; beam_i < beam_size; ++beam_i) {
                DecoderComponents &decoder_components = beam.at(beam_i);
                forwardDecoderByOneStep(graph, decoder_components, i,
                        i == 0 ? nullptr : &last_answers.at(beam_i),
                        hyper_params,
                        model_params);
            }

            graph.compute();
        }
    }
};

#endif
