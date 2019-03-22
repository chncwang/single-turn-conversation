#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include <algorithm>
#include <boost/format.hpp>
#include "N3LDG.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/encoder_decoder/global_context/global_context_decoder_components.h"

struct WordIdAndProbability {
    int word_id;
    dtype probability;

    WordIdAndProbability() = default;
    WordIdAndProbability(const WordIdAndProbability &word_id_and_probability) = default;
    WordIdAndProbability(int wordid, dtype prob) : word_id(wordid), probability(prob) {}
};

struct BeamSearchResult {
    int beam_i;
    std::vector<WordIdAndProbability> path;
    dtype final_log_probability;

    BeamSearchResult() = default;
    BeamSearchResult(const BeamSearchResult &beam_search_result) = default;
    BeamSearchResult(int beami, const std::vector<WordIdAndProbability> &pathh,
            dtype log_probability) : beam_i(beami), path(pathh),
    final_log_probability(log_probability) {}
};

void printWordIds(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const LookupTable &lookup_table) {
    for (const WordIdAndProbability &ids : word_ids_with_probability_vector) {
//        cout << boost::format("%1%(%2%) ") % lookup_table.elems->from_id(ids.word_id) %
//            ids.probability;
        cout << lookup_table.elems->from_id(ids.word_id);
    }
    cout << endl;
}

std::vector<BeamSearchResult> mostProbableResults(
        const std::vector<Node *> &nodes,
        const std::vector<BeamSearchResult> &last_results,
        int k,
        const ModelParams &model_params) {
    if (nodes.size() != last_results.size() && !last_results.empty()) {
        std::cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << std::endl;
        abort();
    }

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.final_log_probability > b.final_log_probability;
    };
    std::priority_queue<BeamSearchResult, std::vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    int stop_id = model_params.lookup_table.getElemId(STOP_SYMBOL);
    std::vector<BeamSearchResult> results;
    for (int i = 0; i < nodes.size(); ++i) {
        const Node &node = *nodes.at(i);
        auto tuple = toExp(node);

        for (int j = 0; j < nodes.at(i)->dim; ++j) {
            dtype value = node.val.v[j] - std::get<1>(tuple).second;
            dtype log_probability = value - log(std::get<2>(tuple));
            dtype word_probability = exp(log_probability);
            std::vector<WordIdAndProbability> word_ids;
            if (!last_results.empty()) {
                log_probability += last_results.at(i).final_log_probability;
                word_ids = last_results.at(i).path;
            }

            word_ids.push_back(WordIdAndProbability(j, word_probability));
//            if (j == stop_id) {
//                log_probability += 3;
//            }
            BeamSearchResult beam_search_result(i, word_ids, log_probability);

            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().final_log_probability < log_probability) {
                queue.pop();
                queue.push(beam_search_result);
//            } else if (j == stop_id) {
//                std::cout << boost::format(
//                        "queue.top().final_log_probability:%1% stop log_probability:%2%") %
//                    queue.top().final_log_probability % log_probability << std::endl;
            }
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        results.push_back(e);
        queue.pop();
    }

//    int i = 0;
//    for (const BeamSearchResult &result : results) {
//        std::cout << boost::format("mostProbableResults - i:%1%") % i << std::endl;
//        printWordIds(result.path, model_params.lookup_table);
//        ++i;
//    }

    return results;
}

struct GraphBuilder {
    std::vector<DropoutNode *> encoder_lookups;
    DynamicLSTMBuilder left_to_right_encoder;
    DynamicLSTMBuilder right_to_left_encoder;
    std::vector<ConcatNode *> concated_encoder_nodes;
    BucketNode *hidden_bucket = new BucketNode;
    BucketNode *word_bucket = new BucketNode;

    void init(const HyperParams &hyper_params) {
        hidden_bucket->init(hyper_params.hidden_dim);
        word_bucket->init(hyper_params.word_dim);
    }

    void forward(Graph &graph, const std::vector<std::string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        hidden_bucket->forward(graph);
        word_bucket->forward(graph);

        for (const std::string &word : sentence) {
            LookupNode* input_lookup(new LookupNode);
            input_lookup->init(hyper_params.word_dim);
            input_lookup->setParam(model_params.lookup_table);
            input_lookup->forward(graph, word);

            DropoutNode* dropout_node(new DropoutNode);
            dropout_node->init(hyper_params.word_dim, hyper_params.dropout);
            dropout_node->is_training = is_training;
            dropout_node->forward(graph, *input_lookup);
            encoder_lookups.push_back(dropout_node);
        }

        for (DropoutNode* node : encoder_lookups) {
            left_to_right_encoder.forward(graph, model_params.left_to_right_encoder_params, *node,
                    *hidden_bucket, *hidden_bucket, hyper_params.dropout, is_training);
        }

        int size = encoder_lookups.size();

        for (int i = size - 1; i >= 0; --i) {
            right_to_left_encoder.forward(graph, model_params.right_to_left_encoder_params,
                    *encoder_lookups.at(i), *hidden_bucket, *hidden_bucket, hyper_params.dropout,
                    is_training);
        }

        if (left_to_right_encoder.size() != right_to_left_encoder.size()) {
            std::cerr << "left_to_right_encoder size is not equal to right_to_left_encoder" <<
                std::endl;
            abort();
        }

        for (int i = 0; i < size; ++i) {
            ConcatNode* concat_node(new ConcatNode);
            concat_node->init(2 * hyper_params.hidden_dim);
            std::vector<Node*> nodes = {left_to_right_encoder._hiddens.at(i),
                    right_to_left_encoder._hiddens.at(i)};
            concat_node->forward(graph, nodes);
            concated_encoder_nodes.push_back(concat_node);
        }
    }

    void forwardDecoder(Graph &graph, DecoderComponents &decoder_components,
            const std::vector<std::string> &answer,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        for (int i = 0; i < answer.size(); ++i) {
            forwardDecoderByOneStep(graph, decoder_components, i,
                    i == 0 ? nullptr : &answer.at(i - 1), hyper_params, model_params, is_training);
        }
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        Node *last_input;
        if (i > 0) {
            LookupNode* before_dropout(new LookupNode);
            before_dropout->init(hyper_params.word_dim);
            before_dropout->setParam(model_params.lookup_table);
            before_dropout->forward(graph, *answer);

            DropoutNode* decoder_lookup(new DropoutNode);
            decoder_lookup->init(hyper_params.word_dim, hyper_params.dropout);
            decoder_lookup->is_training = is_training;
            decoder_lookup->forward(graph, *before_dropout);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            last_input = decoder_components.decoder_lookups.at(i - 1);
        } else {
            last_input = word_bucket;
        }

        std::vector<Node *> encoder_hiddens = transferVector<Node *, ConcatNode*>(
                concated_encoder_nodes, [](ConcatNode *concat) {
                return concat;
                });

        decoder_components.forward(graph, hyper_params, model_params, *last_input,
                encoder_hiddens, is_training);

        LinearNode *decoder_to_wordvector(new LinearNode);
        decoder_to_wordvector->init(hyper_params.word_dim);
        decoder_to_wordvector->setParam(model_params.hidden_to_wordvector_params);
        decoder_to_wordvector->forward(graph, *decoder_components.decoder._hiddens.at(i));
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        LinearWordVectorNode *wordvector_to_onehot(new LinearWordVectorNode);
        wordvector_to_onehot->init(model_params.lookup_table.nVSize);
        wordvector_to_onehot->setParam(model_params.lookup_table.E);
        wordvector_to_onehot->forward(graph, *decoder_to_wordvector);
        decoder_components.wordvector_to_onehots.push_back(wordvector_to_onehot);
    }

    std::pair<std::vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            const std::vector<std::shared_ptr<DecoderComponents>> &decoder_components_beam,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        auto beam = decoder_components_beam;
//        std::cout << boost::format(
//                "forwardDecoderUsingBeamSearch - decoder_components_beam size:%1%") %
//                decoder_components_beam.size() << std::endl;
        std::vector<std::pair<std::vector<WordIdAndProbability>, dtype>> word_ids_result;
        std::vector<BeamSearchResult> most_probable_results;
        std::vector<std::string> last_answers;

        for (int i = 0;; ++i) {
            last_answers.clear();
            if (i > 0) {
                std::vector<Node *> last_outputs;
                int beam_i = 0;
                for (std::shared_ptr<DecoderComponents> &decoder_components : beam) {
                    last_outputs.push_back(
                            decoder_components->wordvector_to_onehots.at(i - 1));
                    ++beam_i;
                }
                most_probable_results = mostProbableResults(last_outputs, most_probable_results,
                        k, model_params);
                auto last_beam = beam;
                beam.clear();
                std::vector<BeamSearchResult> stop_removed_results;
                int j = 0;
                for (BeamSearchResult &beam_search_result : most_probable_results) {
                    const std::vector<WordIdAndProbability> &word_ids = beam_search_result.path;
                    int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                    const std::string &word = model_params.lookup_table.elems->from_id(
                            last_word_id);
                    if (word == STOP_SYMBOL) {
//                        std::cout << boost::format(
//                                "i:%1% word:%2% most_probable_results size:%3% j:%4%") % i % word %
//                            most_probable_results.size() % j << std::endl;
                        word_ids_result.push_back(std::make_pair(word_ids,
                                    beam_search_result.final_log_probability));
                    } else {
                        stop_removed_results.push_back(beam_search_result);
                        last_answers.push_back(word);
                        beam.push_back(last_beam.at(beam_search_result.beam_i));
                    }
                    ++j;
                }

                most_probable_results = stop_removed_results;
            }

            if (beam.empty()) {
                break;
            }

            for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                DecoderComponents &decoder_components = *beam.at(beam_i);
                forwardDecoderByOneStep(graph, decoder_components, i,
                        i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                        model_params, is_training);
            }

            graph.compute();
        }

        if (word_ids_result.size() < k) {
            std::cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << std::endl;
            abort();
        }
        if (word_ids_result.empty()) {
            return std::make_pair(std::vector<WordIdAndProbability>(), 0.0f);
        }

//        for (const auto &pair : word_ids_result) {
//            const std::vector<WordIdAndProbability> ids = pair.first;
//            std::cout << boost::format("beam result:%1%") % exp(pair.second) << std::endl;
//            printWordIds(ids, model_params.lookup_table);
//        }

        auto compair = [](const std::pair<std::vector<WordIdAndProbability>, dtype> &a,
                const std::pair<std::vector<WordIdAndProbability>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = std::max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return std::make_pair(max->first, exp(max->second));
    }
};

#endif
