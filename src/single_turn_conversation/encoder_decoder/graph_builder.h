#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <cmath>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include <algorithm>
#include <boost/format.hpp>
#include "N3LDG.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

struct WordIdAndProbability {
    int word_id;
    dtype probability;

    WordIdAndProbability() = default;
    WordIdAndProbability(const WordIdAndProbability &word_id_and_probability) = default;
    WordIdAndProbability(int wordid, dtype prob) : word_id(wordid), probability(prob) {}
};

class BeamSearchResult {
public:
    BeamSearchResult() {
        ngram_counts_ = {0, 0, 0};
    }
    BeamSearchResult(const BeamSearchResult &beam_search_result) = default;
    BeamSearchResult(const DecoderComponents &decoder_components,
            const std::vector<WordIdAndProbability> &pathh,
            dtype log_probability) : decoder_components_(decoder_components), path_(pathh),
            final_log_probability(log_probability) {
                ngram_counts_ = {0, 0, 0};
            }

    dtype finalScore() const {
        std::set<int> unique_words;
        for (const auto &p : path_) {
            unique_words.insert(p.word_id);
        }
        return (final_log_probability + extra_score_) / unique_words.size();
    }

    dtype finalLogProbability() const {
        return final_log_probability;
    }

    vector<WordIdAndProbability> getPath() const {
        return path_;
    }

    const DecoderComponents &decoderComponents() const {
        return decoder_components_;
    }

    void setExtraScore(dtype extra_score) {
        extra_score_ = extra_score;
    }

    dtype getExtraScore() const {
        return extra_score_;
    }

    const std::array<int, 3> &ngramCounts() const {
        return ngram_counts_;
    }

    void setNgramCounts(const std::array<int, 3> &counts) {
        ngram_counts_ = counts;
    }

private:
    DecoderComponents decoder_components_;
    std::vector<WordIdAndProbability> path_;
    dtype final_log_probability;
    dtype extra_score_;
    std::array<int, 3> ngram_counts_ = {};
};

void printWordIds(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const LookupTable &lookup_table) {
    for (const WordIdAndProbability &ids : word_ids_with_probability_vector) {
        cout << lookup_table.elems.from_id(ids.word_id);
    }
    cout << endl;
}

int countNgramDuplicate(const vector<int> &ids, int n) {
    if (n >= ids.size()) {
        return 0;
    }
    vector<int> target;
    for (int i = 0; i < n; ++i) {
        target.push_back(ids.at(ids.size() - n + i));
    }

    int duplicate_count = 0;

    for (int i = 0; i < ids.size() - n; ++i) {
        bool same = true;
        for (int j = 0; j < n; ++j) {
            if (target.at(j) != ids.at(i + j)) {
                same = false;
                break;
            }
        }
        if (same) {
            ++duplicate_count;
        }
    }

    return duplicate_count;
}

void updateBeamSearchResultScore(BeamSearchResult &beam_search_result,
        const NgramPenalty& penalty) {
    vector<WordIdAndProbability> word_id_and_probability = beam_search_result.getPath();
    vector<int> ids = transferVector<int, WordIdAndProbability>(word_id_and_probability, [](
                const WordIdAndProbability &a) {return a.word_id;});
    dtype extra_score = 0.0f;
    vector<dtype> penalties = {penalty.one, penalty.two, penalty.three};
    std::array<int, 3> counts;
    for (int i = 3; i > 0; --i) {
        int duplicate_count = countNgramDuplicate(ids, i);
        counts.at(i - 1) = duplicate_count;
        extra_score -= penalties.at(i - 1) * duplicate_count;
    }
    beam_search_result.setExtraScore(beam_search_result.getExtraScore() + extra_score);
    std::array<int, 3> original_counts = beam_search_result.ngramCounts();
    std::array<int, 3> new_counts = {original_counts.at(0) + counts.at(0),
        original_counts.at(1) = counts.at(1), original_counts.at(2) + counts.at(2)};
    beam_search_result.setNgramCounts(new_counts);
}

std::vector<BeamSearchResult> mostProbableResults(
        const std::vector<DecoderComponents> &beam,
        const std::vector<BeamSearchResult> &last_results,
        int current_word,
        int k,
        const ModelParams &model_params,
        const DefaultConfig &default_config,
        bool is_first,
        set<int> &searched_word_ids) {
    std::vector<Node *> nodes;
    for (const DecoderComponents &decoder_components : beam) {
        nodes.push_back(decoder_components.wordvector_to_onehots.at(current_word - 1));
    }
    if (nodes.size() != last_results.size() && !last_results.empty()) {
        std::cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << std::endl;
        abort();
    }

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.finalScore() > b.finalScore();
    };
    std::priority_queue<BeamSearchResult, std::vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
//    int stop_id = model_params.lookup_table.getElemId(STOP_SYMBOL);
    std::vector<BeamSearchResult> results;
    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
        const Node &node = *nodes.at(i);
        auto tuple = toExp(node);

        for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
            if (is_first) {
                if (searched_word_ids.find(j) != searched_word_ids.end()) {
                    cout << boost::format("word id searched:%1% word:%2%\n") % j %
                        model_params.lookup_table.elems.from_id(j);
                    continue;
                }
            }
            if (j == model_params.lookup_table.getElemId(::unknownkey)) {
                continue;
            }
            dtype value = node.getVal().v[j] - std::get<1>(tuple).second;
            dtype log_probability = value - log(std::get<2>(tuple));
            dtype word_probability = exp(log_probability);
            std::vector<WordIdAndProbability> word_ids;
            std::array<int, 3> counts = {0, 0, 0};
            dtype extra_score = 0.0f;
            if (!last_results.empty()) {
                log_probability += last_results.at(i).finalLogProbability();
                word_ids = last_results.at(i).getPath();
                counts = last_results.at(i).ngramCounts();
                extra_score = last_results.at(i).getExtraScore();
            }

            word_ids.push_back(WordIdAndProbability(j, word_probability));
            BeamSearchResult beam_search_result(beam.at(i), word_ids, log_probability);
            beam_search_result.setNgramCounts(counts);
            beam_search_result.setExtraScore(extra_score);
            updateBeamSearchResultScore(beam_search_result, default_config.toNgramPenalty());
            int one_gram_count = beam_search_result.ngramCounts().at(0);
            if (one_gram_count > beam_search_result.getPath().size()) {
                cout << boost::format("one_gram_count:%1% path size:%2%") % one_gram_count %
                    beam_search_result.getPath().size() << endl;
                break;
            }

            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                queue.pop();
                queue.push(beam_search_result);
            }
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        if (is_first) {
            int size = e.getPath().size();
            if (size != 1) {
                cerr << boost::format("size is not 1:%1%\n") % size;
                abort();
            }
            searched_word_ids.insert(e.getPath().at(0).word_id);
        }
        results.push_back(e);
        queue.pop();
    }

    int i = 0;
    for (const BeamSearchResult &result : results) {
        std::cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << std::endl;
        printWordIds(result.getPath(), model_params.lookup_table);
        ++i;
    }

    return results;
}

struct GraphBuilder {
    std::vector<Node *> encoder_lookups;
    DynamicLSTMBuilder left_to_right_encoder;

    void forward(Graph &graph, const std::vector<std::string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        BucketNode *hidden_bucket = new BucketNode;
        hidden_bucket->init(hyper_params.hidden_dim);
        hidden_bucket->forward(graph);
        BucketNode *word_bucket = new BucketNode;
        word_bucket->init(hyper_params.word_dim);
        word_bucket->forward(graph);

        for (const std::string &word : sentence) {
            LookupNode* input_lookup(new LookupNode);
            input_lookup->init(hyper_params.word_dim);
            input_lookup->setParam(model_params.lookup_table);
            input_lookup->forward(graph, word);

            DropoutNode* dropout_node(new DropoutNode(hyper_params.dropout, is_training));
            dropout_node->init(hyper_params.word_dim);
            dropout_node->forward(graph, *input_lookup);

            BucketNode *bucket = new BucketNode();
            bucket->init(hyper_params.hidden_dim);
            bucket->forward(graph);

            ConcatNode *concat = new ConcatNode;
            concat->init(dropout_node->getDim() + bucket->getDim());
            concat->forward(graph, {dropout_node, bucket});

            encoder_lookups.push_back(concat);
        }

        for (Node* node : encoder_lookups) {
            left_to_right_encoder.forward(graph, model_params.left_to_right_encoder_params, *node,
                    *hidden_bucket, *hidden_bucket, hyper_params.dropout, is_training);
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

            DropoutNode* decoder_lookup(new DropoutNode(hyper_params.dropout, is_training));
            decoder_lookup->init(hyper_params.word_dim);
            decoder_lookup->forward(graph, *before_dropout);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            last_input = decoder_components.decoder_lookups.at(i - 1);
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
        }

        std::vector<Node *> encoder_hiddens = transferVector<Node *, DropoutNode*>(
                left_to_right_encoder._hiddens, [](DropoutNode *dropout) {
                return dropout;
                });

        decoder_components.forward(graph, hyper_params, model_params, *last_input,
                encoder_hiddens, is_training);

        Node *decoder_to_wordvector = decoder_components.decoderToWordVectors(graph, hyper_params,
                model_params, i);
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        LinearWordVectorNode *wordvector_to_onehot(new LinearWordVectorNode);
        wordvector_to_onehot->init(model_params.lookup_table.nVSize);
        wordvector_to_onehot->setParam(model_params.lookup_table.E);
        wordvector_to_onehot->forward(graph, *decoder_to_wordvector);
        decoder_components.wordvector_to_onehots.push_back(wordvector_to_onehot);
    }

    std::pair<std::vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            const std::vector<DecoderComponents> &decoder_components_beam,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const DefaultConfig &default_config,
            bool is_training) {
        std::vector<std::pair<std::vector<WordIdAndProbability>, dtype>> word_ids_result;
        std::vector<BeamSearchResult> most_probable_results;
        std::vector<std::string> last_answers;
        bool succeeded = false;
        std::set<int> searched_word_ids;

        for (int iter = 0; ; ++iter) {
            cout << boost::format("forwardDecoderUsingBeamSearch iter:%1%\n") % iter;
            most_probable_results.clear();
            auto beam = decoder_components_beam;
            cout << boost::format("beam size:%1%\n") % beam.size();

            int ended_count = word_ids_result.size();
            if (ended_count >= k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                int left_k = k;
                if (left_k <= 0) {
                    cout << boost::format("break for left_k:%1%") % left_k << endl;
                    break;
                }
                last_answers.clear();
                if (i > 0) {
                    most_probable_results = mostProbableResults(beam, most_probable_results, i,
                            left_k, model_params, default_config, i == 1, searched_word_ids);
                    cout << boost::format("most_probable_results size:%1%") %
                        most_probable_results.size() << endl;
                    auto last_beam = beam;
                    beam.clear();
                    std::vector<BeamSearchResult> stop_removed_results;
                    int j = 0;
                    for (BeamSearchResult &beam_search_result : most_probable_results) {
//                        cout << boost::format("1gram:%1% len:%2%") %
//                            beam_search_result.ngramCounts().at(0) %
//                            beam_search_result.getPath().size() << endl;
                        const std::vector<WordIdAndProbability> &word_ids =
                            beam_search_result.getPath();
                        int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                        const std::string &word = model_params.lookup_table.elems.from_id(
                                last_word_id);
                        if (word == STOP_SYMBOL) {
                            word_ids_result.push_back(std::make_pair(word_ids,
                                        beam_search_result.finalScore()));
                            succeeded = word == STOP_SYMBOL;
                        } else {
                            stop_removed_results.push_back(beam_search_result);
                            last_answers.push_back(word);
                            beam.push_back(beam_search_result.decoderComponents());
                        }
                        ++j;
                    }
                    most_probable_results = stop_removed_results;
                }

                if (beam.empty()) {
                    cout << boost::format("break for beam empty\n");
                    break;
                }

                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderComponents &decoder_components = beam.at(beam_i);
                    forwardDecoderByOneStep(graph, decoder_components, i,
                            i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                            model_params, is_training);
                }

                graph.compute();
            }
        }

        if (word_ids_result.size() < k) {
            std::cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << std::endl;
            abort();
        }

        for (const auto &pair : word_ids_result) {
            const std::vector<WordIdAndProbability> ids = pair.first;
            std::cout << boost::format("beam result:%1%") % exp(pair.second) << std::endl;
            printWordIds(ids, model_params.lookup_table);
        }

        auto compair = [](const std::pair<std::vector<WordIdAndProbability>, dtype> &a,
                const std::pair<std::vector<WordIdAndProbability>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = std::max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return std::make_pair(max->first, exp(max->second));
    }
};

#endif
