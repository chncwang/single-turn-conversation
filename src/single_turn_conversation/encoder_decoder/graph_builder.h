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
#include "tinyutf8.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

using namespace std;

struct WordIdAndProbability {
    int word_id;
    dtype probability;

    WordIdAndProbability() = default;
    WordIdAndProbability(const WordIdAndProbability &word_id_and_probability) = default;
    WordIdAndProbability(int wordid, dtype prob) : word_id(wordid), probability(prob) {}
};

string getSentence(const vector<int> &word_ids_vector, const ModelParams &model_params) {
    string words;
    for (const int &w : word_ids_vector) {
        string str = model_params.lookup_table.elems.from_id(w);
        words += str;
    }
    return words;
}

class BeamSearchResult {
public:
    BeamSearchResult() {
        ngram_counts_ = {0, 0, 0};
    }
    BeamSearchResult(const BeamSearchResult &beam_search_result) = default;
    BeamSearchResult(const DecoderComponents &decoder_components,
            const vector<WordIdAndProbability> &pathh,
            dtype log_probability) : decoder_components_(decoder_components), path_(pathh),
            final_log_probability(log_probability) {
                ngram_counts_ = {0, 0, 0};
            }

    dtype finalScore() const {
        set<int> unique_words;
        int i = 0;
        for (const auto &p : path_) {
            if (i % 2 == 1) {
                unique_words.insert(p.word_id);
            }
            ++i;
        }
        return (final_log_probability + extra_score_) / (unique_words.size() + 1e-10);
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
    vector<WordIdAndProbability> path_;
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

vector<BeamSearchResult> mostProbableResults(
        const vector<DecoderComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        int current_word,
        int k,
        const ModelParams &model_params,
        const DefaultConfig &default_config,
        const vector<string> &black_list) {
    vector<Node *> nodes;
    int beam_i = 0;
    for (const DecoderComponents &decoder_components : beam) {
        auto path = last_results.at(beam_i).getPath();
        if (path.size() % 2 == 0) {
            cerr << "path is even" << endl;
            abort();
        }
        int path_size = path.size();
        bool should_append_keyword = path_size == 1 || path.at(path_size - 2).word_id ==
            path.at(path_size - 3).word_id;
        Node *node = should_append_keyword ? decoder_components.keyword_vector_to_onehots.back() :
            decoder_components.wordvector_to_onehots.at(current_word - 1);
        nodes.push_back(node);
        ++beam_i;
    }
    if (nodes.size() != last_results.size() && !last_results.empty()) {
        cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << endl;
        abort();
    }

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
    for (int i = 0; i < nodes.size(); ++i) {
        const Node &node = *nodes.at(i);
        auto tuple = toExp(node);

        for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
            if (j == model_params.lookup_table.getElemId(::unknownkey)) {
                continue;
            }
            dtype value = node.getVal().v[j] - get<1>(tuple).second;
            dtype log_probability = value - log(get<2>(tuple));
            dtype word_probability = exp(log_probability);
            vector<WordIdAndProbability> word_ids;
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
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
        bool contain_black = false;
        for (const string str : black_list) {
            utf8_string utf8_str(str), utf8_sentece(sentence);
            if (utf8_sentece.find(utf8_str) != string::npos) {
                contain_black = true;
                break;
            }
        }
        if (contain_black) {
            continue;
        }
        final_results.push_back(result);
        cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIds(result.getPath(), model_params.lookup_table);
        ++i;
    }

    return final_results;
}

vector<BeamSearchResult> mostProbableKeywords(
        vector<DecoderComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        int word_pos,
        int k,
        Graph &graph,
        ModelParams &model_params,
        const HyperParams &hyper_params,
        bool is_first) {
    vector<Node *> nodes;
    for (int ii = 0; ii < beam.size(); ++ii) {
        bool should_predict_keyword;
        if (last_results.empty()) {
            should_predict_keyword = true;
        } else {
            vector<WordIdAndProbability> path = last_results.at(ii).getPath();
            int size = path.size();
            should_predict_keyword = path.at(size - 2).word_id == path.at(size - 1).word_id;
        }
        Node *node;
        if (should_predict_keyword) {
            DecoderComponents &components = beam.at(ii);

            ConcatNode *concat_node = new ConcatNode();
            int context_dim = components.contexts.at(0)->getDim();
            concat_node->init(context_dim + hyper_params.hidden_dim + 2 * hyper_params.word_dim);
            if (components.decoder_lookups.size() != word_pos) {
                cerr << boost::format("size:%1% word_pos:%2%") % components.decoder_lookups.size()
                    % word_pos << endl;
                abort();
            }
            vector<Node *> concat_inputs = {
                components.contexts.at(word_pos), components.decoder._hiddens.at(word_pos),
                word_pos == 0 ? components.bucket(hyper_params.word_dim, graph) :
                    static_cast<Node*>(components.decoder_lookups.at(word_pos - 1)),
                word_pos == 0 ? components.bucket(hyper_params.word_dim, graph) :
                    static_cast<Node*>(components.decoder_keyword_lookups.at(word_pos - 1))
            };
            concat_node->forward(graph, concat_inputs);

            UniNode *keyword = new UniNode;
            keyword->init(hyper_params.word_dim);
            keyword->setParam(model_params.hidden_to_keyword_params);
            keyword->forward(graph, *concat_node);

            LinearWordVectorNode *keyword_vector_to_onehot = new LinearWordVectorNode;
            keyword_vector_to_onehot->init(model_params.lookup_table.nVSize);
            keyword_vector_to_onehot->setParam(model_params.lookup_table.E);
            keyword_vector_to_onehot->forward(graph, *keyword);
            components.keyword_vector_to_onehots.push_back(keyword_vector_to_onehot);
            node = keyword_vector_to_onehot;
        } else {
            node = nullptr;
        }
        nodes.push_back(node);
    }
    graph.compute();

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
        const Node *node_ptr = nodes.at(i);
        if (node_ptr == nullptr) {
            vector<WordIdAndProbability> new_id_and_probs = last_results.at(i).getPath();
            WordIdAndProbability &last_keyword = new_id_and_probs.at(new_id_and_probs.size() - 2);
            WordIdAndProbability &last_norm = new_id_and_probs.at(new_id_and_probs.size() - 1);
            WordIdAndProbability w = {last_keyword.word_id, last_norm.probability};
            new_id_and_probs.push_back(w);
            BeamSearchResult beam_search_result(beam.at(i), new_id_and_probs,
                    last_results.at(i).finalLogProbability());
            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                queue.pop();
                queue.push(beam_search_result);
            }
        } else {
            const Node &node = *nodes.at(i);
            auto tuple = toExp(node);

            for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
                if (j == model_params.lookup_table.getElemId(::unknownkey)) {
                    continue;
                }
                dtype value = node.getVal().v[j] - get<1>(tuple).second;
                dtype log_probability = value - log(get<2>(tuple));
                dtype word_probability = exp(log_probability);
                vector<WordIdAndProbability> word_ids;
                if (!last_results.empty()) {
                    log_probability += last_results.at(i).finalLogProbability();
                    word_ids = last_results.at(i).getPath();
                }

                word_ids.push_back(WordIdAndProbability(j, word_probability));
                BeamSearchResult beam_search_result(beam.at(i), word_ids, log_probability);

                if (queue.size() < k) {
                    queue.push(beam_search_result);
                } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                    queue.pop();
                    queue.push(beam_search_result);
                }
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
        }
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
        final_results.push_back(result);
        cout << boost::format("mostProbableKeywords - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIds(result.getPath(), model_params.lookup_table);
        ++i;
    }

    return final_results;
}

struct GraphBuilder {
    vector<Node *> encoder_lookups;
    DynamicLSTMBuilder left_to_right_encoder;

    void forward(Graph &graph, const vector<string> &sentence,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        BucketNode *hidden_bucket = new BucketNode;
        hidden_bucket->init(hyper_params.hidden_dim);
        hidden_bucket->forward(graph);
        BucketNode *word_bucket = new BucketNode;
        word_bucket->init(hyper_params.word_dim);
        word_bucket->forward(graph);

        for (const string &word : sentence) {
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
            const std::vector<std::string> &keywords,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        for (int i = 0; i < answer.size(); ++i) {
            forwardDecoderByOneStep(graph, decoder_components, i,
                    i == 0 ? nullptr : &answer.at(i - 1), keywords.at(i),
                    i == 0 ||  answer.at(i - 1) == keywords.at(i - 1), hyper_params,
                    model_params, is_training);
        }
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const std::string &keyword,
            bool should_predict_keyword,
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
            if (decoder_components.decoder_lookups.size() != i) {
                cerr << boost::format("decoder_lookups size:%1% i:%2%") %
                    decoder_components.decoder_lookups.size() % i << endl;
                abort();
            }
            last_input = decoder_components.decoder_lookups.at(i - 1);
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
        }

        LookupNode *keyword_node(new LookupNode);
        keyword_node->init(hyper_params.word_dim);
        keyword_node->setParam(model_params.lookup_table);
        keyword_node->forward(graph, keyword);
        decoder_components.decoder_keyword_lookups.push_back(keyword_node);

        vector<Node *> encoder_hiddens = transferVector<Node *, DropoutNode*>(
                left_to_right_encoder._hiddens, [](DropoutNode *dropout) {
                return dropout;
                });

        decoder_components.forward(graph, hyper_params, model_params, *last_input, encoder_hiddens,
                is_training);

        auto nodes = decoder_components.decoderToWordVectors(graph, hyper_params,
                model_params, i, should_predict_keyword);
        Node *decoder_to_wordvector = nodes.result;
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        LinearWordVectorNode *wordvector_to_onehot(new LinearWordVectorNode);
        wordvector_to_onehot->init(model_params.lookup_table.nVSize);
        wordvector_to_onehot->setParam(model_params.lookup_table.E);
        wordvector_to_onehot->forward(graph, *decoder_to_wordvector);
        decoder_components.wordvector_to_onehots.push_back(wordvector_to_onehot);

        decoder_components.decoder_to_keyword_vectors.push_back(nodes.keyword);

        LinearWordVectorNode *keyword_vector_to_onehot;
        if (nodes.keyword == nullptr) {
            keyword_vector_to_onehot = nullptr;
        } else {
            keyword_vector_to_onehot = new LinearWordVectorNode;
            keyword_vector_to_onehot->init(model_params.lookup_table.nVSize);
            keyword_vector_to_onehot->setParam(model_params.lookup_table.E);
            keyword_vector_to_onehot->forward(graph, *nodes.keyword);
        }
        decoder_components.keyword_vector_to_onehots.push_back(keyword_vector_to_onehot);
    }

    void forwardDecoderResultByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        LookupNode *keyword_lookup = new LookupNode;
        keyword_lookup->init(hyper_params.word_dim);
        keyword_lookup->setParam(model_params.lookup_table);
        keyword_lookup->forward(graph, keyword);
        decoder_components.decoder_keyword_lookups.push_back(keyword_lookup);
    }

    void forwardDecoderHiddenByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        Node *last_input;
        if (i > 0) {
            LookupNode* before_dropout(new LookupNode);
            before_dropout->init(hyper_params.word_dim);
            before_dropout->setParam(model_params.lookup_table);
            before_dropout->forward(graph, *answer);
            decoder_components.decoder_lookups_before_dropout.push_back(before_dropout);

            DropoutNode* decoder_lookup(new DropoutNode(hyper_params.dropout, false));
            decoder_lookup->init(hyper_params.word_dim);
            decoder_lookup->forward(graph, *before_dropout);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            if (decoder_components.decoder_lookups.size() != i) {
                cerr << boost::format("decoder_lookups size:%1% i:%2%") %
                    decoder_components.decoder_lookups.size() % i << endl;
                abort();
            }
            last_input = decoder_components.decoder_lookups.at(i - 1);
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
        }

        vector<Node *> encoder_hiddens = transferVector<Node *, DropoutNode*>(
                left_to_right_encoder._hiddens, [](DropoutNode *dropout) {
                return dropout;
                });

        decoder_components.forward(graph, hyper_params, model_params, *last_input, encoder_hiddens,
                false);
    }

    void forwardDecoderKeywordByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        LookupNode *keyword_embedding = new LookupNode;
        keyword_embedding->init(hyper_params.word_dim);
        keyword_embedding->setParam(model_params.lookup_table);
        keyword_embedding->forward(graph, keyword);
        if (decoder_components.decoder_keyword_lookups.size() != i) {
            cerr << "keyword lookup size:" << decoder_components.decoder_keyword_lookups.size()
                << endl;
            abort();
        }
        decoder_components.decoder_keyword_lookups.push_back(keyword_embedding);
        ResultAndKeywordVectors result =  decoder_components.decoderToWordVectors(graph,
                hyper_params, model_params, i, false);
        Node *result_node = result.result;

        LinearWordVectorNode *one_hot_node = new LinearWordVectorNode;
        one_hot_node->init(model_params.lookup_table.nVSize);
        one_hot_node->setParam(model_params.lookup_table.E);
        one_hot_node->forward(graph, *result_node);
        decoder_components.wordvector_to_onehots.push_back(one_hot_node);
    }

    pair<vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            const vector<DecoderComponents> &decoder_components_beam,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const DefaultConfig &default_config,
            const vector<string> &black_list) {
        vector<pair<vector<WordIdAndProbability>, dtype>> word_ids_result;
        vector<BeamSearchResult> most_probable_results;
        vector<string> last_answers, last_keywords;
        bool succeeded = false;

        auto beam = decoder_components_beam;
        cout << boost::format("beam size:%1%\n") % beam.size();

        for (int i = 0;; ++i) {
            cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
            int left_k = k;
            if (word_ids_result.size() >= k || i > default_config.cut_length) {
                break;
            }

            for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                DecoderComponents &decoder_components = beam.at(beam_i);
                forwardDecoderHiddenByOneStep(graph, decoder_components, i,
                        i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                        model_params);
                if (i > 0) {
                    cout << "last:" << last_answers.at(beam_i) << endl;
                }
            }
            cout << "forwardDecoderHiddenByOneStep:" << endl;
            graph.compute();

            most_probable_results = mostProbableKeywords(beam, most_probable_results, i, k, graph,
                    model_params, hyper_params, i == 0);
            for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                DecoderComponents &decoder_components = beam.at(beam_i);
                int keyword_id = most_probable_results.at(beam_i).getPath().back().word_id;
                string keyword = model_params.lookup_table.elems.from_id(keyword_id);
                forwardDecoderResultByOneStep(graph, decoder_components, i, keyword, hyper_params,
                        model_params);
            }
            cout << "forwardDecoderResultByOneStep:" << endl;
            graph.compute();
            beam.clear();
            vector<BeamSearchResult> search_results;

            for (int j = 0; j < most_probable_results.size(); ++j) {
                BeamSearchResult &beam_search_result = most_probable_results.at(j);
                const vector<WordIdAndProbability> &word_ids = beam_search_result.getPath();
                int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                const string &word = model_params.lookup_table.elems.from_id(last_word_id);
                search_results.push_back(beam_search_result);
                last_keywords.push_back(word);
                beam.push_back(beam_search_result.decoderComponents());
            }
            most_probable_results = search_results;
            
            int beam_i = 0;
            for (auto &decoder_components : beam) {
                forwardDecoderKeywordByOneStep(graph, decoder_components, i,
                        last_keywords.at(beam_i), hyper_params, model_params);
                ++beam_i;
            }
            last_keywords.clear();
            cout << "forwardDecoderKeywordByOneStep:" << endl;
            graph.compute();

            last_answers.clear();
            most_probable_results = mostProbableResults(beam, most_probable_results, i,
                    left_k, model_params, default_config, black_list);
            cout << boost::format("most_probable_results size:%1%") %
                most_probable_results.size() << endl;
            beam.clear();
            vector<BeamSearchResult> stop_removed_results;
            int j = 0;
            for (BeamSearchResult &beam_search_result : most_probable_results) {
                //                        cout << boost::format("1gram:%1% len:%2%") %
                //                            beam_search_result.ngramCounts().at(0) %
                //                            beam_search_result.getPath().size() << endl;
                const vector<WordIdAndProbability> &word_ids =
                    beam_search_result.getPath();

                int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                const string &word = model_params.lookup_table.elems.from_id(
                        last_word_id);
                if (word == STOP_SYMBOL) {
                    word_ids_result.push_back(make_pair(word_ids,
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

            if (beam.empty()) {
                cout << boost::format("break for beam empty\n");
                break;
            }
        }

        if (word_ids_result.size() < k) {
            cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << endl;
            abort();
        }

        for (const auto &pair : word_ids_result) {
            const vector<WordIdAndProbability> ids = pair.first;
            cout << boost::format("beam result:%1%") % exp(pair.second) << endl;
            printWordIds(ids, model_params.lookup_table);
        }

        auto compair = [](const pair<vector<WordIdAndProbability>, dtype> &a,
                const pair<vector<WordIdAndProbability>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return make_pair(max->first, exp(max->second));
    }
};

#endif
