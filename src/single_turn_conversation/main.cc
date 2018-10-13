#include "cxxopts.hpp"
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <memory>
#include <string>
#include "N3LDG.h"
#include "single_turn_conversation/data_manager.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/encoder_decoder/graph_builder.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"

using namespace std;
using namespace cxxopts;

void exportToOptimizer(ModelParams &model_params, ModelUpdate &model_update) {
    model_params.decoder_params.exportAdaParams(model_update);
    model_params.encoder_params.exportAdaParams(model_update);
    model_params.hidden_to_wordvector_params.exportAdaParams(model_update);
    model_params.wordvector_to_onehot_params.exportAdaParams(model_update);
    model_params.lookup_table.exportAdaParams(model_update);
}

void addWord(unordered_map<string, int> &word_counts, const string &word) {
    auto it = word_counts.find(word);
    if (it == word_counts.end()) {
        word_counts.insert(make_pair(word, 1));
    } else {
        it->second++;
    }
}

void addWord(unordered_map<string, int> &word_counts, const vector<string> &sentence) {
    for (const string &word : sentence) {
        addWord(word_counts, word);
    }
}

HyperParams parseHyperParams(INIReader &ini_reader) {
    HyperParams hyper_params;

    int word_dim = ini_reader.GetInteger("hyper", "word_dim", 0);
    if (word_dim <= 0) {
        cerr << "word_dim wrong" << endl;
        abort();
    }
    hyper_params.word_dim = word_dim;

    int hidden_dim = ini_reader.GetInteger("hyper", "hidden_dim", 0);
    if (hidden_dim <= 0) {
        cerr << "hidden_dim wrong" << endl;
        abort();
    }
    hyper_params.hidden_dim = hidden_dim;

    float dropout = ini_reader.GetReal("hyper", "dropout", 0.0);
    if (dropout < -1.0f || dropout >=1.0f) {
        cerr << "dropout wrong" << endl;
        abort();
    }
    hyper_params.dropout = dropout;

    int batchsize = ini_reader.GetInteger("hyper", "batchsize", 0);
    if (batchsize == 0) {
        cerr << "batchsize not found" << endl;
        abort();
    }
    hyper_params.batchsize = batchsize;

    float learning_rate = ini_reader.GetReal("hyper", "learning_rate", 0.001f);
    if (learning_rate <= 0.0f) {
        cerr << "learning_rate wrong" << endl;
        abort();
    }
    hyper_params.learning_rate = learning_rate;

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, LookupTable &lookup_table) {
    vector<int> ids;
    for (const string &word : sentence) {
        ids.push_back(lookup_table.getElemId(word));
    }
    return ids;
}

int main(int argc, char *argv[]) {
    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
        ("config", "config file name", cxxopts::value<string>())
        ("pair", "pair file name", cxxopts::value<string>())
        ("post", "post file name", cxxopts::value<string>())
        ("response", "response file name", cxxopts::value<string>());
    auto args = options.parse(argc, argv);
    string pair_filename = args["pair"].as<string>();

    vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(pair_filename);
    cout << "post_and_responses_vector size:" << post_and_responses_vector.size() << endl;

    const int SEED = 0;
    std::default_random_engine engine(SEED);
    std::shuffle(std::begin(post_and_responses_vector), std::end(post_and_responses_vector),
            engine);
    vector<PostAndResponses> dev_post_and_responses, test_post_and_responses,
        train_post_and_responses;
    vector<ConversationPair> train_conversation_pairs;
    int i = 0;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        if (i < 0) {
            dev_post_and_responses.push_back(post_and_responses);
        } else if (i < 0) {
            test_post_and_responses.push_back(post_and_responses);
        } else {
            train_post_and_responses.push_back(post_and_responses);
            vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
            for (ConversationPair &conversation_pair : conversation_pairs) {
                train_conversation_pairs.push_back(std::move(conversation_pair));
            }
        }
        ++i;
    }

    cout << "train size:" << train_conversation_pairs.size() << " dev size:" <<
        dev_post_and_responses.size() << " test size:" << test_post_and_responses.size() << endl;

    string post_filename = args["post"].as<string>();
    vector<vector<string>> post_sentences = readSentences(post_filename);

    string response_filename = args["response"].as<string>();
    vector<vector<string>> response_sentences = readSentences(response_filename);

    unordered_map<string, int> word_counts;
    for (const ConversationPair &conversation_pair : train_conversation_pairs) {
        const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
        addWord(word_counts, post_sentence);

        const vector<string> &response_sentence = response_sentences.at(
                conversation_pair.response_id);
        addWord(word_counts, response_sentence);
    }
    word_counts[unknownkey] = 1000000;
    word_counts[STOP_SYMBOL] = 1000000;
    Alphabet alphabet;
    alphabet.initial(word_counts, 0);
    ModelParams model_params;

    string configfilename = args["config"].as<string>();
    INIReader ini_reader(configfilename);
    if (ini_reader.ParseError() < 0) {
        cerr << "parse ini failed" << endl;
        abort();
    }
    HyperParams hyper_params = parseHyperParams(ini_reader);
    hyper_params.Print();

    model_params.lookup_table.initial(&alphabet, hyper_params.word_dim, true);
    model_params.encoder_params.initial(hyper_params.hidden_dim, hyper_params.word_dim);
    model_params.decoder_params.initial(hyper_params.hidden_dim, hyper_params.word_dim);
    model_params.hidden_to_wordvector_params.initial(hyper_params.word_dim,
            hyper_params.hidden_dim);
    model_params.wordvector_to_onehot_params.initial(alphabet.size(), hyper_params.word_dim);

    ModelUpdate model_update;
    model_update._alpha = hyper_params.learning_rate;
    exportToOptimizer(model_params, model_update);

    for (int epoch = 0; epoch<100000; ++epoch) {
        cout << "epoch:" << epoch << endl;
        shuffle(begin(train_conversation_pairs), end(train_conversation_pairs), engine);
        for (int batch_i = 0; batch_i < train_conversation_pairs.size() / hyper_params.batchsize;
                ++batch_i) {
            cout << "batch_i:" << batch_i << endl;
            Graph graph;
            graph.train = true;
            vector<shared_ptr<GraphBuilder>> graph_builders;
            for (int i = 0; i < hyper_params.batchsize; ++i) {
                shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                graph_builders.push_back(graph_builder);
                graph_builder->init(hyper_params);
                int instance_index = batch_i * hyper_params.batchsize + i;
                int post_id = train_conversation_pairs.at(instance_index).post_id;
                graph_builder->forward(graph, post_sentences.at(i), hyper_params, model_params);
                int response_id = train_conversation_pairs.at(instance_index).response_id;
                int response_size = response_sentences.at(response_id).size();
                graph_builder->forwardDecoder(graph, response_size, hyper_params, model_params);
            }

            graph.compute();

            dtype loss_sum = 0.0f;
            for (int i = 0; i < hyper_params.batchsize; ++i) {
                int instance_index = batch_i * hyper_params.batchsize + i;
                int response_id = train_conversation_pairs.at(instance_index).response_id;
                vector<int> word_ids = toIds(response_sentences.at(response_id),
                        model_params.lookup_table);
                vector<Node*> result_nodes =
                    toNodePointers(graph_builders.at(i)->wordvector_to_onehots);
                dtype loss = MaxLogProbabilityLoss(result_nodes, word_ids, hyper_params.batchsize);
                loss_sum += loss;
            }
            cout << "loss:" << loss_sum << endl;
            graph.backward();
            model_update.updateAdam(10.0f);
        }
    }

    return 0;
}
