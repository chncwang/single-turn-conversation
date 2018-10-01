#include "data_manager.h"
#include "cxxopts.hpp"
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <string>

using namespace std;
using namespace cxxopts;

void addWord(unordered_map<string, int> &word_counts, const std::string &word) {
    auto it = word_counts.find(word);
    if (it == word_counts.end()) {
        word_counts.insert(make_pair<string, int>(word, 1));
    } else {
        it->second++;
    }
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
    vector<PostAndResponses> dev_post_and_responses, test_post_and_responses;
    vector<ConversationPair> train_conversation_pairs;
    int i = 0;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        if (i < 100) {
            dev_post_and_responses.push_back(post_and_responses);
        } else if (i < 200) {
            test_post_and_responses.push_back(post_and_responses);
        } else {
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

    string configfilename = args["config"].as<string>();
    INIReader ini_reader(configfilename);
    if (ini_reader.ParseError() < 0) {
        cerr << "parse ini failed" << endl;
        abort();
    }
    int batchsize = ini_reader.Get("hyper", "batchsize", 0);
    if (batchsize == 0) {
        cout << "batchsize not found" << endl;
        abort();
    }

    for (int epoch = 0; epoch<1000; ++epoch) {
        std::shuffle(std::begin(train_conversation_pairs), std::end(train_conversation_pairs),
                engine);
        for (int batch_i = 0; batch_i < train_conversation_pairs.size() / batchsize; ++batch_i) {
        }
    }

    return 0;
}
