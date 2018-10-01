#include "data_manager.h"
#include "cxxopts.hpp"
#include <algorithm>
#include <random>

using namespace std;
using namespace cxxopts;

int main(int argc, char *argv[]) {
    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
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

    return 0;
}
