#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H

#include <string>
#include <unordered_map>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <regex>
#include <iostream>
#include <utility>
#include "single_turn_conversation/conversation_structure.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/default_config.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>

using boost::format;

std::vector<PostAndResponses> readPostAndResponsesVector(const std::string &filename) {
    DefaultConfig &default_config = GetDefaultConfig();
    std::vector<PostAndResponses> results;
    std::string line;
    std::ifstream ifs(filename);
    int i = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(":"));
        if (strs.size() != 2) {
            abort();
        }
        int post_id = stoi(strs.at(0));
        PostAndResponses post_and_responses;
        post_and_responses.post_id = post_id;
        std::vector<std::string> strs2;
        boost::split(strs2, strs.at(1), boost::is_any_of(","));
        for (std::string &str : strs2) {
            post_and_responses.response_ids.push_back(stoi(str));
            if (default_config.one_response) {
                break;
            }
        }
        results.push_back(std::move(post_and_responses));
        if (++i >= default_config.max_sample_count) {
            break;
        }
    }

    return results;
}

std::vector<ConversationPair> toConversationPairs(const PostAndResponses &post_and_responses) {
    std::vector<ConversationPair> results;
    for (int response_id : post_and_responses.response_ids) {
        ConversationPair conversation_pair(post_and_responses.post_id, response_id);
        results.push_back(std::move(conversation_pair));
    }
    return results;
}

std::vector<ConversationPair> toConversationPairs(
        const std::vector<PostAndResponses> &post_and_responses_vector) {
    std::vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        std::vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (const ConversationPair & conversation_pair : conversation_pairs) {
            results.push_back(conversation_pair);
        }
    }
    return results;
}

std::vector<ConversationPair> readConversationPairs(const std::string &filename) {
    std::vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(filename);
    std::vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        std::vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            results.push_back(std::move(conversation_pair));
        }
    }

    return results;
}

std::vector<std::vector<std::string>> readSentences(const std::string &filename) {
    std::string line;
    std::ifstream ifs(filename);
    std::vector<std::vector<std::string>> results;

    int i = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::string> strs;
        boost::split_regex(strs, line, boost::regex("##"));
        int index = stoi(strs.at(0));
        if (i != index) {
            abort();
        }

        const std::string &sentence = strs.at(1);
        std::vector<std::string> words;
        boost::split(words, sentence, boost::is_any_of(" "));
        words.push_back(STOP_SYMBOL);
        results.push_back(words);
        ++i;
    }

    return results;
}

bool isPureChinese(const string &word) {
    std::regex expression("^[\u4e00-\u9fff]+$");
    return std::regex_search(word, expression);
}

vector<string> reprocessSentence(const vector<string> &sentence,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    vector<string> processed_sentence;
    for (const string &word : sentence) {
        if (isPureChinese(word)) {
            auto it = word_counts.find(word);
            int occurence;
            if (it == word_counts.end()) {
                cout << format("word not found:%1%\n") % word;
                occurence = 0;
            } else {
                occurence = it->second;
            }
            if (occurence <= min_occurences) {
                for (int i = 0; i < word.size(); i += 3) {
                    processed_sentence.push_back(word.substr(i, 3));
                }
            } else {
                processed_sentence.push_back(word);
            }
        } else {
            processed_sentence.push_back(word);
        }
    }
    return processed_sentence;
}

vector<vector<string>> reprocessSentences(const vector<vector<string>> &sentences,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    cout << boost::format("sentences size:%1%") % sentences.size() << endl;
    vector<vector<string>> result;
    int i = 0;
    for (const auto &sentence : sentences) {
        if (i % 1000 == 0) {
            cout << static_cast<float>(i) / sentences.size() << endl;
        }
        result.push_back(reprocessSentence(sentence, word_counts, min_occurences));
        ++i;
    }
    return result;
}

void reprocessSentences(const vector<PostAndResponses> bundles,
        vector<vector<string>> &posts,
        vector<vector<string>> &responses,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    vector<vector<string>> result;

    cout << "bund count:" << bundles.size() << endl;
    int i = 0;

    for (const PostAndResponses &bundle : bundles) {
        cout << i++ / (float)bundles.size() << endl;
        int post_id = bundle.post_id;
        auto &post = posts.at(post_id);
        post = reprocessSentence(post, word_counts, min_occurences);
        for (int response_id : bundle.response_ids) {
            auto &res = responses.at(response_id);
            res = reprocessSentence(res, word_counts, min_occurences);
        }
    }
}

#endif
