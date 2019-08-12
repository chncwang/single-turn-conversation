#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <regex>
#include <iostream>
#include <algorithm>
#include <utility>
#include <atomic>
#include <mutex>
#include "N3LDG.h"
#include "single_turn_conversation/conversation_structure.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/default_config.h"
#include "tinyutf8.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/asio.hpp>

using namespace std;
using boost::format;
using namespace boost::asio;

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

bool isPureChinese(const string &word) {
    std::regex expression("^[\u4e00-\u9fff]+$");
    return std::regex_search(word, expression);
}

bool containChinese(const utf8_string &word) {
    return word.size() != word.length();
}

bool isPureEnglish(const utf8_string &word) {
    if (containChinese(word)) {
        return false;
    }
    for (int i = 0; i < word.length(); ++i) {
        char c = word.at(i);
        if (!((c == '-' || c == '.' || c == '/' || c == ':' || c == '_') || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
            return false;
        }
    }
    return true;
}

bool isPureNumber(const utf8_string &word) {
    for (int i = 0; i < word.length(); ++i) {
        char c = word.at(i);
        if (!(c == '.' || (c >= '0' && c <= '9'))) {
            return false;
        }
    }
    return true;
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
        std::vector<string> words;
        boost::split(words, sentence, boost::is_any_of(" "));
        std::vector<utf8_string> utf8_words;
        for (const string &word : words) {
            utf8_string s(word);
            utf8_words.push_back(s);
        }

        std::vector<std::string> characters;
        for (const utf8_string &word : utf8_words) {
            if (isPureEnglish(word) && !isPureNumber(word)) {
                string w;
                for (int i = 0; i < word.length(); ++i) {
                    char c = word.at(i);
                    if (c >= 'A' && c <= 'Z') {
                        c += 'a' - 'A';
                    }
                    w += c;
                }
                characters.push_back(w);
            } else {
                characters.push_back(word.cpp_str());
            }
//            if (isPureEnglish(word) && !isPureNumber(word)) {
//                string w;
//                for (int i = 0; i < word.length(); ++i) {
//                    char c = word.at(i);
//                    if (c >= 'A' && c <= 'Z') {
//                        c += 'a' - 'A';
//                    }
//                    w += c;
//                }
//                characters.push_back(w);
//            } else {
//                for (int i = 0; i < word.length(); ++i) {
//                    characters.push_back(word.substr(i, 1).cpp_str());
//                }
//            }
        }

        characters.push_back(STOP_SYMBOL);
        results.push_back(characters);
        if (i % 10000 == 0) {
            cout << boost::format("i:%1%\n") % i;
            for (const string &c : characters) {
                cout << c << " ";
            }
            cout << endl;
        }
        ++i;
    }

    return results;
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
        const unordered_set<string> &words,
        const unordered_set<int> &ids) {
    cout << boost::format("sentences size:%1%") % sentences.size() << endl;

    thread_pool pool(16);
    vector<vector<string>> result;
    map<int, vector<string>> result_map;
    mutex result_mutex;
    mutex cout_mutex;
    atomic_int i(0);
    int id = 0;
    for (const auto &sentence : sentences) {
        auto f = [&, id]() {
            if (i % 1000 == 0) {
                cout_mutex.lock();
                cout << static_cast<float>(i) / sentences.size() << endl;
                cout_mutex.unlock();
            }
            vector<string> processed_sentence;
            if (ids.find(id) == ids.end()) {
                processed_sentence = sentence;
            } else {
                for (const string &word : sentence) {
                    if (isPureChinese(word)) {
                        auto it = words.find(word);
                        if (it == words.end()) {
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
            }
            result_mutex.lock();
            result_map.insert(make_pair(id, processed_sentence));
            result_mutex.unlock();
            ++i;
        };
        post(pool, f);
        ++id;
    }
    pool.join();

    for (int i = 0; i < sentences.size(); ++i) {
        auto it = result_map.find(i);
        if (it == result_map.end()) {
            cerr << boost::format("id %1% not found\n") % i;
            abort();
        }
        result.push_back(it->second);
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

struct WordIdfInfo {
    vector<float> word_idfs;
    vector<string> keywords_behind;
    WordIdfInfo() = default;
    WordIdfInfo(const WordIdfInfo&) = delete;
    WordIdfInfo(WordIdfInfo&&) = default;
};

inline WordIdfInfo getWordIdfInfo(const vector<string> &sentence,
        const unordered_map<string, float> &word_idfs,
        const unordered_map<string, int> word_counts,
        int cutoff) {
    WordIdfInfo word_idf_info;
    word_idf_info.word_idfs.reserve(sentence.size());
    word_idf_info.keywords_behind.reserve(sentence.size());

    for (const string &word : sentence) {
        float idf;
        auto it = word_counts.find(word);
        if (it == word_counts.end()) {
            idf = -1;
        } else if (it->second <= cutoff) {
            idf = -1;
        } else {
            auto it = word_idfs.find(word);
            idf = it->second;
        }
        word_idf_info.word_idfs.push_back(idf);
    }

    auto &word_frequencies = word_idf_info.word_idfs;
    for (int i = 0; i < word_frequencies.size(); ++i) {
        auto it = std::max_element(word_frequencies.begin() + i, word_frequencies.end());
        string word = sentence.at(it - word_frequencies.begin());
        if (word == ::unknownkey) {
            cerr << word_counts.at(::unknownkey) << endl;
            abort();
        }
        word_idf_info.keywords_behind.push_back(word);
    }
    return word_idf_info;
}

std::vector<std::string> readBlackList(const std::string &filename) {
    std::string line;
    std::ifstream ifs(filename);
    std::vector<std::string> result;
    cout << "black:" << endl;
    while (std::getline(ifs, line)) {
        cout << line << endl;
        result.push_back(line);
    }
    return result;
}

#endif
