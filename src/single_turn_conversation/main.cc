#include "cxxopts.hpp"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <mutex>
#include <atomic>
#include <boost/format.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "N3LDG.h"
#include "single_turn_conversation/data_manager.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/bleu.h"
#include "single_turn_conversation/perplex.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/graph_builder.h"
#include "single_turn_conversation/encoder_decoder/hyper_params.h"
#include "single_turn_conversation/encoder_decoder/model_params.h"

using namespace std;
using namespace cxxopts;
using namespace boost::asio;
using boost::is_any_of;
using boost::format;
using boost::filesystem::path;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;

void exportToOptimizer(ModelParams &model_params, ModelUpdate &model_update) {
    model_params.left_to_right_encoder_params.exportAdaParams(model_update);
    model_params.hidden_to_wordvector_params.exportAdaParams(model_update);
    model_params.lookup_table.exportAdaParams(model_update);
    model_params.attention_parrams.exportAdaParams(model_update);
}

void exportToGradChecker(ModelParams &model_params, CheckGrad &grad_checker) {
    grad_checker.add(model_params.lookup_table.E, "lookup_table");
    grad_checker.add(model_params.hidden_to_wordvector_params.W, "hidden_to_wordvector_params W");
    grad_checker.add(model_params.hidden_to_wordvector_params.b, "hidden_to_wordvector_params b");
    grad_checker.add(model_params.left_to_right_encoder_params.cell_hidden.W,
            "left to right encoder cell_hidden W");
    grad_checker.add(model_params.attention_parrams.bi_atten.W1, "attention W1");
    grad_checker.add(model_params.attention_parrams.bi_atten.W2, "attention W2");
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

DefaultConfig parseDefaultConfig(INIReader &ini_reader) {
    DefaultConfig default_config;
    static const string SECTION = "default";
    default_config.pair_file = ini_reader.Get(SECTION, "pair_file", "");
    if (default_config.pair_file.empty()) {
        cerr << "pair file empty" << endl;
        abort();
    }

    default_config.post_file = ini_reader.Get(SECTION, "post_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    default_config.response_file = ini_reader.Get(SECTION, "response_file", "");
    if (default_config.post_file.empty()) {
        cerr << "post file empty" << endl;
        abort();
    }

    string program_mode_str = ini_reader.Get(SECTION, "program_mode", "");
    ProgramMode program_mode;
    if (program_mode_str == "interacting") {
        program_mode = ProgramMode::INTERACTING;
    } else if (program_mode_str == "training") {
        program_mode = ProgramMode::TRAINING;
    } else if (program_mode_str == "decoding") {
        program_mode = ProgramMode::DECODING;
    } else if (program_mode_str == "metric") {
        program_mode = ProgramMode::METRIC;
    } else {
        cout << format("program mode is %1%") % program_mode_str << endl;
        abort();
    }
    default_config.program_mode = program_mode;

    default_config.check_grad = ini_reader.GetBoolean(SECTION, "check_grad", false);
    default_config.one_response = ini_reader.GetBoolean(SECTION, "one_response", false);
    default_config.learn_test = ini_reader.GetBoolean(SECTION, "learn_test", false);
    default_config.save_model_per_batch = ini_reader.GetBoolean(SECTION, "save_model_per_batch",
            false);
    default_config.split_unknown_words = ini_reader.GetBoolean(SECTION, "split_unknown_words",
            true);

    default_config.max_sample_count = ini_reader.GetInteger(SECTION, "max_sample_count",
            1000000000);
    default_config.hold_batch_size = ini_reader.GetInteger(SECTION, "hold_batch_size", 100);
    default_config.dev_size = ini_reader.GetInteger(SECTION, "dev_size", 0);
    default_config.test_size = ini_reader.GetInteger(SECTION, "test_size", 0);
    default_config.device_id = ini_reader.GetInteger(SECTION, "device_id", 0);
    default_config.seed = ini_reader.GetInteger(SECTION, "seed", 0);
    default_config.cut_length = ini_reader.GetInteger(SECTION, "cut_length", 30);
    default_config.output_model_file_prefix = ini_reader.Get(SECTION, "output_model_file_prefix",
            "");
    default_config.input_model_file = ini_reader.Get(SECTION, "input_model_file", "");
    default_config.input_model_dir = ini_reader.Get(SECTION, "input_model_dir", "");
    default_config.black_list_file = ini_reader.Get(SECTION, "black_list_file", "");
    default_config.memory_in_gb = ini_reader.GetReal(SECTION, "memory_in_gb", 0.0f);
    default_config.ngram_penalty_1 = ini_reader.GetReal(SECTION, "ngram_penalty_1", 0.0f);
    default_config.ngram_penalty_2 = ini_reader.GetReal(SECTION, "ngram_penalty_2", 0.0f);
    default_config.ngram_penalty_3 = ini_reader.GetReal(SECTION, "ngram_penalty_3", 0.0f);

    return default_config;
}

HyperParams parseHyperParams(INIReader &ini_reader) {
    HyperParams hyper_params;

    int word_dim = ini_reader.GetInteger("hyper", "word_dim", 0);
    if (word_dim <= 0) {
        cerr << "word_dim wrong" << endl;
        abort();
    }
    hyper_params.word_dim = word_dim;

    int encoding_hidden_dim = ini_reader.GetInteger("hyper", "hidden_dim", 0);
    if (encoding_hidden_dim <= 0) {
        cerr << "hidden_dim wrong" << endl;
        abort();
    }
    hyper_params.hidden_dim = encoding_hidden_dim;

    float dropout = ini_reader.GetReal("hyper", "dropout", 0.0);
    if (dropout < -1.0f || dropout >=1.0f) {
        cerr << "dropout wrong" << endl;
        abort();
    }
    hyper_params.dropout = dropout;

    int batch_size = ini_reader.GetInteger("hyper", "batch_size", 0);
    if (batch_size == 0) {
        cerr << "batch_size not found" << endl;
        abort();
    }
    hyper_params.batch_size = batch_size;

    int beam_size = ini_reader.GetInteger("hyper", "beam_size", 0);
    if (beam_size == 0) {
        cerr << "beam_size not found" << endl;
        abort();
    }
    hyper_params.beam_size = beam_size;

    float learning_rate = ini_reader.GetReal("hyper", "learning_rate", 0.001f);
    if (learning_rate <= 0.0f) {
        cerr << "learning_rate wrong" << endl;
        abort();
    }
    hyper_params.learning_rate = learning_rate;

    float min_learning_rate = ini_reader.GetReal("hyper", "min_learning_rate", 0.0001f);
    if (min_learning_rate <= 0.0f) {
        cerr << "min_learning_rate wrong" << endl;
        abort();
    }
    hyper_params.min_learning_rate = min_learning_rate;

    float learning_rate_decay = ini_reader.GetReal("hyper", "learning_rate_decay", 0.9f);
    if (learning_rate_decay <= 0.0f || learning_rate_decay > 1.0f) {
        cerr << "decay wrong" << endl;
        abort();
    }
    hyper_params.learning_rate_decay = learning_rate_decay;

    int word_cutoff = ini_reader.GetReal("hyper", "word_cutoff", -1);
    if(word_cutoff == -1){
   	cerr << "word_cutoff read error" << endl;
    }
    hyper_params.word_cutoff = word_cutoff;

    bool word_finetune = ini_reader.GetBoolean("hyper", "word_finetune", -1);
    hyper_params.word_finetune = word_finetune;

    string word_file = ini_reader.Get("hyper", "word_file", "");
    hyper_params.word_file = word_file;

    float l2_reg = ini_reader.GetReal("hyper", "l2_reg", 0.0f);
    if (l2_reg < 0.0f || l2_reg > 1.0f) {
        cerr << "l2_reg:" << l2_reg << endl;
        abort();
    }
    hyper_params.l2_reg = l2_reg;
    string optimizer = ini_reader.Get("hyper", "optimzer", "");
    if (optimizer == "adam") {
        hyper_params.optimizer = Optimizer::ADAM;
    } else if (optimizer == "adagrad") {
        hyper_params.optimizer = Optimizer::ADAGRAD;
    } else if (optimizer == "adamw") {
        hyper_params.optimizer = Optimizer::ADAMW;
    } else {
        cerr << "invalid optimzer:" << optimizer << endl;
        abort();
    }

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, const LookupTable &lookup_table) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = lookup_table.getElemId(word);
        if (xid >= lookup_table.nVSize) {
            cerr << "xid:" << xid << " word:" << word << endl;
            for (const string &w :sentence) {
                cerr << w;
            }
            cerr << endl;
            abort();
        }
        ids.push_back(xid);
    }
    return ids;
}

void printWordIds(const vector<int> &word_ids, const LookupTable &lookup_table) {
    for (int word_id : word_ids) {
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
}

void printWordIdsWithKeywords(const vector<int> &word_ids, const LookupTable &lookup_table) {
    for (int i = 0; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
    for (int i = 1; i < word_ids.size(); i += 2) {
        int word_id = word_ids.at(i);
        cout << lookup_table.elems.from_id(word_id) << " ";
    }
    cout << endl;
}

void print(const vector<string> &words) {
    for (const string &w : words) {
        cout << w << " ";
    }
    cout << endl;
}

void analyze(const vector<int> &results, const vector<int> &answers, Metric &metric) {
    if (results.size() != answers.size()) {
        cerr << "results size is not equal to answers size" << endl;
        cerr << boost::format("results size:%1% answers size:%2%\n") % results.size() %
            answers.size();
        abort();
    }

    int size = results.size();
    for (int i = 0; i < size; ++i) {
        ++metric.overall_label_count;
        if (results.at(i) == answers.at(i)) {
            ++metric.correct_label_count;
        }
    }
}

string saveModel(const HyperParams &hyper_params, ModelParams &model_params,
        const string &filename_prefix, int epoch) {
    cout << "saving model file..." << endl;
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    string filename = filename_prefix + oss.str() + "-epoch" + to_string(epoch);
#if USE_GPU
    model_params.copyFromDeviceToHost();
#endif

    Json::Value root;
    root["hyper_params"] = hyper_params.toJson();
    root["model_params"] = model_params.toJson();
    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "";
    string json_str = Json::writeString(builder, root);
    ofstream out(filename);
    out << json_str;
    out.close();
    cout << format("model file %1% saved") % filename << endl;
    return filename;
}

shared_ptr<Json::Value> loadModel(const string &filename) {
    ifstream is(filename.c_str());
    shared_ptr<Json::Value> root(new Json::Value);
    if (is) {
        cout << "loading model..." << endl;
        stringstream sstr;
        sstr << is.rdbuf();
        string str = sstr.str();
        Json::CharReaderBuilder builder;
        auto reader = unique_ptr<Json::CharReader>(builder.newCharReader());
        string error;
        if (!reader->parse(str.c_str(), str.c_str() + str.size(), root.get(), &error)) {
            cerr << boost::format("parse json error:%1%") % error << endl;
            abort();
        }
        cout << "model loaded" << endl;
    } else {
        cerr << format("failed to open is, error when loading %1%") % filename << endl;
        abort();
    }

    return root;
}

void loadModel(const DefaultConfig &default_config, HyperParams &hyper_params,
        ModelParams &model_params,
        const Json::Value *root,
        const function<void(const DefaultConfig &default_config, const HyperParams &hyper_params,
            ModelParams &model_params, const Alphabet*)> &allocate_model_params) {
    hyper_params.fromJson((*root)["hyper_params"]);
    hyper_params.print();
    allocate_model_params(default_config, hyper_params, model_params, nullptr);
    model_params.fromJson((*root)["model_params"]);
#if USE_GPU
    model_params.copyFromHostToDevice();
#endif
}

pair<vector<Node *>, vector<int>> keywordNodesAndIds(const DecoderComponents &decoder_components,
        const vector<WordFrequencyInfo> &word_frequency_infos,
        int response_id,
        const ModelParams &model_params) {
    vector<Node *> keyword_result_nodes = toNodePointers<LinearWordVectorNode>(
            decoder_components.keyword_vector_to_onehots);
    vector<int> keyword_ids = toIds(
            word_frequency_infos.at(response_id).keywords_behind,
            model_params.lookup_table);
    vector<Node *> non_null_nodes;
    vector<int> chnanged_keyword_ids;
    for (int j = 0; j < keyword_result_nodes.size(); ++j) {
        if (keyword_result_nodes.at(j) != nullptr) {
            non_null_nodes.push_back(keyword_result_nodes.at(j));
            chnanged_keyword_ids.push_back(keyword_ids.at(j));
        }
    }

    return {non_null_nodes, chnanged_keyword_ids};
}


float metricTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        unordered_map<string, int> &word_counts,
        const vector<WordFrequencyInfo> &keyword_infos) {
    cout << "metricTestPosts begin" << endl;
    hyper_params.print();
    float rep_perplex(0.0f);
    thread_pool pool(16);
    mutex rep_perplex_mutex;

    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        auto f = [&]() {
            cout << "post:" << endl;
            print(post_sentences.at(post_and_responses.post_id));

            const vector<int> &response_ids = post_and_responses.response_ids;
            float avg_perplex = 0.0f;
            cout << "response size:" << response_ids.size() << endl;
            for (int response_id : response_ids) {
                //            cout << "response:" << endl;
                //            print(response_sentences.at(response_id));
                Graph graph;
                GraphBuilder graph_builder;
                graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                        hyper_params, model_params, false);
                DecoderComponents decoder_components;
                graph_builder.forwardDecoder(graph, decoder_components,
                        response_sentences.at(response_id),
                        keyword_infos.at(response_id).keywords_behind,
                        hyper_params, model_params, false);
                graph.compute();
                vector<Node*> nodes = toNodePointers(decoder_components.wordvector_to_onehots);
                vector<int> word_ids = transferVector<int, string>(
                        response_sentences.at(response_id), [&](const string &w) -> int {
                        return model_params.lookup_table.getElemId(w);
                        });
                int len = word_ids.size();
                auto keyword_nodes_and_ids = keywordNodesAndIds(decoder_components,
                        keyword_infos, response_id, model_params);
                for (int i = 0; i < keyword_nodes_and_ids.first.size(); ++i) {
                    nodes.push_back(keyword_nodes_and_ids.first.at(i));
                    word_ids.push_back(keyword_nodes_and_ids.second.at(i));
                }
                float perplex = computePerplex(nodes, word_ids, len);
                avg_perplex += perplex;
            }
            avg_perplex /= response_ids.size();
            cout << "avg_perplex:" << avg_perplex << endl;
            rep_perplex_mutex.lock();
            rep_perplex += avg_perplex;
            rep_perplex_mutex.unlock();
        };
        post(pool, f);
    }
    pool.join();

    cout << "total avg perplex:" << rep_perplex / post_and_responses_vector.size() << endl;
    return rep_perplex;
}

void decodeTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        DefaultConfig &default_config,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const vector<string> &black_list) {
    cout << "decodeTestPosts begin" << endl;
    hyper_params.print();
    vector<CandidateAndReferences> candidate_and_references_vector;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        cout << "post:" << endl;
        print(post_sentences.at(post_and_responses.post_id));
        Graph graph;
        GraphBuilder graph_builder;
        graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id), hyper_params,
                model_params, false);
        vector<DecoderComponents> decoder_components_vector;
        decoder_components_vector.resize(hyper_params.beam_size);
        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
                hyper_params.beam_size, hyper_params, model_params, default_config, black_list);
        const vector<WordIdAndProbability> &word_ids_and_probability = pair.first;
        cout << "post:" << endl;
        print(post_sentences.at(post_and_responses.post_id));
        cout << "response:" << endl;
        printWordIdsWithKeywords(word_ids_and_probability, model_params.lookup_table);
        dtype probability = pair.second;
        cout << format("probability:%1%") % probability << endl;
        if (word_ids_and_probability.empty()) {
            continue;
        }

        vector<int> decoded_word_ids = transferVector<int, WordIdAndProbability>(
                word_ids_and_probability, [](const WordIdAndProbability &w)->int {
            return w.word_id;
        });
        const vector<int> &response_ids = post_and_responses.response_ids;
        vector<vector<string>> str_references =
            transferVector<vector<string>, int>(response_ids,
                    [&](int response_id) -> vector<string> {
                    return response_sentences.at(response_id);
                    });
        vector<vector<int>> id_references;
        for (const vector<string> &strs : str_references) {
            vector<int> ids = transferVector<int, string>(strs,
                    [&](const string &w) -> int {
                    return model_params.lookup_table.getElemId(w);
                    });
            id_references.push_back(ids);
        }

        CandidateAndReferences candidate_and_references(decoded_word_ids, id_references);
        candidate_and_references_vector.push_back(candidate_and_references);

        float bleu_value = computeBleu(candidate_and_references_vector);
        cout << "bleu_value:" << bleu_value << endl;
    }
}

void interact(const DefaultConfig &default_config, const HyperParams &hyper_params,
        ModelParams &model_params,
        unordered_map<string, int> &word_counts,
        int word_cutoff,
        const vector<string> black_list) {
    hyper_params.print();
    while (true) {
        string post;
        getline(cin >> ws, post);
        utf8_string utf8_post(post);
        vector<string> words;
        for (int i = 0; i < utf8_post.length(); ++i) {
            words.push_back(utf8_post.substr(i, 1).cpp_str());
        }
        words.push_back(STOP_SYMBOL);

        if (default_config.split_unknown_words) {
            words = reprocessSentence(words, word_counts, word_cutoff);
        }

        Graph graph;
        GraphBuilder graph_builder;
        graph_builder.forward(graph, words, hyper_params, model_params, false);
        vector<DecoderComponents> decoder_components_vector;
        decoder_components_vector.resize(hyper_params.beam_size);
        cout << format("decodeTestPosts - beam_size:%1% decoder_components_vector.size:%2%") %
            hyper_params.beam_size % decoder_components_vector.size() << endl;
        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
                hyper_params.beam_size, hyper_params, model_params, default_config, black_list);
        const vector<WordIdAndProbability> &word_ids = pair.first;
        cout << "post:" << endl;
        cout << post << endl;
        cout << "response:" << endl;
        printWordIds(word_ids, model_params.lookup_table);
        dtype probability = pair.second;
        cout << format("probability:%1%") % probability << endl;
    }
}

pair<unordered_set<int>, unordered_set<int>> PostAndResponseIds(
        const vector<PostAndResponses> &post_and_responses_vector) {
    unordered_set<int> post_ids, response_ids;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        post_ids.insert(post_and_responses.post_id);
        for (int id : post_and_responses.response_ids) {
            response_ids.insert(id);
        }
    }
    return make_pair(post_ids, response_ids);
}

unordered_set<string> knownWords(const unordered_map<string, int> &word_counts, int word_cutoff) {
    unordered_set<string> word_set;
    for (auto it : word_counts) {
        if (it.second > word_cutoff) {
            word_set.insert(it.first);
        }
    }
    return word_set;
}

unordered_set<string> knownWords(const vector<string> &words) {
    unordered_set<string> word_set;
    for (const string& w : words) {
        word_set.insert(w);
    }
    return word_set;
}

int main(int argc, char *argv[]) {
    cout << "dtype size:" << sizeof(dtype) << endl;

    Options options("single-turn-conversation", "single turn conversation");
    options.add_options()
        ("config", "config file name", cxxopts::value<string>());
    auto args = options.parse(argc, argv);

    string configfilename = args["config"].as<string>();
    INIReader ini_reader(configfilename);
    if (ini_reader.ParseError() < 0) {
        cerr << "parse ini failed" << endl;
        abort();
    }

    DefaultConfig &default_config = GetDefaultConfig();
    default_config = parseDefaultConfig(ini_reader);
    cout << "default_config:" << endl;
    default_config.print();

#if USE_GPU
    n3ldg_cuda::InitCuda(default_config.device_id, default_config.memory_in_gb);
#endif

    HyperParams hyper_params = parseHyperParams(ini_reader);
    cout << "hyper_params:" << endl;
    hyper_params.print();

    vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(
            default_config.pair_file);
    cout << "post_and_responses_vector size:" << post_and_responses_vector.size() << endl;

    default_random_engine engine(default_config.seed);
    shuffle(begin(post_and_responses_vector), end(post_and_responses_vector),
            engine);
    vector<PostAndResponses> dev_post_and_responses, test_post_and_responses,
        train_post_and_responses;
    vector<ConversationPair> train_conversation_pairs;
    int i = 0;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        auto add_to_train = [&]() {
            if (default_config.program_mode == ProgramMode::TRAINING) {
                train_post_and_responses.push_back(post_and_responses);
                vector<ConversationPair> conversation_pairs =
                    toConversationPairs(post_and_responses);
                for (ConversationPair &conversation_pair : conversation_pairs) {
                    train_conversation_pairs.push_back(move(conversation_pair));
                }
            }
        };
        if (i < default_config.dev_size) {
            dev_post_and_responses.push_back(post_and_responses);
            if (default_config.learn_test) {
                add_to_train();
            }
        } else if (i < default_config.dev_size + default_config.test_size) {
            test_post_and_responses.push_back(post_and_responses);
            if (default_config.learn_test) {
                add_to_train();
            }
        } else {
            if (default_config.program_mode == ProgramMode::TRAINING) {
                add_to_train();
            }
        }
        ++i;
    }

    cout << "train size:" << train_conversation_pairs.size() << " dev size:" <<
        dev_post_and_responses.size() << " test size:" << test_post_and_responses.size() << endl;

    vector<vector<string>> post_sentences = readSentences(default_config.post_file);
    vector<vector<string>> response_sentences = readSentences(default_config.response_file);

    Alphabet alphabet;
    shared_ptr<Json::Value> root_ptr;
    unordered_map<string, int> word_counts;
    auto wordStat = [&]() {
        for (const ConversationPair &conversation_pair : train_conversation_pairs) {
            const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
            addWord(word_counts, post_sentence);

            const vector<string> &response_sentence = response_sentences.at(
                    conversation_pair.response_id);
            addWord(word_counts, response_sentence);
        }

        if (hyper_params.word_file != "" && !hyper_params.word_finetune) {
            for (const PostAndResponses &dev : dev_post_and_responses){
                const vector<string>&post_sentence = post_sentences.at(dev.post_id);
                addWord(word_counts, post_sentence);

                for(int i=0; i<dev.response_ids.size(); i++){
                    const vector<string>&resp_sentence = response_sentences.at(
                            dev.response_ids.at(i));
                    addWord(word_counts, resp_sentence);
                }
            }

            for (const PostAndResponses &test : test_post_and_responses){
                const vector<string>&post_sentence = post_sentences.at(test.post_id);
                addWord(word_counts, post_sentence);

                for(int i =0; i<test.response_ids.size(); i++){
                    const vector<string>&resp_sentence =
                        response_sentences.at(test.response_ids.at(i));
                    addWord(word_counts, resp_sentence);
                }
            }
        }
    };
    wordStat();

    word_counts[unknownkey] = 1000000000;
    alphabet.init(word_counts, hyper_params.word_cutoff);
    cout << boost::format("alphabet size:%1%") % alphabet.size() << endl;

    ModelParams model_params;
    int beam_size = hyper_params.beam_size;

    auto allocate_model_params = [](const DefaultConfig &default_config,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const Alphabet *alphabet) {
        cout << format("allocate word_file:%1%\n") % hyper_params.word_file;
        if (alphabet != nullptr) {
            if(hyper_params.word_file != "" &&
                    default_config.program_mode == ProgramMode::TRAINING &&
                    default_config.input_model_file == "") {
                model_params.lookup_table.init(*alphabet, hyper_params.word_file,
                        hyper_params.word_finetune);
            } else {
                model_params.lookup_table.init(*alphabet, hyper_params.word_dim, true);
            }
        }
        model_params.left_to_right_encoder_params.init(hyper_params.hidden_dim,
                hyper_params.word_dim + hyper_params.hidden_dim);
        model_params.hidden_to_wordvector_params.init(hyper_params.word_dim,
                2 * hyper_params.hidden_dim + 3 * hyper_params.word_dim, false);
        model_params.hidden_to_keyword_params.init(hyper_params.word_dim,
                2 * hyper_params.hidden_dim + 2 * hyper_params.word_dim, false);
        model_params.attention_parrams.init(hyper_params.hidden_dim, hyper_params.hidden_dim);
    };

    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            allocate_model_params(default_config, hyper_params, model_params, &alphabet);
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            word_counts = model_params.lookup_table.elems.m_string_to_id;
        }
    } else {
        if (default_config.input_model_file == "") {
            abort();
        } else {
            root_ptr = loadModel(default_config.input_model_file);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            word_counts = model_params.lookup_table.elems.m_string_to_id;
        }
    }

    auto word_frequency_infos = getWordFrequencyInfo(response_sentences, word_counts);
    auto black_list = readBlackList(default_config.black_list_file);

    if (default_config.program_mode == ProgramMode::INTERACTING) {
        hyper_params.beam_size = beam_size;
        interact(default_config, hyper_params, model_params, word_counts,
                hyper_params.word_cutoff, black_list);
    } else if (default_config.program_mode == ProgramMode::DECODING) {
        hyper_params.beam_size = beam_size;
        decodeTestPosts(hyper_params, model_params, default_config, test_post_and_responses,
                post_sentences, response_sentences, black_list);
    } else if (default_config.program_mode == ProgramMode::METRIC) {
        path dir_path(default_config.input_model_dir);
        if (!is_directory(dir_path)) {
            cerr << format("%1% is not dir path") % default_config.input_model_dir << endl;
            abort();
        }

        vector<string> ordered_file_paths;
        for(auto& entry : boost::make_iterator_range(directory_iterator(dir_path), {})) {
            string basic_name = entry.path().filename().string();
            cout << format("basic_name:%1%") % basic_name << endl;
            if (basic_name.find("model") != 0) {
                continue;
            }

            string model_file_path = entry.path().string();
            ordered_file_paths.push_back(model_file_path);
        }
        std::sort(ordered_file_paths.begin(), ordered_file_paths.end(),
                [](const string &a, const string &b)->bool {
                using boost::filesystem::last_write_time;
                return last_write_time(a) < last_write_time(b);
                });

        float max_rep_perplex = 0.0f;
        for(const string &model_file_path : ordered_file_paths) {
            cout << format("model_file_path:%1%") % model_file_path << endl;
            ModelParams model_params;
            shared_ptr<Json::Value> root_ptr = loadModel(model_file_path);
            loadModel(default_config, hyper_params, model_params, root_ptr.get(),
                    allocate_model_params);
            float rep_perplex = metricTestPosts(hyper_params, model_params, dev_post_and_responses,
                    post_sentences, response_sentences, word_counts, word_frequency_infos);
            cout << format("model %1% rep_perplex is %2%") % model_file_path % rep_perplex << endl;
            if (max_rep_perplex < rep_perplex) {
                max_rep_perplex = rep_perplex;
                cout << format("best model now is %1%, and rep_perplex is %2%") % model_file_path %
                    rep_perplex << endl;
            }
        }
    } else if (default_config.program_mode == ProgramMode::TRAINING) {
        ModelUpdate model_update;
        model_update._alpha = hyper_params.learning_rate;
        model_update._reg = hyper_params.l2_reg;
        exportToOptimizer(model_params, model_update);

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            exportToGradChecker(model_params, grad_checker);
        }

        dtype last_loss_sum = 1e10f;
        dtype loss_sum = 0.0f;

        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.SetEnabled(true);
        profiler.BeginEvent("total");

        int iteration = 0;
        string last_saved_model;

        for (int epoch = 0; ; ++epoch) {
            cout << "epoch:" << epoch << endl;
            auto cmp = [&] (const ConversationPair &a, const ConversationPair &b)->bool {
                auto len = [&] (const ConversationPair &pair)->int {
                    return post_sentences.at(pair.post_id).size() +
                        response_sentences.at(pair.response_id).size();
                };
                return len(a) < len(b);
            };
            sort(begin(train_conversation_pairs), end(train_conversation_pairs), cmp);
            int valid_len = train_conversation_pairs.size() / hyper_params.batch_size *
                hyper_params.batch_size;
            int batch_count = valid_len / hyper_params.batch_size;
            cout << boost::format("valid_len:%1% batch_count:%2%") % valid_len % batch_count <<
                endl;
            for (int i = 0; i < hyper_params.batch_size; ++i) {
                auto begin_pos = begin(train_conversation_pairs) + i * batch_count;
                shuffle(begin_pos, begin_pos + batch_count, engine);
            }

            unique_ptr<Metric> metric = unique_ptr<Metric>(new Metric);
            unique_ptr<Metric> keyword_metric = unique_ptr<Metric>(new Metric);
            for (int batch_i = 0; batch_i < batch_count; ++batch_i) {
                cout << format("batch_i:%1% iteration:%2%") % batch_i % iteration << endl;
                Graph graph;
                vector<shared_ptr<GraphBuilder>> graph_builders;
                vector<DecoderComponents> decoder_components_vector;
                vector<ConversationPair> conversation_pair_in_batch;
                auto getSentenceIndex = [batch_i, batch_count](int i) {
                    return i * batch_count + batch_i;
                };
                for (int i = 0; i < hyper_params.batch_size; ++i) {
                    shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                    graph_builders.push_back(graph_builder);
                    int instance_index = getSentenceIndex(i);
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                instance_index));
                    graph_builder->forward(graph, post_sentences.at(post_id), hyper_params,
                            model_params, true);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    DecoderComponents decoder_components;
                    graph_builder->forwardDecoder(graph, decoder_components,
                            response_sentences.at(response_id),
                            word_frequency_infos.at(response_id).keywords_behind,
                            hyper_params, model_params, true);
                    decoder_components_vector.push_back(decoder_components);
                }

                graph.compute();

                for (int i = 0; i < hyper_params.batch_size; ++i) {
                    int instance_index = getSentenceIndex(i);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    vector<int> word_ids = toIds(response_sentences.at(response_id),
                            model_params.lookup_table);
                    vector<Node*> result_nodes =
                        toNodePointers(decoder_components_vector.at(i).wordvector_to_onehots);
                    auto result = MaxLogProbabilityLoss(result_nodes, word_ids,
                            hyper_params.batch_size);
                    loss_sum += result.first;
                    analyze(result.second, word_ids, *metric);
                    auto keyword_nodes_and_ids = keywordNodesAndIds(
                            decoder_components_vector.at(i), word_frequency_infos, response_id,
                            model_params);
                    auto keyword_result = MaxLogProbabilityLoss(keyword_nodes_and_ids.first,
                        keyword_nodes_and_ids.second, hyper_params.batch_size);
                    loss_sum += keyword_result.first;
                    analyze(keyword_result.second, keyword_nodes_and_ids.second, *keyword_metric);

                    static int count_for_print;
                    if (++count_for_print % 100 == 0) {
                        count_for_print = 0;
                        int post_id = train_conversation_pairs.at(instance_index).post_id;
                        cout << "post:" << post_id << endl;
                        print(post_sentences.at(post_id));

                        cout << "golden answer:" << endl;
                        printWordIds(word_ids, model_params.lookup_table);
                        cout << "output:" << endl;
                        printWordIds(result.second, model_params.lookup_table);

                        cout << "golden keywords:" << endl;
                        printWordIds(keyword_nodes_and_ids.second, model_params.lookup_table);
                        cout << "output:" << endl;
                        printWordIds(keyword_result.second, model_params.lookup_table);
                    }
                }
                cout << "loss:" << loss_sum << endl;
                cout << "normal:" << endl;
                metric->print();
                cout << "keyword:" << endl;
                keyword_metric->print();

                graph.backward();

                if (default_config.check_grad) {
                    auto loss_function = [&](const ConversationPair &conversation_pair) -> dtype {
                        GraphBuilder graph_builder;
                        Graph graph;

                        graph_builder.forward(graph, post_sentences.at(conversation_pair.post_id),
                                hyper_params, model_params, true);

                        DecoderComponents decoder_components;
                        graph_builder.forwardDecoder(graph, decoder_components,
                                response_sentences.at(conversation_pair.response_id),
                                word_frequency_infos.at(
                                    conversation_pair.response_id).keywords_behind,
                                hyper_params, model_params, true);

                        graph.compute();

                        vector<int> word_ids = toIds(response_sentences.at(
                                    conversation_pair.response_id), model_params.lookup_table);
                        vector<Node*> result_nodes = toNodePointers(
                                decoder_components.wordvector_to_onehots);
                        return MaxLogProbabilityLoss(result_nodes, word_ids, 1).first;
                    };
                    cout << format("checking grad - conversation_pair size:%1%") %
                        conversation_pair_in_batch.size() << endl;
                    grad_checker.check<ConversationPair>(loss_function, conversation_pair_in_batch,
                            "");
                }

                if (hyper_params.optimizer == Optimizer::ADAM) {
                    model_update.updateAdam(10.0f);
                } else if (hyper_params.optimizer == Optimizer::ADAGRAD) {
                    model_update.update(10.0f);
                } else if (hyper_params.optimizer == Optimizer::ADAMW) {
                    model_update.updateAdamW(10.0f);
                } else {
                    cerr << "no optimzer set" << endl;
                    abort();
                }

                if (default_config.save_model_per_batch) {
                    saveModel(hyper_params, model_params, default_config.output_model_file_prefix,
                            epoch);
                }

                ++iteration;
            }

            cout << "loss_sum:" << loss_sum << " last_loss_sum:" << endl;
            if (loss_sum > last_loss_sum) {
                if (epoch == 0) {
                    cerr << "loss is larger than last epoch but epoch is 0" << endl;
                    abort();
                }
                model_update._alpha *= 0.1f;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate decay:" << model_update._alpha << endl;
                std::shared_ptr<Json::Value> root = loadModel(last_saved_model);
                model_params.fromJson((*root)["model_params"]);
#if USE_GPU
                model_params.copyFromHostToDevice();
#endif
            } else {
                model_update._alpha = (model_update._alpha - hyper_params.min_learning_rate) *
                    hyper_params.learning_rate_decay + hyper_params.min_learning_rate;
                hyper_params.learning_rate = model_update._alpha;
                cout << "learning_rate now:" << hyper_params.learning_rate << endl;
                last_saved_model = saveModel(hyper_params, model_params,
                        default_config.output_model_file_prefix, epoch);
            }

            last_loss_sum = loss_sum;
            loss_sum = 0;
        }
        profiler.EndCudaEvent();
        profiler.Print();
    } else {
        abort();
    }

    return 0;
}
