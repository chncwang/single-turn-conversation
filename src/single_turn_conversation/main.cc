#include "cxxopts.hpp"
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <random>
#include "INIReader.h"
#include <unordered_map>
#include <memory>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <boost/format.hpp>
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
#include "single_turn_conversation/encoder_decoder/decoder_components_builder.h"

using namespace std;
using namespace cxxopts;
using boost::is_any_of;
using boost::format;
using boost::filesystem::path;
using boost::filesystem::is_directory;
using boost::filesystem::directory_iterator;

void exportToOptimizer(ModelParams &model_params, ModelUpdate &model_update) {
    model_params.decoder_params.exportAdaParams(model_update);
    model_params.encoder_params.exportAdaParams(model_update);
    model_params.hidden_to_wordvector_params.exportAdaParams(model_update);
    model_params.lookup_table.exportAdaParams(model_update);
}

void exportToGradChecker(ModelParams &model_params, CheckGrad &grad_checker) {
    grad_checker.add(model_params.lookup_table.E, "lookup_table");
    grad_checker.add(model_params.hidden_to_wordvector_params.W, "hidden_to_wordvector_params W");
    grad_checker.add(model_params.hidden_to_wordvector_params.b, "hidden_to_wordvector_params b");
    grad_checker.add(model_params.encoder_params.cell_hidden.W, "encoder cell_hidden W");
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
    default_config.max_sample_count = ini_reader.GetInteger(SECTION, "max_sample_count",
            1000000000);
    default_config.dev_size = ini_reader.GetInteger(SECTION, "dev_size", 0);
    default_config.test_size = ini_reader.GetInteger(SECTION, "test_size", 0);
    default_config.device_id = ini_reader.GetInteger(SECTION, "device_id", 0);
    default_config.output_model_file_prefix = ini_reader.Get(SECTION, "output_model_file_prefix",
            "");
    default_config.input_model_file = ini_reader.Get(SECTION, "input_model_file", "");
    default_config.input_model_dir = ini_reader.Get(SECTION, "input_model_dir", "");

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

    int word_cutoff = ini_reader.GetReal("hyper", "word_cutoff", -1);
    if(word_cutoff == -1){
   	cerr << "word_cutoff read error" << endl;
    }
    hyper_params.word_cutoff = word_cutoff;

    bool word_finetune = ini_reader.GetBoolean("hyper", "word_finetune", -1);
    if(word_finetune == -1){
        cerr << "word_finetune read error" << endl;
    }
    hyper_params.word_finetune = word_finetune;

    string word_file = ini_reader.Get("hyper", "word_file", "");
    hyper_params.word_file = word_file;

    return hyper_params;
}

vector<int> toIds(const vector<string> &sentence, LookupTable &lookup_table) {
    vector<int> ids;
    for (const string &word : sentence) {
	int xid = lookup_table.getElemId(word, true);
	if(xid < 0 && lookup_table.getElemId(unknownkey, true) >=0 ){
	    xid = lookup_table.getElemId(unknownkey, true);
	}
        ids.push_back(xid);
    }
    return ids;
}

void printWordIds(const vector<int> &word_ids, const LookupTable &lookup_table) {
    for (int word_id : word_ids) {
        cout << lookup_table.elems->from_id(word_id) << " ";
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

void saveModel(const HyperParams &hyper_params, ModelParams &model_params,
        const string &filename_prefix, int epoch) {
    auto t = time(nullptr);
    auto tm = *localtime(&t);
    ostringstream oss;
    oss << put_time(&tm, "%d-%m-%Y-%H-%M-%S");
    string filename = filename_prefix + oss.str() + "epoch" + to_string(epoch);

    ofstream os(filename.c_str(), ios_base::out | ios::binary);
    if (os.is_open()) {
        hyper_params.save(os);
#if USE_GPU
        model_params.copyFromDeviceToHost();
#endif
        model_params.save(os);
    } else {
        cerr << format("failed to open os, error when saveing %1%") % filename << endl;
        abort();
    }
    cout << format("model file %1% saved") % filename << endl;

    os.close();
}

void loadModel(HyperParams &hyper_params, ModelParams &model_params, const string &filename) {
    ifstream is(filename.c_str());
    if (is) {
        hyper_params.load(is);
        model_params.load(is);
    } else {
        cerr << format("failed to open is, error when loading %1%") % filename << endl;
        abort();
    }
    cout << format("model file %1% loaded") % filename << endl;
}

float metricTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences) {
    cout << "metricTestPosts begin" << endl;
    hyper_params.print();
    float rep_perplex = 0.0f;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        cout << "post:" << endl;
        print(post_sentences.at(post_and_responses.post_id));

        const std::vector<int> &response_ids = post_and_responses.response_ids;
        float min_perplex = -1.0f;
        cout << "response size:" << response_ids.size() << endl;
        for (int response_id : response_ids) {
//            cout << "response:" << endl;
//            print(response_sentences.at(response_id));
            Graph graph;
            graph.train = false;
            GraphBuilder graph_builder;
            graph_builder.init(hyper_params);
            graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id),
                    hyper_params, model_params);
            std::shared_ptr<DecoderComponents> decoder_components(buildDecoderComponents());
            graph_builder.forwardDecoder(graph, *decoder_components,
                    response_sentences.at(response_id), hyper_params, model_params);
            graph.compute();
            vector<Node*> nodes = toNodePointers(decoder_components->wordvector_to_onehots);
            vector<int> word_ids = transferVector<int, string>(response_sentences.at(response_id),
                    [&](const string &w) -> int {
                    return model_params.lookup_table.getElemId(w);
                    });
            float perplex = computePerplex(nodes, word_ids);
//            cout << format("perplex:%1%") % perplex << endl;
            if (min_perplex < 0.0f || perplex < min_perplex) {
                min_perplex = perplex;
            }
        }
        cout << "min_perplex:" << min_perplex << endl;
        rep_perplex += 1.0f / min_perplex;
    }

    cout << "repciprocal perplex:" << rep_perplex << endl;
    return rep_perplex;
}

void decodeTestPosts(const HyperParams &hyper_params, ModelParams &model_params,
        const vector<PostAndResponses> &post_and_responses_vector,
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences) {
    cout << "decodeTestPosts begin" << endl;
    hyper_params.print();
    std::vector<CandidateAndReferences> candidate_and_references_vector;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        Graph graph;
        graph.train = false;
        GraphBuilder graph_builder;
        graph_builder.init(hyper_params);
        graph_builder.forward(graph, post_sentences.at(post_and_responses.post_id), hyper_params,
                model_params);
        std::vector<shared_ptr<DecoderComponents>> decoder_components_vector;
        for (int i = 0; i < hyper_params.beam_size; ++i) {
            decoder_components_vector.push_back(buildDecoderComponents());
        }
        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
                hyper_params.beam_size, hyper_params, model_params);
        const std::vector<WordIdAndProbability> &word_ids_and_probability = pair.first;
        cout << "post:" << endl;
        print(post_sentences.at(post_and_responses.post_id));
        cout << "response:" << endl;
        printWordIds(word_ids_and_probability, model_params.lookup_table);
        dtype probability = pair.second;
        cout << format("probability:%1%") % probability << endl;

        std::vector<int> decoded_word_ids = transferVector<int, WordIdAndProbability>(
                word_ids_and_probability, [](const WordIdAndProbability &w)->int {
            return w.word_id;
        });
        const std::vector<int> &response_ids = post_and_responses.response_ids;
        std::vector<std::vector<std::string>> str_references =
            transferVector<std::vector<std::string>, int>(response_ids,
                    [&](int response_id) -> std::vector<std::string> {
                    return response_sentences.at(response_id);
                    });
        std::vector<std::vector<int>> id_references;
        for (const std::vector<std::string> &strs : str_references) {
            std::vector<int> ids = transferVector<int, std::string>(strs,
                    [&](const std::string &w) -> int {
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

void interact(const HyperParams &hyper_params, ModelParams &model_params) {
    hyper_params.print();
    while (true) {
        string post;
        std::getline(cin >> ws, post);
        vector<string> words;
        split(words, post, is_any_of(" "));
        words.push_back(STOP_SYMBOL);

        Graph graph;
        graph.train = false;
        GraphBuilder graph_builder;
        graph_builder.init(hyper_params);
        graph_builder.forward(graph, words, hyper_params, model_params);
        std::vector<std::shared_ptr<DecoderComponents>> decoder_components_vector;
        decoder_components_vector.push_back(buildDecoderComponents());
        cout << format("decodeTestPosts - beam_size:%1% decoder_components_vector.size:%2%") %
            hyper_params.beam_size % decoder_components_vector.size() << endl;
        auto pair = graph_builder.forwardDecoderUsingBeamSearch(graph, decoder_components_vector,
                hyper_params.beam_size, hyper_params, model_params);
        const std::vector<WordIdAndProbability> &word_ids = pair.first;
        cout << "post:" << endl;
        cout << post << endl;
        cout << "response:" << endl;
        printWordIds(word_ids, model_params.lookup_table);
        dtype probability = pair.second;
        cout << format("probability:%1%") % probability << endl;
    }
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
    n3ldg_cuda::InitCuda(default_config.device_id);
#endif

    HyperParams hyper_params = parseHyperParams(ini_reader);
    cout << "hyper_params:" << endl;
    hyper_params.print();

    vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(
            default_config.pair_file);
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
        auto add_to_train = [&]() {
            train_post_and_responses.push_back(post_and_responses);
            vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
            for (ConversationPair &conversation_pair : conversation_pairs) {
                train_conversation_pairs.push_back(std::move(conversation_pair));
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
            add_to_train();
        }
        ++i;
    }

    cout << "train size:" << train_conversation_pairs.size() << " dev size:" <<
        dev_post_and_responses.size() << " test size:" << test_post_and_responses.size() << endl;

    vector<vector<string>> post_sentences = readSentences(default_config.post_file);

    vector<vector<string>> response_sentences = readSentences(default_config.response_file);

    unordered_map<string, int> word_counts;
    for (const ConversationPair &conversation_pair : train_conversation_pairs) {
        const vector<string> &post_sentence = post_sentences.at(conversation_pair.post_id);
        addWord(word_counts, post_sentence);

        const vector<string> &response_sentence = response_sentences.at(
                conversation_pair.response_id);
        addWord(word_counts, response_sentence);
    }
    
    for (const PostAndResponses &dev : dev_post_and_responses){
    	const vector<string>&post_sentence = post_sentences.at(dev.post_id);
	addWord(word_counts, post_sentence);

	for(int i=0; i<dev.response_ids.size(); i++){
	    const vector<string>&resp_sentence = response_sentences.at(dev.response_ids.at(i));
	    addWord(word_counts, resp_sentence);
	}
    }
    
    for (const PostAndResponses &test : test_post_and_responses){
        const vector<string>&post_sentence = post_sentences.at(test.post_id);
	addWord(word_counts, post_sentence);

	for(int i =0; i<test.response_ids.size(); i++){
	    const vector<string>&resp_sentence = response_sentences.at(test.response_ids.at(i));
	    addWord(word_counts, resp_sentence);
	}
    } 

    word_counts[unknownkey] = hyper_params.word_cutoff + 1;
    word_counts[STOP_SYMBOL] = 1000000;
    Alphabet alphabet;
    alphabet.initial(word_counts, hyper_params.word_cutoff);
    cout << "the size of alphabet is: ";
    cout << alphabet.size() <<endl;
    ModelParams model_params;

    int beam_size = hyper_params.beam_size;

    if (default_config.program_mode != ProgramMode::METRIC) {
        if (default_config.input_model_file == "") {
            if(hyper_params.word_file == "") model_params.lookup_table.initial(&alphabet, hyper_params.word_dim, true);
            else model_params.lookup_table.initial(&alphabet, hyper_params.word_file, hyper_params.word_finetune);
	    model_params.encoder_params.initial(hyper_params.hidden_dim, hyper_params.word_dim);
            model_params.decoder_params.initial(hyper_params.hidden_dim, hyper_params.word_dim);
            model_params.hidden_to_wordvector_params.initial(hyper_params.word_dim,
                    hyper_params.hidden_dim);
        } else {
            model_params.lookup_table.elems = &alphabet;
            loadModel(hyper_params, model_params, default_config.input_model_file);
        }
    }

    if (default_config.program_mode == ProgramMode::INTERACTING) {
        hyper_params.beam_size = beam_size;
        interact(hyper_params, model_params);
    } else if (default_config.program_mode == ProgramMode::DECODING) {
        hyper_params.beam_size = beam_size;
        decodeTestPosts(hyper_params, model_params, test_post_and_responses, post_sentences,
                response_sentences);
    } else if (default_config.program_mode == ProgramMode::METRIC) {
        path dir_path(default_config.input_model_dir);
        if (!is_directory(dir_path)) {
            cerr << format("%1% is not dir path") % default_config.input_model_dir << endl;
            abort();
        }

        float max_rep_perplex = 0.0f;
        for(auto& entry : boost::make_iterator_range(directory_iterator(dir_path), {})) {
            string basic_name = entry.path().filename().string();
            if (basic_name.find("model") != 0) {
                continue;
            }

            string model_file_path = entry.path().string();
            cout << format("model_file_path:%1%") % model_file_path << endl;
            ModelParams model_params;
            model_params.lookup_table.elems = &alphabet;
            loadModel(hyper_params, model_params, model_file_path);
            float rep_perplex = metricTestPosts(hyper_params, model_params,
                    test_post_and_responses, post_sentences, response_sentences);
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
        exportToOptimizer(model_params, model_update);

        CheckGrad grad_checker;
        if (default_config.check_grad) {
            exportToGradChecker(model_params, grad_checker);
        }

        dtype last_loss_sum = 1e10f;

        n3ldg_cuda::Profiler &profiler = n3ldg_cuda::Profiler::Ins();
        profiler.SetEnabled(false);
        profiler.BeginEvent("total");
        int interval = 600;
        for (int epoch = 0; ; ++epoch) {
            cout << "epoch:" << epoch << endl;
            shuffle(begin(train_conversation_pairs), end(train_conversation_pairs), engine);
            unique_ptr<Metric> metric = unique_ptr<Metric>(new Metric);
            dtype loss_sum = 0.0f;

            auto last_timestamp = chrono::steady_clock::now();

            for (int batch_i = 0; batch_i < train_conversation_pairs.size() /
                    hyper_params.batch_size; ++batch_i) {
                cout << "batch_i:" << batch_i << endl;
                Graph graph;
                graph.train = true;
                vector<shared_ptr<GraphBuilder>> graph_builders;
                vector<shared_ptr<DecoderComponents>> decoder_components_vector;
                vector<ConversationPair> conversation_pair_in_batch;
                for (int i = 0; i < hyper_params.batch_size; ++i) {
                    shared_ptr<GraphBuilder> graph_builder(new GraphBuilder);
                    graph_builders.push_back(graph_builder);
                    graph_builder->init(hyper_params);
                    int instance_index = batch_i * hyper_params.batch_size + i;
                    int post_id = train_conversation_pairs.at(instance_index).post_id;
                    conversation_pair_in_batch.push_back(train_conversation_pairs.at(
                                instance_index));
                    graph_builder->forward(graph, post_sentences.at(post_id), hyper_params,
                            model_params);
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    shared_ptr<DecoderComponents> decoder_components(buildDecoderComponents());
                    decoder_components_vector.push_back(decoder_components);
                    graph_builder->forwardDecoder(graph, *decoder_components,
                            response_sentences.at(response_id),
                            hyper_params, model_params);
                }

                graph.compute();

                for (int i = 0; i < hyper_params.batch_size; ++i) {
                    int instance_index = batch_i * hyper_params.batch_size + i;
                    int response_id = train_conversation_pairs.at(instance_index).response_id;
                    vector<int> word_ids = toIds(response_sentences.at(response_id),
                            model_params.lookup_table);
                    vector<Node*> result_nodes =
                        toNodePointers(decoder_components_vector.at(i)->wordvector_to_onehots);
#if USE_GPU
                    vector<const dtype *> vals;
                    vector<dtype*> losses;
                    for (const Node *node : result_nodes) {
                        vals.push_back(node->val.value);
                        losses.push_back(node->loss.value);
                    }
                    auto result = n3ldg_cuda::SoftMaxLoss(vals, vals.size(),
                            result_nodes.at(0)->dim, word_ids, hyper_params.batch_size, losses);
#if TEST_CUDA
                    auto cpu_result = MaxLogProbabilityLoss(result_nodes, word_ids,
                            hyper_params.batch_size);
                    cout << format("result loss:%1% cpu_result loss:%2%") % result.first %
                        cpu_result.first << endl;
                    if (abs(result.first - cpu_result.first) > 0.001) {
                        abort();
                    }

                    for (const Node *node : result_nodes) {
                        n3ldg_cuda::Assert(node->loss.verify("cross entropy loss"));
                    }
#endif
#else
                    auto result = MaxLogProbabilityLoss(result_nodes, word_ids,
                            hyper_params.batch_size);
#endif
                    loss_sum += result.first;

                    analyze(result.second, word_ids, *metric);
                    unique_ptr<Metric> local_metric(unique_ptr<Metric>(new Metric));
                    analyze(result.second, word_ids, *local_metric);

                    if (local_metric->getAccuracy() < 1.0f) {
                        static int count_for_print;
                        if (++count_for_print % 100 == 0) {
                            count_for_print = 0;
                            int post_id = train_conversation_pairs.at(instance_index).post_id;
                            cout << "post:" << endl;
                            print(post_sentences.at(post_id));
                            cout << "golden answer:" << endl;
                            printWordIds(word_ids, model_params.lookup_table);
                            cout << "output:" << endl;
                            printWordIds(result.second, model_params.lookup_table);
                        }
                    }
                }
                cout << "loss:" << loss_sum << endl;
                metric->print();

                graph.backward();

                if (default_config.check_grad) {
                    auto loss_function = [&](const ConversationPair &conversation_pair) -> dtype {
                        GraphBuilder graph_builder;
                        graph_builder.init(hyper_params);
                        Graph graph;
                        graph.train = true;

                        graph_builder.forward(graph, post_sentences.at(conversation_pair.post_id),
                                hyper_params, model_params);

                        std::shared_ptr<DecoderComponents> decoder_components(
                                buildDecoderComponents());
                        graph_builder.forwardDecoder(graph, *decoder_components,
                                response_sentences.at(conversation_pair.response_id),
                                hyper_params, model_params);

                        graph.compute();

                        vector<int> word_ids = toIds(response_sentences.at(
                                    conversation_pair.response_id), model_params.lookup_table);
                        vector<Node*> result_nodes = toNodePointers(
                                decoder_components->wordvector_to_onehots);
                        return MaxLogProbabilityLoss(result_nodes, word_ids, 1).first;
                    };
                    cout << format("checking grad - conversation_pair size:%1%") %
                        conversation_pair_in_batch.size() << endl;
                    grad_checker.check<ConversationPair>(loss_function, conversation_pair_in_batch,
                            "");
                }

                model_update.updateAdam(10.0f);
                
               // auto current_timestamp = chrono::steady_clock::now();
                //if (chrono::duration_cast<chrono::seconds>(current_timestamp -
                  //          last_timestamp).count() >= interval) {
                   // interval *= 2;
                   // if (interval > 3600) {
                    //    interval = 3600;
                   // }
                   // last_timestamp = current_timestamp;
                   // saveModel(hyper_params, model_params, default_config.output_model_file_prefix, epoch);
               // }
            }
	    saveModel(hyper_params, model_params, default_config.output_model_file_prefix, epoch);
            profiler.EndCudaEvent();
            profiler.Print();

            if (last_loss_sum < loss_sum) {
                model_update._alpha *= 0.875;
                cout << "learning_rate:" << model_update._alpha << endl;
            }

            last_loss_sum = loss_sum;
        }
    } else {
        abort();
    }

    return 0;
}
