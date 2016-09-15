// Author: Karl Stratos (jlee2071@bloomberg.net)

#include "bilstm_tagger.h"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "cnn/training.h"

void BiLSTMTagger::SetModelDirectory(const string &model_directory) {
    model_directory_ = model_directory;

    // Remove a file at the path (if it exists).
    if (util_file::exists(model_directory_) &&
        util_file::get_file_type(model_directory_) == "file") {
        ASSERT(system(("rm -f " + model_directory_).c_str()) == 0,
               "Cannot remove file: " << model_directory_);
    }

    // Create the current output directory (if necessary).
    ASSERT(system(("mkdir -p " + model_directory_).c_str()) == 0,
           "Cannot create directory: " << model_directory_);
}

void BiLSTMTagger::ResetModelDirectory() {
    ASSERT(!model_directory_.empty(), "No model directory given.");
    ASSERT(system(("rm -f " + model_directory_ + "/*").c_str()) == 0,
           "Cannot remove the content in: " << model_directory_);
    SetModelDirectory(model_directory_);
}

void BiLSTMTagger::Train(const string &data_path) {
    is_training_ = true;
    ofstream log_file(model_directory_ + "/log", ios::app);

    // Read training data.
    vector<pair<vector<size_t>, vector<size_t> > > training_data;
    vector<vector<string> > sentences;
    size_t num_words_training =
        ReadTrainingData(data_path, &training_data, &sentences);

    // Read dev data.
    ASSERT(!development_path_.empty(), "Requires dev data for training!");
    pair<vector<vector<string> >, vector<vector<string> > > dev_data;
    size_t num_words_dev = ReadTestData(development_path_, &dev_data);

    cerr << "Model directory:        " << model_directory_ << endl;
    cerr << "Training data:          " << data_path << endl;
    cerr << "   Num sentences:       " << training_data.size() << endl;
    cerr << "   Num words:           " << num_words_training << endl;
    cerr << "   Num char types:      " << char_index_.size() << endl;
    cerr << "   Num word types:      " << word_index_.size() << endl;
    cerr << "   Num tag types:       " << tag_index_.size() << endl;
    cerr << endl;
    cerr << "Dev data:               " << development_path_ << endl;
    cerr << "   Num sentences:       " << dev_data.first.size() << endl;
    cerr << "   Num words:           " << num_words_dev << endl;
    cerr << endl;

    log_file << "Model directory:        " << model_directory_ << endl;
    log_file << "Training data:          " << data_path << endl;
    log_file << "   Num sentences:       " << training_data.size() << endl;
    log_file << "   Num words:           " << num_words_training << endl;
    log_file << "   Num char types:      " << char_index_.size() << endl;
    log_file << "   Num word types:      " << word_index_.size() << endl;
    log_file << "   Num tag types:       " << tag_index_.size() << endl;
    log_file << endl;
    log_file << "Dev data:               " << development_path_ << endl;
    log_file << "   Num sentences:       " << dev_data.first.size() << endl;
    log_file << "   Num words:           " << num_words_dev << endl;
    log_file << endl;

    // Read word embeddings, augment word dictionary.
    unordered_map<string, vector<double> > embeddings;
    ReadEmbeddings(&embeddings);
    size_t dim_embeddings = 0;
    if (embeddings.size() > 0) {
        dim_embeddings = embeddings.begin()->second.size();
        dim_word_embeddings_ = dim_embeddings;  // Reset word dim.
        for (auto &map_pair : embeddings) {
            string word = map_pair.first;
            if (word_index_.find(word) == word_index_.end()) {
                size_t new_x = word_index_.size();
                word_index_[word] = new_x;
                word_string_.push_back(word);
            }
        }
    }

    cerr << "Word embeddings:        " << embedding_path_ << endl;
    cerr << "   Dim:                 " << dim_embeddings << endl;
    cerr << "   This many:           " << embeddings.size() << endl;
    cerr << "   Num word types now:  " << word_index_.size() << endl;
    cerr << endl;
    cerr << "Dim word embeddings:    " << dim_word_embeddings_ << endl;
    cerr << "Dim char embeddings:    " << dim_char_embeddings_  << endl;
    cerr << "Dim LSTM outputs:       " << dim_lstm_outputs_ << endl;
    cerr << "Dim hidden layers:      " << dim_hidden_layers_ << endl;
    cerr << "Num LSTM layers:        " << num_lstm_layers_ << endl;
    cerr << "Activation:             " << activation_ << endl;
    cerr << "Learning rate:          " << learning_rate_ << endl;
    cerr << "Dropout rate:           " << dropout_rate_ << endl;
    cerr << "Number of epochs:       " << num_epochs_ << endl;
    cerr << "Performance measure:    " << performance_measure_ << endl;
    cerr << "Loss function:          " << loss_function_ << endl;
    cerr << "Beam size:              " << beam_size_ << endl;
    cerr << endl;

    log_file << "Word embeddings:        " << embedding_path_ << endl;
    log_file << "   Dim:                 " << dim_embeddings << endl;
    log_file << "   This many:           " << embeddings.size() << endl;
    log_file << "   Num word types now:  " << word_index_.size() << endl;
    log_file << endl;
    log_file << "Dim word embeddings:    " << dim_word_embeddings_ << endl;
    log_file << "Dim char embeddings:    " << dim_char_embeddings_  << endl;
    log_file << "Dim LSTM outputs:       " << dim_lstm_outputs_ << endl;
    log_file << "Dim hidden layers:      " << dim_hidden_layers_ << endl;
    log_file << "Num LSTM layers:        " << num_lstm_layers_ << endl;
    log_file << "Activation:             " << activation_ << endl;
    log_file << "Learning rate:          " << learning_rate_ << endl;
    log_file << "Dropout rate:           " << dropout_rate_ << endl;
    log_file << "Number of epochs:       " << num_epochs_ << endl;
    log_file << "Performance measure:    " << performance_measure_ << endl;
    log_file << "Loss function:          " << loss_function_ << endl;
    log_file << "Beam size:              " << beam_size_ << endl;
    log_file << endl;

    // Build model, initialize word parameters.
    BuildModel();
    for (auto &map_pair : embeddings) {
        size_t x = word_index_[map_pair.first];
        vector<float> x_vector;
        for (double value : map_pair.second) {  // Need in floats.
            x_vector.push_back((float) value);
        }
        word_embeddings_->Initialize(x, x_vector);
    }
    for (auto &map_pair : word_index_) {
        string word = map_pair.first;
        string word_lower = util_string::lowercase(word);
        if (embeddings.find(word) == embeddings.end() &&
            embeddings.find(word_lower) != embeddings.end()) {
            // Ex. If "Bloomberg" is not in embeddings but "bloomberg" is,
            // give "bloomberg" embedding to "Bloomberg".
            size_t x = word_index_[word];
            vector<float> x_vector;
            for (double value : embeddings[word_lower]) {  // Need in floats.
                x_vector.push_back((float) value);
            }
            word_embeddings_->Initialize(x, x_vector);
        }
    }

    cnn::AdamTrainer trainer(&model_, 1e-6, learning_rate_, 0.01, 0.9999, 1e-8);

    vector<size_t> permutation(training_data.size());
    for (size_t i = 0; i < permutation.size(); ++i) { permutation[i] = i; }

    double loss_accumulated = 0.0;
    size_t num_predictions_made = 0;
    size_t num_sentences_processed = 0;
    double max_performance =  -numeric_limits<float>::infinity();
    for (size_t epoch_num = 0; epoch_num < num_epochs_; ++epoch_num) {
        cerr << "Epoch " << epoch_num + 1 << endl;
        log_file << "Epoch " << epoch_num + 1 << endl;
        shuffle(permutation.begin(), permutation.end(), *cnn::rndeng);
        for (size_t i_rand : permutation) {
            // Build graph for this tagged sentence.
            vector<size_t> predictions;  // Not used.
            loss_accumulated += BuildGraph(training_data[i_rand].first,
                                           sentences[i_rand],
                                           training_data[i_rand].second,
                                           &predictions);
            trainer.update(1.0);

            num_predictions_made += training_data[i_rand].first.size();
            ++num_sentences_processed;
            if (num_sentences_processed % development_interval_ == 0) {
                cerr << "Processed " << development_interval_ << " sents";
                log_file << "Processed " << development_interval_ << " sents";

                // Report average loss.
                double average_loss = loss_accumulated / num_predictions_made;
                cerr << ",\taverage loss: " << average_loss;
                log_file << ",\taverage loss: " << average_loss;
                loss_accumulated = 0.0;
                num_predictions_made = 0;
                num_sentences_processed = 0;

                // Dev evaluation
                is_training_ = false;
                vector<vector<string> > predicted_tag_sequences;
                Predict(dev_data.first, &predicted_tag_sequences);
                is_training_ = true;

                double performance = EvaluatePerformance(
                    dev_data.second, predicted_tag_sequences);
                cerr << ",  \tdev performance: " << performance;
                log_file << ",  \tdev performance: " << performance;
                if (performance > max_performance) {
                    max_performance = performance;
                    cerr << "  \tsaving model...";
                    log_file << "  \tsaving model...";
                    SaveModel();
                }
                cerr << endl;
                log_file << endl;
            }
        }
    }
}

size_t BiLSTMTagger::ReadTrainingData(
    const string &data_path,
    vector<pair<vector<size_t>, vector<size_t> > > *training_data,
    vector<vector<string> > *sentences) {
    training_data->clear();
    sentences->clear();
    char_index_.clear();
    char_index_[kUnknownChar_] = kUnknownCharIndex_;
    char_count_.clear();
    char_string_.clear();
    word_index_.clear();
    word_index_[kUnknownWord_] = kUnknownWordIndex_;
    word_count_.clear();
    word_string_.clear();
    tag_index_.clear();
    tag_string_.clear();
    unordered_map<size_t, size_t> char_count_hash;
    char_count_hash[kUnknownCharIndex_] = 0;
    unordered_map<size_t, size_t> word_count_hash;
    word_count_hash[kUnknownWordIndex_] = 0;
    unordered_map<size_t, char> char_string_hash;
    char_string_hash[kUnknownCharIndex_] = kUnknownChar_;
    unordered_map<size_t, string> word_string_hash;
    word_string_hash[kUnknownWordIndex_] = kUnknownWord_;
    unordered_map<size_t, string> tag_string_hash;
    vector<size_t> x_sequence;
    vector<size_t> t_sequence;
    vector<string> sentence;
    size_t num_words = 0;

    ifstream file(data_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << data_path);
    while (file.good()) {
	vector<string> tokens;
	util_file::read_line(&file, &tokens);
	if (tokens.size() > 0) {
            ASSERT(tokens.size() == 2, "Each line: <word> <tag>");
            ++num_words;
            string word = tokens[0];
            string tag = tokens[1];
            for (char character : word) {
                if (char_index_.find(character) == char_index_.end()) {
                    size_t new_c = char_index_.size();
                    char_index_[character] = new_c;
                }
                size_t c = char_index_[character];
                if (char_count_hash.find(c) == char_count_hash.end()) {
                    char_count_hash[c] = 0;
                }
                ++char_count_hash[c];
                char_string_hash[c] = character;
            }
            if (word_index_.find(word) == word_index_.end()) {
                size_t new_x = word_index_.size();
                word_index_[word] = new_x;
            }
            if (tag_index_.find(tag) == tag_index_.end()) {
                size_t new_t = tag_index_.size();
                tag_index_[tag] = new_t;
            }
            size_t x = word_index_[word];
            size_t t = tag_index_[tag];
            x_sequence.push_back(x);
            t_sequence.push_back(t);
            sentence.push_back(word);
            if (word_count_hash.find(x) == word_count_hash.end()) {
                word_count_hash[x] = 0;
            }
            ++word_count_hash[x];
            word_string_hash[x] = word;
            tag_string_hash[t] = tag;
        } else {
            if (x_sequence.size() > 0) {
                training_data->push_back(make_pair(x_sequence, t_sequence));
                sentences->push_back(sentence);
                x_sequence.clear();
                t_sequence.clear();
                sentence.clear();
            }
        }
    }
    if (x_sequence.size() > 0) {
        training_data->push_back(make_pair(x_sequence, t_sequence));
        sentences->push_back(sentence);
    }

    char_count_.resize(char_count_hash.size());
    char_count_[kUnknownCharIndex_] = 0;
    word_count_.resize(word_count_hash.size());
    word_count_[kUnknownWordIndex_] = 0;
    char_string_.resize(char_string_hash.size());
    char_string_[kUnknownCharIndex_] = kUnknownChar_;
    word_string_.resize(word_string_hash.size());
    word_string_[kUnknownWordIndex_] = kUnknownWord_;
    tag_string_.resize(tag_string_hash.size());

    for (size_t c = 0; c < char_count_hash.size(); ++c) {
        char_count_[c] = char_count_hash[c];
        char_string_[c] = char_string_hash[c];
    }
    for (size_t x = 0; x < word_count_hash.size(); ++x) {
        word_count_[x] = word_count_hash[x];
        word_string_[x] = word_string_hash[x];
    }
    for (size_t t = 0; t < tag_string_hash.size(); ++t) {
        tag_string_[t] = tag_string_hash[t];
    }

    return num_words;
}

size_t BiLSTMTagger::ReadTestData(
    const string &data_path,
    pair<vector<vector<string> >, vector<vector<string> > > *test_data) {
    vector<vector<string> > word_sequences;
    vector<vector<string> > tag_sequences;
    vector<string> word_sequence;
    vector<string> tag_sequence;
    size_t num_words = 0;

    ifstream file(data_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << data_path);
    while (file.good()) {
	vector<string> tokens;
	util_file::read_line(&file, &tokens);
	if (tokens.size() > 0) {
            ASSERT(tokens.size() == 2, "Each line: <word> <tag>");
            ++num_words;
            string word = tokens[0];
            string tag = tokens[1];
            word_sequence.push_back(word);
            tag_sequence.push_back(tag);
        } else {
            if (word_sequence.size() > 0) {
                word_sequences.push_back(word_sequence);
                tag_sequences.push_back(tag_sequence);
                word_sequence.clear();
                tag_sequence.clear();
            }
        }
    }
    if (word_sequence.size() > 0) {
        word_sequences.push_back(word_sequence);
        tag_sequences.push_back(tag_sequence);
    }
    *test_data = make_pair(word_sequences, tag_sequences);

    return num_words;
}

void BiLSTMTagger::ReadEmbeddings(
    unordered_map<string, vector<double> > *embeddings) {
    if (embedding_path_.empty()) { return; }
    embeddings->clear();
    ifstream file(embedding_path_, ios::in);
    ASSERT(file.is_open(), "Cannot open " << embedding_path_);
    while (file.good()) {
	vector<string> tokens;
	util_file::read_line(&file, &tokens);
	if (tokens.size() > 0) {
            string word = tokens[1];  // <count> <word> <v_1> ... <v_d>
            vector<double> v;
            for (size_t i = 2; i < tokens.size(); ++i) {
                v.push_back(stod(tokens[i]));
            }
            (*embeddings)[word] = v;
        }
    }
}

void BiLSTMTagger::BuildModel() {
    // Shorthands for readability...
    unsigned int num_c = static_cast<unsigned int>(char_index_.size());
    unsigned int num_x = static_cast<unsigned int>(word_index_.size());
    unsigned int num_t = static_cast<unsigned int>(tag_index_.size());
    unsigned int cdim = static_cast<unsigned int>(dim_char_embeddings_);
    unsigned int xdim = static_cast<unsigned int>(dim_word_embeddings_);
    unsigned int ldim = static_cast<unsigned int>(dim_lstm_outputs_);
    unsigned int hdim = static_cast<unsigned int>(dim_hidden_layers_);
    unsigned int nlayer = static_cast<unsigned int>(num_lstm_layers_);

    // Input embeddings
    char_embeddings_ = model_.add_lookup_parameters(num_c, {cdim});
    word_embeddings_ = model_.add_lookup_parameters(num_x, {xdim});

    // Character-level LSTMs
    char_lstm_builder_forward_ = cnn::LSTMBuilder(nlayer, cdim, cdim, &model_);
    char_lstm_builder_backward_ = cnn::LSTMBuilder(nlayer, cdim, cdim, &model_);

    // Word-level LSTMs
    unsigned int d0 = 2 * cdim + xdim;
    word_lstm_builder_forward_ = cnn::LSTMBuilder(nlayer, d0, ldim, &model_);
    word_lstm_builder_backward_ = cnn::LSTMBuilder(nlayer, d0, ldim, &model_);

    // Output layer feed-forward
    unsigned int d1 = 2 * ldim;
    weight_matrix1_ = model_.add_parameters({hdim, d1});
    bias1_ = model_.add_parameters({hdim});
    weight_matrix2_ = model_.add_parameters({num_t, hdim});
    bias2_ = model_.add_parameters({num_t});

    if (loss_function_ != "local") {
        // Tag scoring
        tag_prior_ = model_.add_parameters({num_t});
        tag_transition_ = model_.add_parameters({num_t, num_t});
        tag_end_ = model_.add_parameters({num_t});
    }
}

double BiLSTMTagger::BuildGraph(const vector<size_t> &x_sequence,
                                const vector<string> &sentence,
                                const vector<size_t> &t_sequence,
                                vector<size_t> *predictions) {
    predictions->clear();
    cnn::ComputationGraph graph;

    vector<Expression> word_expressions;
    ComputeWordExpressions(x_sequence, sentence, &graph, &word_expressions);

    Expression W1 = parameter(graph, weight_matrix1_);
    Expression b1 = parameter(graph, bias1_);
    Expression W2 = parameter(graph, weight_matrix2_);
    Expression b2 = parameter(graph, bias2_);

    vector<Expression> tag_weights;

    for (size_t i = 0; i < word_expressions.size(); ++i) {
        Expression z1 = W1 * word_expressions[i] + b1;
        Expression h1;
        Activate(z1, &h1);
        tag_weights.push_back(W2 * h1 + b2);
    }

    if (loss_function_ == "local") {
        return BuildLocalLossGraph(tag_weights, t_sequence, &graph,
                                   predictions);
    } else if (loss_function_ == "global") {
        return BuildGlobalLossGraph(tag_weights, t_sequence, &graph,
                                    predictions);
    } else if (loss_function_ == "beam") {
        return BuildGlobalLossGraphBeam(tag_weights, t_sequence, &graph,
                                        predictions);
    } else {
        ASSERT(false, "Unknown loss function: " << loss_function_);
    }
}

double BiLSTMTagger::BuildLocalLossGraph(const vector<Expression> &tag_weights,
                                         const vector<size_t> &t_sequence,
                                         cnn::ComputationGraph *graph,
                                         vector<size_t> *predictions) {
    vector<Expression> losses;  // Negative log probabilities of gold tags

    for (size_t i = 0; i < tag_weights.size(); ++i) {
        if (is_training_) {
            // loss = - weight_{t_i} + log SUM_{t} exp(weight_{t})
            Expression loss = pick(-log(softmax(tag_weights.at(i))),
                                   t_sequence.at(i));
            losses.push_back(loss);
        } else {  // Test time: compute predictions
            vector<float> scores =
                cnn::as_vector(graph->get_value(tag_weights.at(i)));
            size_t best_t = 0;
            float best_score = -numeric_limits<float>::infinity();
            for (size_t t = 0; t < scores.size(); ++t) {
                if (scores[t] >= best_score) {
                    best_t = t;
                    best_score = scores[t];
                }
            }
            predictions->push_back(best_t);
        }
    }

    double sentence_loss = 0.0;
    if (is_training_) {
        sum(losses);  // Sum of negative log probabilities of gold tags

        // Forward pass on the graph.
        sentence_loss = cnn::as_scalar(graph->incremental_forward());

        // Backprop.
        graph->backward();
    }
    return sentence_loss;
}

double BiLSTMTagger::BuildGlobalLossGraph(const vector<Expression> &tag_weights,
                                          const vector<size_t> &t_sequence,
                                          cnn::ComputationGraph *graph,
                                          vector<size_t> *predictions) {
    double sentence_loss = 0.0;
    if (is_training_) {
        Expression gold_score;  // Score of gold tag sequence
        ComputeScore(tag_weights, t_sequence, tag_weights.size(), true, graph,
                     &gold_score);

        Expression log_global_normalizer;  // log SUM{ exp(score(t_1...t_N)) }
        ComputeLogGlobalNormalizer(tag_weights, graph, &log_global_normalizer);

        log_global_normalizer - gold_score;  // Global loss

        // Forward pass on the graph.
        sentence_loss = cnn::as_scalar(graph->incremental_forward());

        // Backprop.
        graph->backward();
    } else {
        Viterbi(tag_weights, graph, predictions);
    }
    return sentence_loss;
}

double BiLSTMTagger::BuildGlobalLossGraphBeam(
    const vector<Expression> &tag_weights, const vector<size_t> &t_sequence,
    cnn::ComputationGraph *graph, vector<size_t> *predictions) {
    double sentence_loss = 0.0;
    if (is_training_) {
        vector<tuple<vector<size_t>, Expression, float, bool> > beam;
        bool beam_is_complete = ComputeBeam(tag_weights, t_sequence, graph,
                                            &beam);

        Expression gold_score;  // Score of gold tag sequence up to beam length
        ComputeScore(tag_weights, t_sequence, get<0>(beam[0]).size(),
                     beam_is_complete, graph, &gold_score);

        vector<Expression> scores;
        Expression ep = parameter(*graph, tag_end_);
        for (size_t i = 0; i < beam.size(); ++i) {
            size_t t_last = get<0>(beam[i])[get<0>(beam[i]).size() - 1];
            scores.push_back(get<1>(beam[i]) + pick(ep, t_last));
        }
        // log SUM{ exp(score(beam_element)) }
        Expression log_beam_normalizer = logsumexp(scores);

        log_beam_normalizer - gold_score;  // Global loss with beam

        // Forward pass on the graph.
        sentence_loss = cnn::as_scalar(graph->incremental_forward());

        // Backprop.
        graph->backward();
    } else {
        BeamSearch(tag_weights, graph, predictions);
    }
    return sentence_loss;
}

void BiLSTMTagger::BuildGraph(const vector<size_t> &x_sequence,
                              const vector<string> &sentence,
                              vector<size_t> *predictions) {
    vector<size_t> t_sequence_empty;  // Not used.
    BuildGraph(x_sequence, sentence, t_sequence_empty, predictions);
}

void BiLSTMTagger::ComputeWordExpressions(
    const vector<size_t> &x_sequence, const vector<string> &sentence,
    cnn::ComputationGraph *graph, vector<Expression> *word_expressions) {
    // Get "[fw_char bw_char word_emb]" for each word.
    vector<Expression> lstm_inputs;
    for (size_t i = 0; i < x_sequence.size(); ++i) {
        string word = sentence.at(i);

        // Forward character-level LSTM
        char_lstm_builder_forward_.new_graph(*graph);
        if (is_training_) {
            char_lstm_builder_forward_.set_dropout(dropout_rate_);
        } else {
            char_lstm_builder_forward_.disable_dropout();
        }
        char_lstm_builder_forward_.start_new_sequence();
        for (size_t j = 0; j < word.size(); ++j) {
            size_t c = (char_index_.find(word[j]) != char_index_.end()) ?
                char_index_[word[j]] : kUnknownCharIndex_;
            if (is_training_) {
                double drop_probability = 0.25 / (char_count_[c] + 0.25);
                if ((double) rand() / RAND_MAX < drop_probability) {
                    c = kUnknownCharIndex_;
                }
            }
            Expression char_embedding = lookup(*graph, char_embeddings_, c);
            char_lstm_builder_forward_.add_input(char_embedding);
        }
        Expression char_forward = char_lstm_builder_forward_.back();

        // Backward character-level LSTM
        char_lstm_builder_backward_.new_graph(*graph);
        if (is_training_) {
            char_lstm_builder_backward_.set_dropout(dropout_rate_);
        } else {
            char_lstm_builder_backward_.disable_dropout();
        }
        char_lstm_builder_backward_.start_new_sequence();
        for (int j = word.size() - 1; j >= 0; --j) {
            size_t c = (char_index_.find(word[j]) != char_index_.end()) ?
                char_index_[word[j]] : kUnknownCharIndex_;
            if (is_training_) {
                double drop_probability = 0.25 / (char_count_[c] + 0.25);
                if ((double) rand() / RAND_MAX < drop_probability) {
                    c = kUnknownCharIndex_;
                }
            }
            Expression char_embedding = lookup(*graph, char_embeddings_, c);
            char_lstm_builder_backward_.add_input(char_embedding);
        }
        Expression char_backward = char_lstm_builder_backward_.back();

        // Word embedding
        size_t x = x_sequence.at(i);
        if (is_training_) {
            double drop_probability = 0.25 / (word_count_[x] + 0.25);
            if ((double) rand() / RAND_MAX < drop_probability) {
                x = kUnknownWordIndex_;
            }
        }
        Expression word_embedding = lookup(*graph, word_embeddings_, x);

        // Concatenate the three expressions.
        Expression lstm_input =
            concatenate({char_forward, char_backward, word_embedding});

        lstm_inputs.push_back(lstm_input);
    }

    // Then, run LSTMs on these inputs [fw_char bw_char word_emb]'s.
    vector<Expression> word_forwards;
    word_lstm_builder_forward_.new_graph(*graph);
    if (is_training_) {
        word_lstm_builder_forward_.set_dropout(dropout_rate_);
    } else {
        word_lstm_builder_forward_.disable_dropout();
    }
    word_lstm_builder_forward_.start_new_sequence();
    for (size_t i = 0; i < lstm_inputs.size(); ++i) {
        word_forwards.push_back(
            word_lstm_builder_forward_.add_input(lstm_inputs[i]));
    }

    vector<Expression> word_backwards;
    word_lstm_builder_backward_.new_graph(*graph);
    if (is_training_) {
        word_lstm_builder_backward_.set_dropout(dropout_rate_);
    } else {
        word_lstm_builder_backward_.disable_dropout();
    }
    word_lstm_builder_backward_.start_new_sequence();
    for (int i = lstm_inputs.size() - 1; i >= 0; --i) {
        word_backwards.push_back(
            word_lstm_builder_backward_.add_input(lstm_inputs[i]));
    }

    // Finally, return [fw_word bw_word]'s.
    word_expressions->clear();
    for (size_t i = 0; i < word_forwards.size(); ++i) {
        size_t i_reverse = word_forwards.size() - 1 - i;  // Match index!
        Expression word_expression =
            concatenate({word_forwards[i], word_backwards[i_reverse]});
        word_expressions->push_back(word_expression);
    }
}

void BiLSTMTagger::Activate(const Expression &v, Expression *v_activated) {
    if (activation_ == "tanh") {
        *v_activated = tanh(v);
    } else if (activation_ == "relu") {
        *v_activated = rectify(v);
    } else {
        ASSERT(false, "Unknown activation: " << activation_);
    }
}

double BiLSTMTagger::EvaluatePerformance(
    const vector<vector<string> > &gold_tag_sequences,
    const vector<vector<string> > &predicted_tag_sequences) {
    double performance = 0.0;

    if (performance_measure_ == "acc") {  // Per-word accuracy
        double position_accuracy;
        double sequence_accuracy;
        util_eval::compute_accuracy(gold_tag_sequences, predicted_tag_sequences,
                                    &position_accuracy, &sequence_accuracy);
        performance = position_accuracy;
    } else if (performance_measure_ == "f1") {  // Overall labeled F1 score
        unordered_map<string, double> per_entity_precision;
        unordered_map<string, double> per_entity_recall;
        unordered_map<string, size_t> num_guessed;
        double precision;
        double recall;
        util_eval::compute_precision_recall_bio(
            gold_tag_sequences, predicted_tag_sequences, &per_entity_precision,
            &per_entity_recall, &num_guessed, &precision, &recall);

        return (precision > 0.0) ?
            2 * precision * recall / (precision + recall) : 0.0;
    } else {
        ASSERT(false, "Unknown performance measure: " << performance_measure_);
    }

    return performance;
}

double BiLSTMTagger::EvaluatePerformance(const string &data_path,
                                         const string &prediction_path) {
    pair<vector<vector<string> >, vector<vector<string> > > test_data;
    ReadTestData(data_path, &test_data);
    vector<vector<string> > predicted_tag_sequences;
    Predict(test_data.first, &predicted_tag_sequences);
    double performance = EvaluatePerformance(test_data.second,
                                             predicted_tag_sequences);
    if (!prediction_path.empty()) {
        ofstream file(prediction_path, ios::out);
        ASSERT(file.is_open(), "Cannot open file: " << prediction_path);
        for (size_t i = 0; i < test_data.first.size(); ++i) {
	    for (size_t j = 0; j < test_data.first[i].size(); ++j) {
		file << test_data.first[i][j] << "\t";
		file << test_data.second[i][j] << "\t";
		file << predicted_tag_sequences[i][j] << endl;
	    }
	    file << endl;
        }
    }
    return performance;
}

void BiLSTMTagger::Predict(const vector<vector<string> > &sentences,
                           vector<vector<string> > *predicted_tag_sequences) {
    predicted_tag_sequences->clear();
    for (size_t i = 0; i < sentences.size(); ++i) {
        vector<string> predicted_tags;
        Predict(sentences.at(i), &predicted_tags);
        predicted_tag_sequences->push_back(predicted_tags);
    }
}

void BiLSTMTagger::Predict(const vector<string> &sentence,
                           vector<string> *tags) {
    ASSERT(!is_training_, "Training flag is on at prediction!");
    tags->clear();
    vector<size_t> x_sequence;
    for (size_t i = 0; i < sentence.size(); ++i) {
        size_t x = (word_index_.find(sentence.at(i)) != word_index_.end()) ?
            word_index_[sentence.at(i)] : kUnknownWordIndex_;
        x_sequence.push_back(x);
    }

    vector<size_t> t_sequence;
    BuildGraph(x_sequence, sentence, &t_sequence);

    for (size_t i = 0; i < t_sequence.size(); ++i) {
        tags->push_back(tag_string_[t_sequence[i]]);
    }
}

void BiLSTMTagger::SaveModel() {
    // Dictionaries
    ofstream char_index_file(model_directory_ + "/char_index");
    boost::archive::text_oarchive o_char_index(char_index_file);
    o_char_index << char_index_;

    ofstream word_index_file(model_directory_ + "/word_index");
    boost::archive::text_oarchive o_word_index(word_index_file);
    o_word_index << word_index_;

    ofstream tag_string_file(model_directory_ + "/tag_string");
    boost::archive::text_oarchive o_tag_string(tag_string_file);
    o_tag_string << tag_string_;

    // Other values
    ofstream other_file(model_directory_ + "/other");
    util_file::binary_write_primitive(dim_char_embeddings_, other_file);
    util_file::binary_write_primitive(dim_word_embeddings_, other_file);
    util_file::binary_write_primitive(dim_lstm_outputs_, other_file);
    util_file::binary_write_primitive(dim_hidden_layers_, other_file);
    util_file::binary_write_primitive(num_lstm_layers_, other_file);
    util_file::binary_write_string(activation_, other_file);
    util_file::binary_write_string(performance_measure_, other_file);
    util_file::binary_write_string(loss_function_, other_file);

    // Model parameters
    ofstream model_file(model_directory_ + "/model");
    boost::archive::text_oarchive o_model(model_file);
    o_model << model_;
}

void BiLSTMTagger::LoadModel() {
    // Dictionaries
    ifstream char_index_file(model_directory_ + "/char_index");
    boost::archive::text_iarchive i_char_index(char_index_file);
    i_char_index >> char_index_;

    ifstream word_index_file(model_directory_ + "/word_index");
    boost::archive::text_iarchive i_word_index(word_index_file);
    i_word_index >> word_index_;

    ifstream tag_string_file(model_directory_ + "/tag_string");
    boost::archive::text_iarchive i_tag_string(tag_string_file);
    i_tag_string >> tag_string_;

    // Other values
    ifstream other_file(model_directory_ + "/other");
    util_file::binary_read_primitive(other_file, &dim_char_embeddings_);
    util_file::binary_read_primitive(other_file, &dim_word_embeddings_);
    util_file::binary_read_primitive(other_file, &dim_lstm_outputs_);
    util_file::binary_read_primitive(other_file, &dim_hidden_layers_);
    util_file::binary_read_primitive(other_file, &num_lstm_layers_);
    util_file::binary_read_string(other_file, &activation_);
    util_file::binary_read_string(other_file, &performance_measure_);
    util_file::binary_read_string(other_file, &loss_function_);

    // Model parameters
    if (char_embeddings_ == nullptr) {  // Model hasn't been built.
        BuildModel();
    }
    ifstream model_file(model_directory_ + "/model");
    boost::archive::text_iarchive i_model(model_file);
    i_model >> model_;

    is_training_ = false;  // By default, loaded models are not training.
}

double BiLSTMTagger::Viterbi(const vector<Expression> &tag_weights,
                             cnn::ComputationGraph *graph,
                             vector<size_t> *t_sequence) {
    Expression pi = parameter(*graph, tag_prior_);
    Expression T = parameter(*graph, tag_transition_);
    Expression ep = parameter(*graph, tag_end_);
    size_t sentence_length = tag_weights.size();
    size_t num_tags = tag_string_.size();

    // chart[i][t] = MAX{ score(t_1...t_i): t_i = t }
    vector<vector<Expression> > chart(sentence_length);
    vector<vector<size_t> > backpointer(sentence_length);
    for (size_t i = 0; i < sentence_length; ++i) {
	chart[i].resize(num_tags);
	backpointer[i].resize(num_tags);
    }

    // Base case.
    for (size_t t = 0; t < num_tags; ++t) {
	chart[0][t] = pick(pi, t) + pick(tag_weights[0], t);
    }

    // Main body.
    for (size_t i = 1; i < sentence_length; ++i) {
	for (size_t t = 0; t < num_tags; ++t) {
            Expression T_t = select_cols(T, {(unsigned int) t});
            Expression max_score;
            float max_score_float = -numeric_limits<float>::infinity();
	    size_t best_t_previous = 0;
	    for (size_t t_previous = 0; t_previous < num_tags; ++t_previous) {
		Expression score =
                    chart[i - 1][t_previous] +
		    pick(T_t, t_previous) +
                    pick(tag_weights[i], t);
                float score_float =
                    cnn::as_scalar(graph->incremental_forward());
		if (score_float >= max_score_float) {
                    max_score = score;
		    max_score_float = score_float;
		    best_t_previous = t_previous;
		}
	    }
	    chart[i][t] = max_score;
	    backpointer[i][t] = best_t_previous;
	}
    }

    // Maximization over the final position.
    double max_score_float = -numeric_limits<float>::infinity();
    size_t best_t_final = 0;
    for (size_t t = 0; t < num_tags; ++t) {
        chart[sentence_length - 1][t] + pick(ep, t);
        float score_float = cnn::as_scalar(graph->incremental_forward());
	if (score_float >= max_score_float) {
            max_score_float = score_float;
	    best_t_final = t;
	}
    }

    // Backtrack to recover the best tag sequence.
    t_sequence->resize(backpointer.size());
    (*t_sequence)[backpointer.size() - 1] = best_t_final;
    size_t current_best_t = best_t_final;
    for (size_t i = backpointer.size() - 1; i > 0; --i) {
	current_best_t = backpointer[i][current_best_t];
	(*t_sequence)[i - 1] = current_best_t;
    }

    return (double) max_score_float;
}

bool BiLSTMTagger::ComputeBeam(
    const vector<Expression> &tag_weights, const vector<size_t> &true_sequence,
    cnn::ComputationGraph *graph,
    vector<tuple<vector<size_t>, Expression, float, bool> > *beam) {
    Expression pi = parameter(*graph, tag_prior_);
    Expression T = parameter(*graph, tag_transition_);
    size_t num_tags = tag_string_.size();

    BeamQueue beam_queue(beam_size_);
    for (size_t t = 0; t < num_tags; ++t) {
        Expression score = pick(pi, t) + pick(tag_weights[0], t);
        float score_float = cnn::as_scalar(graph->incremental_forward());
        vector<size_t> partial_sequence = {t};
        bool is_gold = (is_training_) && (t == true_sequence.at(0));
        tuple<vector<size_t>, Expression, float, bool> element =
            make_tuple(partial_sequence, score, score_float, is_gold);
        beam_queue.Insert(element);
    }

    for (size_t i = 1; i < tag_weights.size(); ++i) {
        vector<tuple<vector<size_t>, Expression, float, bool> > current_beam;
        beam_queue.Dump(&current_beam);

        if (is_training_) {  // Early updates
            bool gold_in_beam = false;
            for (auto &element : current_beam) {
                if (get<3>(element)) { gold_in_beam = true; }
            }
            if (!gold_in_beam) {
                *beam = current_beam;
                return false;
            }
        }

        for (auto &element : current_beam) {
            for (size_t t = 0; t < num_tags; ++t) {
                Expression T_t = select_cols(T, {(unsigned int) t});
                size_t t_previous = get<0>(element)[i - 1];
                Expression score =
                    get<1>(element) +
                    pick(T_t, t_previous) +
                    pick(tag_weights[i], t);
                float score_float =
                    cnn::as_scalar(graph->incremental_forward());
                vector<size_t> partial_sequence(get<0>(element));
                partial_sequence.push_back(t);
                bool is_gold = (is_training_) && get<3>(element) &&
                    (t == true_sequence.at(i));
                tuple<vector<size_t>, Expression, float, bool> next_element =
                    make_tuple(partial_sequence, score, score_float, is_gold);
                beam_queue.Insert(next_element);
	    }
	}
    }
    beam_queue.Dump(beam);

    if (is_training_) {  // Early updates
        bool gold_in_beam = false;
        for (auto &element : *beam) {
            if (get<3>(element)) { gold_in_beam = true; }
        }
        if (!gold_in_beam) { return false; }
    }
    return true;
}

double BiLSTMTagger::BeamSearch(const vector<Expression> &tag_weights,
                                cnn::ComputationGraph *graph,
                                vector<size_t> *t_sequence) {
    vector<size_t> empty_sequence;
    vector<tuple<vector<size_t>, Expression, float, bool> > beam;
    ComputeBeam(tag_weights, empty_sequence, graph, &beam);

    Expression ep = parameter(*graph, tag_end_);
    float max_score_float = -numeric_limits<float>::infinity();;
    size_t best_element_index = 0;
    for (size_t i = 0; i < beam.size(); ++i) {
        size_t t_last = get<0>(beam[i])[get<0>(beam[i]).size() - 1];
        get<1>(beam[i]) + pick(ep, t_last);  // Final tag score
        float score_float = cnn::as_scalar(graph->incremental_forward());
        if (score_float > max_score_float) {
            best_element_index = i;
            max_score_float = score_float;
        }
    }

    *t_sequence = get<0>(beam[best_element_index]);
    return max_score_float;
}

void BiLSTMTagger::ComputeLogGlobalNormalizer(
    const vector<Expression> &tag_weights, cnn::ComputationGraph *graph,
    Expression *normalizer) {
    Expression pi = parameter(*graph, tag_prior_);
    Expression T = parameter(*graph, tag_transition_);
    Expression ep = parameter(*graph, tag_end_);
    size_t sentence_length = tag_weights.size();
    size_t num_tags = tag_string_.size();

    // chart[i][t] = log SUM{ exp(score(t_1...t_i)): t_i = t }
    vector<vector<Expression> > chart(sentence_length);
    for (size_t i = 0; i < sentence_length; ++i) { chart[i].resize(num_tags); }

    // Base case.
    for (size_t t = 0; t < num_tags; ++t) {
	chart[0][t] = pick(pi, t) + pick(tag_weights[0], t);
    }

    // Main body.
    for (size_t i = 1; i < sentence_length; ++i) {
	for (size_t t = 0; t < num_tags; ++t) {
            Expression T_t = select_cols(T, {(unsigned int) t});
            vector<Expression> scores;
	    for (size_t t_previous = 0; t_previous < num_tags; ++t_previous) {
		Expression score =
                    chart[i - 1][t_previous] +
		    pick(T_t, t_previous) +
                    pick(tag_weights[i], t);
                scores.push_back(score);
	    }
	    chart[i][t] = logsumexp(scores);
	}
    }

    // Sum over the final position.
    vector<Expression> final_scores;
    for (size_t t = 0; t < num_tags; ++t) {
        final_scores.push_back(chart[sentence_length - 1][t] + pick(ep, t));
    }
    *normalizer = logsumexp(final_scores);
}

void BiLSTMTagger::ComputeScore(const vector<Expression> &tag_weights,
                                const vector<size_t> &t_sequence,
                                size_t desired_length,
                                bool add_final_tag_score,
                                cnn::ComputationGraph *graph,
                                Expression *score) {
    Expression pi = parameter(*graph, tag_prior_);
    Expression T = parameter(*graph, tag_transition_);
    Expression ep = parameter(*graph, tag_end_);

    *score = pick(pi, t_sequence.at(0)) + pick(tag_weights[0],
                                               t_sequence.at(0));

    for (size_t i = 1; i < desired_length; ++i) {
        size_t t_previous = t_sequence.at(i - 1);
        size_t t = t_sequence.at(i);
        Expression T_t = select_cols(T, {(unsigned int) t});
        *score = *score + pick(T_t, t_previous) + pick(tag_weights[i], t);
    }

    if (add_final_tag_score) {
        *score = *score + pick(ep, t_sequence.at(t_sequence.size() - 1));
    }
}

void BeamQueue::Insert(const tuple<vector<size_t>, Expression, float, bool>
                       &element) {
    if (q_.size() < beam_size_) {  // Queue is not yet full.
        q_.push(element);
    } else if (get<2>(element) > get<2>(q_.top())) {
        // Queue is full but this one is better than the worst beam element.
        q_.pop();
        q_.push(element);
    } else {
        // Queue is full and this one falls out of the beam: do not add.
    }
}

void BeamQueue::Dump(vector<tuple<vector<size_t>, Expression, float, bool> >
                     *beam) {
    beam->clear();
    while (!q_.empty()) {
        (*beam).push_back(q_.top());
        q_.pop();
    }
}
