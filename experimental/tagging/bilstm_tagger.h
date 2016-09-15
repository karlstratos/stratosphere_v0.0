// Author: Karl Stratos (jlee2071@bloomberg.net)

#include <queue>

#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
#include "tagging/util.h"

class BiLSTMTagger {
  public:
    // Initializes with an output directory.
    BiLSTMTagger(const string &model_directory) {
	SetModelDirectory(model_directory);
    }

    // Sets the model directory.
    void SetModelDirectory(const string &model_directory);

    // Resets the content in the model directory.
    void ResetModelDirectory();

    // Trains the model from a text file of labeled sequences.
    void Train(const string &data_path);

    // Reads training data, also populates various char/word/tag mappings.
    // Returns the total number of words.
    size_t ReadTrainingData(
        const string &data_path,
        vector<pair<vector<size_t>, vector<size_t> > > *training_data,
        vector<vector<string> > *sentences);

    // Reads test data. Returns the total number of words.
    size_t ReadTestData(
        const string &data_path,
        pair<vector<vector<string> >, vector<vector<string> > > *test_data);

    // Reads embeddings.
    void ReadEmbeddings(unordered_map<string, vector<double> > *embeddings);

    // Builds the model.
    void BuildModel();

    // Builds the computation graph for given tagged sentence. Returns loss.
    double BuildGraph(const vector<size_t> &x_sequence,
                      const vector<string> &sentence,
                      const vector<size_t> &t_sequence,
                      vector<size_t> *predictions);

    // Builds the local loss computation graph. Returns loss.
    double BuildLocalLossGraph(const vector<Expression> &tag_weights,
                               const vector<size_t> &t_sequence,
                               cnn::ComputationGraph *graph,
                               vector<size_t> *predictions);

    // Builds the global loss computation graph. Returns loss.
    double BuildGlobalLossGraph(const vector<Expression> &tag_weights,
                                const vector<size_t> &t_sequence,
                                cnn::ComputationGraph *graph,
                               vector<size_t> *predictions);

    // Builds the global loss computation graph with beam. Returns loss.
    double BuildGlobalLossGraphBeam(const vector<Expression> &tag_weights,
                                    const vector<size_t> &t_sequence,
                                    cnn::ComputationGraph *graph,
                                    vector<size_t> *predictions);

    // Builds the computation graph for given sentence (used for prediction).
    void BuildGraph(const vector<size_t> &x_sequence,
                    const vector<string> &sentence,
                    vector<size_t> *predictions);

    // Computes a sequence of word expressions for the given sentence.
    void ComputeWordExpressions(const vector<size_t> &x_sequence,
                                const vector<string> &sentence,
                                cnn::ComputationGraph *graph,
                                vector<Expression> *word_expressions);

    // Evaluates the model performance on the given predicted tag sequences.
    double EvaluatePerformance(
        const vector<vector<string> > &gold_tag_sequences,
        const vector<vector<string> > &predicted_tag_sequences);

    // Evaluates the model performance on the given labeled data.
    double EvaluatePerformance(const string &data_path,
                               const string &prediction_path);

    // Predicts tag sequences for the given sentences.
    void Predict(const vector<vector<string> > &sentences,
                 vector<vector<string> > *predicted_tag_sequences);

    // Predicts a tag sequence for the given word sequence.
    void Predict(const vector<string> &sentence, vector<string> *tags);

    // Applies an activation function on v.
    void Activate(const Expression &v, Expression *v_activated);

    // Saves the model.
    void SaveModel();

    // Loads the model.
    void LoadModel();

    // Performs Viterbi decoding, returns the computed score.
    double Viterbi(const vector<Expression> &tag_weights,
                   cnn::ComputationGraph *graph,
                   vector<size_t> *t_sequence);

    // Computes beam (without the final tag score).
    // Training: stops early if gold falls off the beam, returns false.
    bool ComputeBeam(const vector<Expression> &tag_weights,
                     const vector<size_t> &true_sequence,
                     cnn::ComputationGraph *graph,
                     vector<tuple<vector<size_t>, Expression, float, bool> >
                     *beam);

    // Performs beam search, returns the computed score.
    double BeamSearch(const vector<Expression> &tag_weights,
                      cnn::ComputationGraph *graph,
                      vector<size_t> *t_sequence);

    // Given tag weights for length-N sentence, computes log global normalizer:
    //   log SUM{ exp(score(t_1 ... t_N)) }
    void ComputeLogGlobalNormalizer(const vector<Expression> &tag_weights,
                                    cnn::ComputationGraph *graph,
                                    Expression *normalizer);

    // Given tag weights for length-N sentence, does an early update with
    // beam search. Returns sentence loss.
    double BeamEarlyUpdate(const vector<size_t> &t_sequence,
                         const vector<Expression> &tag_weights,
                         cnn::ComputationGraph *graph);

    // Computes the score of the tag sequence given <tag_weights> up to the
    // desired length.
    void ComputeScore(const vector<Expression> &tag_weights,
                      const vector<size_t> &t_sequence,
                      size_t desired_length, bool add_final_tag_score,
                      cnn::ComputationGraph *graph, Expression *score);

    // Sets the path to development data.
    void set_development_path(string development_path) {
	development_path_ = development_path;
    }

    // Sets the interval to check development accuracy.
    void set_development_interval(size_t development_interval) {
	development_interval_ = development_interval;
    }

    // Sets the path to pre-trained word embeddings.
    void set_embedding_path(string embedding_path) {
	embedding_path_ = embedding_path;
    }

    // Sets the dimension of character embeddings.
    void set_dim_char_embeddings(size_t dim_char_embeddings) {
	dim_char_embeddings_ = dim_char_embeddings;
    }

    // Sets the dimension of word embeddings.
    void set_dim_word_embeddings(size_t dim_word_embeddings) {
	dim_word_embeddings_ = dim_word_embeddings;
    }

    // Sets the dimension of LSTM outputs/hidden vectors.
    void set_dim_lstm_outputs(size_t dim_lstm_outputs) {
	dim_lstm_outputs_ = dim_lstm_outputs;
    }

    // Sets the dimension of the hidden layers in feed-forward networks.
    void set_dim_hidden_layers(size_t dim_hidden_layers) {
	dim_hidden_layers_ = dim_hidden_layers;
    }

    // Sets the number of LSTM layers.
    void set_num_lstm_layers(size_t num_lstm_layers) {
        num_lstm_layers_ = num_lstm_layers;
    }

    // Sets the activation function.
    void set_activation(string activation) { activation_ = activation; }

    // Sets the learning rate.
    void set_learning_rate(double learning_rate) {
        learning_rate_ = learning_rate;
    }

    // Sets the dropout rate.
    void set_dropout_rate(double dropout_rate) {
        dropout_rate_ = dropout_rate;
    }

    // Sets the number of epochs.
    void set_num_epochs(size_t num_epochs) { num_epochs_ = num_epochs; }

    // Sets the performance measure.
    void set_performance_measure(string performance_measure) {
        performance_measure_ = performance_measure;
    }

    // Sets the loss function.
    void set_loss_function(string loss_function) {
        loss_function_ = loss_function;
    }

    // Sets the beam size.
    void set_beam_size(size_t beam_size) { beam_size_ = beam_size; }

  private:
    // Path to the model directory.
    string model_directory_;

    // Path to development data.
    string development_path_;

    // Interval to check development accuracy.
    size_t development_interval_ = 10000;

    // Path to pre-trained word embeddings.
    string embedding_path_;

    // Dimension of character embeddings.
    size_t dim_char_embeddings_ = 25;

    // Dimension of word embeddings.
    size_t dim_word_embeddings_ = 50;

    // Dimension of (word-level) LSTM outputs/hidden vectors.
    size_t dim_lstm_outputs_ = 50;

    // Dimension of the hidden layers in feed-forward networks.
    size_t dim_hidden_layers_ = 50;

    // Number of LSTM layers.
    size_t num_lstm_layers_ = 1;

    // Activation function.
    string activation_ = "tanh";

    // Learning rate.
    double learning_rate_ = 0.0005;  // Also try 0.0004, 0.0006, etc.

    // Dropout rate.
    double dropout_rate_ = 0.5;  // Also try 0.2, 0.3, 0.4, 0.5, 0.6

    // Number of epochs (passes over the entire data).
    size_t num_epochs_ = 100;

    // Performance measure.
    string performance_measure_ = "acc";

    // Loss function.
    string loss_function_ = "local";

    // Beam size.
    size_t beam_size_ = 10;

    // Special character for unknown characters.
    const char kUnknownChar_ = '?';  // Will never be actually used.

    // Index of kUnknownChar_ in char_index_;
    const size_t kUnknownCharIndex_ = 0;

    // Special string for unknown words.
    const string kUnknownWord_ = "<?>";

    // Index of kUnknownWord_ in word_index_;
    const size_t kUnknownWordIndex_ = 0;

    // Are we training the model?
    bool is_training_ = false;

    // Character index: char -> integer
    unordered_map<char, size_t> char_index_;

    // Character count: integer -> count of the corresponding character
    vector<size_t> char_count_;

    // Character string: integer -> char
    vector<char> char_string_;

    // Word index: string -> integer
    unordered_map<string, size_t> word_index_;

    // Word count: integer -> count of the corresponding word
    vector<size_t> word_count_;

    // Word string: integer -> string
    vector<string> word_string_;

    // Tag index: string -> integer
    unordered_map<string, size_t> tag_index_;

    // Tag string: integer -> string
    vector<string> tag_string_;

    // The model.
    cnn::Model model_;

    // Character-level LSTM builder (forward).
    cnn::LSTMBuilder char_lstm_builder_forward_;

    // Character-level LSTM builder (backward).
    cnn::LSTMBuilder char_lstm_builder_backward_;

    // Word-level LSTM builder (forward).
    cnn::LSTMBuilder word_lstm_builder_forward_;

    // Word-level LSTM builder (backward).
    cnn::LSTMBuilder word_lstm_builder_backward_;

    // Character embeddings (number of character types x character dimension).
    cnn::LookupParameters* char_embeddings_ = nullptr;

    // Word embeddings (number of word types x word dimension).
    cnn::LookupParameters* word_embeddings_ = nullptr;

    // Weight matrix at layer 1 (hidden dimension x (2*output LSTM dimension)).
    cnn::Parameters* weight_matrix1_ = nullptr;

    // Bias at layer 1 (hidden dimension x 1).
    cnn::Parameters* bias1_ = nullptr;

    // Weight matrix at output layer (number of tag types x hidden dimension).
    cnn::Parameters* weight_matrix2_ = nullptr;

    // Bias at layer 2 (number of tag types x 1).
    cnn::Parameters* bias2_ = nullptr;

    // Tag prior scores (number of tag types x 1).
    cnn::Parameters* tag_prior_ = nullptr;

    // Tag transition scores (number of tag types x number of tag types), where
    // score(a -> b) is accessed as:
    //    Expression T = parameter(graph, tag_transition_);
    //    Expression T_b = select_cols(T, {b});
    //    Expression score(a -> b) = pick(T_b, a);
    cnn::Parameters* tag_transition_ = nullptr;

    // Tag ending scores (number of tag types x 1).
    cnn::Parameters* tag_end_ = nullptr;
};

// A beam element is a tuple (tag_sequence, score_e, score_f, is_gold), where
// "score_e" and "score_f" are the score of the tag_sequence in Expression and
// float, and is_gold is true if tag_sequence conforms to the gold sequence.
class CompareBeamElements {
public:
    bool operator() (tuple<vector<size_t>, Expression, float, bool> e1,
                     tuple<vector<size_t>, Expression, float, bool> e2) {
        return get<2>(e1) > get<2>(e2);
    }
};

class BeamQueue {
  public:
    BeamQueue(size_t beam_size) : beam_size_(beam_size) { }

    // Inserts a beam element.
    void Insert(const tuple<vector<size_t>, Expression, float, bool> &element);

    // Dumps all the beam elements into a list.
    void Dump(vector<tuple<vector<size_t>, Expression, float, bool> > *beam);

  private:
    size_t beam_size_ = 10;

    // Min heap: q_.top() has the *worst* score in the beam.
    priority_queue<tuple<vector<size_t>, Expression, float, bool>,
                   vector<tuple<vector<size_t>, Expression, float, bool> >,
                   CompareBeamElements> q_;
};
