// Author: Karl Stratos (jlee2071@bloomberg.net)

#include <queue>

#include <cnn/nodes.h>
#include <cnn/cnn.h>
#include <cnn/training.h>
#include <cnn/timing.h>
#include <cnn/rnn.h>
#include <cnn/gru.h>
#include <cnn/lstm.h>
#include <cnn/dict.h>
#include <cnn/expr.h>

#include "util.h"

class BiLSTMTagger {
  public:
    // Initializes with an output directory.
    BiLSTMTagger(const std::string &model_directory) {
	SetModelDirectory(model_directory);
    }

    // Resets the content in the model directory.
    void ResetModelDirectory();

    // Trains the model from a text file of labeled sequences.
    void Train(const std::string &data_path);

    // Gazetteer training.
    void TrainGazetteer(const std::string &data_path);

    // Evaluates the model performance on the given predicted tag sequences.
    double EvaluatePerformance(
        const std::vector<std::vector<std::string> > &gold_tag_sequences,
        const std::vector<std::vector<std::string> > &predicted_tag_sequences);

    // Evaluates the model performance on the given labeled data.
    double EvaluatePerformance(const std::string &data_path,
                               const std::string &prediction_path);

    // Predicts tag sequences for the given sentences. Returns the total time
    // (in seconds).
    double Predict(const std::vector<std::vector<std::string> > &sentences,
                   std::vector<std::vector<std::string> >
                   *predicted_tag_sequences);

    // Predicts a tag sequence for the given word sequence.
    void Predict(const std::vector<std::string> &sentence,
                 std::vector<std::string> *tags);

    // Predicts types for given entities.
    void PredictEntityType(
        const std::vector<std::vector<std::string> > &entities,
        std::vector<std::string> *predicted_types);

    // Predicts type for given entity.
    void PredictEntityType(const std::vector<std::string> &entity,
                           std::string *type);

    // Saves the model.
    void SaveModel();

    // Loads the model.
    void LoadModel();

    // Sets the path to development data.
    void set_development_path(std::string development_path) {
	development_path_ = development_path;
    }

    // Sets the interval to check development accuracy.
    void set_development_interval(size_t development_interval) {
	development_interval_ = development_interval;
    }

    // Sets the path to a gazetteer.
    void set_gazetteer_path(std::string gazetteer_path) {
        gazetteer_path_ = gazetteer_path;
    }


    // Sets the path to pre-trained word embeddings.
    void set_embedding_path(std::string embedding_path) {
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

    // Sets the dimension of gazetteer label embeddings.
    void set_dim_gazetteer_embeddings(
        size_t dim_gazetteer_embeddings) {
	dim_gazetteer_embeddings_ = dim_gazetteer_embeddings;
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
    void set_activation(std::string activation) { activation_ = activation; }

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
    void set_performance_measure(std::string performance_measure) {
        performance_measure_ = performance_measure;
    }

    // Sets the loss function.
    void set_loss_function(std::string loss_function) {
        loss_function_ = loss_function;
    }

    // Sets the beam size.
    void set_beam_size(size_t beam_size) { beam_size_ = beam_size; }

    // Sets the flag for using second LSTMs (word-level).
    void set_use_second_lstms(bool use_second_lstms) {
        use_second_lstms_ = use_second_lstms;
    }

    // Sets the flag for using gazetteer embeddings.
    void set_use_gazemb(bool use_gazemb) { use_gazemb_ = use_gazemb; }

    // Sets the flag for doing gazetteer regularized training.
    void set_do_gaztrain(bool do_gaztrain) { do_gaztrain_ = do_gaztrain; }

  private:
    // Sets the model directory.
    void SetModelDirectory(const std::string &model_directory);

    // Runs training epochs for tagging.
    void RunTaggingEpochs(
        const std::vector<std::pair<std::vector<size_t>, std::vector<size_t> > >
        &training_data, const std::vector<std::vector<std::string> > &sentences,
        const std::pair<std::vector<std::vector<std::string> >,
        std::vector<std::vector<std::string> > > &dev_data,
        size_t num_words_dev, size_t num_epochs, cnn::AdamTrainer *trainer,
        double *max_performance);

    // Runs training epochs for entity classification.
    void RunEntityClassificationEpochs(
        const std::vector<std::pair<std::vector<size_t>, size_t> >
        &entity_type_pairs, const std::vector<std::vector<std::string> >
        &entity_words, size_t num_epochs, cnn::AdamTrainer *trainer);

    // Prints training information.
    void PrintTrainingInfo(const std::string &training_data_path,
                           size_t num_sentences_training,
                           size_t num_words_training,
                           size_t num_sentences_dev,
                           size_t num_words_dev,
                           size_t dim_pretrained_word_embeddings,
                           size_t num_pretrained_word_embeddings);

    // Reads training data, also initializes various char/word/tag mappings.
    // Returns the total number of words.
    size_t ReadTrainingData(
        const std::string &data_path,
        std::vector<std::pair<std::vector<size_t>, std::vector<size_t> > >
        *training_data,
        std::vector<std::vector<std::string> > *sentences);

    // Loads entities from training data (must be in BIO labels).
    void LoadEntities(const std::vector<std::pair<std::vector<size_t>,
                      std::vector<size_t> > > &training_data);

    // Loads entities from a gazetteer file.
    void LoadEntitiesFromGazetteerFile();

    // Reads test data. Returns the total number of words.
    size_t ReadTestData(
        const std::string &data_path,
        std::pair<std::vector<std::vector<std::string> >,
        std::vector<std::vector<std::string> > > *test_data);

    // Reads embeddings, augments word dictionary. Returns the dimension.
    size_t ReadEmbeddings(std::unordered_map<std::string, std::vector<double> >
                          *embeddings);

    // Builds the model, initializes word parameters if given embeddings.
    void BuildModel(const std::unordered_map<std::string, std::vector<double> >
                    &embeddings);

    // Builds the computation graph for predicting entity type. Returns loss.
    double BuildGraphEntityType(const std::vector<size_t> &x_sequence,
                                const std::vector<std::string> &words,
                                size_t gold_type, size_t *prediction);

    // Builds the computation graph for given tagged sentence. Returns loss.
    double BuildGraph(const std::vector<size_t> &x_sequence,
                      const std::vector<std::string> &sentence,
                      const std::vector<size_t> &t_sequence,
                      std::vector<size_t> *predictions);

    // Builds the local loss computation graph. Returns loss.
    double BuildLocalLossGraph(const std::vector<cnn::expr::Expression>
                               &tag_weights,
                               const std::vector<size_t> &t_sequence,
                               cnn::ComputationGraph *graph,
                               std::vector<size_t> *predictions);

    // Builds the global loss computation graph. Returns loss.
    double BuildGlobalLossGraph(const std::vector<cnn::expr::Expression>
                                &tag_weights,
                                const std::vector<size_t> &t_sequence,
                                cnn::ComputationGraph *graph,
                               std::vector<size_t> *predictions);

    // Builds the global loss computation graph with beam. Returns loss.
    double BuildGlobalLossGraphBeam(const std::vector<cnn::expr::Expression>
                                    &tag_weights,
                                    const std::vector<size_t> &t_sequence,
                                    cnn::ComputationGraph *graph,
                                    std::vector<size_t> *predictions);

    // Builds the computation graph for given sentence (used for prediction).
    void BuildGraph(const std::vector<size_t> &x_sequence,
                    const std::vector<std::string> &sentence,
                    std::vector<size_t> *predictions);

    // Builds the computation graph for predicting entity type (for prediction).
    void BuildGraphEntityType(const std::vector<size_t> &x_sequence,
                                const std::vector<std::string> &words,
                                size_t *prediction);

    // Computes a sequence of word expressions for the given sentence.
    void ComputeWordExpressions(const std::vector<size_t> &x_sequence,
                                const std::vector<std::string> &sentence,
                                cnn::ComputationGraph *graph,
                                std::vector<cnn::expr::Expression>
                                *word_expressions);

    // Applies an activation function on v.
    void Activate(const cnn::expr::Expression &v,
                  cnn::expr::Expression *v_activated);

   // Performs Viterbi decoding, returns the computed score.
    double Viterbi(const std::vector<cnn::expr::Expression> &tag_weights,
                   cnn::ComputationGraph *graph,
                   std::vector<size_t> *t_sequence);

    // Computes beam (without the final tag score).
    // Training: stops early if gold falls off the beam, returns false.
    bool ComputeBeam(const std::vector<cnn::expr::Expression> &tag_weights,
                     const std::vector<size_t> &true_sequence,
                     cnn::ComputationGraph *graph,
                     std::vector<std::tuple<std::vector<size_t>,
                     cnn::expr::Expression, float, bool> > *beam);

    // Performs beam search, returns the computed score.
    double BeamSearch(const std::vector<cnn::expr::Expression> &tag_weights,
                      cnn::ComputationGraph *graph,
                      std::vector<size_t> *t_sequence);

    // Given tag weights for length-N sentence, computes log global normalizer:
    //   log SUM{ exp(score(t_1 ... t_N)) }
    void ComputeLogGlobalNormalizer(const std::vector<cnn::expr::Expression>
                                    &tag_weights,
                                    cnn::ComputationGraph *graph,
                                    cnn::expr::Expression *normalizer);

    // Given tag weights for length-N sentence, does an early update with
    // beam search. Returns sentence loss.
    double BeamEarlyUpdate(const std::vector<size_t> &t_sequence,
                         const std::vector<cnn::expr::Expression> &tag_weights,
                         cnn::ComputationGraph *graph);

    // Computes the score of the tag sequence given <tag_weights> up to the
    // desired length.
    void ComputeScore(const std::vector<cnn::expr::Expression> &tag_weights,
                      const std::vector<size_t> &t_sequence,
                      size_t desired_length, bool add_final_tag_score,
                      cnn::ComputationGraph *graph,
                      cnn::expr::Expression *score);

    // Path to the model directory.
    std::string model_directory_;

    // Path to development data.
    std::string development_path_;

    // Interval to check development accuracy.
    size_t development_interval_ = 10000;

    // Path to pre-trained word embeddings.
    std::string embedding_path_;

    // Path to a gazetteer.
    std::string gazetteer_path_;

    // Dimension of character embeddings.
    size_t dim_char_embeddings_ = 25;

    // Dimension of word embeddings.
    size_t dim_word_embeddings_ = 50;

    // Dimension of gazetteer embeddings.
    size_t dim_gazetteer_embeddings_ = 1;

    // Dimension of (word-level) LSTM outputs/hidden vectors.
    size_t dim_lstm_outputs_ = 100;

    // Dimension of the hidden layers in feed-forward networks.
    size_t dim_hidden_layers_ = 100;

    // Number of LSTM layers.
    size_t num_lstm_layers_ = 1;

    // Activation function.
    std::string activation_ = "relu";

    // Learning rate.
    double learning_rate_ = 0.0004;  // Also try 0.0004, 0.0006, etc.

    // Dropout rate.
    double dropout_rate_ = 0.4;  // Also try 0.2, 0.3, 0.4, 0.5, 0.6

    // Number of epochs (passes over the entire data).
    size_t num_epochs_ = 30;

    // Performance measure.
    std::string performance_measure_ = "acc";

    // Loss function.
    std::string loss_function_ = "local";

    // Beam size.
    size_t beam_size_ = 8;

    // Use second LSTMs (word-level)?
    bool use_second_lstms_ = false;

    // Use gazetteer embeddings?
    bool use_gazemb_ = false;

    // Gazetteer regularized training?
    bool do_gaztrain_ = false;

    // Special character for unknown characters.
    const char kUnknownChar_ = '?';  // Will never be actually used.

    // Index of kUnknownChar_ in char_index_;
    const size_t kUnknownCharIndex_ = 0;

    // Special string for unknown words.
    const std::string kUnknownWord_ = "<?>";

    // Index of kUnknownWord_ in word_index_;
    const size_t kUnknownWordIndex_ = 0;

    // Are we training the model?
    bool is_training_ = false;

    // Character index: char -> integer
    std::unordered_map<char, size_t> char_index_;

    // Character count: integer -> count of the corresponding character
    std::vector<size_t> char_count_;

    // Character string: integer -> char
    std::vector<char> char_string_;

    // Word index: string -> integer
    std::unordered_map<std::string, size_t> word_index_;

    // Word count: integer -> count of the corresponding word
    std::vector<size_t> word_count_;

    // Word string: integer -> string
    std::vector<std::string> word_string_;

    // Tag index: string -> integer
    std::unordered_map<std::string, size_t> tag_index_;

    // Tag string: integer -> string
    std::vector<std::string> tag_string_;

    // Entity type index: string -> integer
    std::unordered_map<std::string, size_t> entity_type_index_;

    // Entity type string: integer -> string
    std::unordered_map<size_t, std::string> entity_type_string_;

    // Entity-type mapping ("John F. Kennedy" -> PER)
    std::map<std::vector<size_t>, size_t> entities_;

    // Gazetteer: "John" -> {PER, ORG, MISC}
    std::unordered_map<std::string,
                       std::unordered_map<size_t, bool> > gazetteer_;

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

    // Word-level LSTM builder (forward): second level.
    cnn::LSTMBuilder word_lstm_builder_forward_second_;

    // Word-level LSTM builder (backward): second level.
    cnn::LSTMBuilder word_lstm_builder_backward_second_;

    // LSTM builder for gazetteer classification.
    cnn::LSTMBuilder gaz_lstm_builder_;

    // Character embeddings (number of character types x character dimension).
    cnn::LookupParameters* char_embeddings_ = nullptr;

    // Word embeddings (number of word types x word dimension).
    cnn::LookupParameters* word_embeddings_ = nullptr;

    // In-gazetteer embeddings (number of gazetteer label types x
    //                          gazetteer embedding dimension).
    cnn::LookupParameters* in_gazetteer_embeddings_ = nullptr;

    // Out-gazetteer embeddings (number of gazetteer label types x
    //                           gazetteer embedding dimension).
    cnn::LookupParameters* out_gazetteer_embeddings_ = nullptr;

    // Weight matrix for gazetteer training (number of entity types x
    //                            final word expression dimension from LSTMs).
    cnn::Parameters* weight_matrix_g_ = nullptr;

    // Bias for gazetteer training (number of entity types x 1).
    cnn::Parameters* bias_g_ = nullptr;

    // Weight matrix at layer 1 (hidden dimension x
    //                           final word expression dimension from LSTMs).
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
    //    cnn::expr::Expression T = parameter(graph, tag_transition_);
    //    cnn::expr::Expression T_b = select_cols(T, {b});
    //    cnn::expr::Expression score(a -> b) = pick(T_b, a);
    cnn::Parameters* tag_transition_ = nullptr;

    // Tag ending scores (number of tag types x 1).
    cnn::Parameters* tag_end_ = nullptr;
};

// A beam element is a tuple (tag_sequence, score_e, score_f, is_gold), where
// "score_e" and "score_f" are the score of the tag_sequence in
// cnn::expr::Expression and float, and is_gold is true if tag_sequence
// conforms to the gold sequence.
class CompareBeamElements {
public:
    bool operator() (std::tuple<std::vector<size_t>, cnn::expr::Expression,
                     float, bool> e1,
                     std::tuple<std::vector<size_t>, cnn::expr::Expression,
                     float, bool> e2) {
        return std::get<2>(e1) > std::get<2>(e2);
    }
};

class BeamQueue {
  public:
    BeamQueue(size_t beam_size) : beam_size_(beam_size) { }

    // Inserts a beam element.
    void Insert(const std::tuple<std::vector<size_t>, cnn::expr::Expression,
                float, bool> &element);

    // Dumps all the beam elements into a list.
    void Dump(std::vector<std::tuple<std::vector<size_t>, cnn::expr::Expression,
              float, bool> > *beam);

    // Pops the queue.
    void Pop(std::tuple<std::vector<size_t>, cnn::expr::Expression, float, bool>
             *element);

    // Is the queue empty?
    bool Empty() { return q_.empty(); }

  private:
    size_t beam_size_ = 10;

    // Min heap: q_.top() has the *worst* score in the beam.
    std::priority_queue<std::tuple<std::vector<size_t>, cnn::expr::Expression,
                                   float, bool>,
                        std::vector<std::tuple<std::vector<size_t>,
                                               cnn::expr::Expression,
                                               float, bool> >,
                        CompareBeamElements> q_;
};
