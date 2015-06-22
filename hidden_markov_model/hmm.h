// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// An implementation of hidden Markove models (HMMs).

#ifndef HIDDEN_MARKOV_MODEL_HMM_H_
#define HIDDEN_MARKOV_MODEL_HMM_H_

#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef size_t Observation;
typedef size_t State;

class HMM {
public:
    // Initializes empty.
    HMM() { }

    // Initializes from a model file.
    HMM(const string &model_path) { Load(model_path); }

    // Initializes randomly.
    HMM(size_t num_observations, size_t num_states) {
	CreateRandomly(num_observations, num_states);
    }

    // Clears the model.
    void Clear();

    // Creates a random HMM.
    void CreateRandomly(size_t num_observations, size_t num_states);

    // Saves HMM parameters to a model file.
    void Save(const string &model_path);

    // Loads HMM parameters from a model file.
    void Load(const string &model_path);

    // Trains HMM parameters from a text file of labeled sequences.
    void TrainSupervised(const string &data_path);

    // Trains HMM parameters from a text file of unlabeled sequences.
    void TrainUnsupervised(const string &data_path, size_t num_states);

    // Evaluates the sequence labeling accuracy of the HMM on a labeled dataset,
    // writes predictions in a file (if prediction_path != "").
    void Evaluate(const string &labeled_data_path,
		  const string &prediction_path);

    // Predicts a state sequence.
    void Predict(const vector<string> &observation_string_sequence,
		 vector<string> *state_string_sequence);

    // Computes the log probability of the observation string sequence.
    double ComputeLogProbability(
	const vector<string> &observation_string_sequence);

    // Returns the emission probability.
    double EmissionProbability(string state_string, string observation_string);

    // Returns the transition probability.
    double TransitionProbability(string state1_string, string state2_string);

    // Returns the prior probability.
    double PriorProbability(string state_string);

    // Returns the stopping probability.
    double StoppingProbability(string state_string);

    // Returns the number of observation types.
    size_t NumObservations() { return observation_dictionary_.size(); }

    // Returns the number of state types.
    size_t NumStates() { return state_dictionary_.size(); }

    // Returns the special string for representing rare words.
    string RareObservationString() { return kRareObservationString_; }

    // Sets the rare cutoff.
    void set_rare_cutoff(size_t rare_cutoff) { rare_cutoff_ = rare_cutoff; }

    // Sets the maximum number of EM iterations.
    void set_max_num_em_iterations(size_t max_num_em_iterations) {
	max_num_em_iterations_ = max_num_em_iterations;
    }

    // Sets the path to development data.
    void set_development_path(string development_path) {
	development_path_ = development_path;
    }

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Sets whether to turn on the debug mode.
    void set_debug(bool debug) { debug_ = debug; }

private:
    // Returns the index corresponding to an unknown observation.
    Observation UnknownObservation() { return NumObservations(); }

    // Returns the index corresponding to a special stopping state.
    State StoppingState() { return NumStates(); }

    // Initializes parameters randomly (must already have dictionaries).
    void InitializeParametersRandomly();

    // Check if parameters form proper distributions.
    void CheckProperDistribution();

    // Reads a line from a data file. Returns true if success, false if there is
    // no more non-empty line: while (ReadLine(...)) { /* process line */ }
    bool ReadLine(bool labeled, ifstream *file,
		  vector<string> *observation_string_sequence,
		  vector<string> *state_string_sequence);

    // Constructs observation (and state, if labeled) dictionaries.
    void ConstructDictionaries(const string &data_path, bool labeled);

    // Adds the observation string to the dictionary if not already known.
    Observation AddObservationIfUnknown(const string &observation_string);

    // Adds the state string to the dictionary if not already known.
    State AddStateIfUnknown(const string &state_string);

   // Converts an observation sequence from strings to indices.
    void ConvertObservationSequence(
	const vector<string> &observation_string_sequence,
	vector<Observation> *observation_sequence);

    // Converts a state sequence from strings to indices.
    void ConvertStateSequence(const vector<string> &state_string_sequence,
			      vector<State> *state_sequence);

    // Converts a state sequence from indices to strings.
    void ConvertStateSequence(const vector<State> &state_sequence,
			      vector<string> *state_string_sequence);

   // Performs Viterbi decoding, returns the computed probability.
    double Viterbi(const vector<Observation> &observation_sequence,
		   vector<State> *state_sequence);

    // Recovers the best state sequence from the backpointer.
    void RecoverFromBackpointer(const vector<vector<State> > &backpointer,
				State best_final_state,
				vector<State> *state_sequence);

   // Performs exhaustive Viterbi decoding, returns the computed probability.
    double ViterbiExhaustive(const vector<Observation> &observation_sequence,
			     vector<State> *state_sequence);

    // Populates a vector of all state sequences.
    void PopulateAllStateSequences(const vector<State> &states, size_t length,
				   vector<vector<State> > *all_state_sequences);

    // Computes the log probability of the observation/state sequence pair.
    double ComputeLogProbability(
	const vector<Observation> &observation_sequence,
	const vector<State> &state_sequence);

    // Computes the log probability of the observation sequence.
    double ComputeLogProbability(
	const vector<Observation> &observation_sequence);

    // Computes the log probability of the observation sequence exhaustively.
    double ComputeLogProbabilityExhaustive(
	const vector<Observation> &observation_sequence);

    // Computes forward probabilities:
    //    al[i][h] = log(probability of the observation sequence from position
    //                   1 to i, the i-th state being h)
    void Forward(const vector<Observation> &observation_sequence,
		 vector<vector<double> > *al);

    // Computes backward probabilities:
    //    be[i][h] = log(probability of the observation sequence from position
    //                   i+1 to the end, conditioned on the i-th state being h)
    void Backward(const vector<Observation> &observation_sequence,
		  vector<vector<double> > *be);

    // Performs minimum Bayes risk (MBR) decoding.
    void MinimumBayesRisk(const vector<Observation> &observation_sequence,
			  vector<State> *state_sequence);

    // Special string for separating observation/state in data files.
    const string kObservationStateSeperator_ = "__";

    // Special string for representing rare words.
    const string kRareObservationString_ = "<?>";

    // Maps an observation string to a unique index.
    unordered_map<string, Observation> observation_dictionary_;

    // Maps an observation index to its original string form.
    unordered_map<Observation, string> observation_dictionary_inverse_;

    // Maps a state string to a unique index.
    unordered_map<string, State> state_dictionary_;

    // Maps a state index to its original string form.
    unordered_map<State, string> state_dictionary_inverse_;

    // Emission log probabilities.
    vector<vector<double> > emission_;

    // Transition log probabilities.
    vector<vector<double> > transition_;

    // Prior log probabilities.
    vector<double> prior_;

    // Observation types that occur <= this number in the training data are
    // considered as a single symbol (kRareObservationString_).
    size_t rare_cutoff_ = 0;

    // Maximum number of EM iterations.
    size_t max_num_em_iterations_ = 500;

    // Path to development data.
    string development_path_;

    // Decoding method.
    string decoding_method_ = "mbr";

    // Print messages to stderr?
    bool verbose_ = true;

    // Turn on the debug mode?
    bool debug_ = false;
};

#endif  // HIDDEN_MARKOV_MODEL_HMM_H_
