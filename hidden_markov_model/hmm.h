// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// An implementation of hidden Markove models (HMMs).

#ifndef HIDDEN_MARKOV_MODEL_HMM_H_
#define HIDDEN_MARKOV_MODEL_HMM_H_

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/corpus.h"

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

    // Reads lines from a data file.
    void ReadLines(const string &file_path, bool labeled,
		   vector<vector<string> > *observation_string_sequences,
		   vector<vector<string> > *state_string_sequences);

    // Reads a line from a data file. Returns true if success, false if there is
    // no more non-empty line: while (ReadLine(...)) { /* process line */ }
    bool ReadLine(bool labeled, ifstream *file,
		  vector<string> *observation_string_sequence,
		  vector<string> *state_string_sequence);

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

    // Sets the flag for lowercasing all observation strings.
    void set_lowercase(bool lowercase) { lowercase_ = lowercase; }

    // Sets the rare cutoff.
    void set_rare_cutoff(size_t rare_cutoff) { rare_cutoff_ = rare_cutoff; }

    // Sets the unsupervised learning method.
    void set_unsupervised_learning_method(string unsupervised_learning_method) {
	unsupervised_learning_method_ = unsupervised_learning_method;
    }

    // Sets the maximum number of EM iterations.
    void set_max_num_em_iterations(size_t max_num_em_iterations) {
	max_num_em_iterations_ = max_num_em_iterations;
    }

    // Sets the maximum number of Frank-Wolfe iterations.
    void set_max_num_fw_iterations(size_t max_num_fw_iterations) {
	max_num_fw_iterations_ = max_num_fw_iterations;
    }

    // Sets the interval to check development accuracy.
    void set_development_interval(size_t development_interval) {
	development_interval_ = development_interval;
    }

    // Sets the maximum number of iterations without improvement before
    // stopping.
    void set_max_num_no_improvement(size_t max_num_no_improvement) {
	max_num_no_improvement_ = max_num_no_improvement;
    }

    // Sets the context window size.
    void set_window_size(size_t window_size) { window_size_ = window_size; }

    // Sets the context definition.
    void set_context_definition(string context_definition) {
	context_definition_ = context_definition;
    }

    // Sets the convex hull method.
    void set_convex_hull_method(string convex_hull_method) {
	convex_hull_method_ = convex_hull_method;
    }

    // Sets the context extension.
    void set_context_extension(string context_extension) {
	context_extension_ = context_extension;
    }

    // Sets the additive smoothing value.
    void set_add_smooth(double add_smooth) { add_smooth_ = add_smooth; }

    // Sets the power smoothing value.
    void set_power_smooth(double power_smooth) { power_smooth_ = power_smooth; }

    // Sets the number of anchor candidates.
    void set_num_anchor_candidates(size_t num_anchor_candidates) {
	num_anchor_candidates_ = num_anchor_candidates;
    }

    // Sets the weight for new context features.
    void set_extension_weight(double extension_weight) {
	extension_weight_ = extension_weight;
    }

    // Sets the path to development data.
    void set_development_path(string development_path) {
	development_path_ = development_path;
    }

    // Sets the path to clusters.
    void set_cluster_path(string cluster_path) {
	cluster_path_ = cluster_path;
    }

    // Sets the flag for doing post-training local search.
    void set_post_training_local_search(bool post_training_local_search) {
	post_training_local_search_ = post_training_local_search;
    }

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Sets whether to turn on the debug mode.
    void set_debug(bool debug) { debug_ = debug; }

    // Sets the path to a log file.
    void set_log_path(string log_path) { log_path_ = log_path; }

private:
    // Returns the index corresponding to an unknown observation.
    Observation UnknownObservation() { return NumObservations(); }

    // Returns the index corresponding to a special stopping state.
    State StoppingState() { return NumStates(); }

    // Runs the anchor algorithm (must already have dictionaries).
    void RunAnchor(const string &data_path,
		   const unordered_map<Observation, size_t> &observation_count);

    // Builds a convex hull of observation vectors as rows of a matrix.
    void BuildConvexHull(const string &data_path, Eigen::MatrixXd *convex_hull);

    // Extends the context space (i.e., add additional columns to the
    // observation-context co-occurrence matrix).
    void ExtendContextSpace(unordered_map<string, Context> *context_dictionary,
			    unordered_map<Context, unordered_map<Observation,
			    double> > *context_observation_count);

    // Recover emission parameters by decomposing a convex hull of observation
    // vectors.
    void RecoverEmissionFromConvexHull(const Eigen::MatrixXd &convex_hull,
				       const unordered_map<Observation, size_t>
				       &observation_count);

    // Recovers the prior and transition parameters given the emission
    // parameters.
    void RecoverPriorTransitionGivenEmission(
	const string &data_path,
	const unordered_map<Observation, size_t> &observation_count);

    // Runs the Baum-Welch algorithm (must already have dictionaries).
    // - If parameters exist, simply start from them.
    // - Otherwise, initialize them randomly.
    void RunBaumWelch(const string &data_path);

    // Initializes parameters randomly (must already have dictionaries).
    void InitializeParametersRandomly();

    // Initializes parameters from clusters (must already have dictionaries).
    // - For each state with a cluster, assign nearly all its emission
    //   probabilities to the observations in the cluster.
    // - Otherwise, assign uniformly.
    // Also estimates the transition and prior parameters from unlabeled data.
    void InitializeParametersFromClusters(const string &cluster_path,
					  const string &unlabeled_data_path);

    void InitializeParametersFromClusters(const string &cluster_path) {
	InitializeParametersFromClusters(cluster_path, "");
    }

    // Check if parameters form proper distributions.
    void CheckProperDistribution();

    // Constructs observation (and state, if labeled) dictionaries.
    void ConstructDictionaries(const string &data_path, bool labeled);

    // Constructs observation (and state, if labeled) dictionaries. Also
    // counts filtered observation types.
    void ConstructDictionaries(const string &data_path, bool labeled,
			       unordered_map<Observation, size_t>
			       *observation_count);

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

    // Anchor observations.
    vector<Observation> anchor_observations_;

    // Lowercase all observation strings?
    bool lowercase_ = false;

    // Observation types that occur <= this number in the training data are
    // considered as a single symbol (corpus::kRareString).
    size_t rare_cutoff_ = 0;

    // Unsupervised learning method.
    string unsupervised_learning_method_ = "bw";

    // Maximum number of EM iterations.
    size_t max_num_em_iterations_ = 1000;

    // Maximum number of Frank-Wolfe iterations.
    size_t max_num_fw_iterations_ = 1000;

    // Interval to check development accuracy.
    size_t development_interval_ = 10;

    // Maximum number of iterations without improvement before stopping.
    size_t max_num_no_improvement_ = 10;

    // Size of the sliding window (odd => symmetric, even => assymmetric).
    size_t window_size_ = 5;

    // Context definition.
    string context_definition_ = "list";

    // Convex hull method.
    string convex_hull_method_ = "svd";

    // Context extension.
    string context_extension_ = "";

    // Additive smoothing value.
    double add_smooth_ = 0.0;

    // Power smoothing value.
    double power_smooth_ = 0.5;

    // Number of anchor candidates (most frequent observation types).
    size_t num_anchor_candidates_ = 100;

    // Weight for new context features v: weight * ||u|| = ||v|| (l2 norm).
    double extension_weight_ = 1.0;

    // Path to development data.
    string development_path_;

    // Path to clusters.
    string cluster_path_;

    // Do post-training local search?
    bool post_training_local_search_ = true;

    // Decoding method.
    string decoding_method_ = "mbr";

    // Print messages to stderr?
    bool verbose_ = true;

    // Turn on the debug mode?
    bool debug_ = false;

    // Path to a log file.
    string log_path_;
};

#endif  // HIDDEN_MARKOV_MODEL_HMM_H_
