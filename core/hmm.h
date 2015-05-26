// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// A (first-order) hidden Markove model (HMM) with observation types {0...n-1}
// and state types {0...m-1} is parametrized by
//    o: (m x n)   vector    o[h][x] = log( emission probabilty of x given h )
//    t: (m x m+1) vector  t[h1][h2] = log( transition probility from h1 to h2 )
//   pi: (m x 1)   vector      pi[h] = log( probility of initial h )
// We set m to be a "stopping state" so that t[h][m] is the log probability of
// ending in state h. Under the model:
//
// log p(x(1)...x(N), h(1)...h(N)) =  pi[h(1)] + o[h(1)][x(1)]
//                                 + sum_{i=2...N} t[h(i-1)][h(i)] o[h(i)][x(i)]
//                                 + t[h(N)][m]
#ifndef CORE_HMM_H_
#define CORE_HMM_H_

#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef size_t X;  // Observation space X
typedef size_t H;  // State space H

class HMM {
public:
    // Initializes empty.
    HMM() { }

    // Initialize from a model file.
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

    // Trains HMM parameters from a file of labeled sequences: each line has
    // observation-state string pair, an empty line indicates the end of a
    // sequence.
    void TrainSupervised(const string &labeled_data_path);

    // Trains HMM parameters from sequences of observation-state string pairs.
    void TrainSupervised(const vector<vector<pair<string, string> > >
			 &labeled_sequences);

    // Predicts state sequences for the given data.
    void Predict(const string &data_path) { Predict(data_path, ""); }

    // Predicts state sequences for the given data and writes them in a file.
    void Predict(const string &data_path, const string &prediction_path);

    // Evaluates the predicted label sequences.
    void EvaluatePrediction(const vector<vector<string> > &gold_sequences,
			    const vector<vector<string> > &pred_sequences,
			    double *position_acc, double *sequence_acc,
			    double *many2one_acc,
			    unordered_map<string, string> *many2one_map);

    // Decodes the most likely state sequence with the Viterbi algorithm.
    double Viterbi(const vector<string> &observation_sequence,
		   vector<string> *state_sequence);

    // Decodes the most likely state sequence exhaustively (for debugging).
    double ViterbiExhaustive(const vector<string> &observation_sequence,
			     vector<string> *state_sequence);

    // Decodes the state sequence with the minimum Bayes-risk objective.
    void MinimumBayesRisk(const vector<string> &observation_sequence,
			  vector<string> *state_sequence);


    // Computes log p(x) using the forward algorithm.
    double ComputeObservationLikelihoodForward(const vector<string>
					       &string_sequence);

    // Computes log p(x) using the backward algorithm.
    double ComputeObservationLikelihoodBackward(const vector<string>
						&string_sequence);

    // Computes log p(x) exhaustively.
    double ComputeObservationLikelihoodExhaustive(const vector<string>
						  &string_sequence);

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Returns the number of observation types.
    size_t NumObservations() { return x_dictionary_.size(); }

    // Returns the number of state types.
    size_t NumStates() { return h_dictionary_.size(); }

    // Returns the index corresponding to an unknown observation.
    H UnknownObservation() { return NumObservations(); }

    // Returns the index corresponding to a special stopping state.
    H StoppingState() { return NumStates(); }

    // Returns the integer ID corresponding to an observation string.
    X x_dictionary(const string &x_string);

    // Returns the original string form of an observation integer ID.
    string x_dictionary_inverse(X x);

    // Returns the integer ID corresponding to a state string.
    H h_dictionary(const string &h_string);

    // Returns the original string form of a state integer ID.
    string h_dictionary_inverse(H h);

    // Returns the emission parameter value.
    double o(H h, X x) { return o_[h][x]; }

    // Returns the transition parameter value.
    double t(H h1, H h2) { return t_[h1][h2]; }

    // Returns the prior parameter value.
    double pi(H h) { return pi_[h]; }

private:
    // Check if parameters form proper distributions.
    void CheckProperDistribution();

    // Adds the observation string to the dictionary if not already known.
    X AddObservationIfUnknown(const string &x_string);

    // Adds the state string to the dictionary if not already known.
    H AddStateIfUnknown(const string &h_string);

    // Converts an observation sequence from string to X.
    void ConvertObservation(const vector<string> &string_sequence,
			    vector<X> *x_sequence);

    // Converts an observation sequence from X to string.
    void ConvertObservation(const vector<X> &x_sequence,
			    vector<string> *string_sequence);

    // Converts a state sequence from H to string.
    void ConvertState(const vector<H> &h_sequence,
		      vector<string> *string_sequence);

    // Performs Viterbi decoding, returns max_{h} log p(x, h)
    double Viterbi(const vector<X> &x_sequence, vector<H> *h_sequence);

    // Performs exhaustive decoding, returns max_{h} log p(x, h)
    double ViterbiExhaustive(const vector<X> &x_sequence,
			     vector<H> *h_sequence);

    // Populates a vector of all state sequences.
    void PopulateAllStateSequences(const vector<H> &states, size_t length,
				   vector<vector<H> > *all_state_sequences);

    // Recovers the best state sequence from the backpointer.
    void RecoverFromBackpointer(const vector<vector<H> > &bp, H best_h_final,
				vector<H> *h_sequence);

    // Computes log p(x, h).
    double ComputeLikelihood(const vector<X> &x_sequence,
			     const vector<H> &h_sequence);

    // Computes log p(x) using the forward algorithm.
    double ComputeLikelihoodForward(const vector<X> &x_sequence);

    // Computes log p(x) using the backward algorithm.
    double ComputeLikelihoodBackward(const vector<X> &x_sequence);

    // Computes log p(x) by exhaustively summing over h.
    double ComputeLikelihoodExhaustive(const vector<X> &x_sequence);

    // Computes forward probabilities: al[i][h] = log p(x(1)...x(i), h(i)=h)
    void Forward(const vector<X> &x_sequence, vector<vector<double> > *al);

    // Computes backward probabilities: be[i][h] = log p(x(i+1)...x(N)|h(i)=h)
    void Backward(const vector<X> &x_sequence, vector<vector<double> > *be);

    // Maps an observation string to an integer ID.
    unordered_map<string, X> x_dictionary_;

    // Maps an observation integer ID to its original string form.
    unordered_map<X, string> x_dictionary_inverse_;

    // Maps a state string to an integer ID.
    unordered_map<string, H> h_dictionary_;

    // Maps a state integer ID to its original string form.
    unordered_map<H, string> h_dictionary_inverse_;

    // Emission parameters in log space.
    vector<vector<double> > o_;

    // Transition parameters in log space.
    vector<vector<double> > t_;

    // Prior parameters in log space.
    vector<double> pi_;

    // Decoding method.
    string decoding_method_ = "viterbi";

    // Print messages to stderr?
    bool verbose_ = true;
};

#endif  // CORE_HMM_H_
