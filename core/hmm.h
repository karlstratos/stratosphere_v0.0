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
// log p(x(1)...x(N), h(1)...h(N)) =  pi[h(1)] + o[h(1)][x(1)] +
//                               sum_{i=2...N} t[h(i-1)][h(i)] + o[h(i)][x(i)] +
//                               t[h(N)][m]
#ifndef CORE_HMM_H_
#define CORE_HMM_H_

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

    // Trains HMM parameters from observation and state sequences.
    void TrainSupervised(const vector<vector<string> > &observation_sequences,
			 const vector<vector<string> > &state_sequences);

    // Predicts state sequences for the given data.
    void Predict(const string &data_path) { Predict(data_path, ""); }

    // Predicts state sequences for the given data and writes them in a file.
    void Predict(const string &data_path, const string &prediction_path);

    /*
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
    */

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Returns the number of observation types.
    size_t NumObservations() { return observation_dictionary_.size(); }

    // Returns the number of state types.
    size_t NumStates() { return state_dictionary_.size(); }

    // Returns the index corresponding to an unknown observation.
    H UnknownObservation() { return NumObservations(); }

    // Returns the index corresponding to a special stopping state.
    H StoppingState() { return NumStates(); }

    // Returns the integer ID corresponding to an observation string.
    X observation_dictionary(const string &observation_string);

    // Returns the original string form of an observation integer ID.
    string observation_dictionary_inverse(X observation);

    // Returns the integer ID corresponding to a state string.
    H state_dictionary(const string &state_string);

    // Returns the original string form of a state integer ID.
    string state_dictionary_inverse(H state);

    // Returns the emission parameter value.
    double o(H state, X observation) { return o_[state][observation]; }

    // Returns the transition parameter value.
    double t(H state1, H state2) { return t_[state1][state2]; }

    // Returns the prior parameter value.
    double pi(H state) { return pi_[state]; }

private:
    // Check if parameters form proper distributions.
    void CheckProperDistribution();

    // Reads labeled/unlabeled sequences from a text file.
    void ReadData(const string &data_path,
		  vector<vector<string> > *observation_sequences,
		  vector<vector<string> > *state_sequences,
		  bool *fully_labeled);

    // Adds the observation string to the dictionary if not already known.
    X AddObservationIfUnknown(const string &observation_string);

    // Adds the state string to the dictionary if not already known.
    H AddStateIfUnknown(const string &state_string);

    // Converts an observation sequence from string to X.
    void ConvertObservation(const vector<string> &string_sequence,
			    vector<X> *observation_sequence);

    // Converts an observation sequence from X to string.
    void ConvertObservation(const vector<X> &observation_sequence,
			    vector<string> *string_sequence);

    // Converts a state sequence from H to string.
    void ConvertState(const vector<H> &state_sequence,
		      vector<string> *string_sequence);

    // Performs Viterbi decoding, returns max_{h} log p(x, h)
    double Viterbi(const vector<X> &observation_sequence, vector<H> *state_sequence);

    // Performs exhaustive decoding, returns max_{h} log p(x, h)
    double ViterbiExhaustive(const vector<X> &observation_sequence,
			     vector<H> *state_sequence);

    // Populates a vector of all state sequences.
    void PopulateAllStateSequences(const vector<H> &states, size_t length,
				   vector<vector<H> > *all_state_sequences);

    // Recovers the best state sequence from the backpointer.
    void RecoverFromBackpointer(const vector<vector<H> > &bp, H best_state_final,
				vector<H> *state_sequence);

    // Computes log p(x, h).
    double ComputeLikelihood(const vector<X> &observation_sequence,
			     const vector<H> &state_sequence);

    // Computes log p(x) using the forward algorithm.
    double ComputeLikelihoodForward(const vector<X> &observation_sequence);

    // Computes log p(x) using the backward algorithm.
    double ComputeLikelihoodBackward(const vector<X> &observation_sequence);

    // Computes log p(x) by exhaustively summing over h.
    double ComputeLikelihoodExhaustive(const vector<X> &observation_sequence);

    // Computes forward probabilities: al[i][h] = log p(x(1)...x(i), h(i)=h)
    void Forward(const vector<X> &observation_sequence, vector<vector<double> > *al);

    // Computes backward probabilities: be[i][h] = log p(x(i+1)...x(N)|h(i)=h)
    void Backward(const vector<X> &observation_sequence, vector<vector<double> > *be);

    // Maps an observation string to an integer ID.
    unordered_map<string, X> observation_dictionary_;

    // Maps an observation integer ID to its original string form.
    unordered_map<X, string> observation_dictionary_inverse_;

    // Maps a state string to an integer ID.
    unordered_map<string, H> state_dictionary_;

    // Maps a state integer ID to its original string form.
    unordered_map<H, string> state_dictionary_inverse_;

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
