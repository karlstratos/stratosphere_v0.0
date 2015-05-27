// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "hmm.h"

#include <iomanip>
#include <limits>
#include <random>

#include "util.h"

void HMM::Clear() {
    observation_dictionary_.clear();
    observation_dictionary_inverse_.clear();
    state_dictionary_.clear();
    state_dictionary_inverse_.clear();
    emission_.clear();
    transition_.clear();
    prior_.clear();
}

void HMM::CreateRandomly(size_t num_observations, size_t num_states) {
    Clear();

    // Create an observation dictionary.
    for (Observation observation = 0; observation < num_observations;
	 ++observation) {
	string observation_string = "observation" + to_string(observation);
	observation_dictionary_[observation_string] = observation;
	observation_dictionary_inverse_[observation] = observation_string;
    }

    // Create a state dictionary.
    for (State state = 0; state < num_states; ++state) {
	string state_string = "state" + to_string(state);
	state_dictionary_[state_string] = state;
	state_dictionary_inverse_[state] = state_string;
    }

    random_device device;
    default_random_engine engine(device());
    normal_distribution<double> normal(0.0, 1.0);  // Standard Gaussian.

    // Generate emission parameters.
    emission_.resize(num_states);
    for (State state = 0; state < num_states; ++state) {
	emission_[state].resize(num_observations);
	double state_normalizer = 0.0;
	for (Observation observation = 0; observation < num_observations;
	     ++observation) {
	    double value = fabs(normal(engine));
	    emission_[state][observation] = value;
	    state_normalizer += value;
	}
	for (Observation observation = 0; observation < num_observations;
	     ++observation) {
	    emission_[state][observation] =
		log(emission_[state][observation]) - log(state_normalizer);
	}
    }

    // Generate transition parameters.
    transition_.resize(num_states);
    for (State state1 = 0; state1 < num_states; ++state1) {
	transition_[state1].resize(num_states + 1);  // +stop
	double state1_normalizer = 0.0;
	for (State state2 = 0; state2 < num_states + 1; ++state2) {  // +stop
	    double value = fabs(normal(engine));
	    transition_[state1][state2] = value;
	    state1_normalizer += value;
	}
	for (State state2 = 0; state2 < num_states + 1; ++state2) {  // +stop
	    transition_[state1][state2] =
		log(transition_[state1][state2]) - log(state1_normalizer);
	}
    }

    // Generate prior parameters.
    prior_.resize(num_states);
    double prior_normalizer = 0.0;
    for (State state = 0; state < num_states; ++state) {
	double value = fabs(normal(engine));
	prior_[state] = value;
	prior_normalizer += value;
    }
    for (State state = 0; state < num_states; ++state) {
	prior_[state] = log(prior_[state]) - log(prior_normalizer);
    }

    CheckProperDistribution();
}

void HMM::Save(const string &model_path) {
    ofstream model_file(model_path, ios::out | ios::binary);
    size_t num_observations = NumObservations();
    size_t num_states = NumStates();
    util_file::binary_write_primitive(num_observations, model_file);
    util_file::binary_write_primitive(num_states, model_file);
    for (const auto &observation_pair : observation_dictionary_) {
	string observation_string = observation_pair.first;
	Observation observation = observation_pair.second;
	util_file::binary_write_string(observation_string, model_file);
	util_file::binary_write_primitive(observation, model_file);
    }
    for (const auto &state_pair : state_dictionary_) {
	string state_string = state_pair.first;
	State state = state_pair.second;
	util_file::binary_write_string(state_string, model_file);
	util_file::binary_write_primitive(state, model_file);
    }
    for (State state = 0; state < emission_.size(); ++state) {
	for (Observation observation = 0; observation < emission_[state].size();
	     ++observation) {
	    double value = emission_[state][observation];
	    util_file::binary_write_primitive(value, model_file);
	}
    }
    for (size_t state1 = 0; state1 < transition_.size(); ++state1) {
	for (size_t state2 = 0; state2 < transition_[state1].size(); ++state2) {
	    double value = transition_[state1][state2];
	    util_file::binary_write_primitive(value, model_file);
	}
    }
    for (size_t state = 0; state < prior_.size(); ++state) {
	double value = prior_[state];
	util_file::binary_write_primitive(value, model_file);
    }
}

void HMM::Load(const string &model_path) {
    Clear();
    ifstream model_file(model_path, ios::in | ios::binary);
    size_t num_observations;
    size_t num_states;
    util_file::binary_read_primitive(model_file, &num_observations);
    util_file::binary_read_primitive(model_file, &num_states);
    for (size_t i = 0; i < num_observations; ++i) {
	string observation_string;
	Observation observation;
	util_file::binary_read_string(model_file, &observation_string);
	util_file::binary_read_primitive(model_file, &observation);
	observation_dictionary_[observation_string] = observation;
	observation_dictionary_inverse_[observation] = observation_string;
    }
    for (size_t i = 0; i < num_states; ++i) {
	string state_string;
	State state;
	util_file::binary_read_string(model_file, &state_string);
	util_file::binary_read_primitive(model_file, &state);
	state_dictionary_[state_string] = state;
	state_dictionary_inverse_[state] = state_string;
    }
    emission_.resize(num_states);
    for (State state = 0; state < num_states; ++state) {
	emission_[state].resize(num_observations,
				-numeric_limits<double>::infinity());
	for (Observation observation = 0; observation < num_observations;
	     ++observation) {
	    double value;
	    util_file::binary_read_primitive(model_file, &value);
	    emission_[state][observation] = value;
	}
    }
    transition_.resize(num_states);
    for (State state1 = 0; state1 < num_states; ++state1) {
	transition_[state1].resize(num_states + 1,  // +stop
				   -numeric_limits<double>::infinity());
	for (State state2 = 0; state2 < num_states + 1; ++state2) {  // +stop
	    double value;
	    util_file::binary_read_primitive(model_file, &value);
	    transition_[state1][state2] = value;
	}
    }
    prior_.resize(num_states, -numeric_limits<double>::infinity());
    for (State state = 0; state < num_states; ++state) {
	double value;
	util_file::binary_read_primitive(model_file, &value);
	prior_[state] = value;
    }
    CheckProperDistribution();
}

void HMM::TrainSupervised(const string &data_path) {
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    bool fully_labeled;
    ReadData(data_path, &observation_string_sequences, &state_string_sequences,
	     &fully_labeled);
    ASSERT(fully_labeled, "Data not fully labeled");
    TrainSupervised(observation_string_sequences, state_string_sequences);
}

void HMM::TrainSupervised(
    const vector<vector<string> > &observation_string_sequences,
    const vector<vector<string> > &state_string_sequences) {
    ASSERT(observation_string_sequences.size() == state_string_sequences.size(),
	   "Number of sequences not matching");
    Clear();
    unordered_map<State, unordered_map<Observation, size_t> > emission_count;
    unordered_map<State, unordered_map<State, size_t> > transition_count;
    unordered_map<State, size_t> prior_count;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	size_t length = observation_string_sequences[i].size();
        ASSERT(length > 0 && length == state_string_sequences[i].size(),
	       "Invalid sequence pair");
	State initial_state = AddStateIfUnknown(state_string_sequences[i][0]);
	++prior_count[initial_state];
	for (size_t j = 0; j < length; ++j) {
	    Observation observation =
		AddObservationIfUnknown(observation_string_sequences[i][j]);
	    State state2 = AddStateIfUnknown(state_string_sequences[i][j]);
	    ++emission_count[state2][observation];
	    if (j > 0) {
		State state1 =
		    state_dictionary_[state_string_sequences[i][j - 1]];
		++transition_count[state1][state2];
	    }
	}
	State final_state =
	    state_dictionary_[state_string_sequences[i][length - 1]];
	++transition_count[final_state][StoppingState()];
    }
    emission_.resize(NumStates());
    for (const auto &state_pair : emission_count) {
	State state = state_pair.first;
	size_t state_normalizer = 0;
	for (const auto &observation_pair : state_pair.second) {
	    state_normalizer += observation_pair.second;
	}
	emission_[state].resize(NumObservations(),
				-numeric_limits<double>::infinity());
	for (const auto &observation_pair : state_pair.second) {
	    emission_[state][observation_pair.first] =
		log(observation_pair.second) - log(state_normalizer);
	}
    }
    transition_.resize(NumStates());
    for (const auto &state1_pair : transition_count) {
	State state1 = state1_pair.first;
	size_t state1_normalizer = 0;
	for (const auto &state2_pair : state1_pair.second) {
	    state1_normalizer += state2_pair.second;
	}
	transition_[state1].resize(NumStates() + 1,  // +stop
				   -numeric_limits<double>::infinity());
	for (const auto &state2_pair : state1_pair.second) {
	    transition_[state1][state2_pair.first] =
		log(state2_pair.second) - log(state1_normalizer);
	}
    }
    size_t prior_normalizer = 0;
    for (const auto &state_pair : prior_count) {
	prior_normalizer += state_pair.second;
    }
    prior_.resize(NumStates(), -numeric_limits<double>::infinity());
    for (const auto &state_pair : prior_count) {
	prior_[state_pair.first] =
	    log(state_pair.second) - log(prior_normalizer);
    }
    CheckProperDistribution();
}

double HMM::Viterbi(const vector<string> &observation_string_sequence,
		    vector<string> *state_string_sequence) {
    vector<Observation> observation_sequence;
    ConvertObservationSequence(observation_string_sequence,
			       &observation_sequence);
    vector<State> state_sequence;
    double sequence_log_probability = (debug_) ?
	ViterbiExhaustive(observation_sequence, &state_sequence) :
	Viterbi(observation_sequence, &state_sequence);
    ConvertStateSequence(state_sequence, state_string_sequence);
    return sequence_log_probability;
}

double HMM::EmissionProbability(string state_string,
				string observation_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	State state = state_dictionary_[state_string];
	if (observation_dictionary_.find(observation_string) !=
	    observation_dictionary_.end()) {
	    Observation observation =
		observation_dictionary_[observation_string];
	    return exp(emission_[state][observation]);
	}
    }
    return 0.0;
}

double HMM::TransitionProbability(string state1_string, string state2_string) {
    if (state_dictionary_.find(state1_string) != state_dictionary_.end()) {
	State state1 = state_dictionary_[state1_string];
	if (state_dictionary_.find(state2_string) != state_dictionary_.end()) {
	    State state2 = state_dictionary_[state2_string];
	    return exp(transition_[state1][state2]);
	}
    }
    return 0.0;
}

double HMM::PriorProbability(string state_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	State state = state_dictionary_[state_string];
	return exp(prior_[state]);
    }
    return 0.0;
}

double HMM::StoppingProbability(string state_string) {
    if (state_dictionary_.find(state_string) != state_dictionary_.end()) {
	State state = state_dictionary_[state_string];
	return exp(transition_[state][StoppingState()]);
    }
    return 0.0;
}

void HMM::CheckProperDistribution() {
    ASSERT(NumObservations() > 0 && NumStates() > 0, "Empty dictionary?");
    for (State state = 0; state < NumStates(); ++state) {
	double state_sum = 0.0;
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    state_sum += exp(emission_[state][observation]);
	}
	ASSERT(fabs(state_sum - 1.0) < 1e-10, "Emission: " << state_sum);
    }

    for (State state1 = 0; state1 < NumStates(); ++state1) {
	double state1_sum = 0.0;
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    state1_sum += exp(transition_[state1][state2]);
	}
	ASSERT(fabs(state1_sum - 1.0) < 1e-10, "Transition: " << state1_sum);
    }

    double prior_sum = 0.0;
    for (State state = 0; state < NumStates(); ++state) {
	prior_sum += exp(prior_[state]);
    }
    ASSERT(fabs(prior_sum - 1.0) < 1e-10, "Prior: " << prior_sum);
}

void HMM::ReadData(const string &data_path,
		   vector<vector<string> > *observation_string_sequences,
		   vector<vector<string> > *state_string_sequences,
		   bool *fully_labeled) {
    (*fully_labeled) = true;
    vector<string> observation_string_sequence;
    vector<string> state_string_sequence;
    ifstream data_file(data_path, ios::in);
    while (data_file.good()) {
	vector<string> tokens;
	util_file::read_line(&data_file, &tokens);
	if (tokens.size() > 0) {
	    observation_string_sequence.push_back(tokens[0]);
	    string state_string;
	    if (tokens.size() == 1) {  // "the"
		(*fully_labeled) = false;
	    } else if (tokens.size() == 2) {  // "the DET"
		state_string = tokens[1];
	    } else {  // Invalid
		ASSERT(false, util_string::convert_to_string(tokens));
	    }
	    state_string_sequence.push_back(state_string);
	} else {
	    if (observation_string_sequence.size() > 0) {
		// End of a sequence.
		observation_string_sequences->push_back(
		    observation_string_sequence);
		state_string_sequences->push_back(state_string_sequence);
		observation_string_sequence.clear();
		state_string_sequence.clear();
	    }
	}
    }
}

Observation HMM::AddObservationIfUnknown(const string &observation_string) {
    ASSERT(!observation_string.empty(), "Adding an empty observation string!");
    if (observation_dictionary_.find(observation_string) ==
	observation_dictionary_.end()) {
	Observation observation = observation_dictionary_.size();
	observation_dictionary_[observation_string] = observation;
	observation_dictionary_inverse_[observation] = observation_string;
    }
    return observation_dictionary_[observation_string];
}

State HMM::AddStateIfUnknown(const string &state_string) {
    ASSERT(!state_string.empty(), "Adding an empty state string!");
    if (state_dictionary_.find(state_string) == state_dictionary_.end()) {
	State state = state_dictionary_.size();
	state_dictionary_[state_string] = state;
	state_dictionary_inverse_[state] = state_string;
    }
    return state_dictionary_[state_string];
}

void HMM::ConvertObservationSequence(
    const vector<string> &observation_string_sequence,
    vector<Observation> *observation_sequence) {
    ASSERT(observation_dictionary_.size() > 0, "No observation dictionary");
    observation_sequence->clear();
    for (size_t i = 0; i < observation_string_sequence.size(); ++i) {
	string observation_string = observation_string_sequence[i];
	Observation observation  =
	    (observation_dictionary_.find(observation_string) !=
	     observation_dictionary_.end()) ?
	    observation_dictionary_[observation_string] : UnknownObservation();
	observation_sequence->push_back(observation);
    }
}

void HMM::ConvertStateSequence(const vector<State> &state_sequence,
			       vector<string> *state_string_sequence) {
    ASSERT(state_dictionary_inverse_.size() > 0, "No state dictionary");
    state_string_sequence->clear();
    for (size_t i = 0; i < state_sequence.size(); ++i) {
	State state = state_sequence[i];
	ASSERT(state_dictionary_inverse_.find(state) !=
	       state_dictionary_inverse_.end(), "No state: " << state);
	string state_string = state_dictionary_inverse_[state];
	state_string_sequence->push_back(state_string);
    }
}

double HMM::Viterbi(const vector<Observation> &observation_sequence,
		    vector<State> *state_sequence) {
    size_t length = observation_sequence.size();

    // chart[i][state] = highest log probability of any sequence ending at
    //                   position i in state
    vector<vector<double> > chart(length);
    vector<vector<State> > backpointer(length);
    for (size_t i = 0; i < length; ++i) {
	chart[i].resize(NumStates(), -numeric_limits<double>::infinity());
	backpointer[i].resize(NumStates());
    }

    // Base case.
    Observation initial_observation = observation_sequence[0];
    for (State state = 0; state < NumStates(); ++state) {
	double emission_value = (initial_observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][initial_observation];
	chart[0][state] = prior_[state] + emission_value;
    }

    // Main body.
    for (size_t i = 1; i < length; ++i) {
	Observation observation = observation_sequence[i];
	for (State state = 0; state < NumStates(); ++state) {
	    double emission_value = (observation == UnknownObservation()) ?
		-log(NumObservations()) : emission_[state][observation];
	    double max_log_probability = -numeric_limits<double>::infinity();
	    State best_previous_state = 0;
	    for (State previous_state = 0; previous_state < NumStates();
		 ++previous_state) {
		double log_probability = chart[i - 1][previous_state] +
		    transition_[previous_state][state] + emission_value;
		if (log_probability >= max_log_probability) {
		    max_log_probability = log_probability;
		    best_previous_state = previous_state;
		}
	    }
	    chart[i][state] = max_log_probability;
	    backpointer[i][state] = best_previous_state;
	}
    }

    // Maximization over the final state.
    double max_sequence_log_probability = -numeric_limits<double>::infinity();
    State best_final_state = 0;
    for (State state = 0; state < NumStates(); ++state) {
	double sequence_log_probability =
	    chart[length - 1][state] + transition_[state][StoppingState()];
	if (sequence_log_probability >= max_sequence_log_probability) {
	    max_sequence_log_probability = sequence_log_probability;
	    best_final_state = state;
	}
    }

    // Backtrack to recover the best state sequence.
    RecoverFromBackpointer(backpointer, best_final_state, state_sequence);
    return max_sequence_log_probability;
}

void HMM::RecoverFromBackpointer(const vector<vector<State> > &backpointer,
				 State best_final_state,
				 vector<State> *state_sequence) {
    state_sequence->resize(backpointer.size());
    (*state_sequence)[backpointer.size() - 1] = best_final_state;
    State current_best_state = best_final_state;
    for (size_t i = backpointer.size() - 1; i > 0; --i) {
	current_best_state = backpointer.at(i)[current_best_state];
	(*state_sequence)[i - 1] = current_best_state;
    }
}

double HMM::ViterbiExhaustive(const vector<Observation> &observation_sequence,
			      vector<State> *state_sequence) {
    size_t length = observation_sequence.size();

    // Generate all possible state sequences.
    vector<vector<State> > all_state_sequences;
    vector<State> seed_states;
    PopulateAllStateSequences(seed_states, length, &all_state_sequences);

    // Enumerate each state sequence to find the best one.
    double max_sequence_log_probability = -numeric_limits<double>::infinity();
    size_t best_sequence_index = 0;
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double sequence_log_probability =
	    ComputeLogProbability(observation_sequence, all_state_sequences[i]);
	if (sequence_log_probability >= max_sequence_log_probability) {
	    max_sequence_log_probability = sequence_log_probability;
	    best_sequence_index = i;
	}
    }
    state_sequence->clear();
    for (size_t i = 0; i < length; ++i) {
	state_sequence->push_back(all_state_sequences[best_sequence_index][i]);
    }
    return max_sequence_log_probability;
}

void HMM::PopulateAllStateSequences(const vector<State> &states, size_t length,
				    vector<vector<State> >
				    *all_state_sequences) {
    if (states.size() == length) {
	all_state_sequences->push_back(states);
    } else {
	for (State state = 0; state < NumStates(); ++state) {
	    vector<State> states_appended = states;
	    states_appended.push_back(state);
	    PopulateAllStateSequences(states_appended, length,
				      all_state_sequences);
	}
    }
}

double HMM::ComputeLogProbability(
    const vector<Observation> &observation_sequence,
    const vector<State> &state_sequence) {
    size_t length = observation_sequence.size();
    ASSERT(state_sequence.size() == length, "Lengths not matching");

    Observation initial_observation = observation_sequence[0];
    State initial_state = state_sequence[0];
    double initial_emission_value =
	(initial_observation == UnknownObservation()) ?
	-log(NumObservations()) : emission_[initial_state][initial_observation];

    double sequence_log_probability =
	prior_[initial_state] + initial_emission_value;
    for (size_t i = 1; i < length; ++i) {
	Observation observation = observation_sequence[i];
	State state = state_sequence[i];
	double emission_value = (observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][observation];
	sequence_log_probability +=
	    transition_[state_sequence[i - 1]][state] + emission_value;
    }
    sequence_log_probability +=
	transition_[state_sequence[length - 1]][StoppingState()];
    return sequence_log_probability;
}
