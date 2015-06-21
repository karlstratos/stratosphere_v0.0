// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "hmm.h"

#include <iomanip>
#include <limits>
#include <numeric>
#include <random>

#include "../core/util.h"
#include "../core/evaluate.h"

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

    InitializeParametersRandomly(num_observations, num_states);
}

void HMM::Save(const string &model_path) {
    ofstream model_file(model_path, ios::out | ios::binary);
    util_file::binary_write_primitive(rare_cutoff_, model_file);
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
    util_file::binary_read_primitive(model_file, &rare_cutoff_);
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

void HMM::Train(const string &data_path, bool supervised, size_t num_states) {
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    bool fully_labeled;
    ReadData(data_path, &observation_string_sequences, &state_string_sequences,
	     &fully_labeled);
    if (supervised) {
	ASSERT(fully_labeled, "Data not fully labeled");
	TrainSupervised(observation_string_sequences, state_string_sequences);
    } else {
	ASSERT(num_states > 0, "Number of states needs to be > 0");
	TrainUnsupervised(observation_string_sequences, num_states);
    }
}

void HMM::InitializeParametersRandomly(size_t num_observations,
				       size_t num_states) {
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

void HMM::TrainSupervised(
    const vector<vector<string> > &observation_string_sequences,
    const vector<vector<string> > &state_string_sequences) {
    ASSERT(observation_string_sequences.size() == state_string_sequences.size(),
	   "Number of sequences not matching");
    Clear();

    vector<vector<Observation> > observation_sequences;
    ConstructObservationDictionary(observation_string_sequences,
				   &observation_sequences);
    vector<vector<State> > state_sequences(state_string_sequences.size());
    for (size_t i = 0; i < state_string_sequences.size(); ++i) {
	for (size_t j = 0; j < state_string_sequences[i].size(); ++j) {
	    State state = AddStateIfUnknown(state_string_sequences[i][j]);
	    state_sequences[i].push_back(state);
	}
    }
    TrainSupervised(observation_sequences, state_sequences);
}

void HMM::TrainUnsupervised(
    const vector<vector<string> > &observation_string_sequences,
    size_t num_states) {
    // Populate observation and state dictionaries.
    vector<vector<Observation> > observation_sequences;
    ConstructObservationDictionary(observation_string_sequences,
				   &observation_sequences);
    for (State state = 0; state < num_states; ++state) {
	AddStateIfUnknown("state" + to_string(state));
    }
    InitializeParametersRandomly(NumObservations(), NumStates());

    // Prepare data-driven model development.
    string temp_model_path = tmpnam(nullptr);
    decoding_method_ = "mbr";  // More appropriate than Viterbi for EM?
    double max_development_accuracy = 0.0;
    vector<vector<string> > development_observation_string_sequences;
    vector<vector<string> > development_state_string_sequences;
    if (!development_path_.empty()) {
	bool fully_labeled;
	ReadData(development_path_, &development_observation_string_sequences,
		 &development_state_string_sequences, &fully_labeled);
	ASSERT(fully_labeled, "Development data should be labeled");
    }
    const size_t development_interval = 10;
    const size_t max_no_improvement_count = 5;
    size_t no_improvement_count = 0;

    // Run EM iterations.
    double log_likelihood = -numeric_limits<double>::infinity();
    for (size_t iteration_num = 0; iteration_num < max_num_em_iterations_;
	 ++iteration_num) {
	// Set up expected counts.
	vector<vector<double> > emission_count(NumObservations());
	for (State state = 0; state < NumStates(); ++state) {
	    emission_count[state].resize(NumObservations(), 0.0);
	}
	vector<vector<double> > transition_count(NumStates());
	for (State state = 0; state < NumStates(); ++state) {
	    transition_count[state].resize(NumStates() + 1, 0.0);  // +stop
	}
	vector<double> prior_count(NumStates(), 0.0);

	double new_log_likelihood = 0.0;
	for (size_t i = 0; i < observation_sequences.size(); ++i) {
	    size_t length = observation_sequences[i].size();

	    vector<vector<double> > al;  // Forward probabilities.
	    Forward(observation_sequences[i], &al);

	    // Calculate the (log) probability of the observation sequence.
	    double log_probability = -numeric_limits<double>::infinity();
	    for (State state = 0; state < NumStates(); ++state) {
		log_probability = util_math::sum_logs(
		    log_probability, al[length - 1][state] +
		    transition_[state][StoppingState()]);
	    }
	    new_log_likelihood += log_probability;

	    vector<vector<double> > be;  // Backward probabilities.
	    Backward(observation_sequences[i], &be);

	    // Accumulate initial state probabilities.
	    for (State state = 0; state < NumStates(); ++state) {
		prior_count[state] +=
		    exp(al[0][state] + be[0][state] - log_probability);
	    }

	    for (size_t j = 0; j < length; ++j) {
		Observation observation = observation_sequences[i][j];

		// Accumulate emission probabilities
		for (State state = 0; state < NumStates(); ++state) {
		    emission_count[state][observation] +=
			exp(al[j][state] + be[j][state] - log_probability);
		    if (j > 0) {
			// Accumulate transition probabilities.
			for (State previous_state = 0;
			     previous_state < NumStates(); ++previous_state) {
			    transition_count[previous_state][state] +=
				exp(al[j - 1][previous_state] +
				    transition_[previous_state][state] +
				    emission_[state][observation] +
				    be[j][state] - log_probability);
			}
		    }
		}
	    }
	    // Accumulate final state probabilities.
	    for (State state = 0; state < NumStates(); ++state) {
		transition_count[state][StoppingState()] +=
		    exp(al[length - 1][state] + be[length - 1][state] -
			log_probability);
	    }
	}

	// Update parameters from the expected counts.
	for (State state = 0; state < num_states; ++state) {
	    double state_normalizer = accumulate(emission_count[state].begin(),
						 emission_count[state].end(),
						 0.0);
	    for (Observation observation = 0; observation < NumObservations();
		 ++observation) {
		emission_[state][observation] =
		    log(emission_count[state][observation]) -
		    log(state_normalizer);
	    }
	}
	for (State state1 = 0; state1 < NumStates(); ++state1) {
	    double state1_normalizer =
		accumulate(transition_count[state1].begin(),
			   transition_count[state1].end(), 0.0);
	    for (State state2 = 0; state2 < NumStates() + 1; // +stop
		 ++state2) {
		transition_[state1][state2] =
		    log(transition_count[state1][state2]) -
		    log(state1_normalizer);
	    }
	}
	double prior_normalizer = accumulate(prior_count.begin(),
					     prior_count.end(), 0.0);
	for (State state = 0; state < NumStates(); ++state) {
	    prior_[state] = log(prior_count[state]) - log(prior_normalizer);
	}
	CheckProperDistribution();

	// Must always increase likelihood.
	double likelihood_difference = new_log_likelihood - log_likelihood;
	ASSERT(likelihood_difference > -1e-5, "Likelihood decreased by: "
	       << likelihood_difference);
	log_likelihood = new_log_likelihood;

	// Stopping critera: development accuracy, or likelihood.
	if (verbose_) {
	    string line = util_string::printf_format(
		"Iteration %ld: %.2f (%.2f)   ", iteration_num + 1,
		new_log_likelihood, likelihood_difference);
	    cerr << line;  // Put a newline later.
	}
	if (!development_path_.empty()) {
	    if ((iteration_num + 1) % development_interval != 0) {
		if (verbose_) { cerr << endl; }
		continue;
	    }
	    // Check the current accuracy.
	    vector<vector<string> > predictions;
	    for (size_t i = 0;
		 i < development_observation_string_sequences.size(); ++i) {
		vector<string> prediction;
		Predict(development_observation_string_sequences[i],
			&prediction);
		predictions.push_back(prediction);
	    }

	    unordered_map<string, string> label_mapping;
	    double position_accuracy;
	    double sequence_accuracy;
	    eval_sequential::compute_accuracy_mapping_labels(
		development_state_string_sequences, predictions,
		&position_accuracy, &sequence_accuracy, &label_mapping);
	    if (verbose_) {
		cerr << util_file::get_file_name(development_path_) << ": "
		     << util_string::printf_format("%.2f%% ",
						   position_accuracy);
	    }
	    if (position_accuracy > max_development_accuracy) {
		if (verbose_) { cerr << "(new record)" << endl; }
		max_development_accuracy = position_accuracy;
		no_improvement_count = 0;  // Reset.
		Save(temp_model_path);
	    } else {
		++no_improvement_count;
		if (verbose_) {
		    cerr << "(no improvement " + to_string(no_improvement_count)
			 << ")" << endl;
		}
		if (no_improvement_count >= max_no_improvement_count) {
		    Load(temp_model_path);
		    break;
		}
	    }
	} else {  // No development data is given.
	    if (verbose_) { cerr << endl; }
	    if (likelihood_difference < 1e-10) { break; }  // Stationary point
	}
    }
    remove(temp_model_path.c_str());
}

void HMM::Predict(const string &data_path, const string &prediction_path) {
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    bool fully_labeled;
    ReadData(data_path, &observation_string_sequences, &state_string_sequences,
	     &fully_labeled);

    vector<vector<string> > predictions;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	vector<string> prediction;
	Predict(observation_string_sequences[i], &prediction);
	predictions.push_back(prediction);
    }

    unordered_map<string, string> label_mapping;
    if (fully_labeled) {
	// For simplicity, always report the many-to-one accuracy.
	double position_accuracy;
	double sequence_accuracy;
	eval_sequential::compute_accuracy_mapping_labels(
	    state_string_sequences, predictions, &position_accuracy,
	    &sequence_accuracy, &label_mapping);
	if (verbose_) {
	    string line = util_string::printf_format(
		"(many-to-one) per-position: %.2f%%   per-sequence: %.2f%%",
		position_accuracy, sequence_accuracy);
	    cerr << line << endl;
	}
    }

    if (!prediction_path.empty()) {
	ofstream file(prediction_path, ios::out);
	ASSERT(file.is_open(), "Cannot open file: " << prediction_path);
	for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	    for (size_t j = 0; j < observation_string_sequences[i].size();
		 ++j) {
		string state_string_predicted = (label_mapping.size() > 0) ?
		    label_mapping[predictions[i][j]] : predictions[i][j];
		file << observation_string_sequences[i][j] << " ";
		file << state_string_sequences[i][j] << " ";  // Optional
		file << state_string_predicted << endl;
	    }
	    file << endl;
	}
    }
}

void HMM::Predict(const vector<string> &observation_string_sequence,
		  vector<string> *state_string_sequence) {
    vector<Observation> observation_sequence;
    ConvertObservationSequence(observation_string_sequence,
			       &observation_sequence);
    vector<State> state_sequence;
    if (decoding_method_ == "viterbi") {
	Viterbi(observation_sequence, &state_sequence);
    } else if (decoding_method_ == "mbr") {
	MinimumBayesRisk(observation_sequence, &state_sequence);
    } else {
	ASSERT(false, "Unknown decoding method: " << decoding_method_);
    }
    ConvertStateSequence(state_sequence, state_string_sequence);
}

double HMM::ComputeLogProbability(
    const vector<string> &observation_string_sequence) {
    vector<Observation> observation_sequence;
    ConvertObservationSequence(observation_string_sequence,
			       &observation_sequence);
    return ComputeLogProbability(observation_sequence);
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

void HMM::TrainSupervised(
    const vector<vector<Observation> > &observation_sequences,
    const vector<vector<State> > &state_sequences) {
    // Gather co-occurrence counts.
    vector<vector<size_t> > emission_count(NumObservations());
    for (State state = 0; state < NumStates(); ++state) {
	emission_count[state].resize(NumObservations(), 0);
    }
    vector<vector<size_t> > transition_count(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	transition_count[state].resize(NumStates() + 1, 0);  // +stop
    }
    vector<size_t> prior_count(NumStates(), 0);
    for (size_t i = 0; i < observation_sequences.size(); ++i) {
	size_t length = observation_sequences[i].size();
        ASSERT(length > 0 && length == state_sequences[i].size(),
	       "Invalid sequence pair");
	State initial_state = state_sequences[i][0];
	++prior_count[initial_state];
	for (size_t j = 0; j < length; ++j) {
	    Observation observation = observation_sequences[i][j];
	    State state = state_sequences[i][j];
	    ++emission_count[state][observation];
	    if (j > 0) { ++transition_count[state_sequences[i][j - 1]][state]; }
	}
	++transition_count[state_sequences[i][length - 1]][StoppingState()];
    }

    // Set parameters.
    emission_.resize(NumStates());
    for (State state = 0; state < NumStates(); ++state) {
	size_t state_normalizer = accumulate(emission_count[state].begin(),
					     emission_count[state].end(), 0);
	emission_[state].resize(NumObservations(),
				-numeric_limits<double>::infinity());
	for (Observation observation = 0; observation < NumObservations();
	     ++observation) {
	    emission_[state][observation] =
		log(emission_count[state][observation]) - log(state_normalizer);
	}
    }
    transition_.resize(NumStates());
    for (State state1 = 0; state1 < NumStates(); ++state1) {
	size_t state1_normalizer = accumulate(transition_count[state1].begin(),
					      transition_count[state1].end(),
					      0);
	transition_[state1].resize(NumStates() + 1,  // +stop
				   -numeric_limits<double>::infinity());
	for (State state2 = 0; state2 < NumStates() + 1; ++state2) {  // +stop
	    transition_[state1][state2] =
		log(transition_count[state1][state2]) - log(state1_normalizer);
	}
    }
    size_t prior_normalizer =
	accumulate(prior_count.begin(), prior_count.end(), 0);
    prior_.resize(NumStates(), -numeric_limits<double>::infinity());
    for (State state = 0; state < NumStates(); ++state) {
	prior_[state] = log(prior_count[state]) - log(prior_normalizer);
    }
    CheckProperDistribution();
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

void HMM::ConstructObservationDictionary(
    const vector<vector<string> > observation_string_sequences,
    vector<vector<Observation> > *observation_sequences) {
    unordered_map<string, size_t> observation_string_count;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	for (size_t j = 0; j < observation_string_sequences[i].size(); ++j) {
	    ++observation_string_count[observation_string_sequences[i][j]];
	}
    }
    observation_sequences->resize(observation_string_sequences.size());
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	(*observation_sequences)[i].clear();
	for (size_t j = 0; j < observation_string_sequences[i].size(); ++j) {
	    string observation_string = observation_string_sequences[i][j];
	    if (observation_string_count[observation_string] <= rare_cutoff_) {
		observation_string = kRareObservationString_;
	    }
	    Observation observation =
		AddObservationIfUnknown(observation_string);
	    (*observation_sequences)[i].push_back(observation);
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
	Observation observation;
	if (observation_dictionary_.find(observation_string) !=
	    observation_dictionary_.end()) {  // In dictionary.
	    observation = observation_dictionary_[observation_string];
	} else if (rare_cutoff_ > 0) {  // Not in dictionary, but have rare.
	    observation = observation_dictionary_[kRareObservationString_];
	} else {  // Not in dictionary, no rare -> unknown.
	    observation = UnknownObservation();
	}
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

    // chart[i][h] = log( highest probability of the observation sequence and
    //                    any state sequence from position 1 to i, the i-th
    //                    state being h                                        )
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
    double max_log_probability = -numeric_limits<double>::infinity();
    State best_final_state = 0;
    for (State state = 0; state < NumStates(); ++state) {
	double sequence_log_probability =
	    chart[length - 1][state] + transition_[state][StoppingState()];
	if (sequence_log_probability >= max_log_probability) {
	    max_log_probability = sequence_log_probability;
	    best_final_state = state;
	}
    }
    if (debug_) {
	double answer = ViterbiExhaustive(observation_sequence, state_sequence);
	ASSERT(fabs(answer - max_log_probability) < 1e-8, "Answer: "
	       << answer << ",  Viterbi: " << max_log_probability);
    }

    // Backtrack to recover the best state sequence.
    RecoverFromBackpointer(backpointer, best_final_state, state_sequence);
    return max_log_probability;
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

double HMM::ComputeLogProbability(
    const vector<Observation> &observation_sequence) {
    size_t length = observation_sequence.size();
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    double forward_value = -numeric_limits<double>::infinity();
    for (State state = 0; state < NumStates(); ++state) {
	forward_value = util_math::sum_logs(
	    forward_value, al[length - 1][state] +
	    transition_[state][StoppingState()]);
    }

    if (debug_) {
	double answer = ComputeLogProbabilityExhaustive(observation_sequence);
	vector<vector<double> > be;
	Backward(observation_sequence, &be);
	for (size_t i = 0; i < length; ++i) {
	    double marginal_sum = -numeric_limits<double>::infinity();
	    for (State state = 0; state < NumStates(); ++state) {
		marginal_sum = util_math::sum_logs(
		    marginal_sum, al[i][state] + be[i][state]);
	    }
	    ASSERT(fabs(answer - marginal_sum) < 1e-5, "Answer: "
		   << answer << ",  marginal sum: " << marginal_sum);
	}
    }
    return forward_value;
}

double HMM::ComputeLogProbabilityExhaustive(
    const vector<Observation> &observation_sequence) {
    size_t length = observation_sequence.size();

    // Generate all possible state sequences.
    vector<vector<State> > all_state_sequences;
    vector<State> seed_states;
    PopulateAllStateSequences(seed_states, length, &all_state_sequences);

    // Sum over all state sequences.
    double sum_sequence_log_probability = -numeric_limits<double>::infinity();
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double sequence_log_probability =
	    ComputeLogProbability(observation_sequence, all_state_sequences[i]);
	sum_sequence_log_probability = util_math::sum_logs(
	    sum_sequence_log_probability, sequence_log_probability);
    }
    return sum_sequence_log_probability;
}

void HMM::Forward(const vector<Observation> &observation_sequence,
		  vector<vector<double> > *al) {
    size_t length = observation_sequence.size();

    // al[i][h] = log( probability of the observation sequence from position
    //                 1 to i, the i-th state being h                         )
    al->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*al)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base case.
    Observation initial_observation = observation_sequence[0];
    for (State state = 0; state < NumStates(); ++state) {
	double emission_value = (initial_observation == UnknownObservation()) ?
	    -log(NumObservations()) : emission_[state][initial_observation];
	(*al)[0][state] = prior_[state] + emission_value;
    }

    // Main body.
    for (size_t i = 1; i < length; ++i) {
	Observation observation = observation_sequence[i];
	for (State state = 0; state < NumStates(); ++state) {
	    double emission_value = (observation == UnknownObservation()) ?
		-log(NumObservations()) : emission_[state][observation];
	    double log_summed_probabilities =
		-numeric_limits<double>::infinity();
	    for (State previous_state = 0; previous_state < NumStates();
		 ++previous_state) {
		double log_probability = (*al)[i - 1][previous_state] +
		    transition_[previous_state][state] + emission_value;
		log_summed_probabilities =
		    util_math::sum_logs(log_summed_probabilities,
					log_probability);
	    }
	    (*al)[i][state] = log_summed_probabilities;
	}
    }
}

void HMM::Backward(const vector<Observation> &observation_sequence,
		   vector<vector<double> > *be) {
    size_t length = observation_sequence.size();

    // be[i][h] = log( probability of the observation sequence from position
    //                 i+1 to the end, conditioned on the i-th state being h )
    be->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*be)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base case.
    for (State state = 0; state < NumStates(); ++state) {
	(*be)[length - 1][state] = transition_[state][StoppingState()];
    }

    // Main body.
    for (int i = length - 2; i >= 0; --i) {
	Observation next_observation = observation_sequence[i + 1];
	for (State state = 0; state < NumStates(); ++state) {
	    double log_summed_probabilities =
		-numeric_limits<double>::infinity();
	    for (State next_state = 0; next_state < NumStates(); ++next_state) {
		double emission_value = (next_observation ==
					 UnknownObservation()) ?
		    -log(NumObservations()) :
		    emission_[next_state][next_observation];
		double log_probability = transition_[state][next_state] +
		    emission_value + (*be)[i + 1][next_state];
		log_summed_probabilities =
		    util_math::sum_logs(log_summed_probabilities,
					log_probability);
	    }
	    (*be)[i][state] = log_summed_probabilities;
	}
    }
}

void HMM::MinimumBayesRisk(const vector<Observation> &observation_sequence,
			   vector<State> *state_sequence) {
    state_sequence->clear();
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    vector<vector<double> > be;
    Backward(observation_sequence, &be);
    for (size_t i = 0; i < observation_sequence.size(); ++i) {
	double max_log_probability = -numeric_limits<double>::infinity();
	State best_state = 0;
	for (State state = 0; state < NumStates(); ++state) {
	    double log_probability = al[i][state] + be[i][state];
	    if (log_probability >= max_log_probability) {
		max_log_probability = log_probability;
		best_state = state;
	    }
	}
	state_sequence->push_back(best_state);
    }
}
