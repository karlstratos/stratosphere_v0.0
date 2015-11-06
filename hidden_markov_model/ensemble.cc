// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "../core/eigen_helper.h"
#include "../core/evaluate.h"
#include "hmm.h"

int main (int argc, char* argv[]) {
    string model_paths;
    string data_path;
    string prediction_path;
    string decoding_method = "mbr";
    bool verbose = true;

    // If appropriate, display default options and then close the program.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--help" || arg == "-h"){ display_options_and_quit = true; }
    }
    if (argc == 1 || display_options_and_quit) {
	cout << "--models [-]:        \t"
	     << "path to trained model files (separated by |:|)" << endl;
	cout << "--data [-]:        \t"
	     << "path to a (labeled) test data file" << endl;
	cout << "--pred [-]:        \t"
	     << "path to a prediction file" << endl;
	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr?" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    // Parse command line arguments.
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--models") {
	    model_paths = argv[++i];
	} else if (arg == "--data") {
	    data_path = argv[++i];
	} else if (arg == "--pred") {
	    prediction_path = argv[++i];
	} else if (arg == "--quiet" || arg == "-q") {
	    verbose = false;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    vector<string> models;
    util_string::split_by_chars(model_paths, ":", &models);
    if (models.size() == 0) { exit(0); }

    vector<HMM> hmms(models.size());
    size_t num_observations = 0;
    size_t num_states = 0;
    for (size_t i = 0; i < models.size(); ++i) {
	hmms[i].set_decoding_method(decoding_method);
	hmms[i].set_verbose(false);
	hmms[i].Load(models[i]);
	if (i == 0) {
	    num_observations = hmms[i].NumObservations();
	    num_states = hmms[i].NumStates();
	}
	ASSERT(hmms[i].NumObservations() == num_observations,
	       "Observation spaces mismatch");
	ASSERT(hmms[i].NumStates() == num_states, "State spaces mismatch");
    }

    vector<unordered_map<State, State> > permute(models.size());
    Eigen::MatrixXd O0(num_observations, num_states);  // Model 0's emission
    for (Observation x = 0; x < num_observations; ++x) {
	for (size_t h = 0; h < num_states; ++h) {
	    string x_string = hmms[0].GetObservationString(x);
	    string h_string = hmms[0].GetStateString(h);
	    O0(x, h) = hmms[0].EmissionProbability(h_string, x_string);
	}
    }

    for (size_t model_num = 1; model_num < models.size(); ++model_num) {
	Eigen::MatrixXd O(num_observations, num_states);
	for (Observation x = 0; x < num_observations; ++x) {
	    for (State h = 0; h < num_states; ++h) {
		string x_string = hmms[model_num].GetObservationString(x);
		string h_string = hmms[model_num].GetStateString(h);
		O(x, h) = hmms[model_num].EmissionProbability(h_string,
							      x_string);
	    }
	}
	vector<bool> selected(num_states);
	for (size_t h = 0; h < num_states; ++h) {
	    double min_l2 = numeric_limits<double>::infinity();
	    State closest_h0 = 0;
	    for (size_t h0 = 0; h0 < num_states; ++h0) {
		if (!selected[h0]) {
		    Eigen::VectorXd diff = O0.col(h0) - O.col(h);
		    double l2 = diff.squaredNorm();
		    if (l2 < min_l2) {
			min_l2 = l2;
			closest_h0 = h0;
		    }
		}
	    }
	    selected[closest_h0] = true;
	    permute[model_num][h] = closest_h0;

	    if (verbose) {
		cout << hmms[model_num].GetStateString(h) << " ---> "
		     << hmms[0].GetStateString(closest_h0) << endl;
	    }
	}
	if (verbose) { cout << endl; }
    }

    // Load the test data.
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    hmms[0].ReadLines(data_path, true, &observation_string_sequences,
		      &state_string_sequences);

    // Make predictions.
    vector<vector<string> > predictions;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	size_t length = observation_string_sequences[i].size();
	vector<vector<double> > marginal;
	hmms[0].ComputeLogMarginal(observation_string_sequences[i], &marginal);

	for (size_t model_num = 1; model_num < models.size(); ++model_num) {
	    vector<vector<double> > extra_marginal;
	    hmms[model_num].ComputeLogMarginal(observation_string_sequences[i],
					       &extra_marginal);
	    for (size_t j = 0; j < length; ++j) {
		for (size_t h = 0; h < num_states; ++h) {
		    marginal[j][permute[model_num][h]] += extra_marginal[j][h];
		}
	    }
	}

	vector<string> prediction;
	for (size_t j = 0; j < length; ++j) {
	    double max_value = -numeric_limits<double>::infinity();
	    State max_state = 0;
	    for (size_t h = 0; h < num_states; ++h) {
		if (marginal[j][h] > max_value) {
		    max_value = marginal[j][h];
		    max_state = h;
		}
	    }
	    prediction.push_back(hmms[0].GetStateString(max_state));
	}
	predictions.push_back(prediction);
    }

    unordered_map<string, string> label_mapping;
    double position_accuracy;
    double sequence_accuracy;
    eval_sequential::compute_accuracy_mapping_labels(
	state_string_sequences, predictions, &position_accuracy,
	&sequence_accuracy, &label_mapping);

    if (verbose) {
	string line = util_string::printf_format(
	    "\n---EVALUATION---\n"
	    "(many-to-one) per-position: %.2f%%   per-sequence: %.2f%%",
	    position_accuracy, sequence_accuracy);
	cerr << line << endl;
    }

    // If the prediction path is not "", write predictions in that file.
    if (!prediction_path.empty()) {
	ofstream file(prediction_path, ios::out);
	ASSERT(file.is_open(), "Cannot open file: " << prediction_path);
	for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	    size_t length = observation_string_sequences[i].size();
	    for (size_t j = 0; j < length; ++j) {
		file << observation_string_sequences[i][j] << " ";
		file << state_string_sequences[i][j] << " ";
		file << label_mapping[predictions[i][j]] << endl;
	    }
	    file << endl;
	}
    }
}
