// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "../core/util.h"
#include "../core/evaluate.h"
#include "hmm.h"

int main (int argc, char* argv[]) {
    string cluster_path;
    string test_data_path;
    string prediction_path;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--cluster") {
	    cluster_path = argv[++i];
	} else if (arg == "--test") {
	    test_data_path = argv[++i];
	} else if (arg == "--pred") {
	    prediction_path = argv[++i];
	} else if (arg == "--help" || arg == "-h"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--cluster [-]:        \t"
	     << "path to a cluster file" << endl;
	cout << "--test [-]:        \t"
	     << "path to a labeled test data file" << endl;
	cout << "--pred [-]:        \t"
	     << "path to a prediction file" << endl;
	exit(0);
    }

    if (cluster_path.empty() || test_data_path.empty()) { return 0; }

    unordered_map<string, string> observation_to_cluster;
    ifstream cluster_file(cluster_path, ios::in);
    ASSERT(cluster_file.is_open(), "Cannot open " << cluster_path);
    while (cluster_file.good()) {
	vector<string> tokens;
	util_file::read_line(&cluster_file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.

	string cluster_string = tokens[0];
	string observation_string = tokens[1];
	observation_to_cluster[observation_string] = cluster_string;
    }

    HMM hmm;
    vector<vector<string> > observation_string_sequences;
    vector<vector<string> > state_string_sequences;
    hmm.ReadLines(test_data_path, true, &observation_string_sequences,
		  &state_string_sequences);


    // Find the most frequent cluster.
    unordered_map<string, size_t> cluster_count;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	for (size_t j = 0; j < observation_string_sequences[i].size(); ++j) {
	    string observation_string = observation_string_sequences[i][j];
	    if (observation_to_cluster.find(observation_string) !=
		observation_to_cluster.end()) {
		++cluster_count[observation_to_cluster[observation_string]];
	    }
	}
    }
    size_t max_count = 0;
    string most_frequent_cluster_string;
    for (const auto &cluster_pair : cluster_count) {
	if (cluster_pair.second > max_count) {
	    max_count = cluster_pair.second;
	    most_frequent_cluster_string = cluster_pair.first;
	}
    }

    // Make predictions.
    vector<vector<string> > predictions;
    for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	vector<string> prediction;
	for (size_t j = 0; j < observation_string_sequences[i].size(); ++j) {
	    string observation_string = observation_string_sequences[i][j];
	    string cluster_string;
	    if (observation_to_cluster.find(observation_string) !=
		observation_to_cluster.end()) {
		cluster_string = observation_to_cluster[observation_string];
	    } else {
		cluster_string = most_frequent_cluster_string;
	    }
	    prediction.push_back(cluster_string);
	}
	predictions.push_back(prediction);
    }

    unordered_map<string, string> label_mapping;
    double position_accuracy;
    double sequence_accuracy;
    eval_sequential::compute_accuracy_mapping_labels(
	state_string_sequences, predictions, &position_accuracy,
	&sequence_accuracy, &label_mapping);
    string line = util_string::printf_format(
	"(many-to-one) per-position: %.2f%%   per-sequence: %.2f%%",
	position_accuracy, sequence_accuracy);
    cerr << line << endl;

    if (!prediction_path.empty()) {
	ofstream prediction_file(prediction_path, ios::out);
	ASSERT(prediction_file.is_open(), "Cannot open file: "
		   << prediction_path);
	for (size_t i = 0; i < observation_string_sequences.size(); ++i) {
	    for (size_t j = 0; j < observation_string_sequences[i].size();
		 ++j) {
		string state_string_predicted = (label_mapping.size() > 0) ?
		    label_mapping[predictions[i][j]] : predictions[i][j];
		prediction_file << observation_string_sequences[i][j] << " ";
		prediction_file << state_string_sequences[i][j] << " ";
		prediction_file << state_string_predicted << endl;
	    }
	    prediction_file << endl;
	}
    }
}
