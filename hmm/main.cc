// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "../core/hmm.h"

int main (int argc, char* argv[]) {
    string model_path;
    string data_path;
    string prediction_path;
    size_t rare_cutoff = 5;
    bool train = false;
    size_t num_states = 10;
    size_t max_num_em_iterations = 500;
    string decoding_method = "viterbi";
    bool verbose = true;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--model") {
	    model_path = argv[++i];
	} else if (arg == "--data") {
	    data_path = argv[++i];
	} else if (arg == "--pred") {
	    prediction_path = argv[++i];
	} else if (arg == "--rare") {
	    rare_cutoff = stol(argv[++i]);
	} else if (arg == "--train") {
	    train = true;
	} else if (arg == "--states") {
	    num_states = stol(argv[++i]);
	} else if (arg == "--iter") {
	    max_num_em_iterations = stol(argv[++i]);
	} else if (arg == "--decode") {
	    decoding_method = argv[++i];
	} else if (arg == "--quiet" || arg == "-q") {
	    verbose = false;
	} else if (arg == "--help" || arg == "-h"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--model [-]:        \t"
	     << "path to a model file" << endl;
	cout << "--data [-]:        \t"
	     << "path to a data file" << endl;
	cout << "--pred [-]:        \t"
	     << "path to a prediction file" << endl;
	cout << "--rare [" << rare_cutoff << "]:       \t"
	     << "word types occurring <= this are considered rare" << endl;
	cout << "--train:          \t"
	     << "train a model?" << endl;
	cout << "--states [" << num_states << "]:       \t"
	     << "number of states" << endl;
	cout << "--iter [" << max_num_em_iterations << "]:       \t"
	     << "maximum number of EM iterations" << endl;
	cout << "--decode [" << decoding_method << "]: \t"
	     << "decoding method: viterbi, mbr"  << endl;
	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr?" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    HMM hmm;
    hmm.set_rare_cutoff(rare_cutoff);
    hmm.set_max_num_em_iterations(max_num_em_iterations);
    hmm.set_decoding_method(decoding_method);
    hmm.set_verbose(verbose);
    if (train) {
	bool supervised = (num_states == 0);
	hmm.Train(data_path, supervised, num_states);
	hmm.Save(model_path);
    } else {
	hmm.Load(model_path);
	hmm.Predict(data_path, prediction_path);
    }
}
