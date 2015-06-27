// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "hmm.h"

int main (int argc, char* argv[]) {
    string model_path;
    string data_path;
    string prediction_path;
    size_t rare_cutoff = 5;
    bool train = false;
    string unsupervised_learning_method = "bw";
    size_t num_states = 0;
    size_t max_num_em_iterations = 500;
    size_t max_num_fw_iterations = 1000;
    size_t window_size = 5;
    string context_definition = "list";
    size_t num_anchor_candidates = 100;
    string development_path;
    string decoding_method = "mbr";
    bool verbose = true;
    string log_path;

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
	} else if (arg == "--unsup") {
	    unsupervised_learning_method = argv[++i];
	} else if (arg == "--states") {
	    num_states = stol(argv[++i]);
	} else if (arg == "--emiter") {
	    max_num_em_iterations = stol(argv[++i]);
	} else if (arg == "--fwiter") {
	    max_num_fw_iterations = stol(argv[++i]);
	} else if (arg == "--window") {
	    window_size = stol(argv[++i]);
	} else if (arg == "--context") {
	    context_definition = argv[++i];
	} else if (arg == "--cand") {
	    num_anchor_candidates = stol(argv[++i]);
	} else if (arg == "--dev") {
	    development_path = argv[++i];
	} else if (arg == "--decode") {
	    decoding_method = argv[++i];
	} else if (arg == "--quiet" || arg == "-q") {
	    verbose = false;
	} else if (arg == "--log") {
	    log_path = argv[++i];
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
	cout << "--unsup [" << unsupervised_learning_method << "]:     \t"
	     << "unsupervised learning method: bw, anchor"  << endl;
	cout << "--states [" << num_states << "]:       \t"
	     << "number of states" << endl;
	cout << "--emiter [" << max_num_em_iterations << "]:       \t"
	     << "maximum number of EM iterations" << endl;
	cout << "--fwiter [" << max_num_fw_iterations << "]:       \t"
	     << "maximum number of Frank-Wolfe iterations" << endl;
	cout << "--window [" << window_size << "]:     \t"
	     << "window size: \"word\"=center, \"context\"=non-center" << endl;
	cout << "--context [" << context_definition << "]: \t"
	     << "context definition: bag, list"  << endl;
	cout << "--cand [" << num_anchor_candidates << "]:   \t"
	     << "number of candidates to consider for anchors" << endl;
	cout << "--dev [-]:        \t"
	     << "path to a development data file" << endl;
	cout << "--decode [" << decoding_method << "]: \t"
	     << "decoding method: viterbi, mbr"  << endl;
	cout << "--log [-]:        \t"
	     << "path to a log file" << endl;
	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr?" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    HMM hmm;
    hmm.set_rare_cutoff(rare_cutoff);
    hmm.set_unsupervised_learning_method(unsupervised_learning_method);
    hmm.set_max_num_em_iterations(max_num_em_iterations);
    hmm.set_max_num_fw_iterations(max_num_fw_iterations);
    hmm.set_window_size(window_size);
    hmm.set_context_definition(context_definition);
    hmm.set_num_anchor_candidates(num_anchor_candidates);
    hmm.set_development_path(development_path);
    hmm.set_decoding_method(decoding_method);
    hmm.set_log_path(log_path);
    hmm.set_verbose(verbose);
    if (train) {
	if (num_states == 0) {  // Supervised learning
	    hmm.TrainSupervised(data_path);
	} else {  // Unsupervised learning
	    hmm.TrainUnsupervised(data_path, num_states);
	}
	hmm.Save(model_path);
    } else {
	hmm.Load(model_path);
	hmm.Evaluate(data_path, prediction_path);
    }
}
