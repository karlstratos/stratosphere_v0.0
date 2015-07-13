// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "hmm.h"

int main (int argc, char* argv[]) {
    string model_path;
    string data_path;
    string prediction_path;
    bool lowercase = false;
    size_t rare_cutoff = 5;
    bool train = false;
    string unsupervised_learning_method = "bw";
    size_t num_states = 0;
    size_t max_num_em_iterations = 1000;
    size_t max_num_fw_iterations = 1000;
    size_t development_interval = 10;
    size_t max_num_no_improvement = 10;
    size_t window_size = 5;
    string context_definition = "list";
    string convex_hull_method = "svd";
    string context_extension = "";
    double add_smooth = 10.0;
    double power_smooth = 0.5;
    size_t num_anchor_candidates = 100;
    double extension_weight = 1.0;
    string development_path;
    string cluster_path;
    bool post_training_local_search = false;
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
	} else if (arg == "--lowercase") {
	    lowercase = true;
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
	} else if (arg == "--check") {
	    development_interval = stol(argv[++i]);
	} else if (arg == "--lives") {
	    max_num_no_improvement = stol(argv[++i]);
	} else if (arg == "--window") {
	    window_size = stol(argv[++i]);
	} else if (arg == "--context") {
	    context_definition = argv[++i];
	} else if (arg == "--hull") {
	    convex_hull_method = argv[++i];
	} else if (arg == "--extend") {
	    context_extension = argv[++i];
	} else if (arg == "--add") {
	    add_smooth = stod(argv[++i]);
	} else if (arg == "--power") {
	    power_smooth = stod(argv[++i]);
	} else if (arg == "--cand") {
	    num_anchor_candidates = stol(argv[++i]);
	} else if (arg == "--extweight") {
	    extension_weight = stod(argv[++i]);
	} else if (arg == "--dev") {
	    development_path = argv[++i];
	} else if (arg == "--cluster") {
	    cluster_path = argv[++i];
	} else if (arg == "--postmortem" || arg == "-p") {
	    post_training_local_search = true;
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
	cout << "--lowercase:          \t"
	     << "lowercase all observation strings?" << endl;
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
	cout << "--check [" << development_interval << "]:        \t"
	     << "interval to check development accuracy" << endl;
	cout << "--lives [" << max_num_no_improvement << "]:       \t"
	     << "maximum number of iterations without improvement before "
	     << "stopping" << endl;
	cout << "--window [" << window_size << "]:     \t"
	     << "window size: \"word\"=center, \"context\"=non-center" << endl;
	cout << "--context [" << context_definition << "]: \t"
	     << "context definition: bag, list"  << endl;
	cout << "--hull [" << convex_hull_method << "]:     \t"
	     << "convex hull method: brown, svd, cca, rand"  << endl;
	cout << "--add [" << add_smooth << "]:          \t"
	     << "additive smoothing" << endl;
	cout << "--power [" << power_smooth << "]:    \t"
	     << "power smoothing" << endl;
	cout << "--cand [" << num_anchor_candidates << "]:   \t"
	     << "number of candidates to consider for anchors" << endl;
	cout << "--dev [-]:        \t"
	     << "path to a development data file" << endl;
	cout << "--cluster [-]:   \t"
	     << "path to clusters" << endl;
	cout << "--postmortem, -p:   \t"
	     << "do post-training local search?" << endl;
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
    hmm.set_lowercase(lowercase);
    hmm.set_rare_cutoff(rare_cutoff);
    hmm.set_unsupervised_learning_method(unsupervised_learning_method);
    hmm.set_max_num_em_iterations(max_num_em_iterations);
    hmm.set_max_num_fw_iterations(max_num_fw_iterations);
    hmm.set_development_interval(development_interval);
    hmm.set_max_num_no_improvement(max_num_no_improvement);
    hmm.set_window_size(window_size);
    hmm.set_context_definition(context_definition);
    hmm.set_convex_hull_method(convex_hull_method);
    hmm.set_context_extension(context_extension);
    hmm.set_add_smooth(add_smooth);
    hmm.set_power_smooth(power_smooth);
    hmm.set_num_anchor_candidates(num_anchor_candidates);
    hmm.set_extension_weight(extension_weight);
    hmm.set_development_path(development_path);
    hmm.set_cluster_path(cluster_path);
    hmm.set_post_training_local_search(post_training_local_search);
    hmm.set_decoding_method(decoding_method);
    hmm.set_log_path(log_path);
    hmm.set_verbose(verbose);
    if (train) {
	if (num_states == 0) {  // Supervised learning
	    hmm.TrainSupervised(data_path);
	} else {  // Unsupervised learning
	    hmm.TrainUnsupervised(data_path, num_states);
	}
	if (!development_path.empty()) {
	    hmm.Evaluate(development_path, prediction_path);
	}
	hmm.Save(model_path);
    } else {
	hmm.Load(model_path);
	hmm.Evaluate(data_path, prediction_path);
    }
}
