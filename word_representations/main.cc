// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "wordrep.h"

int main (int argc, char* argv[]) {
    string corpus_path;
    string output_directory;
    bool from_scratch = false;
    size_t rare_cutoff = 10;
    bool sentence_per_line = false;
    size_t window_size = 11;
    string context_definition = "bag";
    size_t num_context_hashed = 0;  // 0 means no hashing.
    size_t dim = 500;
    string transformation_method = "power";
    double add_smooth = 10.0;
    double power_smooth = 0.5;
    string scaling_method = "cca";
    bool verbose = true;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--corpus") {
	    corpus_path = argv[++i];
	} else if (arg == "--output") {
	    output_directory = argv[++i];
	} else if (arg == "--force" || arg == "-f") {
	    from_scratch = true;
	} else if (arg == "--rare") {
	    rare_cutoff = stol(argv[++i]);
	} else if (arg == "--sentences") {
	    sentence_per_line = true;
	} else if (arg == "--window") {
	    window_size = stol(argv[++i]);
	} else if (arg == "--context") {
	    context_definition = argv[++i];
	} else if (arg == "--hash") {
	    num_context_hashed = stol(argv[++i]);
	} else if (arg == "--dim") {
	    dim = stol(argv[++i]);
	} else if (arg == "--transform") {
	    transformation_method = argv[++i];
	} else if (arg == "--add") {
	    add_smooth = stod(argv[++i]);
	} else if (arg == "--power") {
	    power_smooth = stod(argv[++i]);
	} else if (arg == "--scale") {
	    scaling_method = argv[++i];
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
	cout << "--corpus [-]:        \t"
	     << "path to a text file or a directory of text files" << endl;
	cout << "--output [-]:        \t"
	     << "path to an output directory" << endl;
	cout << "--force, -f:         \t"
	     << "forcefully recompute from scratch" << endl;
	cout << "--rare [" << rare_cutoff << "]:       \t"
	     << "word types occurring <= this are considered rare" << endl;
	cout << "--sentences:         \t"
	     << "have a sentence per line in the corpus?" << endl;
	cout << "--window [" << window_size << "]:     \t"
	     << "window size: \"word\"=center, \"context\"=non-center" << endl;
	cout << "--context [" << context_definition << "]: \t"
	     << "context definition: bag, list"  << endl;
	cout << "--hash [" << num_context_hashed << "]:          \t"
	     << "hash size for context (0 means no hashing)" << endl;
	cout << "--dim [" << dim << "]:        \t"
	     << "dimension of word vectors" << endl;
	cout << "--transform [" << transformation_method << "]: \t"
	     << "data transform: none, log, power"  << endl;
	cout << "--add [" << add_smooth << "]:          \t"
	     << "additive smoothing" << endl;
	cout << "--power [" << power_smooth << "]:    \t"
	     << "power smoothing" << endl;
	cout << "--scale [" << scaling_method << "]:    \t"
	     << "data scaling: none, ppmi, reg, cca" << endl;
	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit" << endl;
	exit(0);
    }

    /*
    // Initialize a WordRep object.
    WordRep wordrep(output_directory);
    wordrep.set_rare_cutoff(rare_cutoff);
    wordrep.set_sentence_per_line(sentence_per_line);
    wordrep.set_window_size(window_size);
    wordrep.set_context_definition(context_definition);
    wordrep.set_dim(dim);
    wordrep.set_transformation_method(transformation_method);
    wordrep.set_scaling_method(scaling_method);
    wordrep.set_num_context_hashed(num_context_hashed);
    wordrep.set_add_smooth(add_smooth);
    wordrep.set_power_smooth(power_smooth);
    wordrep.set_verbose(verbose);

    // If given a corpus, extract statistics from it.
    if (!corpus_path.empty()) {
	if (from_scratch) { wordrep.ResetOutputDirectory(); }
	wordrep.ExtractStatistics(corpus_path);
    }

    // Induce word representations from cached statistics.
    wordrep.InduceWordRepresentations();
    */
}
