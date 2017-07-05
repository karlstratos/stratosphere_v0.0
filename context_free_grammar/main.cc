// Author: Karl Stratos (me@karlstratos.com)

#include <iostream>
#include <string>

#include "grammar.h"

int main (int argc, char* argv[]) {
    string treebank_path;
    string model_path;
    string binarization_method = "left";
    size_t horizontal_markovization_order = 0;
    size_t vertical_markovization_order = 0;
    bool train = false;
    bool use_gold_tags = false;
    size_t max_sentence_length = 1000;
    string prediction_path;
    string decoding_method = "viterbi";
    bool verbose = true;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--trees") {
	    treebank_path = argv[++i];
	} else if (arg == "--model") {
	    model_path = argv[++i];
	} else if (arg == "--bin") {
	    binarization_method = argv[++i];
	} else if (arg == "--hor") {
	    horizontal_markovization_order = stol(argv[++i]);
	} else if (arg == "--ver") {
	    vertical_markovization_order = stol(argv[++i]);
	} else if (arg == "--train") {
	    train = true;
	} else if (arg == "--pos") {
	    use_gold_tags = true;
	} else if (arg == "--len") {
	    max_sentence_length = stol(argv[++i]);
	} else if (arg == "--pred") {
	    prediction_path = argv[++i];
	} else if (arg == "--decode") {
	    decoding_method = argv[++i];
	} else if (arg == "--quiet" || arg == "-q") {
	    verbose = false;
	} else if (arg == "-h" || arg == "--help"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--raw    [-]        \t"
	     << "path to raw treebank (tree per line)" << endl;
	cout << "--trees  [-]      \t"
	     << "path to processed treebank for training/parsing" << endl;
	cout << "--model  [-]      \t"
	     << "path to model" << endl;
	cout << "--bin    [" << binarization_method << "] \t"
	     << "binarization method (left, right)" << endl;
	cout << "--hor    [" << horizontal_markovization_order << "]      \t"
	     << "horizontal Markovization order" << endl;
	cout << "--ver    [" << vertical_markovization_order << "]      \t"
	     << "vertical Markovization order" << endl;
	cout << "--train           \t"
	     << "train a parser?" << endl;
	cout << "--pos             \t"
	     << "use gold POS tags at test time?" << endl;
	cout << "--len    [" << max_sentence_length << "]       \t"
	     << "maximum sentence length for parsing" << endl;
	cout << "--pred   [-]        \t"
	     << "path to predicted trees" << endl;
	cout << "--decode [" << decoding_method << "] \t"
	     << "decoding method (viterbi, marginal)" << endl;
	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr?" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    Grammar grammar;
    grammar.set_max_sentence_length(max_sentence_length);
    grammar.set_binarization_method(binarization_method);
    grammar.set_vertical_markovization_order(vertical_markovization_order);
    grammar.set_horizontal_markovization_order(horizontal_markovization_order);
    grammar.set_use_gold_tags(use_gold_tags);
    grammar.set_decoding_method(decoding_method);
    grammar.set_verbose(verbose);

    if (train) {
	ASSERT(!model_path.empty(), "Specify the model path!");
	grammar.Train(treebank_path);
	grammar.Save(model_path);
    } else {
	grammar.Load(model_path);
	grammar.Evaluate(treebank_path, prediction_path);
    }
}
