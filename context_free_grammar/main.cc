// Author: Karl Stratos (stratos@cs.columbia.edu)

#include <iostream>
#include <string>

#include "grammar.h"

int main (int argc, char* argv[]) {
    string raw_treebank_path_;
    string treebank_path_;
    string model_directory_;
    string binarization_method_ = "left";
    int horizontal_markovization_order_ = 0;
    int vertical_markovization_order_ = 0;
    bool train_ = false;
    bool use_gold_tags_ = false;
    int max_sentence_length_ = 1000;
    string prediction_path_;
    string decoding_method_ = "viterbi";

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--raw") {
	    raw_treebank_path_ = argv[++i];
	} else if (arg == "--trees") {
	    treebank_path_ = argv[++i];
	} else if (arg == "--model") {
	    model_directory_ = argv[++i];
	} else if (arg == "--bin") {
	    binarization_method_ = argv[++i];
	} else if (arg == "--hor") {
	    horizontal_markovization_order_ = stoi(argv[++i]);
	} else if (arg == "--ver") {
	    vertical_markovization_order_ = stoi(argv[++i]);
	} else if (arg == "--train") {
	    train_ = true;
	} else if (arg == "--pos") {
	    use_gold_tags_ = true;
	} else if (arg == "--len") {
	    max_sentence_length_ = stoi(argv[++i]);
	} else if (arg == "--pred") {
	    prediction_path_ = argv[++i];
	} else if (arg == "--decode") {
	    decoding_method_ = argv[++i];
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
	     << "path to model directory" << endl;
	cout << "--bin    [" << binarization_method_ << "] \t"
	     << "binarization method (left, right)" << endl;
	cout << "--hor    [" << horizontal_markovization_order_ << "]      \t"
	     << "horizontal Markovization order" << endl;
	cout << "--ver    [" << vertical_markovization_order_ << "]      \t"
	     << "vertical Markovization order" << endl;
	cout << "--train           \t"
	     << "train a parser?" << endl;
	cout << "--pos             \t"
	     << "use gold POS tags at test time?" << endl;
	cout << "--len    [" << max_sentence_length_ << "]       \t"
	     << "maximum sentence length for parsing" << endl;
	cout << "--pred   [-]        \t"
	     << "path to predicted trees" << endl;
	cout << "--decode [" << decoding_method_ << "] \t"
	     << "decoding method (viterbi, marginal)" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    // Given a raw treebank, process it to the standard format.
    if (!raw_treebank_path_.empty()) {
	ASSERT(!treebank_path_.empty(), "specify output path");
	TreeSet raw_trees(raw_treebank_path_);
	raw_trees.ProcessToStandardForm();
	raw_trees.Write(treebank_path_);
    }

    // Given a treebank and a model, either train or parse.
    if (!treebank_path_.empty() && !model_directory_.empty()) {
	Grammar grammar;
	grammar.set_model_directory(model_directory_);
	grammar.set_max_sentence_length(max_sentence_length_);
	TreeSet trees(treebank_path_);
	if (train_) {  // Train a parser on the treebank.
	    grammar.set_binarization_method(binarization_method_);
	    grammar.set_vertical_markovization_order(
		vertical_markovization_order_);
	    grammar.set_horizontal_markovization_order(
		horizontal_markovization_order_);
	    grammar.Train(&trees);
	} else {  // Parse the sentences in the treebank using the given parser.
	    grammar.Load();
	    grammar.set_use_gold_tags(use_gold_tags_);
	    grammar.set_decoding_method(decoding_method_);
	    TreeSet *predicted_trees = grammar.Parse(&trees, false);
	    if (!prediction_path_.empty()) {
		predicted_trees->Write(prediction_path_);
	    }
	    delete predicted_trees;
	}
    }
}
