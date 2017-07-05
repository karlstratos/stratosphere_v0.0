// Author: Karl Stratos (me@karlstratos.com)

#include <iostream>
#include <string>

#include "../core/util.h"
#include "../core/trees.h"

int main (int argc, char* argv[]) {
    string raw_treebank_path;
    string treebank_path;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (size_t i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--raw") {
	    raw_treebank_path = argv[++i];
	} else if (arg == "--trees") {
	    treebank_path = argv[++i];
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
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    size_t num_interminal_types;
    size_t num_preterminal_types;
    size_t num_terminal_types;
    if (!raw_treebank_path.empty()) {
	TreeSet raw_trees(raw_treebank_path);
	raw_trees.NumSymbolTypes(&num_interminal_types,
				 &num_preterminal_types,
				 &num_terminal_types);
	cerr << endl << "[Raw treebank]" << endl;
	cerr << "   Path: " << raw_treebank_path << endl;
	cerr << "   " << raw_trees.NumTrees() << " trees" << endl;
	cerr << "   " << num_interminal_types << " interminal types" << endl;
	cerr << "   " << num_preterminal_types << " preterminal types" << endl;
	cerr << "   " << num_terminal_types << " terminal types" << endl;

	if (!treebank_path.empty()) {
	    raw_trees.ProcessToStandardForm();
	    raw_trees.Write(treebank_path);
	}
    }

    if (!treebank_path.empty()) {
	TreeSet trees(treebank_path);
	trees.NumSymbolTypes(&num_interminal_types,
				  &num_preterminal_types,
				  &num_terminal_types);
	cerr << endl << "[Treebank]" << endl;
	cerr << "   Path: " << treebank_path << endl;
	cerr << "   " << trees.NumTrees() << " trees" << endl;
	cerr << "   " << num_interminal_types << " interminal types" << endl;
	cerr << "   " << num_preterminal_types << " preterminal types" << endl;
	cerr << "   " << num_terminal_types << " terminal types" << endl;
    }

}
