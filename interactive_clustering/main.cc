// Author: Karl Stratos (me@karlstratos.com)

#include <iostream>
#include <string>

#include "../core/icluster.h"
#include "../core/util.h"

int main (int argc, char* argv[]) {
    string proposed_path;
    string desired_path;
    string split_method = "clean";
    size_t random_seed = 42;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--proposed") {
	    proposed_path = argv[++i];
	} else if (arg == "--desired") {
	    desired_path = argv[++i];
	} else if (arg == "--split") {
	    split_method = argv[++i];
	} else if (arg == "--seed") {
	    random_seed = stoi(argv[++i]);
	} else if (arg == "--help" || arg == "-h"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--proposed [-]:        \t"
	     << "path to a proposed clustering" << endl;
	cout << "--desired [-]:        \t"
	     << "path to a desired clustering" << endl;
	cout << "--split [" << split_method << "]:        \t"
	     << "split method" << endl;
	cout << "--seed [" << random_seed << "]:        \t"
	     << "random seed" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    // Load data points.
    Data data(proposed_path);

    // Load clusterings to fixer.
    ClusterFixer fixer;
    fixer.proposed.Read(&data, proposed_path);
    fixer.desired.Read(&data, desired_path);
    fixer.split_method = split_method;
    fixer.mt.seed(random_seed);

    cout << "---------" << endl;
    cout << "| Stats |" << endl;
    cout << "---------" << endl;
    cout << "  # data points:         " << data.NumPoints() << endl;
    cout << "  # proposed clusters:   " << fixer.proposed.NumClusters() << endl;
    cout << "  # desired clusters:    " << fixer.desired.NumClusters() << endl;
    cout << "  Overclustering error:  " << fixer.Overclustering() << endl;
    cout << "  Underclustering error: " << fixer.Underclustering() << endl;
    cout << endl;

    cout << "----------" << endl;
    cout << "| Fixing |" << endl;
    cout << "----------" << endl;
    size_t num_edits = 0;
    Request r = fixer.Fix();
    while (r != Request::none) {
	++num_edits;
	if (r == Request::split) {
	    cout << "  Split(" << fixer.cluster_to_split << "):";
	} else if (r == Request::merge) {
	    cout << "  Merge(" << fixer.clusters_to_merge.first << ", "
		 << fixer.clusters_to_merge.second << "):";
	}
	cout << "   num_clusters=" << fixer.proposed.NumClusters()
	     << "   overcluster=" << fixer.Overclustering()
	     << "   undercluster=" << fixer.Underclustering()
	     << endl;
	r = fixer.Fix();
    }
    cout << endl << "# of edits: " << num_edits << endl;

    for (const auto &c : fixer.proposed.c2x) {
	for (const auto &x: c.second) {
	    cerr << c.first << "\t" << x << endl;
	}
    }
}
