// Author: Karl Stratos (me@karlstratos.com)

#include <iostream>
#include <string>

#include "../core/pruner.h"
#include "../core/util.h"

int main (int argc, char* argv[]) {
    string tree_path;
    string prototype_path;
    string oracle_path;
    size_t num_queries = 100;
    string output_path;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--tree") {
	    tree_path = argv[++i];
	} else if (arg == "--proto") {
	    prototype_path = argv[++i];
	} else if (arg == "--oracle") {
	    oracle_path = argv[++i];
	} else if (arg == "--num-queries") {
	    num_queries = stoi(argv[++i]);
	} else if (arg == "--out") {
	    output_path = argv[++i];
	} else if (arg == "--help" || arg == "-h"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--tree [-]:        \t"
	     << "path to a tree" << endl;
	cout << "--proto [-]:        \t"
	     << "path to a list of labeled prototypes" << endl;
	cout << "--oracle [-]:        \t"
	     << "path to an oracle labeler" << endl;
	cout << "--num-queries [-]:        \t"
	     << "number of queries in active learning" << endl;
	cout << "--out [-]:        \t"
	     << "output path for propagated labels" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    Pruner pruner;
    pruner.ReadTree(tree_path);
    unordered_map<string, string> proto2label;
    if (!prototype_path.empty()) {
	proto2label = pruner.ReadPrototypes(prototype_path);
    } else {
	ASSERT(!oracle_path.empty(), "Need oracle access to find prototypes");
	unordered_map<string, string> oracle = pruner.ReadOracle(oracle_path);

    }

    unordered_map<string, vector<Node *> > pure_subtrees;
    vector<pair<Node *, string> > unknown_subtrees;
    pruner.FindConsistentSubtrees(pruner.tree(), proto2label, &pure_subtrees,
				  &unknown_subtrees);

    um2v propagation = pruner.LabelConsistentSubtrees(pure_subtrees,
						      unknown_subtrees);

    // Quick sanity check on if the labeling is consistent.
    for (const auto &pair : propagation) {
	const string &propagated_label = pair.first;
	for (const string &point : pair.second) {
	    auto search = proto2label.find(point);
	    if (search != proto2label.end()) {
		ASSERT(propagated_label == search->second,
		       "Inconsistent label propagation for prototype: "
		       << point << "\n"
		       << "-Specified " << search->second << "\n"
		       << "-Propagated " << propagated_label);
	    }
	}
    }

    size_t num_covered = 0;
    for (const auto &pair : pure_subtrees) {
	for (Node *node : pair.second) {
	    num_covered += node->NumLeaves();
	}
    }
    size_t num_uncovered = 0;
    for (const auto &pair : unknown_subtrees) {
	Node *node = pair.first;
	num_uncovered += node->NumLeaves();
    }
    size_t num_labeled_points = 0;
    ofstream file;
    if (!output_path.empty()) { file.open(output_path, ios::out); }
    for (const auto &pair : propagation) {
	const string &label = pair.first;
	for (const string &point : pair.second) {
	    if (file.is_open()) {
		file << label << "\t" << point << endl;
	    }
	    ++num_labeled_points;
	}
    }

    cout << "---------" << endl;
    cout << "| Stats |" << endl;
    cout << "---------" << endl;
    cout << "  # leaves:               " << pruner.tree()->NumLeaves() << endl;
    cout << "  # prototypes:           " << proto2label.size() << endl;
    cout << "  # pure subtrees:        " << pure_subtrees.size() << endl;
    cout << "    - # covered leaves:   " << num_covered << endl;
    cout << "  # unknown subtrees:     " << unknown_subtrees.size() << endl;
    cout << "    - # uncovered leaves: " << num_uncovered << endl;
    cout << "  # labeled points:       " << num_labeled_points << endl;
    cout << endl;

    vector<pair<string, size_t> > clusters;
    for (const auto &pair : propagation) {
	clusters.emplace_back(pair.first, pair.second.size());
    }
    sort(clusters.begin(), clusters.end(),
	 util_misc::sort_pairs_second<string, size_t, greater<size_t> >());

    cout << "------------" << endl;
    cout << "| Clusters |" << endl;
    cout << "------------" << endl;
    for (const auto &pair : clusters) {
	double percent = double(pair.second) / pruner.tree()->NumLeaves() * 100;
	cout << "   " << pair.first << ": " << pair.second
	     << " (" << util_string::to_string_with_precision(percent, 0)
	     << "%)" << endl;
    }
}
