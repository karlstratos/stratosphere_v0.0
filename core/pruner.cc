// Author: Karl Stratos (me@karlstratos.com)

#include "pruner.h"

#include "util.h"

void Pruner::ReadTree(const string &file_path) {
    unordered_map<string, string> bitstring;
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << file_path);
    while (file.good()) {
	vector<string> tokens;  // <cluster> <instance> <freq>
	util_file::read_line(&file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.
	ASSERT(tokens.size() == 3, "Expected <cluster> <instance> <freq>: "
	       "received " << util_string::convert_to_string(tokens));
	ASSERT(tokens[1] != kRRB_ && tokens[1] != kLRB_,
	       "Parenthesis symbols are reserved: " << kRRB_ << " " <<kLRB_);
	bitstring[tokens[0]] = tokens[1];
    }
    ReadTree(bitstring);
}

void Pruner::ReadTree(const unordered_map<string, string> &bitstring) {
    // Sort bit strings by lexicographic order.
    vector<string> list;
    for (const auto &pair : bitstring) {
	list.push_back(pair.first);
    }
    sort(list.begin(), list.end());

    // Build the tree left-to-right.
    string tree_string = "(TOP ";
    for (size_t j = 0; j < list[0].size(); ++j) {
	tree_string += "(" + list[0].substr(0, j + 1) + " ";
    }
    tree_string += ReplaceParenthesis(bitstring.at(list[0]));
    string bits_prev = list[0];

    for (size_t i = 1; i < list.size(); ++i) {
	string bits_now = list[i];

	// Close completed subtrees.
	int j = bits_prev.size() - 1;
	while (j >= 0 &&
	       (j > bits_now.size() - 1 || bits_prev[j] != bits_now[j])) {
	    tree_string += ")";
	    --j;
	}
	++j;

	// Open new subtrees.
	tree_string += " ";
	while (j < bits_now.size()) {
	    tree_string += "(" + bits_now.substr(0, j + 1) + " ";
	    ++j;
	}
	tree_string += ReplaceParenthesis(bitstring.at(bits_now));

	bits_prev = bits_now;
    }
    for (size_t j = 0; j < bits_prev.size(); ++j) { tree_string += ")"; }
    tree_string += ")";  // Close TOP.

    // Convert the tree string into a tree.
    TreeReader tree_reader;
    if (tree_ != nullptr) { tree_->DeleteSelfAndDescendents(); }
    tree_ = tree_reader.CreateTreeFromTreeString(tree_string);
}

unordered_map<string, string> Pruner::ReadPrototypes(const string &file_path) {
    unordered_map<string, string> proto2label;
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << file_path);
    while (file.good()) {
	vector<string> tokens;  // <label> <p_1> ... <p_k>
	util_file::read_line(&file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.
	for (size_t i = 1; i < tokens.size(); ++i) {
	    proto2label[tokens[i]] = tokens[0];
	}
    }
    return proto2label;
}

unordered_map<string, string> Pruner::ReadOracle(const string &file_path) {
    unordered_map<string, string> oracle;
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << file_path);
    while (file.good()) {
	vector<string> tokens;  // <C(x)> <x>
	util_file::read_line(&file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.
	oracle[tokens[1]] = tokens[0];
    }
    return oracle;
}

void Pruner::SamplePrototypes(const unordered_map<string, string> &oracle,
			      const vector<string> &leaves,
			      size_t num_proto,
			      unordered_map<string, string> *proto2label) {
    // TODO: implement prototype sampling.
}

unordered_map<string, vector<string> > Pruner::PropagateLabels(
    const unordered_map<string, vector<string> > &prototypes) {
    ASSERT(prototypes.size() > 0, "No prototype to propagate");
    ASSERT(tree_ != nullptr, "Need a tree to propagate labels");
    ASSERT(prototypes.find(kUnknown_) == prototypes.end() &&
	   prototypes.find(kConflict_) == prototypes.end(),
	   "Cannot use reserved symbols " << kUnknown_ << ", " << kConflict_);

    // Map each prototype to its label for convenience.
    unordered_map<string, string> proto2label;
    for (const auto &pair : prototypes) {
	for (const auto &proto : pair.second) {
	    proto2label[proto] = pair.first;
	}
    }

    // Find subtrees consistent with the prototypes.
    unordered_map<string, vector<Node *> > pure_subtrees;
    vector<pair<Node *, string> > unknown_subtrees;
    auto root_pair = FindConsistentSubtrees(tree_, proto2label,
					    &pure_subtrees,
					    &unknown_subtrees);
    string root_status = root_pair.first;

    // If there's no conflict, the entire tree is consistent with one label.
    if (root_status != kConflict_) {
	pure_subtrees[root_status].push_back(tree_);
    }

    // Process consistent subtrees into full labeling.
    unordered_map<string, vector<string> > propagation = \
	LabelConsistentSubtrees(pure_subtrees, unknown_subtrees);

    return propagation;
}

pair<string, string> Pruner::FindConsistentSubtrees(
    Node *node,
    const unordered_map<string, string> &proto2label,
    unordered_map<string, vector<Node *> > *pure_subtrees,
    vector<pair<Node *, string> > *unknown_subtrees) {
    ASSERT(node != nullptr, "We should never reach a null pointer!");

    // Edge case: Only happens if the whole tree is a singleton.
    if (node->NumChildren() == 1) {
	return FindConsistentSubtrees(node->Child(0), proto2label,
				      pure_subtrees, unknown_subtrees);
    }

    // Leaf case
    if (node->IsPreterminal()) {
	string leaf_string = RestoreParenthesis(node->terminal_string());
	auto search = proto2label.find(leaf_string);
	return (search != proto2label.end()) ?
	    make_pair(search->second, search->second) :  // Pure:    (c, c)
	    make_pair(kUnknown_, "");                    // Unknown: (?, "")
    }

    // At this point, the node has exacty 2 children. Do postorder traversal.
    Node *left = node->Child(0);
    Node *right = node->Child(1);
    auto left_pair = FindConsistentSubtrees(left, proto2label,
					    pure_subtrees,
					    unknown_subtrees);
    auto right_pair = FindConsistentSubtrees(right, proto2label,
					     pure_subtrees,
					     unknown_subtrees);
    string left_status = left_pair.first;
    string right_status = right_pair.first;
    string left_label = left_pair.second;
    string right_label = right_pair.second;
    string majority_label =
	(left->NumLeaves() > right->NumLeaves()) ? left_label : right_label;

    // Left/right status can be
    //
    //      !   (conflict - this guy is done)
    //      ?   (unspecified and mergeable)
    //      c   (prototype label - "y" for left and "z" for right)
    //
    // Handle each of the 9 possible cases below.
    //
    string label_status;
    string label;
    if (left_status == kConflict_ || right_status == kConflict_) {
	label_status = kConflict_;
	if (left_status == kConflict_ && right_status == kConflict_) {
	    // _______
	    // | ! ! |        ->        !
	    // |_____|
	    //
	    label = majority_label;
	} else if (left_status == kConflict_) {
	    if (right_status == kUnknown_) {
		// _______
		// | ! ? |        ->       !
		// |_____|
		//
		(*unknown_subtrees).push_back(make_pair(right, left_label));
		label = left_label;
	    } else {
		// _______
		// | ! z |        ->       !
		// |_____|
		//
		(*pure_subtrees)[right_status].push_back(right);
		label = majority_label;
	    }
	} else {
	    if (left_status == kUnknown_) {
		// _______
		// | ? ! |        ->       !
		// |_____|
		//
		(*unknown_subtrees).push_back(make_pair(left, right_label));
		label = right_label;
	    } else {
		// _______
		// | y ! |        ->       !
		// |_____|
		//
		(*pure_subtrees)[left_status].push_back(left);
		label = majority_label;
	    }
	}
    } else if (left_status == kUnknown_ && right_status == kUnknown_) {
	// _______
	// | ? ? |        ->        ?
	// |_____|
	//
	label_status = kUnknown_;
	label = "";
    } else if (left_status == kUnknown_) {
	// _______
	// | ? z |        ->        z
	// |_____|
	//
	label_status = right_status;
	label = right_label;
    } else if (right_status == kUnknown_) {
	// _______
	// | y ? |        ->        y
	// |_____|
	//
	label_status = left_status;
	label = left_label;
    } else {
	if (left_status == right_status) {
	    // _______
	    // | y y |        ->       y
	    // |_____|
	    //
	    label_status = left_status;
	    label = left_label;
	} else {
	    // _______
	    // | y z |        ->       !
	    // |_____|
	    //
	    (*pure_subtrees)[left_status].push_back(left);
	    (*pure_subtrees)[right_status].push_back(right);
	    label_status = kConflict_;
	    label = majority_label;
	}
    }

    ASSERT(!label.empty() ||
	   (left_status == kUnknown_ && right_status == kUnknown_),
	   "Node isn't labeled even though a child is labeled: "
	   << left_status << " " << right_status);

    return make_pair(label_status, label);
}

unordered_map<string, vector<string> > Pruner::LabelConsistentSubtrees(
    const unordered_map<string, vector<Node *> > &pure_subtrees,
    const vector<pair<Node *, string> > &unknown_subtrees) {
    unordered_map<string, vector<string> > propagation;

    if (true) {
	// Merge pure subtrees under the same prototype label.
	for (const auto &pair : pure_subtrees) {
	    string label = pair.first;
	    for (Node *subtree : pair.second) {
		vector<string> leaves;
		subtree->Leaves(&leaves);
		for (const string &leaf : leaves) {
		    propagation[label].push_back(RestoreParenthesis(leaf));
		}
	    }
	}

	// Assign each unknown subtree its nearest prototype label.
	for (const auto &pair : unknown_subtrees) {
	    Node *subtree = pair.first;
	    string best_label = pair.second;
	    vector<string> leaves;
	    subtree->Leaves(&leaves);
	    for (const string &leaf : leaves) {
		propagation[best_label].push_back(RestoreParenthesis(leaf));
	    }
	}
    }  // TODO: Add a split/merge procedure?

    return propagation;
}

string Pruner::ReplaceParenthesis(const string &leaf) {
    if (leaf == ")") { return kRRB_; }
    if (leaf == "(") { return kLRB_; }
    return leaf;
}

string Pruner::RestoreParenthesis(const string &leaf) {
    if (leaf == kRRB_) { return ")"; }
    if (leaf == kLRB_) { return "("; }
    return leaf;
}
