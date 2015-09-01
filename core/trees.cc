// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "trees.h"

#include <algorithm>
#include <fstream>

#include "util.h"

void Node::DeleteSelfAndDescendents(Node *node) {
    for (Node *child : node->children_) {
	DeleteSelfAndDescendents(child);
    }
    delete node;
}

bool Node::Compare(Node *node) {
    ASSERT(IsRoot() && node->IsRoot(),
	   "Comparing non-root nodes is not permitted!");
    return Compare(this, node);
}

bool Node::Compare(string node_string) {
    TreeReader tree_reader;
    Node *node = tree_reader.CreateTreeFromTreeString(node_string);
    bool is_same = Compare(node);
    node->DeleteSelfAndDescendents();
    return is_same;
}

Node *Node::Copy() {
    ASSERT(IsRoot(), "Copying a non-root node!");
    return Copy(this);
}

void Node::AppendToChildren(Node *child) {
    children_.push_back(child);
    child->parent_ = this;
    child->child_index_ = children_.size() - 1;
}

void Node::Leaves(vector<string> *terminal_strings) {
    terminal_strings->clear();
    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();
	if (node->IsPreterminal()) {
	    terminal_strings->push_back(node->terminal_string());
	}
	// Putting on the stack right-to-left means popping left-to-right.
	for (int i = node->NumChildren() - 1; i >= 0; --i) {
	    dfs_stack.push(node->Child(i));
	}
    }
}

void Node::Preterminals(vector<string> *preterminal_strings) {
    preterminal_strings->clear();
    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();
	if (node->IsPreterminal()) {
	    preterminal_strings->push_back(node->nonterminal_string());
	}
	// Putting on the stack right-to-left means popping left-to-right.
	for (int i = node->NumChildren() - 1; i >= 0; --i) {
	    dfs_stack.push(node->Child(i));
	}
    }
}

Node *Node::Child(size_t i) {
    ASSERT(i < NumChildren(), "Children index out of bound: " << i << " / "
	   << NumChildren());
    return children_[i];
}

string Node::ToString() {
    string tree_string = "";
    if (IsPreterminal()) {
	if (!IsEmpty()) {
	    // Print nothing if the node is empty.
	    tree_string =
		"(" + nonterminal_string_ + " " + terminal_string_ + ")";
	}
    } else {
	string children_string = "";
	for (size_t i = 0; i < NumChildren(); ++i) {
	    children_string += Child(i)->ToString();
	    if (i < NumChildren() - 1) children_string += " ";
	}
	tree_string += "(" + nonterminal_string_ + " " + children_string + ")";
    }
    return tree_string;
}

void Node::set_terminal_string(const string &terminal_string) {
    ASSERT(NumChildren() == 0,
	   "Assigning terminal string for a node with a child!");
    terminal_string_ = terminal_string;
}

void Node::set_terminal_number(Terminal terminal_number) {
    ASSERT(NumChildren() == 0,
	   "Assigning terminal number for a node with a child!");
    ASSERT((terminal_string_.empty() && terminal_number == -1) ||
	   (!terminal_string_.empty() && terminal_number != -1),
	   "Setting inconsistent terminal string/number!");
    terminal_number_ = terminal_number;
}

void Node::set_span(int span_begin, int span_end) {
    ASSERT(span_begin >= 0 && span_end >= 0 && span_begin <= span_end,
	   "Setting invalid span: (" << span_begin << ", " << span_end << ")");
    span_begin_ = span_begin;
    span_end_ = span_end;
}

void Node::RemoveFunctionTags() {
    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();

	// Ignore -NONE-, -LRB-, etc.
	if (node->nonterminal_string_.at(0) != '-') {
	    string new_nonterminal_string = "";
	    for (const char &c : node->nonterminal_string_) {

		// TODO: confirm that this is the correct way of stripping
		// function tags. In particular, confirm the ambiguity tag "|"
		// can be safely discarded like this.
		if (c == '-' || c == '=' || c == '|') break;
		new_nonterminal_string.push_back(c);
	    }
	    node->nonterminal_string_ = new_nonterminal_string;
	}
	for (Node *child : node->children_) { dfs_stack.push(child); }
    }
}

void Node::RemoveNullProductions() {
    vector<Node *> null_preterminals;
    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();
	if (node->IsPreterminal() &&
	    node->nonterminal_string_ == kNullPreterminalString_) {
	    null_preterminals.push_back(node);
	}
	for (Node *child : node->children_) { dfs_stack.push(child); }
    }

    // Remove the subtrees associated with null preterminal nodes.
    for (Node *null_preterminal : null_preterminals) {
	Node *subtree_neck = null_preterminal;
	while (subtree_neck->parent() != nullptr &&
	       subtree_neck->parent()->NumChildren() == 1) {
	    subtree_neck = subtree_neck->parent();
	}
	subtree_neck->parent()->DeleteChild(subtree_neck->child_index());
    }
}

void Node::AddRootNode() {
    ASSERT(IsRoot(), "Cannot add root to a non-root node");

    // Make sure the special string for the new root node is absent.
    ASSERT(!AppearsInDerivedNonterminalString(kRootInterminalString_),
	   "Special string used for the new root node is already present: "
	   << kRootInterminalString_);

    Node *self_copy = Copy();
    DeleteAllDescendents();
    nonterminal_string_ = kRootInterminalString_;
    terminal_string_ = "";
    nonterminal_number_ = -1;
    terminal_number_ = -1;
    AppendToChildren(self_copy);
    span_begin_ = self_copy->span_begin();
    span_end_ = self_copy->span_end();
}

void Node::Binarize(string binarization_method,
		    size_t vertical_markovization_order,
		    size_t horizontal_markovization_order) {
    // Make sure the special strings are absent.
    ASSERT(!AppearsInDerivedNonterminalString(kGivenString_) &&
	   !AppearsInDerivedNonterminalString(kUpString_) &&
	   !AppearsInDerivedNonterminalString(kChainString_),
	   "Special strings used for binarization already present: "
	   << kGivenString_ << ", " << kUpString_ << ", " << kChainString_);

    // Vertical Markovization.
    stack<Node *> ver_stack;  // Depth-first search (DFS)
    ver_stack.push(this);
    while (!ver_stack.empty()) {
	Node *node = ver_stack.top();
	ver_stack.pop();
	if (!node->IsRoot()) {
	    vector<string> markovized_strings;
	    util_string::split_by_string(node->parent()->nonterminal_string(),
					 kUpString_, &markovized_strings);

	    // Only collect nonterminal strings up to Markovization.
	    string decoration;
	    size_t num_vertical_labels =min(vertical_markovization_order,
					    size_t(markovized_strings.size()));
	    for (size_t i = 0; i < num_vertical_labels; ++i) {
		decoration += markovized_strings[i];
		if (i < num_vertical_labels - 1) { decoration += kUpString_; }
	    }
	    if (!decoration.empty()) {
		node->nonterminal_string_ += kUpString_ + decoration;
	    }
	}
	for (Node *child : node->children_) { ver_stack.push(child); }
    }

    // Binarization and horizontal Markovization.
    stack<Node *> hor_stack;  // Depth-first search (DFS)
    hor_stack.push(this);
    while (!hor_stack.empty()) {
	Node *node = hor_stack.top();
	hor_stack.pop();

	if (node->NumChildren() > 2) {
	    // Group together children to shift.
	    vector<Node *> shifted_children;
	    Node *unshifted_child = nullptr;
	    if (binarization_method == "right") {
		//        A
		//     / | \ \
		//    B (C  D E)
		for (size_t i = 1; i < node->NumChildren(); ++i) {
		    shifted_children.push_back(node->Child(i));
		}
		unshifted_child = node->Child(0);
	    } else if (binarization_method == "left") {
		//         A
		//      / /| \
		//    (B C D) E
		for (size_t i = 0; i < node->NumChildren() - 1; ++i) {
		    shifted_children.push_back(node->Child(i));
		}
		unshifted_child = node->Child(node->NumChildren() - 1);
	    } else {
		ASSERT(false, "Unknown binarization: " << binarization_method);
	    }

	    // Determine the label of the artificial node.
	    size_t dividing_index =
		node->nonterminal_string_.find(kGivenString_);
	    string artificial_node_label =
		node->nonterminal_string_.substr(0, dividing_index);
	    artificial_node_label += kGivenString_;  // "|" is always needed.
	    if (dividing_index != string::npos) {
		vector<string> markovized_strings;
		util_string::split_by_string(node->nonterminal_string_.substr(
						 dividing_index + 1),
					     kChainString_,
					     &markovized_strings);

		// Only collect nonterminal strings up to Markovization.
		size_t start_index = (markovized_strings.size() + 1 >=
				      horizontal_markovization_order) ?
		    markovized_strings.size() + 1  // Put +1 room for incoming.
		    - horizontal_markovization_order : 0;
		for (size_t i = start_index; i < markovized_strings.size();
		     ++i) {
		    artificial_node_label +=
			markovized_strings[i] + kChainString_;
		}
	    }
	    if (horizontal_markovization_order > 0) {
		artificial_node_label += unshifted_child->nonterminal_string_;
	    }

	    // Create and insert the artificial node.
	    Node *artificial_node = new Node(artificial_node_label, "");
	    for (Node *shifted_child : shifted_children) {
		artificial_node->AppendToChildren(shifted_child);
	    }
	    int span_begin = shifted_children[0]->span_begin();
	    int span_end = shifted_children[
		shifted_children.size() - 1]->span_end();
	    artificial_node->set_span(span_begin, span_end);

	    node->children_.resize(0);
	    if (binarization_method == "right") {
		//         A
		//        / \
		//       B  A|B
		//          /|\
		//         C D E
		node->AppendToChildren(unshifted_child);
		node->AppendToChildren(artificial_node);
	    } else if (binarization_method == "left") {
		//         A
		//        / \
		//      A|E  E
		//      /|\
		//     B C D
		node->AppendToChildren(artificial_node);
		node->AppendToChildren(unshifted_child);
	    } else {
		ASSERT(false, "Unknown binarization: " << binarization_method);
	    }
	}
	for (Node *child : node->children_) { hor_stack.push(child); }
    }
}

void Node::Debinarize() {
    stack<Node *> hor_stack;  // Depth-first search (DFS)
    hor_stack.push(this);
    while (!hor_stack.empty()) {
	Node *node = hor_stack.top();
	hor_stack.pop();
	size_t boundary_position =
	    node->nonterminal_string_.find(kGivenString_);
	if (boundary_position != string::npos) {
	    if (node->IsRoot()) {
		// For a root, simply remove special symbols. (This can happen
		// during the inference.)
		//      A|Y~Z                  A
		//      / \          =>       / \
		//     B   C                 B   C
		node->set_nonterminal_string(
		    node->nonterminal_string_.substr(0, boundary_position));
	    } else {
		if (node->child_index() > 0) {
		    // Recover from right binarization:
		    //      A                      A
		    //     /|\           =>      /| |\         (A|B~C dangling)
		    //    B C A|B~C             B C D E
		    //        / \
		    //       D   E
		    node->parent()->children_.resize(
			node->parent()->NumChildren() - 1);
		    for (Node *child : node->children_) {
			node->parent()->AppendToChildren(child);
		    }
		} else {
		    // Recover from left binarization:
		    //       A                      A
		    //      / \\           =>     /| |\       (A|E~D dangling)
		    //  A|E~D  D E               B C D E
		    //    /  \
		    //   B    C

		    vector<Node *> nonfirst_nodes;
		    for (size_t i = 1; i < node->parent()->NumChildren(); ++i) {
			nonfirst_nodes.push_back(node->parent()->Child(i));
		    }
		    node->parent()->children_.resize(0);
		    for (Node *child : node->children_) {
			node->parent()->AppendToChildren(child);
		    }
		    for (Node *child : nonfirst_nodes) {
			node->parent()->AppendToChildren(child);
		    }
		}
	    }
	}
	for (Node *child : node->children_) { hor_stack.push(child); }

	// Delete this node if it was a dangling artificial node.
	if (boundary_position != string::npos && !node->IsRoot()) {
	    delete node;
	}
    }

    // Remove vertical Markovization.
    stack<Node *> ver_stack;  // Depth-first search (DFS)
    ver_stack.push(this);
    while (!ver_stack.empty()) {
	Node *node = ver_stack.top();
	ver_stack.pop();
	size_t boundary_position =
	    node->nonterminal_string_.find(kUpString_);
	if (boundary_position != string::npos) {
	    node->set_nonterminal_string(
		node->nonterminal_string_.substr(0, boundary_position));
	}
	for (Node *child : node->children_) { ver_stack.push(child); }
    }
}

void Node::CollapseUnaryProductions() {
    // Special string used for glueing nonterminal strings should be absent.
    ASSERT(!AppearsInDerivedNonterminalString(kGlueString_),
	   "A nonterminal string contains the glue string reserved for "
	   "removing unary productions: " << kGlueString_);

    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();

	// When a node has a unary production, "jump over" that child. E.g.,
	//
	//     A            A+B             A+B
	//     |            /|\             / \
	//     B      =>   | B |     =>    |   |
	//    / \          |/ \|           |   |
	//   C   D         C   D           C   D
	while (node->NumChildren() == 1) {
	    Node *child = node->Child(0);
	    node->nonterminal_string_ +=
		kGlueString_ + child->nonterminal_string_;
	    node->terminal_string_ = child->terminal_string_;
	    node->children_.clear();
	    for (Node *grandchild : child->children_) {
		node->AppendToChildren(grandchild);
	    }
	    delete child;
	}

	for (Node *child : node->children_) { dfs_stack.push(child); }
    }
}

void Node::ExpandUnaryProductions() {
    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();
	size_t boundary_position =
	    node->nonterminal_string_.find(kGlueString_);
	if (boundary_position != string::npos) {
	    // "A+B+C" => head "A", tail "B+C"
	    string head_string = node->nonterminal_string_.substr(
		0, boundary_position);
	    string tail_string = node->nonterminal_string_.substr(
		boundary_position + kGlueString_.size());

	    // Create a new node from the collapsed node.
	    //
	    //    A+B+C                    A                  A
	    //     / \           =>       / \        =>       |
	    //    D   E                  |B+C|               B+C
	    //                           |/ \|               / \
	    //                           D   E              D   E
	    Node *expanded_node = new Node(tail_string,
					   node->terminal_string_);
	    expanded_node->set_span(node->span_begin(), node->span_end());
	    for (Node *child : node->children_) {
		expanded_node->AppendToChildren(child);
	    }
	    node->nonterminal_string_ = head_string;
	    node->terminal_string_ = "";
	    node->children_.clear();
	    node->AppendToChildren(expanded_node);
	}

	for (Node *child : node->children_) { dfs_stack.push(child); }
    }
}

void Node::ProcessToStandardForm() {
    RemoveFunctionTags();
    RemoveNullProductions();
    AddRootNode();
}

void Node::ProcessToChomskyNormalForm(string binarization_method,
				      size_t vertical_markovization_order,
				      size_t horizontal_markovization_order) {
    Binarize(binarization_method, vertical_markovization_order,
	     horizontal_markovization_order);
    CollapseUnaryProductions();
}

void Node::RecoverFromChomskyNormalForm() {
    ExpandUnaryProductions();
    Debinarize();
}

bool Node::Compare(Node *node1, Node *node2) {
    // First off, they need to have the same nonterminal strings.
    if (node1->nonterminal_string_ != node2->nonterminal_string_) {
	return false;
    }

    // If they do, they must have the same number of children.
    size_t num_children = node1->NumChildren();
    if (node2->NumChildren() != num_children) { return false; }

    if (num_children == 0) {
	// If they are both preterminals, compare their terminal strings.
	return (node1->terminal_string_ == node2->terminal_string_);
    } else {
	// If they are both interminals, compare their children nodes.
	for (size_t i = 0; i < num_children; ++i) {
	    if (!Compare(node1->Child(i), node2->Child(i))) { return false; }
	}
	return true;
    }
}

Node *Node::Copy(Node *node) {
    Node *new_node = new Node(
	node->nonterminal_string_, node->terminal_string_,
	node->nonterminal_number_, node->terminal_number_);
    new_node->span_begin_ = node->span_begin_;
    new_node->span_end_ = node->span_end_;
    for (Node *child : node->children_) {
	new_node->AppendToChildren(Copy(child));
    }
    return new_node;
}

void Node::DeleteChild(size_t i) {
    ASSERT(i < NumChildren(), "Children index out of bound!");
    children_[i]->DeleteSelfAndDescendents();
    children_.erase(children_.begin() + i);
}

void Node::DeleteAllDescendents() {
    while (NumChildren() > 0) { DeleteChild(0); }
}

bool Node::AppearsInDerivedNonterminalString(string symbol) {
    bool appear_in_nonterminal_string = false;
    stack<Node *> dfs_stack;  // Depth-first search (DFS)
    dfs_stack.push(this);
    while (!dfs_stack.empty()) {
	Node *node = dfs_stack.top();
	dfs_stack.pop();
	if (node->nonterminal_string_.find(symbol) != string::npos) {
	    appear_in_nonterminal_string = true;
	    break;
	}
	for (Node *child : node->children_) { dfs_stack.push(child); }
    }
    return appear_in_nonterminal_string;
}

void TreeSet::Clear() {
    for (Node *tree : trees_) { tree->DeleteSelfAndDescendents(); }
    trees_.resize(0);
}

TreeSet *TreeSet::Copy() {
    TreeSet *copied_trees = new TreeSet();
    for (Node *tree : trees_) {
	copied_trees->AddTree(tree->Copy());
    }
    return copied_trees;
}

void TreeSet::EvaluateAgainstGold(TreeSet *gold_trees, size_t *num_skipped,
				  double *precision, double *recall,
				  double *f1_score, double *tagging_accuracy) {
    ASSERT(NumTrees() == gold_trees->NumTrees(), NumTrees() << " trees against "
	   << gold_trees->NumTrees() << " gold trees");
    gold_trees->RecoverFromChomskyNormalForm();  // If in CNF, recover from it.

    // Define scoring details.
    unordered_map<string, bool> nonterminals_to_skip;
    nonterminals_to_skip[SPECIAL_ROOT_SYMBOL];
    unordered_map<string, vector<string> > equivalent_nonterminals;
    equivalent_nonterminals["ADVP"].push_back("PRT");
    equivalent_nonterminals["PRT"].push_back("ADVP");

    // A "bracket" is simply an interminal span.
    double num_gold_brackets = 0.0;
    double num_pred_brackets = 0.0;
    double num_correct_brackets = 0.0;
    double num_tags = 0.0;
    double num_correct_tags = 0.0;
    *num_skipped = 0;

    for (size_t tree_num = 0; tree_num < NumTrees(); ++tree_num) {
	Node *pred_tree = Tree(tree_num);
	if (pred_tree->IsEmpty()) {  // Empty predicted tree = skipped tree
	    ++(*num_skipped);
	    continue;
	}
	Node *gold_tree = gold_trees->Tree(tree_num);
	ASSERT(!gold_tree->IsEmpty(), "Empty gold tree");
	TerminalSequence pred_terminals(pred_tree);
	TerminalSequence gold_terminals(gold_tree);
	ASSERT(pred_terminals.ToString() == gold_terminals.ToString(),
	       "Terminal strings don't match: \""  << pred_terminals.ToString()
	       << "\" vs \"" << gold_terminals.ToString() << "\"");
	num_tags += pred_terminals.Length();

	// Collect all nonterminal spans in the gold tree.
	unordered_map<int, unordered_map<int, unordered_map<string, size_t> > >
	    gold_span_count;
	stack<Node *> dfs_stack;  // Depth-first search (DFS)
	dfs_stack.push(gold_tree);
	while (!dfs_stack.empty()) {
	    Node *node = dfs_stack.top();
	    dfs_stack.pop();
	    int i = node->span_begin();
	    int j = node->span_end();
	    string a_string = node->nonterminal_string();
	    if (nonterminals_to_skip.find(a_string) ==
		nonterminals_to_skip.end()) {  // Skip special symbols.
		++gold_span_count[i][j][a_string];
		if (node->IsInterminal()) { ++num_gold_brackets; }
	    }
	    for (size_t child_num = 0; child_num < node->NumChildren();
		 ++child_num) {
		dfs_stack.push(node->Child(child_num));
	    }
	}

	// Collect all nonterminal spans in the predicted tree.
	dfs_stack.push(pred_tree);  // Should already be empty.
	while (!dfs_stack.empty()) {
	    Node *node = dfs_stack.top();
	    dfs_stack.pop();
	    int i = node->span_begin();
	    int j = node->span_end();
	    string a_string = node->nonterminal_string();
	    if (nonterminals_to_skip.find(a_string) ==
		nonterminals_to_skip.end()) {  // Skip special symbols.
		if (node->IsInterminal()) { ++num_pred_brackets; }
		bool is_correct = (gold_span_count[i][j][a_string] > 0);
		if (!is_correct && equivalent_nonterminals.find(a_string)
		    != equivalent_nonterminals.end()) {
		    // Consider nonterminal equivalents.
		    for (const auto &equivalent :
			     equivalent_nonterminals[a_string]) {
			is_correct = is_correct ||
			    (gold_span_count[i][j][equivalent] > 0);
		    }
		}

		if (is_correct) {
		    // If the predicted span was in gold, mark it off correct.
		    if (node->IsInterminal()) {
			++num_correct_brackets;
		    } else {
			++num_correct_tags;
		    }
		    --gold_span_count[i][j][a_string];
		}
	    }
	    for (size_t child_num = 0; child_num < node->NumChildren();
		 ++child_num) {
		dfs_stack.push(node->Child(child_num));
	    }
	}
    }
    *precision = num_correct_brackets / num_pred_brackets * 100;
    *recall = num_correct_brackets / num_gold_brackets * 100;
    *f1_score = 2 * (*precision) * (*recall) / (*precision + *recall);
    *tagging_accuracy = num_correct_tags / num_tags * 100;
}

Node *TreeSet::Tree(size_t i) {
    ASSERT(i < NumTrees(), "Tree index out of bound: " << i << " / "
	   << NumTrees());
    return trees_[i];
}

void TreeSet::NumSymbolTypes(size_t *num_interminal_types,
			     size_t *num_preterminal_types,
			     size_t *num_terminal_types) {
    unordered_map<string, size_t> interminal_count;
    unordered_map<string, size_t> preterminal_count;
    unordered_map<string, size_t> terminal_count;
    CountTypes(&interminal_count, &preterminal_count, &terminal_count);
    *num_interminal_types = interminal_count.size();
    *num_preterminal_types = preterminal_count.size();
    *num_terminal_types = terminal_count.size();
}

size_t TreeSet::NumInterminalTypes() {
    unordered_map<string, size_t> interminal_count;  // Only use this.
    unordered_map<string, size_t> preterminal_count;
    unordered_map<string, size_t> terminal_count;
    CountTypes(&interminal_count, &preterminal_count, &terminal_count);
    return interminal_count.size();
}

size_t TreeSet::NumPreterminalTypes() {
    unordered_map<string, size_t> interminal_count;
    unordered_map<string, size_t> preterminal_count;  // Only use this.
    unordered_map<string, size_t> terminal_count;
    CountTypes(&interminal_count, &preterminal_count, &terminal_count);
    return preterminal_count.size();
}

size_t TreeSet::NumTerminalTypes() {
    unordered_map<string, size_t> interminal_count;
    unordered_map<string, size_t> preterminal_count;
    unordered_map<string, size_t> terminal_count;  // Only use this.
    CountTypes(&interminal_count, &preterminal_count, &terminal_count);
    return terminal_count.size();
}

void TreeSet::Write(string file_path) {
    ofstream file(file_path, ios::out);
    for (Node *tree : trees_) { file << tree->ToString() << endl; }
    file.close();
}

void TreeSet::ProcessToStandardForm() {
    for (Node *tree : trees_) { tree->ProcessToStandardForm(); }
}

void TreeSet::ProcessToChomskyNormalForm(
    string binarization_method, size_t vertical_markovization_order,
    size_t horizontal_markovization_order) {
    for (Node *tree : trees_) {
	tree->ProcessToChomskyNormalForm(binarization_method,
					 vertical_markovization_order,
					 horizontal_markovization_order);
    }
}

void TreeSet::RecoverFromChomskyNormalForm() {
    for (Node *tree : trees_) { tree->RecoverFromChomskyNormalForm(); }
}

void TreeSet::ReadTreesFromFile(const string &file_path) {
    ASSERT(trees_.empty(), "Trying to read trees into a non-empty tree set!");
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open tree data: " << file_path);

    TreeReader tree_reader;
    string line;
    while (file.good()) {
	getline(file, line);
	if (!line.empty()){
	    Node *tree = tree_reader.CreateTreeFromTreeString(line);
	    AddTree(tree);
	}
    }
    file.close();
}

void TreeSet::CountTypes(unordered_map<string, size_t> *interminal_count,
			 unordered_map<string, size_t> *preterminal_count,
			 unordered_map<string, size_t> *terminal_count) {
    interminal_count->clear();
    preterminal_count->clear();
    terminal_count->clear();
    for (size_t tree_index = 0; tree_index < NumTrees(); ++tree_index) {
	stack<Node *> dfs_stack;  // Depth-first search (DFS)
	Node *root = Tree(tree_index);
        dfs_stack.push(root);
	while (!dfs_stack.empty()) {
	    Node *node = dfs_stack.top();
	    dfs_stack.pop();

	    if (node->IsInterminal()) {
		++(*interminal_count)[node->nonterminal_string()];
	    } else {
		++(*preterminal_count)[node->nonterminal_string()];
		++(*terminal_count)[node->terminal_string()];
	    }

	    for (size_t i = 0; i < node->NumChildren(); ++i) {
		dfs_stack.push(node->Child(i));
	    }
	}
    }

    // Check that interminal types and preterminal types do not overlap.
    for (const auto &preterminal_pair : *preterminal_count) {
	const string &preterminal_type = preterminal_pair.first;
	ASSERT(interminal_count->find(preterminal_type) ==
	       interminal_count->end(),
	       preterminal_type << " is used as either preterminal or "
	       "interminal!");
    }
}

Node *TreeReader::CreateTreeFromTreeString(const string &tree_string) {
    vector<string> toks;
    TokenizeTreeString(tree_string, &toks);
    Node *tree = CreateTreeFromTokenSequence(toks);
    return tree;
}

Node *TreeReader::CreateTreeFromTokenSequence(const vector<string> &toks) {
    size_t num_left_parentheses = 0;
    size_t num_right_parentheses = 0;
    string error_message = "Invalid tree string: ";
    for (const string &tok : toks) { error_message += " " + tok; }

    stack<Node *> node_stack;
    size_t leaf_num = 0;  // tracks the position of leaf nodes
    for (size_t tok_index = 0; tok_index < toks.size(); ++tok_index) {
	if (toks[tok_index] == "(") {
	    // We have an opening parenthesis: begin a new subtree.
	    ++num_left_parentheses;
	    Node *node = new Node("", "");
	    node_stack.push(node);
	} else if (toks[tok_index] == ")") {
	    // We have a closing parenthesis.
	    ++num_right_parentheses;

	    // Must have something on the stack.
	    ASSERT(node_stack.size() > 0, error_message);

	    if (node_stack.size() < 2) {
		// If there is only a single node on the stack, we should have
		// reached the end of the tokens.
		ASSERT(tok_index == toks.size() - 1, error_message);

		if (node_stack.top()->IsPreterminal()) {
		    // A singleton tree must consist of a valid preterminal.
		    ASSERT(!node_stack.top()->nonterminal_string().empty() &&
			   !node_stack.top()->terminal_string().empty(),
			   error_message);
		}
		break;
	    }

	    // Otherwise pop node, make it the next child of the top.
	    Node *popped_node = node_stack.top();
	    node_stack.pop();
	    if (popped_node->IsEmpty()) {
		// If the child is empty, just remove it.
		popped_node->DeleteSelfAndDescendents();
		continue;
	    } else {
		if (node_stack.top()->IsEmpty()) {
		    // If the parent is empty, skip it.
		    Node *parent_node = node_stack.top();
		    parent_node->DeleteSelfAndDescendents();
		    node_stack.pop();
		    node_stack.push(popped_node);
		} else {
		    // If the parent is non-empty, add the child.
		    node_stack.top()->AppendToChildren(popped_node);
		    int span_begin = (node_stack.top()->span_begin() >= 0) ?
			node_stack.top()->span_begin() :
			popped_node->span_begin();
		    node_stack.top()->set_span(span_begin,
					       popped_node->span_end());
		}
	    }
	} else {
	    // We have a symbol.
	    if (node_stack.top()->IsEmpty()) {
		// We must have a nonterminal symbol.
		node_stack.top()->set_nonterminal_string(toks[tok_index]);
	    } else {
		// We must have a terminal symbol: make this a terminal symbol
		// of the node on top of the stack.
		ASSERT(node_stack.top()->terminal_string().empty(),
		       error_message);
		node_stack.top()->set_terminal_string(toks[tok_index]);
		node_stack.top()->set_span(leaf_num, leaf_num);
		++leaf_num;
	    }
	}
    }
    // There should be a single node on the stack.
    ASSERT(node_stack.size() == 1, error_message);

    // The number of parentheses should match.
    ASSERT(num_left_parentheses == num_right_parentheses, error_message);

    return node_stack.top();
}

void TreeReader::TokenizeTreeString(const string &tree_string,
				    vector<string> *toks) {
    toks->clear();
    string tok = "";

    // Are we currently building letters?
    bool building_letters = false;

    for (const char &c : tree_string) {
	if (c == '(' || c == ')') {
	    // A parenthesis is a boundary.
	    if (building_letters) {
		toks->push_back(tok);
		tok = "";
		building_letters = false;
	    }
	    toks->emplace_back(1, c);
	} else if (c != ' ' && c != '\t') {
	    // A non-empty, non-parenthesis character contributes letters.
	    building_letters = true;
	    tok += c;
	} else {
	    // An empty character (either space or tab) is a boundary.
	    if (building_letters) {
		toks->push_back(tok);
		tok = "";
		building_letters = false;
	    }
	}
    }
}

TerminalSequence::TerminalSequence(Node *tree) {
    vector<string> terminal_strings;
    vector<string> preterminal_strings;
    tree->Leaves(&terminal_strings);
    tree->Preterminals(&preterminal_strings);
    terminal_strings_ = terminal_strings;
    preterminal_strings_ = preterminal_strings;
}

string TerminalSequence::ToString() {
    string sequence_string;
    for (size_t i = 0; i < Length(); ++i) {
	sequence_string += TerminalString(i);
	if (i < Length() - 1) sequence_string += " ";
    }
    return sequence_string;
}

void TerminalSequence::AdjustPreterminalVectors() {
    preterminal_strings_.clear();
    preterminal_numbers_.clear();
    preterminal_strings_.resize(terminal_strings_.size(), "");
    preterminal_numbers_.resize(terminal_strings_.size(), -1);
}

TerminalSequences::TerminalSequences(TreeSet *trees) {
    for (size_t i = 0; i < trees->NumTrees(); ++i) {
	Node *tree = trees->Tree(i);
	TerminalSequence *terminal_sequence = new TerminalSequence(tree);
	AddSequence(terminal_sequence);
    }
}

TerminalSequences::~TerminalSequences() {
    for (TerminalSequence *terminal_sequence : terminal_sequences_) {
	delete terminal_sequence;
    }
}
