// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for manipulating constituency trees.

#ifndef CORE_TREES_H_
#define CORE_TREES_H_

#include <iostream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef int Nonterminal;
typedef int Terminal;

const string SPECIAL_ROOT_SYMBOL = "TOP";

// A node is a recursive structure representing a nonterminal in a context-free
// derivation. When a node has at least one child node, it is called an
// interminal node. When a node has no children, it is called a preterminal
// node.
class Node {
public:
    // Creates a tree node with the given nonterminal and terminal information.
    // If the node is not a terminal node, "" and -1 should be passed for the
    // terminal string and number.
    Node(const string &nonterminal_string, const string &terminal_string,
	 Nonterminal nonterminal_number, Terminal terminal_number) :
	nonterminal_string_(nonterminal_string),
	terminal_string_(terminal_string),
	nonterminal_number_(nonterminal_number),
	terminal_number_(terminal_number) { }

    Node(const string &nonterminal_string, const string &terminal_string) :
	nonterminal_string_(nonterminal_string),
	terminal_string_(terminal_string) { }

    Node(Nonterminal nonterminal_number, Terminal terminal_number) :
	nonterminal_number_(nonterminal_number),
	terminal_number_(terminal_number) { }

    ~Node() { }

    // Returns true if this node is a root.
    bool IsRoot() { return parent_ == nullptr; }

    // Compares the node with the given node (applicable only for roots).
    bool Compare(Node *node);

    // Compares the node with the given node string (applicable only for roots).
    bool Compare(string node_string);

    // Returns a copy of this node (applicable only for roots).
    Node *Copy();

    // Deletes the node and all its descendent nodes.
    void DeleteSelfAndDescendents() { DeleteSelfAndDescendents(this); }

    // Appends the given node to the children vector to the right.
    void AppendToChildren(Node *child);

    // Gets the leaves of this node as a sequence of terminal strings.
    void Leaves(vector<string> *terminal_strings);

    // Gets the preterminal strings of this node.
    void Preterminals(vector<string> *preterminal_strings);

    // Returns the number of children nodes.
    int NumChildren() { return children_.size(); }

    // Returns the i-th child node.
    Node *Child(int i);

    // Returns the parent node.
    Node *parent() { return parent_; }

    // Returns the string form of this node.
    string ToString();

    // Returns the nonterminal string of this node.
    string nonterminal_string() { return nonterminal_string_; }

    // Returns the terminal string of this node.
    string terminal_string() { return terminal_string_; }

    // Returns the numeric identity of the nonterminal string of this node.
    Nonterminal nonterminal_number() { return nonterminal_number_; }

    // Returns the numeric identity of the terminal string of this node.
    Terminal terminal_number() { return terminal_number_; }

    // Returns the child index of this node.
    int child_index() { return child_index_; }

    // Returns the position of the first leaf this node spans.
    int span_begin() { return span_begin_; }

    // Returns the position of the last leaf this node spans.
    int span_end() { return span_end_; }

    // Sets the nonterminal string of this node.
    void set_nonterminal_string(const string &nonterminal_string) {
	nonterminal_string_ = nonterminal_string;
    }

    // Sets the terminal string of this node.
    void set_terminal_string(const string &terminal_string);

    // Sets the numeric identity of the nonterminal string of this node.
    void set_nonterminal_number(Nonterminal nonterminal_number) {
	nonterminal_number_ = nonterminal_number;
    }

    // Sets the numeric identity of the terminal string of this node.
    void set_terminal_number(Terminal terminal_number);

    // Sets the span of this node.
    void set_span(int span_begin, int span_end);

    // Is the node empty?
    bool IsEmpty() {
	return (nonterminal_string_.empty() && terminal_string_.empty() &&
		nonterminal_number_ == -1 && nonterminal_number_ == -1);
    }

    // Is the node an interminal?
    bool IsInterminal() { return !IsPreterminal(); }

    // Is the node a preterminal?
    bool IsPreterminal() { return children_.size() == 0; }

    // Removes the function tags in nonterminal strings of the derivation.
    void RemoveFunctionTags();

    // Removes null productions in the derivation.
    void RemoveNullProductions();

    // Adds a new root node.
    void AddRootNode();

    // Binarizes the derivation (+ vertical/horizontal Markovization).
    void Binarize(string binarization_method,
		  int vertical_markovization_order,
		  int horizontal_markovization_order);

    // Recovers the original derivation from binarization.
    void Debinarize();

    // Collapses unary productions in the derivation.
    void CollapseUnaryProductions();

    // Expands unary productions in the derivation.
    void ExpandUnaryProductions();

    // Processes the tree to standard form: removes function tags and null
    // productions.
    void ProcessToStandardForm();

    // Processes the tree to Chomsky normal form.
    void ProcessToChomskyNormalForm(string binarization_method,
				    int vertical_markovization_order,
				    int horizontal_markovization_order);

    // Recovers the original tree from Chomsky normal form transformation.
    void RecoverFromChomskyNormalForm();

private:
    // Compares the given nodes. This only compares the string components of the
    // nodes and ignores numeric components.
    bool Compare(Node *node1, Node *node2);

    // Copies the given node downward - does not copy its parent.
    Node *Copy(Node *node);

    // Deletes the given node and all its descendent nodes.
    void DeleteSelfAndDescendents(Node *node);

    // Deletes the i-th child node.
    void DeleteChild(int i);

    // Delete all descendent nodes.
    void DeleteAllDescendents();

    // Does the given symbol appear in a derived nonterminal string?
    bool AppearsInDerivedNonterminalString(string symbol);

    // Nonterminal string of the node.
    string nonterminal_string_ = "";

    // Terminal string of the node (stays "" if the node is an interminal).
    string terminal_string_ = "";

    // Numeric identity of the nonterminal string of the node.
    Nonterminal nonterminal_number_ = -1;

    // Numeric identity of the terminal string of the node (stays -1 if the node
    // is an interminal).
    Terminal terminal_number_ = -1;

    // Children Nodes.
    vector<Node *> children_;

    // Parent Node.
    Node *parent_ = nullptr;

    // Index of this node in the children vector of its parent (stays -1 if it
    // has no parent).
    int child_index_ = -1;

    // Position of the first leaf this node spans (-1 if no span).
    int span_begin_ = -1;

    // Position of the last leaf this node spans (-1 if no span).
    int span_end_ = -1;

    // Special string used as the interminal symbol of the added root node.
    const string kRootInterminalString_ = SPECIAL_ROOT_SYMBOL;

    // Special strings used for binarization.
    const string kGivenString_ = "|";
    const string kUpString_ = "^";
    const string kChainString_ = "~";

    // Special string used for removing unary productions.
    const string kGlueString_ = "+";

    // Special string used for preterminal nodes in null productions.
    const string kNullPreterminalString_ = "-NONE-";
};

// A tree set is a set of nodes representing trees.
class TreeSet {
public:
    // Creates a tree set from a file containing a tree string per line.
    TreeSet(const string &file_path) { ReadTreesFromFile(file_path); }

    // Creates an empty tree set.
    TreeSet() { }

    // Deletes all trees in the tree set.
    ~TreeSet() { Clear(); }

    // Deletes all trees in the tree set.
    void Clear();

    // Returns a copy of this tree set.
    TreeSet *Copy();

    // Evalutes against the given gold trees.
    void EvaluateAgainstGold(TreeSet *gold_trees, int *num_skipped,
			     double *precision, double *recall,
			     double *f1_score, double *tagging_accuracy);

    // Adds a new tree to the tree set.
    void AddTree(Node *tree) { trees_.push_back(tree); }

    // Returns the number of trees.
    int NumTrees() { return trees_.size(); }

    // Returns the i-th tree.
    Node *Tree(int i);

    // Returns the number of interminal types.
    int NumInterminalTypes();

    // Returns the number of preterminal types.
    int NumPreterminalTypes();

    // Returns the number of terminal types.
    int NumTerminalTypes();

    // Writes all tree strings to a file (tree per line).
    void Write(string file_path);

    // Processes the tree set to standard form: removes function tags and null
    // productions.
    void ProcessToStandardForm();

    // Processes the tree set to Chomsky normal form.
    void ProcessToChomskyNormalForm(string binarization_method,
				    int vertical_markovization_order,
				    int horizontal_markovization_order);

    // Recovers the original tree set from Chomsky normal form transformation.
    void RecoverFromChomskyNormalForm();

    // Populate the tree set with trees read from the given file.
    void ReadTreesFromFile(const string &file_path);

private:
    // Counts interminal/preterminal/observation types occurring in the trees.
    void CountTypes(unordered_map<string, int> *interminal_count,
		    unordered_map<string, int> *preterminal_count,
		    unordered_map<string, int> *observation_count);

    // Trees in this tree set.
    vector<Node *> trees_;
};

// Class for reading constituency trees from string.
class TreeReader {
public:
    // Creates a tree from the given tree string.
    Node *CreateTreeFromTreeString(const string &tree_string);

private:
    // Creates a tree from the given token sequence.
    Node *CreateTreeFromTokenSequence(const vector<string> &toks);

    // Tokenizes the given tree string: "(A (BB	b2))" -> "(", "A", "(", "BB",
    // "b2", ")", ")".
    void TokenizeTreeString(const string &tree_string, vector<string> *toks);
};

// A sequence of terminals represents the yield of a context-free derivation.
class TerminalSequence {
public:
    // Creates a terminal sequence from the given terminal strings.
    TerminalSequence(const vector<string> &terminal_strings) :
	terminal_strings_(terminal_strings) { AdjustPreterminalVectors(); }

    // Creates a terminal sequence from the given terminal strings and their
    // numeric identities.
    TerminalSequence(const vector<string> &terminal_strings,
		     const vector<Terminal> &terminal_numbers) :
	terminal_strings_(terminal_strings),
	terminal_numbers_(terminal_numbers)  { AdjustPreterminalVectors(); }

    // Creates a terminal sequence from a tree.
    TerminalSequence(Node *tree);

    ~TerminalSequence() { }

    // Returns the length of the sequence.
    int Length() { return terminal_strings_.size(); }

    // Returns the string form of the sequence.
    string ToString();

    // Sets the numeric IDs of terminals as the give sequence.
    void set_terminal_numbers(const vector<Terminal> &terminal_numbers) {
	terminal_numbers_ = terminal_numbers;
    }

    // Sets the numeric IDs of preterminals as the give sequence.
    void set_preterminal_numbers(const vector<Nonterminal>
				 &preterminal_numbers) {
	preterminal_numbers_ = preterminal_numbers;
    }

    // Returns the i-th terminal string.
    string TerminalString(int i) { return terminal_strings_[i]; }

    // Returns the i-th terminal number.
    Terminal TerminalNumber(int i) { return terminal_numbers_[i]; }

    // Returns the i-th preterminal string.
    string PreterminalString(int i) { return preterminal_strings_[i]; }

    // Returns the i-th preterminal number.
    Nonterminal PreterminalNumber(int i) { return preterminal_numbers_[i]; }

private:
    // Adjusts preterminal vectors to have the same length as terminal vectors.
    void AdjustPreterminalVectors();

    // Sequence of terminal strings.
    vector<string> terminal_strings_;

    // Sequence of terminal numbers.
    vector<Terminal> terminal_numbers_;

    // Sequence of preterminal strings.
    vector<string> preterminal_strings_;

    // Sequence of preterminal numbers.
    vector<Terminal> preterminal_numbers_;
};

// A set of terminal sequences.
class TerminalSequences {
public:
    // Creates a set of terminal sequences from a tree set.
    TerminalSequences(TreeSet *trees);

    // Creates an empty set of terminal sequences.
    TerminalSequences() { }

    ~TerminalSequences();

    // Adds a terminal sequence.
    void AddSequence(TerminalSequence *terminal_sequence) {
	terminal_sequences_.push_back(terminal_sequence);
    }

    // Returns the number of terminal sequences in this set.
    int NumSequences() { return terminal_sequences_.size(); }

    // Returns the i-th terminal sequence.
    TerminalSequence *Sequence(int i) { return terminal_sequences_[i]; }

private:
    // Terminal sequences.
    vector<TerminalSequence *> terminal_sequences_;
};

#endif  // CORE_TREES_H_
