// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for manipulating grammar.

#ifndef CORE_GRAMMAR_H_
#define CORE_GRAMMAR_H_

#include <iostream>
#include <limits.h>
#include <string>
#include <tuple>
#include <unordered_map>

#include "../core/trees.h"
#include "../core/util.h"

using namespace std;

typedef vector<vector<vector<double> > > Chart;
typedef vector<vector<vector<tuple<Nonterminal, Nonterminal, int> > > >
Backpointer;

class Grammar {
public:
    // Creates an empty grammar.
    Grammar() { }

    ~Grammar() { }

    // Trains the grammar on the given trees.
    void Train(TreeSet *trees);

    // Parses the given sequence of strings.
    Node *Parse(const vector<string> &terminal_strings);

    // Parses the terminal sequences in the given trees.
    TreeSet *Parse(TreeSet *trees, bool silent);

    // Writes the grammar to a file.
    void Write(const string &file_path);

    // Load the grammar (from: model_directory_ + "model").
    void Load();

    // Returns the numeric ID of the given nonterminal string.
    Nonterminal nonterminal_str2num(const string &a_string) {
	return (nonterminal_str2num_.find(a_string) !=
		nonterminal_str2num_.end()) ? nonterminal_str2num_[a_string]
	    : -1;
    }

    // Returns the numeric ID of the given terminal string.
    Terminal terminal_str2num(const string &x_string) {
	return (terminal_str2num_.find(x_string) !=
		terminal_str2num_.end()) ? terminal_str2num_[x_string] : -1;
    }

    // Returns the special string used to represent an unknown terminal string.
    string unknown_terminal() { return kUnknownTerminal_; }

    // Sets the path to the parser model directory.
    void set_model_directory(string model_directory) {
	model_directory_ = model_directory;
    }

    // Sets the binarization method.
    void set_binarization_method(string binarization_method) {
	binarization_method_ = binarization_method;
    }

    // Sets the order of vertical Markovization.
    void set_vertical_markovization_order(
	int vertical_markovization_order) {
	vertical_markovization_order_ = vertical_markovization_order;
    }

    // Sets the order of horizontal Markovization.
    void set_horizontal_markovization_order(
	int horizontal_markovization_order) {
	horizontal_markovization_order_ = horizontal_markovization_order;
    }

    // Sets the flag for using gold part-of-speech tags at test time.
    void set_use_gold_tags(bool use_gold_tags) {
	use_gold_tags_ = use_gold_tags;
    }

    // Sets the maximum length of a sentence to parse.
    void set_max_sentence_length(int max_sentence_length) {
	max_sentence_length_ = max_sentence_length;
    }

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Computes the probability of the given tree under PCFG.
    double ComputePCFGTreeProbability(Node *tree);

    // Computes the marginal probabilities for the given strings under PCFG.
    void ComputeMarginalsPCFG(const vector<string> &terminal_strings,
			      Chart *marginal);

    // Returns the number of nonterminal types.
    int NumNonterminalTypes() { return nonterminal_num2str_.size(); }

    // Returns the number of interminal types.
    int NumInterminalTypes() { return interminal_.size(); }

    // Returns the number of preterminal types.
    int NumPreterminalTypes() { return preterminal_.size(); }

    // Returns the number of terminal types.
    int NumTerminalTypes() { return terminal_num2str_.size(); }

    // Returns the number of binary rule types.
    int NumBinaryRuleTypes();

    // Returns the number of unary rule types.
    int NumUnaryRuleTypes();

    // Returns the number of nonterminal types that can be roots.
    int NumRootNonterminalTypes() { return lprob_root_.size(); }

private:
    // Parses the given terminal sequence.
    Node *Parse(TerminalSequence *terminals);

    // Computes the most likely tree under PCFG.
    Node *ParsePCFGCKY(TerminalSequence *terminals);

    // Computes the tree that maximizes the sum of marginal probabilities under
    // PCFG.
    Node *ParsePCFGMarginal(TerminalSequence *terminals);

    // Computes all the marginal probabilities for the given terminal sequence.
    void ComputeMarginalsPCFG(TerminalSequence *terminals, Chart *marginal);

    // Computes inside probabilities for the given terminal sequence.
    void ComputeInsideProbabilitiesPCFG(TerminalSequence *terminals,
					Chart *inside);

    // Computes outside probabilities for the given terminal sequence.
    void ComputeOutsideProbabilitiesPCFG(int sequence_length,
					 const Chart &inside, Chart *outside);

    // Computes outside probability of subtree a -*-> x_i...x_j.
    double ComputeOutsideProbabilityPCFG(
	Nonterminal a, int i, int j, int sequence_length,
	const Chart &inside, const Chart &outside);

    // Recovers the tree that maximizes the sum of marginals.
    Node *RecoverMaxMarginalTree(TerminalSequence *terminals,
				 const Chart &marginal);

    // Recovers the tree recorded in the backpointer from the start point.
    Node *RecoverFromBackpointer(
	const Backpointer &bp, TerminalSequence *terminals,
	int start_position, int end_position, Nonterminal a);

    // Estimate PCFG parameters from the given trees.
    void EstimatePCFG(TreeSet *trees);

    // Add the nonterminal to the nonterminal dictionary if not already known.
    Nonterminal AddNonterminalIfUnknown(const string &a_string);

    // Add the terminal to the terminal dictionary if not already known.
    Terminal AddTerminalIfUnknown(const string &x_string);

    // Assert that the model probability distributions are all proper.
    void AssertProperDistributions();

    // Assigns numeric identities for the nonterminals/terminals in the tree.
    void AssignNumericIdentities(Node *tree);

    // Assigns string identities for the nonterminals/terminals in the tree.
    void AssignStringIdentities(Node *tree);

    // Assigns numeric identities for the given terminal sequence.
    void AssignNumericIdentities(TerminalSequence *terminals);

    // Maps a nonterminal string to an integer ID.
    unordered_map<string, Nonterminal> nonterminal_str2num_;

    // Maps a nonterminal ID to its original string form.
    unordered_map<Nonterminal, string> nonterminal_num2str_;

    // Maps a terminal string to an integer ID.
    unordered_map<string, Terminal> terminal_str2num_;

    // Maps a terminal ID to its original string form.
    unordered_map<Terminal, string> terminal_num2str_;

    // Keys of this map consist of interminal types.
    unordered_map<Nonterminal, bool> interminal_;

    // Keys of this map consist of preterminal types.
    unordered_map<Nonterminal, bool> preterminal_;

    // Path to the parser model directory.
    string model_directory_;

    // Path to the log file.
    ofstream log_;

    // Binarization method.
    string binarization_method_ = "left";

    // Order of vertical Markovization.
    int vertical_markovization_order_ = 0;

    // Order of horizontal Markovization.
    int horizontal_markovization_order_ = 0;

    // Use gold part-of-speech tags at test time?
    bool use_gold_tags_ = false;

    // Maximum length of a sentence to parse.
    int max_sentence_length_ = 1000;

    // Decoding method.
    string decoding_method_;

    // Special string used to represent an unknown terminal string.
    const string kUnknownTerminal_ = "<?>";

    // log p(a -> b c|a): conditional probability of binary rules in log space.
    unordered_map<Nonterminal,
		  unordered_map<Nonterminal,
				unordered_map<Nonterminal, double> > >
    lprob_binary_;

    // log p(a -> x|a): conditional probability of unary rules in log space.
    unordered_map<Nonterminal, unordered_map<Terminal, double> >
    lprob_unary_;

    // log p(a): prior probability of root nonterminals in log space.
    unordered_map<Nonterminal, double> lprob_root_;

    // interminal a => {(b, c, log p(a -> b c|a)): p(a -> b c|a) > 0}
    unordered_map<Nonterminal,
		  vector<tuple<Nonterminal, Nonterminal, double> > >
    binary_rhs_;

    // nonterminal a => {(b, c, log p(b -> a c|b)): p(b -> a c|b) > 0}
    unordered_map<Nonterminal,
		  vector<tuple<Nonterminal, Nonterminal, double> > >
    left_parent_sibling_;

    // nonterminal a => {(b, c, log p(b -> c a|b)): p(b -> c a|b) > 0}
    unordered_map<Nonterminal,
		  vector<tuple<Nonterminal, Nonterminal, double> > >
    right_parent_sibling_;
};

#endif  // CORE_GRAMMAR_H_