// Author: Karl Stratos (me@karlstratos.com)
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
typedef vector<vector<vector<tuple<Nonterminal, Nonterminal, size_t> > > >
Backpointer;

class Grammar {
public:
    // Creates an empty grammar.
    Grammar() { }

    ~Grammar() { }

    // Clears the model.
    void Clear();

    // Trains the grammar from a treebank file.
    void Train(const string &treebank_path);

    // Trains the grammar on the given trees.
    void Train(TreeSet *trees);

    // Evalutes parsing on a treebank file.
    void Evaluate(const string &treebank_path, const string prediction_path);

    // Parses the terminal sequences in the given trees.
    TreeSet *Parse(TreeSet *trees);

    // Parses the given sequence of strings.
    Node *Parse(const vector<string> &terminal_strings);

    // Saves the grammar to a file.
    void Save(const string &model_path);

    // Load the grammar from a file.
    void Load(const string &model_path);

    // Returns the numeric ID of the given nonterminal string.
    Nonterminal nonterminal_dictionary(const string &a_string) {
	return (nonterminal_dictionary_.find(a_string) !=
		nonterminal_dictionary_.end()) ?
	    nonterminal_dictionary_[a_string] : -1;
    }

    // Returns the numeric ID of the given terminal string.
    Terminal terminal_dictionary(const string &x_string) {
	return (terminal_dictionary_.find(x_string) !=
		terminal_dictionary_.end()) ?
	    terminal_dictionary_[x_string] : -1;
    }

    // Returns the special string used to represent an unknown terminal string.
    string unknown_terminal() { return kUnknownTerminal_; }

    // Sets the binarization method.
    void set_binarization_method(string binarization_method) {
	binarization_method_ = binarization_method;
    }

    // Sets the order of vertical Markovization.
    void set_vertical_markovization_order(
	size_t vertical_markovization_order) {
	vertical_markovization_order_ = vertical_markovization_order;
    }

    // Sets the order of horizontal Markovization.
    void set_horizontal_markovization_order(
	size_t horizontal_markovization_order) {
	horizontal_markovization_order_ = horizontal_markovization_order;
    }

    // Sets the flag for using gold part-of-speech tags at test time.
    void set_use_gold_tags(bool use_gold_tags) {
	use_gold_tags_ = use_gold_tags;
    }

    // Sets the maximum length of a sentence to parse.
    void set_max_sentence_length(size_t max_sentence_length) {
	max_sentence_length_ = max_sentence_length;
    }

    // Sets the decoding method.
    void set_decoding_method(string decoding_method) {
	decoding_method_ = decoding_method;
    }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Computes the probability of the given tree under PCFG.
    double ComputePCFGTreeProbability(Node *tree);

    // Computes the marginal probabilities for the given strings under PCFG.
    void ComputeMarginalsPCFG(const vector<string> &terminal_strings,
			      Chart *marginal);

    // Returns the number of nonterminal types.
    size_t NumNonterminalTypes() { return nonterminal_dictionary_.size(); }

    // Returns the number of interminal types.
    size_t NumInterminalTypes() { return interminal_.size(); }

    // Returns the number of preterminal types.
    size_t NumPreterminalTypes() { return preterminal_.size(); }

    // Returns the number of terminal types.
    size_t NumTerminalTypes() { return terminal_dictionary_.size(); }

    // Returns the number of binary rule types.
    size_t NumBinaryRuleTypes();

    // Returns the number of unary rule types.
    size_t NumUnaryRuleTypes();

    // Returns the number of nonterminal types that can be roots.
    size_t NumRootNonterminalTypes() { return lprob_root_.size(); }

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
    void ComputeOutsideProbabilitiesPCFG(size_t sequence_length,
					 const Chart &inside, Chart *outside);

    // Computes outside probability of subtree a -*-> x_i...x_j.
    double ComputeOutsideProbabilityPCFG(
	Nonterminal a, size_t i, size_t j, size_t sequence_length,
	const Chart &inside, const Chart &outside);

    // Recovers the tree that maximizes the sum of marginals.
    Node *RecoverMaxMarginalTree(TerminalSequence *terminals,
				 const Chart &marginal);

    // Recovers the tree recorded in the backpointer from the start point.
    Node *RecoverFromBackpointer(
	const Backpointer &bp, TerminalSequence *terminals,
	size_t start_position, size_t end_position, Nonterminal a);

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
    unordered_map<string, Nonterminal> nonterminal_dictionary_;

    // Maps a nonterminal ID to its original string form.
    unordered_map<Nonterminal, string> nonterminal_dictionary_inverse_;

    // Maps a terminal string to an integer ID.
    unordered_map<string, Terminal> terminal_dictionary_;

    // Maps a terminal ID to its original string form.
    unordered_map<Terminal, string> terminal_dictionary_inverse_;

    // Keys of this map consist of interminal types.
    unordered_map<Nonterminal, bool> interminal_;

    // Keys of this map consist of preterminal types.
    unordered_map<Nonterminal, bool> preterminal_;

    // Binarization method.
    string binarization_method_ = "left";

    // Order of vertical Markovization.
    size_t vertical_markovization_order_ = 0;

    // Order of horizontal Markovization.
    size_t horizontal_markovization_order_ = 0;

    // Use gold part-of-speech tags at test time?
    bool use_gold_tags_ = false;

    // Maximum length of a sentence to parse.
    size_t max_sentence_length_ = 1000;

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

    // Print messages to stderr?
    bool verbose_ = true;
};

#endif  // CORE_GRAMMAR_H_
