// Author: Karl Stratos (me@karlstratos.com)
//
// Code for pruning a binary tree.

#ifndef CORE_PRUNER_H_
#define CORE_PRUNER_H_

#include <iostream>
#include <random>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "trees.h"

typedef unordered_map<string, vector<string> > um2v;

class Pruner {
public:
    // Deallocates memory assigned to the tree.
    ~Pruner() {
	if (tree_ != nullptr) { tree_->DeleteSelfAndDescendents(); }
    }

    // Reads a tree from a cluster file: <bits> <point> <count>.
    void ReadTree(const string &file_path);

    // Reads a tree from the bit string format: {100:x, 101:y, ...}.
    void ReadTree(const unordered_map<string, string> &bitstring);

    // Reads labeled prototypes from a file: <label> <p_1> ... <p_k>
    unordered_map<string, string> ReadPrototypes(const string &file_path);

    // Reads an oracle labeler x->C(x) from a file: <C(x)> <x>.
    unordered_map<string, string> ReadOracle(const string &file_path);

    // Sample prototypes.
    void SamplePrototypes(const unordered_map<string, string> &oracle,
			  const vector<string> &leaves,
			  size_t num_proto,
			  unordered_map<string, string> *proto2label);

    // Propagates the labels of the given prototypes through the tree.
    um2v PropagateLabels(const um2v &prototypes);

    // At the given node: adds all subtrees purely in label c into
    // pure_subtrees[c], all subtrees whose labels cannot be inferred into
    // unknown_subtrees[c] along with their best-guess labels from siblings.
    // Returns a pair ([1], [2]) where [1] is the label status and [2] is the
    // best-guess label of the node.
    //
    //      - If the node is pure in c:   [1]=c, [2]=c
    //      - If the node is unspecified: [1]=?, [2]=""
    //      - If the node is conflicted:  [1]=!, [2]=majority_label
    pair<string, string> FindConsistentSubtrees(
	Node *node,
	const unordered_map<string, string> &proto2label,
	unordered_map<string, vector<Node *> > *pure_subtrees,
	vector<pair<Node *, string> > *unknown_subtrees);

    // Processes consistent subtrees into full labeling.
    um2v LabelConsistentSubtrees(
	const unordered_map<string, vector<Node *> > &pure_subtrees,
	const vector<pair<Node *, string> > &unknown_subtrees);

    // Returns the root node of the tree.
    Node *tree() { return tree_; }

private:
    // Replaces a parenthesis with a special symbol.
    string ReplaceParenthesis(const string &leaf);

    // Restores a parenthesis from a special symbol.
    string RestoreParenthesis(const string &leaf);

    // Tree.
    Node *tree_ = nullptr;

    // Special string to represent an unknown.
    const string kUnknown_ = "<?>";

    // Special string to represent a conflict.
    const string kConflict_ = "<!>";

    // Special strings to represent parentheses.
    const string kRRB_ = "<[-RRB-]>";
    const string kLRB_ = "<[-LRB-]>";

    // Randomness engine
    mt19937 mt;
};

#endif  // CORE_PRUNER_H_
