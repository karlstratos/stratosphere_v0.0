// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for evaluation.

#ifndef CORE_EVALUATE_H_
#define CORE_EVALUATE_H_

#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace evaluate {
    // Evaluates sequence predictions.
    void evaluate_sequences(const vector<vector<string> > &true_sequences,
			    const vector<vector<string> > &predicted_sequences,
			    double *position_accuracy,
			    double *sequence_accuracy);

    // Evaluates sequence predictions with mapping labels (many-to-one).
    void evaluate_sequences_mapping_labels(
	const vector<vector<string> > &true_sequences,
	const vector<vector<string> > &predicted_sequences,
	double *position_accuracy, double *sequence_accuracy,
	unordered_map<string, string> *label_mapping);

    // Evaluate word vectors on their correlation with gold similarity scores.
    void evaluate_similarity(
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	const vector<tuple<string, string, double> > &word_pair_scores,
	bool normalized, size_t *num_handled, double *correlation);

    // Evaluate word vectors on a word similarity data file.
    void evaluate_similarity(const unordered_map<string, Eigen::VectorXd>
			     &word_vectors, const string &similarity_path,
			     bool normalized, size_t *num_instances,
			     size_t *num_handled, double *correlation);

    // Returns word v2 (not in {w1, w2, v1}) such that "w1:w2 ~ v1:v2". Word
    // vectors must contain {w1, w2, v1}.
    string infer_analogous_word(string w1, string w2, string v1,
				const unordered_map<string, Eigen::VectorXd>
				&word_vectors, bool normalized);

    // Evaluate word vectors on word analogy questions.
    void evaluate_analogy(
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	const vector<tuple<string, string, string, string> > &questions,
	size_t *num_instances, size_t *num_handled, double *accuracy);

    // Evaluate word vectors on a word analogy data file.
    void evaluate_analogy(const unordered_map<string, Eigen::VectorXd>
			  &word_vectors, const string &analogy_path,
			  size_t *num_instances, size_t *num_handled,
			  double *accuracy);
}  // namespace evaluate

#endif  // CORE_EVALUATE_H_
