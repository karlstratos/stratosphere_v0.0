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
	size_t *num_handled, double *correlation);

    // Evaluate word vectors on a word similarity data file.
    void evaluate_similarity(const unordered_map<string, Eigen::VectorXd>
			     &word_vectors, const string &similarity_path,
			     size_t *num_instances, size_t *num_handled,
			     double *correlation);
}  // namespace evaluate

#endif  // CORE_EVALUATE_H_
