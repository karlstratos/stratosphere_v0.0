// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for evaluation.

#ifndef CORE_EVALUATE_H_
#define CORE_EVALUATE_H_

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
}  // namespace evaluate

#endif  // CORE_EVALUATE_H_
