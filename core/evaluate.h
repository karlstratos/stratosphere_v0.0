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
    // TODO: separate many_to_one.
    // Evaluates sequence predictions in per-position, per-sentence, and
    // many-to-one accuracy.
    void evaluate_sequences(const vector<vector<string> > &true_sequences,
			    const vector<vector<string> > &predicted_sequences,
			    double *position_accuracy,
			    double *sequence_accuracy,
			    double *many_to_one_accuracy,
			    unordered_map<string, string> *many_to_one_map);
}  // namespace evaluate

#endif  // CORE_EVALUATE_H_
