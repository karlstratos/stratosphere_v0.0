// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "evaluate.h"

#include "util.h"

namespace evaluate {
    void evaluate_sequences(const vector<vector<string> > &true_sequences,
			    const vector<vector<string> > &predicted_sequences,
			    double *position_accuracy,
			    double *sequence_accuracy) {
	size_t num_items = 0;
	size_t num_items_correct = 0;
	size_t num_sequences_correct = 0;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    num_items += true_sequences[i].size();
	    bool entire_sequence_is_correct = true;
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		string true_string = true_sequences[i][j];
		string predicted_string = predicted_sequences[i][j];
		if (predicted_string == true_string) {
		    num_items_correct += 1;
		} else {
		    entire_sequence_is_correct = false;
		}
	    }
	    if (entire_sequence_is_correct) { num_sequences_correct += 1; }
	}
	(*position_accuracy) = ((double) num_items_correct) / num_items * 100;
	(*sequence_accuracy) = ((double) num_sequences_correct) /
	    true_sequences.size() * 100;
    }

    void evaluate_sequences_mapping_labels(
	const vector<vector<string> > &true_sequences,
	const vector<vector<string> > &predicted_sequences,
	double *position_accuracy, double *sequence_accuracy,
	unordered_map<string, string> *label_mapping) {
	// Create many-to-one label mapping.
	unordered_map<string, unordered_map<string, size_t> > count_matches;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		++count_matches[predicted_sequences[i][j]][
		    true_sequences[i][j]];
	    }
	}
	for (const auto &predicted_pair: count_matches) {
	    vector<pair<string, size_t> > matches;
	    for (const auto &true_pair: predicted_pair.second) {
		matches.emplace_back(true_pair.first, true_pair.second);
	    }
	    sort(matches.begin(), matches.end(),
		 util_misc::sort_pairs_second<string, size_t,
		 greater<size_t> >());
	    (*label_mapping)[predicted_pair.first] = matches[0].first;
	}

	// Use the mapping to match label sets.
	vector<vector<string> > predicted_sequences_mapped(
	    predicted_sequences.size());
	for (size_t i = 0; i < predicted_sequences.size(); ++i) {
	    predicted_sequences_mapped[i].resize(predicted_sequences[i].size());
	    for (size_t j = 0; j < predicted_sequences[i].size(); ++j) {
		predicted_sequences_mapped[i][j] =
		    (*label_mapping)[predicted_sequences[i][j]];
	    }
	}
	evaluate_sequences(true_sequences, predicted_sequences_mapped,
			   position_accuracy, sequence_accuracy);
    }
}  // namespace evaluate
