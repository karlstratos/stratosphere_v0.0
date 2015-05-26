// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "evaluate.h"

#include "util.h"

namespace evaluate {
    void evaluate_sequences(const vector<vector<string> > &true_sequences,
			    const vector<vector<string> > &predicted_sequences,
			    double *position_accuracy,
			    double *sequence_accuracy,
			    double *many_to_one_accuracy,
			    unordered_map<string, string> *many_to_one_map) {
	size_t num_items = 0;
	size_t num_items_correct = 0;
	size_t num_sequences_correct = 0;
	unordered_map<string, unordered_map<string, size_t> >
	    count_predicted_true;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    ASSERT(true_sequences[i].size() == predicted_sequences[i].size(),
		   "Sequence lengths not matching");
	    num_items += true_sequences[i].size();
	    bool entire_sequence_is_correct = true;
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		string true_string = true_sequences[i][j];
		string predicted_string = predicted_sequences[i][j];
		++count_predicted_true[predicted_string][true_string];
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

	// Map predicted string => most frequently co-occurring true string.
	for (const auto &predicted_pair: count_predicted_true) {
	    vector<pair<string, size_t> > cooccurrence_counts;
	    for (const auto &true_pair: predicted_pair.second) {
		cooccurrence_counts.emplace_back(true_pair.first,
						 true_pair.second);
	    }
	    sort(cooccurrence_counts.begin(), cooccurrence_counts.end(),
		 util_misc::sort_pairs_second<string, size_t,
		 greater<size_t> >());
	    (*many_to_one_map)[predicted_pair.first] =
		cooccurrence_counts[0].first;
	}

	// Use the mapping to obtain many-to-1 accuracy.
	size_t num_items_correct_many_to_one = 0;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		string true_string = true_sequences[i][j];
		string predicted_string = predicted_sequences[i][j];
		string predicted_string_mapped =
		    (*many_to_one_map)[predicted_string];
		if (predicted_string_mapped == true_string) {
		    num_items_correct_many_to_one += 1;
		}
	    }
	}
	(*many_to_one_accuracy) =
	    ((double) num_items_correct_many_to_one) / num_items * 100;
    }
}  // namespace evaluate
