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

    void evaluate_similarity(
	const unordered_map<string, Eigen::VectorXd> &word_vectors,
	const vector<tuple<string, string, double> > &word_pair_scores,
	size_t *num_handled, double *correlation) {
	vector<double> gold_scores;
	vector<double> cosine_scores;
	*num_handled = 0;
	for (const auto &word_pair_score : word_pair_scores) {
	    string word1 = get<0>(word_pair_score);
	    string word2 = get<1>(word_pair_score);
	    double gold_score = get<2>(word_pair_score);
	    string word1_lowercase = util_string::lowercase(word1);
	    string word2_lowercase = util_string::lowercase(word2);
	    Eigen::VectorXd word1_vector;
	    Eigen::VectorXd word2_vector;

	    // Try to find the original string. If not found, try lowercasing.
	    if (word_vectors.find(word1) != word_vectors.end()) {
		word1_vector = word_vectors.at(word1);
	    } else if (word_vectors.find(word1_lowercase) !=
		       word_vectors.end()) {
		word1_vector = word_vectors.at(word1_lowercase);
	    }
	    if (word_vectors.find(word2) != word_vectors.end()) {
		word2_vector = word_vectors.at(word2);
	    } else if (word_vectors.find(word2_lowercase) !=
		       word_vectors.end()) {
		word2_vector = word_vectors.at(word2_lowercase);
	    }

	    // If we have vectors for both word types, compute similarity.
	    if (word1_vector.size() > 0 && word2_vector.size() > 0) {
		word1_vector.normalize();
		word2_vector.normalize();
		double cosine_score = word1_vector.dot(word2_vector);
		gold_scores.push_back(gold_score);
		cosine_scores.push_back(cosine_score);
		++(*num_handled);
	    }
	}
	*correlation = util_math::compute_spearman(gold_scores, cosine_scores);
    }

    void evalute_similarity(const unordered_map<string, Eigen::VectorXd>
			    &word_vectors, const string &similarity_path,
			    size_t *num_instances, size_t *num_handled,
			    double *correlation) {
	ifstream similarity_file(similarity_path, ios::in);
	ASSERT(similarity_file.is_open(), "Cannot open: " << similarity_path);
	vector<tuple<string, string, double> > word_pair_scores;
	while (similarity_file.good()) {
	    vector<string> tokens;
	    util_file::read_line(&similarity_file, &tokens);
	    if (tokens.size() > 0) {
		ASSERT(tokens.size() == 3, "Need [word1] [word2] [similarity]");
		string word1 = tokens[0];
		string word2 = tokens[1];
		double gold_score = stod(tokens[2]);
		word_pair_scores.push_back(make_tuple(word1, word2,
						      gold_score));
	    }
	}
	*num_instances = word_pair_scores.size();
	evaluate_similarity(word_vectors, word_pair_scores, num_handled,
			    correlation);
    }

}  // namespace evaluate
