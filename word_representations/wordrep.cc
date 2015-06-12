// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "wordrep.h"

#include <iomanip>
#include <limits>
#include <map>

#include "../core/cluster.h"
#include "../core/corpus.h"
#include "../core/evaluate.h"

void WordRep::SetOutputDirectory(const string &output_directory) {
    ASSERT(!output_directory.empty(), "Empty output directory.");
    output_directory_ = output_directory;

    // Prepare the output directory.
    ASSERT(system(("mkdir -p " + output_directory_).c_str()) == 0,
	   "Cannot create directory: " << output_directory_);
}

void WordRep::ResetOutputDirectory() {
    ASSERT(!output_directory_.empty(), "No output directory given.");
    ASSERT(system(("rm -f " + output_directory_ + "/*").c_str()) == 0,
	   "Cannot remove the content in: " << output_directory_);
    SetOutputDirectory(output_directory_);
}

void WordRep::ExtractStatistics(const string &corpus_file) {
    Corpus corpus(corpus_file, verbose_);

    // Get the word dictionary.
    unordered_map<string, Word> word_dictionary;
    if (!util_file::exists(SortedWordTypesPath()) ||
	!util_file::exists(WordDictionaryPath())) {
	corpus.WriteWords(rare_cutoff_, SortedWordTypesPath(),
			  WordDictionaryPath());
    }
    util_file::binary_read(WordDictionaryPath(), &word_dictionary);

    // Get the context dictionary and co-occurrence counts.
    if (!util_file::exists(ContextDictionaryPath()) ||
	!util_file::exists(ContextWordCountPath())) {
	corpus.WriteContexts(word_dictionary, sentence_per_line_,
			     context_definition_, window_size_, hash_size_,
			     ContextDictionaryPath(), ContextWordCountPath());
    }
}

void WordRep::InduceWordVectors() {
    if (util_file::exists(SingularValuesPath()) &&
	util_file::exists(WordVectorsPath())) { return; }
    SMat matrix = sparsesvd::binary_read_sparse_matrix(ContextWordCountPath());
    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
    Eigen::VectorXd singular_values;
    corpus::decompose(matrix, dim_, transformation_method_, add_smooth_,
		      power_smooth_, scaling_method_, &left_singular_vectors,
		      &right_singular_vectors, &singular_values);
    svdFreeSMat(matrix);

    // Write singular values.
    ofstream singular_values_file(SingularValuesPath(), ios::out);
    singular_values_file << singular_values << endl;

    // Write word vectors in decreasing word frequency.
    vector<pair<string, size_t> > sorted_word_types;
    corpus::load_sorted_word_types(rare_cutoff_, SortedWordTypesPath(),
				   &sorted_word_types);
    unordered_map<string, Word> word_dictionary;
    util_file::binary_read(WordDictionaryPath(), &word_dictionary);
    ofstream word_vectors_file(WordVectorsPath(), ios::out);
    for (size_t i = 0; i < sorted_word_types.size(); ++i) {
	string word_string = sorted_word_types[i].first;
	size_t word_count = sorted_word_types[i].second;
	ASSERT(word_dictionary.find(word_string) != word_dictionary.end(),
	       "Word not in dictionary: " << word_string);
	word_vectors_file << word_count << " " << word_string;

	// Word vector is the corresponding row of the left singular vector
	// matrix, normalized to have unit 2-norm.
	Word word = word_dictionary[word_string];
	left_singular_vectors.row(word).normalize();
	for (size_t j = 0; j < left_singular_vectors.row(word).size(); ++ j) {
	    word_vectors_file << " " << left_singular_vectors.row(word)(j);
	}
	word_vectors_file << endl;
    }
}

void WordRep::EvaluateWordVectors() {
    if (!util_file::exists(WordVectorsPath())) { return; }

    /*
    // Find development datasets for evaluation.
    string wordsim353_path = "third_party/public_datasets/wordsim353.dev";
    string men_path = "third_party/public_datasets/men.dev";
    string rw_path = "third_party/public_datasets/rw.dev";
    string mturk_path = "third_party/public_datasets/mturk.dev";
    string syn_path = "third_party/public_datasets/syntactic_analogies.dev";
    string mixed_path = "third_party/public_datasets/mixed_analogies.dev";
    FileManipulator file_manipulator;
    if (!file_manipulator.Exists(wordsim353_path) ||
	!file_manipulator.Exists(men_path) ||
	!file_manipulator.Exists(rw_path) ||
	!file_manipulator.Exists(mturk_path) ||
	!file_manipulator.Exists(syn_path) ||
	!file_manipulator.Exists(mixed_path)) {
	// Skip evaluation (e.g., in unit tests) if files are not found.
	return;
    }
    */

    unordered_map<string, Eigen::VectorXd> word_vectors;
    corpus::load_word_vectors(WordVectorsPath(), &word_vectors);

}
/*

void WordRep::TestQualityOfWordVectors() {
    string wordsim353_path = "third_party/public_datasets/wordsim353.dev";
    string men_path = "third_party/public_datasets/men.dev";
    string rw_path = "third_party/public_datasets/rw.dev";
    string mturk_path = "third_party/public_datasets/mturk.dev";
    string syn_path = "third_party/public_datasets/syntactic_analogies.dev";
    string mixed_path = "third_party/public_datasets/mixed_analogies.dev";
    FileManipulator file_manipulator;
    if (!file_manipulator.Exists(wordsim353_path) ||
	!file_manipulator.Exists(men_path) ||
	!file_manipulator.Exists(rw_path) ||
	!file_manipulator.Exists(mturk_path) ||
	!file_manipulator.Exists(syn_path) ||
	!file_manipulator.Exists(mixed_path)) {
	// Skip evaluation (e.g., in unit tests) if files are not found.
	return;
    }
    log_ << endl << "[Dev performance]" << endl;

    // Use 3 decimal places for word similartiy.
    log_ << fixed << setprecision(3);
    Evaluator eval;

    // Word similarity with wordsim353.dev.
    size_t num_instances_wordsim353;
    size_t num_handled_wordsim353;
    double corr_wordsim353;
    eval.EvaluateWordSimilarity(wordvectors_, wordsim353_path,
				&num_instances_wordsim353,
				&num_handled_wordsim353, &corr_wordsim353);
    log_ << "   WS353: \t" << corr_wordsim353 << " ("
	 << num_handled_wordsim353 << "/" << num_instances_wordsim353
	 << " evaluated)" << endl;

    // Word similarity with men.dev.
    size_t num_instances_men;
    size_t num_handled_men;
    double corr_men;
    eval.EvaluateWordSimilarity(wordvectors_, men_path, &num_instances_men,
				&num_handled_men, &corr_men);
    log_ << "   MEN:  \t" << corr_men << " (" << num_handled_men << "/"
	 << num_instances_men << " evaluated)" << endl;

    // Word similarity with rw.dev (rare words).
    size_t num_instances_rw;
    size_t num_handled_rw;
    double corr_rw;
    eval.EvaluateWordSimilarity(wordvectors_, rw_path, &num_instances_rw,
				&num_handled_rw, &corr_rw);
    log_ << "   RW:    \t" << corr_rw << " (" << num_handled_rw << "/"
	 << num_instances_rw << " evaluated)" << endl;

    // Word similarity with mturk.dev.
    size_t num_instances_mturk;
    size_t num_handled_mturk;
    double corr_mturk;
    eval.EvaluateWordSimilarity(wordvectors_, mturk_path, &num_instances_mturk,
				&num_handled_mturk, &corr_mturk);
    log_ << "   MTURK: \t" << corr_mturk << " (" << num_handled_mturk << "/"
	 << num_instances_mturk << " evaluated)" << endl;

    // Word analogy with syntactic_analogies.dev.
    log_ << fixed << setprecision(2);
    size_t num_instances_syn;
    size_t num_handled_syn;
    double acc_syn;
    eval.EvaluateWordAnalogy(wordvectors_, syn_path, &num_instances_syn,
			     &num_handled_syn, &acc_syn);
    log_ << "   SYN: \t" << acc_syn << " (" << num_handled_syn
	 << "/" << num_instances_syn << " evaluated)" << endl;

    // Word analogy with mixed_analogies.dev.
    size_t num_instances_mixed;
    size_t num_handled_mixed;
    double acc_mixed;
    eval.EvaluateWordAnalogy(wordvectors_, mixed_path, &num_instances_mixed,
			     &num_handled_mixed, &acc_mixed);
    log_ << "   MIXED: \t" << acc_mixed << " (" << num_handled_mixed << "/"
	 << num_instances_mixed << " evaluated)" << endl;
}

void WordRep::PerformAgglomerativeClustering(size_t num_clusters) {
    FileManipulator file_manipulator;  // Do not repeat the work.
    if (file_manipulator.Exists(AgglomerativePath())) { return; }

    // Prepare a list of word vectors sorted in decreasing frequency.
    ASSERT(wordvectors_.size() > 0, "No word vectors to cluster!");
    vector<Eigen::VectorXd> sorted_vectors(sorted_wordcount_.size());
    for (size_t i = 0; i < sorted_wordcount_.size(); ++i) {
	string word_string = sorted_wordcount_[i].first;
	sorted_vectors[i] = wordvectors_[word_string];
    }

    // Do agglomerative clustering over the sorted word vectors.
    if (verbose_) { cerr << "Clustering" << endl; }
    time_t begin_time_greedo = time(NULL);
    log_ << endl << "[Agglomerative clustering]" << endl;
    log_ << "   Number of clusters: " << num_clusters << endl;
    Greedo greedo;
    greedo.Cluster(sorted_vectors, num_clusters);
    double time_greedo = difftime(time(NULL), begin_time_greedo);
    StringManipulator string_manipulator;
    log_ << "   Average number of tightenings: "
	 << greedo.average_num_extra_tightening() << " (versus exhaustive "
	 << num_clusters << ")" << endl;
    log_ << "   Time taken: " << string_manipulator.TimeString(time_greedo)
	 << endl;

    // Lexicographically sort bit strings for enhanced readability.
    vector<string> bitstring_types;
    for (const auto &bitstring_pair : *greedo.bit2cluster()) {
	bitstring_types.push_back(bitstring_pair.first);
    }
    sort(bitstring_types.begin(), bitstring_types.end());

    // Write the bit strings and their associated word types.
    ofstream greedo_file(AgglomerativePath(), ios::out);
    unordered_map<string, vector<size_t> > *bit2cluster = greedo.bit2cluster();
    for (const auto &bitstring : bitstring_types) {
	vector<pair<string, size_t> > sorting_vector;  // Sort each cluster.
	for (size_t cluster : bit2cluster->at(bitstring)) {
	    string word_string = sorted_wordcount_[cluster].first;
	    size_t count = sorted_wordcount_[cluster].second;
	    sorting_vector.push_back(make_pair(word_string, count));
	}
	sort(sorting_vector.begin(), sorting_vector.end(),
	     sort_pairs_second<string, size_t, greater<size_t> >());

	for (const auto &word_pair : sorting_vector) {
	    greedo_file << bitstring << " " << word_pair.first << " "
			<< word_pair.second << endl;
	}
    }
}
*/

string WordRep::Signature(size_t version) {
    ASSERT(version <= 2, "Unrecognized signature version: " << version);

    string signature = "rare" + to_string(rare_cutoff_);  // Version 0
    if (version >= 1) {
	if (sentence_per_line_) { signature += "_sentences"; }
	signature += "_window" + to_string(window_size_);
	signature += "_" + context_definition_;
	signature += "_hash" + to_string(hash_size_);
    }
    if (version >= 2) {
	signature += "_dim" + to_string(dim_);
	signature += "_" + transformation_method_;
	signature += "_add" +
	    util_string::convert_to_alphanumeric_string(add_smooth_, 2);
	signature += "_power" +
	    util_string::convert_to_alphanumeric_string(power_smooth_, 2);
	signature += "_" + scaling_method_;
    }

    return signature;
}
