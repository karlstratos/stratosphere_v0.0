// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "wordrep.h"

#include <iomanip>
#include <libgen.h>
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
    string dev_path = "../data/lexical/dev/";  // Path to dev files.
    vector<string> similarity_files = {dev_path + "wordsim353.dev",
				       dev_path + "men.dev",
				       dev_path + "rw.dev"};
    vector<string> analogy_files = {dev_path + "microsoft2013.dev",
				    dev_path + "google2013.dev",
				    dev_path + "google2013_syntactic.dev",
				    dev_path + "google2013_semantic.dev"};

    // Load word vectors.
    unordered_map<string, Eigen::VectorXd> word_vectors;
    bool normalized = true;
    corpus::load_word_vectors(WordVectorsPath(), &word_vectors, normalized);

    // Evaluate on word similarity/relatedness datasets.
    vector<size_t> num_instances;
    vector<size_t> num_handled;
    vector<double> correlation;
    eval_lexical::compute_correlation(similarity_files, word_vectors,
				      normalized, &num_instances, &num_handled,
				      &correlation);
    /*
    for (size_t i = 0; i < similarity_files.size(); ++i) {
	string file_name =
	    basename(const_cast<char*>(similarity_files[i].c_str()));
	cout << file_name << " " << num_instances[i] << " "
	     << num_handled[i] << " " << correlation[i] << endl;
    }
    */

    // Evaluate on word analogy datasets.
    vector<double> accuracy;
    vector<unordered_map<string, double> > per_type_accuracy;
    eval_lexical::compute_analogy_accuracy(
	analogy_files, word_vectors, normalized, &num_instances, &num_handled,
	&accuracy, &per_type_accuracy);

    /*
    for (size_t i = 0; i < analogy_files.size(); ++i) {
	string file_name =
	    basename(const_cast<char*>(analogy_files[i].c_str()));
	cout << file_name << " " << num_instances[i] << " "
	     << num_handled[i] << " " << accuracy[i] << endl;
	for (const auto &type_pair : per_type_accuracy[i]) {
	    cout << type_pair.first << " " << type_pair.second << endl;
	}
	cout << endl;
    }
    */
}

void WordRep::ClusterWordVectors() {
    if (!util_file::exists(WordVectorsPath())) { return; }
    if (util_file::exists(ClustersPath())) { return; }

    // Load word vectors sorted in decreasing frequency.
    unordered_map<string, Eigen::VectorXd> word_vectors;
    vector<size_t> sorted_word_counts;
    vector<string> sorted_word_strings;
    vector<Eigen::VectorXd> sorted_word_vectors;
    bool normalized = true;
    corpus::load_sorted_word_vectors(WordVectorsPath(), &sorted_word_counts,
				     &sorted_word_strings, &sorted_word_vectors,
				     normalized);

    // Agglomeratively cluster the sorted vectors to a tree with dim_ leaves.
    AgglomerativeClustering cluster;
    cluster.ClusterOrderedVectors(sorted_word_vectors, dim_);

    // Lexicographically sort bit strings for enhanced readability.
    vector<string> bitstring_types;
    for (const auto &bitstring_pair : *cluster.leaves()) {
	bitstring_types.push_back(bitstring_pair.first);
    }
    sort(bitstring_types.begin(), bitstring_types.end());

    // Write the bit strings and their associated word types.
    ofstream clusters_file(ClustersPath(), ios::out);
    unordered_map<string, vector<size_t> > *leaves = cluster.leaves();
    for (const auto &bitstring : bitstring_types) {
	vector<pair<string, size_t> > v;  // Sort word types in each cluster.
	for (Word word : leaves->at(bitstring)) {
	    string word_string = sorted_word_strings[word];
	    size_t word_count = sorted_word_counts[word];
	    v.emplace_back(word_string, word_count);
	}
	sort(v.begin(), v.end(),
	     util_misc::sort_pairs_second<string, size_t, greater<size_t> >());

	for (const auto &word_pair : v) {
	    cout << bitstring << " " << word_pair.first << " "
		 << word_pair.second << endl;
	}
    }
}

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
