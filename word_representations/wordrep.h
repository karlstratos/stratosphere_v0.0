// Author: Karl Stratos (stratos@cs.columbia.edu)
//
//  Code for inducing word representations.

#ifndef WORD_REPRESENTATIONS_WORDREP_H_
#define WORD_REPRESENTATIONS_WORDREP_H_

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class WordRep {
public:
    // Initializes empty.
    WordRep() { }

    // Initializes with an output directory.
    WordRep(const string &output_directory) {
	SetOutputDirectory(output_directory);
    }

    ~WordRep() { }

    // Sets the output directory.
    void SetOutputDirectory(const string &output_directory);

    // Resets the content in the output directory.
    void ResetOutputDirectory();

    // Extracts statistics from a corpus.
    void ExtractStatistics(const string &corpus_file);

    // Induces word vectors from cached word counts.
    void InduceWordVectors();

    // Evaluate word vectors on lexical tasks.
    void EvaluateWordVectors();

    // Sets the rare word cutoff value.
    void set_rare_cutoff(size_t rare_cutoff) { rare_cutoff_ = rare_cutoff; }

    // Sets whether there is a sentence per line in the text corpus.
    void set_sentence_per_line(bool sentence_per_line) {
	sentence_per_line_ = sentence_per_line;
    }

    // Sets the context window size.
    void set_window_size(size_t window_size) { window_size_ = window_size; }

    // Sets the context definition.
    void set_context_definition(string context_definition) {
	context_definition_ = context_definition;
    }

    // Sets the target dimension of word vectors.
    void set_dim(size_t dim) { dim_ = dim; }

    // Sets the transformation method.
    void set_transformation_method(string transformation_method) {
	transformation_method_ = transformation_method;
    }

    // Sets the scaling method.
    void set_scaling_method(string scaling_method) {
	scaling_method_ = scaling_method;
    }

    // Sets the number of hash bins for context types.
    void set_hash_size(size_t hash_size) { hash_size_ = hash_size; }

    // Sets the additive smoothing value.
    void set_add_smooth(double add_smooth) { add_smooth_ = add_smooth; }

    // Sets the power smoothing value.
    void set_power_smooth(double power_smooth) { power_smooth_ = power_smooth; }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    // Evalutes word vectors on simple tasks.
    void EvaluateWordVectors();

    // Performs greedy agglomerative clustering over word vectors.
    void PerformAgglomerativeClustering(size_t num_clusters);

    // Returns the path to the corpus information file.
    string CorpusInfoPath() { return output_directory_ + "/corpus_info.txt"; }

    // Returns the path to the log file.
    string LogPath() {
	return output_directory_ + "/log_" + Signature(2) + ".txt";
    }

    // Returns the path to the sorted word types file.
    string SortedWordTypesPath() {
	return output_directory_ + "/sorted_word_types.txt";
    }

    // Returns the path to the word dictionary file.
    string WordDictionaryPath() {
	return output_directory_ + "/word_dictionary_" + Signature(0) + ".bin";
    }

    // Returns the path to the context dictionary file.
    string ContextDictionaryPath() {
	return output_directory_ + "/context_dictionary_" + Signature(1) +
	    ".bin";
    }

    // Returns the path to the word-context count file.
    string ContextWordCountPath() {
	return output_directory_ + "/context_word_count_" + Signature(1) +
	    ".bin";
    }

    // Returns the path to the singular values.
    string SingularValuesPath() {
	return output_directory_ + "/singular_values_" + Signature(2) + ".txt";
    }

    // Returns the path to the word vectors.
    string WordVectorsPath() {
	return output_directory_ + "/word_vectors_" + Signature(2) + ".txt";
    }

    // Returns the path to the agglomeratively clusterered word vectors.
    string AgglomerativePath() {
	return output_directory_ + "/agglomerative_" + Signature(2);
    }

    // Returns a string signature of parameters.
    //    version=0: {rare_cutoff_}
    //    version=1: 0 + {sentence_per_line_, window_size_, context_defintion_,
    //                    hash_size_}
    //    version=2: 1 + {dim_, transformation_method_, add_smooth_,
    //                    power_smooth_, scaling_method_}
    string Signature(size_t version);

    // Maximum word length to consider.
    const size_t kMaxWordLength_ = 100;

    // Maximum sentence length to consider.
    const size_t kMaxSentenceLength_ = 1000;

    // Path to the output directory.
    string output_directory_;

    // Log string.
    string log_;

    // If a word type appears <= this number, treat it as a rare symbol.
    size_t rare_cutoff_ = 1;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the sliding window (odd => symmetric, even => assymmetric).
    size_t window_size_ = 3;

    // Context definition.
    string context_definition_ = "bag";

    // Number of hash bins for context types (0 means no hashing).
    size_t hash_size_ = 0;

    // Target dimension of word vectors.
    size_t dim_;

    // Data transformation method.
    string transformation_method_ = "power";

    // Additive smoothing value.
    double add_smooth_ = 0.0;

    // Power smoothing value.
    double power_smooth_ = 0.5;

    // Scaling method.
    string scaling_method_ = "cca";

    // Print messages to stderr?
    bool verbose_ = true;
};

#endif  // WORD_REPRESENTATIONS_WORDREP_H_
