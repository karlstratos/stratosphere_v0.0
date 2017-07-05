// Author: Karl Stratos (me@karlstratos.com)
//
//  Code for inducing word representations.

#ifndef WORD_REPRESENTATION_WORDREP_H_
#define WORD_REPRESENTATION_WORDREP_H_

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

    // Clusters cached word vectors.
    void ClusterWordVectors(const string &corpus_file);

    // Sets the flag for lowercasing all word strings.
    void set_lowercase(bool lowercase) { lowercase_ = lowercase; }

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

    // Sets the subsampling threshold.
    void set_subsampling_threshold(double subsampling_threshold) {
	subsampling_threshold_ = subsampling_threshold;
    }

    // Sets the co-occurrence weight method.
    void set_cooccur_weight_method(string cooccur_weight_method) {
	cooccur_weight_method_ = cooccur_weight_method;
    }

    // Sets the number of hash bins for context types.
    void set_hash_size(size_t hash_size) { hash_size_ = hash_size; }

    // Sets the target dimension of word vectors.
    void set_dim(size_t dim) { dim_ = dim; }

    // Sets the transformation method.
    void set_transformation_method(string transformation_method) {
	transformation_method_ = transformation_method;
    }

    // Sets the additive smoothing value.
    void set_add_smooth(double add_smooth) { add_smooth_ = add_smooth; }

    // Sets the power smoothing value.
    void set_power_smooth(double power_smooth) { power_smooth_ = power_smooth; }

    // Sets the context power smoothing value.
    void set_context_power_smooth(double context_power_smooth) {
	context_power_smooth_ = context_power_smooth;
    }

    // Sets the scaling method.
    void set_scaling_method(string scaling_method) {
	scaling_method_ = scaling_method;
    }

    // Sets the clustering method.
    void set_clustering_method(string clustering_method) {
	clustering_method_ = clustering_method;
    }

    // Sets the number of clusters.
    void set_num_clusters(size_t num_clusters) { num_clusters_ = num_clusters; }

    // Sets the maximum number of iterations in k-means.
    void set_max_num_iterations_kmeans(size_t max_num_iterations_kmeans) {
	max_num_iterations_kmeans_ = max_num_iterations_kmeans;
    }

    // Sets the number of threads.
    void set_num_threads(size_t num_threads) { num_threads_ = num_threads; }

    // Sets the distance type in k-means.
    void set_distance_type(size_t distance_type) {
	distance_type_ = distance_type;
    }

    // Sets the seed method in k-means.
    void set_seed_method(const string &seed_method) {
	seed_method_ = seed_method;
    }

    // Sets the number of restarts in k-means.
    void set_num_restarts(size_t num_restarts) {
	num_restarts_ = num_restarts;
    }

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

private:
    // Evaluates word vectors on lexical tasks.
    void EvaluateWordVectors(const unordered_map<string, Eigen::VectorXd>
			     &word_vectors);

    // Computes mutual information of adjacent clusters in the corpus.
    double ComputeClusterMI(const string &corpus_file, const
			    unordered_map<string, string> &word_to_cluster);

    // Returns a string signature of parameters.
    //    version=0: {lowercase_}
    //    version=1: 0 + {rare_cutoff_}
    //    version=2: 1 + {sentence_per_line_, window_size_, context_defintion_,
    //                    subsampling_threshold_, cooccur_weight_method,
    //                    hash_size_}
    //    version=3: 2 + {dim_, transformation_method_, add_smooth_,
    //                    power_smooth_, context_power_smooth_, scaling_method_}
    //    version=4: 3 + {num_clusters_, clustering_method_,
    //                    max_num_iterations_kmeans_, distance_type_,
    //                    seed_method_, num_restarts_}
    string Signature(size_t version);

    // Reports status in a log file and optionally the standard output.
    void Report(const string &report_string);

    // Returns the path to the log file.
    string LogPath() {
	return output_directory_ + "/log_" + Signature(4) + ".txt";
    }

    // Returns the path to the sorted word types file.
    string SortedWordTypesPath() {
	return output_directory_ + "/wcount_" + Signature(0) + ".txt";
    }

    // Returns the path to the word dictionary file.
    string WordDictionaryPath() {
	return output_directory_ + "/wdict_" + Signature(1) + ".bin";
    }

    // Returns the path to the context dictionary file.
    string ContextDictionaryPath() {
	return output_directory_ + "/cdict_" + Signature(2) + ".bin";
    }

    // Returns the path to the word-context count file.
    string ContextWordCountPath() {
	return output_directory_ + "/cooccur_" + Signature(2) + ".bin";
    }

    // Returns the path to the singular values.
    string SingularValuesPath() {
	return output_directory_ + "/sigma_" + Signature(3) + ".txt";
    }

    // Returns the path to the word vectors.
    string WordVectorsPath() {
	return output_directory_ + "/vector_" + Signature(3) + ".txt";
    }

    // Returns the path to the clusterered word vectors.
    string ClustersPath() {
	return output_directory_ + "/cluster_" + Signature(4) + ".txt";
    }

    // Report details to stderr.
    const bool report_details_ = false;

    // Path to the output directory.
    string output_directory_;

    // Lowercase all word strings?
    bool lowercase_ = false;

    // If a word type appears <= this number, treat it as a rare symbol.
    size_t rare_cutoff_ = 1;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the sliding window (odd => symmetric, even => assymmetric).
    size_t window_size_ = 3;

    // Context definition.
    string context_definition_ = "bag";

    // Subsampling threshold.
    double subsampling_threshold_ = 0.0;

    // Co-occurrence weight method.
    string cooccur_weight_method_ = "unif";

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

    // Context power smoothing value.
    double context_power_smooth_ = 0.75;

    // Scaling method.
    string scaling_method_ = "cca";

    // Clustering method.
    string clustering_method_ = "agglo";

    // Number of clusters.
    size_t num_clusters_;

    // Maximum number of iterations in k-means.
    size_t max_num_iterations_kmeans_ = 100;

    // Number of threads.
    size_t num_threads_ = 24;

    // Distance type in k-means.
    size_t distance_type_ = 0;  // Squared Euclidean distance

    // Seed method in k-means.
    string seed_method_ = "pp";  // k-means++ initialization

    // Number of restarts in k-means.
    size_t num_restarts_ = 3;

    // Print messages to stderr?
    bool verbose_ = true;
};

#endif  // WORD_REPRESENTATION_WORDREP_H_
