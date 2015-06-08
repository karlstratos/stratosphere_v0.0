// Author: Karl Stratos (stratos@cs.columbia.edu)
//
//  Code for inducing lexical representations.

/*
#ifndef WORDREP_H
#define WORDREP_H

#include <Eigen/Dense>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

typedef size_t Word;
typedef size_t Context;

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

    // Extracts statistics from a corpus (file or a directory of files).
    void ExtractStatistics(const string &corpus_file);

    // Induces lexical representations from cached word counts.
    void InduceLexicalRepresentations();

    // Sets the rare word cutoff value.
    void set_rare_cutoff(size_t rare_cutoff) { rare_cutoff_ = rare_cutoff; }

    // Sets the flag for indicating that there is a sentence per line in the
    // text corpus.
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

    // Sets the flag for printing messages to stderr.
    void set_verbose(bool verbose) { verbose_ = verbose; }

    // Sets the number of context types to hash.
    void set_num_context_hashed(size_t num_context_hashed) {
	num_context_hashed_ = num_context_hashed;
    }

    // Sets the pseudocount for smoothing.
    void set_pseudocount(size_t pseudocount) { pseudocount_ = pseudocount; }

    // Sets the context smoothing exponent.
    void set_context_smoothing_exponent(double context_smoothing_exponent) {
	context_smoothing_exponent_ = context_smoothing_exponent;
    }

    // Sets the singular value exponent.
    void set_singular_value_exponent(double singular_value_exponent) {
	singular_value_exponent_ = singular_value_exponent;
    }

    // Returns the computed word vectors.
    unordered_map<string, Eigen::VectorXd> *wordvectors() {
	return &wordvectors_;
    }

    // Returns the singular values of the scaled count matrix.
    Eigen::VectorXd *singular_values() { return &singular_values_; }

    // Returns the special string for representing rare words.
    string kRareString() { return kRareString_; }

    // Returns the special string for buffering.
    string kBufferString() { return kBufferString_; }

    // Returns the path to the word-context count file.
    string CountWordContextPath() {
	return output_directory_ + "/count_word_context_" + Signature(1);
    }

    // Returns the path to the word count file.
    string CountWordPath() {
	return output_directory_ + "/count_word_" + Signature(0);
    }

    // Returns the path to the context count file.
    string CountContextPath() {
	return output_directory_ + "/count_context_" + Signature(1);
    }

    // Loads a filtered word dictionary from a cached file.
    void LoadWordDictionary();

    // Loads a filtered context dictionary from a cached file.
    void LoadContextDictionary();

    // Returns the integer ID corresponding to a word string.
    Word word_str2num(const string &word_string);

    // Returns the original string form of a word integer ID.
    string word_num2str(Word word);

    // Returns the integer ID corresponding to a context string.
    Context context_str2num(const string &context_string);

    // Returns the original string form of a context integer ID.
    string context_num2str(Context context);

private:
    // Extracts the count of each word type appearing in the given corpus.
    void CountWords(const string &corpus_file);

    // Adds the word to the word dictionary if not already known.
    Word AddWordIfUnknown(const string &word_string);

    // Returns true if the given word string will be skipped.
    bool SkipThisString(const string &word_string);

    // Determines rare word types.
    void DetermineRareWords();

    // Slides a window across a corpus to collect statistics.
    void SlideWindow(const string &corpus_file);

    void FinishWindow(size_t word_index,
		      const vector<string> &position_markers,
		      const hash<string> &context_hash,
		      deque<string> *window,
		      unordered_map<Context, unordered_map<Word, double> >
		      *count_word_context);

    // Increments word/context counts from a window of text.
    void ProcessWindow(const deque<string> &window,
		       size_t word_index,
		       const vector<string> &position_markers,
		       const hash<string> &context_hash,
		       unordered_map<Context, unordered_map<Word, double> >
		       *count_word_context);

    // Adds the context to the context dictionary if not already known.
    Context AddContextIfUnknown(const string &context_string_given,
				const hash<string> &context_hash);

    // Induces vector representations of word types based on cached count files.
    void InduceWordVectors();

    // Load a sorted list of word-count pairs from a cached file.
    void LoadSortedWordCounts();

    // Calculate SVD of cached count files.
    void CalculateSVD();

    // Scales a joint value by individual values.
    double ScaleJointValue(double joint_value, double value1, double value2,
			   size_t num_samples, double smoothed_sum);

    // Tests the quality of word vectors on simple tasks.
    void TestQualityOfWordVectors();

    // Performs greedy agglomerative clustering over word vectors.
    void PerformAgglomerativeClustering(size_t num_clusters);

    // Returns a string signature of tunable parameters.
    //    version=0: rare_cutoff_
    //    version=1: 0 + sentence_per_line_, window_size_, context_defintion_
    //    version=2: 1 + dim_, transformation_method_, scaling_method_,
    //                   context_smoothing_exponent_, singular_value_exponent_
    string Signature(size_t version);

    // Returns the path to the corpus information file.
    string CorpusInfoPath() { return output_directory_ + "/corpus_info"; }

    // Returns the path to the log file.
    string LogPath() { return output_directory_ + "/log"; }

    // Returns the path to the sorted word types file.
    string SortedWordTypesPath() {
	return output_directory_ + "/sorted_word_types";
    }

    // Returns the path to the rare word file.
    string RarePath() {
	return output_directory_ + "/rare_words_" + Signature(0);
    }

    // Returns the path to the str2num mapping for words.
    string WordStr2NumPath() {
	return output_directory_ + "/word_str2num_" + Signature(0);
    }

    // Returns the path to the str2num mapping for context.
    string ContextStr2NumPath() {
	return output_directory_ + "/context_str2num_" + Signature(1);
    }

    // Returns the path to the singular values.
    string SingularValuesPath() {
	return output_directory_ + "/singular_values_" + Signature(2);
    }

    // Returns the path to the word vectors.
    string WordVectorsPath() {
	return output_directory_ + "/wordvectors_" + Signature(2);
    }

    // Returns the path to the agglomeratively clusterered word vectors.
    string AgglomerativePath() {
	return output_directory_ + "/agglomerative_" + Signature(2);
    }

    // Word-count pairs sorted in decreasing frequency.
    vector<pair<string, size_t> > sorted_wordcount_;

    // Maps a word string to an integer ID.
    unordered_map<string, Word> word_str2num_;

    // Maps a word integer ID to its original string form.
    unordered_map<Word, string> word_num2str_;

    // Maps a context string to an integer ID.
    unordered_map<string, Context> context_str2num_;

    // Maps a context integer ID to its original string form.
    unordered_map<Context, string> context_num2str_;

    // Path to the log file.
    ofstream log_;

    // Special string for representing rare words.
    const string kRareString_ = "<?>";

    // Special string for representing the out-of-sentence buffer.
    const string kBufferString_ = "<!>";

    // Special string for glueing words to n-gram features.
    const string kNGramGlueString_ = "<+>";

    // Maximum word length to consider.
    const size_t kMaxWordLength_ = 100;

    // Maximum sentence length to consider.
    const size_t kMaxSentenceLength_ = 1000;

    // Interval to report progress.
    const double kReportInterval_ = 0.1;

    // Computed word vectors.
    unordered_map<string, Eigen::VectorXd> wordvectors_;

    // Matrix of word vectors (as rows).
    Eigen::MatrixXd word_matrix_;

    // Matrix of context vectors (as rows).
    Eigen::MatrixXd context_matrix_;

    // Singular values of the correlation matrix.
    Eigen::VectorXd singular_values_;

    // Path to the output directory.
    string output_directory_;

    // If a word type appears <= this number, treat it as a rare symbol.
    size_t rare_cutoff_ = 1;

    // Have a sentence per line in the text corpus?
    bool sentence_per_line_ = false;

    // Size of the context to compute covariance on. Note that it needs to be
    // odd if we want the left and right context to have the same length.
    size_t window_size_ = 3;

    // Context definition.
    string context_definition_ = "bag";

    // Target dimension of word vectors.
    size_t dim_;

    // Data transformation method.
    string transformation_method_ = "raw";

    // Scaling method.
    string scaling_method_ = "cca";

    // Number of context types to hash (0 means no hashing).
    size_t num_context_hashed_ = 0;

    // Pseudocount for smoothing.
    size_t pseudocount_ = 0;

    // Context smoothing exponent.
    double context_smoothing_exponent_ = 0.75;

    // Singular value exponent.
    double singular_value_exponent_ = 0.0;

    // Print messages to stderr?
    bool verbose_ = true;
};

#endif  // WORDREP_H
*/
