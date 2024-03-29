// Author: Karl Stratos (me@karlstratos.com)

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
    if (util_file::exists(output_directory_) &&
	util_file::get_file_type(output_directory_) == "file") {
	ASSERT(system(("rm -f " + output_directory_).c_str()) == 0,
	       "Cannot remove file: " << output_directory_);
    }
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
    if (corpus_file.empty()) { return; }
    Corpus corpus(corpus_file, verbose_);
    corpus.set_lowercase(lowercase_);
    corpus.set_subsampling_threshold(subsampling_threshold_);
    corpus.set_cooccur_weight_method(cooccur_weight_method_);

    // Get the word dictionary.
    unordered_map<string, Word> word_dictionary;
    if (!util_file::exists(WordDictionaryPath())) {
	size_t num_words = 0;
	size_t num_word_types = 0;
	size_t vocabulary_size = 0;

	if (!util_file::exists(SortedWordTypesPath())) {
	    // No word list found, must read in the corpus.
	    Report(util_string::buffer_string("[EXTRACTING WORD COUNTS]", 80,
					      '-', "right"));
	    time_t begin_time = time(NULL);
	    corpus.WriteWords(rare_cutoff_, SortedWordTypesPath(),
			      WordDictionaryPath(), &num_words, &num_word_types,
			      &vocabulary_size);
	    string duration = util_string::difftime_string(time(NULL),
							   begin_time);
	    Report(util_string::buffer_string("[" + duration + "]", 80, '-',
					      "right"));
	} else {
	    // Already have a word list, build a dictionary from it.
	    unordered_map<string, size_t> word_count;
	    ifstream file(SortedWordTypesPath(), ios::in);
	    ASSERT(file.is_open(), "Cannot open file: "
		   << SortedWordTypesPath());
	    while (file.good()) {
		vector<string> tokens;
		util_file::read_line(&file, &tokens);
		if (tokens.size() > 0) {
		    word_count[tokens[0]] = stol(tokens[1]);
		    num_words += stol(tokens[1]);
		}
	    }
	    num_word_types = word_count.size();

	    unordered_map<string, Word> word_dictionary_temp;
	    corpus.BuildWordDictionary(word_count, rare_cutoff_,
				       &word_dictionary_temp);
	    vocabulary_size = word_dictionary_temp.size();
	    util_file::binary_write(word_dictionary_temp, WordDictionaryPath());
	}
	Report("   Corpus: " + util_file::get_file_name(corpus_file));
	Report(util_string::printf_format(
		   "   Number of words: %ld\n"
		   "   Number of word types: %ld\n"
		   "   Vocabulary size: %d (with frequency cutoff > %ld)",
		   num_words, num_word_types, vocabulary_size, rare_cutoff_));
    }
    util_file::binary_read(WordDictionaryPath(), &word_dictionary);

    // Get the context dictionary and co-occurrence counts.
    if (!util_file::exists(ContextDictionaryPath()) ||
	!util_file::exists(ContextWordCountPath())) {
	if (subsampling_threshold_ > 0.0) {  // Prepare for random subsampling.
	    corpus.LoadWordCounts(SortedWordTypesPath(), rare_cutoff_);
	}
	Report("\n");
	Report(util_string::buffer_string(
		   "[EXTRACTING WORD-CONTEXT CO-OCCURRENCE COUNTS]", 80, '-',
		   "right"));
	Report(util_string::printf_format(
		   "   Sentence-per-line: %s\n"
		   "   Window size: %ld\n"
		   "   Context definition: %s\n"
		   "   Subsampling threshold: %.2e\n"
		   "   Co-occurrence weight method: %s\n"
		   "   Hash size: %ld",
		   sentence_per_line_ ? "true" : "false",
		   window_size_, context_definition_.c_str(),
		   subsampling_threshold_, cooccur_weight_method_.c_str(),
		   hash_size_));
	time_t begin_time = time(NULL);
	size_t num_nonzeros;
	corpus.WriteContexts(word_dictionary, sentence_per_line_,
			     context_definition_, window_size_, hash_size_,
			     ContextDictionaryPath(), ContextWordCountPath(),
			     &num_nonzeros);
	Report(util_string::printf_format(
		   "   Number of distinct word-context pairs: %ld",
		   num_nonzeros));
	string duration = util_string::difftime_string(time(NULL), begin_time);
	Report(util_string::buffer_string("[" + duration + "]", 80, '-',
					  "right"));
    }
}

void WordRep::InduceWordVectors() {
    if (util_file::exists(SingularValuesPath()) &&
	util_file::exists(WordVectorsPath())) { return; }
    SMat matrix = sparsesvd::binary_read_sparse_matrix(ContextWordCountPath());
    Report("\n");
    Report(util_string::buffer_string("[SINGULAR VALUE DECOMPOSITION]",
				      80, '-', "right"));
    Report(util_string::printf_format(
	       "   Matrix dimensions: %ld x %ld (%ld nonzeros)\n"
	       "   Desired rank: %ld\n"
	       "   Transformation method: %s\n"
	       "   Additive smoothing: %.2f\n"
	       "   Power smoothing: %.2f\n"
	       "   Context power smoothing: %.2f\n"
	       "   Scaling method: %s",
	       matrix->rows, matrix->cols, matrix->vals, dim_,
	       transformation_method_.c_str(), add_smooth_, power_smooth_,
	       context_power_smooth_, scaling_method_.c_str()));
    time_t begin_time = time(NULL);
    Eigen::MatrixXd left_singular_vectors;
    Eigen::MatrixXd right_singular_vectors;
    Eigen::VectorXd singular_values;
    corpus::decompose(matrix, dim_, transformation_method_, add_smooth_,
		      power_smooth_, context_power_smooth_, scaling_method_,
		      &left_singular_vectors, &right_singular_vectors,
		      &singular_values);
    string duration = util_string::difftime_string(time(NULL), begin_time);
    Report(util_string::buffer_string("[" + duration + "]", 80, '-',
				      "right"));
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
    unordered_map<string, Eigen::VectorXd> word_vectors;
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
	word_vectors[word_string] = left_singular_vectors.row(word);
	word_vectors[word_string].normalize();
	for (size_t j = 0; j < word_vectors[word_string].size(); ++ j) {
	    word_vectors_file << " " << word_vectors[word_string](j);
	}
	word_vectors_file << endl;
    }
    // Evaluate word vectors on lexical tasks.
    EvaluateWordVectors(word_vectors);
}

void WordRep::ClusterWordVectors(const string &corpus_file) {
    if (!util_file::exists(WordVectorsPath())) { return; }
    if (util_file::exists(ClustersPath())) { return; }

    // Load word vectors sorted in decreasing frequency.
    vector<size_t> sorted_word_counts;
    vector<string> sorted_word_strings;
    vector<Eigen::VectorXd> sorted_word_vectors;
    bool normalized = true;
    corpus::load_sorted_word_vectors(WordVectorsPath(), &sorted_word_counts,
				     &sorted_word_strings, &sorted_word_vectors,
				     normalized);

    Report(util_string::buffer_string("[CLUSTERING]", 80, '-', "right"));
    Report(util_string::printf_format("   Number of clusters: %ld\n"
				      "   Clustering method: %s",
				      num_clusters_,
				      clustering_method_.c_str()));
    if (clustering_method_ != "agglo") {
	Report(util_string::printf_format(
		   "   Maximum number of iterations: %ld\n"
		   "   Number of threads: %ld\n"
		   "   Distance type: %ld\n"
		   "   Seed method: %s\n"
		   "   Number of restarts: %ld",
		   max_num_iterations_kmeans_, num_threads_, distance_type_,
		   seed_method_.c_str(), num_restarts_));
    }
    unordered_map<string, vector<Word> > leaves;
    string duration;

    if (clustering_method_ == "agglo") {
	AgglomerativeClustering agglo;
	agglo.set_verbose(verbose_);
	time_t begin_time = time(NULL);
	double gamma = agglo.ClusterOrderedVectors(sorted_word_vectors,
						   num_clusters_);
	duration = util_string::difftime_string(time(NULL), begin_time);
	Report(util_string::printf_format("   Average number of tightening: "
					  "%.2f", gamma));
	leaves = *agglo.leaves();
    } else if (clustering_method_ == "div") {
	DivisiveClustering divisive;
	divisive.set_max_num_iterations_kmeans(max_num_iterations_kmeans_);
	divisive.set_num_threads(num_threads_);
	divisive.set_distance_type(distance_type_);
	divisive.set_seed_method(seed_method_);
	divisive.set_num_restarts(num_restarts_);
	divisive.set_verbose(verbose_);
	time_t begin_time = time(NULL);
	divisive.Cluster(sorted_word_vectors, num_clusters_, &leaves);
	duration = util_string::difftime_string(time(NULL), begin_time);
    } else {
	ASSERT(false, "Unknown clustering method: " << clustering_method_);
    }
    Report(util_string::buffer_string("[" + duration + "]", 80, '-', "right"));

    // Lexicographically sort bit strings for enhanced readability.
    vector<string> bitstring_types;
    for (const auto &bitstring_pair : leaves) {
	bitstring_types.push_back(bitstring_pair.first);
    }
    sort(bitstring_types.begin(), bitstring_types.end());

    // Write the bit strings and their associated word types.
    ofstream clusters_file(ClustersPath(), ios::out);
    unordered_map<string, string> word_to_cluster;  // Used later.
    for (const auto &bitstring : bitstring_types) {
	vector<pair<string, size_t> > v;  // Sort word types in each cluster.
	for (Word word : leaves[bitstring]) {
	    string word_string = sorted_word_strings[word];
	    size_t word_count = sorted_word_counts[word];
	    v.emplace_back(word_string, word_count);
	    word_to_cluster[word_string] = bitstring;
	}
	sort(v.begin(), v.end(),
	     util_misc::sort_pairs_second<string, size_t, greater<size_t> >());

	for (const auto &word_pair : v) {
	    clusters_file << bitstring << " " << word_pair.first << " "
			  << word_pair.second << endl;
	}
    }

    if (!corpus_file.empty()) {
	double mi = ComputeClusterMI(corpus_file, word_to_cluster);
	Report(util_string::printf_format("   Cluster MI: %.3f", mi));
    }
}

void WordRep::EvaluateWordVectors(const unordered_map<string, Eigen::VectorXd>
				  &word_vectors) {
    string dev_path = "../data/lexical/dev/";  // Path to dev files.
    vector<string> similarity_files = {dev_path + "wordsim353.dev",
				       dev_path + "men.dev",
				       dev_path + "rw.dev"};
    vector<string> analogy_files = {dev_path + "microsoft2013.dev",
				    dev_path + "google2013.dev",
				    dev_path + "google2013_syntactic.dev",
				    dev_path + "google2013_semantic.dev"};
    bool normalized = true;  // Word vectors are already normalized.

    // Evaluate on word similarity/relatedness datasets.
    Report("\n");
    Report(util_string::buffer_string("[CORRELATION IN SIMILARITY SCORES]",
				      80, '-', "right"));
    time_t begin_time = time(NULL);
    vector<size_t> num_instances;
    vector<size_t> num_handled;
    vector<double> correlation;
    size_t max_file_name_length = 0;
    for (const string &file_name : similarity_files) {
	max_file_name_length = max(max_file_name_length,
				   file_name.size() - dev_path.size());
    }
    eval_lexical::compute_correlation(similarity_files, word_vectors,
				      normalized, &num_instances, &num_handled,
				      &correlation);
    double average_correlation = 0.0;
    for (size_t i = 0; i < similarity_files.size(); ++i) {
	string file_name = util_string::buffer_string(
	    util_file::get_file_name(similarity_files[i]),
	    max_file_name_length + 3, ' ', "right");
	Report(util_string::printf_format(
		   "%s   %.3f   (%d/%d evaluated)",
		   file_name.c_str(), correlation[i], num_handled[i],
		   num_instances[i]));
	average_correlation += correlation[i] / similarity_files.size();
    }
    Report(util_string::printf_format("   Average correlation: %.3f",
				      average_correlation));
    string duration = util_string::difftime_string(time(NULL), begin_time);
    Report(util_string::buffer_string("[" + duration + "]", 80, '-',
				      "right"));

    // Evaluate on word analogy datasets.
    Report("\n");
    Report(util_string::buffer_string("[ACCURACY IN ANALOGY QUESTIONS]",
				      80, '-', "right"));
    begin_time = time(NULL);
    for (const string &file_name : analogy_files) {
	max_file_name_length = max(max_file_name_length,
				   file_name.size() - dev_path.size());
    }
    vector<double> accuracy;
    vector<unordered_map<string, double> > per_type_accuracy;
    eval_lexical::compute_analogy_accuracy(
	analogy_files, word_vectors, normalized, &num_instances, &num_handled,
	&accuracy, &per_type_accuracy);
    for (size_t i = 0; i < analogy_files.size(); ++i) {
	string file_name = util_string::buffer_string(
	    util_file::get_file_name(analogy_files[i]),
	    max_file_name_length + 3, ' ', "right");
	Report(util_string::printf_format(
		   "%s   %.2f   (%d/%d evaluated)",
		   file_name.c_str(), accuracy[i], num_handled[i],
		   num_instances[i]));
	if (report_details_) {
	    vector<pair<string, double> > v;
	    size_t max_type_name_length = 0;
	    for (const auto &type_pair : per_type_accuracy[i]) {
		v.emplace_back(type_pair.first, type_pair.second);
		max_type_name_length = max(type_pair.first.size(),
					   max_type_name_length);
	    }
	    sort(v.begin(), v.end(), util_misc::sort_pairs_second<string,
		 size_t, greater<size_t> >());
	    Report(util_string::buffer_string("_____", 80, ' ', "right"));
	    for (const auto &sorted_pair : v) {
		string type_name = util_string::buffer_string(
		    sorted_pair.first, max_type_name_length + 5, ' ',
		    "center");
		string line = util_string::buffer_string(
		    util_string::printf_format("%s %.2f", type_name.c_str(),
					       sorted_pair.second),
		    80, ' ', "right");
		Report(line);
	    }
	    Report(util_string::buffer_string("-----", 80, ' ', "right"));
	}
    }
    duration = util_string::difftime_string(time(NULL), begin_time);
    Report(util_string::buffer_string("[" + duration + "]", 80, '-',
				      "right"));
}

double WordRep::ComputeClusterMI(const string &corpus_file,
				 const unordered_map<string, string>
				 &word_to_cluster) {
    // Count cluster cooccurrences.
    size_t num_samples = 0;
    unordered_map<string, unordered_map<string, size_t> > c1c2_count;
    unordered_map<string, size_t> c1_count;
    unordered_map<string, size_t> c2_count;

    vector<string> file_list;
    util_file::list_files(corpus_file, &file_list);
    bool not_first = false;
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string c1;
	string file_path = file_list[file_num];
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	while (file.good()) {
	    vector<string> word_strings;
	    util_file::read_line(&file, &word_strings);
	    for (size_t i = 0; i < word_strings.size(); ++i) {
		string word_string = (!lowercase_) ? word_strings[i] :
		    util_string::lowercase(word_strings[i]);
		if (word_to_cluster.find(word_string) ==
		    word_to_cluster.end()) {
		    word_string = corpus::kRareString;
		}

		string c = word_to_cluster.at(word_string);
		if (not_first) {
		    ++num_samples;
		    ++c1c2_count[c1][c];
		    ++c1_count[c1];
		    ++c2_count[c];
		} else {
		    not_first = true;
		}
		c1 = c;
	    }
	}
    }

    double mi = 0.0;
    for (auto &c1_pair : c1c2_count) {
	string c1 = c1_pair.first;
	for (auto &c2_pair : c1c2_count[c1]) {
	    string c2 = c2_pair.first;
	    double prob_c1c2 = float(c2_pair.second) / num_samples;
	    double pmi = log2(float(c2_pair.second) * num_samples /
			      c1_count[c1] / c2_count[c2]);  // Base 2 log.
	    mi += prob_c1c2 * pmi;
	}
    }

    return mi;
}

string WordRep::Signature(size_t version) {
    ASSERT(version <= 4, "Unrecognized signature version: " << version);

    string signature = (lowercase_) ? "lowercased" : "caseintact";  // Version 0

    if (version >= 1) { signature += "_rare" + to_string(rare_cutoff_); }

    if (version >= 2) {
	if (sentence_per_line_) { signature += "_sentences"; }
	signature += "_window" + to_string(window_size_);
	signature += "_" + context_definition_;
	signature += "_sub" +
	    util_string::convert_to_alphanumeric_string(subsampling_threshold_,
							10);
	signature += "_" + cooccur_weight_method_;
	signature += "_hash" + to_string(hash_size_);
    }

    if (version >= 3) {
	signature += "_dim" + to_string(dim_);
	signature += "_" + transformation_method_;
	signature += "_add" +
	    util_string::convert_to_alphanumeric_string(add_smooth_, 2);
	signature += "_power" +
	    util_string::convert_to_alphanumeric_string(power_smooth_, 2);
	signature += "_cpower" +
	    util_string::convert_to_alphanumeric_string(context_power_smooth_,
							2);
	signature += "_" + scaling_method_;
    }

    if (version >= 4) {
	signature += "_c" + to_string(num_clusters_);
	signature += "_" + clustering_method_;
	if (clustering_method_ != "agglo") {
	    signature += "_iter" + to_string(max_num_iterations_kmeans_);
	    signature += "_dist" + to_string(distance_type_);
	    signature += "_" + seed_method_;
	    signature += "_restarts" + to_string(num_restarts_);
	}
    }

    return signature;
}

void WordRep::Report(const string &report_string) {
    ofstream log_file(LogPath(), ios::out | ios::app);
    log_file << report_string << endl;
    if (verbose_) { cerr << report_string << endl; }
}
