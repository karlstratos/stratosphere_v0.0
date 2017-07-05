// Author: Karl Stratos (me@karlstratos.com)

#include <iostream>
#include <string>

#include "wordrep.h"

int main (int argc, char* argv[]) {
    string corpus_path;
    string output_directory;
    bool from_scratch = false;
    bool lowercase = false;
    size_t rare_cutoff = 10;
    bool sentence_per_line = false;
    size_t window_size = 5;
    string context_definition = "bag";
    size_t hash_size = 0;  // 0 means no hashing.
    double subsampling_threshold = 1e-5;  // 0 means no subsampling.
    string cooccur_weight_method = "inv";
    size_t dim = 500;
    string transformation_method = "power";
    double add_smooth = 0.0;
    double power_smooth = 0.5;
    double context_power_smooth = 0.75;
    string scaling_method = "cca";
    string clustering_method = "agglo";
    size_t num_clusters = 0;  // 0 means same as dimension.
    size_t max_num_iterations_kmeans = 100;
    size_t num_threads = 24;
    size_t distance_type = 0;  // Squared Euclidean distance
    string seed_method = "pp";  // k-means++ initialization
    size_t num_restarts = 3;
    bool verbose = true;

    // Parse command line arguments.
    bool display_options_and_quit = false;
    for (int i = 1; i < argc; ++i) {
	string arg = (string) argv[i];
	if (arg == "--corpus") {
	    corpus_path = argv[++i];
	} else if (arg == "--output") {
	    output_directory = argv[++i];
	} else if (arg == "--force" || arg == "-f") {
	    from_scratch = true;
	} else if (arg == "--lowercase") {
	    lowercase = true;
	} else if (arg == "--rare") {
	    rare_cutoff = stol(argv[++i]);
	} else if (arg == "--sentences") {
	    sentence_per_line = true;
	} else if (arg == "--window") {
	    window_size = stol(argv[++i]);
	} else if (arg == "--context") {
	    context_definition = argv[++i];
	} else if (arg == "--hash") {
	    hash_size = stol(argv[++i]);
	} else if (arg == "--sub") {
	    subsampling_threshold = stod(argv[++i]);
	} else if (arg == "--cooccur") {
	    cooccur_weight_method = argv[++i];
	} else if (arg == "--dim") {
	    dim = stol(argv[++i]);
	} else if (arg == "--transform") {
	    transformation_method = argv[++i];
	} else if (arg == "--add") {
	    add_smooth = stod(argv[++i]);
	} else if (arg == "--power") {
	    power_smooth = stod(argv[++i]);
	} else if (arg == "--cpower") {
	    context_power_smooth = stod(argv[++i]);
	} else if (arg == "--scale") {
	    scaling_method = argv[++i];
	} else if (arg == "--cluster") {
	    clustering_method = argv[++i];
	} else if (arg == "--c") {
	    num_clusters = stol(argv[++i]);
	} else if (arg == "--iter") {
	    max_num_iterations_kmeans = stol(argv[++i]);
	} else if (arg == "--threads") {
	    num_threads = stol(argv[++i]);
	} else if (arg == "--dist") {
	    distance_type = stol(argv[++i]);
	} else if (arg == "--seed") {
	    seed_method = argv[++i];
	} else if (arg == "--restart") {
	    num_restarts = stol(argv[++i]);
	} else if (arg == "--quiet" || arg == "-q") {
	    verbose = false;
	} else if (arg == "--help" || arg == "-h"){
	    display_options_and_quit = true;
	} else {
	    cerr << "Invalid argument \"" << arg << "\": run the command with "
		 << "-h or --help to see possible arguments." << endl;
	    exit(-1);
	}
    }

    if (display_options_and_quit || argc == 1) {
	cout << "--corpus [-]:        \t"
	     << "path to a text file or a directory of text files" << endl;
	cout << "--output [-]:        \t"
	     << "path to an output directory" << endl;
	cout << "--force, -f:         \t"
	     << "forcefully recompute from scratch?" << endl;
	cout << "--lowercase:          \t"
	     << "lowercase all word strings?" << endl;
	cout << "--rare [" << rare_cutoff << "]:       \t"
	     << "word types occurring <= this are considered rare" << endl;
	cout << "--sentences:         \t"
	     << "have a sentence per line in the corpus?" << endl;
	cout << "--window [" << window_size << "]:     \t"
	     << "window size: \"word\"=center, \"context\"=non-center" << endl;
	cout << "--context [" << context_definition << "]: \t"
	     << "context definition: bag, list"  << endl;
	cout << "--hash [" << hash_size << "]:          \t"
	     << "number of hash bins for context (0 means no hashing)" << endl;
	cout << "--sub [" << subsampling_threshold << "]:      \t"
	     << "subsampling threshold (0 means no subsampling)"  << endl;
	cout << "--cooccur [" << cooccur_weight_method << "]: \t"
	     << "co-occurrence weight method: unif, inv"  << endl;
	cout << "--dim [" << dim << "]:        \t"
	     << "dimension of word vectors" << endl;
	cout << "--transform [" << transformation_method << "]: \t"
	     << "data transform: power, log"  << endl;
	cout << "--add [" << add_smooth << "]:          \t"
	     << "additive smoothing" << endl;
	cout << "--power [" << power_smooth << "]:    \t"
	     << "power smoothing" << endl;
	cout << "--cpower [" << context_power_smooth << "]:    \t"
	     << "context power smoothing" << endl;
	cout << "--scale [" << scaling_method << "]:    \t"
	     << "data scaling: none, ppmi, reg, cca" << endl;
	cout << "--cluster [" << clustering_method << "]:    \t"
	     << "clustering: agglo, div" << endl;
	cout << "--c [" << num_clusters << "]:        \t"
	     << "number of clusters (0 means same as dimension)" << endl;
	cout << "--iter [" << max_num_iterations_kmeans << "]:    \t"
	     << "maximum number of iterations in k-means" << endl;
	cout << "--threads [" << num_threads << "]:        \t"
	     << "number of threads" << endl;
	cout << "--dist [" << distance_type << "]:        \t"
	     << "distance type in k-means: 0, 1" << endl;
	cout << "--seed [" << seed_method << "]:         \t"
	     << "seed method in k-means: pp, uniform"  << endl;
	cout << "--restart [" << num_restarts << "]:        \t"
	     << "number of restarts in k-means" << endl;
	cout << "--quiet, -q:          \t"
	     << "do not print messages to stderr?" << endl;
	cout << "--help, -h:           \t"
	     << "show options and quit?" << endl;
	exit(0);
    }

    // If the number of clusters is 0, set it to the dimension.
    if (num_clusters == 0) { num_clusters = dim; }

    // Initialize a WordRep object.
    WordRep wordrep(output_directory);
    wordrep.set_lowercase(lowercase);
    wordrep.set_rare_cutoff(rare_cutoff);
    wordrep.set_sentence_per_line(sentence_per_line);
    wordrep.set_window_size(window_size);
    wordrep.set_context_definition(context_definition);
    wordrep.set_hash_size(hash_size);
    wordrep.set_subsampling_threshold(subsampling_threshold);
    wordrep.set_cooccur_weight_method(cooccur_weight_method);
    wordrep.set_dim(dim);
    wordrep.set_transformation_method(transformation_method);
    wordrep.set_add_smooth(add_smooth);
    wordrep.set_power_smooth(power_smooth);
    wordrep.set_context_power_smooth(context_power_smooth);
    wordrep.set_scaling_method(scaling_method);
    wordrep.set_clustering_method(clustering_method);
    wordrep.set_num_clusters(num_clusters);
    wordrep.set_max_num_iterations_kmeans(max_num_iterations_kmeans);
    wordrep.set_num_threads(num_threads);
    wordrep.set_distance_type(distance_type);
    wordrep.set_seed_method(seed_method);
    wordrep.set_num_restarts(num_restarts);
    wordrep.set_verbose(verbose);

    // Extract statistics from a corpus.
    if (from_scratch) { wordrep.ResetOutputDirectory(); }
    wordrep.ExtractStatistics(corpus_path);

    // Induce word vectors from cached statistics.
    wordrep.InduceWordVectors();

    // Cluster cached word vectors.
    wordrep.ClusterWordVectors(corpus_path);
}
