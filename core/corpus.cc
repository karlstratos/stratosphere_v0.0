// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "corpus.h"

#include <iomanip>
#include <limits>

#include "sparsesvd.h"

void Window::Add(const string &word_string) {
    // Filter words before putting in the window.
    string word_string_filtered = (word_dictionary_.find(word_string) !=
				   word_dictionary_.end()) ?
	word_string : rare_symbol_;
    queue_.push_back(word_string_filtered);  // [dog saw the]--[dog saw the cat]
    if (queue_.size() == window_size_) {
	ProcessFull();
	queue_.pop_front();  // [dog saw the cat]--[saw the cat]
    }
}

void Window::Finish() {
    size_t num_window_elements = queue_.size();
    while (queue_.size() < window_size_) {
	// This can happen if the number of words added to the window was
	// smaller than the window size (note that in this case the window was
	// never processed). So first fill up the window:
	queue_.push_back(buffer_symbol_);  // [<!> he did]--[<!> he did <!>]
    }
    for (size_t i = center_index_; i < num_window_elements; ++i) {
	ProcessFull();
	queue_.pop_front();  // [<!> he did <!>]--[he did <!>]
	queue_.push_back(buffer_symbol_);  // [he did <!>]--[he did <!> <!>]
    }
    queue_.clear();
    for (size_t i = 0; i < center_index_; ++i) {
	queue_.push_back(buffer_symbol_);
    }
}

void Window::PrepareWindow() {
    center_index_ = (window_size_ - 1) / 2;  // Right-biased center index.

    // Buffer the window up to before the center index.
    for (size_t i = 0; i < center_index_; ++i) {
	queue_.push_back(buffer_symbol_);
    }

    // Initialize the string markers for position-sensitive contexts.
    position_markers_.resize(window_size_);
    for (size_t i = 0; i < window_size_; ++i) {
	if (i != center_index_) {
	    position_markers_[i] =
		"c(" + to_string((int(i)) - (int(center_index_))) + ")=";
	}
    }
}

void Window::ProcessFull() {
    Word word = word_dictionary_.at(queue_[center_index_]);

    for (size_t i = 0; i < window_size_; ++i) {
	if (i == center_index_) { continue; }
	if (context_definition_ == "bag") {  // Bag-of-words contexts
	    Context bag_context = AddContextString(queue_[i]);
	    context_word_count_[bag_context][word] += 1;
	} else if (context_definition_ == "list") {  // List-of-words contexts
	    Context list_context = AddContextString(position_markers_[i] +
						    queue_[i]);
	    context_word_count_[list_context][word] += 1;
	} else {
	    ASSERT(false, "Unknown context definition: " <<
		   context_definition_);
	}
    }
}

Context Window::AddContextString(const string &context_string) {
    string context_string_hashed = (hash_size_ == 0) ?  // Context hashing
	context_string : to_string(context_hash_(context_string) % hash_size_);

    if (context_dictionary_.find(context_string_hashed) ==
	context_dictionary_.end()) {  // Add to dictionary if not known.
	Context new_context_index = context_dictionary_.size();
	context_dictionary_[context_string_hashed] = new_context_index;
    }
    return context_dictionary_[context_string_hashed];
}

/*
size_t Corpus::CountWords(unordered_map<string, size_t> *count) {
    size_t num_words = 0;
    vector<string> file_list;
    file_manipulator_.ListFiles(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = file_manipulator_.NumLines(file_path);
	if (verbose_) {
	    cerr << "Counting words in file " << file_num + 1 << "/"
		 << file_list.size() << " " << flush;
	}
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	double portion_so_far = kReportInterval_;
	double line_num = 0.0;  // Float for division
	while (file.good()) {
	    vector<string> word_strings;
	    ++line_num;
	    string_manipulator_.ReadLine(&file, &word_strings);
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    for (string word_string : word_strings) {
		if (Skip(word_string)) { continue; }
		if (lowercase_) {
		    word_string = string_manipulator_.Lowercase(word_string);
		}
		++(*count)[word_string];
		++num_words;
	    }
	    if (verbose_ && (line_num / num_lines >= portion_so_far)) {
		portion_so_far += kReportInterval_;
		cerr << "." << flush;
	    }
	}
	if (verbose_) { cerr << " " << count->size() << " types" << endl; }
    }
    return num_words;
}

size_t Corpus::BuildWordDictionary(const unordered_map<string, size_t> &count,
				   size_t rare_cutoff,
				   unordered_map<string, Word>
				   *word_dictionary) {
    size_t num_considered_words = 0;
    bool have_rare = false;
    word_dictionary->clear();
    for (const auto &string_count_pair : count) {
	string word_string = string_count_pair.first;
	size_t word_count = string_count_pair.second;
	if (word_count > rare_cutoff) {
	    num_considered_words += word_count;
	    (*word_dictionary)[word_string] = word_dictionary->size();
	} else {
	    have_rare = true;
	}
    }
    if (have_rare) {  // The rare word symbol gets the highest index.
	(*word_dictionary)[kRareString()] = word_dictionary->size();
    }
    return num_considered_words;
}

void Corpus::SlideWindow(const unordered_map<string, Word> &word_dictionary,
			 bool sentence_per_line,
			 const string &context_definition, size_t window_size,
			 size_t hash_size,
			 unordered_map<string, Context> *context_dictionary,
			 unordered_map<Context, unordered_map<Word, double> >
			 *context_word_count) {
    ASSERT(window_size >= 2, "Window size less than 2: " << window_size);
    Window window(window_size, word_dictionary, kBufferString_, kRareString_,
		  context_definition, hash_size, context_dictionary,
		  context_word_count);

    vector<string> file_list;
    file_manipulator_.ListFiles(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = file_manipulator_.NumLines(file_path);
	if (verbose_) {
	    cerr << "Sliding window in file " << file_num + 1 << "/"
		 << file_list.size() << " " << flush;
	}
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	double portion_so_far = kReportInterval_;
	double line_num = 0.0;  // Float for division
	while (file.good()) {
	    vector<string> word_strings;
	    string_manipulator_.ReadLine(&file, &word_strings);
	    ++line_num;
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    for (string word_string : word_strings) {
		if (Skip(word_string)) { continue; }
		if (lowercase_) {
		    word_string = string_manipulator_.Lowercase(word_string);
		}
		window.Add(word_string);
	    }
	    if (sentence_per_line) { window.Finish(); } // Finish the line.
	    if (verbose_ && (line_num / num_lines >= portion_so_far)) {
		portion_so_far += kReportInterval_;
		cerr << "." << flush;
	    }
	}
	if (!sentence_per_line) { window.Finish(); } // Finish the file.
	if (verbose_) { cerr << endl; }
    }
}

void Corpus::CountTransitions(
    const unordered_map<string, Word> &word_dictionary,
    unordered_map<Word, unordered_map<Word, size_t> > *bigram_count,
    unordered_map<Word, size_t> *start_count,
    unordered_map<Word, size_t> *end_count) {
    bigram_count->clear();
    start_count->clear();
    end_count->clear();

    vector<string> file_list;
    file_manipulator_.ListFiles(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = file_manipulator_.NumLines(file_path);
	if (verbose_) {
	    cerr << "Counting transitions in file " << file_num + 1 << "/"
		 << file_list.size() << " " << flush;
	}
	ifstream file(file_path, ios::in);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	double portion_so_far = kReportInterval_;
	double line_num = 0.0;  // Float for division
	while (file.good()) {
	    vector<string> word_strings;
	    string_manipulator_.ReadLine(&file, &word_strings);
	    ++line_num;
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    Word w_prev;
	    for (size_t i = 0; i < word_strings.size(); ++i) {
		string word_string = (!lowercase_) ? word_strings[i] :
		    string_manipulator_.Lowercase(word_strings[i]);
		if (Skip(word_string)) { continue; }
		if (word_dictionary.find(word_string) ==
		    word_dictionary.end()) { word_string = kRareString_; }
		Word w = word_dictionary.at(word_string);

		if (i == 0) { ++(*start_count)[w]; }
		if (i == word_strings.size() - 1) { ++(*end_count)[w]; }
		if (i > 0) { ++(*bigram_count)[w_prev][w]; }
		w_prev = w;
	    }
	    if (verbose_ && (line_num / num_lines >= portion_so_far)) {
		portion_so_far += kReportInterval_;
		cerr << "." << flush;
	    }
	}
	if (verbose_) { cerr << endl; }
    }
}

void Corpus::CreateSortedWordTypesFile(bool recompute,
				       const string &sorted_word_types_path) {
    if (!recompute && file_manipulator_.Exists(sorted_word_types_path)) {
	return;  // Already have the file.
    }
    unordered_map<string, size_t> word_count;

    //size_t num_words = CountWords(&word_count);
    CountWords(&word_count);

    // Sort word types in decreasing frequency.
    vector<pair<string, size_t> > sorted_word_count(word_count.begin(),
						    word_count.end());
    sort(sorted_word_count.begin(), sorted_word_count.end(),
	 sort_pairs_second<string, size_t, greater<size_t> >());

    ofstream sorted_word_types_file(sorted_word_types_path, ios::out);
    for (size_t i = 0; i < sorted_word_count.size(); ++i) {
	sorted_word_types_file << sorted_word_count[i].first << " "
			       << sorted_word_count[i].second << endl;
    }
}

void Corpus::CreateWordDictionaryFile(bool recompute,
				      const unordered_map<string, size_t>
				      &word_count, size_t rare_cutoff,
				      const string &word_dictionary_path) {
    if (!recompute && file_manipulator_.Exists(word_dictionary_path)) {
	return;  // Already have the file.
    }

    unordered_map<string, size_t> word_dictionary;
    BuildWordDictionary(word_count, rare_cutoff, &word_dictionary);
    file_manipulator_.Write(word_dictionary, word_dictionary_path);
}

void Corpus::CreateCoOccurrenceFiles(bool recompute,
				     const unordered_map<string, Word>
				     &word_dictionary, bool sentence_per_line,
				     const string &context_definition,
				     size_t window_size, size_t hash_size,
				     const string &context_word_count_path,
				     const string &context_dictionary_path) {
    if (!recompute && file_manipulator_.Exists(context_word_count_path) &&
	file_manipulator_.Exists(context_dictionary_path)) {
	return;  // Already have the files.
    }

    unordered_map<string, Context> context_dictionary;
    unordered_map<Context, unordered_map<Word, double> > context_word_count;
    SlideWindow(word_dictionary, sentence_per_line, context_definition,
		window_size, hash_size, &context_dictionary,
		&context_word_count);

    file_manipulator_.Write(context_dictionary, context_dictionary_path);

    // Write co-occurrence counts as a sparse matrix for SVDLIBC.
    size_t num_samples = 0;
    for (const auto &context_pair : context_word_count) {
	num_samples += context_pair.second.size();
    }
    SparseSVDSolver sparsesvd_solver;
    sparsesvd_solver.WriteSparseMatrix(context_word_count,
				       context_word_count_path,
				       word_dictionary.size(),
				       context_dictionary.size(), num_samples);
}

void Corpus::CreateTransitionFiles(bool recompute,
				   const unordered_map<string, Word>
				   &word_dictionary,
				   const string &bigram_count_path,
				   const string &start_count_path,
				   const string &end_count_path) {
    unordered_map<Word, unordered_map<Word, size_t> > bigram_count;
    unordered_map<Word, size_t> start_count;
    unordered_map<Word, size_t> end_count;
    CountTransitions(word_dictionary, &bigram_count, &start_count, &end_count);

    file_manipulator_.Write(bigram_count, bigram_count_path);
    file_manipulator_.Write(start_count, start_count_path);
    file_manipulator_.Write(end_count, end_count_path);
}

bool Corpus::Skip(const string &word_string) {
    return (word_string == kRareString_ ||  // Is the special "rare" symbol.
	    word_string == kBufferString_ ||  // Is the special "buffer" symbol.
	    word_string.size() > kMaxWordLength_);  // Is too long.
}
*/
