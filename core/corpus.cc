// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "corpus.h"

#include <iomanip>
#include <limits>

#include "sparsesvd.h"

size_t Corpus::CountWords(unordered_map<string, size_t> *word_count) {
    size_t num_words = 0;
    vector<string> file_list;
    util_file::list_files(corpus_path_, &file_list);
    for (size_t file_num = 0; file_num < file_list.size(); ++file_num) {
	string file_path = file_list[file_num];
	size_t num_lines = util_file::get_num_lines(file_path);
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
	    util_string::read_line(&file, &word_strings);
	    if (word_strings.size() > kMaxSentenceLength_) { continue; }
	    for (string word_string : word_strings) {
		if (Skip(word_string)) { continue; }
		if (lowercase_) {
		    word_string = util_string::lowercase(word_string);
		}
		++(*word_count)[word_string];
		++num_words;

		// If the vocabulary is too large, subtract by the median count
		// and eliminate at least half of the word types.
		if (word_count->size() >= kMaxVocabularySize_) {
		    util_misc::subtract_by_median(word_count);
		}
	    }
	    if (line_num / num_lines >= portion_so_far) {
		portion_so_far += kReportInterval_;
		if (verbose_) { cerr << "." << flush; }
	    }
	}
	if (verbose_) { cerr << " " << word_count->size() << " types" << endl; }
    }
    return num_words;
}

bool Corpus::Skip(const string &word_string) {
    return (word_string == kRareString_ ||  // Is the special "rare" symbol.
	    word_string == kBufferString_ ||  // Is the special "buffer" symbol.
	    word_string.size() > kMaxWordLength_);  // Is too long.
}

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
