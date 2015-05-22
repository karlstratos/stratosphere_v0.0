// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for processing text corpora.

#ifndef CORE_CORPUS_H_
#define CORE_CORPUS_H_

#include <deque>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "util.h"

typedef size_t Word;
typedef size_t Context;

// A Corpus object is designged to extract useful statistics about text easily.
class Corpus {
public:
    // Always initializes with a corpus.
    Corpus(const string &corpus_path, bool verbose) : corpus_path_(corpus_path),
						      verbose_(verbose) { }

    ~Corpus() { }

    // Counts words appearing in the corpus, returns the total count.
    size_t CountWords(unordered_map<string, size_t> *word_count);

    // Sets the flag for lowercasing all strings.
    void set_lowercase(bool lowercase) { lowercase_ = lowercase; }

    // Returns the corpus path.
    string corpus_path() { return corpus_path_; }

    // Returns the special string for representing rare words.
    string kRareString() { return kRareString_; }

    // Returns the special string for buffering.
    string kBufferString() { return kBufferString_; }

private:
    // Returns true if the string is to be skipped.
    bool Skip(const string &word_string);

    // Maximum sentence length to consider.
    const size_t kMaxSentenceLength_ = 1000;

    // Maximum word length to consider.
    const size_t kMaxWordLength_ = 100;

    // Maximum vocabulary size allowed.
    const size_t kMaxVocabularySize_ = 10000000;  // 10 million

    // Interval to report progress (0.1 => every 10%).
    const double kReportInterval_ = 0.1;

    // Special string for representing rare words.
    const string kRareString_ = "<?>";

    // Special string for representing the out-of-sentence buffer.
    const string kBufferString_ = "<!>";

    // Path to a corpus (a single text file, or a directory of text files).
    string corpus_path_;

    // Lowercase all strings?
    bool lowercase_ = false;

    // Print messages to stderr?
    bool verbose_ = true;
};

// A Window object is used to "slide" a window of fixed size and count
// word-context co-occurrences. The definition of a "context" is customized
// (bag-of-words, list, etc.) and a context dictionary is built on the fly.
// Optionally, contexts can be randomly hashed to control their number.
//
// E.g., Bag-of-words contexts, window size 4, sentence "the dog saw the cat":
//        window            call                count([word], [context])
//   [<!>]
//   [<!> the]           Add("the")
//   [<!> the dog]       Add("dog")
//   [<!> the dog saw]   Add("saw")       ++(the, <!>) ++(the, dog) ++(the, saw)
//   [the dog saw the]   Add("the")       ++(dog, the) ++(dog, saw) ++(dog, the)
//   [dog saw the cat]   Add("cat")       ++(saw, dog) ++(saw, the) ++(saw, cat)
//   ______________________________
//   [saw the cat <!>]   Finish()         ++(the, saw) ++(the, cat) ++(the, <!>)
//   [the cat <!> <!>]                    ++(cat, the) ++(cat, <!>) ++(cat, <!>)
//   [<!>]
class Window {
public:
    // Initializes a window.
    Window(size_t window_size, const string &context_definition,
	   const unordered_map<string, Word> &word_dictionary,
	   const string &rare_symbol, const string &buffer_symbol,
	   size_t hash_size) :
	window_size_(window_size),
	context_definition_(context_definition),
	word_dictionary_(word_dictionary),
	rare_symbol_(rare_symbol),
	buffer_symbol_(buffer_symbol),
	hash_size_(hash_size) { PrepareWindow(); }

    // Adds a word string to the window and processes it if full.
    void Add(const string &word_string);

    // Processes whatever is in the window and resets.
    void Finish();

    // Returns a pointer to the context dictionary.
    unordered_map<string, Context> *context_dictionary() {
	return &context_dictionary_;
    }

    // Returns a pointer to the context-word co-occurrence counts.
    unordered_map<Context, unordered_map<Word, double> > *context_word_count() {
	return &context_word_count_;
    }

private:
    // Prepares a window with given parameters.
    void PrepareWindow();

    // Increments co-occurrence counts from a full window of text.
    void ProcessFull();

    // Adds the given string to the context dictionary (if not already in it).
    // Returns a unique index corresponding to this string.
    Context AddContextString(const string &context_string);

    // FIFO queue of strings.
    deque<string> queue_;

    // Center index of the window.
    size_t center_index_;

    // Strings for marking position-sensitive contexts.
    vector<string> position_markers_;

    // Hash function for hashing context strings.
    hash<string> context_hash_;

    // Window size.
    size_t window_size_;

    // Context definition.
    string context_definition_;

    // Address of the word dictionary.
    const unordered_map<string, Word> &word_dictionary_;

    // Rare symbol.
    string rare_symbol_;

    // Buffer symbol.
    string buffer_symbol_;

    // Number of hash bins (0 means no hashing).
    size_t hash_size_;

    // Context dictionary.
    unordered_map<string, Context> context_dictionary_;

    // Context-word co-occurrence counts.
    unordered_map<Context, unordered_map<Word, double> > context_word_count_;
};

#endif  // CORE_CORPUS_H_
