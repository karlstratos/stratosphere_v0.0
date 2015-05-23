// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the corpus code.

#include "gtest/gtest.h"

#include <stdio.h>
#include <stdlib.h>

#include "../corpus.h"

// Test class that provides a simple corpus file.
class CorpusExample : public testing::Test {
protected:
    virtual void SetUp() {
	corpus_file_path_ = tmpnam(nullptr);
	ofstream corpus_file(corpus_file_path_, ios::out);
	corpus_file << "DOG" << endl;
	corpus_file << "the dog saw the cat ." << endl;
	corpus_file << "the dog barked ." << endl;
	corpus_file << "the cat laughed !" << endl;
    }

    virtual void TearDown() { remove(corpus_file_path_.c_str()); }

    string corpus_file_path_;
};

// Checks word counting.
TEST_F(CorpusExample, CheckCountWords) {
    Corpus corpus(corpus_file_path_, false);
    unordered_map<string, size_t> word_count;
    size_t num_words = corpus.CountWords(&word_count);

    EXPECT_EQ(15, num_words);
    EXPECT_EQ(9, word_count.size());
    EXPECT_EQ(4, word_count["the"]);
    EXPECT_EQ(2, word_count["dog"]);
    EXPECT_EQ(2, word_count["cat"]);
    EXPECT_EQ(2, word_count["."]);
    EXPECT_EQ(1, word_count["saw"]);
    EXPECT_EQ(1, word_count["barked"]);
    EXPECT_EQ(1, word_count["laughed"]);
    EXPECT_EQ(1, word_count["!"]);
    EXPECT_EQ(1, word_count["DOG"]);
}

// Checks word counting with limited vocabulary size.
TEST_F(CorpusExample, CheckCountWordsWithLimitedVocabularySize) {
    Corpus corpus(corpus_file_path_, false);
    corpus.set_max_vocabulary_size(9);
    unordered_map<string, size_t> word_count;
    size_t num_words = corpus.CountWords(&word_count);

    // The median count was 1, so all counts are subtracted by 1.
    EXPECT_EQ(6, num_words);
    EXPECT_EQ(4, word_count.size());
    EXPECT_EQ(3, word_count["the"]);
    EXPECT_EQ(1, word_count["dog"]);
    EXPECT_EQ(1, word_count["cat"]);
    EXPECT_EQ(1, word_count["."]);
}

// Checks lowercasing.
TEST_F(CorpusExample, CheckLowercase) {
    Corpus corpus(corpus_file_path_, false);
    corpus.set_lowercase(true);
    unordered_map<string, size_t> word_count;
    corpus.CountWords(&word_count);
    EXPECT_EQ(3, word_count["dog"]);  // DOG -> dog
}

// Checks word dictionary building.
TEST_F(CorpusExample, CheckBuildWordDictionary) {
    Corpus corpus(corpus_file_path_, false);
    unordered_map<string, size_t> word_count;
    corpus.CountWords(&word_count);

    unordered_map<string, size_t> word_dictionary_cutoff0;
    size_t num_considered_words_cutoff0 =
	corpus.BuildWordDictionary(word_count, 0, &word_dictionary_cutoff0);
    EXPECT_EQ(15, num_considered_words_cutoff0);
    EXPECT_EQ(9, word_dictionary_cutoff0.size());
    EXPECT_FALSE(word_dictionary_cutoff0.find(corpus.kRareString()) !=
		 word_dictionary_cutoff0.end());  // No rare symbol.

    unordered_map<string, size_t> word_dictionary_cutoff1;
    size_t num_considered_words_cutoff1 =
	corpus.BuildWordDictionary(word_count, 1, &word_dictionary_cutoff1);
    EXPECT_EQ(10, num_considered_words_cutoff1);
    EXPECT_EQ(5, word_dictionary_cutoff1.size());
    EXPECT_EQ(word_dictionary_cutoff1.size() - 1,
	      word_dictionary_cutoff1[corpus.kRareString()]);  // Highest index.
}

// Checks window sliding with sentence-per-line, bag contexts, and size 2.
TEST_F(CorpusExample, CheckSlideWindowSentencesBagSize2) {
    bool sentence_per_line = true;
    string context_definition = "bag";
    size_t window_size = 2;

    Corpus corpus(corpus_file_path_, false);
    unordered_map<string, size_t> word_count;
    corpus.CountWords(&word_count);
    unordered_map<string, size_t> word_dictionary;
    corpus.BuildWordDictionary(word_count, 1, &word_dictionary);  // Cutoff 1

    unordered_map<string, Context> context_dictionary;
    unordered_map<Context, unordered_map<Word, double> > context_word_count;
    corpus.SlideWindow(word_dictionary, sentence_per_line, context_definition,
		       window_size, 0, &context_dictionary,
		       &context_word_count);

    size_t num_samples = util_misc::sum_values(context_word_count);
    EXPECT_EQ(15, num_samples);  // num_samples = (window_size - 1) * num_words

    Word w_the = word_dictionary["the"];
    Word w_dog = word_dictionary["dog"];
    Word w_cat = word_dictionary["cat"];
    Word w_period = word_dictionary["."];
    Word w_rare = word_dictionary[corpus.kRareString()];
    Context c_dog = context_dictionary["dog"];
    Context c_the = context_dictionary["the"];
    Context c_cat = context_dictionary["cat"];
    Context c_period = context_dictionary["."];
    Context c_rare = context_dictionary[corpus.kRareString()];
    Context c_buffer = context_dictionary[corpus.kBufferString()];

    // <?>
    // the dog <?> the cat .
    // the dog <?> .
    // the cat <?> <?>
    EXPECT_EQ(2, context_word_count[c_dog][w_the]);  // the dog
    EXPECT_EQ(2, context_word_count[c_rare][w_dog]);  // dog <?>
    EXPECT_EQ(1, context_word_count[c_the][w_rare]);  // saw the
    EXPECT_EQ(1, context_word_count[c_period][w_cat]);  // cat .
    EXPECT_EQ(2, context_word_count[c_buffer][w_period]);  // . <!>
    EXPECT_EQ(2, context_word_count[c_cat][w_the]);  // the cat
    EXPECT_EQ(1, context_word_count[c_period][w_rare]);  // <?> .
    EXPECT_EQ(1, context_word_count[c_rare][w_cat]);  // cat <?>
    EXPECT_EQ(1, context_word_count[c_rare][w_rare]);  // <?> <?>
    EXPECT_EQ(2, context_word_count[c_buffer][w_rare]);  // <?> <!>
}

// Checks window sliding with list contexts and size 3.
TEST_F(CorpusExample, CheckSlideWindowListSize3) {
    bool sentence_per_line = false;
    string context_definition = "list";
    size_t window_size = 3;

    Corpus corpus(corpus_file_path_, false);
    unordered_map<string, size_t> word_count;
    corpus.CountWords(&word_count);
    unordered_map<string, size_t> word_dictionary;
    corpus.BuildWordDictionary(word_count, 2, &word_dictionary);  // Cutoff 2

    unordered_map<string, Context> context_dictionary;
    unordered_map<Context, unordered_map<Word, double> > context_word_count;
    corpus.SlideWindow(word_dictionary, sentence_per_line, context_definition,
		       window_size, 0, &context_dictionary,
		       &context_word_count);

    size_t num_samples = util_misc::sum_values(context_word_count);
    EXPECT_EQ(30, num_samples);  // num_samples = (window_size - 1) * num_words

    Word w_the = word_dictionary["the"];
    Word w_rare = word_dictionary[corpus.kRareString()];
    Context c_the_next = context_dictionary["c(1)=the"];
    Context c_the_prev = context_dictionary["c(-1)=the"];
    Context c_rare_next = context_dictionary["c(1)=" + corpus.kRareString()];
    Context c_rare_prev = context_dictionary["c(-1)=" + corpus.kRareString()];
    Context c_buffer_next = context_dictionary["c(1)=" +
					       corpus.kBufferString()];
    Context c_buffer_prev = context_dictionary["c(-1)=" +
					       corpus.kBufferString()];

    // <?> the <?> <?> the <?> <?> the <?> <?> <?> the <?> <?> <?>
    EXPECT_EQ(4, context_word_count[c_rare_next][w_the]);  // the c(1)=<?>
    EXPECT_EQ(4, context_word_count[c_rare_prev][w_the]);  // the c(-1)=<?>
    EXPECT_EQ(6, context_word_count[c_rare_next][w_rare]);  // <?> c(1)=<?>
    EXPECT_EQ(6, context_word_count[c_rare_prev][w_rare]);  // <?> c(-1)=<?>
    EXPECT_EQ(4, context_word_count[c_the_next][w_rare]);  // <?> c(1)=the
    EXPECT_EQ(4, context_word_count[c_the_prev][w_rare]);  // <?> c(-1)=the
    EXPECT_EQ(4, context_word_count[c_the_next][w_rare]);  // <?> c(1)=the
    EXPECT_EQ(4, context_word_count[c_the_prev][w_rare]);  // <?> c(-1)=the
    EXPECT_EQ(1, context_word_count[c_buffer_next][w_rare]);  // <?> c(1)=<!>
    EXPECT_EQ(1, context_word_count[c_buffer_prev][w_rare]);  // <?> c(-1)=<!>
}

// Checks transition counting.
TEST_F(CorpusExample, CheckTransitionCounting) {
    Corpus corpus(corpus_file_path_, false);
    unordered_map<string, size_t> word_count;
    corpus.CountWords(&word_count);
    unordered_map<string, size_t> word_dictionary;
    corpus.BuildWordDictionary(word_count, 1, &word_dictionary);  // Cutoff 1

    unordered_map<Word, unordered_map<Word, size_t> > bigram_count;
    unordered_map<Word, size_t> start_count;
    unordered_map<Word, size_t> end_count;
    corpus.CountTransitions(word_dictionary, &bigram_count, &start_count,
			    &end_count);

    Word w_the = word_dictionary["the"];
    Word w_dog = word_dictionary["dog"];
    Word w_cat = word_dictionary["cat"];
    Word w_period = word_dictionary["."];
    Word w_rare = word_dictionary[corpus.kRareString()];

    // <?>
    // the dog <?> the cat .
    // the dog <?> .
    // the cat <?> <?>
    EXPECT_EQ(2, bigram_count[w_the][w_dog]);
    EXPECT_EQ(2, bigram_count[w_dog][w_rare]);
    EXPECT_EQ(1, bigram_count[w_rare][w_the]);
    EXPECT_EQ(2, bigram_count[w_the][w_cat]);
    EXPECT_EQ(1, bigram_count[w_cat][w_period]);
    EXPECT_EQ(1, bigram_count[w_rare][w_period]);
    EXPECT_EQ(1, bigram_count[w_rare][w_rare]);
    EXPECT_EQ(1, start_count[w_rare]);
    EXPECT_EQ(3, start_count[w_the]);
    EXPECT_EQ(2, end_count[w_rare]);
    EXPECT_EQ(2, end_count[w_period]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
