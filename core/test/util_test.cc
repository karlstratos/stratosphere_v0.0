// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the utility code.

#include "gtest/gtest.h"

//#include <stdio.h>
//#include <stdlib.h>

#include "../util.h"

// Test class for string tokenization.
class StringTokenization : public testing::Test {
protected:
    virtual void SetUp() {
	example_ = "I have	some\n tabs	and spaces";
    }
    string example_;
};

// Checks spliting by a string delimiter.
TEST_F(StringTokenization, SplitByString) {
    vector<string> tokens_by_phrase;
    util::split_by_string(example_, "some\n tabs", &tokens_by_phrase);
    EXPECT_EQ(2, tokens_by_phrase.size());
    EXPECT_EQ("I have\t", tokens_by_phrase[0]);
    EXPECT_EQ("\tand spaces", tokens_by_phrase[1]);

    vector<string> tokens_by_space;
    util::split_by_string(example_, " ", &tokens_by_space);
    EXPECT_EQ(4, tokens_by_space.size());
    EXPECT_EQ("I", tokens_by_space[0]);
    EXPECT_EQ("have	some\n", tokens_by_space[1]);
    EXPECT_EQ("tabs	and", tokens_by_space[2]);
    EXPECT_EQ("spaces", tokens_by_space[3]);
}

// Checks spliting by char delimiters.
TEST_F(StringTokenization, SplitByChars) {
    vector<string> tokens_by_whitespace;
    util::split_by_chars(example_, " \t\n", &tokens_by_whitespace);
    EXPECT_EQ(6, tokens_by_whitespace.size());
    EXPECT_EQ("I", tokens_by_whitespace[0]);
    EXPECT_EQ("have", tokens_by_whitespace[1]);
    EXPECT_EQ("some", tokens_by_whitespace[2]);
    EXPECT_EQ("tabs", tokens_by_whitespace[3]);
    EXPECT_EQ("and", tokens_by_whitespace[4]);
    EXPECT_EQ("spaces", tokens_by_whitespace[5]);
}

// Checks reading lines from a text file.
TEST(StringUtil, FileNextLineTokenization) {
    string text_file_path = tmpnam(nullptr);
    ofstream text_file_out(text_file_path, ios::out);
    text_file_out << "a b	c" << endl;
    text_file_out << endl;
    text_file_out << "		d e f" << endl;
    text_file_out << endl;
    text_file_out.close();

    ifstream text_file_in(text_file_path, ios::in);
    vector<string> tokens;

    //  "a b\tc"
    util::read_line(&text_file_in, &tokens);
    EXPECT_EQ(3, tokens.size());
    EXPECT_EQ("a", tokens[0]);
    EXPECT_EQ("b", tokens[1]);
    EXPECT_EQ("c", tokens[2]);

    //  ""
    util::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    //  "\t\td e f"
    util::read_line(&text_file_in, &tokens);
    EXPECT_EQ(3, tokens.size());
    EXPECT_EQ("d", tokens[0]);
    EXPECT_EQ("e", tokens[1]);
    EXPECT_EQ("f", tokens[2]);

    //  ""
    util::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    //  ""
    util::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    remove(text_file_path.c_str());
}

// Checks converting seconds to string.
TEST(StringUtil, ConvertSecondsToString) {
    EXPECT_EQ("20h7m18s", util::convert_seconds_to_string(72438.1));
    EXPECT_EQ("20h7m18s", util::convert_seconds_to_string(72438.9));
}

// Checks lowercasing a string.
TEST(StringUtil, Lowercase) {
    EXPECT_EQ("ab12345cd@#%! ?ef", util::lowercase("AB12345Cd@#%! ?eF"));
}

// Checks converting a vector to string.
TEST(StringUtil, ConvertVectorToString) {
    EXPECT_EQ("a b c", util::convert_to_string({"a", "b", "c"}));
    EXPECT_EQ("1 2 0.54", util::convert_to_string({1.0, 2.0, 0.538}));
    EXPECT_EQ("0.53", util::convert_to_string({0.532}));
    EXPECT_EQ("1 2 3", util::convert_to_string({1, 2, 3}));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
