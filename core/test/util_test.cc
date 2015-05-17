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
    util_string::split_by_string(example_, "some\n tabs", &tokens_by_phrase);
    EXPECT_EQ(2, tokens_by_phrase.size());
    EXPECT_EQ("I have\t", tokens_by_phrase[0]);
    EXPECT_EQ("\tand spaces", tokens_by_phrase[1]);

    vector<string> tokens_by_space;
    util_string::split_by_string(example_, " ", &tokens_by_space);
    EXPECT_EQ(4, tokens_by_space.size());
    EXPECT_EQ("I", tokens_by_space[0]);
    EXPECT_EQ("have	some\n", tokens_by_space[1]);
    EXPECT_EQ("tabs	and", tokens_by_space[2]);
    EXPECT_EQ("spaces", tokens_by_space[3]);
}

// Checks spliting by char delimiters.
TEST_F(StringTokenization, SplitByChars) {
    vector<string> tokens_by_whitespace;
    util_string::split_by_chars(example_, " \t\n", &tokens_by_whitespace);
    EXPECT_EQ(6, tokens_by_whitespace.size());
    EXPECT_EQ("I", tokens_by_whitespace[0]);
    EXPECT_EQ("have", tokens_by_whitespace[1]);
    EXPECT_EQ("some", tokens_by_whitespace[2]);
    EXPECT_EQ("tabs", tokens_by_whitespace[3]);
    EXPECT_EQ("and", tokens_by_whitespace[4]);
    EXPECT_EQ("spaces", tokens_by_whitespace[5]);
}

// Checks reading lines from a text file.
TEST(ReadFromFile, LineTokenization) {
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
    util_string::read_line(&text_file_in, &tokens);
    EXPECT_EQ(3, tokens.size());
    EXPECT_EQ("a", tokens[0]);
    EXPECT_EQ("b", tokens[1]);
    EXPECT_EQ("c", tokens[2]);

    //  ""
    util_string::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    //  "\t\td e f"
    util_string::read_line(&text_file_in, &tokens);
    EXPECT_EQ(3, tokens.size());
    EXPECT_EQ("d", tokens[0]);
    EXPECT_EQ("e", tokens[1]);
    EXPECT_EQ("f", tokens[2]);

    //  ""
    util_string::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    //  ""
    util_string::read_line(&text_file_in, &tokens);
    EXPECT_EQ(0, tokens.size());

    remove(text_file_path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
