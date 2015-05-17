// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various utility functions for the C++ standard library.

#ifndef UTIL_H_
#define UTIL_H_

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace util_string {
    // Splits a line by char delimiters.
    void split_by_chars(const string &line, const string &char_delimiters,
			vector<string> *tokens);

    // Splits a line by whitespace (space, tab, or newline).
    void split_by_whitespace(const string &line, vector<string> *tokens);

    // Reads a line from a file into tokens separated by whitespace.
    void read_line(ifstream *file,  vector<string> *tokens);

    // Splits a line by a string delimiter.
    void split_by_string(const string &line, const string &string_delimiter,
			 vector<string> *tokens);
}  // namespace util_strings

#endif  // UTIL_H_
