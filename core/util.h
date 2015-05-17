// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various utility functions for the C++ standard library.

#ifndef CORE_UTIL_H_
#define CORE_UTIL_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace util {
    // Splits a line by char delimiters.
    void split_by_chars(const string &line, const string &char_delimiters,
			vector<string> *tokens);

    // Splits a line by space or tab.
    void split_by_space_tab(const string &line, vector<string> *tokens);

    // Reads the next line from a file into tokens separated by space or tab.
    void read_line(ifstream *file,  vector<string> *tokens);

    // Splits a line by a string delimiter.
    void split_by_string(const string &line, const string &string_delimiter,
			 vector<string> *tokens);

    // Converts seconds to an hour/minute/second string: 6666 => "1h51m6s".
    string convert_seconds_to_string(double num_seconds);

    // Lowercases a string.
    string lowercase(const string &original_string);

    template <typename T>
    string to_string_with_precision(const T value,
				    const size_t num_decimal_places = 6) {
	ostringstream out;
	out << setprecision(num_decimal_places) << value;
	return out.str();
    }

    // Converts a string vector to string.
    string convert_to_string(const vector<string> &sequence);

    // Converts a double vector to string.
    string convert_to_string(const vector<double> &sequence);
}  // namespace util

#endif  // CORE_UTIL_H_
