// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "util.h"

namespace util_string {
    void split_by_chars(const string &line, const string &char_delimiters,
			vector<string> *tokens) {
	tokens->clear();
	size_t start = 0;  // Keep track of the current position.
	size_t end = 0;
	string token;
	while (end != string::npos) {
	    // Find where the delimiter occurs next.
	    end = line.find_first_of(char_delimiters, start);

	    // Collect a corresponding portion of the line into a token.
	    token = (end == string::npos) ? line.substr(start, string::npos) :
		line.substr(start, end - start);
	    if(token != "") { tokens->push_back(token); }

	    // Update the current position.
	    start = (end > string::npos - 1) ?  string::npos : end + 1;
	}
    }

    void split_by_whitespace(const string &line, vector<string> *tokens) {
	split_by_chars(line, " \t\n", tokens);
    }

    void read_line(ifstream *file,  vector<string> *tokens) {
	string line;
	getline(*file, line);
	split_by_whitespace(line, tokens);
    }

    void split_by_string(const string &line, const string &string_delimiter,
			 vector<string> *tokens) {
	tokens->clear();
	size_t start = 0;  // Keep track of the current position.
	size_t end = 0;
	string token;
	while (end != string::npos) {
	    // Find where the delimiter occurs next.
	    end = line.find(string_delimiter, start);

	    // Collect a corresponding portion of the line into a token.
	    token = (end == string::npos) ? line.substr(start, string::npos) :
		line.substr(start, end - start);
	    if(token != "") { tokens->push_back(token); }

	    // Update the current position.
	    start = (end > string::npos - string_delimiter.size()) ?
		string::npos : end + string_delimiter.size();
	}
    }
}  // namespace util_strings
