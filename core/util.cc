// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "util.h"

namespace util {
    void split_by_chars(const string &line, const string &char_delimiters,
			vector<string> *tokens) {
	tokens->clear();
	size_t start = 0;  // Keep track of the current position.
	size_t end = 0;
	string token;
	while (end != string::npos) {
	    // Find the first index a delimiter char occurs.
	    end = line.find_first_of(char_delimiters, start);

	    // Collect a corresponding portion of the line into a token.
	    token = (end == string::npos) ? line.substr(start, string::npos) :
		line.substr(start, end - start);
	    if(token != "") { tokens->push_back(token); }

	    // Update the current position.
	    start = (end > string::npos - 1) ?  string::npos : end + 1;
	}
    }

    void split_by_space_tab(const string &line, vector<string> *tokens) {
	split_by_chars(line, " \t\n", tokens);
    }

    void read_line(ifstream *file,  vector<string> *tokens) {
	string line;
	getline(*file, line);
	split_by_space_tab(line, tokens);
    }

    void split_by_string(const string &line, const string &string_delimiter,
			 vector<string> *tokens) {
	tokens->clear();
	size_t start = 0;  // Keep track of the current position.
	size_t end = 0;
	string token;
	while (end != string::npos) {
	    // Find where the string delimiter occurs next.
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

    string convert_seconds_to_string(double num_seconds) {
	size_t num_hours = (int) floor(num_seconds / 3600.0);
	double num_seconds_minus_h = num_seconds - (num_hours * 3600);
	int num_minutes = (int) floor(num_seconds_minus_h / 60.0);
	int num_seconds_minus_hm = num_seconds_minus_h - (num_minutes * 60);
	string time_string = to_string(num_hours) + "h" + to_string(num_minutes)
	    + "m" + to_string(num_seconds_minus_hm) + "s";
	return time_string;
    }

    string lowercase(const string &original_string) {
	string lowercased_string;
	for (const char &character : original_string) {
	    lowercased_string.push_back(tolower(character));
	}
	return lowercased_string;
    }

    string convert_to_string(const vector<string> &sequence) {
	string sequence_string;
	for (size_t i = 0; i < sequence.size(); ++i) {
	    sequence_string += sequence[i];
	    if (i < sequence.size() - 1) { sequence_string += " "; }
	}
	return sequence_string;
    }

    string convert_to_string(const vector<double> &sequence) {
	string sequence_string;
	for (size_t i = 0; i < sequence.size(); ++i) {
	    sequence_string += to_string_with_precision(sequence[i], 2);
	    if (i < sequence.size() - 1) { sequence_string += " "; }
	}
	return sequence_string;
    }
}  // namespace util
