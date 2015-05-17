// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "util.h"

#include <dirent.h>
#include <math.h>
#include <sys/stat.h>

namespace util_string {
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
}  // namespace util_string

namespace util_file {
    bool exists(const string &file_path) {
	struct stat buffer;
	return (stat(file_path.c_str(), &buffer) == 0);
    }

    string get_file_type(const string &file_path) {
	string file_type;
	struct stat stat_buffer;
	if (stat(file_path.c_str(), &stat_buffer) == 0) {
	    if (stat_buffer.st_mode & S_IFREG) {
		file_type = "file";
	    } else if (stat_buffer.st_mode & S_IFDIR) {
		file_type = "dir";
	    } else {
		file_type = "other";
	    }
	} else {
	    ASSERT(false, "Problem with " << file_path);
	}
	return file_type;
    }

    void list_files(const string &file_path, vector<string> *list) {
	(*list).clear();
	string file_type = get_file_type(file_path);
	if (file_type == "dir") {
	    DIR *pDIR = opendir(file_path.c_str());
	    if (pDIR != NULL) {
		struct dirent *entry = readdir(pDIR);
		while (entry != NULL) {
		    if (strcmp(entry->d_name, ".") != 0 &&
			strcmp(entry->d_name, "..") != 0) {
			(*list).push_back(file_path + "/" + entry->d_name);
		    }
		    entry = readdir(pDIR);
		}
	    }
	    closedir(pDIR);
	} else {
	    (*list).push_back(file_path);
	}
    }

    size_t get_num_lines(const string &file_path) {
	size_t num_lines = 0;
	string file_type = get_file_type(file_path);
	ifstream file(file_path, ios::in);
	if (file_type == "file") {
	    string line;
	    while (getline(file, line)) { ++num_lines; }
	}
	return num_lines;
    }
}  // namespace util_file
