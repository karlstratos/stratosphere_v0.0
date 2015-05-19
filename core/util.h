// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various utility functions for the C++ standard library.

#ifndef CORE_UTIL_H_
#define CORE_UTIL_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

// Assert macro that allows adding a message to an assertion upon failure. It
// implictly performs string conversion: ASSERT(x > 0, "Negative x: " << x);
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

namespace util_string {
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

    // Converts a value to string up to a certain number of decimal places.
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
}  // namespace util_string

namespace util_file {
    // Returns true if the file exists, false otherwise.
    bool exists(const string &file_path);

    // Returns the type of the given file path: "file", "dir", or "other".
    string get_file_type(const string &file_path);

    // Lists files. If given a single file, the list contains the path to that
    // file. If given a directory, the list contains the paths to the files
    // inside that directory (non-recursively).
    void list_files(const string &file_path, vector<string> *list);

    // Returns the number of lines in a file.
    size_t get_num_lines(const string &file_path);

    // Writes a primitive value to a binary file.
    template<typename T>
    void binary_write_primitive(const T &value, ostream& file){
	file.write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    // Reads a primitive value from a binary file.
    template<typename T>
    void binary_read_primitive(istream& file, T *value){
	file.read(reinterpret_cast<char*>(value), sizeof(T));
    }

    // Writes a primitive unordered_map.
    template <typename T1, typename T2>
    void binary_write_primitive(const unordered_map<T1, T2> &table,
				const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	binary_write_primitive(table.size(), file);
	for (const auto &pair : table) {
	    binary_write_primitive(pair.first, file);
	    binary_write_primitive(pair.second, file);
	}
    }

    // Reads a primitive unordered_map.
    template <typename T1, typename T2>
    void binary_read_primitive(const string &file_path,
			       unordered_map<T1, T2> *table) {
	table->clear();
	ifstream file(file_path, ios::in | ios::binary);
	size_t num_keys;
	binary_read_primitive(file, &num_keys);
	for (size_t i = 0; i < num_keys; ++i) {
	    T1 key;
	    T2 value;
	    binary_read_primitive(file, &key);
	    binary_read_primitive(file, &value);
	    (*table)[key] = value;
	}
    }

    // Writes a primitive 2-nested unordered_map.
    template <typename T1, typename T2, typename T3>
    void binary_write_primitive(
	const unordered_map<T1, unordered_map<T2, T3> > &table,
	const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	binary_write_primitive(table.size(), file);
	for (const auto &pair1 : table) {
	    binary_write_primitive(pair1.first, file);
	    binary_write_primitive(pair1.second.size(), file);
	    for (const auto &pair2 : pair1.second) {
		binary_write_primitive(pair2.first, file);
		binary_write_primitive(pair2.second, file);
	    }
	}
    }

    // Reads a primitive 2-nested unordered_map.
    template <typename T1, typename T2, typename T3>
    void binary_read_primitive(
	const string &file_path,
	unordered_map<T1, unordered_map<T2, T3> > *table) {
	table->clear();
	ifstream file(file_path, ios::in | ios::binary);
	size_t num_first_keys;
	binary_read_primitive(file, &num_first_keys);
	for (size_t i = 0; i < num_first_keys; ++i) {
	    T1 first_key;
	    size_t num_second_keys;
	    binary_read_primitive(file, &first_key);
	    binary_read_primitive(file, &num_second_keys);
	    for (size_t j = 0; j < num_second_keys; ++j) {
		T2 second_key;
		T3 value;
		binary_read_primitive(file, &second_key);
		binary_read_primitive(file, &value);
		(*table)[first_key][second_key] = value;
	    }
	}
    }

    // Writes a string to a binary file.
    void binary_write_string(const string &value, ofstream& file);

    // Reads a string from a binary file.
    void binary_read_string(ofstream& file, string *value);

    // Writes a (string, size_t) unordered_map to a binary file.
    void binary_write(const unordered_map<string, size_t> &table,
		      const string &file_path);

    // Reads a (string, size_t) unordered_map from a binary file.
    void binary_read(const string &file_path,
		     unordered_map<string, size_t> *table);
}  // namespace util_file

namespace util_misc {
    // Template for a struct used to sort a vector of pairs by the second
    // values. Use it like this:
    //    sort(v.begin(), v.end(), util_misc::sort_pairs_second<int, int>());
    //    sort(v.begin(), v.end(),
    //         util_misc::sort_pairs_second<int, int, greater<int> >());
    template <class T1, class T2, class Predicate = less<T2> >
    struct sort_pairs_second {
	bool operator()(const pair<T1, T2> &left, const pair<T1, T2> &right) {
	    return Predicate()(left.second, right.second);
	}
    };

    // Inverts an unordered_map.
    template <typename T1, typename T2>
    void invert(const unordered_map<T1, T2> &table1,
		unordered_map<T2, T1> *table2) {
	table2->clear();
	for (const auto &pair : table1) { (*table2)[pair.second] = pair.first; }
    }
}  // namespace util_misc

#endif  // CORE_UTIL_H_
