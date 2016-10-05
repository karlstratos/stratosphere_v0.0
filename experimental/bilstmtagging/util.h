// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various utility functions for the C++ standard library.

#ifndef CORE_UTIL_H_
#define CORE_UTIL_H_

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Assert macro that allows adding a message to an assertion upon failure. It
// implictly performs string conversion: ASSERT(x > 0, "Negative x: " << x);
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

namespace util_eval {
    // Computes accuracy for sequence predictions.
    void compute_accuracy(const std::vector<std::vector<std::string> >
                          &true_sequences,
			  const std::vector<std::vector<std::string> >
                          &predicted_sequences,
			  double *position_accuracy,
			  double *sequence_accuracy);

    // Computes precision/recall for entity predictions in BIO format.
    void compute_precision_recall_bio(
        const std::vector<std::vector<std::string> > &true_sequences,
        const std::vector<std::vector<std::string> > &predicted_sequences,
        std::unordered_map<std::string, double> *per_entity_precision,
        std::unordered_map<std::string, double> *per_entity_recall,
        std::unordered_map<std::string, size_t> *num_guessed,
        double *precision, double *recall);
}  // namespace util_eval

namespace util_string {
    // Buffers a string to have a certain length.
    std::string buffer_string(const std::string &given_string, size_t length,
			 char buffer_char, const std::string &align);

    // Returns the string form of a printf format string.
    std::string printf_format(const char *format, ...);

    // Splits a line by char delimiters.
    void split_by_chars(const std::string &line,
                        const std::string &char_delimiters,
			std::vector<std::string> *tokens);

    // Splits a line by space or tab.
    void split_by_space_tab(const std::string &line, std::vector<std::string>
                            *tokens);

    // Splits a line by a string delimiter.
    void split_by_string(const std::string &line,
                         const std::string &string_delimiter,
			 std::vector<std::string> *tokens);

    // Converts seconds to an hour/minute/second string: 6666 => "1h51m6s".
    std::string convert_seconds_to_string(double num_seconds);

    // Returns an hour/minute/second string of the difftime output.
    std::string difftime_string(time_t time_now, time_t time_before);

    // Lowercases a string.
    std::string lowercase(const std::string &original_string);

    // Converts a value to string up to a certain number of decimal places.
    template <typename T>
    std::string to_string_with_precision(const T value,
				    const size_t num_decimal_places = 6) {
	std::ostringstream out;
	out << std::setprecision(num_decimal_places) << value;
	return out.str();
    }

    // Returns an alphanumeric string of a double, e.g., 1.3503 -> "1p35".
    std::string convert_to_alphanumeric_string(double value,
                                               size_t decimal_place);

    // Converts a string vector to string.
    std::string convert_to_string(const std::vector<std::string> &sequence);

    // Converts a double vector to string.
    std::string convert_to_string(const std::vector<double> &sequence);
}  // namespace util_string

namespace util_file {
    // Gets the file name from a file path.
    std::string get_file_name(std::string file_path);

    // Reads the next line from a file into tokens separated by space or tab.
    // while (file.good()) {
    //     vector<string> tokens;
    //     util_file::read_line(&file, &tokens);
    //     /* (Do stuff with tokens.) */
    // }
    void read_line(std::ifstream *file,  std::vector<std::string> *tokens);

    // Returns true if the file exists, false otherwise.
    bool exists(const std::string &file_path);

    // Returns the type of the given file path: "file", "dir", or "other".
    std::string get_file_type(const std::string &file_path);

    // Lists files. If given a single file, the list contains the path to that
    // file. If given a directory, the list contains the paths to the files
    // inside that directory (non-recursively).
    void list_files(const std::string &file_path,
                    std::vector<std::string> *list);

    // Returns the number of lines in a file.
    size_t get_num_lines(const std::string &file_path);

    // Writes a primitive value to a binary file.
    // *WARNING* Do not pass a value stored in a temporary variable!
    //     // binary_write_primiative(0, file);  // Bad: undefined behavior
    //     size_t zero = 0;
    //     binary_write_primiative(zero, file);  // Good
    template<typename T>
    void binary_write_primitive(const T &value, std::ostream& file){
	file.write(reinterpret_cast<const char *>(&value), sizeof(T));
    }

    // Reads a primitive value from a binary file.
    template<typename T>
    void binary_read_primitive(std::istream& file, T *value){
	file.read(reinterpret_cast<char*>(value), sizeof(T));
    }

    // Writes a primitive unordered_map.
    template <typename T1, typename T2>
    void binary_write_primitive(const std::unordered_map<T1, T2> &table,
				const std::string &file_path) {
        std::ofstream file(file_path, std::ios::out | std::ios::binary);
	    ASSERT(file.is_open(), "Cannot open file: " << file_path);
	    binary_write_primitive(table.size(), file);
	    for (const auto &pair : table) {
	        binary_write_primitive(pair.first, file);
	        binary_write_primitive(pair.second, file);
	    }
    }

    // Reads a primitive unordered_map.
    template <typename T1, typename T2>
    void binary_read_primitive(const std::string &file_path,
			       std::unordered_map<T1, T2> *table) {
		table->clear();
        std::ifstream file(file_path, std::ios::in | std::ios::binary);
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
	const std::unordered_map<T1, std::unordered_map<T2, T3> > &table,
	const std::string &file_path) {
        std::ofstream file(file_path, std::ios::out | std::ios::binary);
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
	const std::string &file_path,
	std::unordered_map<T1, std::unordered_map<T2, T3> > *table) {
	table->clear();
    std::ifstream file(file_path, std::ios::in | std::ios::binary);
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
    void binary_write_string(const std::string &value, std::ostream& file);

    // Reads a string from a binary file.
    void binary_read_string(std::istream& file, std::string *value);

    // Writes a (string, size_t) unordered_map to a binary file.
    void binary_write(const std::unordered_map<std::string, size_t> &table,
		      const std::string &file_path);

    // Reads a (std::string, size_t) unordered_map from a binary file.
    void binary_read(const std::string &file_path,
		     std::unordered_map<std::string, size_t> *table);
}  // namespace util_file

namespace util_math {
    // Computes a random permutation of {1...n}, runtime/memory O(n).
    //    n = 100 million => ~2.5G memory, ~2.5 minutes
    void permute_indices(size_t num_indices,
                         std::vector<size_t> *permuted_indices);

    // Returns -inf if a = 0, returns log(a) otherwise (error if a < 0).
    double log0(double a);

    // Given two log values log(a) and log(b), computes log(a + b) without
    // exponentiating log(a) and log(b).
    double sum_logs(double log_a, double log_b);

    // Computes Spearman's rank correlation coefficient between two sequences.
    double compute_spearman(const std::vector<double> &sequence1,
			    const std::vector<double> &sequence2);

    // Computes the average-rank transformation of a sequence.
    void transform_average_rank(const std::vector<double> &sequence,
				std::vector<double> *transformed_sequence);
}  // namespace util_math

namespace util_misc {
    // Template for a struct used to sort a vector of pairs by the second
    // values. Use it like this:
    //    sort(v.begin(), v.end(), util_misc::sort_pairs_second<int, int>());
    //    sort(v.begin(), v.end(),
    //         util_misc::sort_pairs_second<int, int, greater<int> >());
    template <typename T1, typename T2, typename Predicate = std::less<T2> >
    struct sort_pairs_second {
	bool operator()(const std::pair<T1, T2> &left,
                    const std::pair<T1, T2> &right) {
	    return Predicate()(left.second, right.second);
	}
    };

    // Inverts an unordered_map.
    template <typename T1, typename T2>
    void invert(const std::unordered_map<T1, T2> &table1,
		std::unordered_map<T2, T1> *table2) {
	table2->clear();
	for (const auto &pr : table1) { (*table2)[pr.second] = pr.first; }
    }

    // Subtracts the median value from all values, guaranteeing the elimination
    // of at least half of the elements in the hash table. When counting items
    // with only k slots in a stream of N instances, calling this every time all
    // slots are filled guarantees that |#(i) - #'(i)| <= 2N/k where #(i) is the
    // true count of item i and #'(i) is the approximate count obtained through
    // this process.
    template <typename T1, typename T2>
    void subtract_by_median(std::unordered_map<T1, T2> *table) {
	std::vector<std::pair<T1, T2> > sorted_key_values(table->begin(),
                                                          table->end());
    std::sort(sorted_key_values.begin(), sorted_key_values.end(),
	     sort_pairs_second<T1, T2, std::greater<T2> >());
	T2 median_value = sorted_key_values[(table->size() - 1) / 2].second;

	for (auto iterator = table->begin(); iterator != table->end();) {
	    if (iterator->second <= median_value) {
		iterator = table->erase(iterator);
	    } else {
		iterator->second -= median_value;
		++iterator;
	    }
	}
    }

    // Returns the sum of values in an unordered map.
    template <typename T1, typename T2>
    T2 sum_values(const std::unordered_map<T1, T2> &table) {
	T2 sum = 0.0;
	for (const auto &pair : table) { sum += pair.second; }
	return sum;
    }

    // Returns the sum of values in a 2-nested unordered map.
    template <typename T1, typename T2, typename T3>
    T3 sum_values(const std::unordered_map<T1, std::unordered_map<T2, T3> >
                  &table) {
	T3 sum = 0.0;
	for (const auto &pair1 : table) {
	    for (const auto &pair2 : pair1.second) { sum += pair2.second; }
	}
	return sum;
    }

    // Returns true if two unordered maps have the same entries and are close
    // in value, else false.
    template <typename T1, typename T2>
    T2 check_near(const std::unordered_map<T1, T2> &table1,
		  const std::unordered_map<T1, T2> &table2) {
	if (table1.size() != table2.size()) { return false; }
	for (const auto &pair1 : table1) {
	    T1 key = pair1.first;
	    if (table2.find(key) == table2.end()) { return false; }
	    if (fabs(table1.at(key) - table2.at(key)) > 1e-10) { return false; }
	}
	return true;
    }

    // Returns true if two 2-nested unordered maps have the same entries and are
    // close in value, else false.
    template <typename T1, typename T2, typename T3>
    T3 check_near(const std::unordered_map<T1, std::unordered_map<T2, T3> >
                  &table1,
		  const std::unordered_map<T1, std::unordered_map<T2, T3> >
                  &table2) {
	if (table1.size() != table2.size()) { return false; }
	for (const auto &pair1 : table1) {
	    T1 key1 = pair1.first;
	    if (table2.find(key1) == table2.end()) { return false; }
	    if (table1.at(key1).size() != table2.at(key1).size()) {
		return false;
	    }
	    for (const auto &pair2 : table1.at(key1)) {
		T2 key2 = pair2.first;
		if (table2.at(key1).find(key2) == table2.at(key1).end()) {
		    return false;
		}
		if (fabs(table1.at(key1).at(key2) -
			 table2.at(key1).at(key2)) > 1e-10) { return false; }
	    }
	}
	return true;
    }

}  // namespace util_misc

#endif  // CORE_UTIL_H_
