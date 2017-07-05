// Author: Karl Stratos (me@karlstratos.com)

#include "util.h"

#include <dirent.h>
#include <libgen.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>
#include <random>

namespace {

using namespace std;

}

namespace util_eval {
    void compute_accuracy(const vector<vector<string> > &true_sequences,
			  const vector<vector<string> > &predicted_sequences,
			  double *position_accuracy,
			  double *sequence_accuracy) {
	size_t num_items = 0;
	size_t num_items_correct = 0;
	size_t num_sequences_correct = 0;
	for (size_t i = 0; i < true_sequences.size(); ++i) {
	    num_items += true_sequences[i].size();
	    bool entire_sequence_is_correct = true;
	    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
		string true_string = true_sequences[i][j];
		string predicted_string = predicted_sequences[i][j];
		if (predicted_string == true_string) {
		    num_items_correct += 1;
		} else {
		    entire_sequence_is_correct = false;
		}
	    }
	    if (entire_sequence_is_correct) { num_sequences_correct += 1; }
	}
	(*position_accuracy) = ((double) num_items_correct) / num_items * 100;
	(*sequence_accuracy) = ((double) num_sequences_correct) /
	    true_sequences.size() * 100;
    }

    void compute_precision_recall_bio(
        const vector<vector<string> > &true_sequences,
        const vector<vector<string> > &predicted_sequences,
        unordered_map<string, double> *per_entity_precision,
        unordered_map<string, double> *per_entity_recall,
        unordered_map<string, size_t> *num_guessed,
        double *precision, double *recall) {
        per_entity_precision->clear();
        per_entity_recall->clear();
        num_guessed->clear();

        unordered_map<string, size_t> tp;  // tp[e] = TP for entity e
        unordered_map<string, size_t> fp;  // fp[e] = FP for entity e
        unordered_map<string, size_t> fn;  // fn[e] = FP for entity e
        size_t tp_all = 0;  // TP for all entities (labeled)
        size_t fp_all = 0;  // FP for all entities (labeled)
        size_t fn_all = 0;  // FN for all entities (labeled)

	for (size_t i = 0; i < true_sequences.size(); ++i) {
            const vector<string> &true_sequence = true_sequences[i];
            unordered_map<
                size_t, unordered_map<
                            size_t, unordered_map<
                                        string, bool > > > true_entities;
            size_t j = 0;
            while (j < true_sequence.size()) {
                if (tolower(true_sequence.at(j)[0]) == 'b') {
                    size_t start = j;
                    size_t end = j;
                    string entity_type = true_sequence.at(j).substr(2);
                    while (j + 1 < true_sequence.size() &&
                           tolower(true_sequence.at(j + 1)[0]) == 'i' &&
                           true_sequence.at(j + 1).substr(2) == entity_type) {
                        ++end;
                        ++j;
                    }
                    true_entities[start][end][entity_type] = true;
                }
                ++j;
            }

            const vector<string> &predicted_sequence = predicted_sequences[i];
            unordered_map<
                size_t, unordered_map<
                            size_t, unordered_map<
                                        string, bool > > > predicted_entities;
            j = 0;
            while (j < predicted_sequence.size()) {
                // Start of an entity can happen with 'I' in addition to 'B'.
                if (tolower(predicted_sequence.at(j)[0]) != 'o') {
                    size_t start = j;
                    size_t end = j;
                    string entity_type = predicted_sequence.at(j).substr(2);
                    while (j + 1 < predicted_sequence.size() &&
                           tolower(predicted_sequence.at(j + 1)[0]) == 'i' &&
                           predicted_sequence.at(j + 1).substr(2) ==
                           entity_type) {
                        ++end;
                        ++j;
                    }
                    predicted_entities[start][end][entity_type] = true;
                }
                ++j;
            }

            for (auto &p1 : true_entities) {
                size_t start = p1.first;
                for (auto &p2 : p1.second) {
                    size_t end = p2.first;
                    for (auto &p3 : p2.second) {
                        string entity_type = p3.first;
                        if (predicted_entities.find(start) !=
                            predicted_entities.end() &&
                            predicted_entities[start].find(end) !=
                            predicted_entities[start].end() &&
                            predicted_entities[start][end].find(entity_type) !=
                            predicted_entities[start][end].end()) {
                            if (tp.find(entity_type) == tp.end()) {
                                tp[entity_type] = 0;
                            }
                            ++tp[entity_type];
                            ++tp_all;
                        } else {
                            if (fn.find(entity_type) == fn.end()) {
                                fn[entity_type] = 0;
                            }
                            ++fn[entity_type];
                            ++fn_all;
                        }
                    }
                }
            }

           for (auto &p1 : predicted_entities) {
                size_t start = p1.first;
                for (auto &p2 : p1.second) {
                    size_t end = p2.first;
                    for (auto &p3 : p2.second) {
                        string entity_type = p3.first;
                        if (true_entities.find(start) ==
                            true_entities.end() ||
                            true_entities[start].find(end) ==
                            true_entities[start].end() ||
                            true_entities[start][end].find(entity_type) ==
                            true_entities[start][end].end()) {
                            if (fp.find(entity_type) == fp.end()) {
                                fp[entity_type] = 0;
                            }
                            ++fp[entity_type];
                            ++fp_all;
                        }
                    }
                }
            }

           for (auto &p1 : tp) {
               string entity_type = p1.first;
               size_t num_predictions = tp[entity_type] + fp[entity_type];
               size_t num_occurrences = tp[entity_type] + fn[entity_type];
               (*per_entity_precision)[entity_type] =
                   ((double) tp[entity_type]) / num_predictions * 100.0;
               (*per_entity_recall)[entity_type] =
                   ((double) tp[entity_type]) / num_occurrences * 100.0;
               (*num_guessed)[entity_type] = num_predictions;
           }

           size_t num_predictions_all = tp_all + fp_all;
           size_t num_occurrences_all = tp_all + fn_all;
           if (tp_all > 0) {
               *precision = ((double) tp_all) / num_predictions_all * 100.0;
               *recall = ((double) tp_all) / num_occurrences_all * 100.0;
           } else {
               *precision = 0.0;
               *recall = 0.0;
           }
        }
    }
}  // namespace util_eval

namespace util_string {
    string buffer_string(const string &given_string, size_t length,
			 char buffer_char, const string &align) {
	string buffered_string =
	    given_string.substr(0, min(given_string.size(), length));
	bool left_turn = true;  // For align = "center".
	string buffer(1, buffer_char);
	while (buffered_string.size() < length) {
	    if (align == "left" || (align == "center" && left_turn)) {
		buffered_string = buffered_string + buffer;
		left_turn = false;
	    } else if (align == "right" || (align == "center" && !left_turn)) {
		buffered_string = buffer + buffered_string;
		left_turn = true;
	    } else {
		ASSERT(false, "Unknown alignment method: " << align);
	    }
	}
	return buffered_string;
    }

    string printf_format(const char *format, ...) {
	char buffer[16384];
	va_list variable_argument_list;
	va_start(variable_argument_list, format);
	vsnprintf(buffer, sizeof(buffer), format, variable_argument_list);
	va_end(variable_argument_list);
	return buffer;
    }

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

    string difftime_string(time_t time_now, time_t time_before) {
	double num_seconds = difftime(time_now, time_before);
	return convert_seconds_to_string(num_seconds);
    }

    string lowercase(const string &original_string) {
	string lowercased_string;
	for (const char &character : original_string) {
	    lowercased_string.push_back(tolower(character));
	}
	return lowercased_string;
    }

    string convert_to_alphanumeric_string(double value, size_t decimal_place) {
	string value_string = to_string_with_precision(value, decimal_place);
	for (size_t i = 0; i < value_string.size(); ++i) {
	    if (value_string[i] == '.') { value_string[i] = 'p'; }  // Decimal
	    if (value_string[i] == '+') { value_string[i] = 'P'; }  // Plus
	    if (value_string[i] == '-') { value_string[i] = 'M'; }  // Minus
	}
	return value_string;
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
    string get_file_name(string file_path) {
	return string(basename(const_cast<char *>(file_path.c_str())));
    }

    void read_line(ifstream *file,  vector<string> *tokens) {
	string line;
	getline(*file, line);
	util_string::split_by_space_tab(line, tokens);
    }

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

    void binary_write_string(const string &value, ostream& file) {
	size_t string_length = value.length();
	binary_write_primitive(string_length, file);
	file.write(value.c_str(), string_length);
    }

    void binary_read_string(istream& file, string *value){
	size_t string_length;
	binary_read_primitive(file, &string_length);
	char* buffer = new char[string_length];
	file.read(buffer, string_length);
	value->assign(buffer, string_length);
	delete[] buffer;
    }

    void binary_write(const unordered_map<string, size_t> &table,
		      const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	binary_write_primitive(table.size(), file);
	for (const auto &pair : table) {
	    binary_write_string(pair.first, file);
	    binary_write_primitive(pair.second, file);
	}
    }

    void binary_read(const string &file_path,
		     unordered_map<string, size_t> *table) {
	table->clear();
	ifstream file(file_path, ios::in | ios::binary);
	size_t num_keys;
	binary_read_primitive(file, &num_keys);
	for (size_t i = 0; i < num_keys; ++i) {
	    string key;
	    size_t value;
	    binary_read_string(file, &key);
	    binary_read_primitive(file, &value);
	    (*table)[key] = value;
	}
    }
}  // namespace util_file

namespace util_math {
    void permute_indices(size_t num_indices, vector<size_t> *permuted_indices) {
	permuted_indices->clear();
	for (size_t i = 0; i < num_indices; ++i) {
	    permuted_indices->push_back(i);
	}
	size_t seed = chrono::system_clock::now().time_since_epoch().count();
	shuffle(permuted_indices->begin(), permuted_indices->end(),
		default_random_engine(seed));
    }

    double log0(double a) {
	if (a > 0.0) {
	    return log(a);
	} else if (a == 0.0) {
	    return -numeric_limits<double>::infinity();
	} else {
	    ASSERT(false, "Cannot take log of negative value: " << a);
	}
    }

    double sum_logs(double log_a, double log_b) {
	if (log_a < log_b) {
	    double temp = log_a;
	    log_a = log_b;
	    log_b = temp;
	}
	if (log_a <= -numeric_limits<double>::infinity()) { return log_a; }

	double negative_difference = log_b - log_a;
	return (negative_difference < -20) ?
	    log_a : log_a + log(1.0 + exp(negative_difference));
    }

    double compute_spearman(const vector<double> &sequence1,
			    const vector<double> &sequence2) {
	ASSERT(sequence1.size() == sequence2.size(), "Different lengths: "
	       << sequence1.size() << " vs " << sequence2.size());

	vector<double> sequence1_transformed;
	vector<double> sequence2_transformed;
	transform_average_rank(sequence1, &sequence1_transformed);
	transform_average_rank(sequence2, &sequence2_transformed);

	size_t num_instances = sequence1.size();
	double sum_squares = 0;
	for (size_t i = 0; i < num_instances; ++i) {
	    sum_squares += pow(sequence1_transformed[i] -
			       sequence2_transformed[i], 2);
	}
	double uncorrelatedness = 6.0 * sum_squares /
	    (num_instances * (pow(num_instances, 2) - 1));
	double corrleation = 1.0 - uncorrelatedness;
	return corrleation;
    }

    void transform_average_rank(const vector<double> &sequence,
				vector<double> *transformed_sequence) {
	transformed_sequence->clear();
	vector<double> sorted_values = sequence;
	sort(sorted_values.begin(), sorted_values.end());
	vector<double> averaged_ranks;
	size_t index = 0;
	while (index < sorted_values.size()) {
	    size_t num_same = 1;
	    size_t rank_sum = index + 1;
	    while (index + 1 < sorted_values.size() &&
		   fabs(sorted_values[index + 1] -
			sorted_values[index]) < 1e-15) {
		++index;
		++num_same;
		rank_sum += index + 1;
	    }

	    double averaged_rank = ((double) rank_sum) / num_same;
	    for (size_t j = 0; j < num_same; ++j) {
		// Assign the average rank to all tied elements.
		averaged_ranks.push_back(averaged_rank);
	    }
	    ++index;
	}

	// Map each value to the corresponding index in averaged_ranks. A value
	// can appear many times but it doesn't matter since it will have the
	// same averaged rank.
	unordered_map<double, size_t> value2index;
	for (size_t index = 0; index < sorted_values.size(); ++index) {
	    value2index[sorted_values[index]] = index;
	}
	for (double value : sequence) {
	    size_t index = value2index[value];
	    transformed_sequence->push_back(averaged_ranks[index]);
	}
    }
}  // namespace util_math
