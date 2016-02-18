// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "features.h"

#include <assert.h>

namespace features {
    string basic_word_shape(const string &word_string) {
	bool is_all_digit = true;
	bool is_all_uppercase = true;
	bool is_all_lowercase = true;
	bool is_capitalized = true;
	int index = 0;
	for (const char &c : word_string) {
	    if (!isdigit(c)) is_all_digit = false;
	    if (!isupper(c)) is_all_uppercase = false;
	    if (!islower(c)) is_all_lowercase = false;
	    if ((index == 0 && !isupper(c)) || (index > 0 && !islower(c))) {
		is_capitalized = false;
	    }
	    ++index;
	}

	if (is_all_digit) {
	    return "<shape5>=all-digit";
	} else if (is_all_uppercase) {
	    return "<shape5>=all-upper";
	} else if (is_all_lowercase) {
	    return "<shape5>=all-lower";
	} else if (is_capitalized) {
	    return "<shape5>=capitalized";
	} else {
	    return "<shape5>=other";
	}
    }

    string word_identity(const string &word_string) {
	return "<id>=" + word_string;
    }

    string prefix(const string &word_string, size_t prefix_length) {
	assert(prefix_length <= word_string.size());
	return "<prefix" + to_string(prefix_length) + ">=" +
	    word_string.substr(0, prefix_length);
    }

    string suffix(const string &word_string, size_t suffix_length) {
	assert(suffix_length <= word_string.size());
	return "<suffix" + to_string(suffix_length) + ">=" +
	    word_string.substr(word_string.size() - suffix_length);
    }

    string contains_digit(const string &word_string) {
	for (char c : word_string) {
	    if (isdigit(c)) { return "<contains-digit>=t"; }
	}
	return "<contains-digit>=f";
    }

    string contains_hyphen(const string &word_string) {
	for (char c : word_string) {
	    if (c == 45) { return "<contains-hyphen>=t"; }
	}
	return "<contains-hyphen>=f";
    }

    string is_capitalized(const string &word_string) {
	return isupper(word_string.at(0)) ? "<is-capitalized>=t" :
	    "<is-capitalized>=f";
    }
}
