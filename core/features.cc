// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "features.h"

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
	    return "ALL-DIGIT";
	} else if (is_all_uppercase) {
	    return "ALL-UPPER";
	} else if (is_all_lowercase) {
	    return "ALL-LOWER";
	} else if (is_capitalized) {
	    return "CAPITALIZED";
	} else {
	    return "OTHER";
	}
    }
}
