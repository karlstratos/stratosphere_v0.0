// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code related to feature extraction.

#ifndef CORE_FEATURES_H_
#define CORE_FEATURES_H_

#include <string>

using namespace std;

namespace features {
    // A basic 5-way word classifier: ALL-DIGIT, ALL-UPPER, ALL-LOWER,
    // CAPITALIZED, and OTHER.
    string basic_word_shape(const string &word_string);
}  // namespace features

#endif  // CORE_FEATURES_H_
