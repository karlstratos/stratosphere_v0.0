// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the evaluation code.

#include "gtest/gtest.h"

#include "../evaluate.h"

// Test class that provides example sequences.
class SequenceExample : public testing::Test {
protected:
    vector<vector<string> > true_sequences_ = {{"a", "b", "c"}, {"b", "c"}};
    vector<vector<string> > predicted_sequences_matching_labels_ =
    {{"a", "c", "c"}, {"b", "c"}};
    vector<vector<string> > predicted_sequences_unmatching_labels_ =
    {{"0", "1", "2"}, {"1", "2"}};
    double position_accuracy_;
    double sequence_accuracy_;
};


// Checks evaluating sequence predictions with matching labels.
TEST_F(SequenceExample, MatchingLabels) {
    evaluate::evaluate_sequences(true_sequences_,
				 predicted_sequences_matching_labels_,
				 &position_accuracy_, &sequence_accuracy_);
    EXPECT_NEAR(80.0, position_accuracy_, 1e-15);
    EXPECT_NEAR(50.0, sequence_accuracy_, 1e-15);
}

// Checks evaluating sequence predictions with unmatching labels.
TEST_F(SequenceExample, UnmatchingLabels) {
    evaluate::evaluate_sequences(true_sequences_,
				 predicted_sequences_unmatching_labels_,
				 &position_accuracy_, &sequence_accuracy_);
    EXPECT_NEAR(0.0, position_accuracy_, 1e-15);
    EXPECT_NEAR(0.0, sequence_accuracy_, 1e-15);

    unordered_map<string, string> label_mapping;
    evaluate::evaluate_sequences_mapping_labels(
	true_sequences_, predicted_sequences_unmatching_labels_,
	&position_accuracy_, &sequence_accuracy_, &label_mapping);
    EXPECT_NEAR(100.0, position_accuracy_, 1e-15);
    EXPECT_NEAR(100.0, sequence_accuracy_, 1e-15);

    EXPECT_EQ("a", label_mapping["0"]);
    EXPECT_EQ("b", label_mapping["1"]);
    EXPECT_EQ("c", label_mapping["2"]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
