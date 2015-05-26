// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the evaluation code.

#include "gtest/gtest.h"

#include "../evaluate.h"

// Checks evaluating sequence predictions with matching labels.
TEST(EvaluateSequences, MatchingLabels) {
    vector<vector<string> > true_sequences = {{"a", "b", "c"}, {"b", "c"}};
    vector<vector<string> > predicted_sequences =
	{{"a", "c", "c"}, {"b", "c"}};
    double position_accuracy;
    double sequence_accuracy;
    double many_to_one_accuracy;
    unordered_map<string, string> many_to_one_map;
    evaluate::evaluate_sequences(true_sequences, predicted_sequences,
				 &position_accuracy, &sequence_accuracy,
				 &many_to_one_accuracy, &many_to_one_map);
    EXPECT_NEAR(80.0, position_accuracy, 1e-15);
    EXPECT_NEAR(50.0, sequence_accuracy, 1e-15);
    EXPECT_NEAR(80.0, many_to_one_accuracy, 1e-15);
}

// Checks evaluating sequence predictions with unmatching labels.
TEST(EvaluateSequences, UnmatchingLabels) {
    vector<vector<string> > true_sequences = {{"a", "b", "c"}, {"b", "c"}};
    vector<vector<string> > predicted_sequences =
	{{"0", "1", "2"}, {"1", "2"}};
    double position_accuracy;
    double sequence_accuracy;
    double many_to_one_accuracy;
    unordered_map<string, string> many_to_one_map;
    evaluate::evaluate_sequences(true_sequences, predicted_sequences,
				 &position_accuracy, &sequence_accuracy,
				 &many_to_one_accuracy, &many_to_one_map);
    EXPECT_NEAR(0.0, position_accuracy, 1e-15);
    EXPECT_NEAR(0.0, sequence_accuracy, 1e-15);
    EXPECT_NEAR(100.0, many_to_one_accuracy, 1e-15);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
