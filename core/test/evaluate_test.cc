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

// Checks evaluting word similarity with a random example.
TEST(WordSimilarity, ManualExample) {
    Eigen::VectorXd x1(2);
    Eigen::VectorXd x2(2);
    Eigen::VectorXd x3(2);
    Eigen::VectorXd x4(2);
    Eigen::VectorXd x5(2);
    Eigen::VectorXd x6(2);
    Eigen::VectorXd y1(2);
    Eigen::VectorXd y2(2);
    Eigen::VectorXd y3(2);
    Eigen::VectorXd y4(2);
    Eigen::VectorXd y5(2);
    Eigen::VectorXd y6(2);
    x1 << 0.9134, 0.6324;
    x2 << 0.0975, 0.2785;
    x3 << 0.5469, 0.9575;
    x4 << 0.1419, 0.4218;
    x5 << 0.9157, 0.7922;
    x6 << 0.9595, 0.6557;
    y1 << 0.9649, 0.1576;
    y2 << 0.9706, 0.9572;
    y3 << 0.4854, 0.8003;
    y4 << 0.0357, 0.8491;
    y5 << 0.9340, 0.6787;
    y6 << 0.7577, 0.7431;
    unordered_map<string, Eigen::VectorXd> word_vectors;
    word_vectors["x1"] = x1;
    word_vectors["x2"] = x2;
    word_vectors["x3"] = x3;
    word_vectors["x4"] = x4;
    word_vectors["x5"] = x5;
    word_vectors["x6"] = x6;
    word_vectors["y1"] = y1;
    word_vectors["y2"] = y2;
    word_vectors["y3"] = y3;
    word_vectors["y4"] = y4;
    word_vectors["y5"] = y5;
    word_vectors["y6"] = y6;
    vector<tuple<string, string, double> > word_pair_scores;
    word_pair_scores.push_back(make_tuple("x1", "y1", -1));
    word_pair_scores.push_back(make_tuple("x2", "y2", 100));
    word_pair_scores.push_back(make_tuple("x3", "y3", 101));
    word_pair_scores.push_back(make_tuple("x4", "y4", -3));
    word_pair_scores.push_back(make_tuple("x5", "y5", 1000));
    word_pair_scores.push_back(make_tuple("x6", "y6", 1002));
    word_pair_scores.push_back(make_tuple("x7", "y1", 1002));
    size_t num_handled;
    double correlation;
    evaluate::evaluate_similarity(word_vectors, word_pair_scores, false,
				  &num_handled, &correlation);
    EXPECT_EQ(6, num_handled);
    EXPECT_NEAR(0.5429, correlation, 1e-4);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
