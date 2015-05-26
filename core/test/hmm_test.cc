// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the HMM code.

#include "gtest/gtest.h"

#include <stdio.h>
#include <stdlib.h>

#include "../src/hmm.h"
#include "../src/util.h"

// Test class that provides a simple labeled dataset.
class LabeledDataExample : public testing::Test {
protected:
    virtual void SetUp() {
	// the_D dog_N saw_V the_D cat_N ._S
	// the_D dog_N barked_V ._S
	// the_D cat_N laughed_V !_S
	model_file_path_ = tmpnam(nullptr);
	data_file_path_ = tmpnam(nullptr);
	ofstream data_file(data_file_path_, ios::out);
	data_file << "the D" << endl;
	data_file << "dog N" << endl;
	data_file << "saw V" << endl;
	data_file << "the D" << endl;
	data_file << "cat N" << endl;
	data_file << ". S" << endl;
	data_file << endl;
	data_file << "the D" << endl;
	data_file << "dog N" << endl;
	data_file << "barked V" << endl;
	data_file << ". S" << endl;
	data_file << endl;
	data_file << "the D" << endl;
	data_file << "cat N" << endl;
	data_file << "laughed V" << endl;
	data_file << "! S" << endl;

	// True MLE parameter values.
	true_o_["D"]["the"] = 1.0;
	true_o_["N"]["dog"] = 0.5;
	true_o_["N"]["cat"] = 0.5;
	true_o_["V"]["saw"] = 1.0 / 3.0;
	true_o_["V"]["barked"] = 1.0 / 3.0;
	true_o_["V"]["laughed"] = 1.0 / 3.0;
	true_o_["S"]["."] = 2.0 / 3.0;
	true_o_["S"]["!"] = 1.0 / 3.0;
	true_t_["D"]["N"] = 1.0;
	true_t_["N"]["V"] = 3.0 / 4.0;
	true_t_["N"]["S"] = 1.0 / 4.0;
	true_t_["V"]["D"] = 1.0 / 3.0;
	true_t_["V"]["S"] = 2.0 / 3.0;
	true_pi_["D"] = 1.0;

    }

    virtual void TearDown() { }

    string data_file_path_;
    string model_file_path_;
    StringManipulator string_manipulator_;
    string line_;
    vector<string> tokens_;
    double tol_ = 1e-4;

    unordered_map<string, unordered_map<string, double> > true_o_;
    unordered_map<string, unordered_map<string, double> > true_t_;
    unordered_map<string, double> true_pi_;
};

// Checks supervised training.
TEST_F(LabeledDataExample, CheckSupervisedTraining) {
    HMM hmm;
    hmm.TrainSupervised(data_file_path_);

    for (const auto &h_pair: true_o_) {
	H h = hmm.h_str2num(h_pair.first);
	for (const auto &x_pair: h_pair.second) {
	    X x = hmm.x_str2num(x_pair.first);
	    EXPECT_NEAR(x_pair.second, exp(hmm.o(h, x)), tol_);
	}
    }

    for (const auto &h1_pair: true_t_) {
	H h1 = hmm.h_str2num(h1_pair.first);
	for (const auto &h2_pair: h1_pair.second) {
	    H h2 = hmm.h_str2num(h2_pair.first);
	    EXPECT_NEAR(h2_pair.second, exp(hmm.t(h1, h2)), tol_);
	}
    }
    // Additionally check the stopping probability.
    EXPECT_NEAR(1.0, exp(hmm.t(hmm.h_str2num("S"), hmm.StoppingState())), tol_);

    for (const auto &h_pair: true_pi_) {
	H h = hmm.h_str2num(h_pair.first);
	EXPECT_NEAR(h_pair.second, exp(hmm.pi(h)), tol_);
    }
}

// Checks saving and loading a trained model
TEST_F(LabeledDataExample, CheckSavingAndLoadingTrainedModel) {
    HMM hmm1;
    hmm1.TrainSupervised(data_file_path_);
    hmm1.Save(model_file_path_);

    HMM hmm2(model_file_path_);
    for (const auto &h_pair: true_o_) {
	H h = hmm2.h_str2num(h_pair.first);
	for (const auto &x_pair: h_pair.second) {
	    X x = hmm2.x_str2num(x_pair.first);
	    EXPECT_NEAR(x_pair.second, exp(hmm2.o(h, x)), tol_);
	}
    }

    for (const auto &h1_pair: true_t_) {
	H h1 = hmm2.h_str2num(h1_pair.first);
	for (const auto &h2_pair: h1_pair.second) {
	    H h2 = hmm2.h_str2num(h2_pair.first);
	    EXPECT_NEAR(h2_pair.second, exp(hmm2.t(h1, h2)), tol_);
	}
    }
    // Additionally check the stopping probability.
    EXPECT_NEAR(1.0, exp(hmm2.t(hmm2.h_str2num("S"), hmm2.StoppingState())),
		tol_);

    for (const auto &h_pair: true_pi_) {
	H h = hmm2.h_str2num(h_pair.first);
	EXPECT_NEAR(h_pair.second, exp(hmm2.pi(h)), tol_);
    }
}

// Test class that provides a random HMM.
class RandomHMM : public testing::Test {
protected:
    virtual void SetUp() {
	hmm_.CreateRandomly(num_observations_, num_states_);
	srand(time(NULL));
	for (size_t i = 0; i < length_; ++i) {
	    X x = rand() % num_observations_;
	    x_sequence_.push_back("x" + to_string(x));
	}
    }

    virtual void TearDown() { }

    HMM hmm_;
    size_t num_observations_ = 6;
    size_t num_states_ = 3;
    double tol_ = 1e-4;
    size_t length_ = 8;
    vector<string> x_sequence_;
};

// Checks the correctness of Viterbi decoding.
TEST_F(RandomHMM, Viterbi) {
    vector<string> viterbi_sequence;
    double viterbi_lprob = hmm_.Viterbi(x_sequence_, &viterbi_sequence);

    vector<string> viterbi_exhaustive_sequence;
    double viterbi_exhaustive_lprob =
	hmm_.ViterbiExhaustive(x_sequence_, &viterbi_exhaustive_sequence);

    EXPECT_NEAR(viterbi_exhaustive_lprob, viterbi_lprob, tol_);
}

// Checks the correctness of the forward-backward algorithm.
TEST_F(RandomHMM, ForwardBackward) {
    double lprob_forward =
	hmm_.ComputeObservationLikelihoodForward(x_sequence_);
    double lprob_backward =
	hmm_.ComputeObservationLikelihoodForward(x_sequence_);
    double lprob_exhaustive =
	hmm_.ComputeObservationLikelihoodExhaustive(x_sequence_);

    EXPECT_NEAR(lprob_exhaustive, lprob_forward, tol_);
    EXPECT_NEAR(lprob_exhaustive, lprob_backward, tol_);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
