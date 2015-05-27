// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the HMM code.

#include "gtest/gtest.h"

#include "../hmm.h"
#include "../util.h"

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
	data_file << "the D\ndog N\nsaw V\nthe D\ncat N\n. S\n" << endl;
	data_file << "the D\ndog N\nbarked V\n. S\n" << endl;
	data_file << "the D\ncat N\nlaughed V\n! S" << endl;

	// True MLE parameter estimates.
	true_emission_["D"]["the"] = 1.0;
	true_emission_["N"]["dog"] = 0.5;
	true_emission_["N"]["cat"] = 0.5;
	true_emission_["V"]["saw"] = 1.0 / 3.0;
	true_emission_["V"]["barked"] = 1.0 / 3.0;
	true_emission_["V"]["laughed"] = 1.0 / 3.0;
	true_emission_["S"]["."] = 2.0 / 3.0;
	true_emission_["S"]["!"] = 1.0 / 3.0;
	true_transition_["D"]["N"] = 1.0;
	true_transition_["N"]["V"] = 0.75;
	true_transition_["N"]["S"] = 0.25;
	true_transition_["V"]["D"] = 1.0 / 3.0;
	true_transition_["V"]["S"] = 2.0 / 3.0;
	true_prior_["D"] = 1.0;
	true_stop_["S"] = 1.0;
    }

    virtual void TearDown() {
	remove(data_file_path_.c_str());
	remove(model_file_path_.c_str());
    }

    string data_file_path_;
    string model_file_path_;
    double tol_ = 1e-10;
    unordered_map<string, unordered_map<string, double> > true_emission_;
    unordered_map<string, unordered_map<string, double> > true_transition_;
    unordered_map<string, double> true_prior_;
    unordered_map<string, double> true_stop_;
};

// Checks supervised training.
TEST_F(LabeledDataExample, CheckSupervisedTraining) {
    HMM hmm;
    hmm.TrainSupervised(data_file_path_);
    for (const auto &state_pair: true_emission_) {
	for (const auto &observation_pair: state_pair.second) {
	    EXPECT_NEAR(observation_pair.second,
			hmm.EmissionProbability(state_pair.first,
						observation_pair.first), tol_);
	}
    }
    for (const auto &state1_pair: true_transition_) {
	for (const auto &state2_pair: state1_pair.second) {
	    EXPECT_NEAR(state2_pair.second,
			hmm.TransitionProbability(state1_pair.first,
						  state2_pair.first), tol_);
	}
    }
    for (const auto &state_pair: true_prior_) {
	EXPECT_NEAR(state_pair.second,
		    hmm.PriorProbability(state_pair.first), tol_);
    }
    for (const auto &state_pair: true_stop_) {
	EXPECT_NEAR(state_pair.second,
		    hmm.StoppingProbability(state_pair.first), tol_);
    }
}

// Checks saving and loading a trained model
TEST_F(LabeledDataExample, CheckSavingAndLoadingTrainedModel) {
    HMM hmm1;
    hmm1.TrainSupervised(data_file_path_);
    hmm1.Save(model_file_path_);

    HMM hmm2(model_file_path_);  // Loading.
    for (const auto &state_pair: true_emission_) {
	for (const auto &observation_pair: state_pair.second) {
	    EXPECT_NEAR(observation_pair.second,
			hmm2.EmissionProbability(state_pair.first,
						 observation_pair.first), tol_);
	}
    }
    for (const auto &state1_pair: true_transition_) {
	for (const auto &state2_pair: state1_pair.second) {
	    EXPECT_NEAR(state2_pair.second,
			hmm2.TransitionProbability(state1_pair.first,
						   state2_pair.first), tol_);
	}
    }
    for (const auto &state_pair: true_prior_) {
	EXPECT_NEAR(state_pair.second,
		    hmm2.PriorProbability(state_pair.first), tol_);
    }
    for (const auto &state_pair: true_stop_) {
	EXPECT_NEAR(state_pair.second,
		    hmm2.StoppingProbability(state_pair.first), tol_);
    }
}

// Test class that provides a random HMM.
class RandomHMM : public testing::Test {
protected:
    virtual void SetUp() {
	hmm_.CreateRandomly(num_observations_, num_states_);
	srand(time(NULL));
	for (size_t i = 0; i < length_; ++i) {
	    observation_string_sequence_.push_back(
		"observation" + to_string(rand() % num_observations_));
	}
    }
    HMM hmm_;
    size_t num_observations_ = 6;
    size_t num_states_ = 3;
    double tol_ = 1e-10;
    size_t length_ = 8;
    vector<string> observation_string_sequence_;
};

// Checks the correctness of Viterbi decoding.
TEST_F(RandomHMM, Viterbi) {
    vector<string> state_string_sequence1;
    double max_sequence_log_probability_viterbi =
	hmm_.Viterbi(observation_string_sequence_, &state_string_sequence1);

    hmm_.set_debug(true);
    vector<string> state_string_sequence2;
    double max_sequence_log_probability_viterbi_exhaustive =
	hmm_.Viterbi(observation_string_sequence_, &state_string_sequence2);

    EXPECT_NEAR(max_sequence_log_probability_viterbi_exhaustive,
		max_sequence_log_probability_viterbi, tol_);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
