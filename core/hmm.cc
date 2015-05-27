// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "hmm.h"

#include <iomanip>
#include <limits>
#include <random>

#include "util.h"

void HMM::Clear() {
    observation_dictionary_.clear();
    observation_dictionary_inverse_.clear();
    state_dictionary_.clear();
    state_dictionary_inverse_.clear();
    o_.clear();
    t_.clear();
    pi_.clear();
}

void HMM::CreateRandomly(size_t num_observations, size_t num_states) {
    Clear();

    // Create an observation dictionary.
    for (X x = 0; x < num_observations; ++x) {
	string observation_string = "x" + to_string(x);
	observation_dictionary_[observation_string] = x;
	observation_dictionary_inverse_[x] = observation_string;
    }

    // Create a state dictionary.
    for (H h = 0; h < num_states; ++h) {
	string state_string = "h" + to_string(h);
	state_dictionary_[state_string] = h;
	state_dictionary_inverse_[h] = state_string;
    }

    random_device device;
    default_random_engine engine(device());
    normal_distribution<double> normal(0.0, 1.0);  // Standard Gaussian.

    // Generate emission parameters.
    o_.resize(num_states);
    for (H h = 0; h < num_states; ++h) {
	o_[h].resize(num_observations);
	double normalizer_o_h = 0.0;
	for (X x = 0; x < num_observations; ++x) {
	    double value = fabs(normal(engine));
	    o_[h][x] = value;
	    normalizer_o_h += value;
	}
	for (X x = 0; x < num_observations; ++x) {
	    o_[h][x] = util_math::log0(o_[h][x] / normalizer_o_h);
	}
    }

    // Generate transition parameters.
    t_.resize(num_states);
    for (H h1 = 0; h1 < num_states; ++h1) {
	t_[h1].resize(num_states + 1);  // +1 for stopping state.
	double normalizer_t_h1 = 0.0;
	for (H h2 = 0; h2 < num_states + 1; ++h2) {  // +1 for stopping state.
	    double value = fabs(normal(engine));
	    t_[h1][h2] = value;
	    normalizer_t_h1 += value;
	}
	for (H h2 = 0; h2 < num_states + 1; ++h2) {  // +1 for stopping state.
	    t_[h1][h2] = util_math::log0(t_[h1][h2] / normalizer_t_h1);
	}
    }

    // Generate prior parameters.
    pi_.resize(num_states);
    double normalizer_pi = 0.0;
    for (H h = 0; h < num_states; ++h) {
	double value = fabs(normal(engine));
	pi_[h] = util_math::log0(value);
	normalizer_pi += value;
    }
    for (H h = 0; h < num_states; ++h) {
	pi_[h] = util_math::log0(pi_[h] / normalizer_pi);
    }

    CheckProperDistribution();
}

void HMM::Save(const string &model_path) {
    ofstream model_file(model_path, ios::out | ios::binary);
    size_t num_observations = NumObservations();
    size_t num_states = NumObservations();
    util_file::binary_write_primitive(num_observations, model_file);
    util_file::binary_write_primitive(num_states, model_file);
    for (const auto &observation_pair : observation_dictionary_) {
	string observation_string = observation_pair.first;
	X x = observation_pair.second;
	util_file::binary_write_string(observation_string, model_file);
	util_file::binary_write_primitive(x, model_file);
    }
    for (const auto &state_pair : state_dictionary_) {
	string state_string = state_pair.first;
	H h = state_pair.second;
	util_file::binary_write_string(state_string, model_file);
	util_file::binary_write_primitive(h, model_file);
    }
    for (H h = 0; h < o_.size(); ++h) {
	for (X x = 0; x < o_[h].size(); ++x) {
	    double value_o_h_x = o_[h][x];
	    util_file::binary_write_primitive(value_o_h_x, model_file);
	}
    }
    for (size_t h1 = 0; h1 < t_.size(); ++h1) {
	for (size_t h2 = 0; h2 < t_[h1].size(); ++h2) {
	    double value_t_h1_h2 = t_[h1][h2];
	    util_file::binary_write_primitive(value_t_h1_h2, model_file);
	}
    }
    for (size_t h = 0; h < pi_.size(); ++h) {
	double value_pi_h = pi_[h];
	util_file::binary_write_primitive(value_pi_h, model_file);
    }
}

void HMM::Load(const string &model_path) {
    Clear();
    ifstream model_file(model_path, ios::in | ios::binary);
    size_t num_observations;
    size_t num_states;
    util_file::binary_read_primitive(model_file, &num_observations);
    util_file::binary_read_primitive(model_file, &num_states);
    for (size_t i = 0; i < num_observations; ++i) {
	string observation_string;
	X x;
	util_file::binary_read_string(model_file, &observation_string);
	util_file::binary_read_primitive(model_file, &x);
	observation_dictionary_[observation_string] = x;
	observation_dictionary_inverse_[x] = observation_string;
    }
    for (size_t i = 0; i < num_states; ++i) {
	string state_string;
	H h;
	util_file::binary_read_string(model_file, &state_string);
	util_file::binary_read_primitive(model_file, &h);
	state_dictionary_[state_string] = h;
	state_dictionary_inverse_[h] = state_string;
    }
    o_.resize(m);
    for (H h = 0; h < num_states; ++h) {
	o_[h].resize(num_observations, -numeric_limits<double>::infinity());
	for (X x = 0; x < num_observations; ++x) {
	    double value_o_h_x;
	    util_file::binary_read_primitive(model_file, &value_o_h_x);
	    o_[h][x] = value_o_h_x;
	}
    }
    t_.resize(num_states);
    for (H h1 = 0; h1 < num_states; ++h1) {
	t_[h1].resize(num_states + 1,  // +1 for stopping state.
		      -numeric_limits<double>::infinity());
	for (H h2 = 0; h2 < num_states + 1; ++h2) {  // +1 for stopping state.
	    double value_t_h1_h2;
	    util_file::binary_read_primitive(model_file, &value_t_h1_h2);
	    t_[h1][h2] = value_t_h1_h2;
	}
    }
    pi_.resize(num_states, -numeric_limits<double>::infinity());
    for (H h = 0; h < num_states; ++h) {
	double value_pi_h;
	util_file::binary_read_primitive(model_file, &value_pi_h);
    }
    CheckProperDistribution();
}

void HMM::TrainSupervised(const string &labeled_data_path) {
    vector<vector<string> > observation_sequences;
    vector<vector<string> > state_sequences;
    bool fully_labeled;
    ReadData(labeled_data_path, &observation_sequences, &state_sequences,
	     &fully_labeled);
    ASSERT(fully_labeled, "Data not fully labeled");
    TrainSupervised(observation_sequences, state_sequences);
}

void HMM::TrainSupervised(const vector<vector<string> > &observation_sequences,
			  const vector<vector<string> > &state_sequences) {
    ASSERT(observation_sequences.size() == state_sequences.size(), "Mismatch");
    Clear();
    unordered_map<H, unordered_map<X, size_t> > count_h_x;
    unordered_map<H, unordered_map<H, size_t> > count_h1_h2;
    unordered_map<H, size_t> count_h_initial;
    for (size_t i = 0; i < observation_sequences.size(); ++i) {
	size_t length = observation_sequences[i].size();
        ASSERT(length > 0 && length == state_sequences[i].size(), "Invalid");
	H h_initial = AddStateIfUnknown(state_sequences[i][0]);
	++count_h_initial[h_initial];
	for (size_t j = 0; j < length; ++j) {
	    X x = AddObservationIfUnknown(observation_sequences[i][j]);
	    H h2 = AddStateIfUnknown(state_sequences[i][j]);
	    ++count_h_x[h2][x];
	    if (j > 0) {
		H h1 = h_dictionary_[state_sequences[i][j - 1]];
		++count_h1_h2[h1][h2];
	    }
	}
	H h_final = h_dictionary_[state_sequences[i][length - 1]];
	++count_h1_h2[h_final][StoppingState()];
    }
    o_.resize(NumStates());
    for (const auto &h_pair : count_h_x) {
	size_t sum = 0;
	for (const auto &observation_pair : h_pair.second) { sum += observation_pair.second; }
	o_[h_pair.first].resize(NumObservations(),
				-numeric_limits<double>::infinity());
	for (const auto &observation_pair : h_pair.second) {
	    o_[h_pair.first][observation_pair.first] = log(observation_pair.second) - log(sum);
	}
    }
    t_.resize(NumStates());
    for (const auto &h1_pair : count_h1_h2) {
	size_t sum = 0;
	for (const auto &h2_pair : h1_pair.second) { sum += h2_pair.second; }
	t_[h1_pair.first].resize(NumStates() + 1,  // +1 for stopping state.
				 -numeric_limits<double>::infinity());
	for (const auto &h2_pair : h1_pair.second) {
	    t_[h1_pair.first][h2_pair.first] = log(h2_pair.second) - log(sum);
	}
    }
    size_t sum = 0;
    for (const auto &h_pair : count_h_initial) { sum += h_pair.second; }
    pi_.resize(NumStates(), -numeric_limits<double>::infinity());
    for (const auto &h_pair : count_h_initial) {
	pi_[h_pair.first] = log(h_pair.second) - log(sum);
    }
}

void HMM::Predict(const string &data_path, const string &prediction_path) {
    vector<vector<string> > observation_sequences;
    vector<vector<string> > state_sequences;
    bool fully_labeled;
    ReadData(labeled_data_path, &observation_sequences, &state_sequences,
	     &fully_labeled);
    vector<vector<string> > pred_sequences;
    for (size_t i = 0; i < observation_sequences.size(); ++i) {
	vector<string> pred_sequence;
	if (decoding_method_ == "viterbi") {
	    Viterbi(observation_sequences[i], &pred_sequence);
	} else if (decoding_method_ == "mbr") {
	    MinimumBayesRisk(observation_sequences[i], &pred_sequence);
	} else {
	    ASSERT(false, "Unknown decoding method: " << decoding_method_);
	}
	pred_sequences.push_back(pred_sequence);
    }

    // TODO: from here.
    double position_acc;
    double sequence_acc;
    double many2one_acc;
    unordered_map<string, string> many2one_map;
    EvaluatePrediction(h_sequences, pred_sequences, &position_acc,
		       &sequence_acc, &many2one_acc, &many2one_map);
    if (fully_labeled && verbose_) {
	cerr << setprecision(4);
	cerr << "Position: " << position_acc << "     \tSequence: "
	     << sequence_acc << "     \tManyToOne: " << many2one_acc << endl;
    }

    bool report_many2one = (position_acc == 0.0 && many2one_acc > 0.0);
    if (!prediction_path.empty()) {
	ofstream prediction_file(prediction_path, ios::out);
	for (size_t i = 0; i < observation_sequences.size(); ++i) {
	    for (size_t j = 0; j < observation_sequences[i].size(); ++j) {
		string observation_string = observation_sequences[i][j];
		string h_string = h_sequences[i][j];
		string pred_string = pred_sequences[i][j];
		if (report_many2one) {
		    pred_string = many2one_map[pred_string];
		}
		prediction_file << observation_string << " " << h_string << " "
				<< pred_string << endl;
	    }
	    if (i < observation_sequences.size() - 1) { prediction_file << endl; }
	}
    }
}

void HMM::EvaluatePrediction(const vector<vector<string> > &gold_sequences,
			     const vector<vector<string> > &pred_sequences,
			     double *position_acc, double *sequence_acc,
			     double *many2one_acc,
			     unordered_map<string, string> *many2one_map) {
    size_t num_states = 0;
    size_t num_states_correct = 0;
    size_t num_sequences_correct = 0;
    unordered_map<string, unordered_map<string, size_t> > count_pred_gold;
    for (size_t i = 0; i < gold_sequences.size(); ++i) {
	num_states += gold_sequences[i].size();
	bool entire_sequence_is_correct = true;
	for (size_t j = 0; j < gold_sequences[i].size(); ++j) {
	    string gold_string = gold_sequences[i][j];
	    string pred_string = pred_sequences[i][j];
	    ++count_pred_gold[pred_string][gold_string];  // pred -> gold
	    if (pred_string == gold_string) {
		num_states_correct += 1;
	    } else {
		entire_sequence_is_correct = false;
	    }
	}
	if (entire_sequence_is_correct) { num_sequences_correct += 1; }
    }
    (*position_acc) = ((double) num_states_correct) / num_states * 100;
    (*sequence_acc) = ((double) num_sequences_correct) /
	gold_sequences.size() * 100;

    // Map each predicted string to the most frequently co-occurring state
    // string.
    for (const auto &pred_pair: count_pred_gold) {
	vector<pair<string, size_t> > v;
	for (const auto &gold_pair: pred_pair.second) {
	    v.emplace_back(gold_pair.first, gold_pair.second);
	}
        sort(v.begin(), v.end(),
	     sort_pairs_second<string, size_t, greater<size_t> >());
	(*many2one_map)[pred_pair.first] = v[0].first;
    }

    // Use the mapping to obtain many-to-1 accuracy.
    size_t num_states_correct_many2one = 0;
    for (size_t i = 0; i < gold_sequences.size(); ++i) {
	for (size_t j = 0; j < gold_sequences[i].size(); ++j) {
	    string gold_string = gold_sequences[i][j];
	    string pred_string = pred_sequences[i][j];
	    string pred_string_mapped = (*many2one_map)[pred_string];
	    if (pred_string_mapped == gold_string) {
		num_states_correct_many2one += 1;
	    }
	}
    }
    (*many2one_acc) = ((double) num_states_correct_many2one) / num_states * 100;
}

double HMM::Viterbi(const vector<string> &string_sequence,
		    vector<string> *state_sequence) {
    vector<X> observation_sequence;
    ConvertObservation(string_sequence, &observation_sequence);
    vector<H> h_sequence;
    double lprob = Viterbi(observation_sequence, &h_sequence);
    ConvertState(h_sequence, state_sequence);
    return lprob;
}

double HMM::ViterbiExhaustive(const vector<string> &string_sequence,
			      vector<string> *state_sequence) {
    vector<X> observation_sequence;
    ConvertObservation(string_sequence, &observation_sequence);
    vector<H> h_sequence;
    double lprob = ViterbiExhaustive(observation_sequence, &h_sequence);
    ConvertState(h_sequence, state_sequence);
    return lprob;
}

void HMM::MinimumBayesRisk(const vector<string> &string_sequence,
			   vector<string> *state_sequence) {
    vector<X> observation_sequence;
    ConvertObservation(string_sequence, &observation_sequence);
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    vector<vector<double> > be;
    Backward(observation_sequence, &be);
    vector<H> h_sequence;
    for (size_t i = 0; i < observation_sequence.size(); ++i) {
	double max_lprob = -numeric_limits<double>::infinity();
	H h_best = 0;
	for (H h = 0; h < NumStates(); ++h) {
	    double lprob = al[i][h] + be[i][h];
	    if (lprob >= max_lprob) {
		max_lprob = lprob;
		h_best = h;
	    }
	}
	h_sequence.push_back(h_best);
    }
    ConvertState(h_sequence, state_sequence);
}

double HMM::ComputeObservationLikelihoodForward(const vector<string>
						&string_sequence) {
    vector<X> observation_sequence;
    ConvertObservation(string_sequence, &observation_sequence);
    return ComputeLikelihoodForward(observation_sequence);
}

double HMM::ComputeObservationLikelihoodBackward(const vector<string>
						 &string_sequence) {
    vector<X> observation_sequence;
    ConvertObservation(string_sequence, &observation_sequence);
    return ComputeLikelihoodBackward(observation_sequence);
}

double HMM::ComputeObservationLikelihoodExhaustive(const vector<string>
						   &string_sequence) {
    vector<X> observation_sequence;
    ConvertObservation(string_sequence, &observation_sequence);
    return ComputeLikelihoodExhaustive(observation_sequence);
}

X HMM::observation_dictionary(const string &observation_string) {
    ASSERT(observation_dictionary_.find(observation_string) != observation_dictionary_.end(),
	   "Requesting integer ID of an unknown x string: " << observation_string);
    return observation_dictionary_[observation_string];
}

string HMM::observation_dictionary_inverse(X x) {
    if (x == UnknownObservation()) { return "<UNK>"; }
    ASSERT(observation_dictionary_inverse_.find(x) != observation_dictionary_inverse_.end(),
	   "Requesting string of an unknown x integer: " << x);
    return observation_dictionary_inverse_[x];
}

H HMM::h_dictionary(const string &h_string) {
    ASSERT(h_dictionary_.find(h_string) != h_dictionary_.end(),
	   "Requesting integer ID of an unknown h string: "
	   << h_string);
    return h_dictionary_[h_string];
}

string HMM::h_dictionary_inverse(H h) {
    if (h == StoppingState()) { return "<STOP>"; }
    ASSERT(h_dictionary_inverse_.find(h) != h_dictionary_inverse_.end(),
	   "Requesting string of an unknown h integer: " << h);
    return h_dictionary_inverse_[h];
}

void HMM::CheckProperDistribution() {
    ASSERT(NumObservations() > 0 && NumStates() > 0, "Empty dictionary?");
    for (H h = 0; h < NumStates(); ++h) {
	double mass_o_h = 0.0;
	for (X x = 0; x < NumObservations(); ++x) {
	    mass_o_h += exp(o_[h][x]);
	}
	ASSERT(fabs(mass_o_h - 1.0) < 1e-10, "Improper o: " << mass_o_h);
    }

    for (H h1 = 0; h1 < NumStates(); ++h1) {
	double mass_t_h1 = 0.0;
	for (H h2 = 0; h2 < NumStates() + 1; ++h2) {  // +1 for stopping state.
	    mass_t_h1 += exp(t_[h1][h2]);
	}
	ASSERT(fabs(mass_t_h1 - 1.0) < 1e-10, "Improper t: " << mass_t_h1);
    }

    double mass_pi = 0.0;
    for (H h = 0; h < NumStates(); ++h) { mass_pi += exp(pi_[h]); }
    ASSERT(fabs(mass_pi - 1.0) < 1e-10, "Improper pi: " << mass_pi);
}

void HMM::ReadData(const string &data_path,
		   vector<vector<string> > *observation_sequences,
		   vector<vector<string> > *label_sequences,
		   bool *fully_labeled) {
    (*fully_labeled) = true;
    vector<vector<string> > observation_sequences;
    vector<vector<string> > label_sequences;
    vector<string> observation_sequence;
    vector<string> label_sequence;
    ifstream data_file(data_path, ios::in);
    while (data_file.good()) {
	vector<string> tokens;
	util_string::read_line(data_file, &tokens);
	if (tokens.size() > 0) {
	    observation_sequence.push_back(tokens[0]);
	    string label_string;
	    if (tokens.size() == 1) {  // the
		(*fully_labeled) = false;
	    } else if (tokens.size() == 2) {  // the DET
		label_string = tokens[1];
	    } else {
		ASSERT(false, util_string::convert_to_string(tokens));
	    }
	    label_sequence.push_back(label_string);
	} else {
	    if (observation_sequence.size() > 0) {  // End of a sequence.
		observation_sequences.push_back(observation_sequence);
		label_sequences.push_back(label_sequence);
		observation_sequence.clear();
		label_sequence.clear();
	    }
	}
    }
}

X HMM::AddObservationIfUnknown(const string &observation_string) {
    ASSERT(!observation_string.empty(), "Adding an empty observation string!");
    if (observation_dictionary_.find(observation_string) == observation_dictionary_.end()) {
	X x = observation_dictionary_.size();
	observation_dictionary_[observation_string] = x;
	observation_dictionary_inverse_[x] = observation_string;
    }
    return observation_dictionary_[observation_string];
}

H HMM::AddStateIfUnknown(const string &h_string) {
    ASSERT(!h_string.empty(), "Adding an empty state string!");
    if (h_dictionary_.find(h_string) == h_dictionary_.end()) {
	H h = h_dictionary_.size();
	h_dictionary_[h_string] = h;
	h_dictionary_inverse_[h] = h_string;
    }
    return h_dictionary_[h_string];
}

void HMM::ConvertObservation(const vector<string> &string_sequence,
			     vector<X> *observation_sequence) {
    ASSERT(observation_dictionary_.size() > 0, "No observation dictionary");
    observation_sequence->clear();
    for (size_t i = 0; i < string_sequence.size(); ++i) {
	string observation_string = string_sequence[i];
	X x = (observation_dictionary_.find(observation_string) != observation_dictionary_.end()) ?
	    observation_dictionary_[observation_string] : UnknownObservation();
	observation_sequence->push_back(x);
    }
}

void HMM::ConvertObservation(const vector<X> &observation_sequence,
			     vector<string> *string_sequence) {
    ASSERT(observation_dictionary_inverse_.size() > 0, "No observation dictionary");
    string_sequence->clear();
    for (size_t i = 0; i < observation_sequence.size(); ++i) {
	string_sequence->push_back(observation_dictionary_inverse(observation_sequence[i]));
    }
}

void HMM::ConvertState(const vector<H> &h_sequence,
		       vector<string> *string_sequence) {
    ASSERT(h_dictionary_inverse_.size() > 0, "No state dictionary");
    string_sequence->clear();
    for (size_t i = 0; i < h_sequence.size(); ++i) {
	H h = h_sequence[i];
	ASSERT(h_dictionary_inverse_.find(h) != h_dictionary_inverse_.end(), h
	       << " not in state dictionary");
	string h_string = h_dictionary_inverse_[h];
	string_sequence->push_back(h_string);
    }
}

double HMM::Viterbi(const vector<X> &observation_sequence, vector<H> *h_sequence) {
    size_t length = observation_sequence.size();

    // chart[i][h] = highest prob of a sequence ending in state h at position i
    vector<vector<double> > chart(length);
    vector<vector<H> > bp(length);  // bp[i][h] = h_prev (backpointer)
    for (size_t i = 0; i < length; ++i) {
	chart[i].resize(NumStates(), -numeric_limits<double>::infinity());
	bp[i].resize(NumStates());
    }

    // Base: chart[0][h] = pi[h] + o[h][x(0)]
    X x0 = observation_sequence[0];
    for (H h = 0; h < NumStates(); ++h) {
	double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	    -log(NumObservations()) : o_[h][x0];
	chart[0][h] = pi_[h] + lprob_emission_x0;
    }

    // Main body: chart[i][h] =
    //         max_{h_prev} chart[i-1][h_prev] + t[h_prev][h] + o[h][x(i)]
    for (size_t i = 1; i < length; ++i) {
	X xi = observation_sequence[i];
	for (H h = 0; h < NumStates(); ++h) {
	    double lprob_emission_xi = (xi == UnknownObservation()) ?
		-log(NumObservations()) : o_[h][xi];
	    double max_lprob = -numeric_limits<double>::infinity();
	    H best_h_prev = 0;
	    for (H h_prev = 0; h_prev < NumStates(); ++h_prev) {
		double particular_lprob =
		    chart[i - 1][h_prev] + t_[h_prev][h] + lprob_emission_xi;
		if (particular_lprob >= max_lprob) {
		    max_lprob = particular_lprob;
		    best_h_prev = h_prev;
		}
	    }
	    chart[i][h] = max_lprob;
	    bp[i][h] = best_h_prev;
	}
    }

    double best_sequence_lprob = -numeric_limits<double>::infinity();
    H best_h_final = 0;
    for (H h = 0; h < NumStates(); ++h) {
	double sequence_lprob = chart[length - 1][h] + t_[h][StoppingState()];
	if (sequence_lprob >= best_sequence_lprob) {
	    best_sequence_lprob = sequence_lprob;
	    best_h_final = h;
	}
    }
    RecoverFromBackpointer(bp, best_h_final, h_sequence);
    return best_sequence_lprob;
}

double HMM::ViterbiExhaustive(const vector<X> &observation_sequence,
			      vector<H> *h_sequence) {
    size_t length = observation_sequence.size();

    // Generate all possible (length^num_states) state sequences.
    vector<vector<H> > all_state_sequences;
    vector<H> seed;
    PopulateAllStateSequences(seed, length, &all_state_sequences);

    double best_sequence_lprob = -numeric_limits<double>::infinity();
    size_t best_index = 0;
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double lprob = ComputeLikelihood(observation_sequence, all_state_sequences[i]);
	if (lprob >= best_sequence_lprob) {
	    best_sequence_lprob = lprob;
	    best_index = i;
	}
    }
    h_sequence->clear();
    for (size_t i = 0; i < length; ++i) {
	h_sequence->push_back(all_state_sequences[best_index][i]);
    }

    return best_sequence_lprob;
}

void HMM::PopulateAllStateSequences(const vector<H> &states, size_t length,
				    vector<vector<H> > *all_state_sequences) {
    if (states.size() == length) {
	all_state_sequences->push_back(states);
    } else {
	for (H h = 0; h < NumStates(); ++h) {
	    vector<H> states2 = states;
	    states2.push_back(h);
	    PopulateAllStateSequences(states2, length, all_state_sequences);
	}
    }
}

void HMM::RecoverFromBackpointer(const vector<vector<H> > &bp, H best_h_final,
				 vector<H> *h_sequence) {
    h_sequence->resize(bp.size());
    H current_best_h = best_h_final;
    (*h_sequence)[bp.size() - 1] = current_best_h;
    for (size_t i = bp.size() - 1; i > 0; --i) {
	current_best_h = bp.at(i)[current_best_h];
	(*h_sequence)[i - 1] = current_best_h;
    }
}

double HMM::ComputeLikelihood(const vector<X> &observation_sequence,
			      const vector<H> &h_sequence) {
    size_t length = observation_sequence.size();
    ASSERT(h_sequence.size() == length, "Different lengths");

    X x0 = observation_sequence[0];
    H h0 = h_sequence[0];
    double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	-log(NumObservations()) : o_[h0][x0];

    double lprob = pi_[h0] + lprob_emission_x0;
    for (size_t i = 1; i < length; ++i) {
	X xi = observation_sequence[i];
	H hi = h_sequence[i];
	double lprob_emission_xi = (xi == UnknownObservation()) ?
	    -log(NumObservations()) : o_[hi][xi];
	lprob += t_[h_sequence[i - 1]][hi] + lprob_emission_xi;
    }
    lprob += t_[h_sequence[length - 1]][StoppingState()];
    return lprob;
}

double HMM::ComputeLikelihoodForward(const vector<X> &observation_sequence) {
    vector<vector<double> > al;
    Forward(observation_sequence, &al);
    LogHandler log_handler;
    double sum_lprob = -numeric_limits<double>::infinity();
    for (H h = 0; h < NumStates(); ++h) {
	sum_lprob = log_handler.SumLogs(sum_lprob,
					al[observation_sequence.size() - 1][h] +
					t_[h][StoppingState()]);
    }
    return sum_lprob;
}

double HMM::ComputeLikelihoodBackward(const vector<X> &observation_sequence) {
    vector<vector<double> > be;
    Backward(observation_sequence, &be);
    LogHandler log_handler;
    double sum_lprob = -numeric_limits<double>::infinity();
    X x0 = observation_sequence.at(0);
    for (H h = 0; h < NumStates(); ++h) {
	double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	    -log(NumObservations()) : o_[h][x0];
	sum_lprob = log_handler.SumLogs(sum_lprob,
					pi_[h] + lprob_emission_x0 +
					be[0][h]);
    }
    return sum_lprob;
}

double HMM::ComputeLikelihoodExhaustive(const vector<X> &observation_sequence) {
    size_t length = observation_sequence.size();

    // Generate all possible (length^num_states) state sequences.
    vector<vector<H> > all_state_sequences;
    vector<H> seed;
    PopulateAllStateSequences(seed, length, &all_state_sequences);

    LogHandler log_handler;
    double sum_lprob = -numeric_limits<double>::infinity();
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double lprob = ComputeLikelihood(observation_sequence, all_state_sequences[i]);
	sum_lprob = log_handler.SumLogs(sum_lprob, lprob);
    }
    return sum_lprob;
}

void HMM::Forward(const vector<X> &observation_sequence, vector<vector<double> > *al) {
    size_t length = observation_sequence.size();

    // al[i][h] = log p(x(1)...x(i), h(i)=h)
    al->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*al)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base: al[0][h] = pi[h] + o[h][x(0)]
    X x0 = observation_sequence[0];
    for (H h = 0; h < NumStates(); ++h) {
	double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	    -log(NumObservations()) : o_[h][x0];
	(*al)[0][h] = pi_[h] + lprob_emission_x0;
    }

    // Main body: al[i][h] =
    //         logsum_{h_prev} al[i-1][h_prev] + t[h_prev][h] + o[h][x(i)]
    LogHandler log_handler;
    for (size_t i = 1; i < length; ++i) {
	X xi = observation_sequence[i];
	for (H h = 0; h < NumStates(); ++h) {
	    double lprob_emission_xi = (xi == UnknownObservation()) ?
		-log(NumObservations()) : o_[h][xi];
	    double sum_lprob = -numeric_limits<double>::infinity();
	    for (H h_prev = 0; h_prev < NumStates(); ++h_prev) {
		double particular_lprob =
		    (*al)[i - 1][h_prev] + t_[h_prev][h] + lprob_emission_xi;
		sum_lprob = log_handler.SumLogs(sum_lprob, particular_lprob);
	    }
	    (*al)[i][h] = sum_lprob;
	}
    }
}

void HMM::Backward(const vector<X> &observation_sequence, vector<vector<double> > *be) {
    size_t length = observation_sequence.size();

    // be[i][h] = log p(x(i+1)...x(N)|h(i)=h)
    be->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*be)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base: be[N][h] = log p(.|h(N)=h) = t_[h][<STOP>]
    for (H h = 0; h < NumStates(); ++h) {
	(*be)[length - 1][h] = t_[h][StoppingState()];
    }

    // Main body: be[i][h] =
    //       logsum_{h_next} t[h][h_next] + o[h_next][x(i+1)] + be[i+1][h_next]
    LogHandler log_handler;
    for (int i = length - 2; i >= 0; --i) {
	X observation_next = observation_sequence[i + 1];
	for (H h = 0; h < NumStates(); ++h) {
	    double sum_lprob = -numeric_limits<double>::infinity();
	    for (H h_next = 0; h_next < NumStates(); ++h_next) {
		double lprob_emission_next = (observation_next == UnknownObservation()) ?
		    -log(NumObservations()) : o_[h_next][observation_next];
		double particular_lprob =
		    t_[h][h_next] + lprob_emission_next + (*be)[i + 1][h_next];
		sum_lprob = log_handler.SumLogs(sum_lprob, particular_lprob);
	    }
	    (*be)[i][h] = sum_lprob;
	}
    }
}
