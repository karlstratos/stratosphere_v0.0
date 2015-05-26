// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "hmm.h"

#include <iomanip>
#include <limits>
#include <random>

#include "util.h"

void HMM::Clear() {
    x_dictionary_.clear();
    x_dictionary_inverse_.clear();
    h_dictionary_.clear();
    h_dictionary_inverse_.clear();
    o_.clear();
    t_.clear();
    pi_.clear();
}

void HMM::CreateRandomly(size_t num_observations, size_t num_states) {
    Clear();

    // Create the observation dictionary.
    for (X x = 0; x < num_observations; ++x) {
	string x_string = "x" + to_string(x);
	x_dictionary_[x_string] = x;
	x_dictionary_inverse_[x] = x_string;
    }

    // Create the state dictionary.
    for (H h = 0; h < num_states; ++h) {
	string h_string = "h" + to_string(h);
	h_dictionary_[h_string] = h;
	h_dictionary_inverse_[h] = h_string;
    }

    random_device device;
    default_random_engine engine(device());
    normal_distribution<double> normal(0.0, 1.0);  // Standard Gaussian.

    // Generate the emission parameters.
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

    // Generate the transition parameters.
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

    // Generate the prior parameters.
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
    for (const auto &x_pair : x_dictionary_) {
	string x_string = x_pair.first;
	X x = x_pair.second;
	util_file::binary_write_string(x_string, model_file);
	util_file::binary_write_primitive(x, model_file);
    }
    for (const auto &h_pair : h_dictionary_) {
	string h_string = h_pair.first;
	H h = h_pair.second;
	util_file::binary_write_string(h_string, model_file);
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
    for (size_t x_num = 0; x_num < num_observations; ++x_num) {
	string x_string;
	X x;
	util_file::binary_read_string(model_file, &x_string);
	util_file::binary_read_primitive(model_file, &x);
	x_dictionary_[x_string] = x;
	x_dictionary_inverse_[x] = x_string;
    }
    for (size_t h_num = 0; h_num < num_states; ++h_num) {
	string h_string;
	H h;
	util_file::binary_read_string(model_file, &h_string);
	util_file::binary_read_primitive(model_file, &h);
	h_dictionary_[h_string] = h;
	h_dictionary_inverse_[h] = h_string;
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
    vector<vector<pair<string, string> > > labeled_sequences;
    vector<pair<string, string> > labeled_sequence;
    ifstream labeled_data_file(labeled_data_path, ios::in);
    while (labeled_data_file.good()) {
	vector<string> tokens;
	util_string::read_line(labeled_data_file, &tokens);
	if (tokens.size() > 0) {
	    ASSERT(tokens.size() == 2, "Invalid format");
	    labeled_sequence.emplace_back(tokens[0], tokens[1]);
	} else {
	    if (labeled_sequence.size() > 0) {  // End of a sequence.
		labeled_sequences.push_back(labeled_sequence);
		labeled_sequence.clear();
	    }
	}
    }
    TrainSupervised(labeled_sequences);
}

void HMM::TrainSupervised(const vector<vector<pair<string, string> > >
			  &labeled_sequences) {
    Clear();
    unordered_map<H, unordered_map<X, size_t> > count_h_x;
    unordered_map<H, unordered_map<H, size_t> > count_h1_h2;
    unordered_map<H, size_t> count_h_initial;
    for (size_t i = 0; i < labeled_sequences.size(); ++i) {
        ASSERT(labeled_sequences[i].size() > 0, "Empty training sequence");
	H h_initial = AddStateIfUnknown(labeled_sequences[i][0].second);
	++count_h_initial[h_initial];
	size_t length = labeled_sequences[i].size();
	for (size_t j = 0; j < length; ++j) {
	    X x = AddObservationIfUnknown(labeled_sequences[i][j].first);
	    H h2 = AddStateIfUnknown(labeled_sequences[i][j].second);
	    ++count_h_x[h2][x];
	    if (j > 0) {
		H h1 = h_dictionary_[labeled_sequences[i][j - 1].second];
		++count_h1_h2[h1][h2];
	    }
	}
	H h_final = h_dictionary_[labeled_sequences[i][length - 1].second];
	++count_h1_h2[h_final][StoppingState()];
    }
    o_.resize(NumStates());
    for (const auto &h_pair : count_h_x) {
	size_t sum = 0;
	for (const auto &x_pair : h_pair.second) { sum += x_pair.second; }
	o_[h_pair.first].resize(NumObservations(),
				-numeric_limits<double>::infinity());
	for (const auto &x_pair : h_pair.second) {
	    o_[h_pair.first][x_pair.first] = log(x_pair.second) - log(sum);
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
    vector<vector<string> > x_sequences;
    vector<vector<string> > h_sequences;
    vector<string> x_sequence;
    vector<string> h_sequence;
    bool is_labeled = true;  // Is the test data labeled?
    ifstream data_file(data_path, ios::in);
    while (data_file.good()) {
	vector<string> tokens;
	util_string::read_line(data_file, &tokens);
	if (tokens.size() > 0) {
	    x_sequence.push_back(tokens[0]);
	    string h_string;
	    if (tokens.size() == 1) {  // the
		is_labeled = false;
	    } else if (tokens.size() == 2) {  // the DET
		h_string = tokens[1];
	    } else {
		ASSERT(false, "Invalid format");
	    }
	    h_sequence.push_back(h_string);
	} else {
	    if (x_sequence.size() > 0) {  // End of a sequence.
		x_sequences.push_back(x_sequence);
		h_sequences.push_back(h_sequence);
		x_sequence.clear();
		h_sequence.clear();
	    }
	}
    }
    vector<vector<string> > pred_sequences;
    for (size_t i = 0; i < x_sequences.size(); ++i) {
	vector<string> pred_sequence;
	if (decoding_method_ == "viterbi") {
	    Viterbi(x_sequences[i], &pred_sequence);
	} else if (decoding_method_ == "mbr") {
	    MinimumBayesRisk(x_sequences[i], &pred_sequence);
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
    if (is_labeled && verbose_) {
	cerr << setprecision(4);
	cerr << "Position: " << position_acc << "     \tSequence: "
	     << sequence_acc << "     \tManyToOne: " << many2one_acc << endl;
    }

    bool report_many2one = (position_acc == 0.0 && many2one_acc > 0.0);
    if (!prediction_path.empty()) {
	ofstream prediction_file(prediction_path, ios::out);
	for (size_t i = 0; i < x_sequences.size(); ++i) {
	    for (size_t j = 0; j < x_sequences[i].size(); ++j) {
		string x_string = x_sequences[i][j];
		string h_string = h_sequences[i][j];
		string pred_string = pred_sequences[i][j];
		if (report_many2one) {
		    pred_string = many2one_map[pred_string];
		}
		prediction_file << x_string << " " << h_string << " "
				<< pred_string << endl;
	    }
	    if (i < x_sequences.size() - 1) { prediction_file << endl; }
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
    vector<X> x_sequence;
    ConvertObservation(string_sequence, &x_sequence);
    vector<H> h_sequence;
    double lprob = Viterbi(x_sequence, &h_sequence);
    ConvertState(h_sequence, state_sequence);
    return lprob;
}

double HMM::ViterbiExhaustive(const vector<string> &string_sequence,
			      vector<string> *state_sequence) {
    vector<X> x_sequence;
    ConvertObservation(string_sequence, &x_sequence);
    vector<H> h_sequence;
    double lprob = ViterbiExhaustive(x_sequence, &h_sequence);
    ConvertState(h_sequence, state_sequence);
    return lprob;
}

void HMM::MinimumBayesRisk(const vector<string> &string_sequence,
			   vector<string> *state_sequence) {
    vector<X> x_sequence;
    ConvertObservation(string_sequence, &x_sequence);
    vector<vector<double> > al;
    Forward(x_sequence, &al);
    vector<vector<double> > be;
    Backward(x_sequence, &be);
    vector<H> h_sequence;
    for (size_t i = 0; i < x_sequence.size(); ++i) {
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
    vector<X> x_sequence;
    ConvertObservation(string_sequence, &x_sequence);
    return ComputeLikelihoodForward(x_sequence);
}

double HMM::ComputeObservationLikelihoodBackward(const vector<string>
						 &string_sequence) {
    vector<X> x_sequence;
    ConvertObservation(string_sequence, &x_sequence);
    return ComputeLikelihoodBackward(x_sequence);
}

double HMM::ComputeObservationLikelihoodExhaustive(const vector<string>
						   &string_sequence) {
    vector<X> x_sequence;
    ConvertObservation(string_sequence, &x_sequence);
    return ComputeLikelihoodExhaustive(x_sequence);
}

X HMM::x_dictionary(const string &x_string) {
    ASSERT(x_dictionary_.find(x_string) != x_dictionary_.end(),
	   "Requesting integer ID of an unknown x string: " << x_string);
    return x_dictionary_[x_string];
}

string HMM::x_dictionary_inverse(X x) {
    if (x == UnknownObservation()) { return "<UNK>"; }
    ASSERT(x_dictionary_inverse_.find(x) != x_dictionary_inverse_.end(),
	   "Requesting string of an unknown x integer: " << x);
    return x_dictionary_inverse_[x];
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

X HMM::AddObservationIfUnknown(const string &x_string) {
    ASSERT(!x_string.empty(), "Adding an empty observation string!");
    if (x_dictionary_.find(x_string) == x_dictionary_.end()) {
	X x = x_dictionary_.size();
	x_dictionary_[x_string] = x;
	x_dictionary_inverse_[x] = x_string;
    }
    return x_dictionary_[x_string];
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
			     vector<X> *x_sequence) {
    ASSERT(x_dictionary_.size() > 0, "No observation dictionary");
    x_sequence->clear();
    for (size_t i = 0; i < string_sequence.size(); ++i) {
	string x_string = string_sequence[i];
	X x = (x_dictionary_.find(x_string) != x_dictionary_.end()) ?
	    x_dictionary_[x_string] : UnknownObservation();
	x_sequence->push_back(x);
    }
}

void HMM::ConvertObservation(const vector<X> &x_sequence,
			     vector<string> *string_sequence) {
    ASSERT(x_dictionary_inverse_.size() > 0, "No observation dictionary");
    string_sequence->clear();
    for (size_t i = 0; i < x_sequence.size(); ++i) {
	string_sequence->push_back(x_dictionary_inverse(x_sequence[i]));
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

double HMM::Viterbi(const vector<X> &x_sequence, vector<H> *h_sequence) {
    size_t length = x_sequence.size();

    // chart[i][h] = highest prob of a sequence ending in state h at position i
    vector<vector<double> > chart(length);
    vector<vector<H> > bp(length);  // bp[i][h] = h_prev (backpointer)
    for (size_t i = 0; i < length; ++i) {
	chart[i].resize(NumStates(), -numeric_limits<double>::infinity());
	bp[i].resize(NumStates());
    }

    // Base: chart[0][h] = pi[h] + o[h][x(0)]
    X x0 = x_sequence[0];
    for (H h = 0; h < NumStates(); ++h) {
	double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	    -log(NumObservations()) : o_[h][x0];
	chart[0][h] = pi_[h] + lprob_emission_x0;
    }

    // Main body: chart[i][h] =
    //         max_{h_prev} chart[i-1][h_prev] + t[h_prev][h] + o[h][x(i)]
    for (size_t i = 1; i < length; ++i) {
	X xi = x_sequence[i];
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

double HMM::ViterbiExhaustive(const vector<X> &x_sequence,
			      vector<H> *h_sequence) {
    size_t length = x_sequence.size();

    // Generate all possible (length^num_states) state sequences.
    vector<vector<H> > all_state_sequences;
    vector<H> seed;
    PopulateAllStateSequences(seed, length, &all_state_sequences);

    double best_sequence_lprob = -numeric_limits<double>::infinity();
    size_t best_index = 0;
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double lprob = ComputeLikelihood(x_sequence, all_state_sequences[i]);
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

double HMM::ComputeLikelihood(const vector<X> &x_sequence,
			      const vector<H> &h_sequence) {
    size_t length = x_sequence.size();
    ASSERT(h_sequence.size() == length, "Different lengths");

    X x0 = x_sequence[0];
    H h0 = h_sequence[0];
    double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	-log(NumObservations()) : o_[h0][x0];

    double lprob = pi_[h0] + lprob_emission_x0;
    for (size_t i = 1; i < length; ++i) {
	X xi = x_sequence[i];
	H hi = h_sequence[i];
	double lprob_emission_xi = (xi == UnknownObservation()) ?
	    -log(NumObservations()) : o_[hi][xi];
	lprob += t_[h_sequence[i - 1]][hi] + lprob_emission_xi;
    }
    lprob += t_[h_sequence[length - 1]][StoppingState()];
    return lprob;
}

double HMM::ComputeLikelihoodForward(const vector<X> &x_sequence) {
    vector<vector<double> > al;
    Forward(x_sequence, &al);
    LogHandler log_handler;
    double sum_lprob = -numeric_limits<double>::infinity();
    for (H h = 0; h < NumStates(); ++h) {
	sum_lprob = log_handler.SumLogs(sum_lprob,
					al[x_sequence.size() - 1][h] +
					t_[h][StoppingState()]);
    }
    return sum_lprob;
}

double HMM::ComputeLikelihoodBackward(const vector<X> &x_sequence) {
    vector<vector<double> > be;
    Backward(x_sequence, &be);
    LogHandler log_handler;
    double sum_lprob = -numeric_limits<double>::infinity();
    X x0 = x_sequence.at(0);
    for (H h = 0; h < NumStates(); ++h) {
	double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	    -log(NumObservations()) : o_[h][x0];
	sum_lprob = log_handler.SumLogs(sum_lprob,
					pi_[h] + lprob_emission_x0 +
					be[0][h]);
    }
    return sum_lprob;
}

double HMM::ComputeLikelihoodExhaustive(const vector<X> &x_sequence) {
    size_t length = x_sequence.size();

    // Generate all possible (length^num_states) state sequences.
    vector<vector<H> > all_state_sequences;
    vector<H> seed;
    PopulateAllStateSequences(seed, length, &all_state_sequences);

    LogHandler log_handler;
    double sum_lprob = -numeric_limits<double>::infinity();
    for (size_t i = 0; i < all_state_sequences.size(); ++i) {
	double lprob = ComputeLikelihood(x_sequence, all_state_sequences[i]);
	sum_lprob = log_handler.SumLogs(sum_lprob, lprob);
    }
    return sum_lprob;
}

void HMM::Forward(const vector<X> &x_sequence, vector<vector<double> > *al) {
    size_t length = x_sequence.size();

    // al[i][h] = log p(x(1)...x(i), h(i)=h)
    al->resize(length);
    for (size_t i = 0; i < length; ++i) {
	(*al)[i].resize(NumStates(), -numeric_limits<double>::infinity());
    }

    // Base: al[0][h] = pi[h] + o[h][x(0)]
    X x0 = x_sequence[0];
    for (H h = 0; h < NumStates(); ++h) {
	double lprob_emission_x0 = (x0 == UnknownObservation()) ?
	    -log(NumObservations()) : o_[h][x0];
	(*al)[0][h] = pi_[h] + lprob_emission_x0;
    }

    // Main body: al[i][h] =
    //         logsum_{h_prev} al[i-1][h_prev] + t[h_prev][h] + o[h][x(i)]
    LogHandler log_handler;
    for (size_t i = 1; i < length; ++i) {
	X xi = x_sequence[i];
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

void HMM::Backward(const vector<X> &x_sequence, vector<vector<double> > *be) {
    size_t length = x_sequence.size();

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
	X x_next = x_sequence[i + 1];
	for (H h = 0; h < NumStates(); ++h) {
	    double sum_lprob = -numeric_limits<double>::infinity();
	    for (H h_next = 0; h_next < NumStates(); ++h_next) {
		double lprob_emission_next = (x_next == UnknownObservation()) ?
		    -log(NumObservations()) : o_[h_next][x_next];
		double particular_lprob =
		    t_[h][h_next] + lprob_emission_next + (*be)[i + 1][h_next];
		sum_lprob = log_handler.SumLogs(sum_lprob, particular_lprob);
	    }
	    (*be)[i][h] = sum_lprob;
	}
    }
}
