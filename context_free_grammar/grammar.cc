// Author: Karl Stratos (me@karlstratos.com)

#include "grammar.h"

#include <fstream>
#include <iomanip>
#include <math.h>
#include <stack>

void Grammar::Clear() {
    nonterminal_dictionary_.clear();
    nonterminal_dictionary_inverse_.clear();
    terminal_dictionary_.clear();
    terminal_dictionary_inverse_.clear();
    interminal_.clear();
    preterminal_.clear();
    lprob_binary_.clear();
    lprob_unary_.clear();
    lprob_root_.clear();
    binary_rhs_.clear();
    left_parent_sibling_.clear();
    right_parent_sibling_.clear();
}

void Grammar::Train(const string &treebank_path) {
    TreeSet trees(treebank_path);
    Train(&trees);
}

void Grammar::Train(TreeSet *trees) {
    size_t num_interminal_types_original;
    size_t num_preterminal_types_original;
    size_t num_terminal_types_original;
    trees->NumSymbolTypes(&num_interminal_types_original,
			  &num_preterminal_types_original,
			  &num_terminal_types_original);

    if (verbose_) {
	cerr << "[CNF transform]" << endl;
	cerr << "   " << binarization_method_ << " binarization" << endl;
	cerr << "   Vertical Markovization order: "
	     << vertical_markovization_order_ << endl;
	cerr << "   Horizontal Markovization order: "
	     << horizontal_markovization_order_ << endl;
	trees->ProcessToChomskyNormalForm(binarization_method_,
					  vertical_markovization_order_,
					  horizontal_markovization_order_);
    }
    EstimatePCFG(trees);

    if (verbose_) {
	cerr << endl << "[Treebank statistics]" << endl;
	cerr << "   " << trees->NumTrees() << " trees" << endl;
	cerr << "   " << NumInterminalTypes() << " interminal types"
	     << " (original " << num_interminal_types_original << ")" << endl;
	cerr << "   " << NumPreterminalTypes() << " preterminal types"
	     << " (original " << num_preterminal_types_original << ")" << endl;
	cerr << "   " << NumTerminalTypes() << " terminal types" << endl;
	cerr << "   " << NumBinaryRuleTypes() << " binary rule types" << endl;
	cerr << "   " << NumUnaryRuleTypes() << " unary rule types" << endl;
	cerr << "   " << NumRootNonterminalTypes() << " root nonterminal types"
	     << endl << flush;
    }
}

void Grammar::Evaluate(const string &treebank_path,
		       const string prediction_path) {
    TreeSet trees(treebank_path);
    TreeSet *predicted_trees = Parse(&trees);
    if (!prediction_path.empty()) {
	predicted_trees->Write(prediction_path);
    }
    delete predicted_trees;
}

TreeSet *Grammar::Parse(TreeSet *trees) {
    time_t begin_time = time(NULL);
    size_t num_parsed = 0;
    size_t sum_lengths = 0;

    // The same CNF transformation used in training is necessary if we decide to
    // use gold tags (e.g., TOP+NP+NNP).
    trees->ProcessToChomskyNormalForm(binarization_method_,
				      vertical_markovization_order_,
				      horizontal_markovization_order_);

    TreeSet *predicted_trees = new TreeSet();
    TerminalSequences terminal_sequences(trees);
    for (size_t i = 0; i < terminal_sequences.NumSequences(); ++i) {
	TerminalSequence *terminal_sequence = terminal_sequences.Sequence(i);
	AssignNumericIdentities(terminal_sequence);
	Node *predicted_tree = Parse(terminal_sequence);
	if (!predicted_tree->IsEmpty()) {
	    ++num_parsed;
	    sum_lengths += terminal_sequence->Length();
	}
	AssignStringIdentities(predicted_tree);
	predicted_trees->AddTree(predicted_tree);
    }
    predicted_trees->RecoverFromChomskyNormalForm();

    if (verbose_) {
	double parsing_time = difftime(time(NULL), begin_time);
	double avg_time = parsing_time / num_parsed;
	double avg_length = ((double) sum_lengths) / num_parsed;
	size_t num_skipped = terminal_sequences.NumSequences() - num_parsed;
	cerr << fixed << setprecision(2);

	cerr << "Parsing time: "
	     << util_string::convert_seconds_to_string(parsing_time)
	     << " (" << avg_time << " SPS, " << avg_length << " AL, "
	     << num_parsed << " parsed, " << num_skipped << " skipped)" << endl;

	double precision, recall, f1_score, tagging_accuracy;
	predicted_trees->EvaluateAgainstGold(
	    trees, &num_skipped, &precision, &recall, &f1_score,
	    &tagging_accuracy);
	cerr << "F1: " << f1_score << "   ";
	cerr << "P: " << precision << "   ";
	cerr << "R: " << recall << "   ";
	cerr << "T: " << tagging_accuracy << "   ";
	cerr << "SK: " << num_skipped << endl;
    }

    return predicted_trees;
}

Node *Grammar::Parse(const vector<string> &terminal_strings) {
    TerminalSequence terminal_sequence(terminal_strings);
    AssignNumericIdentities(&terminal_sequence);
    Node *predicted_tree = Parse(&terminal_sequence);
    AssignStringIdentities(predicted_tree);
    return predicted_tree;
}

void Grammar::Save(const string &model_path) {
    ofstream model_file(model_path, ios::out | ios::binary);
    util_file::binary_write_string(binarization_method_, model_file);
    util_file::binary_write_primitive(vertical_markovization_order_,
				      model_file);
    util_file::binary_write_primitive(horizontal_markovization_order_,
				      model_file);
    size_t num_interminal_types = NumInterminalTypes();
    size_t num_preterminal_types = NumPreterminalTypes();
    size_t num_terminal_types = NumTerminalTypes();
    size_t num_binary_rule_types = NumBinaryRuleTypes();
    size_t num_unary_rule_types = NumUnaryRuleTypes();
    size_t num_root_nonterminal_types = NumRootNonterminalTypes();
    util_file::binary_write_primitive(num_interminal_types, model_file);
    util_file::binary_write_primitive(num_preterminal_types, model_file);
    util_file::binary_write_primitive(num_terminal_types, model_file);
    util_file::binary_write_primitive(num_binary_rule_types, model_file);
    util_file::binary_write_primitive(num_unary_rule_types, model_file);
    util_file::binary_write_primitive(num_root_nonterminal_types, model_file);

    for (Nonterminal a = 0; a < NumNonterminalTypes(); ++a) {
	string a_string = nonterminal_dictionary_inverse_[a];
	util_file::binary_write_string(a_string, model_file);
	util_file::binary_write_primitive(a, model_file);
    }

    for (Nonterminal x = 0; x < NumTerminalTypes(); ++x) {
	string x_string = terminal_dictionary_inverse_[x];
	util_file::binary_write_string(x_string, model_file);
	util_file::binary_write_primitive(x, model_file);
    }

    for (const auto &interminal_pair : interminal_) {
	Nonterminal interminal = interminal_pair.first;
	util_file::binary_write_primitive(interminal, model_file);
    }

    for (const auto &a_pair : lprob_binary_) {
	Nonterminal a = a_pair.first;
	for (const auto &b_pair : lprob_binary_[a]) {
	    Nonterminal b = b_pair.first;
	    for (const auto &c_pair : lprob_binary_[a][b]) {
		Nonterminal c = c_pair.first;
		double lprob_abc = c_pair.second;
		util_file::binary_write_primitive(a, model_file);
		util_file::binary_write_primitive(b, model_file);
		util_file::binary_write_primitive(c, model_file);
		util_file::binary_write_primitive(lprob_abc, model_file);
	    }
	}
    }

    for (const auto &a_pair : lprob_unary_) {
	Nonterminal a = a_pair.first;
	for (const auto &x_pair : lprob_unary_[a]) {
	    Terminal x = x_pair.first;
	    double lprob_ax = x_pair.second;
	    util_file::binary_write_primitive(a, model_file);
	    util_file::binary_write_primitive(x, model_file);
	    util_file::binary_write_primitive(lprob_ax, model_file);
	}
    }

    for (const auto &a_pair : lprob_root_) {
	Nonterminal a = a_pair.first;
	double lprob_root = a_pair.second;
	util_file::binary_write_primitive(a, model_file);
	util_file::binary_write_primitive(lprob_root, model_file);
    }

    model_file.close();
}

void Grammar::Load(const string &model_path) {
    Clear();
    ifstream model_file(model_path, ios::in | ios::binary);
    ASSERT(model_file.is_open(), "Can't load the model: " << model_path);

    util_file::binary_read_string(model_file, &binarization_method_);
    util_file::binary_read_primitive(model_file,
				     &vertical_markovization_order_);
    util_file::binary_read_primitive(model_file,
				     &horizontal_markovization_order_);
    size_t num_interminal_types;
    size_t num_preterminal_types;
    size_t num_terminal_types;
    size_t num_binary_rule_types;
    size_t num_unary_rule_types;
    size_t num_root_nonterminal_types;
    util_file::binary_read_primitive(model_file, &num_interminal_types);
    util_file::binary_read_primitive(model_file, &num_preterminal_types);
    util_file::binary_read_primitive(model_file, &num_terminal_types);
    size_t num_nonterminal_types = num_interminal_types + num_preterminal_types;
    util_file::binary_read_primitive(model_file, &num_binary_rule_types);
    util_file::binary_read_primitive(model_file, &num_unary_rule_types);
    util_file::binary_read_primitive(model_file, &num_root_nonterminal_types);

    for (size_t i = 0; i < num_nonterminal_types; ++i) {
	string a_string;
	Nonterminal a;
	util_file::binary_read_string(model_file, &a_string);
	util_file::binary_read_primitive(model_file, &a);
	nonterminal_dictionary_inverse_[a] = a_string;
	nonterminal_dictionary_[a_string] = a;
    }

    for (size_t i = 0; i < num_terminal_types; ++i) {
	string x_string;
	Terminal x;
	util_file::binary_read_string(model_file, &x_string);
	util_file::binary_read_primitive(model_file, &x);
	terminal_dictionary_inverse_[x] = x_string;
	terminal_dictionary_[x_string] = x;
    }

    for (size_t i = 0; i < num_interminal_types; ++i) {
	Nonterminal interminal;
	util_file::binary_read_primitive(model_file, &interminal);
	interminal_[interminal] = true;
    }

    for (Nonterminal a = 0; a < num_nonterminal_types; ++a) {
	if (interminal_.find(a) == interminal_.end()) {
	    preterminal_[a] = true;
	}
    }

    for (size_t i = 0; i < num_binary_rule_types; ++i) {
	Nonterminal a;
	Nonterminal b;
	Nonterminal c;
	double lprob_abc;
	util_file::binary_read_primitive(model_file, &a);
	util_file::binary_read_primitive(model_file, &b);
	util_file::binary_read_primitive(model_file, &c);
	util_file::binary_read_primitive(model_file, &lprob_abc);
	lprob_binary_[a][b][c] = lprob_abc;
	binary_rhs_[a].push_back(make_tuple(b, c, lprob_abc));
	left_parent_sibling_[b].push_back(make_tuple(a, c, lprob_abc));
	right_parent_sibling_[c].push_back(make_tuple(a, b, lprob_abc));
    }

    for (size_t i = 0; i < num_unary_rule_types; ++i) {
	Nonterminal a;
	Nonterminal x;
	double lprob_ax;
	util_file::binary_read_primitive(model_file, &a);
	util_file::binary_read_primitive(model_file, &x);
	util_file::binary_read_primitive(model_file, &lprob_ax);
	lprob_unary_[a][x] = lprob_ax;
    }

    for (size_t i = 0; i < num_root_nonterminal_types; ++i) {
	Nonterminal a;
	double lprob_root;
	util_file::binary_read_primitive(model_file, &a);
	util_file::binary_read_primitive(model_file, &lprob_root);
	lprob_root_[a] = lprob_root;
    }

    model_file.close();
    AssertProperDistributions();
}

double Grammar::ComputePCFGTreeProbability(Node *tree) {
    if (tree->IsEmpty()) { return -numeric_limits<double>::infinity(); }
    try {
	AssignNumericIdentities(tree);
	Nonterminal root = tree->nonterminal_number();
	if (lprob_root_.find(root) == lprob_root_.end()) {
	    // Return probability 0 if the root is unknown.
	    return -numeric_limits<double>::infinity();
	}
	double lprob_tree = lprob_root_[root];
	stack<Node *> dfs_stack;  // Depth-first search (DFS)
	dfs_stack.push(tree);
	while (!dfs_stack.empty()) {
	    Node *node = dfs_stack.top();
	    dfs_stack.pop();
	    Nonterminal a = node->nonterminal_number();
	    if (node->NumChildren() == 2) {
		Nonterminal b = node->Child(0)->nonterminal_number();
		Nonterminal c = node->Child(1)->nonterminal_number();
		if (lprob_binary_.find(a) == lprob_binary_.end() ||
		    lprob_binary_[a].find(b) == lprob_binary_[a].end() ||
		    lprob_binary_[a][b].find(c) == lprob_binary_[a][b].end()) {
		    // Return probability 0 if a binary rule is unknown.
		    return -numeric_limits<double>::infinity();
		}
		lprob_tree += lprob_binary_[a][b][c];
	    } else if (node->NumChildren() == 0) {
		Terminal x = node->terminal_number();
		if (lprob_unary_.find(a) == lprob_unary_.end() ||
		    lprob_unary_[a].find(x) == lprob_unary_[a].end()) {
		    // Return probability 0 if a unary rule is unknown.
		    return -numeric_limits<double>::infinity();
		}
		lprob_tree += lprob_unary_[a][x];
	    } else {
		ASSERT(false, "Computing probability of a tree not in CNF!");
	    }

	    for (size_t i = 0; i < node->NumChildren(); ++i) {
		dfs_stack.push(node->Child(i));
	    }
	}
	return lprob_tree;
    } catch (int e) {
	// Likely in AssignNumericIdentities(tree) because the tree has unknown
	// symbols - return probability 0.
	return -numeric_limits<double>::infinity();
    }
}

void Grammar::ComputeMarginalsPCFG(const vector<string> &terminal_strings,
				   Chart *marginal) {
    TerminalSequence terminal_sequence(terminal_strings);
    AssignNumericIdentities(&terminal_sequence);
    ComputeMarginalsPCFG(&terminal_sequence, marginal);
}

size_t Grammar::NumBinaryRuleTypes() {
    size_t num_binary_rule_types = 0;
    for (const auto & a_pair : lprob_binary_) {
	for (const auto & b_pair : a_pair.second) {
	    num_binary_rule_types += b_pair.second.size();
	}
    }
    return num_binary_rule_types;
}

size_t Grammar::NumUnaryRuleTypes() {
    size_t num_unary_rule_types = 0;
    for (const auto & a_pair : lprob_unary_) {
	num_unary_rule_types += a_pair.second.size();
    }
    return num_unary_rule_types;
}

Node *Grammar::Parse(TerminalSequence *terminals) {
    if (terminals->Length() > max_sentence_length_) { return new Node("", ""); }
    if (decoding_method_ == "viterbi") {
	return ParsePCFGCKY(terminals);
    } else if (decoding_method_ == "marginal") {
	return ParsePCFGMarginal(terminals);
    } else {
	ASSERT(false, "Unknown decoding method: " << decoding_method_);
    }
}

Node *Grammar::ParsePCFGCKY(TerminalSequence *terminals) {
    size_t sequence_length = terminals->Length();

    // chart[i][j][a] = highest probability of any tree rooted at a spanning the
    // sequence from position i to position j.
    Chart chart(sequence_length);

    // bp[i][j][a] = (b, c, k) corresponding to chart[i][j][a].
    Backpointer bp(sequence_length);
    for (size_t i = 0; i < sequence_length; ++i) {
	chart[i].resize(sequence_length);
	bp[i].resize(sequence_length);
	for (size_t j = 0; j < sequence_length; ++j) {
	    chart[i][j].resize(NumNonterminalTypes(),
			       -numeric_limits<double>::infinity());
	    bp[i][j].resize(NumNonterminalTypes());
	}
    }

    // Base: chart[i][i][a] = p(a -> x_i | a).
    for (size_t i = 0; i < sequence_length; ++i) {
	Terminal x = terminals->TerminalNumber(i);
	if (use_gold_tags_ && terminals->PreterminalNumber(i) >= 0) {
	    // If we are using the gold tag and the gold tag is known to the
	    // grammar, softly enforce this gold tag above others.
	    for (const auto &preterminal_pair : preterminal_) {
		Nonterminal a = preterminal_pair.first;
		chart[i][i][a] = (a == terminals->PreterminalNumber(i)) ?
		    0.0 : -log(NumTerminalTypes());
	    }
	} else {
	    // Otherwise consider all possible tags, while softly enforcing
	    // p(a -> x|a) > 0 estimated from the training data.
	    for (const auto &preterminal_pair : preterminal_) {
		Nonterminal a = preterminal_pair.first;
		if (lprob_unary_[a].find(x) != lprob_unary_[a].end()) {
		    chart[i][i][a] = lprob_unary_[a][x];
		} else {
		    chart[i][i][a] = -log(NumTerminalTypes());
		}
	    }
	}
    }

    // Main body.
    for (size_t span_length = 1; span_length < sequence_length; ++span_length) {
	for (size_t i = 0; i < sequence_length - span_length; ++i) {
	    size_t j = i + span_length;
	    for (const auto &interminal_pair : interminal_) {
		Nonterminal a = interminal_pair.first;
		double max_lprob = -numeric_limits<double>::infinity();
		Nonterminal best_b = -1;
		Nonterminal best_c = -1;
		size_t best_split_point = -1;

		vector<tuple<Nonterminal, Nonterminal, double> > *bcs =
		    &binary_rhs_[a];
		for (size_t idx = 0; idx < bcs->size(); ++idx) {
		    tuple<Nonterminal, Nonterminal, double> *bc_tuple =
			&bcs->at(idx);
		    Nonterminal b = get<0>(*bc_tuple);
		    Nonterminal c = get<1>(*bc_tuple);
		    double lprob_abc = get<2>(*bc_tuple);
		    for (size_t split_point = i; split_point < j;
			 ++split_point) {
			double particular_lprob = lprob_abc +
			    chart[i][split_point][b] +
			    chart[split_point+1][j][c];
			if (particular_lprob > max_lprob) {
			    max_lprob = particular_lprob;
			    best_b = b;
			    best_c = c;
			    best_split_point = split_point;
			}
		    }
		}
		chart[i][j][a] = max_lprob;
		bp[i][j][a] = make_tuple(best_b, best_c, best_split_point);
	    }
	}
    }

    double best_tree_lprob = -numeric_limits<double>::infinity();
    Nonterminal best_root = -1;
    for (const auto &root_pair : lprob_root_) {
	Nonterminal a = root_pair.first;
	double root_prior = root_pair.second;
	double tree_lprob = chart[0][sequence_length - 1][a] + root_prior;
	if (tree_lprob > best_tree_lprob) {
	    best_tree_lprob = tree_lprob;
	    best_root = a;
	}
    }

    ASSERT(best_root >= 0, "Parse failure: " << terminals->ToString());
    Node *best_tree =
	RecoverFromBackpointer(bp, terminals,
			       0, sequence_length - 1, best_root);
    return best_tree;
}

Node *Grammar::ParsePCFGMarginal(TerminalSequence *terminals) {
    Chart marginal;
    ComputeMarginalsPCFG(terminals, &marginal);
    Node *tree = RecoverMaxMarginalTree(terminals, marginal);
    return tree;
}

void Grammar::ComputeMarginalsPCFG(TerminalSequence *terminals,
				   Chart *marginal) {
    Chart inside;
    ComputeInsideProbabilitiesPCFG(terminals, &inside);

    Chart outside;
    size_t sequence_length = terminals->Length();
    ComputeOutsideProbabilitiesPCFG(sequence_length, inside, &outside);

    (*marginal).clear();
    (*marginal).resize(sequence_length);
    for (size_t i = 0; i < sequence_length; ++i) {
	(*marginal)[i].resize(sequence_length);
	for (size_t j = 0; j < sequence_length; ++j) {
	    (*marginal)[i][j].resize(NumNonterminalTypes(),
				     -numeric_limits<double>::infinity());
	    for (Nonterminal a = 0; a < NumNonterminalTypes(); ++a) {
		(*marginal)[i][j][a] = inside[i][j][a] + outside[i][j][a];
	    }
	}
    }
}

void Grammar::ComputeInsideProbabilitiesPCFG(TerminalSequence *terminals,
					     Chart *inside) {
    size_t sequence_length = terminals->Length();
    inside->clear();
    inside->resize(sequence_length);
    for (size_t i = 0; i < sequence_length; ++i) {
	(*inside)[i].resize(sequence_length);
	for (size_t j = 0; j < sequence_length; ++j) {
	    (*inside)[i][j].resize(NumNonterminalTypes(),
				   -numeric_limits<double>::infinity());
	}
    }

    // Base: inside[i][i][a] = p(a -> x_i | a).
    for (size_t i = 0; i < sequence_length; ++i) {
	Terminal x = terminals->TerminalNumber(i);
	if (use_gold_tags_ && terminals->PreterminalNumber(i) >= 0) {
	    // If we are using the gold tag and the gold tag is known to the
	    // grammar, softly enforce this gold tag above others.
	    for (const auto &preterminal_pair : preterminal_) {
		Nonterminal a = preterminal_pair.first;
		(*inside)[i][i][a] = (a == terminals->PreterminalNumber(i)) ?
		    0.0 : -log(NumTerminalTypes());
	    }
	} else {
	    // Otherwise consider all possible tags, while softly enforcing
	    // p(a -> x|a) > 0 estimated from the training data.
	    for (const auto &preterminal_pair : preterminal_) {
		Nonterminal a = preterminal_pair.first;
		if (lprob_unary_[a].find(x) != lprob_unary_[a].end()) {
		    (*inside)[i][i][a] = lprob_unary_[a][x];
		} else {
		    (*inside)[i][i][a] = -log(NumTerminalTypes());
		}
	    }
	}
    }

    // Main body.
    for (size_t span_length = 1; span_length < sequence_length; ++span_length) {
	for (size_t i = 0; i < sequence_length - span_length; ++i) {
	    size_t j = i + span_length;
	    for (const auto &interminal_pair : interminal_) {
		Nonterminal a = interminal_pair.first;
		double inside_lprob = -numeric_limits<double>::infinity();

		vector<tuple<Nonterminal, Nonterminal, double> > *bcs =
		    &binary_rhs_[a];
		for (size_t idx = 0; idx < bcs->size(); ++idx) {
		    tuple<Nonterminal, Nonterminal, double> *bc_tuple =
			&bcs->at(idx);
		    Nonterminal b = get<0>(*bc_tuple);
		    Nonterminal c = get<1>(*bc_tuple);
		    double lprob_abc = get<2>(*bc_tuple);
		    for (size_t split_point = i; split_point < j;
			 ++split_point) {
			double particular_lprob = lprob_abc +
			    (*inside)[i][split_point][b] +
			    (*inside)[split_point+1][j][c];
			inside_lprob = util_math::sum_logs(inside_lprob,
							   particular_lprob);
		    }
		}
		(*inside)[i][j][a] = inside_lprob;
	    }
	}
    }
}

void Grammar::ComputeOutsideProbabilitiesPCFG(size_t sequence_length,
					      const Chart &inside,
					      Chart *outside) {
    outside->clear();
    outside->resize(sequence_length);
    for (size_t i = 0; i < sequence_length; ++i) {
	(*outside)[i].resize(sequence_length);
	for (size_t j = 0; j < sequence_length; ++j) {
	    (*outside)[i][j].resize(NumNonterminalTypes(),
				    -numeric_limits<double>::infinity());
	}
    }

    // Base: outside[0][n-1][a] = p(a)
    for (const auto &root_pair : lprob_root_) {
	Nonterminal a = root_pair.first;
	double root_prior = root_pair.second;
	(*outside)[0][sequence_length - 1][a] = root_prior;
    }

    // Main body.
    for (int span_length = sequence_length - 2; span_length >= 0;
	 --span_length) {
	// Span length starts from (length - 2). The base covers (length - 1)
	for (size_t i = 0; i < sequence_length - span_length; ++i) {
	    size_t j = i + span_length;

	    // Below, we only consider relevant nonterminals for span (i, j) and
	    // leave others with probability 0. This is technically wrong since
	    // the outside probability is still not necessarily 0, but it *will*
	    // be 0 when multiplied by the (zero) inside probability.
	    if (span_length > 0) {
		//                        a
		// Want outside tree of  / \    Only consider interminals for a.
		//                    x_i...x_j
		for (const auto &interminal_pair : interminal_) {
		    Nonterminal a = interminal_pair.first;
		    (*outside)[i][j][a] =
			ComputeOutsideProbabilityPCFG(a, i, j, sequence_length,
						      inside, *outside);
		}
	    } else {
		//                        a
		// Want outside tree of   |   Only consider preterminals for a.
		//                       x_i
		for (const auto &preterminal_pair : preterminal_) {
		    Nonterminal a = preterminal_pair.first;
		    (*outside)[i][j][a] =
			ComputeOutsideProbabilityPCFG(a, i, j, sequence_length,
						      inside, *outside);
		}
	    }
	}
    }
}

double Grammar::ComputeOutsideProbabilityPCFG(
    Nonterminal a, size_t i, size_t j, size_t sequence_length,
    const Chart &inside, const Chart &outside) {
    double outside_lprob = -numeric_limits<double>::infinity();

    // Consider all cases of being a left child: for all nonterminals b, c and
    // all split points k = [j+1, length-1],
    //                    ...
    //                     b
    //                 /      \
    //                a        c
    //               / \      / \
    //          ..  i   j   j+1  k ...
    vector<tuple<Nonterminal, Nonterminal, double> > *bcs =
	&left_parent_sibling_[a];
    for (size_t idx = 0; idx < bcs->size(); ++idx) {
	tuple<Nonterminal, Nonterminal, double> *bc_tuple = &bcs->at(idx);
	Nonterminal b = get<0>(*bc_tuple);
	Nonterminal c = get<1>(*bc_tuple);
	double lprob_bac = get<2>(*bc_tuple);
	for (size_t split_point = j + 1; split_point < sequence_length;
	     ++split_point) {
	    double outside_lprob_left = lprob_bac +
		inside[j+1][split_point][c] + outside[i][split_point][b];
	    outside_lprob = util_math::sum_logs(outside_lprob,
						outside_lprob_left);
	}
    }

    // Consider all cases of being a right child: for all nonterminals b, c and
    // all split points k = [0, i-1],
    //                    ...
    //                     b
    //                 /      \
    //                c        a
    //               / \      / \
    //          ..  k  i-1   i   j ...
    bcs = &right_parent_sibling_[a];
    for (size_t idx = 0; idx < bcs->size(); ++idx) {
	tuple<Nonterminal, Nonterminal, double> *bc_tuple = &bcs->at(idx);
	Nonterminal b = get<0>(*bc_tuple);
	Nonterminal c = get<1>(*bc_tuple);
	double lprob_bca = get<2>(*bc_tuple);
	for (size_t split_point = 0; split_point < i; ++split_point) {
	    double outside_lprob_right = lprob_bca +
		inside[split_point][i-1][c] + outside[split_point][j][b];
	    outside_lprob = util_math::sum_logs(outside_lprob,
						outside_lprob_right);
	}
    }
    return outside_lprob;
}

Node *Grammar::RecoverMaxMarginalTree(TerminalSequence *terminals,
				      const Chart &marginal) {
    size_t sequence_length = terminals->Length();

    // chart[i][j][a] = highest score of a tree rooted at a spanning terminals
    // from position i to position j.
    Chart chart(sequence_length);

    // bp[i][j][a] = backpointer corresponding to chart[i][j][a].
    Backpointer bp(sequence_length);
    for (size_t i = 0; i < sequence_length; ++i) {
	chart[i].resize(sequence_length);
	bp[i].resize(sequence_length);
	for (size_t j = 0; j < sequence_length; ++j) {
	    chart[i][j].resize(NumNonterminalTypes(),
			       -numeric_limits<double>::infinity());
	    bp[i][j].resize(NumNonterminalTypes());
	}
    }

    // Base case.
    for (size_t i = 0; i < sequence_length; ++i) {
	for (const auto &preterminal_pair : preterminal_) {
	    Nonterminal a = preterminal_pair.first;
	    chart[i][i][a] = marginal[i][i][a];
	}
    }

    // Main body.
    for (size_t span_length = 1; span_length < sequence_length; ++span_length) {
	for (size_t i = 0; i < sequence_length - span_length; ++i) {
	    size_t j = i + span_length;
	    for (const auto &interminal_pair : interminal_) {
		Nonterminal a = interminal_pair.first;
		double max_total_score = -numeric_limits<double>::infinity();
		Nonterminal best_b = -1;
		Nonterminal best_c = -1;
		size_t best_split_point = -1;
		double node_score = marginal[i][j][a];

		vector<tuple<Nonterminal, Nonterminal, double> > *bcs =
		    &binary_rhs_[a];
		for (size_t idx = 0; idx < bcs->size(); ++idx) {
		    tuple<Nonterminal, Nonterminal, double> *bc_tuple =
			&bcs->at(idx);
		    Nonterminal b = get<0>(*bc_tuple);
		    Nonterminal c = get<1>(*bc_tuple);
		    for (size_t split_point = i; split_point < j;
			 ++split_point) {
			double total_score = node_score +
			    chart[i][split_point][b] +
			    chart[split_point+1][j][c];
			if (total_score > max_total_score) {
			    max_total_score = total_score;
			    best_b = b;
			    best_c = c;
			    best_split_point = split_point;
			}
		    }
		}
		chart[i][j][a] = max_total_score;
		bp[i][j][a] = make_tuple(best_b, best_c, best_split_point);
	    }
	}
    }

    double best_root_score = -numeric_limits<double>::infinity();
    Nonterminal best_root = -1;
    for (const auto &nonterminal_pair : nonterminal_dictionary_inverse_) {
	Nonterminal a = nonterminal_pair.first;
	if (chart[0][sequence_length - 1][a] > best_root_score) {
	    best_root_score = chart[0][sequence_length - 1][a];
	    best_root = a;
	}
    }

    ASSERT(best_root >= 0, "Parse failure: " << terminals->ToString());
    Node *best_tree =
	RecoverFromBackpointer(bp, terminals,
			       0, sequence_length - 1, best_root);
    return best_tree;
}

Node *Grammar::RecoverFromBackpointer(
    const Backpointer &bp, TerminalSequence *terminals,
    size_t start_position, size_t end_position, Nonterminal a) {
    if (start_position == end_position) {
	Terminal x = terminals->TerminalNumber(start_position);
	string x_string = terminals->TerminalString(start_position);
	Node *leaf = new Node(a, x);
	leaf->set_terminal_string(x_string);
	leaf->set_span(start_position, end_position);
	return leaf;
    } else {
	Nonterminal b = get<0>(bp[start_position][end_position][a]);
	Nonterminal c = get<1>(bp[start_position][end_position][a]);
	size_t split_point = get<2>(bp[start_position][end_position][a]);
	Node *node_a = new Node(a, -1);
	Node *node_b = RecoverFromBackpointer(bp, terminals, start_position,
					      split_point, b);
	Node *node_c = RecoverFromBackpointer(bp, terminals, split_point + 1,
					      end_position, c);
	node_a->AppendToChildren(node_b);
	node_a->AppendToChildren(node_c);
	node_a->set_span(start_position, end_position);
	return node_a;
    }
}

void Grammar::EstimatePCFG(TreeSet *trees) {
    Clear();
    // Collect rule and nonterminal statistics.
    unordered_map<Nonterminal, size_t> count_nonterminal;
    unordered_map<Terminal, size_t> count_terminal;
    unordered_map<Nonterminal,
		  unordered_map<Nonterminal,
				unordered_map<Nonterminal, size_t> > >
	count_binary;
    unordered_map<Nonterminal, unordered_map<Terminal, size_t> > count_unary;
    unordered_map<Nonterminal, size_t> count_root;

    for (size_t i = 0; i < trees->NumTrees(); ++i) {
	Node *tree = trees->Tree(i);
	string root_string = tree->nonterminal_string();
	AddNonterminalIfUnknown(root_string);
	Nonterminal root = nonterminal_dictionary_[root_string];
	++count_root[root];  // root nonterminal

	stack<Node *> dfs_stack;  // Depth-first search (DFS)
	dfs_stack.push(tree);
	while (!dfs_stack.empty()) {
	    Node *node = dfs_stack.top();
	    dfs_stack.pop();
	    string a_string = node->nonterminal_string();
	    Nonterminal a = AddNonterminalIfUnknown(a_string);
	    ++count_nonterminal[a];  // either interminal or preterminal
	    if (node->NumChildren() == 2) {
		if (interminal_.find(a) == interminal_.end()) {
		    interminal_[a] = true;  // recording interminal
		}
		string b_string = node->Child(0)->nonterminal_string();
		string c_string = node->Child(1)->nonterminal_string();
		Nonterminal b = AddNonterminalIfUnknown(b_string);
		Nonterminal c = AddNonterminalIfUnknown(c_string);
		++count_binary[a][b][c];  // binary rule
	    } else if (node->NumChildren() == 0) {
		if (preterminal_.find(a) == preterminal_.end()) {
		    preterminal_[a] = true;  // recording preterminal
		}
		string x_string = node->terminal_string();
		Terminal x = AddTerminalIfUnknown(x_string);
		++count_terminal[x];  // terminal
		++count_unary[a][x];  // unary rule
	    } else {
		ASSERT(false, "Derivation not in CNF!");
	    }

	    for (size_t j = 0; j < node->NumChildren(); ++j) {
		dfs_stack.push(node->Child(j));
	    }
	}
    }

    // Check every nonterminal is either an interminal or a preterminal.
    for (const auto &a_pair : nonterminal_dictionary_inverse_) {
	Nonterminal a = a_pair.first;
	string a_string = a_pair.second;
	bool a_is_interminal = (interminal_.find(a) != interminal_.end());
	bool a_is_preterminal = (preterminal_.find(a) != preterminal_.end());
	ASSERT(a_is_interminal != a_is_preterminal,
	       "Found a nonterminal used as in/preterminal: " << a_string);
    }

    // Estimate binary rule probababilities from counts.
    for (const auto &a_pair : count_binary) {
	Nonterminal a = a_pair.first;
	size_t count_a = count_nonterminal[a];
	for (const auto &b_pair : a_pair.second) {
	    Nonterminal b = b_pair.first;
	    for (const auto &c_pair : b_pair.second) {
		Nonterminal c = c_pair.first;
		size_t count_abc = c_pair.second;
		double prob = log(count_abc) - log(count_a);
		lprob_binary_[a][b][c] = prob;
		binary_rhs_[a].push_back(make_tuple(b, c, prob));
		left_parent_sibling_[b].push_back(make_tuple(a, c, prob));
		right_parent_sibling_[c].push_back(make_tuple(a, b, prob));
	    }
	}
    }

    // Estimate unary rule probabilities from counts.
    for (const auto &a_pair : count_unary) {
	Nonterminal a = a_pair.first;
	size_t count_a = count_nonterminal[a];
	for (const auto &x_pair : a_pair.second) {
	    Terminal x = x_pair.first;
	    size_t count_ax = x_pair.second;
	    lprob_unary_[a][x] = log(count_ax) - log(count_a);
	}
    }

    // Estimate root nonterminal probabilities from counts.
    size_t num_root_occurrences = 0;
    for (const auto &a_pair : count_root) {
	num_root_occurrences += a_pair.second;
    }
    for (const auto &a_pair : count_root) {
	Nonterminal a = a_pair.first;
	lprob_root_[a] = log(count_root[a]) - log(num_root_occurrences);
    }
    AssertProperDistributions();
}

Nonterminal Grammar::AddNonterminalIfUnknown(const string &a_string) {
    ASSERT(!a_string.empty(), "Trying to add an empty string for nonterminal!");
    if (nonterminal_dictionary_.find(a_string) ==
	nonterminal_dictionary_.end()) {
	Nonterminal nonterminal_type = nonterminal_dictionary_.size();
	nonterminal_dictionary_[a_string] = nonterminal_type;
	nonterminal_dictionary_inverse_[nonterminal_type] = a_string;
    }
    return nonterminal_dictionary_[a_string];
}

Terminal Grammar::AddTerminalIfUnknown(const string &x_string) {
    ASSERT(!x_string.empty(), "Trying to add an empty string for terminal!");
    if (terminal_dictionary_.find(x_string) == terminal_dictionary_.end()) {
	Terminal terminal_type = terminal_dictionary_.size();
	terminal_dictionary_[x_string] = terminal_type;
	terminal_dictionary_inverse_[terminal_type] = x_string;
    }
    return terminal_dictionary_[x_string];
}

void Grammar::AssertProperDistributions() {
    // Assert proper binary rule distributions.
    for (const auto a_pair : lprob_binary_) {
	Nonterminal a = a_pair.first;
	double mass = 0.0;
	for (const auto &b_pair : a_pair.second) {
	    for (const auto &c_pair : b_pair.second) {
		mass += exp(c_pair.second);
	    }
	}
	ASSERT(fabs(1.0 - mass) < 1e-5,
	       "Improper binary rule distribution for interminal string "
	       << nonterminal_dictionary_inverse_[a] << ": " << mass);
    }

    // Assert proper unary rule distributions.
    for (const auto a_pair : lprob_unary_) {
	Nonterminal a = a_pair.first;
	double mass = 0.0;
	for (const auto x_pair : a_pair.second) {
	    mass += exp(x_pair.second);
	}
	ASSERT(fabs(1.0 - mass) < 1e-5,
	       "Improper unary rule distribution for preterminal string "
	       << nonterminal_dictionary_inverse_[a] << ": " << mass);
    }

    // Assert a proper root nonterminal distribution.
    double mass = 0.0;
    for (const auto a_pair : lprob_root_) {
	mass += exp(a_pair.second);
    }
    ASSERT(fabs(1.0 - mass) < 1e-5,
	   "Improper root nonterminal distribution: " << mass);
}

void Grammar::AssignNumericIdentities(Node *tree) {
    if (tree->IsEmpty()) { return; }
    string a_string = tree->nonterminal_string();
    ASSERT(nonterminal_dictionary_.find(a_string) !=
	   nonterminal_dictionary_.end(),
	   "Assigning numeric ID to an unknown nonterminal: " << a_string);
    tree->set_nonterminal_number(nonterminal_dictionary_[a_string]);
    if (tree->IsPreterminal()) {
	string x_string = tree->terminal_string();
	if (terminal_dictionary_.find(x_string) != terminal_dictionary_.end()) {
	    tree->set_terminal_number(terminal_dictionary_[x_string]);
	} else {
	    // Use -1 for an unknown terminal string.
	    tree->set_terminal_number(-1);
	}
    } else {
	AssignNumericIdentities(tree->Child(0));
	AssignNumericIdentities(tree->Child(1));
    }
}

void Grammar::AssignStringIdentities(Node *tree) {
    if (tree->IsEmpty()) { return; }
    Nonterminal a = tree->nonterminal_number();
    ASSERT(nonterminal_dictionary_inverse_.find(a) !=
	   nonterminal_dictionary_inverse_.end(),
	   "Assigning string ID to an unknown nonterminal: " << a);
    tree->set_nonterminal_string(nonterminal_dictionary_inverse_[a]);
    if (tree->IsPreterminal()) {
	ASSERT(!tree->terminal_string().empty(), "Terminal nodes must have "
	       "string IDs at all time!");
    } else {
	AssignStringIdentities(tree->Child(0));
	AssignStringIdentities(tree->Child(1));
    }
}

void Grammar::AssignNumericIdentities(TerminalSequence *terminals) {
    vector<Terminal> terminal_numbers;
    vector<Nonterminal> preterminal_numbers;

    for (size_t i = 0; i < terminals->Length(); ++i) {
	string terminal_string = terminals->TerminalString(i);
	if (terminal_dictionary_.find(terminal_string) !=
	    terminal_dictionary_.end()) {
	    terminal_numbers.push_back(terminal_dictionary_[terminal_string]);
	} else {
	    // Use -1 for an unknown terminal string.
	    terminal_numbers.push_back(-1);
	}

	string preterminal_string = terminals->PreterminalString(i);
	if (nonterminal_dictionary_.find(preterminal_string) !=
	    nonterminal_dictionary_.end()) {
	    preterminal_numbers.push_back(
		nonterminal_dictionary_[preterminal_string]);
	} else {
	    // Use -1 for an unknown preterminal string.
	    preterminal_numbers.push_back(-1);
	}
    }
    terminals->set_terminal_numbers(terminal_numbers);
    terminals->set_preterminal_numbers(preterminal_numbers);
}
