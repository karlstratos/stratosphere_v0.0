// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the code for grammar.

#include "gtest/gtest.h"

#include <math.h>

#include "../grammar.h"

// Test class for checking probabilities.
class ProbabilityTest : public testing::Test {
protected:
    virtual void SetUp() {
	tree_equal_ = tree_reader_.CreateTreeFromTreeString(
	    "(A (A (B b) (A (B b) (B b))) (B b))");
	tree_AB_ = tree_reader_.CreateTreeFromTreeString(
	    "(A (A (B b) (B b)) (B b))");
	tree_BA_ = tree_reader_.CreateTreeFromTreeString(
	    "(A (B b) (A (B b) (B b)))");
	TreeSet treeset_equal, treeset_AB, treeset_BA;
	treeset_equal.AddTree(tree_equal_->Copy());
	treeset_AB.AddTree(tree_equal_->Copy());
	treeset_AB.AddTree(tree_AB_->Copy());
	treeset_BA.AddTree(tree_equal_->Copy());
	treeset_BA.AddTree(tree_BA_->Copy());

	// A -> A B    0.3333
	//   -> B A    0.3333
	//   -> B B    0.3333
	grammar_equal_.set_model_directory("/tmp/grammar_equal_");
	grammar_equal_.Train(&treeset_equal);

	// A -> A B    0.4
	//   -> B A    0.2
	//   -> B B    0.4
	grammar_AB_.set_model_directory("/tmp/grammar_AB_");
	grammar_AB_.Train(&treeset_AB);

	// A -> A B    0.2
	//   -> B A    0.4
	//   -> B B    0.4
	grammar_BA_.set_model_directory("/tmp/grammar_BA_");
	grammar_BA_.Train(&treeset_BA);
    }

    virtual void TearDown() {
	tree_equal_->DeleteSelfAndDescendents();
	tree_AB_->DeleteSelfAndDescendents();
	tree_BA_->DeleteSelfAndDescendents();
	ASSERT(system("rm -rf /tmp/grammar_equal_") == 0 &&
	       system("rm -rf /tmp/grammar_AB_") == 0 &&
	       system("rm -rf /tmp/grammar_BA_") == 0,
	       "Cannot remove test directories in /tmp/");
    }

    TreeReader tree_reader_;
    Grammar grammar_equal_;
    Grammar grammar_AB_;
    Grammar grammar_BA_;
    Node *tree_equal_;
    Node *tree_AB_;
    Node *tree_BA_;
    double tol_ = 1e-4;
};

// Tests the estimated tree probabilities under PCFG.
TEST_F(ProbabilityTest, TestComputePCFGTreeProbability) {
    // Probabilities under grammar_equal_.
    EXPECT_NEAR(0.0370,
		exp(grammar_equal_.ComputePCFGTreeProbability(tree_equal_)),
		tol_);
    EXPECT_NEAR(0.1111,
		exp(grammar_equal_.ComputePCFGTreeProbability(tree_AB_)),
		tol_);
    EXPECT_NEAR(0.1111,
		exp(grammar_equal_.ComputePCFGTreeProbability(tree_BA_)),
		tol_);

    // Probabilities under grammar_AB_.
    EXPECT_NEAR(0.0320,
		exp(grammar_AB_.ComputePCFGTreeProbability(tree_equal_)),
		tol_);
    EXPECT_NEAR(0.1600,
		exp(grammar_AB_.ComputePCFGTreeProbability(tree_AB_)),
		tol_);
    EXPECT_NEAR(0.0800,
		exp(grammar_AB_.ComputePCFGTreeProbability(tree_BA_)),
		tol_);

    // Probabilities under grammar_BA_.
    EXPECT_NEAR(0.0320,
		exp(grammar_BA_.ComputePCFGTreeProbability(tree_equal_)),
		tol_);
    EXPECT_NEAR(0.0800,
		exp(grammar_BA_.ComputePCFGTreeProbability(tree_AB_)),
		tol_);
    EXPECT_NEAR(0.1600,
		exp(grammar_BA_.ComputePCFGTreeProbability(tree_BA_)),
		tol_);
}

// Tests the Viterbi decoding with CKY algorithm under PCFG.
TEST_F(ProbabilityTest, TestCKYAlgorithmPCFG) {
  vector<string> terminal_strings;
  for (int i = 0; i < 6; ++i) { terminal_strings.push_back("b"); }

    // We must obtain a left-branching tree with AB grammar.
    grammar_AB_.set_decoding_method("viterbi");
    Node *AB_parse = grammar_AB_.Parse(terminal_strings);
    Node *AB_answer = tree_reader_.CreateTreeFromTreeString(
	"(A (A (A (A (A (B b) (B b)) (B b)) (B b)) (B b)) (B b))");
    EXPECT_TRUE(AB_parse->Compare(AB_answer));

    // We must obtain a right-branching tree with BA grammar.
    grammar_BA_.set_decoding_method("viterbi");
    Node *BA_parse = grammar_BA_.Parse(terminal_strings);
    Node *BA_answer = tree_reader_.CreateTreeFromTreeString(
	"(A (B b) (A (B b) (A (B b) (A (B b) (A (B b) (B b))))))");
    EXPECT_TRUE(BA_parse->Compare(BA_answer));

    AB_parse->DeleteSelfAndDescendents();
    AB_answer->DeleteSelfAndDescendents();
    BA_parse->DeleteSelfAndDescendents();
    BA_answer->DeleteSelfAndDescendents();
}

// Tests the marginal probabilities under PCFG.
TEST_F(ProbabilityTest, TestMarginalsPCFG) {
    vector<string> terminal_strings;
    for (int i = 0; i < 4; ++i) { terminal_strings.push_back("b"); }

    Chart marginal;
    grammar_equal_.ComputeMarginalsPCFG(terminal_strings, &marginal);
    Nonterminal A = grammar_equal_.nonterminal_str2num("A");
    Nonterminal B = grammar_equal_.nonterminal_str2num("B");

    // Under grammar_equal_, all possible 4 derivations are equally likely:
    // 1. (A (A (A (B b) (B b)) (B b)) (B b))   wp 0.0370
    // 2. (A (B b) (A (B b) (A (B b) (B b))))   wp 0.0370
    // 3. (A (A (B b) (A (B b) (B b))) (B b))   wp 0.0370
    // 4. (A (B b) (A (A (B b) (B b)) (B b)))   wp 0.0370
    //
    // All four trees have (B,i,i).
    for (size_t i = 0; i < terminal_strings.size(); ++i) {
	EXPECT_NEAR(0.1481, exp(marginal[i][i][B]), tol_);
    }

    // Only tree 1 has (A,0,1).
    EXPECT_NEAR(0.0370, exp(marginal[0][1][A]), tol_);

    // Trees 2 and 3 have (A,1,2).
    EXPECT_NEAR(0.0740, exp(marginal[1][2][A]), tol_);

    // Only tree 4 has (A,2,3).
    EXPECT_NEAR(0.0370, exp(marginal[2][3][A]), tol_);

    // Trees 1 and 3 have (A,0,2).
    EXPECT_NEAR(0.0740, exp(marginal[0][2][A]), tol_);

    // Trees 2 and 4 have (A,1,3).
    EXPECT_NEAR(0.0740, exp(marginal[1][3][A]), tol_);

    // All four trees have (A,0,3).
    EXPECT_NEAR(0.1481, exp(marginal[0][3][A]), tol_);
}

// Tests max-marginal parsing under PCFG.
TEST_F(ProbabilityTest, TestMaxMarginalParsingPCFG) {
    vector<string> terminal_strings;
    for (int i = 0; i < 4; ++i) { terminal_strings.push_back("b"); }

    // Even under grammar_equal_, not all trees are equally likely anymore.
    grammar_equal_.set_decoding_method("marginal");
    Node *equal_parse = grammar_equal_.Parse(terminal_strings);
    Node *equal_answer1 = tree_reader_.CreateTreeFromTreeString(
	"(A (A (B b) (A (B b) (B b))) (B b))");
    Node *equal_answer2 = tree_reader_.CreateTreeFromTreeString(
	"(A (B b) (A (A (B b) (B b)) (B b)))");
    EXPECT_TRUE(equal_parse->Compare(equal_answer1) ||
		equal_parse->Compare(equal_answer2));

    // Under grammar_AB_, we should have one of the two equally scored trees.
    grammar_AB_.set_decoding_method("marginal");
    Node *AB_parse = grammar_AB_.Parse(terminal_strings);
    Node *AB_answer1 = tree_reader_.CreateTreeFromTreeString(
	"(A (A (A (B b) (B b)) (B b)) (B b))");
    Node *AB_answer2 = tree_reader_.CreateTreeFromTreeString(
	"(A (A (B b) (A (B b) (B b))) (B b))");
    EXPECT_TRUE(AB_parse->Compare(AB_answer1) || AB_parse->Compare(AB_answer2));

    // Under grammar_BA_, we should have one of the two equally scored trees.
    grammar_BA_.set_decoding_method("marginal");
    Node *BA_parse = grammar_BA_.Parse(terminal_strings);
    Node *BA_answer1 = tree_reader_.CreateTreeFromTreeString(
	"(A (B b) (A (B b) (A (B b) (B b))))");
    Node *BA_answer2 = tree_reader_.CreateTreeFromTreeString(
	"(A (B b) (A (A (B b) (B b)) (B b)))");
    EXPECT_TRUE(BA_parse->Compare(BA_answer1) || BA_parse->Compare(BA_answer2));

    equal_parse->DeleteSelfAndDescendents();
    equal_answer1->DeleteSelfAndDescendents();
    equal_answer2->DeleteSelfAndDescendents();
    AB_parse->DeleteSelfAndDescendents();
    AB_answer1->DeleteSelfAndDescendents();
    AB_answer2->DeleteSelfAndDescendents();
    BA_parse->DeleteSelfAndDescendents();
    BA_answer1->DeleteSelfAndDescendents();
    BA_answer2->DeleteSelfAndDescendents();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
