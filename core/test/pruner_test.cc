// Author: Karl Stratos (me@karlstratos.com)
//
// Check the correctness of the code for pruning.

#include "gtest/gtest.h"

#include <limits.h>

#include "../pruner.h"
#include "../util.h"

// Test class with example trees.
class PrunerTest : public testing::Test {
protected:
    virtual void SetUp() {
	// bitstring1
	//                   _______ TOP____________
	//                  /                       \
	//                 0                         1
	//              /      \               /          \
	//            00       01             10           11
	//          /    \   /    \         /   \        /     \
	//        000   001 010   011     100    101   110    111
	//        |      |   |     |       |      |     |      |
	//      dog     cat tea   coffee  walk   run  walked  ran
	//
	bitstring1["000"] = "dog";
	bitstring1["001"] = "cat";
	bitstring1["010"] = "tea";
	bitstring1["011"] = "coffee";
	bitstring1["100"] = "walk";
	bitstring1["101"] = "run";
	bitstring1["110"] = "walked";
	bitstring1["111"] = "ran";

	// bitstring2
	//                   _______ TOP____________
	//                  /                       \
	//                 0               _________ 1
	//                 |              /           \
	//                 x         ___ 10           11
	//                          /      \           |
	//                        100      101         q
	//                         |        |
	//                         y        z
	//
	bitstring2["0"] = "x";
	bitstring2["100"] = "y";
	bitstring2["101"] = "z";
	bitstring2["11"] = "q";

	// bitstring3
	//                   _______________ TOP_________________
	//                  /                                    \
	//               _ 0________                         ____ 1_____
	//              /           \                       /           \
	//            00             01                   10             11
	//          /    \         /     \              /   \          /     \
	//        000     001     010    011          100    101      110    111
	//      /   |      |      / \      |           |    /   \      |      |
	//   0000  0001   x3   0100 0101   x6         x7  1010 1011   x10    x11
	//     |    |           |    |                     |     |
	//    x1    x2          x4   x5                    x8    x9
	//
	bitstring3["0000"] = "x1";
	bitstring3["0001"] = "x2";
	bitstring3["001"] = "x3";
	bitstring3["0100"] = "x4";
	bitstring3["0101"] = "x5";
	bitstring3["011"] = "x6";
	bitstring3["100"] = "x7";
	bitstring3["1010"] = "x8";
	bitstring3["1011"] = "x9";
	bitstring3["110"] = "x10";
	bitstring3["111"] = "x11";

	// bitstring4
	//                   TOP
	//                  /
	//                 0
	//                 |
	//                 x
	bitstring4["0"] = "x";

	// bitstring5
	//                  TOP
	//                    \
	//                    1
	//                    |
	//                    x
	bitstring5["1"] = "x";

	// bitstring6 (bug: closing parentheses)
	//
	//              TOP ________________
	//           /                      \
	//          0        _______________ 1
	//          |       /                  \
	//         x1    _ 10________           11
	//              /            \          |
	//            100            101       x5
	//            |             /  \
	//           x2          1010  1011
	//                         |      |
	//                        x3      x4
	//
	// BUG: When reading 11, the code used to close parentheses by
	//
	// int j = bits_prev.size() - 1;
	// while (j >= 0 && bits_prev[j] != bits_now[j]) {
	//
	// but the index j=2 is out of bound for 11!
	// Unfortunately, the compiler does not report an error. Fix:
	//
	// int j = bits_prev.size() - 1;
	// while (j >= 0 && (j > bits_now.size() - 1 ||
	//                   bits_prev[j] != bits_now[j])) {
	//
	bitstring6["0"] = "x1";
	bitstring6["100"] = "x2";
	bitstring6["1010"] = "x3";
	bitstring6["1011"] = "x4";
	bitstring6["11"] = "x5";
    }
    unordered_map<string, string> bitstring1;
    unordered_map<string, string> bitstring2;
    unordered_map<string, string> bitstring3;
    unordered_map<string, string> bitstring4;
    unordered_map<string, string> bitstring5;
    unordered_map<string, string> bitstring6;
};

// Tests Read.
TEST_F(PrunerTest, Read) {
    Pruner pruner;
    pruner.ReadTree(bitstring1);
    EXPECT_TRUE(pruner.tree()->Compare(
		    "(TOP (0 (00 (000 dog) (001 cat)) "
		    "        (01 (010 tea) (011 coffee)))"
		    "     (1 (10 (100 walk) (101 run))"
		    "        (11 (110 walked) (111 ran))))"));

    pruner.ReadTree(bitstring2);
    EXPECT_TRUE(pruner.tree()->Compare("(TOP (0 x)"
				       "     (1 (10 (100 y) (101 z))"
				       "        (11 q)))"));

    pruner.ReadTree(bitstring3);
    EXPECT_TRUE(pruner.tree()->Compare(
		    "(TOP (0 (00 (000 (0000 x1) (0001 x2)) (001 x3))"
		    "        (01 (010 (0100 x4) (0101 x5)) (011 x6)))"
		    "     (1 (10 (100 x7) (101 (1010 x8) (1011 x9)))"
		    "        (11 (110 x10) (111 x11))))"));

    // Singleton trees: the only case where there is no left/right distinction.
    pruner.ReadTree(bitstring4);
    EXPECT_TRUE(pruner.tree()->Compare("(TOP (0 x))"));
    pruner.ReadTree(bitstring5);
    EXPECT_TRUE(pruner.tree()->Compare("(TOP (1 x))"));

    // Closing parentheses bug.
    pruner.ReadTree(bitstring6);
    EXPECT_TRUE(pruner.tree()->Compare(
		    "(TOP (0 x1) (1 (10 (100 x2) (101 (1010 x3)"
		    "                                 (1011 x4)))"
		    "               (11 x5)))"));

    // TODO: Check memory leaks.
}

// Tests label propagation for a singleton tree.
TEST_F(PrunerTest, LabelPropagationSingleton) {
    Pruner pruner;
    pruner.ReadTree(bitstring4);
    unordered_map<string, vector<string> > prototypes;
    prototypes["1"] = {"x"};
    unordered_map<string, vector<string> > propagation = \
	pruner.PropagateLabels(prototypes);
    EXPECT_EQ(1, propagation.size());
    EXPECT_EQ(1, propagation["1"].size());
    EXPECT_EQ("x", propagation["1"][0]);
}

// Tests finding consistent subtrees for a prunable case.
TEST_F(PrunerTest, FindConsistentSubtreesPrunable) {
    Pruner pruner;
    pruner.ReadTree(bitstring3);

    // The true clustering is a pruning of bitstring3.
    //
    //                   _______________ TOP_________________
    //                  /                                    \
    //               _ 0________                         ____ 1_____
    //              /           \                       /           \
    //            00             01                   10             11
    //          /    \         /     \              /   \          /     \
    //        000     001     010    011          100    101      110    111
    //      /   |      |      / \      |           |    /   \      |      |
    //   0000  0001  a.x3  0100 0101  c.x6      d.x7  1010 1011  d.x10 d.x11
    //     |    |           |    |                     |     |
    //  a.x1   a.x2       b.x4  b.x5                  d.x8  d.x9
    //
    unordered_map<string, vector<string> > clustering;
    clustering["a"] = {"x1", "x2", "x3"};
    clustering["b"] = {"x4", "x5"};
    clustering["c"] = {"x6"};
    clustering["d"] = {"x7", "x8", "x9", "x10", "x11"};

    // Given a prototype for each label, we should recover the clustering.
    unordered_map<string, string> proto2label;
    proto2label["x2"] = "a";
    proto2label["x4"] = "b";
    proto2label["x6"] = "c";
    proto2label["x9"] = "d";

    unordered_map<string, vector<Node *> > pure_subtrees;
    vector<pair<Node *, string> > unknown_subtrees;
    pruner.FindConsistentSubtrees(pruner.tree(), proto2label, &pure_subtrees,
				  &unknown_subtrees);

    // No unknown subtrees.
    EXPECT_EQ(0, unknown_subtrees.size());

    // For each label, we have a single pure subtree for a corr. gold cluster.
    EXPECT_EQ(clustering.size(), pure_subtrees.size());
    for (const auto &pair : pure_subtrees) {
	vector<string> proposed;
	pair.second[0]->Leaves(&proposed);
	EXPECT_EQ(clustering[pair.first], proposed);
    }
}

// Tests finding consistent subtrees for a non-prunable case.
TEST_F(PrunerTest, FindConsistentSubtreesNonPrunable) {
    Pruner pruner;
    pruner.ReadTree(bitstring3);

    // The true clustering is *not* a pruning of bitstring3.
    //
    //                   _______________ TOP_________________
    //                  /                                    \
    //               _ 0________                         ____ 1_____
    //              /           \                       /           \
    //            00             01                   10             11
    //          /    \         /     \              /   \          /     \
    //        000     001     010    011          100    101      110    111
    //      /   |      |      / \      |           |    /   \      |      |
    //   0000  0001   a.x3 0100 0101  b.x6      a.x7  1010 1011  c.x10  c.x11
    //     |    |           |    |                     |     |
    //  b.x1   b.x2        b.x4  b.x5                  c.x8  c.x9
    //
    unordered_map<string, vector<string> > clustering;
    clustering["a"] = {"x3", "x7"};
    clustering["b"] = {"x1", "x2", "x4", "x5", "x6"};
    clustering["c"] = {"x8", "x9", "x10", "x11"};

    // Prototypes don't cover all consistent subtrees.
    unordered_map<string, string> proto2label;
    proto2label["x2"] = "b";
    proto2label["x3"] = "a";  // Subtree 01 cannot be inferred.
    proto2label["x7"] = "a";
    proto2label["x8"] = "c";  // Subtree 11 cannot be inferred.

    unordered_map<string, vector<Node *> > pure_subtrees;
    vector<pair<Node *, string> > unknown_subtrees;
    pruner.FindConsistentSubtrees(pruner.tree(), proto2label, &pure_subtrees,
				  &unknown_subtrees);

    // Two unknown trees.
    EXPECT_EQ(2, unknown_subtrees.size());
    EXPECT_EQ(3, unknown_subtrees[0].first->NumLeaves());  // Subtree 01
    EXPECT_EQ(2, unknown_subtrees[1].first->NumLeaves());  // Subtree 11

    // Label "a" has two pure subtrees under the prototypes.
    EXPECT_EQ(2, pure_subtrees["a"].size());
    EXPECT_EQ(1, pure_subtrees["b"].size());
    EXPECT_EQ(1, pure_subtrees["c"].size());

    // But mapping unknown subtrees to majority sibling label works here.
    unordered_map<string, vector<string> > propagation = \
	pruner.LabelConsistentSubtrees(pure_subtrees, unknown_subtrees);
    EXPECT_EQ(clustering.size(), propagation.size());
    for (const auto &pair : clustering) {
	EXPECT_EQ(pair.second, propagation[pair.first]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
