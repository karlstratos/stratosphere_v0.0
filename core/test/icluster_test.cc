// Author: Karl Stratos (me@karlstratos.com)
//
// Check the correctness of the code for interactive clustering.

#include "gtest/gtest.h"

#include <limits.h>

#include "../icluster.h"
#include "../util.h"

// Test class with data points and clusters.
class ClusterTest : public testing::Test {
protected:
    virtual void SetUp() {
	data.x2f["coffee"] = 5;
	data.x2f["tea"] = 3;
	data.x2f["dog"] = 8;
	data.x2f["cat"] = 7;
	data.x2f["walk"] = 2;
	data.x2f["run"] = 1;
	data.x2f["walked"] = 7;
	data.x2f["ran"] = 5;

	// A 4-clustering
	c2x1["00"] = { "coffee", "tea" };
	c2x1["01"] = { "dog", "cat" };
	c2x1["10"] = { "walk", "run" };
	c2x1["11"] = { "walked", "ran" };

	// Another 4-clustering
	c2x2["00"] = { "coffee", "tea" };
	c2x2["01"] = { "dog", "cat" };
	c2x2["10"] = { "walk", "walked" };
	c2x2["11"] = { "run", "ran" };

	// A 2-clustering
	c2x3["like"] = { "coffee", "dog", "walk", "walked" };
	c2x3["hate"] = { "tea", "cat", "run", "ran" };
    }
    Data data;
    unordered_map<string, unordered_set<string> > c2x1, c2x2, c2x3;
};

// Tests Read.
TEST_F(ClusterTest, Read) {
    EXPECT_EQ(8, data.NumPoints());
    EXPECT_EQ(7, data.Frequency("cat"));

    Cluster C1(&data, c2x1);
    EXPECT_EQ(4, C1.NumClusters());
    EXPECT_EQ(2, C1.ClusterSize("01"));

    Cluster C3(&data, c2x3);
    EXPECT_EQ(2, C3.NumClusters());
    EXPECT_EQ(4, C3.ClusterSize("hate"));
}

// Tests finding splittable/mergeable clusters.
TEST_F(ClusterTest, FindSplitMerge) {
    ClusterFixer fixer;
    fixer.proposed.Read(&data, c2x2);
    fixer.desired.Read(&data, c2x1);

    // 10 and 11 are splittable.
    vector<string> s = fixer.FindSplittable();
    EXPECT_EQ(2, s.size());

    // With small eta, {10, 11} is issued by both 10 and 11.
    fixer.eta = 0.1;
    vector<unordered_set<string> > m = fixer.FindMergeable();
    EXPECT_EQ(2, m.size());
    EXPECT_EQ(2, m[0].size());
    EXPECT_EQ(2, m[1].size());

    // With big eta, no merge is possible.
    fixer.eta = 1.0;
    m = fixer.FindMergeable();
    EXPECT_EQ(0, m.size());
}


// Tests split/merge operations.
TEST_F(ClusterTest, SplitMerge) {
    ClusterFixer fixer;
    fixer.proposed.Read(&data, c2x1);
    fixer.desired.Read(&data, c2x3);

    // Clean split. 10:{walk, run} -> 100:{walk} 101:{run}
    size_t overclustering_before = fixer.Overclustering();
    fixer.split_method = "clean";
    fixer.cluster_to_split = "10";
    fixer.Split();
    EXPECT_EQ(5, fixer.proposed.NumClusters());
    EXPECT_EQ(1, fixer.proposed.c2x["100"].size());
    EXPECT_TRUE(fixer.proposed.c2x["100"].find("walk") !=
		fixer.proposed.c2x["100"].end());
    EXPECT_EQ(1, fixer.proposed.c2x["101"].size());
    EXPECT_TRUE(fixer.proposed.c2x["101"].find("run") !=
		fixer.proposed.c2x["101"].end());

    // Lemma: a clean split always reduces overclutering error by 1.
    EXPECT_EQ(overclustering_before - 1, fixer.Overclustering());

    // Another clean split. 11:{walked, ran} -> 110:{walked} 111:{ran}
    fixer.cluster_to_split = "11";
    fixer.Split();
    EXPECT_EQ(6, fixer.proposed.NumClusters());

    // Merge. 100:{walk} 110:{walked} -> 100:{walk, walked}
    size_t underclustering_before = fixer.Underclustering();
    fixer.clusters_to_merge.first = "100";
    fixer.clusters_to_merge.second = "110";
    fixer.Merge();
    EXPECT_EQ(5, fixer.proposed.NumClusters());
    EXPECT_EQ(2, fixer.proposed.c2x["100"].size());
    EXPECT_TRUE(fixer.proposed.c2x["100"].find("walk") !=
		fixer.proposed.c2x["100"].end());
    EXPECT_TRUE(fixer.proposed.c2x["100"].find("walked") !=
		fixer.proposed.c2x["100"].end());

    // Lemma: a pure merge always reduces underclutering error by 1.
    EXPECT_EQ(underclustering_before - 1, fixer.Underclustering());

    fixer.clusters_to_merge.first = "100";
    fixer.clusters_to_merge.second = "111";
    fixer.Merge();
    EXPECT_EQ(4, fixer.proposed.NumClusters());
    EXPECT_EQ(3, fixer.proposed.c2x["100"].size());
    EXPECT_TRUE(fixer.proposed.c2x["100"].find("walk") !=
		fixer.proposed.c2x["100"].end());
    EXPECT_TRUE(fixer.proposed.c2x["100"].find("walked") !=
		fixer.proposed.c2x["100"].end());
    EXPECT_TRUE(fixer.proposed.c2x["100"].find("ran") !=
		fixer.proposed.c2x["100"].end());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
