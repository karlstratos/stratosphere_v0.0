// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the clustering code.

#include "gtest/gtest.h"

#include <limits>

#include "../cluster.h"

// Test class that provides a set of vectors for clustering.
class VectorsForClustering : public testing::Test {
protected:
    virtual void SetUp() {
	Eigen::VectorXd v0(1);
	Eigen::VectorXd v1(1);
	Eigen::VectorXd v2(1);
	Eigen::VectorXd v3(1);
	Eigen::VectorXd v4(1);
	Eigen::VectorXd v5(1);
	v0 << 0.0;
	v3 << 0.3;
	v1 << 3.0;
	v4 << 3.9;
	v2 << 9.0;
	v5 << 9.6;
	//------v0-v3------------v1----v4----------------------------v2--v5-----
	ordered_vectors_.push_back(v0);
	ordered_vectors_.push_back(v1);
	ordered_vectors_.push_back(v2);
	ordered_vectors_.push_back(v3);
	ordered_vectors_.push_back(v4);
	ordered_vectors_.push_back(v5);
    }
    vector<Eigen::VectorXd> ordered_vectors_;
    GreedyLazyFocusedAgglomerativeClustering agglomerative_;
};

// Checks agglomerative clustering with unlimited number of leaf clusters.
TEST_F(VectorsForClustering, AgglomerativeAllLeaves) {
    //size_t num_leaf_clusters = numeric_limits<size_t>::max();
    size_t num_leaf_clusters = 6;
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    cout << gamma << endl;
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (const auto &pair : *agglomerative_.leaves()) {
	cout << pair.first << ": ";
	for (size_t i = 0; i < pair.second.size(); ++i) {
	    cout << pair.second[i] << " ";
	}
	cout << endl;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
