// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the clustering code.

#include "gtest/gtest.h"

#include <limits>

#include "../cluster.h"

// Test class that provides a k-means example with an empty cluster problem:
// http://www.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html
class KMeansEmptyClusterVectors : public testing::Test {
protected:
    virtual void SetUp() {
	kmeans_.set_verbose(false);
	Eigen::VectorXd v0(2);
	Eigen::VectorXd v1(2);
	Eigen::VectorXd v2(2);
	Eigen::VectorXd v3(2);
	Eigen::VectorXd v4(2);
	Eigen::VectorXd v5(2);
	Eigen::VectorXd v6(2);
	v0 << 1.0, 1.0;
	v1 << 2.0, 2.8;
	v2 << 2.0, 2.9;
	v3 << 2.0, 3.0;
	v4 << 2.6, 3.0;
	v5 << 5.0, 3.0;
	v6 << 5.5, 2.0;
	vectors_.push_back(v0);
	vectors_.push_back(v1);
	vectors_.push_back(v2);
	vectors_.push_back(v3);
	vectors_.push_back(v4);
	vectors_.push_back(v5);
	vectors_.push_back(v6);
    }
    KMeans kmeans_;
    vector<Eigen::VectorXd> vectors_;
};

// Checks that 3-means can result in an empty cluster.
TEST_F(KMeansEmptyClusterVectors, EmptyClusterWith3Means) {
    // Given centers
    //   |
    //   |         v3    v4         [v5]
    //   |         v2
    //   |         v1                    [v6]
    //   |
    //   |    [v0]
    //   |
    //   |______________________________________
    vector<Eigen::VectorXd> centers = {vectors_[0], vectors_[5], vectors_[6]};

    // Iteration 1.
    //   |
    //   |    (    v3)   (v4         v5)
    //   |    (    v2)
    //   |    (    v1)                   (v6)
    //   |    (      )
    //   |    (v0    )
    //   |
    //   |______________________________________
    //
    //   |
    //   |         v3    v4    []    v5
    //   |         v2
    //   |       []v1                     [v6]
    //   |
    //   |     v0
    //   |
    //   |______________________________________
    //
    // Iteration 2. EMPTY CLUSTER
    //   |
    //   |    (    v3    v4)   ()   (v5     )
    //   |    (    v2    )            (     )
    //   |    (    v1    )               (v6)
    //   |    (          )
    //   |    (v0        )
    //   |
    //   |______________________________________

    unordered_map<size_t, size_t> clustering;
    unordered_map<size_t, vector<size_t> > clustering_inverse;

    // With 2 iterations, the example results in an empty cluster.
    kmeans_.Cluster(vectors_, 2, &centers, &clustering, &clustering_inverse);
    EXPECT_EQ(2, clustering_inverse.size());

    // But the implementation should handle the empty cluster!
    centers = {vectors_[0], vectors_[5], vectors_[6]};  // Reset centers.
    kmeans_.Cluster(vectors_, 10, &centers, &clustering, &clustering_inverse);
    EXPECT_EQ(3, clustering_inverse.size());
}

// Test class that provides a set of vectors for clustering.
class VectorsForClustering : public testing::Test {
protected:
    virtual void SetUp() {
	kmeans_.set_verbose(false);
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
	//       0 0.3           3     3.9                           9  9.6
	//------v0-v3------------v1----v4----------------------------v2--v5-----
	ordered_vectors_.push_back(v0);
	ordered_vectors_.push_back(v1);
	ordered_vectors_.push_back(v2);
	ordered_vectors_.push_back(v3);
	ordered_vectors_.push_back(v4);
	ordered_vectors_.push_back(v5);

	three_clusters_mean1_ = (v0 + v3)[0] / 2;
	three_clusters_mean2_ = (v1 + v4)[0] / 2;
	three_clusters_mean3_ = (v2 + v5)[0] / 2;

	// For agglomerative clustering, structures are disambiguated by the
	// "left < right" children ordering.
	// All 6 leaves:
	//                                 /\
	//                                /  \
	//                               /    \
	//                              /\    / \
	//                            v2 v5  /   \
	//                                  /\   /\
	//                                v0 v3 v1 v4
	all_leaves_paths_ = {"100", "110", "00", "101", "111", "01"};
	// 2 leaves without pruning:
	//                                 /\
	//                                /  \
	//                               /\   \
	//                             v4 /\   \
	//                              v3 /\   /\
	//                               v0 v1 v2 v5
	two_leaves_paths_ = {"0110", "0111", "10", "010", "00", "11"};
	// 2 leaves with pruning:
	//                                   /\
	//                      {v0,v1,v3,v4} {v2,v5}
	two_leaves_pruned_paths_ = {"0", "0", "1", "0", "0", "1"};
    }
    vector<Eigen::VectorXd> ordered_vectors_;
    AgglomerativeClustering agglomerative_;
    KMeans kmeans_;
    vector<string> all_leaves_paths_;
    vector<string> two_leaves_paths_;
    vector<string> two_leaves_pruned_paths_;
    double three_clusters_mean1_;
    double three_clusters_mean2_;
    double three_clusters_mean3_;

};

// Checks that 3-means converges to the optimum point given good centers (in 1
// iteration).
TEST_F(VectorsForClustering, KMeans3ClustersGoodCenters) {
    // Given centers
    //----[v0]-v3---------[v1]----v4------------------------------[v2]--v5------
    vector<Eigen::VectorXd> centers = {ordered_vectors_[0], ordered_vectors_[1],
				       ordered_vectors_[2]};
    // Iteration 1. OPTIMAL
    //----(v0-v3)---------(v1----v4)------------------------------(v2--v5)------
    //----v0[]v3-----------v1-[]-v4--------------------------------v2[]v5-------
    size_t num_iterations = 1;
    unordered_map<size_t, size_t> clustering;
    unordered_map<size_t, vector<size_t> > clustering_inverse;
    kmeans_.Cluster(ordered_vectors_, num_iterations, &centers, &clustering,
		    &clustering_inverse);
    EXPECT_EQ(clustering[0], clustering[3]);
    EXPECT_EQ(clustering[1], clustering[4]);
    EXPECT_EQ(clustering[2], clustering[5]);
    EXPECT_NEAR(centers[0][0], three_clusters_mean1_, 1e-15);
    EXPECT_NEAR(centers[1][0], three_clusters_mean2_, 1e-15);
    EXPECT_NEAR(centers[2][0], three_clusters_mean3_, 1e-15);
}

// Checks that 3-means (still) converges to the optimum point given bad centers
// (in 3 iterations).
TEST_F(VectorsForClustering, KMeans3ClustersBadCenters) {
    // Given centers
    //----[v0]-[v3]--------[v1]----v4-------------------------------v2--v5------
    vector<Eigen::VectorXd> centers = {ordered_vectors_[0], ordered_vectors_[3],
				       ordered_vectors_[1]};
    // Iteration 1.
    //----(v0)-(v3)--------(v1----v4--------------------------------v2--v5)-----
    //----[v0]-[v3]---------v1----v4------------[]------------------v2--v5------
    //
    // Iteration 2.
    //----(v0)-(v3----------v1)--(v4--------------------------------v2--v5)-----
    //----[v0]--v3----[]----v1----v4--------------------[]----------v2--v5)-----
    //
    // Iteration 3. OPTIMAL
    //----(v0--v3)---------(v1----v4)------------------------------(v2--v5)-----
    //-----v0[]v3-----------v1-[]-v4--------------------------------v2[]v5------
    size_t num_iterations = 3;
    unordered_map<size_t, size_t> clustering;
    unordered_map<size_t, vector<size_t> > clustering_inverse;
    kmeans_.Cluster(ordered_vectors_, num_iterations, &centers, &clustering,
		    &clustering_inverse);
    EXPECT_EQ(clustering[0], clustering[3]);
    EXPECT_EQ(clustering[1], clustering[4]);
    EXPECT_EQ(clustering[2], clustering[5]);
    EXPECT_NEAR(centers[0][0], three_clusters_mean1_, 1e-15);
    EXPECT_NEAR(centers[1][0], three_clusters_mean2_, 1e-15);
    EXPECT_NEAR(centers[2][0], three_clusters_mean3_, 1e-15);
}

// Checks agglomerative clustering with all 6 leaves.
TEST_F(VectorsForClustering, AgglomerativeAll6Leaves) {
    size_t num_leaf_clusters = numeric_limits<size_t>::max();
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < all_leaves_paths_.size(); ++i) {
	EXPECT_EQ(all_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering twice.
TEST_F(VectorsForClustering, AgglomerativeAll6LeavesTwice) {
    size_t num_leaf_clusters = numeric_limits<size_t>::max();
    double gamma1 = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							 num_leaf_clusters);
    // Cluster again.
    double gamma2 = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							 num_leaf_clusters);
    EXPECT_NEAR(gamma1, gamma2, 1e-15);
    for (size_t i = 0; i < all_leaves_paths_.size(); ++i) {
	EXPECT_EQ(all_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering with 2 leaf clusters without pruning.
TEST_F(VectorsForClustering, Agglomerative2LeavesNotPruned) {
    size_t num_leaf_clusters = 2;
    agglomerative_.set_prune(false);  // Do not prune.
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < two_leaves_pruned_paths_.size(); ++i) {
	EXPECT_EQ(two_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering with 2 leaf clusters with pruning.
TEST_F(VectorsForClustering, Agglomerative2LeavesPruned) {
    size_t num_leaf_clusters = 2;
    agglomerative_.set_prune(true);  // Prune.
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < two_leaves_pruned_paths_.size(); ++i) {
	EXPECT_EQ(two_leaves_pruned_paths_[i],
		  agglomerative_.path_from_root(i));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
