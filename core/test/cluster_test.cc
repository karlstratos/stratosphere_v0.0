// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the clustering code.

#include "gtest/gtest.h"

#include <limits>
#include <random>
#include <thread>

#include "../cluster.h"

// Test class that provides a random set of vectors for clustering.
class RandomVectorsForClustering : public testing::Test {
protected:
    virtual void SetUp() {
	vector<Eigen::VectorXd> vectors(num_vectors_);
	for (size_t i = 0; i < num_vectors_; ++i) {
	    vectors_.push_back(Eigen::VectorXd::Random(dim_));
	}
    }
    vector<Eigen::VectorXd> vectors_;
    size_t num_vectors_ = 197;
    size_t dim_ = 10;
    size_t num_clusters_ = 10;
    size_t max_num_iterations_ = 1000;
    bool verbose_ = false;
};

// Checks that k-means with Euclidean distance behaves correctly under
// multithreading.
TEST_F(RandomVectorsForClustering, KMeansEuclideanMultithreading) {
    size_t distance_type = 0;  // Squared Euclidean distance
    vector<size_t> clustering;
    vector<size_t> clustering_multi;

    // Use front vectors as initial centers.
    vector<Eigen::VectorXd> centers;
    kmeans::select_centers(vectors_, num_clusters_, "front", &centers);
    double value_thread1 = kmeans::cluster(vectors_, max_num_iterations_,
					   1, distance_type, verbose_, &centers,
					   &clustering);  // 1 thread

    // Centers have been modified, reset.
    kmeans::select_centers(vectors_, num_clusters_, "front", &centers);
    double value_thread4 = kmeans::cluster(vectors_, max_num_iterations_,
					   4, distance_type, verbose_, &centers,
					   &clustering_multi);  // 4 threads

    EXPECT_NEAR(value_thread1, value_thread4, 1e-5);
    for (size_t i = 0; i < num_vectors_; ++i) {
	EXPECT_EQ(clustering[i], clustering_multi[i]);
    }
}

// Checks that k-means with Manhattan distance behaves correctly under
// multithreading.
TEST_F(RandomVectorsForClustering, KMeansManhattanMultithreading) {
    size_t distance_type = 1;  // Manhattan distance
    vector<size_t> clustering;
    vector<size_t> clustering_multi;

    // Use front vectors as initial centers.
    vector<Eigen::VectorXd> centers;
    kmeans::select_centers(vectors_, num_clusters_, "front", &centers);
    double value_thread1 = kmeans::cluster(vectors_, max_num_iterations_,
					   1, distance_type, verbose_, &centers,
					   &clustering);  // 1 thread

    // Centers have been modified, reset.
    kmeans::select_centers(vectors_, num_clusters_, "front", &centers);
    double value_thread4 = kmeans::cluster(vectors_, max_num_iterations_,
					   4, distance_type, verbose_, &centers,
					   &clustering_multi);  // 4 threads

    EXPECT_NEAR(value_thread1, value_thread4, 1e-5);
    for (size_t i = 0; i < num_vectors_; ++i) {
	EXPECT_EQ(clustering[i], clustering_multi[i]);
    }
}

// Test class that provides a k-means example with an empty cluster problem:
// http://www.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html
class KMeansEmptyClusterVectors : public testing::Test {
protected:
    virtual void SetUp() {
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
    vector<Eigen::VectorXd> vectors_;
};

// Checks that 3-means can result in an empty cluster.
TEST_F(KMeansEmptyClusterVectors, EmptyClusterWith3Means) {
    bool verbose = false;
    size_t num_threads = 1;
    size_t distance_type = 0;
    vector<size_t> clustering;

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
    // Iteration 2. EMPTY CLUSTER 1
    //   |
    //   |    (    v3    v4)   ()   (v5     )
    //   |    (    v2    )            (     )
    //   |    (    v1    )               (v6)
    //   |    (          )
    //   |    (v0        )
    //   |
    //   |______________________________________

    // With 2 iterations, the example results in an empty cluster.
    kmeans::cluster(vectors_, 2, num_threads, distance_type, verbose, &centers,
		    &clustering);
    vector<vector<size_t> > clustering_inverse;
    kmeans::invert_clustering(clustering, &clustering_inverse);
    EXPECT_EQ(0, clustering_inverse[1].size());

    // But the implementation should handle the empty cluster!
    centers = {vectors_[0], vectors_[5], vectors_[6]};  // Reset centers.
    kmeans::cluster(vectors_, 10, num_threads, distance_type, verbose, &centers,
		    &clustering);
    kmeans::invert_clustering(clustering, &clustering_inverse);
    EXPECT_TRUE(clustering_inverse[1].size() > 0);
}

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
	//       0 0.3           3     3.9                           9  9.6
	//------v0-v3------------v1----v4----------------------------v2--v5-----
	ordered_vectors_.push_back(v0);
	ordered_vectors_.push_back(v1);
	ordered_vectors_.push_back(v2);
	ordered_vectors_.push_back(v3);
	ordered_vectors_.push_back(v4);
	ordered_vectors_.push_back(v5);

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
    vector<string> all_leaves_paths_;
    vector<string> two_leaves_paths_;
    vector<string> two_leaves_pruned_paths_;
};

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
