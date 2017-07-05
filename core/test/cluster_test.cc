// Author: Karl Stratos (me@karlstratos.com)
//
// Check the correctness of the clustering code.

#include "gtest/gtest.h"

#include <limits>
#include <random>
#include <thread>

#include "../cluster.h"

// Checks the correctness of k-means cost computation.
TEST(KMeansCosts, CheckCorrect) {
    vector<Eigen::VectorXd> vectors;
    size_t max_num_iterations = 1000;
    size_t num_threads = 4;
    size_t distance_type = 1;  // Manhattan distance
    bool verbose = false;

    Eigen::VectorXd v0(1);
    Eigen::VectorXd v1(1);
    Eigen::VectorXd v2(1);
    Eigen::VectorXd v3(1);
    Eigen::VectorXd v4(1);
    Eigen::VectorXd v5(1);
    v3 << 0.0;
    v5 << 1.0;
    v2 << 7.0;
    v0 << 10.0;
    v4 << 30.0;
    v1 << 31.0;

    //        0   1         7     10                                 30  31
    //------[v3]--v5-------v2----[v0]--------------------------------v4--v1-----
    vectors = {v0, v1, v2, v3, v4, v5};
    vector<Eigen::VectorXd> centers = {v3, v0};

    // If we use indices {0,2,3,5}, we shoud instead get
    //         0.5            8.5
    //------(v3---v5)-----(v2------v0)-------------------------------v4--v1-----
    vector<size_t> indices = {0, 2, 3, 5};
    vector<size_t> clustering;
    kmeans::cluster(vectors, indices, max_num_iterations, num_threads,
		    distance_type, verbose, &centers, &clustering);
    vector<double> costs;
    double total_cost = kmeans::compute_cost(vectors, indices, num_threads,
					     distance_type, centers, clustering,
					     &costs);
    EXPECT_NEAR(1.0, costs[0], 1e-15);  // 0.5 + 0.5 = 1
    EXPECT_NEAR(3.0, costs[1], 1e-15);  // 1.5 + 1.5 = 3
    EXPECT_NEAR(4, total_cost, 1e-15);
}

// Checks the correctness of k-means for clustering a strict subset of vectors.
TEST(KMeansOnStrictSubset, CheckClustering) {
    vector<Eigen::VectorXd> vectors;
    size_t max_num_iterations = 1000;
    size_t num_threads = 1;
    size_t distance_type = 0;  // Squared Euclidean distance
    bool verbose = false;

    Eigen::VectorXd v0(1);
    Eigen::VectorXd v1(1);
    Eigen::VectorXd v2(1);
    Eigen::VectorXd v3(1);
    Eigen::VectorXd v4(1);
    Eigen::VectorXd v5(1);
    v3 << 0.0;
    v5 << 1.0;
    v2 << 4.0;
    v0 << 5.0;
    v4 << 20.0;
    v1 << 21.0;

    //        0   1         4   5                                    20  21
    //------[v3]--v5------ v2--[v0]----------------------------------v4--v1-----
    vectors = {v0, v1, v2, v3, v4, v5};
    vector<Eigen::VectorXd> centers = {v3, v0};

    // If we use all indices, we should get
    //                2.5                                             20.5
    //------(v3---v5------v2---v0)----------------------------------(v4--v1)----
    vector<Eigen::VectorXd> centers_all(centers);
    vector<size_t> indices_all = {0, 1, 2, 3, 4, 5};
    vector<size_t> clustering_all;
    kmeans::cluster(vectors, indices_all, max_num_iterations, num_threads,
		    distance_type, verbose, &centers_all, &clustering_all);
    EXPECT_EQ(2, centers_all.size());
    EXPECT_EQ(6, clustering_all.size());
    EXPECT_NEAR(2.5, centers_all[0](0), 1e-15);  // Mean of cluster 1
    EXPECT_NEAR(20.5, centers_all[1](0), 1e-15);  // Mean of cluster 2
    EXPECT_EQ(0, clustering_all[3]);
    EXPECT_EQ(0, clustering_all[5]);
    EXPECT_EQ(0, clustering_all[2]);
    EXPECT_EQ(0, clustering_all[0]);
    EXPECT_EQ(1, clustering_all[4]);
    EXPECT_EQ(1, clustering_all[1]);

    // If we use indices {0,2,3,5}, we shoud instead get
    //         0.5          4.5
    //------(v3---v5)----(v2---v0)-----------------------------------v4--v1-----
    vector<Eigen::VectorXd> centers_subset(centers);
    vector<size_t> indices_subset = {0, 2, 3, 5};
    vector<size_t> clustering_subset;
    kmeans::cluster(vectors, indices_subset, max_num_iterations, num_threads,
		    distance_type, verbose, &centers_subset,
		    &clustering_subset);
    EXPECT_EQ(2, centers_subset.size());
    EXPECT_EQ(4, clustering_subset.size());
    EXPECT_NEAR(0.5, centers_subset[0](0), 1e-15);  // Mean of cluster 1
    EXPECT_NEAR(4.5, centers_subset[1](0), 1e-15);  // Mean of cluster 2
    EXPECT_EQ(0, clustering_subset[2]);  // indices_subset[2] -> v3
    EXPECT_EQ(0, clustering_subset[3]);  // indices_subset[3] -> v5
    EXPECT_EQ(1, clustering_subset[1]);  // indices_subset[1] -> v2
    EXPECT_EQ(1, clustering_subset[0]);  // indices_subset[0] -> v0
}

// Checks the correctness of k-means center selection from a strict subset of
// vectors.
TEST(KMeansOnStrictSubset, CheckCenterSelection) {
    size_t num_vectors = 100;
    size_t dim = 10;
    vector<Eigen::VectorXd> vectors;  // Random vectors
    vector<size_t> indices_even;
    for (size_t i = 0; i < num_vectors; ++i) {
	vectors.push_back(Eigen::VectorXd::Random(dim));
	if (i % 2 == 0) { indices_even.push_back(i); }
    }
    size_t num_centers = 5;
    size_t num_threads = 1;
    size_t distance_type = 0;
    size_t num_restarts = 10;

    // Check that odd indices are never selected as centers.
    vector<size_t> center_indices;

    // k-means++
    for (size_t restart_num = 0; restart_num < num_restarts; ++restart_num) {
	kmeans::select_center_indices(vectors, indices_even, num_centers, "pp",
				      num_threads, distance_type,
				      &center_indices);
	EXPECT_EQ(num_centers, center_indices.size());
	for (size_t j = 0; j < num_centers; ++j) {
	    EXPECT_TRUE(center_indices[j] % 2 == 0);
	}
    }

    // Uniform
    for (size_t restart_num = 0; restart_num < num_restarts; ++restart_num) {
	kmeans::select_center_indices(vectors, indices_even, num_centers,
				      "uniform", num_threads, distance_type,
				      &center_indices);
	EXPECT_EQ(num_centers, center_indices.size());
	for (size_t j = 0; j < num_centers; ++j) {
	    EXPECT_TRUE(center_indices[j] % 2 == 0);
	}
    }

    // Front
    kmeans::select_center_indices(vectors, indices_even, num_centers,
				  "front", num_threads, distance_type,
				  &center_indices);
    EXPECT_EQ(num_centers, center_indices.size());
    for (size_t j = 0; j < num_centers; ++j) {
	EXPECT_TRUE(center_indices[j] % 2 == 0);
    }
}

// Test class that provides a random set of vectors.
class RandomVectors : public testing::Test {
protected:
    virtual void SetUp() {
	for (size_t i = 0; i < num_vectors_; ++i) {
	    vectors_.push_back(Eigen::VectorXd::Random(dim_));
	    indices_.push_back(i);
	}
    }
    vector<Eigen::VectorXd> vectors_;
    vector<size_t> indices_;
    size_t num_vectors_ = 197;
    size_t dim_ = 10;
    size_t num_clusters_ = 10;
    size_t max_num_iterations_ = 1000;
    bool verbose_ = false;
};

// Checks the behavior of the k-means++ initialization under multithreading.
TEST_F(RandomVectors, KMeansPlusPlusInitialization) {
    string seed_method = "pp";
    size_t distance_type = 0;  // Any distance is fine (checked in other test).
    size_t num_threads = 4;
    size_t num_fewer_vectors = (num_vectors_ > 40) ? 40 : num_vectors_;
    vector<Eigen::VectorXd> fewer_vectors(num_fewer_vectors);
    vector<size_t> fewer_indices(num_fewer_vectors);
    for (size_t i = 0; i < num_fewer_vectors; ++i) {
	fewer_vectors[i] = vectors_[i];
	fewer_indices[i] = i;
    }
    vector<size_t> center_indices;

    // Draw as many centers as there are vectors.
    kmeans::select_center_indices(fewer_vectors, fewer_indices,
				  num_fewer_vectors, seed_method, num_threads,
				  distance_type, &center_indices);

    // Then every vector must be selected as a center under k-means++.
    unordered_map<size_t, bool> covered_indices;
    for (size_t index : center_indices) { covered_indices[index] = true; }
    EXPECT_EQ(num_fewer_vectors, covered_indices.size());
}

// Checks that k-means with restarts behaves correctly under multithreading.
TEST_F(RandomVectors, KMeansWithRestarts) {
    size_t distance_type = 0;  // Any distance is fine (checked in other test).
    vector<vector<Eigen::VectorXd> > list_centers;
    vector<vector<size_t> > list_clustering;
    vector<double> list_objective;

    // Vary numbers of restarts and threads and check with deterministic seeds.
    vector<pair<size_t, size_t> > pairs;
    pairs.push_back(make_pair(4, 4));  // Equal number of restarts and threads
    pairs.push_back(make_pair(7, 2));  // More restarts than threads
    pairs.push_back(make_pair(3, 11));  // More threads than restarts
    for (auto &restart_thread_pair : pairs) {
	size_t num_restarts = restart_thread_pair.first;
	size_t num_threads = restart_thread_pair.second;
	kmeans::cluster(vectors_, indices_, max_num_iterations_, num_threads,
			distance_type, num_clusters_, "front", num_restarts,
			&list_centers, &list_clustering, &list_objective);
	for (size_t r = 1; r < num_restarts; ++r) {
	    EXPECT_NEAR(list_objective[r - 1], list_objective[r], 1e-10);
	}
    }
}

// Checks that k-means behaves correctly under multithreading.
TEST_F(RandomVectors, KMeansEuclideanMultithreading) {
    // Use front vectors as initial centers.
    vector<size_t> center_indices;
    kmeans::select_center_indices(vectors_, indices_, num_clusters_, "front", 1,
				  0, &center_indices);
    vector<Eigen::VectorXd> centers;
    for (size_t center_index : center_indices) {
	centers.push_back(vectors_[center_index]);
    }

    vector<size_t> distance_types = {0, 1};  // Specify distance types.
    for (size_t distance_type : distance_types) {
	vector<size_t> clustering;
	vector<size_t> clustering_multi;

	vector<Eigen::VectorXd> centers_thread1(centers);
	double value_thread1 = kmeans::cluster(vectors_, indices_,
					       max_num_iterations_, 1,
					       distance_type, verbose_,
					       &centers_thread1, &clustering);

	vector<Eigen::VectorXd> centers_thread4(centers);
	double value_thread4 = kmeans::cluster(vectors_, indices_,
					       max_num_iterations_, 4,
					       distance_type, verbose_,
					       &centers_thread4,
					       &clustering_multi);

	EXPECT_NEAR(value_thread1, value_thread4, 1e-5);
	for (size_t i = 0; i < num_vectors_; ++i) {
	    EXPECT_EQ(clustering[i], clustering_multi[i]);
	}
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
	for (size_t i = 0; i < vectors_.size(); ++i) { indices_.push_back(i); }
    }
    vector<Eigen::VectorXd> vectors_;
    vector<size_t> indices_;
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
    kmeans::cluster(vectors_, indices_, 2, num_threads, distance_type, verbose,
		    &centers, &clustering);
    vector<vector<size_t> > clustering_inverse;
    kmeans::invert_clustering(clustering, &clustering_inverse);
    EXPECT_EQ(0, clustering_inverse[1].size());

    // But the implementation should handle the empty cluster!
    centers = {vectors_[0], vectors_[5], vectors_[6]};  // Reset centers.
    kmeans::cluster(vectors_, indices_, 10, num_threads, distance_type, verbose,
		    &centers, &clustering);
    kmeans::invert_clustering(clustering, &clustering_inverse);
    EXPECT_TRUE(clustering_inverse[1].size() > 0);
}

// Test class that provides a set of vectors for divisive clustering.
class VectorsDivisive : public testing::Test {
protected:
    virtual void SetUp() {
	Eigen::VectorXd v0(1);
	Eigen::VectorXd v1(1);
	Eigen::VectorXd v2(1);
	Eigen::VectorXd v3(1);
	Eigen::VectorXd v4(1);
	Eigen::VectorXd v5(1);
	v0 << 0.0;
	v3 << 1.0;
	v1 << 3.0;
	v4 << 5.0;
	v2 << 10.0;
	v5 << 14.0;
	//       0 1          3     5                             10     14
	//------v0-v3---------v1----v4----------------------------v2-----v5-----
	vectors_.push_back(v0);
	vectors_.push_back(v1);
	vectors_.push_back(v2);
	vectors_.push_back(v3);
	vectors_.push_back(v4);
	vectors_.push_back(v5);

	divisive_.set_max_num_iterations_kmeans(max_num_iterations_kmeans_);
	divisive_.set_num_threads(num_threads_);
	divisive_.set_distance_type(distance_type_);
	divisive_.set_seed_method(seed_method_);
	divisive_.set_num_restarts(num_restarts_);
	divisive_.set_verbose(false);
    }
    vector<Eigen::VectorXd> vectors_;
    DivisiveClustering divisive_;
    size_t max_num_iterations_kmeans_ = 100;
    size_t num_threads_ = 4;
    size_t distance_type_ = 0;  // Squared Euclidean distance
    string seed_method_ = "pp";  // k-means++ initialization
    size_t num_restarts_ = 4;
};

// Checks divisive clustering with all 6 leaves.
TEST_F(VectorsDivisive, DivisiveAll6Leaves) {
    unordered_map<string, vector<size_t> > leaves;
    double total_cost = divisive_.Cluster(vectors_, 10, &leaves);
    EXPECT_EQ(6, leaves.size());
    EXPECT_NEAR(0.0, total_cost, 1e-15);  // Singleton clusters

    unordered_map<size_t, string> path_from_root;
    divisive_.Invert(leaves, &path_from_root);
    //                                 /\
    //                                /  \
    //                               /    \
    //                              /\    / \
    //                            v2 v5  /   \
    //                                  /\   /\
    //                                v0 v3 v1 v4
    EXPECT_EQ(path_from_root[2].substr(0, 1), path_from_root[5].substr(0, 1));
    EXPECT_EQ(path_from_root[1].substr(0, 2), path_from_root[4].substr(0, 2));
    EXPECT_EQ(path_from_root[0].substr(0, 2), path_from_root[3].substr(0, 2));
}

// Checks divisive clustering with 3 leaves.
TEST_F(VectorsDivisive, Divisive3Leaves) {
    unordered_map<string, vector<size_t> > leaves;
    double total_cost = divisive_.Cluster(vectors_, 3, &leaves);
    EXPECT_EQ(3, leaves.size());
    EXPECT_NEAR(10.5, total_cost, 1e-15);  // 0.5 + 2 + 8 = 10.5

    unordered_map<size_t, string> path_from_root;
    divisive_.Invert(leaves, &path_from_root);
    //                                 /\
    //                                /  \
    //                               /    \
    //                           {v2 v5}  /\
    //                                   /   \
    //                               {v0 v3}{v1 v4}
    EXPECT_EQ(path_from_root[2], path_from_root[5]);
    EXPECT_EQ(path_from_root[1], path_from_root[4]);
    EXPECT_EQ(path_from_root[0], path_from_root[3]);
}

// Test class that provides a set of vectors for agglomerative clustering.
class VectorsAgglomerative : public testing::Test {
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

	agglomerative_.set_verbose(false);
    }
    vector<Eigen::VectorXd> ordered_vectors_;
    AgglomerativeClustering agglomerative_;
    vector<string> all_leaves_paths_;
    vector<string> two_leaves_paths_;
    vector<string> two_leaves_pruned_paths_;
};

// Checks agglomerative clustering with all 6 leaves.
TEST_F(VectorsAgglomerative, AgglomerativeAll6Leaves) {
    size_t num_leaf_clusters = numeric_limits<size_t>::max();
    double gamma = agglomerative_.ClusterOrderedVectors(ordered_vectors_,
							num_leaf_clusters);
    EXPECT_TRUE(gamma <= num_leaf_clusters);
    for (size_t i = 0; i < all_leaves_paths_.size(); ++i) {
	EXPECT_EQ(all_leaves_paths_[i], agglomerative_.path_from_root(i));
    }
}

// Checks agglomerative clustering twice.
TEST_F(VectorsAgglomerative, AgglomerativeAll6LeavesTwice) {
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
TEST_F(VectorsAgglomerative, Agglomerative2LeavesNotPruned) {
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
TEST_F(VectorsAgglomerative, Agglomerative2LeavesPruned) {
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
