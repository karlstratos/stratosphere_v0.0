// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for clustering algorithms. References:
//
// - Fast and memory efficient implementation of the exact pnn (Franti et al.,
//   2000).

#ifndef CLUSTER_H
#define CLUSTER_H

#include <Eigen/Dense>
#include <unordered_map>

#include "util.h"

namespace kmeans {
    // Runs k-means on n vectors of length d for T iterations, returns the
    // final objective value. Calculates:
    //    - centers[j]   : mean of cluster j (initialized from given centers)
    //    - clustering[i]: cluster of vector i (an index in {1...k})
    //
    // The code is optimized for large n and small k.
    //    * RUNTIME: O(Tndk / num_threads)
    //    * MEMORY: O(ndk)
    double cluster(const vector<Eigen::VectorXd> &vectors,
		   size_t max_num_iterations, size_t num_threads, bool verbose,
		   vector<Eigen::VectorXd> *centers,
		   vector<size_t> *clustering);

    // Selects centers from given vectors.
    void select_centers(const vector<Eigen::VectorXd> &vectors,
			size_t num_centers, const string &select_method,
			vector<Eigen::VectorXd> *centers);

    // Inverts the mapping vector->cluster to the mapping cluster->{vectors}.
    void invert_clustering(const vector<size_t> &clustering,
			   vector<vector<size_t> > *clustering_inverse);
}

// Class for agglomerative clustering. Since complex index manipulation is
// needed, we will have the following convention.
//        n := number of vectors to cluster
//        m := number of leaf clusters in the final hierarchy (m <= n)
//    gamma := average number of clusters searched at each merge
//             (data-dependent constant upper bounded by m)
//
// RUNTIME OF THE ALGORITHM: O(gamma * n * m^2)
class AgglomerativeClustering {
public:
    // Clusters ordered vectors. The leftmost vectors are initialized as
    // singleton clusters, and an additional vector is added as a new cluster
    // at each merge left to right. Returns the value of gamma.
    double ClusterOrderedVectors(const vector<Eigen::VectorXd> &ordered_vectors,
				 size_t num_leaf_clusters);

    // Returns a pointer to the mapping from a leaf cluster to vectors:
    // "01001101" => {5, 973, 60}.
    unordered_map<string, vector<size_t> > *leaves() { return &leaves_; }

    // Returns the path from the root for a clustered vector (as bits):
    // 973 => "01001101".
    string path_from_root(size_t vector_index);

    // Sets whether to prune the hierarchy (have only m leaf clusters, not n).
    void set_prune(bool prune) { prune_ = prune; }

private:
    // Computes the distance between two active clusters.
    double ComputeDistance(const vector<Eigen::VectorXd> &ordered_vectors,
			   size_t active_index1, size_t active_index2);

    // Updates two active clusters' lowerbounds/twins given their distance.
    void UpdateLowerbounds(size_t active_index1, size_t active_index2,
			   double distance);

    // Computes the new mean resulting from merging two active clusters.
    void ComputeMergedMean(const vector<Eigen::VectorXd> &ordered_vectors,
			   size_t active_index1, size_t active_index2,
			   Eigen::VectorXd *new_mean);

    // Based on the computed hierarchy, create a mapping from a leaf-node bit
    // string indicating the path from the root to the associated clusters.
    //                    ...
    //                   /  \
    //                1010  1011
    //             {0,3,9}   {77,1,8}
    void LabelLeaves(size_t num_leaf_clusters);

    // Information of the n-1 merges. For i in {0 ... n-2}:
    //    get<0>(Z_[i]) = left child of cluster n+i
    //    get<1>(Z_[i]) = right child of cluster n+i
    //    get<2>(Z_[i]) = distance between children for cluster n+i
    // Merges are ordered so that get<0>(Z_[i]) < get<1>(Z_[i]).
    vector<tuple<size_t, size_t, double> > Z_;

    // For c = 0 ... 2n-2:
    //    size_[c] = number of elements in cluster c.
    vector<size_t> size_;

    // For i = 0 ... m:
    //    active_[i] = i-th active cluster, an element in {0 ... 2n-2}.
    vector<size_t> active_;

    // For i = 0 ... m:
    //    mean_[i] = mean of the i-th active cluster.
    vector<Eigen::VectorXd> mean_;

    // For i = 0 ... m:
    //    lb_[i] = lowerbound on the distance from the i-th active cluster to
    //             any other active cluster.
    vector<double> lb_;

    // For i = 0 ... m:
    //    twin_[i] = index in {0 ... m}\{i} that indicates which active cluster
    //               is estimated as the nearest to the i-th active cluster.
    vector<size_t> twin_;

    // For i = 0 ... m:
    //    tight_[i] = true if lb_[i] is tight.
    vector<bool> tight_;

    // Prune the hierarchy (have only m leaf clusters, not n)?
    bool prune_ = true;

    // Mapping from a leaf cluster to vectors: "01001101" => {5, 973, 60}.
    unordered_map<string, vector<size_t> > leaves_;

    // Mapping from a vector index to the path from the root (as bits).
    unordered_map<size_t, string> path_from_root_;
};

#endif  // CLUSTER_H
