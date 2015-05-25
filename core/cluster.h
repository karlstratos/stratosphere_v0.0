// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for clustering algorithms.

#ifndef CLUSTER_H
#define CLUSTER_H

#include <Eigen/Dense>
#include <unordered_map>

#include "util.h"

// Greedy, lazy, focused agglomerative clustering. By convention, we will denote
//        n := number of vectors to cluster
//        m := number of leaf clusters in the final hierarchy.
//    gamma := average number of clusters searched at each merge
//             (data-dependent constant upper bounded by m)
//
// RUNTIME OF THE ALGORITHM: O(gamma * n * m^2)
//
// CHARACTERISTICS OF THE ALGORITHM:
//    (Greedy) Greedily merges a pair of clusters (for n-1 times).
//      (Lazy) Updates pairwise distances only when they are needed.
//   (Focused) Considers at most m+1 clusters at each merge.
//
// This is a "focused" variant of the algorithm in: Fast and memory efficient
// implementation of the exact pnn (Franti et al., 2000).
class GreedyLazyFocusedAgglomerativeClustering {
public:
    // Clusters *ordered* vectors "from left to right". The leftmost vectors are
    // initialized as singleton clusters, and an additional vector is added as a
    // new cluster at each merge. Returns the value of gamma.
    double ClusterOrderedVectors(const vector<Eigen::VectorXd> &ordered_vectors,
				 size_t num_leaf_clusters);

    // Returns a pointer to the mapping from a leaf cluster to vectors:
    // "01001101" => {5, 973, 60}.
    unordered_map<string, vector<size_t> > *leaves() { return &leaves_; }

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

    // Total number of tightening operations performed because lowerbounds were
    // not tight.
    size_t num_extra_tightening_ = 0;

    // Mapping from a leaf cluster to vectors: "01001101" => {5, 973, 60}.
    unordered_map<string, vector<size_t> > leaves_;
};

#endif  // CLUSTER_H
