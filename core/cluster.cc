// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "cluster.h"

#include <cfloat>
#include <limits>
#include <stack>
#include <thread>

namespace kmeans {
    double cluster(const vector<Eigen::VectorXd> &vectors,
		   size_t max_num_iterations, size_t num_threads, bool verbose,
		   vector<Eigen::VectorXd> *centers,
		   vector<size_t> *clustering) {
	clustering->resize(vectors.size());

	// Infer the number of clusters from given centers.
	size_t num_clusters = centers->size();
	ASSERT(num_clusters > 0, "No initial centers given!");
	vector<size_t> cluster_sizes(num_clusters);

	// Prepare variables for multithreading.
	ASSERT(num_threads > 0, "Number of threads must be at least 1!");
	size_t num_vectors_per_worker = vectors.size() / num_threads;
	size_t extra_num_vectors_last_worker = vectors.size() % num_threads;
	vector<vector<size_t> > partial_cluster_sizes(num_threads);
	for (size_t t = 0; t < num_threads; ++t) {
	    partial_cluster_sizes[t].resize(num_clusters);
	}
	vector<double> partial_objective(num_threads);
	vector<vector<Eigen::VectorXd> > partial_centers(num_threads);
	for (size_t t = 0; t < num_threads; ++t) {
	    partial_centers[t].resize(num_clusters);
	    for (size_t j = 0; j < num_clusters; ++j) {
		partial_centers[t][j] =
		    Eigen::VectorXd::Zero(vectors.at(0).size());
	    }
	}

	// Start the k-means loop.
	double old_objective = numeric_limits<double>::infinity();
	double new_objective = numeric_limits<double>::infinity();
	for (size_t iteration_num = 1; iteration_num <= max_num_iterations;
	     ++iteration_num) {
	    if (verbose) { cerr << "Iteration " << iteration_num << ":\t"; }

	    // STEP 1: Assign vectors to their closest centers. Each worker
	    //         accumulates necessary quantities in parallel.
	    for (size_t t = 0; t < num_threads; ++t) {
		fill(partial_cluster_sizes[t].begin(),
		     partial_cluster_sizes[t].end(), 0);
	    }
	    fill(partial_objective.begin(), partial_objective.end(), 0.0);

	    // Lambda function for assigning vectors [start ... end).
	    auto cluster_code = [&vectors, &centers, &clustering,
				 &partial_objective, &partial_cluster_sizes]
		(size_t start, size_t end, size_t thread_num) -> void {
		for (size_t i = start; i < end; ++i) {
		    double min_dist = numeric_limits<double>::infinity();
		    size_t closest_center_index = 0;
		    for (size_t j = 0; j < centers->size(); ++j) {
			Eigen::VectorXd diff = vectors.at(i) - centers->at(j);
			double dist = diff.squaredNorm();
			if (dist < min_dist) {
			    min_dist = dist;
			    closest_center_index = j;
			}
		    }
		    (*clustering)[i] = closest_center_index;
		    ++partial_cluster_sizes[thread_num][closest_center_index];
		    partial_objective[thread_num] += min_dist;
		}
	    };

	    size_t start = 0;
	    size_t end = num_vectors_per_worker;
	    vector<thread> workers;
	    for (size_t t = 0; t < num_threads; ++t) {
		if (t == num_threads - 1) {  // Last worker does extra work.
		    end += extra_num_vectors_last_worker;
		}
		workers.push_back(thread(cluster_code, start, end, t));
		start = end;
		end += num_vectors_per_worker;
	    }
	    for (thread &worker : workers) { worker.join(); }
	    workers.clear();

	    // Sum over workers to get global cluster sizes and objective.
	    fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
	    new_objective = 0.0;
	    for (size_t t = 0; t < num_threads; ++t) {
		for (size_t j = 0; j < num_clusters; ++j) {
		    cluster_sizes[j] += partial_cluster_sizes[t][j];
		}
		new_objective += partial_objective[t];
	    }

	    // Check if there exist centers that yielded empty clusters.
	    vector<size_t> lone_center_indices;
	    for (size_t j = 0; j < cluster_sizes.size(); ++j) {
		if (cluster_sizes[j] == 0) { lone_center_indices.push_back(j); }
	    }
	    if (lone_center_indices.size() > 0) {  // Hopefully not often.
		// Reset each lone center with a random point.
		vector<size_t> permuted_indices;
		util_math::permute_indices(vectors.size(), &permuted_indices);
		for (size_t j : lone_center_indices) {
		    if (verbose) {
			cerr << "Center " << j << " yielded an empty cluster! "
			     << "Replacing it with point "
			     << permuted_indices[j] << endl;
		    }
		    (*centers)[j] = vectors.at(permuted_indices[j]);
		}
		continue;  // Optimize clusters again (counted as an iteration).
	    }
	    if (verbose) { cerr << new_objective << "\t"; }

	    // STEP 2: Calculate cluster means for new centers in parallel.
	    for (size_t j = 0; j < num_clusters; ++j) {
		(*centers)[j] = Eigen::VectorXd::Zero(vectors.at(0).size());
	    }
	    for (size_t t = 0; t < num_threads; ++t) {
		for (size_t j = 0; j < num_clusters; ++j) {
		    partial_centers[t][j].setZero();
		}
	    }

	    // Lambda function for accumulating vectors [start ... end).
	    auto center_code = [&vectors, &clustering, &partial_centers]
		(size_t start, size_t end, size_t thread_num) -> void {
		for (size_t i = start; i < end; ++i) {
		    partial_centers[thread_num][clustering->at(i)] +=
		    vectors.at(i);
		}
	    };

	    start = 0;
	    end = num_vectors_per_worker;
	    for (size_t t = 0; t < num_threads; ++t) {
		if (t == num_threads - 1) {  // Last worker does extra work.
		    end += extra_num_vectors_last_worker;
		}
		workers.push_back(thread(center_code, start, end, t));
		start = end;
		end += num_vectors_per_worker;
	    }
	    for (thread &worker : workers) { worker.join(); }
	    workers.clear();

	    // Sum over workers to get global cluster means (new centers).
	    for (size_t t = 0; t < num_threads; ++t) {
		for (size_t j = 0; j < num_clusters; ++j) {
		    (*centers)[j] += partial_centers[t][j] / cluster_sizes[j];
		}
	    }

	    // Check if converged.
	    if (old_objective - new_objective < 1e-15) {
		if (verbose) { cerr << " CONVERGED" << endl; }
		break;
	    }
	    old_objective = new_objective;
	    if (verbose) { cerr << endl; }
	}

	return new_objective;
    }

    void select_centers(const vector<Eigen::VectorXd> &vectors,
			size_t num_centers, const string &select_method,
			vector<Eigen::VectorXd> *centers) {
	centers->resize(num_centers);
	if (select_method == "rand") {  // Uniform sampling.
	    vector<size_t> permuted_indices;
	    util_math::permute_indices(vectors.size(), &permuted_indices);

	    // Copy vectors corresponding to the top k shuffled indices.
	    for (size_t j = 0; j < num_centers; ++j) {
		(*centers)[j] = vectors.at(permuted_indices[j]);
	    }
	} else {
	    ASSERT(false, "Unknown selection method: " << select_method);
	}
    }

    void invert_clustering(const vector<size_t> &clustering,
			   vector<vector<size_t> > *clustering_inverse) {
	for (size_t i = 0; i < clustering.size(); ++i) {
	    size_t cluster_num = clustering.at(i);
	    if (cluster_num >= clustering_inverse->size()) {
		clustering_inverse->resize(cluster_num + 1);
	    }
	    (*clustering_inverse)[cluster_num].push_back(i);
	}
    }
}  // namespace kmeans

double AgglomerativeClustering::ClusterOrderedVectors(
    const vector<Eigen::VectorXd> &ordered_vectors, size_t num_leaf_clusters) {
    size_t n = ordered_vectors.size();
    size_t m = (num_leaf_clusters <= n) ? num_leaf_clusters : n;

    //--------------------------------------------------------------------------
    // (Sketch of the algorithm)
    // We compute an ordered list of 2n-1 clusters (n original vectors as
    // singletons + n-1 merges):
    //    [Singletons]       0     1     2   ...    n-2   n-1
    //    [Non-singletons]   n   n+1   n+2   ...   2n-2
    // Normally in agglomerative clustering, we would start with the first n
    // singleton clusters and repeatedly merge: with the Franti et al. (2000)
    // trick, this would take O(gdn^2) where g is a data-dependent constant
    // and d is the dimension of the vector space.
    //
    // Here, we will instead start with the first m singleton clusters and
    // repeatedly merge. At every iteration, we will include the next singleton
    // cluster for consideration. Thus in any given moment, we handle at most
    // m+1 "active" clusters. Again applying the Franti et al. (2000) trick,
    // this now takes O(gdmn).
    //--------------------------------------------------------------------------

    Z_.resize(n - 1);  // Information about the n-1 merges.
    size_.resize(2 * n - 1);  // Clusters' sizes.
    active_.resize(m + 1);  // Active clusters.
    mean_.resize(m + 1);  // Clusters' means.
    lb_.resize(m + 1);  // Lowerbounds.
    twin_.resize(m + 1);  // Indices in {0 ... m} for merge candidates.
    tight_.resize(m + 1);  // Is the current lowerbound tight?
    size_t num_extra_tightening = 0;  // Number of tightening operations.

    // Initialize the first m clusters.
    for (size_t a1 = 0; a1 < m; ++a1) {  // Tightening m clusters: O(dm^2).
	size_[a1] = 1;
	active_[a1] = a1;
	mean_[a1] = ordered_vectors[a1];
	lb_[a1] = DBL_MAX;
	for (size_t a2 = 0; a2 < a1; ++a2) {
	    double dist = ComputeDistance(ordered_vectors, a1, a2);
	    UpdateLowerbounds(a1, a2, dist);
	}
    }

    // Main loop: Perform n-1 merges.
    size_t next_singleton = m;
    for (size_t merge_num = 0; merge_num < n - 1; ++merge_num) {
	if (next_singleton < n) {
	    // Set the next remaining vector as the (m+1)-th active cluster.
	    size_[next_singleton] = 1;
	    active_[m] = next_singleton;
	    mean_[m] = ordered_vectors[next_singleton];
	    lb_[m] = DBL_MAX;
	    for (int a = 0; a < m; ++a) {  // Tightening 1 cluster: O(dm).
		double dist = ComputeDistance(ordered_vectors, m, a);
		UpdateLowerbounds(m, a, dist);
	    }
	    ++next_singleton;
	}

	// Number of active clusters is m+1 until all singleton clusters are
	// active. Then it decreases m ... 2.
	size_t num_active_clusters = min(m + 1, n - merge_num);

	// Find which active cluster has the smallest lowerbound: O(m).
	size_t candidate_index = 0;
	double smallest_lowerbound = DBL_MAX;
	for (size_t a = 0; a < num_active_clusters; ++a) {
	    if (lb_[a] < smallest_lowerbound) {
		smallest_lowerbound = lb_[a];
		candidate_index = a;
	    }
	}

	while (!tight_[candidate_index]) {
	    // The current candidate turns out to have a loose lowerbound.
	    // Tighten it: O(dm).
	    lb_[candidate_index] = DBL_MAX;  // Recompute lowerbound.
	    for (size_t a = 0; a < num_active_clusters; ++a) {
		if (a == candidate_index) continue;  // Skip self.
		double dist =
		    ComputeDistance(ordered_vectors, candidate_index, a);
		UpdateLowerbounds(candidate_index, a, dist);
	    }
	    ++num_extra_tightening;

	    // Again, find an active cluster with the smallest lowerbound: O(m).
	    smallest_lowerbound = DBL_MAX;
	    for (size_t a = 0; a < num_active_clusters; ++a) {
		if (lb_[a] < smallest_lowerbound) {
		    smallest_lowerbound = lb_[a];
		    candidate_index = a;
		}
	    }
	}

	// At this point, we have a pair of active clusters with minimum
	// pairwise distance. Denote their active indices by "alpha" and "beta".
	size_t alpha = candidate_index;
	size_t beta = twin_[alpha];
	if (alpha > beta) {  // WLOG, we will maintain alpha < beta.
	    size_t temp = alpha;
	    alpha = beta;
	    beta = temp;
	}

	// Cluster whose twin was in {alpha, beta} has a loose lowerbound.
	for (size_t a = 0; a < num_active_clusters; ++a) {
	    if (twin_[a] == alpha || twin_[a] == beta) { tight_[a] = false; }
	}

	// Record the merge in Z_.
	size_t merged_cluster = n + merge_num;
	get<0>(Z_[merge_num]) = active_[alpha];
	get<1>(Z_[merge_num]) = active_[beta];
	get<2>(Z_[merge_num]) = smallest_lowerbound;

	// Compute the size of the merged cluster.
	size_[merged_cluster] = size_[active_[alpha]] + size_[active_[beta]];

	// MUST compute the merge mean before modifying active clusters!
	ComputeMergedMean(ordered_vectors, alpha, beta, &mean_[alpha]);

	//----------------------------------------------------------------------
	// SHIFTING (Recall: alpha < beta)
	// We now replace the (alpha)-th active cluster with the new merged
	// cluster. Then we will shift active clusters past index beta to the
	// left by one position to overwrite beta. Graphically speaking, the
	// current M <= m+1 active clusters will change in structure (1 element
	// shorter) as follows:
	//
	//     a_1   ...   alpha        ...  a   b   beta   c   d   ...   a_M
	// =>
	//     a_1   ...   alpha+beta   ...  a   b   c   d   ...   a_M
	//----------------------------------------------------------------------

	// Set the merged cluster as the (alpha)-th active cluster and tighten.
	active_[alpha] = merged_cluster;
	lb_[alpha] = DBL_MAX;
	for (size_t a = 0; a < num_active_clusters; ++a) {
	    if (a == alpha) continue;  // Skip self.
	    if (a == beta) continue;  // beta will be overwritten anyway.
	    double dist = ComputeDistance(ordered_vectors, alpha, a);
	    UpdateLowerbounds(alpha, a, dist);
	}

	// Shift the elements past beta to the left by one (overwriting beta).
	for (size_t a = 0; a < num_active_clusters - 1; ++a) {
	    if (a < beta && twin_[a] > beta) {
		// Even for non-shifting elements, if their twin index is
		// greater than beta, we must shift accordingly.
		twin_[a] = twin_[a] - 1;
	    }

	    if (a >= beta) {
		active_[a] = active_[a + 1];
		mean_[a] = mean_[a + 1];
		lb_[a] = lb_[a + 1];
		tight_[a] = tight_[a + 1];

		if (twin_[a + 1] < beta) {
		    twin_[a] = twin_[a + 1];
		} else {  // Again, need to shift twin indices accordingly.
		    twin_[a] = twin_[a + 1] - 1;
		}
	    }
	    ASSERT(a != twin_[a], "Active index " << a << " has itself for "
		   "twin: something got screwed while shifting");
	}
    }

    // Order the left and right children.
    for (size_t i = 0; i < n - 1; ++i) {
	if (get<0>(Z_[i]) > get<1>(Z_[i])) {
	    double temp = get<0>(Z_[i]);
	    get<0>(Z_[i]) = get<1>(Z_[i]);
	    get<1>(Z_[i]) = temp;
	}
    }
    LabelLeaves(m);  // Clustering done: label bit strings.
    double gamma = ((double) num_extra_tightening) / (n - 1);
    return gamma;
}

string AgglomerativeClustering::path_from_root(size_t vector_index) {
    ASSERT(vector_index < path_from_root_.size(), "No index: " << vector_index);
    return path_from_root_[vector_index];
}

double AgglomerativeClustering::ComputeDistance(
    const vector<Eigen::VectorXd> &ordered_vectors, size_t active_index1,
    size_t active_index2) {
    size_t size1 = size_[active_[active_index1]];
    size_t size2 = size_[active_[active_index2]];
    double scale = 2.0 * size1 * size2 / (size1 + size2);
    Eigen::VectorXd diff = mean_[active_index1] - mean_[active_index2];
    return scale * diff.squaredNorm();
}

void AgglomerativeClustering::UpdateLowerbounds(size_t active_index1,
						size_t active_index2,
						double distance) {
    if (distance < lb_[active_index1]) {
	lb_[active_index1] = distance;
	twin_[active_index1] = active_index2;
	tight_[active_index1] = true;
    }
    if (distance < lb_[active_index2]) {
	lb_[active_index2] = distance;
	twin_[active_index2] = active_index1;
	tight_[active_index2] = true;
    }
}

void AgglomerativeClustering::ComputeMergedMean(
    const vector<Eigen::VectorXd> &ordered_vectors, size_t active_index1,
    size_t active_index2, Eigen::VectorXd *new_mean) {
    double size1 = size_[active_[active_index1]];
    double size2 = size_[active_[active_index2]];
    double total_size = size1 + size2;
    double scale1 = size1 / total_size;
    double scale2 = size2 / total_size;
    *new_mean = scale1 * mean_[active_index1] + scale2 * mean_[active_index2];
}

void AgglomerativeClustering::LabelLeaves(size_t num_leaf_clusters) {
    ASSERT(Z_.size() > 0, "No merge information to label leaves!");
    ASSERT(active_.size() > 0, "Active clusters missing!");
    leaves_.clear();
    path_from_root_.clear();
    size_t n = Z_.size() + 1;

    // Use breadth-first search (BFS) to traverse the tree. Maintain bit strings
    // to mark the path from the root.
    stack<pair<size_t, string> > bfs_stack;  // [  ... (77, "10011") ]

    // Push the root cluster (2n-2) with an empty bit string.
    bfs_stack.push(make_pair(2 * n - 2, ""));

    while(!bfs_stack.empty()){
        std::pair<size_t,string> cluster_bitstring_pair = bfs_stack.top();
        bfs_stack.pop();
        size_t cluster = cluster_bitstring_pair.first;
        string bitstring = cluster_bitstring_pair.second;

        if (cluster < n) {
	    // We have a leaf cluster. Add to the current bit string.
	    leaves_[bitstring].push_back(cluster);
	    path_from_root_[cluster] = bitstring;
	} else {
	    // We have a non-leaf cluster. Branch to its two children.
            size_t left_child_cluster = get<0>(Z_[cluster - n]);
            size_t right_child_cluster = get<1>(Z_[cluster - n]);

            string left_bitstring = bitstring;
            string right_bitstring = bitstring;

            if (!prune_ || cluster >= 2 * n - num_leaf_clusters) {
		// Prune branches to have m leaf clusters.
                left_bitstring += "0";
                right_bitstring += "1";
            }
            bfs_stack.push(make_pair(left_child_cluster, left_bitstring));
            bfs_stack.push(make_pair(right_child_cluster, right_bitstring));
        }
    }
}
