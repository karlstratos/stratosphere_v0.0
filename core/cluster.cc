// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "cluster.h"

#include <cfloat>
#include <limits>
#include <stack>
#include <thread>
#include <queue>
#include <random>

namespace kmeans {
    double compute_distance(const Eigen::VectorXd &v1,
			    const Eigen::VectorXd &v2, size_t distance_type) {
	double dist;
	switch (distance_type) {
	case 0: {  // Squared Euclidean distance
	    dist = (v1 - v2).squaredNorm();
	    break;
	}
	case 1: {  // Manhattan distance
	    dist = (v1 - v2).lpNorm<1>();
	    break;
	}
	default: {
	    ASSERT(false, "Unknown dist: " << distance_type);
	}
	}
	return dist;
    }

    double compute_cost(const vector<Eigen::VectorXd> &vectors,
			const vector<size_t> &indices,
			size_t num_threads, size_t distance_type,
			const vector<Eigen::VectorXd> &centers,
			const vector<size_t> &clustering,
			vector<double> *costs) {
	costs->resize(centers.size());

	// Prepare variables for multithreading.
	ASSERT(num_threads > 0, "Number of threads must be at least 1!");
	size_t num_vectors_per_worker = indices.size() / num_threads;
	size_t extra_num_vectors_last_worker = indices.size() % num_threads;
	vector<vector<double> > partial_costs(num_threads);
	for (size_t t = 0; t < num_threads; ++t) {
	    partial_costs[t].resize(centers.size());
	}

	// Lambda function for handling vectors(indices.at(i)) for i in
	// [start ... end).
	auto code = [&vectors, &indices, distance_type, &centers,
		     &clustering, &partial_costs]
	    (size_t start, size_t end, size_t thread_num) -> void {
	    for (size_t i = start; i < end; ++i) {
		size_t j = clustering.at(i);
		partial_costs[thread_num][j] +=
		compute_distance(vectors.at(indices.at(i)), centers.at(j),
				 distance_type);
	    }
	};

	size_t start = 0;
	size_t end = num_vectors_per_worker;
	vector<thread> workers;
	for (size_t t = 0; t < num_threads; ++t) {
	    if (t == num_threads - 1) {  // Last worker does extra work.
		end += extra_num_vectors_last_worker;
	    }
	    workers.push_back(thread(code, start, end, t));
	    start = end;
	    end += num_vectors_per_worker;
	}
	for (thread &worker : workers) { worker.join(); }
	workers.clear();

	// Sum over workers to get global costs.
	fill(costs->begin(), costs->end(), 0.0);
	double total_cost = 0.0;
	for (size_t t = 0; t < num_threads; ++t) {
	    for (size_t j = 0; j < centers.size(); ++j) {
		(*costs)[j] += partial_costs[t][j];
		total_cost += partial_costs[t][j];
	    }
	}
	return total_cost;
    }

    double cluster(const vector<Eigen::VectorXd> &vectors,
		   const vector<size_t> &indices,
		   size_t max_num_iterations, size_t num_threads,
		   size_t distance_type, bool verbose,
		   vector<Eigen::VectorXd> *centers,
		   vector<size_t> *clustering) {
	// clustering[i] = cluster of vectors.at(indices.at(i))
	clustering->resize(indices.size());
	size_t dim = vectors.at(indices.at(0)).size();

	// Infer the number of clusters from given centers.
	size_t num_clusters = centers->size();
	ASSERT(num_clusters > 0, "No initial centers given!");
	vector<size_t> cluster_sizes(num_clusters);

	// Prepare variables for multithreading.
	ASSERT(num_threads > 0, "Number of threads must be at least 1!");
	size_t num_vectors_per_worker = indices.size() / num_threads;
	size_t num_clusters_per_worker = num_clusters / num_threads;
	size_t extra_num_vectors_last_worker = indices.size() % num_threads;
	size_t extra_num_clusters_last_worker = num_clusters % num_threads;
	vector<vector<size_t> > partial_cluster_sizes(num_threads);
	for (size_t t = 0; t < num_threads; ++t) {
	    partial_cluster_sizes[t].resize(num_clusters);
	}
	vector<double> partial_objective(num_threads);
	vector<vector<Eigen::VectorXd> > partial_centers(num_threads);
	for (size_t t = 0; t < num_threads; ++t) {
	    partial_centers[t].resize(num_clusters);
	    for (size_t j = 0; j < num_clusters; ++j) {
		partial_centers[t][j] = Eigen::VectorXd::Zero(dim);
	    }
	}

	// Start the k-means loop.
	bool converged = false;
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

	    // Lambda function for assigning vectors(indices.at(i)) for i in
	    // [start ... end).
	    auto cluster_code = [&vectors, &indices, distance_type, &centers,
				 &clustering, &partial_objective,
				 &partial_cluster_sizes]
		(size_t start, size_t end, size_t thread_num) -> void {
		for (size_t i = start; i < end; ++i) {
		    double min_dist = numeric_limits<double>::infinity();
		    size_t closest_center_index = 0;
		    for (size_t j = 0; j < centers->size(); ++j) {
			double dist =
			    compute_distance(vectors.at(indices.at(i)),
					     centers->at(j), distance_type);
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
		vector<size_t> permuted_indices(indices);
		size_t seed =
		    chrono::system_clock::now().time_since_epoch().count();
		shuffle(permuted_indices.begin(), permuted_indices.end(),
			default_random_engine(seed));
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

	    // STEP 2: Calculate new centers.
	    switch (distance_type) {
	    case 0: {  // Squared Euclidean distance
		// Accumulate cluster means in parallel.
		for (size_t j = 0; j < num_clusters; ++j) {
		    (*centers)[j] = Eigen::VectorXd::Zero(dim);
		}
		for (size_t t = 0; t < num_threads; ++t) {
		    for (size_t j = 0; j < num_clusters; ++j) {
			partial_centers[t][j].setZero();
		    }
		}

		// Lambda function for accumulating vectors.at(indices.at(i))
		// for i in [start ... end).
		auto center_code = [&vectors, &indices, &clustering,
				    &partial_centers]
		    (size_t start, size_t end, size_t thread_num) -> void {
		    for (size_t i = start; i < end; ++i) {
			partial_centers[thread_num][clustering->at(i)] +=
			vectors.at(indices.at(i));
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
			(*centers)[j] +=
			    partial_centers[t][j] / cluster_sizes[j];
		    }
		}
		break;
	    }
	    case 1: {  // Manhattan distance
		// In each cluster, sort vectors dimension-wise.
		vector<vector<vector<double> > > values(num_clusters);
		for (size_t j = 0; j < num_clusters; ++j) {
		    values[j].resize(dim);
		}
		for (size_t i = 0; i < indices.size(); ++i) {  // O(#vectors)...
		    for (size_t d = 0; d < dim; ++d) {
			// values[j][d][z] = d-th value of (indices.at(z))-th
			//                   vector in cluster j
			values[clustering->at(i)][d].push_back(
			    vectors.at(indices.at(i))(d));
		    }
		}

		// Lambda function for sorting clusters [start ... end).
		auto sort_code = [&values, dim, &centers]
		    (size_t start, size_t end) -> void {
		    for (size_t j = start; j < end; ++j) {
			size_t cluster_size = values[j][0].size();  // Know > 0.
			size_t middle = (cluster_size - 1) / 2;
			for (size_t d = 0; d < dim; ++d) {
			    sort(values[j][d].begin(), values[j][d].end());
			    double median_value = (cluster_size % 2 == 1) ?
				values[j][d][middle] :
				(values[j][d][middle] +
				 values[j][d][middle + 1]) / 2;
			    (*centers)[j](d) = median_value;
			}
		    }
		};

		start = 0;
		end = num_clusters_per_worker;
		for (size_t t = 0; t < num_threads; ++t) {
		    if (t == num_threads - 1) {  // Last worker does extra work.
			end += extra_num_clusters_last_worker;
		    }
		    workers.push_back(thread(sort_code, start, end));
		    start = end;
		    end += num_clusters_per_worker;
		}
		for (thread &worker : workers) { worker.join(); }
		workers.clear();

		break;
	    }
	    default: {
		ASSERT(false, "Unknown dist: " << distance_type);
	    }
	    }

	    // Check if converged.
	    ASSERT(old_objective - new_objective > -1e-32,
		   "Clustering objective increased, something is wrong!");
	    if (old_objective - new_objective < 1e-15) {
		if (verbose) { cerr << " CONVERGED" << endl; }
		converged = true;
		break;
	    }
	    old_objective = new_objective;
	    if (verbose) { cerr << endl; }
	}

	vector<double> costs;
	double final_objective = (converged) ?
	    new_objective : compute_cost(vectors, indices, num_threads,
					 distance_type, *centers, *clustering,
					 &costs);
	return final_objective;
    }

    void select_center_indices(const vector<Eigen::VectorXd> &vectors,
			       const vector<size_t> &indices,
			       size_t num_centers, const string &seed_method,
			       size_t num_threads, size_t distance_type,
			       vector<size_t> *center_indices) {
	ASSERT(indices.size() >= num_centers, "More centers than vectors?");
	center_indices->resize(num_centers);
	if (seed_method == "pp") {  // k-means++
	    random_device rd;
	    mt19937 gen(rd());
	    ASSERT(num_threads > 0, "Number of threads must be at least 1!");
	    size_t num_vectors_per_worker = indices.size() / num_threads;
	    size_t extra_num_vectors_last_worker = indices.size() % num_threads;

	    // Draw the first center uniformly at random.
	    uniform_int_distribution<> uniform_dis(0, indices.size() - 1);
	    (*center_indices)[0] = indices.at(uniform_dis(gen));

	    // Draw the remaining centers.
	    for (size_t j = 1; j < num_centers; ++j) {
		vector<double> vector_weights(indices.size());

		// Lambda function for computing vector weights [start ... end)
		// when selecting center_indices->at(center_index).
		auto weight_code = [&vectors, &indices, distance_type,
				    &center_indices, &vector_weights]
		    (size_t start, size_t end, size_t center_index) -> void {
		    for (size_t i = start; i < end; ++i) {
			double min_dist = numeric_limits<double>::infinity();
			for (size_t q = 0; q < center_index; ++q) {
			    // Search over the previously selected centers.
			    double dist = compute_distance(
				vectors.at(indices.at(i)),
				vectors.at(center_indices->at(q)),
				distance_type);
			    if (dist < min_dist) { min_dist = dist; }
			}
			vector_weights[i] = min_dist;
		    }
		};

		size_t start = 0;
		size_t end = num_vectors_per_worker;
		vector<thread> workers;
		for (size_t t = 0; t < num_threads; ++t) {
		    if (t == num_threads - 1) {  // Last worker does extra work.
			end += extra_num_vectors_last_worker;
		    }
		    workers.push_back(thread(weight_code, start, end, j));
		    start = end;
		    end += num_vectors_per_worker;
		}
		for (thread &worker : workers) { worker.join(); }

		discrete_distribution<> weighted_dis(vector_weights.begin(),
						     vector_weights.end());
		// Weighted draw
		(*center_indices)[j] = indices.at(weighted_dis(gen));
	    }
	} else if (seed_method == "uniform") {  // Uniform sampling.
	    vector<size_t> permuted_indices(indices);
	    size_t seed =
		chrono::system_clock::now().time_since_epoch().count();
	    shuffle(permuted_indices.begin(), permuted_indices.end(),
		    default_random_engine(seed));
	    for (size_t j = 0; j < num_centers; ++j) {
		(*center_indices)[j] = permuted_indices[j];
	    }
	} else if (seed_method == "front") {  // Front k vectors.
	    for (size_t j = 0; j < num_centers; ++j) {
		(*center_indices)[j] = indices.at(j);
	    }
	} else {
	    ASSERT(false, "Unknown seed method: " << seed_method);
	}
    }

    void cluster(const vector<Eigen::VectorXd> &vectors,
		 const vector<size_t> &indices, size_t max_num_iterations,
		 size_t num_threads, size_t distance_type, size_t num_centers,
		 const string &seed_method, size_t num_restarts,
		 vector<vector<Eigen::VectorXd> > *list_centers,
		 vector<vector<size_t> > *list_clustering,
		 vector<double> *list_objective) {
	list_centers->clear();
	list_clustering->clear();
	list_objective->clear();
	list_centers->resize(num_restarts);
	list_clustering->resize(num_restarts);
	list_objective->resize(num_restarts);

	size_t start = 0;  // Restart start index
	size_t end = start;  // Restart end index (TBD)
	size_t num_restarts_left = num_restarts;
	while (num_restarts_left > 0) { // Handle as many restarts as we can.
	    size_t num_threads_per_worker;
	    size_t extra_num_threads_last_worker;
	    if (num_restarts_left <= num_threads) {
		// More threads than restarts: finished.
		num_threads_per_worker = num_threads / num_restarts_left;
		extra_num_threads_last_worker = num_threads % num_restarts_left;
		end += num_restarts_left;
		num_restarts_left = 0;
	    } else {
		// More restarts than threads: assign 1 thread per restart.
		num_threads_per_worker = 1;
		extra_num_threads_last_worker = 0;
		end += num_threads;
		num_restarts_left -= num_threads;
	    }

	    // Lambda function for 1 restart.
	    auto restart_code = [&vectors, &indices, max_num_iterations,
				 distance_type, num_centers, seed_method,
				 &list_centers, &list_clustering,
				 &list_objective]
		(size_t restart_num, size_t num_threads_assigned) -> void {
		vector<size_t> center_indices;
		select_center_indices(vectors, indices, num_centers,
				      seed_method, num_threads_assigned,
				      distance_type, &center_indices);
		for (size_t center_index : center_indices) {
		    list_centers->at(restart_num).push_back(
			vectors.at(center_index));
		}

		(*list_objective)[restart_num] =
		cluster(vectors, indices, max_num_iterations,
			num_threads_assigned, distance_type, false,
			&list_centers->at(restart_num),
			&list_clustering->at(restart_num));
	    };

	    vector<thread> workers;
	    for (size_t restart_num = start; restart_num < end; ++restart_num) {
		size_t num_threads_assigned = num_threads_per_worker;
		if (restart_num == end - start - 1) {
		    num_threads_assigned += extra_num_threads_last_worker;
		}
		workers.push_back(thread(restart_code, restart_num,
					 num_threads_assigned));
	    }
	    for (thread &worker : workers) { worker.join(); }

	    start = end;
	    end = start;  // TBD
	}
    }

    void invert_clustering(const vector<size_t> &clustering,
			   vector<vector<size_t> > *clustering_inverse) {
	clustering_inverse->clear();
	for (size_t i = 0; i < clustering.size(); ++i) {
	    size_t cluster_num = clustering.at(i);
	    if (cluster_num >= clustering_inverse->size()) {
		clustering_inverse->resize(cluster_num + 1);
	    }
	    (*clustering_inverse)[cluster_num].push_back(i);
	}
    }
}  // namespace kmeans

double DivisiveClustering::Cluster(const vector<Eigen::VectorXd> &vectors,
				   size_t num_leaf_clusters,
				   unordered_map<string, vector<size_t> >
				   *leaves) {
    // Number of clusters shouldn't be more than number of vectors.
    num_leaf_clusters = min(num_leaf_clusters, vectors.size());
    leaves->clear();

    // Greedily split cluster with highest cost:
    //    q.top() = {(cluster_indices, higest_cost, path_from_root)}
    auto cmp = [](tuple<vector<size_t>, double, string> triple1,
		  tuple<vector<size_t>, double, string> triple2) {
	return get<1>(triple1) < get<1>(triple2);
    };
    std::priority_queue<tuple<vector<size_t>, double, string>,
			vector<tuple<vector<size_t>, double, string> >,
			decltype(cmp)> q(cmp);
    vector<size_t> indices;
    for (size_t i = 0; i < vectors.size(); ++i) { indices.push_back(i); }
    q.emplace(indices, 0.0, "");  // Initially a single cluster.

    // Start splitting.
    while (q.size() < num_leaf_clusters) {
	const vector<size_t> &current_indices = get<0>(q.top());
	string current_path = get<2>(q.top());
	if (verbose_) {
	    cerr << "(" << q.size() << "/" << num_leaf_clusters - 1 << ") "
		 << "2-means on " << current_indices.size() << " vectors: ";
	}

	vector<vector<Eigen::VectorXd> > list_centers;
	vector<vector<size_t> > list_clustering;
	vector<double> list_objective;
	kmeans::cluster(vectors, current_indices, max_num_iterations_kmeans_,
			num_threads_, distance_type_, 2, seed_method_,
			num_restarts_, &list_centers, &list_clustering,
			&list_objective);

	// Use the restart with smallest objective value.
	double min_objective = numeric_limits<double>::infinity();
	size_t best_restart_num = 0;
	for (size_t restart_num = 0; restart_num < num_restarts_;
	     ++restart_num) {
	    if (list_objective[restart_num] < min_objective) {
		min_objective = list_objective[restart_num];
		best_restart_num = restart_num;
	    }
	    if (verbose_) { cerr << list_objective[restart_num] << " "; }
	}
	if (verbose_) { cerr << endl; }

	// Compute each cluster's cost.
	vector<double> costs;
	kmeans::compute_cost(vectors, current_indices, num_threads_,
			     distance_type_, list_centers[best_restart_num],
			     list_clustering[best_restart_num], &costs);

	// Recover indices of vectors in each cluster.
	vector<vector<size_t> > split_indices(2);
	for (size_t i = 0; i < list_clustering[best_restart_num].size(); ++i) {
	    size_t cluster_num = list_clustering[best_restart_num][i];  // 0, 1
	    split_indices[cluster_num].push_back(current_indices.at(i));
	}

	q.pop();  // Do not pop until done with current_indices!
	q.emplace(split_indices[0], costs[0], current_path + "0");
	q.emplace(split_indices[1], costs[1], current_path + "1");
    }

    double total_cost = 0.0;
    while (q.size() > 0) {
	(*leaves)[get<2>(q.top())] = get<0>(q.top());
	total_cost += get<1>(q.top());
	q.pop();
    }

    return total_cost;
}

void DivisiveClustering::Invert(
    const unordered_map<string, vector<size_t> > &leaves,
    unordered_map<size_t, string> *path_from_root) {
    path_from_root->clear();
    for (auto &leaf : leaves) {
	for (size_t i : leaf.second) { (*path_from_root)[i] = leaf.first; }
    }
}

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
