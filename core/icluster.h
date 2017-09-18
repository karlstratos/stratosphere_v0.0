// Author: Karl Stratos (me@karlstratos.com)
//
// Code for interactive clustering.

#ifndef CORE_ICLUSTER_H_
#define CORE_ICLUSTER_H_

#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// Struct that represents data points for clustering.
struct Data {
    // Initializes empty.
    Data() { }

    // Initializes from a file.
    Data(const string &file_path) { Read(file_path); }

    // Reads data from a file.
    void Read(const string &file_path);

    // Clears the object.
    void Clear();

    // Returns information of the data points.
    size_t NumPoints() { return x2f.size(); }
    size_t Frequency(string x);

    // Maps a data point to its frequency (if provided).
    unordered_map<string, double> x2f;
};

// Struct that represents a clustering.
struct Cluster {
    // Initializes empty.
    Cluster() { }

    // Initializes from data and a file/mapping.
    Cluster(Data *data_given, const string &file_path) {
	Read(data_given, file_path);
    }
    Cluster(Data *data_given,
	    const unordered_map<string, unordered_set<string> > &c2x_given) {
	Read(data_given, c2x_given);
    }

    // Clears the object.
    void Clear();

    // Reads a clustering from data and a file/mapping.
    void Read(Data *data_given, const string &file_path);
    void Read(Data *data_given,
	      const unordered_map<string, unordered_set<string> > &c2x_given);

    // Returns information of the clustering.
    size_t NumClusters() { return c2x.size(); }
    size_t NumPoints() { return (data == nullptr) ? 0 : data->x2f.size(); }
    size_t ClusterSize(string c);

    // Pointer to associated data points. Take care to make sure that the
    // pointed Data instance is alive during the whole life span of the Cluster
    // instance!
    Data *data = nullptr;

    // Maps a cluster to its data points.
    unordered_map<string, unordered_set<string> > c2x;

    // Maps a data point to its cluster.
    unordered_map<string, string> x2c;
};

namespace icluster {
    // Do c1 and c2 intersect?
    bool intersect_bool(const unordered_set<string> &c1,
			const unordered_set<string> &c2);

    // What is the intersection between c1 and c2?
    unordered_set<string> intersect(const unordered_set<string> &c1,
				    const unordered_set<string> &c2);

    // Computes dist(c,C): how many *extraneous* clusters in C overlaps with c,
    // where c is a cluster and C is a clustering over the same data.
    double dist(const unordered_set<string> &c, const Cluster &C);

    // Computes dist(C,C') := sum_{c in C} dist(c, C').
    double dist(const Cluster &C, const Cluster &C_other);

}  // namespace icluster

// Enum class for edit requests.
enum class Request { split, merge, none };

// Struct that fixes a proposed clustering.
struct ClusterFixer {
    // Initializes empty.
    ClusterFixer() { }

    // Finds proposed clusters that overlap with multiple desired clusters.
    vector<string> FindSplittable();

    // Finds proposed clusters that overlap with the same desired cluster c^* by
    // at least eta portion: when eta=1, they are contained in c^*.
    vector<unordered_set<string> > FindMergeable();

    // Fixes proposed w.r.t. desired with one edit.
    Request Fix();

    // Gets an edit request.
    Request GetRequest();

    // Splits the proposed cluster in variable cluster_to_split.
    void Split();

    // Chooses how to split the given list of data points.
    pair<unordered_set<string>, unordered_set<string> > ChooseSplit(string c);

    // Merges the proposed clusters in variable clusters_to_merge.
    void Merge();

    // Computes the overclustering error.
    size_t Overclustering() { return icluster::dist(proposed, desired); }

    // Computes the underclustering error.
    size_t Underclustering() { return icluster::dist(desired, proposed); }

    // Proposed clustering: we only modify this, not the desired clustering.
    Cluster proposed;

    // Desired clustering: constant.
    Cluster desired;

    // Cluster to split.
    string cluster_to_split;

    // Clusters to merge.
    pair<string, string> clusters_to_merge;

    // Overlap parameter for merge. Needs to be >0.5 to make progress.
    double eta = 1.0;

    // Split method.
    string split_method = "clean";

    // Randomness engine
    mt19937 mt;
};

#endif  // CORE_ICLUSTER_H_
