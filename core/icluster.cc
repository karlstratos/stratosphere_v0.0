// Author: Karl Stratos (me@karlstratos.com)

#include "icluster.h"

#include <fstream>
#include <unordered_set>

#include "util.h"

void Data::Read(const string &file_path) {
    Clear();
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << file_path);
    while (file.good()) {
	vector<string> tokens;  // <cluster> <instance> <freq>
	util_file::read_line(&file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.
	x2f[tokens[1]] = (tokens.size() > 2) ? stod(tokens[2]) : 1.0;
    }
}

void Data::Clear() {
    x2f.clear();
}

size_t Data::Frequency(string x) {
    auto search = x2f.find(x);
    ASSERT(search != x2f.end(), "No point named: " << x);
    return search->second;
}

void Cluster::Read(Data *data_given, const string &file_path) {
    Clear();
    data = data_given;
    ifstream file(file_path, ios::in);
    ASSERT(file.is_open(), "Cannot open " << file_path);
    while (file.good()) {
	vector<string> tokens;  // <cluster> <instance> <freq>
	util_file::read_line(&file, &tokens);
	if (tokens.size() == 0) { continue; }  // Skip empty lines.
	c2x[tokens[0]].insert(tokens[1]);
	x2c[tokens[1]] = tokens[0];
    }
}

void Cluster::Read(Data *data_given,
		   const unordered_map<string, unordered_set<string> >
		   &c2x_given) {
    Clear();
    data = data_given;
    c2x = c2x_given;
    for (const auto &kv : c2x) {
	for (const auto &x : kv.second) { x2c[x] = kv.first; }
    }
}

void Cluster::Clear() {
    data = nullptr;
    c2x.clear();
    x2c.clear();
}

size_t Cluster::ClusterSize(string c) {
    auto search = c2x.find(c);
    ASSERT(search != c2x.end(), "No cluster named: " << c);
    return search->second.size();
}

namespace icluster {
    bool intersect_bool(const unordered_set<string> &c1,
			const unordered_set<string> &c2) {
	// Faster when iterate over a smaller set.
	if (c1.size() > c2.size()) { return intersect_bool(c2, c1); }
	for (const auto &x : c1) {
	    if (c2.find(x) != c2.end()) { return true; } // Overlap
	}
	return false;
    }

    unordered_set<string> intersect(const unordered_set<string> &c1,
				    const unordered_set<string> &c2) {
	// Faster when iterate over a smaller set.
	if (c1.size() > c2.size()) { return intersect(c2, c1); }
	unordered_set<string> intersection;
	for (const auto &x : c1) {
	    if (c2.find(x) != c2.end()) { intersection.insert(x); }
	}
	return intersection;
    }

    double dist(const unordered_set<string> &c, const Cluster &C) {
	size_t num_overlapping_clusters = 0;
	for (const auto &c0 : C.c2x) {
	    if (intersect_bool(c, c0.second)) { ++num_overlapping_clusters; }
	}
	return num_overlapping_clusters - 1;  // # extraneous clusters in C
    }

    double dist(const Cluster &C, const Cluster &C_other) {
	size_t num_extraneous_clusters_total = 0;
	for (const auto &c : C.c2x) {
	    num_extraneous_clusters_total += dist(c.second, C_other);
	}
	return num_extraneous_clusters_total;
    }
}  // namespace icluster

vector<string> ClusterFixer::FindSplittable() {
    vector<string> splittable;
    for (const auto &c : proposed.c2x) {
	if (icluster::dist(c.second, desired) > 0) {
	    splittable.push_back(c.first);
	}
    }
    return splittable;
}

vector<unordered_set<string> > ClusterFixer::FindMergeable() {
    unordered_map<string, unordered_set<string> > eta_overlap;
    for (const auto &c_desired : desired.c2x) {
	for (const auto &c : proposed.c2x) {
	    unordered_set<string> intersection = \
		icluster::intersect(c.second, c_desired.second);
	    if (intersection.size() >= eta * c.second.size()) {
		eta_overlap[c_desired.first].insert(c.first);
	    }
	}
    }
    // Only return groups that have more than 1 cluster.
    vector<unordered_set<string> > mergeable;
    for (const auto &group : eta_overlap) {
	if (group.second.size() > 1) { mergeable.push_back(group.second); }
    }
    return mergeable;
}

Request ClusterFixer::Fix() {
    Request r = GetRequest();

    if (r == Request::split) {
	Split();
    } else if (r == Request::merge) {
	Merge();
    } else { } // No request, nothing to do.

    return r;
}

Request ClusterFixer::GetRequest() {
    vector<string> splittable = FindSplittable();
    vector<unordered_set<string> > mergeable = FindMergeable();
    size_t num_edits = splittable.size() + mergeable.size();
    if (num_edits == 0) { return Request::none; }

    uniform_int_distribution<> dis(0, num_edits - 1);
    size_t i = dis(mt);

    if (i < splittable.size()) {
	cluster_to_split = splittable[i];
	return Request::split;
    } else {
	string c1, c2;
	size_t j = 0;
	for (const auto &c : mergeable[i - splittable.size()]) {
	    if (j == 0) {
		c1 = c;
	    } else if (j == 1) {
		c2 = c;
	    } else {
		break;
	    }
	    ++j;
	}
	clusters_to_merge = make_pair(c1, c2);
	return Request::merge;
    }
}

void ClusterFixer::Split() {
    string c = cluster_to_split;
    pair<unordered_set<string>, unordered_set<string> > split = ChooseSplit(c);

    proposed.c2x.erase(c);
    proposed.c2x[c + "0"] = split.first;
    for (const auto &x : split.first) {
	proposed.x2c[x] = c + "0";
    }
    proposed.c2x[c + "1"] = split.second;
    for (const auto &x : split.second) {
	proposed.x2c[x] = c + "1";
    }
}

pair<unordered_set<string>, unordered_set<string> >
ClusterFixer::ChooseSplit(string c) {
    unordered_set<string> *xs = &proposed.c2x[c];
    unordered_set<string> group1, group2;

    if (split_method == "clean") {
	string c0 = "";
	for (auto x : *xs) {
	    if (c0.size() == 0) { c0 = desired.x2c[x]; }
	    if (desired.x2c[x] == c0) {
		group1.insert(x);
	    } else {
		group2.insert(x);
	    }
	}
    } else if (split_method == "random") {
	uniform_int_distribution<> dis(0, 1);
	bool flag = (dis(mt) == 0);
	size_t i = 0;
	for (auto x : *xs) {
	    if (i == 0) {
		if (flag) {
		    group1.insert(x);
		} else {
		    group2.insert(x);
		}
	    } else if (i == 1) {
		if (flag) {
		    group2.insert(x);
		} else {
		    group1.insert(x);
		}
	    } else {
		if (dis(mt) == 0) {
		    group1.insert(x);
		} else {
		    group2.insert(x);
		}
	    }
	    ++i;
	}
    } else {
	ASSERT(false, "Unknown split method: " << split_method);
    }

    return make_pair(group1, group2);
}


void ClusterFixer::Merge() {
    string c1 = clusters_to_merge.first;
    string c2 = clusters_to_merge.second;

    // Add c2 into c1.
    proposed.c2x[c1].insert(proposed.c2x[c2].begin(), proposed.c2x[c2].end());
    for (const auto &x : proposed.c2x[c2]) {
	proposed.x2c[x] = c1;
    }

    // Delete c2.
    proposed.c2x.erase(c2);
}
