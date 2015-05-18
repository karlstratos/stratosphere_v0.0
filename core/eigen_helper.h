// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various helper functions for the Eigen library.

#ifndef CORE_EIGEN_HELPER_H_
#define CORE_EIGEN_HELPER_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

namespace eigen_helper {
    // Writes an Eigen matrix to a binary file.
    void binary_write(const Eigen::MatrixXd &m, const string &file_path);
}  // namespace eigen_helper

#endif  // CORE_EIGEN_HELPER_H_
