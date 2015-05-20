// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Code for optimization.

#ifndef CORE_OPTIMIZE_H_
#define CORE_OPTIMIZE_H_

#include <Eigen/Dense>

#include "util.h"

namespace optimize {
    // Finds the convex coefficients for the columns of a matrix that
    // minimize a loss wrt. to a target vector.
    void compute_convex_coefficients(const Eigen::MatrixXd &columns,
				     const Eigen::VectorXd &target_vector,
				     const string &loss_type,
				     size_t max_num_updates,
				     double improvement_threshold,
				     bool verbose,
				     Eigen::VectorXd *convex_coefficients);
}  // namespace optimize

#endif  // CORE_OPTIMIZE_H_
