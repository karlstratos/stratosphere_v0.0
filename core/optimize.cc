// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "optimize.h"

namespace optimize {
    void compute_convex_coefficients(const Eigen::MatrixXd &columns,
				     const Eigen::VectorXd &target_vector,
				     const string &loss_type,
				     size_t max_num_updates,
				     double tol,
				     bool verbose,
				     Eigen::VectorXd *convex_coefficients) {
	size_t dimension = columns.rows();
	ASSERT(target_vector.size() == dimension, "Dimensions mismatch");

	// Initialize the convex coefficients uniformly.
	convex_coefficients->resize(dimension);
	for (size_t i = 0; i < dimension; ++i) {
	    (*convex_coefficients)(i) = 1.0 / dimension;
	}

	// Repeat the Frank-Wolfe updates:
	double old_loss = numeric_limits<double>::infinity();
	for (size_t update_num = 0; update_num < max_num_updates;
	     ++update_num) {
	    Eigen::VectorXd current_vector = columns * (*convex_coefficients);
	    Eigen::VectorXd residual = current_vector - target_vector;
	    double current_loss;
	    if (loss_type == "squared") {
		current_loss = residual.squaredNorm();
	    } else {
		ASSERT(false, "Unknown loss type: " << loss_type);
	    }
	    double improvement = old_loss - current_loss;
	    if (verbose) {
		cerr << update_num + 1 << ": " << current_loss << " (imp. "
		     << improvement << ")" << endl;
	    }
	    if (improvement <= tol) {
		if (verbose) { cerr << "Converged" << endl; }
		break;
	    }
	    old_loss = current_loss;

	    // Step 1. Minimize the linear approximation function around the
	    // current solution within the probability simplex.
	    Eigen::VectorXd gradient;
	    if (loss_type == "squared") {
		gradient = columns.transpose() * residual;
	    } else {
		ASSERT(false, "Unknown loss type: " << loss_type);
	    }
	    double min_value = numeric_limits<double>::infinity();
	    size_t min_i = 0;
	    for (size_t i = 0; i < dimension; ++i) {
		if (gradient(i) < min_value) {
		    min_value = gradient(i);
		    min_i = i;
		}
	    }  // TODO: start from here.

	    // Step 2. Find the optimal step size.
	    Eigen::VectorXd Ax_minus_Ai = current_vector - columns.col(min_i);  // This can be zero.
	    double step_size = Ax_minus_Ai.dot(residual);
	    step_size /= Ax_minus_Ai.squaredNorm();  // So this can be a NaN.
	    step_size = min(step_size, 1.0);
	    step_size = max(step_size, 0.0);
	    if (std::isnan(step_size)) { step_size = 0.0; }  // Set to 0 if NaN.

	    // Step 3. Move the current x along the coordinate min_i by step_size.
	    (*convex_coefficients) = (1 - step_size) * (*convex_coefficients);
	    (*convex_coefficients)(min_i) += step_size;
	}

	// Check if x is a proper distribution.
	double l1_mass = 0.0;
	for (size_t i = 0; i < dimension; ++i) {
	    double ith_prob = (*convex_coefficients)(i);
	    ASSERT(ith_prob >= 0.0, "Non-probability value? " << ith_prob);
	    l1_mass += (*convex_coefficients)(i);
	}
	ASSERT(fabs(l1_mass - 1.0) < 1e-10, "Does not sum to 1?" << l1_mass);

    }
}  // namespace optimize
