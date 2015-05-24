// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "optimize.h"

namespace optimize {
    void compute_convex_coefficients_squared_loss(
	const Eigen::MatrixXd &columns, const Eigen::VectorXd &target_vector,
	size_t max_num_updates, double stopping_threshold, bool verbose,
	Eigen::VectorXd *convex_coefficients) {
	ASSERT(target_vector.size() == columns.rows(), "Dimensions mismatch");
	size_t num_columns = columns.cols();
	convex_coefficients->resize(num_columns);
	for (size_t i = 0; i < num_columns; ++i) {  // Uniform initialization.
	    (*convex_coefficients)(i) = 1.0 / num_columns;
	}

	// Iterate the Frank-Wolfe steps:
	for (size_t update_num = 1; update_num <= max_num_updates;
	     ++update_num) {
	    Eigen::VectorXd current_vector = columns * (*convex_coefficients);
	    Eigen::VectorXd residual = current_vector - target_vector;

	    // Step 1. Minimize the linear approximation function around the
	    // current solution inside the probability simplex.
	    Eigen::VectorXd gradient = columns.transpose() * residual;
	    double min_value = numeric_limits<double>::infinity();
	    size_t min_i = 0;
	    for (size_t i = 0; i < num_columns; ++i) {
		if (gradient(i) < min_value) {
		    min_value = gradient(i);
		    min_i = i;
		}
	    }

	    // If the duality gap (an upper bound on the difference between the
	    // current loss and the optimal loss) is small, stop optimizing.
	    if (update_num % 100 == 0) {
		Eigen::VectorXd deflated_vector = *convex_coefficients;
		deflated_vector(min_i) -= 1.0;
		double duality_gap = gradient.dot(deflated_vector);
		if (verbose) {
		    cerr << "\r" << update_num << ": CURRENT_LOSS - "
			 << "OPTIMAL_LOSS <= " << duality_gap << "   " << flush;
		}
		if (duality_gap <= stopping_threshold) {
		    if (verbose) { cerr << endl; }
		    break;
		}
	    }

	    // Step 2. Find the optimal step size.
	    Eigen::VectorXd vector1 = current_vector - columns.col(min_i);
	    double vector1_norm = vector1.squaredNorm();
	    double step_size = vector1.dot(residual);  // 0 if vector1 = 0
	    if (vector1_norm > 0.0) {
		step_size /= vector1_norm;
		step_size = min(step_size, 1.0);
		step_size = max(step_size, 0.0);
	    }

	    // Step 3. Move the current x along coordinate min_i by step_size.
	    (*convex_coefficients) = (1 - step_size) * (*convex_coefficients);
	    (*convex_coefficients)(min_i) += step_size;
	}

	// Check if x is a proper distribution.
	double l1_mass = 0.0;
	for (size_t i = 0; i < num_columns; ++i) {
	    double ith_prob = (*convex_coefficients)(i);
	    ASSERT(ith_prob >= 0.0, "Non-probability value? " << ith_prob);
	    l1_mass += (*convex_coefficients)(i);
	}
	ASSERT(fabs(l1_mass - 1.0) < 1e-10, "Does not sum to 1?" << l1_mass);
    }
}  // namespace optimize
