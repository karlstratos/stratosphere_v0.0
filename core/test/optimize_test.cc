// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the optimization code.

#include "gtest/gtest.h"

#include <limits>

#include "../optimize.h"

// Test class for convex-constrained optimization.
class ConvexConstrainedOptimization : public testing::Test {
protected:
    // (1 x 3): short and fat
    // (3 x 3): square
    // (6 x 3): tall and thin
    vector<size_t> list_num_rows_ = {1, 3, 6};
    vector<size_t> list_num_columns_ = {3};
};

// Checks minimizing the squared-loss for convex-constrained optimization.
TEST_F(ConvexConstrainedOptimization, SquaredLoss) {
    size_t max_num_updates = numeric_limits<size_t>::max();
    double stopping_threshold = 1e-10;
    bool verbose = false;

    for (size_t num_rows : list_num_rows_) {
	for (size_t num_columns : list_num_columns_) {
	    // Set up a problem.
	    Eigen::MatrixXd columns = Eigen::MatrixXd::Random(num_rows,
							      num_columns);
	    Eigen::VectorXd convex_coefficients =
		Eigen::VectorXd::Random(num_columns).cwiseAbs();
	    double l1_norm = convex_coefficients.lpNorm<1>();
	    convex_coefficients /= l1_norm;
	    Eigen::VectorXd target_vector = columns * convex_coefficients;

	    // Solve the problem.
	    Eigen::VectorXd computed_coefficients;
	    optimize::compute_convex_coefficients_squared_loss(
		columns, target_vector, max_num_updates, stopping_threshold,
		verbose, &computed_coefficients);
	    Eigen::VectorXd estimate = columns * computed_coefficients;

	    // Check if the error (= loss of the estimate since the optimal loss
	    // is 0) is at least as small as specified.
	    double error = 0.5 * (target_vector - estimate).squaredNorm();
	    EXPECT_TRUE(error <= stopping_threshold);
	}
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
