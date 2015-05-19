// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the Eigen helper code.

#include "gtest/gtest.h"

#include "../eigen_helper.h"

// Test class for checking Eigen helper functions.
class EigenHelper : public testing::Test {
protected:
    // (1 x 3): short and fat
    // (3 x 3): square
    // (5 x 3): tall and thin
    vector<size_t> list_num_rows_ = {1, 3, 5};
    vector<size_t> list_num_columns_ = {3};
    double tol_ = 1e-10;
};

// Checks reading and writing a double matrix.
TEST_F(EigenHelper, WritingReadingDoubleMatrix) {
    for (size_t num_rows : list_num_rows_) {
	for (size_t num_columns : list_num_columns_) {
	    string file_path = tmpnam(nullptr);
	    Eigen::MatrixXd matrix1 = Eigen::MatrixXd::Random(num_rows,
							      num_columns);
	    eigen_helper::binary_write_matrix(matrix1, file_path);
	    Eigen::MatrixXd matrix2;
	    eigen_helper::binary_read_matrix(file_path, &matrix2);
	    EXPECT_EQ(num_rows, matrix2.rows());
	    EXPECT_EQ(num_columns, matrix2.cols());
	    for (size_t row = 0; row < num_rows; ++row) {
		for (size_t col = 0; col < num_columns; ++col) {
		    EXPECT_NEAR(matrix1(row, col), matrix2(row, col), tol_);
		}
	    }
	    remove(file_path.c_str());
	}
    }
}

// Checks reading and writing an integer matrix.
TEST_F(EigenHelper, WritingReadingIntMatrix) {
    for (size_t num_rows : list_num_rows_) {
	for (size_t num_columns : list_num_columns_) {
	    string file_path = tmpnam(nullptr);
	    Eigen::MatrixXi matrix1 = Eigen::MatrixXi::Random(num_rows,
							      num_columns);
	    eigen_helper::binary_write_matrix(matrix1, file_path);
	    Eigen::MatrixXi matrix2;
	    eigen_helper::binary_read_matrix(file_path, &matrix2);
	    EXPECT_EQ(num_rows, matrix2.rows());
	    EXPECT_EQ(num_columns, matrix2.cols());
	    for (size_t row = 0; row < num_rows; ++row) {
		for (size_t col = 0; col < num_columns; ++col) {
		    EXPECT_EQ(matrix1(row, col), matrix2(row, col));
		}
	    }
	    remove(file_path.c_str());
	}
    }
}

// Checks reading and writing a float vector.
TEST_F(EigenHelper, WritingReadingFloatVector) {
    size_t length = 3;
    string file_path = tmpnam(nullptr);
    Eigen::VectorXf vector1 = Eigen::VectorXf::Random(length);
    eigen_helper::binary_write_matrix(vector1, file_path);
    Eigen::VectorXf vector2;
    eigen_helper::binary_read_matrix(file_path, &vector2);
    EXPECT_EQ(length, vector2.rows());
    for (size_t i = 0; i < length; ++i) {
	EXPECT_NEAR(vector1(i), vector2(i), tol_);
    }
    remove(file_path.c_str());
}

// Checks computing the matrix pseudo-inverse.
TEST_F(EigenHelper, PseudoInverse) {
    for (size_t num_rows : list_num_rows_) {
	for (size_t num_columns : list_num_columns_) {
	    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(num_rows,
							     num_columns);
	    Eigen::MatrixXd matrix_pseudoinverse;
	    eigen_helper::compute_pseudoinverse(matrix, &matrix_pseudoinverse);
	    Eigen::MatrixXd product = (num_rows >= num_columns) ?
		matrix_pseudoinverse * matrix : matrix * matrix_pseudoinverse;
	    size_t rank = min(num_rows, num_columns);
	    for (size_t row = 0; row < rank; ++row) {
		for (size_t col = 0; col < rank; ++col) {
		    if (row == col) {
			EXPECT_NEAR(1.0, product(row, col), tol_);
		    } else {
			EXPECT_NEAR(0.0, product(row, col), tol_);
		    }
		}
	    }
	}
    }
}

// Checks finding the matrix range.
TEST_F(EigenHelper, FindMatrixRange) {
    for (size_t num_rows : list_num_rows_) {
	for (size_t num_columns : list_num_columns_) {
	    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(num_rows,
							     num_columns);
	    Eigen::MatrixXd orthonormal_basis;
	    eigen_helper::find_range(matrix, &orthonormal_basis);

	    // Check if orthonormal.
	    Eigen::MatrixXd inner_product =
		orthonormal_basis.transpose() * orthonormal_basis;
	    size_t rank = min(num_rows, num_columns);
	    for (size_t row = 0; row < rank; ++row) {
		for (size_t col = 0; col < rank; ++col) {
		    if (row == col) {
			EXPECT_NEAR(1.0, inner_product(row, col), tol_);
		    } else {
			EXPECT_NEAR(0.0, inner_product(row, col), tol_);
		    }
		}
	    }

	    // Check if the same range.
	    Eigen::VectorXd v = Eigen::VectorXd::Random(num_rows);
	    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU);
	    Eigen::MatrixXd projection1 =
		svd.matrixU() * svd.matrixU().transpose() * v;
	    Eigen::MatrixXd projection2 =
		orthonormal_basis * orthonormal_basis.transpose() * v;
	    for (size_t i = 0; i < projection1.size(); ++i) {
		EXPECT_NEAR(projection1(i), projection2(i), tol_);
	    }
	}
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
