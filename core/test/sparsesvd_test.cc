// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Check the correctness of the sparse SVD code.

#include <random>

#include "gtest/gtest.h"
#include "../sparsesvd.h"

// Test class that provides a dense random matrix.
class DenseRandomMatrix : public testing::Test {
protected:
    virtual void SetUp() {
	for (size_t column_index = 0; column_index < num_columns_;
	     ++column_index) {
	    for (size_t row_index = 0; row_index < num_rows_; ++row_index) {
		random_device device;
		default_random_engine engine(device());
		normal_distribution<double> normal(0.0, 1.0);
		double random_value = normal(engine);
		column_map_[column_index][row_index] = random_value;
	    }
	}
    }

    virtual void TearDown() { }

    size_t num_rows_ = 5;
    size_t num_columns_ = 4;
    size_t full_rank_ = min(num_rows_, num_columns_);
    unordered_map<size_t, unordered_map<size_t, double> > column_map_;
    SparseSVDSolver sparsesvd_solver_;
};

// Tests a full SVD a random (full-rank) matrix.
TEST_F(DenseRandomMatrix, CheckFullRank) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);  // Full SVD.
    EXPECT_EQ(full_rank_, sparsesvd_solver_.rank());
}

// Test class that provides an identity matrix.
class IdentityMatrix : public testing::Test {
protected:
    virtual void SetUp() {
	for (size_t column_index = 0; column_index < num_columns_;
	     ++column_index) {
	    vector<pair<size_t, double> > row_index_value_pairs;
	    for (size_t row_index = 0; row_index < num_rows_; ++row_index) {
		if (row_index == column_index) {
		    column_map_[column_index][row_index] = 1.0;
		}
	    }
	}
    }

    virtual void TearDown() { }

    size_t num_rows_ = 4;
    size_t num_columns_ = num_rows_;
    size_t full_rank_ = num_rows_;
    unordered_map<size_t, unordered_map<size_t, double> > column_map_;
    SparseSVDSolver sparsesvd_solver_;
};

// Demonstrates that SVDLIBC breaks without eigengaps.
TEST_F(IdentityMatrix, CheckSVDLIBCBreaksWithoutEigengap) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);
    EXPECT_NE(full_rank_, sparsesvd_solver_.rank());
}

// Demonstrates that SVDLIBC breaks even with a nonzero eigengap if small.
TEST_F(IdentityMatrix, CheckSVDLIBCBreaksWithTrivialEigengap) {
    // Introduce a nonzero eigengap in an identity matrix.
    column_map_[0][0] = 1.0000001;

    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);
    EXPECT_NE(full_rank_, sparsesvd_solver_.rank());
}

// Demonstrates that SVDLIBC works correctly with a not-so-small eigengap.
TEST_F(IdentityMatrix, ChecksSVDLIBCWorksWithNontrivialEigengap) {
    // Introduce eigengaps in an identity matrix.
    size_t value = num_rows_;
    for (size_t i = 0; i < num_rows_; ++i) {
	column_map_[i][i] = value--;  // diag(4, 3, 2, 1)
    }

    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(full_rank_);
    EXPECT_EQ(full_rank_, sparsesvd_solver_.rank());
}

// Test class that provides a sparse matrix with empty columns.
class SparseMatrixWithEmptyColumns : public testing::Test {
protected:
    virtual void SetUp() {
	//      Empty columns
	//        |     |
	//        |     |
	//        v     v
	//
	//     0  0  1  0
	//     0  0  0  0
	//     2  0  3  0
	//     0  0  4  0
	column_map_[0][2] = 2.0;
	column_map_[2][0] = 1.0;
	column_map_[2][2] = 3.0;
	column_map_[2][3] = 4.0;
    }

    virtual void TearDown() { }

    size_t num_rows_ = 4;
    size_t num_columns_ = 4;
    unordered_map<size_t, unordered_map<size_t, double> > column_map_;
    SparseSVDSolver sparsesvd_solver_;
    double tol_ = 1e-4;
};

// Confirms that SVDLIBC works correctly on this matrix.
TEST_F(SparseMatrixWithEmptyColumns, CheckCorrect) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    sparsesvd_solver_.SolveSparseSVD(2);
    EXPECT_EQ(2, sparsesvd_solver_.rank());
    EXPECT_NEAR(5.2469, fabs(*(sparsesvd_solver_.singular_values() + 0)), tol_);
    EXPECT_NEAR(1.5716, fabs(*(sparsesvd_solver_.singular_values() + 1)), tol_);
}

// Confirms that writing and loading this sparse matrix is correct.
TEST_F(SparseMatrixWithEmptyColumns, CheckWritingAndLoading) {
    // Write the matrix to a temporary file.
    string temp_file_path = tmpnam(nullptr);
    svdlibc_helper::binary_write_sparse_matrix(column_map_, temp_file_path);

    // Load the matrix from that file.
    sparsesvd_solver_.LoadSparseMatrix(temp_file_path);

    // Solve SVD and check the result.
    sparsesvd_solver_.SolveSparseSVD(2);
    EXPECT_EQ(2, sparsesvd_solver_.rank());
    EXPECT_NEAR(5.2469, fabs(*(sparsesvd_solver_.singular_values() + 0)), tol_);
    EXPECT_NEAR(1.5716, fabs(*(sparsesvd_solver_.singular_values() + 1)), tol_);
    remove(temp_file_path.c_str());
}

// Checks summing rows/columns.
TEST_F(SparseMatrixWithEmptyColumns, CheckSumRowsColumns) {
    sparsesvd_solver_.LoadSparseMatrix(column_map_);
    unordered_map<size_t, double> row_sum;
    unordered_map<size_t, double> column_sum;
    sparsesvd_solver_.SumRowsColumns(&row_sum, &column_sum);
    //     0  0  1  0
    //     0  0  0  0
    //     2  0  3  0
    //     0  0  4  0
    EXPECT_NEAR(1.0, row_sum[0], tol_);
    EXPECT_NEAR(0.0, row_sum[1], tol_);
    EXPECT_NEAR(5.0, row_sum[2], tol_);
    EXPECT_NEAR(4.0, row_sum[3], tol_);
    EXPECT_NEAR(2.0, column_sum[0], tol_);
    EXPECT_NEAR(0.0, column_sum[1], tol_);
    EXPECT_NEAR(8.0, column_sum[2], tol_);
    EXPECT_NEAR(0.0, column_sum[3], tol_);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
