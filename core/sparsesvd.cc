// Author: Karl Stratos (stratos@cs.columbia.edu)

#include "sparsesvd.h"

#include <fstream>
#include <iomanip>
#include <math.h>
#include <sstream>

namespace svdlibc_helper {
    SMat binary_read_sparse_matrix(const string &file_path) {
	ifstream file(file_path, ios::in | ios::binary);
	size_t num_rows;
	size_t num_columns;
	size_t num_nonzeros;
	util_file::binary_read_primitive(file, &num_rows);
	util_file::binary_read_primitive(file, &num_columns);
	util_file::binary_read_primitive(file, &num_nonzeros);

	// Load the sparse matrix variable.
	SMat sparse_matrix = svdNewSMat(num_rows, num_columns, num_nonzeros);

	size_t current_nonzero_index = 0;  // Keep track of nonzero values.
	for (size_t col = 0; col < num_columns; ++col) {
	    sparse_matrix->pointr[col] = current_nonzero_index;
	    size_t num_nonzero_rows;
	    util_file::binary_read_primitive(file, &num_nonzero_rows);
	    for (size_t i = 0; i < num_nonzero_rows; ++i) {
		size_t row;
		double value;
		util_file::binary_read_primitive(file, &row);
		util_file::binary_read_primitive(file, &value);
		sparse_matrix->rowind[current_nonzero_index] = row;
		sparse_matrix->value[current_nonzero_index] = value;
		++current_nonzero_index;
	    }
	}
	sparse_matrix->pointr[num_columns] = num_nonzeros;
	return sparse_matrix;
    }

    void compute_svd(SMat sparse_matrix, size_t desired_rank,
		     Eigen::MatrixXd *left_singular_vectors,
		     Eigen::MatrixXd *right_singular_vectors,
		     Eigen::VectorXd *singular_values, size_t *actual_rank) {
	if (desired_rank == 0) {
	    left_singular_vectors->resize(0, 0);
	    right_singular_vectors->resize(0, 0);
	    singular_values->resize(0);
	    (*actual_rank) = 0;
	    return;
	}
	size_t rank_upper_bound = min(sparse_matrix->rows, sparse_matrix->cols);
	if (desired_rank > rank_upper_bound) {  // Adjust the oversized rank.
	    desired_rank = rank_upper_bound;
	}

	// Run the Lanczos algorithm with default parameters.
	SVDRec svd_result = svdLAS2A(sparse_matrix, desired_rank);

	left_singular_vectors->resize(sparse_matrix->rows, desired_rank);
	for (size_t row = 0; row < sparse_matrix->rows; ++row) {
	    for (size_t col = 0; col < desired_rank; ++col) {
		(*left_singular_vectors)(row, col) =
		    svd_result->Ut->value[col][row];  // Transpose.
	    }
	}

	right_singular_vectors->resize(sparse_matrix->cols, desired_rank);
	for (size_t row = 0; row < sparse_matrix->cols; ++row) {
	    for (size_t col = 0; col < desired_rank; ++col) {
		(*right_singular_vectors)(row, col) =
		    svd_result->Vt->value[col][row];  // Transpose.
	    }
	}

	singular_values->resize(desired_rank);
	for (size_t i = 0; i < desired_rank; ++i) {
	    (*singular_values)(i) = *(svd_result->S + i);
	}

	(*actual_rank) = svd_result->d;

	svdFreeSVDRec(svd_result);
    }
}  // namespace svdlibc_helper

SparseSVDSolver::~SparseSVDSolver() {
    FreeSparseMatrix();
    FreeSVDResult();
}

void SparseSVDSolver::SolveSparseSVD(size_t rank) {
    ASSERT(rank > 0, "SVD rank is given as <= 0: " << rank);
    ASSERT(HasMatrix(), "No matrix for SVD computation.");
    ASSERT(rank <= min(sparse_matrix_->rows, sparse_matrix_->cols), "SVD rank "
	   "is given as > min(num_rows, num_cols): " << rank << " > min("
	   << sparse_matrix_->rows << ", " << sparse_matrix_->cols << ")");

    // Free the current SVD result in case it's filled.
    FreeSVDResult();

    // Run the Lanczos algorithm with default parameters.
    svd_result_ = svdLAS2A(sparse_matrix_, rank);
}

void SparseSVDSolver::SumRowsColumns(
    unordered_map<size_t, double> *row_sum,
    unordered_map<size_t, double> *column_sum) {
    ASSERT(HasMatrix(), "No matrix for SVD computation.");
    row_sum->clear();
    column_sum->clear();

    size_t current_nonzero_index = 0;
    for (size_t col = 0; col < sparse_matrix_->cols; ++col) {
	while (current_nonzero_index < sparse_matrix_->pointr[col + 1]) {
	    size_t row = sparse_matrix_->rowind[current_nonzero_index];
	    double value = sparse_matrix_->value[current_nonzero_index];
	    (*row_sum)[row] += value;
	    (*column_sum)[col] += value;
	    ++current_nonzero_index;
	}
    }
}

void SparseSVDSolver::FreeSparseMatrix() {
    if (HasMatrix()) {
	svdFreeSMat(sparse_matrix_);
	sparse_matrix_ = nullptr;
    }
}

void SparseSVDSolver::FreeSVDResult() {
    if (HasSVDResult()) {
	svdFreeSVDRec(svd_result_);
	svd_result_ = nullptr;
    }
}
