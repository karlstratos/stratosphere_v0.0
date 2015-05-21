// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// This is a wrapper around SVDLIBC that provides a clean interface for
// performing singular value decomposition (SVD) on sparse matrices in
// Standard C++.

#ifndef CORE_SPARSESVD_H_
#define CORE_SPARSESVD_H_

#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>

#include "util.h"

extern "C" {  // For using C code from C++ code.
#include "../third_party/SVDLIBC/svdlib.h"
}

namespace svdlibc_helper {
    // Writes an unordered_map (column -> {row: value}) as a binary file for
    // SVDLIBC.
    template<typename T>
    void binary_write_sparse_matrix(
	const unordered_map<size_t, unordered_map<size_t, T> > &column_map,
	size_t num_rows, size_t num_columns, size_t num_nonzeros,
	const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	util_file::binary_write_primitive(num_rows, file);
	util_file::binary_write_primitive(num_columns, file);
	util_file::binary_write_primitive(num_nonzeros, file);
	for (size_t col = 0; col < num_columns; ++col) {
	    if (column_map.find(col) == column_map.end()) {
		size_t zero = 0;  // No nonzero rows for this column.
		util_file::binary_write_primitive(zero, file);
		continue;
	    }
	    util_file::binary_write_primitive(column_map.at(col).size(), file);
	    for (const auto &row_pair: column_map.at(col)) {
		size_t row = row_pair.first;
		T value = row_pair.second;  // TODO: might have convert to double.
		util_file::binary_write_primitive(row, file);
		util_file::binary_write_primitive(value, file);
	    }
	}
    }

    // Writes an unordered_map (column -> {row: value}) as a binary file for
    // SVDLIBC. Need to first compute the number of dimensions and nonzeros.
    template<typename T>
    void binary_write_sparse_matrix(
	const unordered_map<size_t, unordered_map<size_t, T> > &column_map,
	const string &file_path) {
	size_t num_rows = 0;
	size_t num_columns = 0;
	size_t num_nonzeros = 0;
	for (const auto &col_pair: column_map) {
	    size_t col = col_pair.first;
	    if (col >= num_columns) { num_columns = col + 1; }
	    for (const auto &row_pair: col_pair.second) {
		size_t row = row_pair.first;
		if (row >= num_rows) { num_rows = row + 1; }
		++num_nonzeros;
	    }
	}
	binary_write_sparse_matrix(column_map, num_rows, num_columns,
				   num_nonzeros, file_path);
    }

    // Reads an SVDLIBC sparse matrix from a binary file.
    SMat binary_read_sparse_matrix(const string &file_path);

    // Converts a column map to an SVDLIBC sparse matrix:
    //    column_map[j][i] = value at (i, j) in sparse_matrix
    template<typename T>
    SMat convert_column_map(
	const unordered_map<size_t, unordered_map<size_t, T> > &column_map) {
	size_t num_rows = 0;
	size_t num_columns = 0;
	size_t num_nonzeros = 0;
	for (const auto &col_pair: column_map) {
	    size_t col = col_pair.first;
	    if (col >= num_columns) { num_columns = col + 1; }
	    for (const auto &row_pair: col_pair.second) {
		size_t row = row_pair.first;
		if (row >= num_rows) { num_rows = row + 1; }
		++num_nonzeros;
	    }
	}

	// Load the sparse matrix variable.
	SMat sparse_matrix = svdNewSMat(num_rows, num_columns, num_nonzeros);

	size_t current_nonzero_index = 0;  // Keep track of nonzero values.
	for (size_t col = 0; col < num_columns; ++col) {
	    sparse_matrix->pointr[col] = current_nonzero_index;
	    if (column_map.find(col) == column_map.end()) { continue; }
	    for (const auto &row_pair: column_map.at(col)) {
		size_t row = row_pair.first;
		T value = row_pair.second;
		sparse_matrix->rowind[current_nonzero_index] = row;
		sparse_matrix->value[current_nonzero_index] = value;
		++current_nonzero_index;
	    }
	}
	sparse_matrix->pointr[num_columns] = num_nonzeros;
	return sparse_matrix;
    }

    // Computes a low-rank SVD of a sparse matrix in the SVDLIBC format. The
    // singular vectors are organized as columns of a matrix. The given matrix
    // has rank smaller than the desired rank if actual_rank < desired_rank.
    void compute_svd(SMat sparse_matrix, size_t desired_rank,
		     Eigen::MatrixXd *left_singular_vectors,
		     Eigen::MatrixXd *right_singular_vectors,
		     Eigen::VectorXd *singular_values, size_t *actual_rank);
}  // namespace svdlibc_helper

class SparseSVDSolver {
public:
    // Initializes an empty SVD solver.
    SparseSVDSolver() { }

    // Initializes with a sparse matrix in a file.
    SparseSVDSolver(const string &file_path) { LoadSparseMatrix(file_path); }

    // Cleans up memory at deletion.
    ~SparseSVDSolver();

    // Loads a sparse matrix from a text file into the class object.
    void LoadSparseMatrix(const string &file_path) {
	sparse_matrix_ = svdlibc_helper::binary_read_sparse_matrix(file_path);
    }

    // Loads a sparse matrix M for SVD: column_map[j][i] = M_{i,j}.
    void LoadSparseMatrix(
	const unordered_map<size_t, unordered_map<size_t, double> >
	&column_map) {
	sparse_matrix_ = svdlibc_helper::convert_column_map(column_map);
    }

    // Computes a thin SVD of the loaded sparse matrix.
    void SolveSparseSVD(size_t rank);

    // Sums the rows/columns: row_sum[i] = sum_j M_{i, j}, column_sum[j] =
    // sum_i M_{i, j}.
    void SumRowsColumns(unordered_map<size_t, double> *row_sum,
			unordered_map<size_t, double> *column_sum);

    // Does it have some matrix loaded?
    bool HasMatrix() const { return sparse_matrix_ != nullptr; }

    // Does it have some SVD result?
    bool HasSVDResult() const { return svd_result_ != nullptr; }

    // Returns a pointer to the sparse matrix for SVD.
    SMat sparse_matrix() { return sparse_matrix_; }

    // Returns a pointer to a matrix whose i-th row is the left singular vector
    // corresponding to the i-th largest singular value.
    DMat left_singular_vectors() const { return svd_result_->Ut; }

    // Returns a pointer to a matrix whose i-th row is the right singular vector
    // corresponding to the i-th largest singular value.
    DMat right_singular_vectors() const { return svd_result_->Vt; }

    // Returns a pointer to computed singular values.
    double *singular_values() const { return svd_result_->S; }

    // Returns the rank of the computed SVD.
    size_t rank() const { return svd_result_->d; }

    // Frees the loaded sparse matrix and sets it to nullptr.
    void FreeSparseMatrix();

    // Frees the loaded SVD result and sets it to nullptr.
    void FreeSVDResult();

private:
    // Sparse matrix for SVD.
    SMat sparse_matrix_ = nullptr;

    // Result of the latest SVD computation.
    SVDRec svd_result_ = nullptr;
};

#endif  // CORE_SPARSESVD_H_
