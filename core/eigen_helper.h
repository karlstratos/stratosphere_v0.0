// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various helper functions for the Eigen library. Some conventions:
//    - A "basis" is always a matrix whose columns are the basis elements.

#ifndef CORE_EIGEN_HELPER_H_
#define CORE_EIGEN_HELPER_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "util.h"

namespace eigen_helper {
    // Writes an Eigen dense matrix to a binary file.
    template<typename EigenDenseMatrix>
    void binary_write_matrix(const EigenDenseMatrix& matrix,
			     const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	typename EigenDenseMatrix::Index num_rows = matrix.rows();
	typename EigenDenseMatrix::Index num_columns = matrix.cols();
	util_file::binary_write_primitive(num_rows, file);
	util_file::binary_write_primitive(num_columns, file);
	file.write(reinterpret_cast<const char *>(matrix.data()), num_rows *
		   num_columns * sizeof(typename EigenDenseMatrix::Scalar));
    }

    // Reads an Eigen dense matrix from a binary file.
    template<typename EigenDenseMatrix>
    void binary_read_matrix(const string &file_path, EigenDenseMatrix *matrix) {
	ifstream file(file_path, ios::in | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	typename EigenDenseMatrix::Index num_rows;
	typename EigenDenseMatrix::Index num_columns;
	util_file::binary_read_primitive(file, &num_rows);
	util_file::binary_read_primitive(file, &num_columns);
	matrix->resize(num_rows, num_columns);
	file.read(reinterpret_cast<char*>(matrix->data()), num_rows *
		  num_columns * sizeof(typename EigenDenseMatrix::Scalar));
    }

    // Computes the Moore–Penrose pseudo-inverse.
    void compute_pseudoinverse(const Eigen::MatrixXd &matrix,
			       Eigen::MatrixXd *matrix_pseudoinverse);

    // Extends an orthonormal basis to subsume the given vector v.
    void extend_orthonormal_basis(const Eigen::VectorXd &v,
				  Eigen::MatrixXd *orthonormal_basis);

    // Finds an orthonormal basis that spans the range of the matrix.
    void find_range(const Eigen::MatrixXd &matrix,
		    Eigen::MatrixXd *orthonormal_basis);

    // Generates a random projection matrix.
    void generate_random_projection(size_t original_dimension,
				    size_t reduced_dimension,
				    Eigen::MatrixXd *projection_matrix);

    // Returns true if two Eigen dense matrices are close in value.
    template<typename EigenDenseMatrix>
    bool check_near(const EigenDenseMatrix& matrix1,
		    const EigenDenseMatrix& matrix2, double error_threshold) {
	if (matrix1.rows() != matrix2.rows() ||
	    matrix2.cols() != matrix2.cols()) { return false; }
	for (size_t row = 0; row < matrix1.rows(); ++row) {
	    for (size_t col = 0; col < matrix1.cols(); ++col) {
		if (fabs(matrix1(row, col) - matrix2(row, col))
		    > error_threshold) { return false; }
	    }
	}
	return true;
    }

    // Returns true if two Eigen dense matrices are close in absolute value.
    template<typename EigenDenseMatrix>
    bool check_near_abs(const EigenDenseMatrix& matrix1,
			const EigenDenseMatrix& matrix2,
			double error_threshold) {
	if (matrix1.rows() != matrix2.rows() ||
	    matrix2.cols() != matrix2.cols()) { return false; }
	for (size_t row = 0; row < matrix1.rows(); ++row) {
	    for (size_t col = 0; col < matrix1.cols(); ++col) {
		if (fabs(fabs(matrix1(row, col)) - fabs(matrix2(row, col)))
		    > error_threshold) { return false; }
	    }
	}
	return true;
    }
}  // namespace eigen_helper

#endif  // CORE_EIGEN_HELPER_H_
