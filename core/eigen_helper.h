// Author: Karl Stratos (stratos@cs.columbia.edu)
//
// Various helper functions for the Eigen library. Some conventions:
//    - A "basis" is always a matrix whose columns are the basis elements.

#ifndef CORE_EIGEN_HELPER_H_
#define CORE_EIGEN_HELPER_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "util.h"

using namespace std;

namespace eigen_helper {
    // Writes an Eigen Matrix object to a binary file.
    template<typename EigenMatrix>
    void binary_write_matrix(const EigenMatrix& matrix,
			     const string &file_path) {
	ofstream file(file_path, ios::out | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	typename EigenMatrix::Index num_rows = matrix.rows();
	typename EigenMatrix::Index num_columns = matrix.cols();
	util_file::binary_write_primitive(num_rows, file);
	util_file::binary_write_primitive(num_columns, file);
	file.write(reinterpret_cast<const char *>(matrix.data()), num_rows *
		   num_columns * sizeof(typename EigenMatrix::Scalar));
    }

    // Reads an Eigen Matrix object from a binary file.
    template<typename EigenMatrix>
    void binary_read_matrix(const string &file_path, EigenMatrix *matrix) {
	ifstream file(file_path, ios::in | ios::binary);
	ASSERT(file.is_open(), "Cannot open file: " << file_path);
	typename EigenMatrix::Index num_rows;
	typename EigenMatrix::Index num_columns;
	util_file::binary_read_primitive(file, &num_rows);
	util_file::binary_read_primitive(file, &num_columns);
	matrix->resize(num_rows, num_columns);
	file.read(reinterpret_cast<char*>(matrix->data()), num_rows *
		  num_columns * sizeof(typename EigenMatrix::Scalar));
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
}  // namespace eigen_helper

#endif  // CORE_EIGEN_HELPER_H_
