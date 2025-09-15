# Project Structure

## Data

This directory contains tests for library functions encoded in the JSON format with the name `<lib file>.json`.

- Numpy does not contain functions for gramm-schmidt operations so this are complemented in the `qr.json` file.
- Numpy does not contain a function for a rref decomposition and with not wanting to incorperate non standart libraries the `rref.json` file contains special test cases for an rref fcuntion.

## SRC

This directory contains the library source files.

- The `types.py` file contains custom data types used by the library
  
  - The `mat` type is 2d array of row arrays that represents a matrix
  
  - The `vec` type is a array that represents a vector

- The `errors.py` file contains cutom errors used by the library
  
  - The `ShapeMismatchedError` is raised when the dimensions of matrices or vector are not correct
  
  - The `SingularError` is raised when a calculation can not be completed because a matrix is singular

- The `vector.py` file contains functions to manipulate vectors
  
  - The function `vec_add` adds two vectors
    - **Parameters:**
      - First vector
      - Second vector
    - **Return:**
      - Sum of vectors
    - **Errors:**
      - ShapeMismatchedError when dimensions are not equal
  - The function `vec_scl` scales a vector by a real number
    - **Parameters:**
      - First vector
      - Scaler
    - **Return:**
      - Scaled Vector
  - The function `vec_len` returns the length of the vector
    - **Parameters:**
      - Vector
    - **Return:**
      - Length
  - The function `vec_nor` normalizes a vector
    - **Parameters:**
      - Vector
    - **Return:**
      - Normalized Vector
    - **Errors:**
      - ZeroDivisionError when vector has length 0

- The `matrix.py` file contains functions to manipulate matrices
  
  - The function `mat_ide` returns the identity matrix
    - **Parameters:**
      - Size of the matrix
    - **Return:**
      - The identity matrix
  - The function `mat_siz` returns the size of a matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Number of rows
      - Number of columns
  - The function `mat_scl` scales a matrix by a scalar
    - **Parameters:**
      - Matrix
      - Scalar
    - **Return:**
      - Scaled matrix
  - The function `mat_add` adds two matrices
    - **Parameters:**
      - First matrix
      - Second matrix
    - **Return:**
      - Sum of matrices
    - **Errors:**
      - ShapeMismatchedError when dimensions are not equal
  - The function `mat_col` returns the ith column of a matrix
    - **Parameters:**
      - First matrix
      - Index of column
    - **Return:**
      - Column
  - The function `mat_row` returns the jth row of a matrix
    - **Parameters:**
      - First matrix
      - Index of row
    - **Return:**
      - Row
  - The function `mat_mul` multiplies two matrices
    - **Parameters:**
      - First matrix
      - Second matrix
    - **Return:**
      - Product
    - **Errors:**
      - ShapeMismatchedError when number of rows of the first matrix and number of columns of second matrix do not match
  - The function `mat_tra` returns the transpose of a matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Transposed matrix

- The `mat_vec.py` file contains operations between matrices and vectors
  
  - The function `mat_vec_mul` multiplies a matrix by a vector
    - **Parameters:**
      - Matrix
      - Vector
    - **Return:**
      - The product vector
    - **Errors:**
      - ShapeMismatchedError when the number of columns do not match the dimensionality of the vector

- The `lu.py` file contains operations for lu decompositions and solving of linear equations
  
  - The function `lu` computes the lu decomposition for a matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Lower triangular matrix
      - Upper triangular matrix
      - Permutation matrix
  - The function `solve` solves Ax=b
    - **Parameters:**
      - Matrix
      - Vector b
    - **Return:**
      - Vector x
    - **Errors:**
      - ShapeMismatchedError when the matrix is not square
      - SingularError when the matrix is singular to machine precision

- The `inverse.py` file contains functions for inverse calculations
  
  - The function `inv` computes the inverse of a matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Inverse of matrix
    - **Errors:**
      - SingularError when the matrix is singular

- The `rref.py` file contains operations for lu decompositions and solving of linear equations
  
  - The function `rref` computes the row reduced echelon form of a matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Row reduced echelon form

- The `qr.py` files contains functions conserning orthonormalization for vector and matrices
  
  - The function `vec_prj` computes the projection of two vectors
    - **Parameters:**
      - Vector to project onto
      - Vector to project
    - **Return:**
      - Projected vector
    - **Errors:**
      - ShapeMismatchedError when the dimensions do not match
  - The function `mat_prj` computes the projection matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Projection matrix
    - **Errors:**
      - SingularError when the matrix has dependent columns
  - The function `ortho` orthogonolizes a vector in terms of a set of orthogonalized vectors using gramm-schmid
    - **Parameters:**
      - Orthogonormalized Vectors
      - Vector to orthonormalize
      - Optinoal `show_factors` to aswell return the factors for the substituted vectors in gramm-schmid
    - **Return:**
      - Orthonormalized vector if `show_factors=False`
      - Orthonormalized vector and factors if `show-factors=True`
  - The function `ortho_base` orthogonalizes a set of vectors in respect to the first vector
    - **Parameters:**
      - Set of vectors
    - **Return:**
      - Orthonormalized vectors
  - The function `qr` computes the qr decomposition of a matrix using householder reflections
    - **Parameters:**
      - Matrix
    - **Return:**
      - Orthonormal matrix Q
      - Upper triangular matrix R

- The `determinant.py` file contains functions for computing the determinant
  
  - The function `det` computes the terminant of a matrix
    - **Parameters:**
      - Matrix
    - **Return:**
      - Determinant
    - **Errors:**
      - ShapeMismatchedError when the matrix is not square

## Tests

This directory contains tests for library functions with the name `test_<lib file>.py`. The tests can be started from the root directory using the 'pytest' command.

Most data is rndomized using numpy matrices except for the ones mentioned in the data part.

- The `const.py` file contains important constants for the testing process
  
  - DATA_PATH is the relative path from the root directory to the `.json` test cases
  
  - TEST_CASES is the number of randomized cases to execute per function
  
  - ZERO is the threshold from which a number is considered zero
  
  - UNSTABLE_ZERO is the threshold from which a number is considered zero for unstable operations like inverses 
