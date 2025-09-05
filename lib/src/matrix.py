from src.errors import ShapeMismatchedError
from src.types import mat, vec
from src.vector import vec_dot
from copy import copy


def mat_ide(size: int) -> mat:
    result: mat = [[0 for _ in range(size)] for _ in range(size)]

    for i in range(size):
        result[i][i] = 1

    print("Result", result)
    return result


def mat_siz(a: mat) -> tuple[int]:
    return (len(a), len(a[0]))

def mat_scl(a: mat, s: int) -> mat:
    result: vec = []
    for row in a:
        result.append([])
        for e in row:
            result[-1].append(e * s)

    return result

def mat_add(a: mat, b: mat) -> mat:
    size_a = mat_siz(a)
    size_b = mat_siz(b)

    if size_a != size_b:
        raise ShapeMismatchedError(f"The size of the matrix a ({size_a}) does not match the size of b ({size_b})")

    result = []
    for row_a, row_b in zip(a, b):
        result.append([])
        for e_a, e_b in zip(row_a, row_b):
            result[-1].append(e_a + e_b)

    return result

def mat_sub(a: mat, b: mat) -> mat:
    return mat_add(a, mat_scl(b, -1))

def mat_col(a: mat, index: int) -> vec:
    cols, _ = mat_siz(a)

    col: vec = []
    for i in range(cols):
        col.append(a[i][index])

    return col

def mat_row(a: mat, index: int) -> vec:
    return copy(a[index])

def mat_mul(a: mat, b: mat) -> mat:
    size_a = mat_siz(a) 
    size_b = mat_siz(b)

    if size_a[1] != size_b[0]:
        raise ShapeMismatchedError(
            f"The number of columns of a ({size_a[1]}) does not match the number of rows of b ({size_b[0]})"
        )

    r_rows = size_a[0]
    r_cols = size_b[1]
    result: mat = [[0 for _ in range(r_cols)] for _ in range(r_rows)]


    for i in range(r_cols):
        for j in range(r_rows):
            row = mat_row(a, j)
            col = mat_col(b, i)
            result[j][i] = vec_dot(row, col)

    return result

def mat_tra(a: mat) -> mat:
    cols, rows = mat_siz(a)
    result: mat = [[0 for _ in range(cols)] for _ in range(rows)]

    for col in range(cols):
        for row in range(rows):
            result[row][col] = a[col][row]
    
    return result

