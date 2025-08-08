from src.types import mat
from src.errors import ShapeMismatchedError
from src.matrix import mat_ide, mat_siz, mat_col, mat_tra
from src.mat_vec import mat_vec_mul
from src.lu import lu, for_sub, bck_sub


def inverse(a: mat) -> mat:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    L, U, P = lu(a)

    I = mat_ide(rows)
    a_inv_t = []

    for i in range(rows):
        b = mat_col(i, i)
        bp = mat_vec_mul(P, b)
        y = for_sub(L, bp)
        x = bck_sub(U, y)
        a_inv_t.append(x)

    return mat_tra(a_inv_t)

