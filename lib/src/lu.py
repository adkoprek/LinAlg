from types import mat, vec
from errors import ShapeMismatchedError, SingularError
from matrix import mat_siz, mat_ide, mat_tra
from mat_vec import mat_vec_mul
from copy import copy


def lu(a: mat) -> tuple[mat, mat, mat]:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    L: mat = mat_ide(rows)
    U: mat = copy(a)
    P: mat = rows * [0 for _ in range(cols)]

    for i in range(rows):
        pivot = max(range(i, rows), key=lambda r: abs(U[r][i]))

        if U[pivot][i] == 0:
            raise SingularError("The matrix a is singular")
        
        L[i][:i], L[pivot][:i] = L[pivot][:i], L[i][:i]
        U[i], U[pivot] = U[pivot], U[i]
        P[pivot][i] = 1

        for j in range(i, rows):
            fac = a[j][i] / a[i][i]            
            L[i][i] = fac

            for k in range(i, cols):
                U[j][k] -= fac * U[i][k]

    return (L, U, P)

def ldu(a: mat) -> tuple[mat, mat, mat, mat]:
    rows, cols = mat_siz(a)
    L, U, P = lu(a)
    D: mat = rows * [0 for _ in range(cols)]

    for i in range(rows):
        D[i][i] = U[i][i]
        fac = U[i][i]
        for j in range(i, cols):
            U[i][j] /= fac

    return (L, D, U, P)

def for_sub(l: mat, b: vec) -> vec:
    rows, _ = mat_siz(l) 
    y = []

    for i in range(rows):
        temp = b[i]
        for col in range(0, i - 1):
            temp -= y[col] * l[i][col]
        y.append(temp / l[i][i])

    return y
        
def bck_sub(u: mat, y: vec) -> vec:
    rows, _ = mat_siz(u) 
    y = []

    for i in range(rows - 1, 0, -1):
        temp = y[i]
        for col in range(rows - 1, i - 1, -1):
            temp -= y[col] * u[i][col]
        y.append(temp / u[i][i])

def solve(a: mat, b: vec) -> vec:
    L, U, P = lu(a)
    bp = mat_vec_mul(mat_tra(P), b)
    y = for_sub(L, bp)
    x = bck_sub(U, y)
    return x

