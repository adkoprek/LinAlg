from src.types import mat, vec
from src.errors import ShapeMismatchedError, SingularError
from src.matrix import mat_siz, mat_ide, mat_tra
from src.mat_vec import mat_vec_mul
from copy import copy


def swap_rows(a: mat, i: int, j: int) -> mat:
    rows, cols = mat_siz(a)

    for k in range(cols):
        temp = a[i][k]
        a[i][k] = a[j][k]
        a[j][k] = temp

def lu(a: mat) -> tuple[mat, mat, mat]:
    rows, cols = mat_siz(a)

    U = copy(a)
    L = [[0]*cols for _ in range(rows)]
    P = mat_ide(rows)

    for i in range(rows):
        pivot = max(range(i, rows), key=lambda r: abs(U[r][i]))

        if pivot != i:
            swap_rows(U, i, pivot)
            swap_rows(P, i, pivot)

            L[i][:i], L[pivot][:i] = L[pivot][:i], L[i][:i]

        for j in range(i+1, rows):
            if U[i][i] == 0:
                continue

            fac = U[j][i] / U[i][i]
            L[j][i] = fac

            for k in range(i, cols):
                U[j][k] -= fac * U[i][k]

        L[i][i] = 1.0

    return L, U, P

def for_sub(l: mat, b: vec) -> vec:
    rows, _ = mat_siz(l) 
    y = []

    for i in range(rows):
        temp = b[i]
        for col in range(0, i):
            temp -= y[col] * l[i][col]
        y.append(temp / l[i][i])

    return y
        
def bck_sub(u: mat, y: vec) -> vec:
    rows, cols = mat_siz(u) 
    x = []

    for i in range(rows - 1, -1, -1):
        temp = y[i]
        for col in range(rows - 1, i, -1):
            temp -= x[cols - col - 1] * u[i][col]
        x.append(temp / u[i][i])

    x.reverse()
    return x

def solve(a: mat, b: vec) -> vec:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    L, U, P = lu(a)
    bp = mat_vec_mul(P, b)
    y = for_sub(L, bp)
    x = bck_sub(U, y)
    return x

