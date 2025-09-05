from src.types import mat, vec
from src.errors import ShapeMismatchedError, SingularError
from src.matrix import mat_siz, mat_ide, mat_tra
from src.mat_vec import mat_vec_mul
from copy import copy


def lu(a: mat) -> tuple[mat, mat, mat]:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    L: mat = mat_ide(rows)
    U: mat = copy(a)
    P: mat = mat_ide(rows)

    for i in range(rows):
        pivot = max(range(i, rows), key=lambda r: abs(U[r][i]))

        if U[pivot][i] == 0:
            raise SingularError("The matrix a is singular")
        
        L[i][:i], L[pivot][:i] = L[pivot][:i], L[i][:i]
        U[i], U[pivot] = U[pivot], U[i]
        P[i], P[pivot] = P[pivot], P[i]

        for j in range(i + 1, rows):
            fac = U[j][i] / U[i][i]            
            L[j][i] = fac

            for k in range(i, cols):
                U[j][k] -= fac * U[i][k]

    return (L, U, P)

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
    L, U, P = lu(a)
    print(L, U, P, b)
    bp = mat_vec_mul(mat_tra(P), b)
    print(bp)
    y = for_sub(L, bp)
    print(y)
    x = bck_sub(U, y)
    print(x)
    return x

