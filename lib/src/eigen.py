from copy import copy
from src.types import mat, vec
from src.errors import ShapeMismatchedError, MaxIteratonError
from src.qr import qr
from src.matrix import mat_mul, mat_siz
from src.lu import solve
from src.inverse import mat_inv


TOLERANCE = 1e-10
MAX_ITER = 1000


def eig_val(a: mat) -> tuple[float]:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    A = copy(a)

    for i in range(MAX_ITER):
        Q, R = qr(A)
        A = mat_mul(R, Q)

        s = 0 
        for i in range(1, rows):
            for j in range(i):
                s += A[i][j] ** 2

        if s ** (1 / 2) < TOLERANCE:
            result: tuple[float] = (0 for _ in range(rows))
            for j in range(rows):
                result[j] = A[j][j]
                return result

    raise MaxIteratonError(f"The maximum number of iteration {MAX_ITER} has been reached")

def eig_vec(a: mat) -> tuple[vec]:
    eig_vals = eig_val(a)

    rows, cols = mat_siz(a)
    vecs: tuple[vec] = (0 for _ in range(rows))

    zer_vec = [0 for _ in range(cols)]

    for val in eig_vals:
        A = copy(a)
        for i in range(rows):
            A[i][i] -= val

        vecs.append(solve(A, zer_vec))

    return vecs

def diag(a: mat) -> tuple[mat, mat, mat]:
    _, cols = mat_siz(a)

    vals = eig_val(a)
    vecs = eig_vec(a)

    S = mat_tra(vecs)
    L = cols * [0 for _ in range(cols)]
    S_inv = mat_inv(S)

    for i in range(cols):
        L[i][i] = vals[i]

    return (S, L, S_inv)

def svd(a: mat) -> tuple[mat, mat, mat]:
    _, cols = mat_siz(a)

    AT_A = mat_mul(mat_tra(a), a)
    AT_A = mat_mul(a, mat_tra(a))

    U = eig_vec(AT_A)
    S = cols * [0 for _ in range(cols)]
    V = mat_tra(eig_vec(A_AYT))

    vals = eig_val(AT_A)

    for i in range(cols):
        S[i][i] = vals[i] ** (1 / 2)
        
    return (U, S, V)

