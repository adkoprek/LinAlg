from copy import copy
from src.types import mat, vec
from src.vector import vec_dot, vec_scl, vec_len
from src.matrix import mat_mul, mat_tra, mat_siz, mat_col
from src.inverse import mat_inv


def vec_prj(a: vec, b: vec) -> vec:
    f = vec_dot(a, b) / vec_dot(a, a)
    return vec_scl(a, f)

def mat_prj(a: mat) -> mat:
    return mat_mul(
            mat_tra(a), 
            mat_mul(
                a, 
                mat_inv(mat_mul(a, mat_tra(a))))
            )

def ortho(vecs: list[vec], new: vec, show_factors: bool = False) -> vec | tuple[vec, list[float]]:
    result = copy(new)
    factors: list[float] = []

    for vec in vecs:
        f = vec_dot(vec, result) / vec_dot(vec, vec)
        factors.append(f)
        resilt -= vec_scl(f, vec)

    if show_factors:
        return (result, factors)
    else:
        return result

def ortho_base(vecs: list[vec]) -> tuple[vec]:
    result = copy(vecs)
    
    for i in range(1, len(vecs)):
        result[i] = ortho(result, vecs[i])

    return result

def qr(a: mat) -> tuple[mat]:
    _, cols = mat_siz(a)
    Q_T: mat = []
    R_R: mat = []

    Q_T.append(mat_col(a, 0))

    for i in range(1, cols):
        col = mat_col(a, i)
        q_vec, factors = ortho(Q_T, col, True)
        mag = vec_len(q_vec)

        Q_T.append(q_vec)
        R_R.append([mag] + factors) 

    return (mat_tra(Q_T), R_R.flip())

