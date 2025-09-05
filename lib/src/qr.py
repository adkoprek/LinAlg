from copy import copy
from src.types import mat, vec
from src.vector import *
from src.matrix import *
from src.inverse import inv

TOLERANCE = 1e-10


def vec_prj(a: vec, b: vec) -> vec:
    f = vec_dot(a, b) / vec_dot(a, a)
    return vec_scl(a, f)

def mat_prj(a: mat) -> mat:
    return mat_mul(
                mat_mul(
                    a,
                    inv(
                        mat_mul(
                            mat_tra(a),
                            a
                        )
                    )
                ),
                mat_tra(a)
            )

def ortho(vecs: list[vec], new: vec, show_factors: bool = False) -> vec | tuple[vec, list[float]]:
    result = copy(new)
    factors: list[float] = []

    for o_vec in vecs:
        f = vec_dot(o_vec, result) / vec_dot(o_vec, o_vec)
        factors.append(f)
        result = vec_add(result, vec_scl(o_vec, -f))

    if show_factors:
        return (result, factors)
    else:
        return result

def ortho_base(vecs: list[vec]) -> tuple[vec]:
    result = [copy(vecs[0])]
    
    for i in range(1, len(vecs)):
        result.append(ortho(result, vecs[i]))

    return tuple(result)

def qr(a: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    rows, cols = mat_siz(a)
    R = copy(a)
    Q = mat_ide(rows)

    for i in range(min(rows - 1, cols)):
        x = copy([row[i] for row in R[i:]])

        length = vec_len(x)
        if length == 0:
            continue

        if x[0] == 0:
            alpha = -length

        elif x[0] < 1:
            alpha = length

        else:
            alpha = -length

        e1 = [0 for _ in range(len(x))]
        e1[0] = 1

        u = vec_add(x, vec_scl(e1, -alpha))

        length = vec_len(u)
        if length < TOLERANCE:
            continue

        v = vec_nor(u)

        H_sub = mat_add(mat_ide(len(v)), mat_scl(mat_mul(mat_tra([v]), [v]), - 2))
        H_full = mat_ide(rows)
        for row_idx, row in enumerate(H_sub):
            for col_idx, val in enumerate(row):
                H_full[i + row_idx][i + col_idx] = val

        R = mat_mul(H_full, R)
        Q = mat_mul(Q, mat_tra(H_full))

    return (Q, R)

