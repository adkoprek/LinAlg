from copy import copy
from src.types import mat, vec
from src.vector import vec_dot, vec_scl, vec_add, vec_nor, vec_len
from src.matrix import mat_mul, mat_tra, mat_siz, mat_col
from src.inverse import inv


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
    rows, cols = mat_siz(a)  # m, n
    Q: list[list[float]] = [[0.0 for _ in range(rows)] for _ in range(rows)] 
    R: list[list[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]  

    for j in range(cols):
        v = mat_col(a, j)

        # Subtract projections of v on all previous q_i
        for i in range(j):
            o_col = mat_col(Q, i)
            R[i][j] = vec_dot(o_col, v)
            v = vec_add(v, vec_scl(o_col, -R[i][j]))

        if j < rows:
            R[j][j] = vec_len(v)
            if R[j][j] == 0:
                qj = [0.0] * rows

            else:
                qj = vec_nor(v)
                if R[j][j] < 0:
                    R[j][j] = -R[j][j]
                    qj = vec_scl(qj, -1)

            # Store q_j as the j-th column of Q
            for i in range(rows):
                Q[i][j] = qj[i]

    return Q, R

