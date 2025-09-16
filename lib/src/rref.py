from copy import copy
from src.types import mat, vec
from src.matrix import mat_siz
from src.vector import vec_scl, vec_add
from src.consts import *


def rref(a: mat) -> mat:
    rref = copy(a)
    rows, cols = mat_siz(a)

    row = 0
    for col in range(cols):
        if row >= rows:
            break

        pivot_row = max(range(row, rows), key=lambda r: abs(rref[r][col]))

        if abs(rref[pivot_row][col]) < UNSTABLE_ZERO:
            continue 

        rref[row], rref[pivot_row] = rref[pivot_row], rref[row]

        rref[row] = vec_scl(rref[row], 1 / rref[row][col])

        for r in range(rows):
            if r != row:
                rref[r] = vec_add(rref[r], vec_scl(rref[row], -rref[r][col]))

        row += 1

    return rref
    
