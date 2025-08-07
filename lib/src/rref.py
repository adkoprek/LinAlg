from types import mat, vec
from copy import copy
from matrix import mat_siz


TOLERANCE = 1e-12

def rref(a: mat) -> mat:
    rref = copy(a)
    cols, rows = mat_siz(a)

    row = 0
    for col in range(cols):
        if row >= rows:
            break

        pivot_row = max(range(row, rows), key=lambda r: abs(rref[r][col])) + row
        if abs(rref[pivot_row, col]) < TOLERANCE:
            continue 

        rref[[row, pivot_row]] = rref[[pivot_row, row]]

        rref[row] = rref[row] / rref[row, col]

        for r in range(rows):
            if r != row:
                rref[r] -= rref[r, col] * rref[row]

        row += 1

    return rref
    
