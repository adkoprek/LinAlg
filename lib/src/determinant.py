from types import mat, vec
from errors import ShapeMismatchedError
from matrix import mat_siz
from qr import qr

def det(a: list[list[float]]) -> float:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    _, R = qr(a)

    det = 1

    for i in range(rows):
        det *= R[i]

    return det
