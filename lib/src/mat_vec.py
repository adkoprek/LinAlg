from src.matrix import mat_siz, mat_row
from src.vector import vec_dot
from src.errors import ShapeMismatchedError
from src.types import mat, vec


def mat_vec_mul(a: list[list[float]], b: list[float]) -> list[float]:
    _, rows = mat_siz(a)
    if rows != len(b):
        raise ShapeMismatchedError(f"The number of rows ({rows}) in a does not match the length ({len(b)}) of vector b")

    result: vec = []

    for i in range(rows):
        row = mat_row(a, i)
        result.append(vec_dot(row, b))

    return result

