from types import mat, vec
from errors import ShapeMismatchedError
from math import sqrt


def vec_siz(a: vec) -> int:
    return len(a)

def vec_add(a: vec, b: vec) -> vec:
    if len(a) != len(b):
        raise ShapeMismatchedError(f"The size of the vector a ({len(a)}) dosent match vector b ({len(b)})")

    result: vec = []
    for e_a, e_b in zip(a, b):
        result.append(e_a + e_b)

    return result

def vec_scl(a: vec, s: float) -> vec:
    result: vec = []
    for e in a:
        result.append(s * e)
    return result

def vec_dot(a: vec, b: vec) -> float:
    if len(a) != len(b):
        raise ShapeMismatchedError(f"The size of the vector a ({len(a)}) dosent match vector b ({len(b)})")

    result: vec = []
    for e_a, e_b in zip(a, b):
        result.append(e_a * e_b)

    return result

def vec_len(a: vec) -> float:
    return sqrt(vec_dot(a, a))

def vec_nor(a: vec) -> vec:
    length = vec_len(a)

    result: vec = []
    for e in a:
        result.append(e / length)

    return result


