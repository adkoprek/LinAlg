import numpy as np
from src.types import mat, vec
from random import randint


DATA_PATH = "./lib/data"
TEST_CASES = 50
ERROR_TEST_CASES = 5
ZERO = 1e-15
UNSTABLE_ZERO = 1e-6


def random_matrix(shape: tuple[int, int] = None, square: bool = False) -> mat:
    if shape == None:
        rows = randint(2, 10)
        cols = randint(2, 10)
        if square:
            rows = cols
        return np.random.random((rows, cols))

    return np.random.random(shape)

def random_vector(shape: int = None) -> vec:
    if shape == None:
        length = randint(2, 10)
        return np.random.random(length)

    return np.random.random(shape)

