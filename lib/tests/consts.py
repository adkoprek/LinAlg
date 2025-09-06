import numpy as np
from random import randint


DATA_PATH = "./lib/data"
TEST_CASES = 50


def random_matrix(shape: tuple[int, int] = None):
    if shape == None:
        rows = randint(1, 10)
        cols = randint(1, 10)
        return np.random.random((rows, cols))

    return np.random.random(shape)

def random_vector(shape: int = None):
    if shape == None:
        length = randint(1, 10)
        return np.random.random(length)

    return np.random.random(shape)

