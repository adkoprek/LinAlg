from src.mat_vec import mat_vec_mul
from src.types import mat, vec
from src.errors import ShapeMismatchedError
from dataclasses import dataclass
from tests.consts import *
import numpy as np
import pytest
import random

@dataclass
class MatVecMulTestCase:
    a: mat
    v: vec
    result: vec | Exception 
    error: bool = False

def load_mat_vec_mul():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1]).T
        cases.append(MatVecMulTestCase(A, x, np.matmul(A, x)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1] + random.randint(1, 10)).T
        cases.append(MatVecMulTestCase(A, x, ShapeMismatchedError, error=True))

    return cases

@pytest.mark.parametrize("test_case", load_mat_vec_mul())
def test_mat_vec_mul(test_case: MatVecMulTestCase):
    if test_case.error:
        with pytest.raises(test_case.result):
            mat_vec_mul(test_case.a, test_case.v)

    else:
        result = mat_vec_mul(test_case.a, test_case.v)
        np.testing.assert_allclose(result, test_case.result)
