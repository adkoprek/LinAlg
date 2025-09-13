from src.mat_vec import mat_vec_mul
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import *
import numpy as np
import pytest
import json
from tabulate import tabulate

@dataclass
class MatVecMulTestCase:
    a: mat
    v: vec
    result: vec

def load_mat_vec_mul():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix()
        x = random_vector(A.shape[1]).T
        cases.append(MatVecMulTestCase(A, x, np.matmul(A, x)))

    return cases

@pytest.mark.parametrize("test_case", load_mat_vec_mul())
def test_mat_vec_mul(test_case: MatVecMulTestCase):
    result = mat_vec_mul(test_case.a, test_case.v)
    np.testing.assert_allclose(result, test_case.result)
