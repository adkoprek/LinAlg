from src.determinant import det
from src.errors import ShapeMismatchedError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import *
import numpy as np
import pytest
import json


@dataclass
class DeterminantTestCase:
    a: mat
    result: float | str 

def load_determinant():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)
        cases.append(DeterminantTestCase(A, np.linalg.det(A)))

    return cases

@pytest.mark.parametrize("test_case", load_determinant())
def test_determinant(test_case: DeterminantTestCase):
    result = det(test_case.a)
    assert abs(result - test_case.result) < ZERO
