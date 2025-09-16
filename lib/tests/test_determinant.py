from src.determinant import det
from src.types import mat
from src.errors import ShapeMismatchedError
from dataclasses import dataclass
from tests.consts import *
import numpy as np
import pytest


@dataclass
class DeterminantTestCase:
    a: mat
    result: float | Exception
    error: bool = False

def load_determinant():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)
        cases.append(DeterminantTestCase(A, np.linalg.det(A)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()

        # Unlikly
        if A.shape[0] == A.shape[1]:
            continue

        cases.append(DeterminantTestCase(A, ShapeMismatchedError, error=True))


    return cases

@pytest.mark.parametrize("test_case", load_determinant())
def test_determinant(test_case: DeterminantTestCase):
    if test_case.error: 
        with pytest.raises(ShapeMismatchedError):
            det(test_case.a)

    else:
        result = det(test_case.a)
        assert abs(result - test_case.result) < ZERO
