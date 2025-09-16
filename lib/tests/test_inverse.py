from src.inverse import inv
from src.types import mat
from src.errors import SingularError
from dataclasses import dataclass
from tests.consts import * 
import numpy as np
import pytest
import random


@dataclass
class MatInvTestCase:
    a: mat
    result: mat | Exception 
    error: bool = False

def load_inv():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)

        # Very low probability
        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(MatInvTestCase(A, np.linalg.inv(A)))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix(square=True)

        # Make matrix singular
        first_row = random.randint(0, A.shape[0] - 2)
        second_row = first_row + 1
        for i in range(len(A[0])):
            A[first_row][i] = A[second_row][i]

        assert abs(np.linalg.det(A)) < ZERO, "This is an error conserning the testing"

        cases.append(MatInvTestCase(A, SingularError, error=True))

    return cases

@pytest.mark.parametrize("test_case", load_inv())
def test_inv(test_case: MatInvTestCase):
    if test_case.error: 
        with pytest.raises(SingularError):
            inv(test_case.a)

    else:
        result = inv(test_case.a)
        np.testing.assert_allclose(result, test_case.result)

