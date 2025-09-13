from src.inverse import inv
from src.errors import ShapeMismatchedError, SingularError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import * 
import numpy as np
import pytest
import json


@dataclass
class MatInvTestCase:
    a: mat
    result: mat | str

def load_inv():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)

        # Very low probability
        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(MatInvTestCase(A, np.linalg.inv(A)))

    return cases

@pytest.mark.parametrize("test_case", load_inv())
def test_inv(test_case: MatInvTestCase):
    result = inv(test_case.a)
    np.testing.assert_allclose(result, test_case.result)

