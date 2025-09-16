from src.lu import lu, solve
from src.types import mat, vec
from src.errors import SingularError, ShapeMismatchedError
from dataclasses import dataclass
from tests.consts import * 
import numpy as np
import pytest
import random


@dataclass
class LUTestCase:
    A: mat

@dataclass
class SolveTestCase:
    A: mat
    b: vec
    error: None | Exception = None

def load_lu():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)

        # Very low probability
        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(LUTestCase(A))

    return cases

def load_solve():
    cases = []
    for _ in range(TEST_CASES):
        A = random_matrix(square=True)
        b = random_vector(A.shape[0])

        # Very low probability
        if abs(np.linalg.det(A)) < ZERO:
            continue

        cases.append(SolveTestCase(A, b))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix(square=True)
        b = random_vector(A.shape[0])

        # Make matrix singular
        first_row = random.randint(0, A.shape[0] - 2)
        second_row = first_row + 1
        for i in range(len(A[0])):
            A[first_row][i] = A[second_row][i]

        assert abs(np.linalg.det(A)) < ZERO, "This is an error conserning the testing"

        cases.append(SolveTestCase(A, b, SingularError))

    for _ in range(ERROR_TEST_CASES):
        A = random_matrix()
        b = random_vector(A.shape[0])

        # Unlikly
        if A.shape[0] == A.shape[1]:
            continue

        cases.append(SolveTestCase(A, b, ShapeMismatchedError))

    return cases

@pytest.mark.parametrize("test_case", load_lu())
def test_lu(test_case: LUTestCase):
    A = test_case.A

    L, U, P = lu(A)
    Lm = np.array(L)
    Um = np.array(U)
    Pm = np.array(P)

    n = A.shape[0]
    assert Lm.shape == (n, n), "Lm must be same shape as A"
    assert Um.shape == (n, n), "Um must be same shape as A"
    assert Pm.shape == (n, n), "Pm must be same shape as A"

    assert np.allclose(Pm @ Pm.T, np.eye(n), atol=ZERO), "Pm is not orthogonal (not permutation)"
    assert np.allclose(np.sum(Pm, axis=0), 1), "Each column of Pm must have exactly one 1"
    assert np.allclose(np.sum(Pm, axis=1), 1), "Each row of Pm must have exactly one 1"

    assert np.allclose(np.tril(Lm), Lm, atol=ZERO), "Lm must be lower triangular"
    assert np.allclose(np.diag(Lm), np.ones(n), atol=ZERO), "Diagonal of Lm must be all ones"

    assert np.allclose(np.triu(Um), Um, atol=ZERO), "Um must be upper triangular"

    assert np.allclose(Pm @ A, Lm @ Um, atol=ZERO), "Decomposition check failed: Pm*A != Lm*Um"

@pytest.mark.parametrize("test_case", load_solve())
def test_solve(test_case: SolveTestCase):
    if test_case.error == SingularError: 
        with pytest.raises(SingularError):
            solve(test_case.A, test_case.b)

    elif test_case.error == ShapeMismatchedError: 
        with pytest.raises(ShapeMismatchedError):
            solve(test_case.A, test_case.b)

    else:
        A = test_case.A
        b = test_case.b
        x = solve(test_case.A, test_case.b)

        A = np.asarray(A, dtype=float)
        x = np.asarray(x, dtype=float)
        b = np.asarray(b, dtype=float)

        m, n = A.shape
        assert x.shape == (n,), f"x must be shape ({n},), got {x.shape}"
        assert b.shape == (m,), f"b must be shape ({m},), got {b.shape}"

        np.testing.assert_allclose(A @ x, b, atol=ZERO)
