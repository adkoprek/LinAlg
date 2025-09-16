from src.matrix import *
from src.errors import ShapeMismatchedError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import *
from random import randint
import numpy as np
import pytest
import json


@dataclass
class MatIdeTestCase:
    size: int
    result: mat

@dataclass
class MatSizTestCase:
    a: mat
    result: tuple[int]

@dataclass
class MatColTestCase:
    a: mat
    col_index: int
    result: vec

@dataclass
class MatRowTestCase:
    a: mat
    row_index: int
    result: vec

@dataclass
class MatAddTestCase:
    a: mat
    b: mat
    result: mat

@dataclass
class MatMulTestCase:
    a: mat
    b: mat
    result: mat

@dataclass
class MatSclTestCase:
    a: mat
    s: float
    result: mat

@dataclass
class MatTransTestCase:
    a: mat
    result: mat

def load_mat_ide():
    cases = []
    for i in range(TEST_CASES):
        size = randint(1, 10)
        correct = np.eye(size)
        cases.append(MatIdeTestCase(size, np.eye(size)))

    return cases

def load_mat_siz():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        cases.append(MatSizTestCase(A, A.shape))

    return cases

def load_mat_col():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        col = A.shape[1] - 1
        cases.append(MatColTestCase(A, col, A[:, col]))

    return cases

def load_mat_row():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        row = A.shape[0] - 1
        cases.append(MatRowTestCase(A, row, A[row, :]))

    return cases

def load_mat_add():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        B = random_matrix(A.shape)
        AsB = A + B
        cases.append(MatAddTestCase(A, B, AsB))

    return cases

def load_mat_mul():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        B = random_matrix(A.T.shape)
        AmB = A @ B
        cases.append(MatMulTestCase(A, B, AmB))

    return cases

def load_mat_scl():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        s = randint(0, 100)
        As = s * A
        cases.append(MatSclTestCase(A, s, As))

    return cases

def load_mat_trans():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        AT = A.T
        cases.append(MatTransTestCase(A, AT))

    return cases


@pytest.mark.parametrize("test_case", load_mat_ide())
def test_mat_ide(test_case: MatIdeTestCase):
    result = mat_ide(test_case.size)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_siz())
def test_mat_siz(test_case: MatSizTestCase):
    result = mat_siz(test_case.a)
    assert result == test_case.result

@pytest.mark.parametrize("test_case", load_mat_col())
def test_mat_col(test_case: MatColTestCase):
    result = mat_col(test_case.a, test_case.col_index)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_row())
def test_mat_row(test_case: MatRowTestCase):
    result = mat_row(test_case.a, test_case.row_index)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_add())
def test_mat_add(test_case: MatAddTestCase):
    result = mat_add(test_case.a, test_case.b)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_mul())
def test_mat_mul(test_case: MatMulTestCase):
    result = mat_mul(test_case.a, test_case.b)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_scl())
def test_mat_scl(test_case: MatSclTestCase):
    result = mat_scl(test_case.a, test_case.s)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_trans())
def test_mat_trans(test_case: MatTransTestCase):
    result = mat_tra(test_case.a)
    np.testing.assert_allclose(result, test_case.result)

