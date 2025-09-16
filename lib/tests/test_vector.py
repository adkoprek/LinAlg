from src.vector import vec_add, vec_scl, vec_dot, vec_len, vec_nor
from src.types import vec
from src.errors import ShapeMismatchedError
from dataclasses import dataclass, field
from tests.consts import *
import numpy as np
import pytest
import random

@dataclass
class VecAddTestCase:
    a: vec
    b: vec
    result: vec | Exception
    error: bool = False

@dataclass
class VecSclTestCase:
    a: vec
    s: float
    result: vec

@dataclass
class VecDotTestCase:
    a: vec
    b: vec
    result: float | Exception 
    error: bool = False

@dataclass
class VecLenTestCase:
    a: vec
    result: float

@dataclass
class VecNorTestCase:
    a: vec
    result: vec 


def load_vec_add():
    cases = []
    for _ in range(TEST_CASES):
        a = random_vector()
        b = random_vector(a.shape)
        cases.append(VecAddTestCase(a, b, a + b))

    for _ in range(ERROR_TEST_CASES):
        a = random_vector()
        diff_shape = (a.shape[0] + random.randint(1, 10))
        b = random_vector(diff_shape)
        cases.append(VecAddTestCase(a, b, ShapeMismatchedError, error=True))

    return cases

def load_vec_scl():
    cases = []
    for _ in range(TEST_CASES):
        a = random_vector()
        s = randint(1, 100)
        cases.append(VecSclTestCase(a, s, s * a))

    return cases

def load_vec_dot():
    cases = []
    for _ in range(TEST_CASES):
        a = random_vector()
        b = random_vector(a.shape)
        cases.append(VecDotTestCase(a, b, np.dot(a, b)))

    for _ in range(ERROR_TEST_CASES):
        a = random_vector()
        diff_shape = (a.shape[0] + random.randint(1, 10))
        b = random_vector(diff_shape)
        cases.append(VecDotTestCase(a, b, ShapeMismatchedError, error=True))

    return cases

def load_vec_len():
    cases = []
    for _ in range(TEST_CASES):
        a = random_vector()
        cases.append(VecLenTestCase(a, np.linalg.norm(a)))

    return cases

def load_vec_nor():
    cases = []
    for _ in range(TEST_CASES):
        a = random_vector()
        cases.append(VecLenTestCase(a, a / np.linalg.norm(a)))

    return cases

@pytest.mark.parametrize("test_case", load_vec_add())
def test_vec_add(test_case: VecAddTestCase):
    if test_case.error:
        with pytest.raises(test_case.result):
            vec_add(test_case.a, test_case.b)

    else:
        result = vec_add(test_case.a, test_case.b)
        np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_vec_scl())
def test_vec_scl(test_case: VecSclTestCase):
    result = vec_scl(test_case.a, test_case.s)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_vec_dot())
def test_vec_dot(test_case: VecDotTestCase):
    if test_case.error:
        with pytest.raises(test_case.result):
            vec_dot(test_case.a, test_case.b)

    else:
        result = vec_dot(test_case.a, test_case.b)
        np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_vec_len())
def test_vec_len(test_case: VecLenTestCase):
    result = vec_len(test_case.a)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_vec_nor())
def test_vec_nor(test_case: VecNorTestCase):
    result = vec_nor(test_case.a)
    np.testing.assert_allclose(result, test_case.result)
