from src.vector import vec_add, vec_scl, vec_dot, vec_len, vec_nor
from src.errors import ShapeMismatchedError
from src.types import vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json

@dataclass
class VecAddTestCase:
    a: vec
    b: vec
    result: vec | str 

@dataclass
class VecSclTestCase:
    a: vec
    s: float
    result: vec

@dataclass
class VecDotTestCase:
    a: vec
    b: vec
    result: float | str

@dataclass
class VecLenTestCase:
    a: vec
    result: float

@dataclass
class VecNorTestCase:
    a: vec
    result: vec


data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/vector.json", 'r') as file:
        data = json.load(file)

def load_vec_add():
    load_data()
    return [VecAddTestCase(tc["a"], tc["b"], tc["result"]) for tc in data["vec_add"]]

def load_vec_scl():
    load_data()
    return [VecSclTestCase(tc["a"], tc["s"], tc["result"]) for tc in data["vec_scl"]]

def load_vec_dot():
    load_data()
    return [VecDotTestCase(tc["a"], tc["b"], tc["result"]) for tc in data["vec_dot"]]

def load_vec_len():
    load_data()
    return [VecLenTestCase(tc["a"], tc["result"]) for tc in data["vec_len"]]

def load_vec_nor():
    load_data()
    return [VecNorTestCase(tc["a"], tc["result"]) for tc in data["vec_nor"]]

@pytest.mark.parametrize("test_case", load_vec_add())
def test_vec_add(test_case: VecAddTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
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
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
            vec_dot(test_case.a, test_case.b)
    else:
        result = vec_dot(test_case.a, test_case.b)
        assert abs(result - test_case.result) < 1e-9

@pytest.mark.parametrize("test_case", load_vec_len())
def test_vec_len(test_case: VecLenTestCase):
    result = vec_len(test_case.a)
    assert abs(result - test_case.result) < 1e-9

@pytest.mark.parametrize("test_case", load_vec_nor())
def test_vec_nor(test_case: VecNorTestCase):
    result = vec_nor(test_case.a)
    np.testing.assert_allclose(result, test_case.result)
