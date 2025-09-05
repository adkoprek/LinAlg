from src.matrix import mat_ide, mat_siz, mat_scl, mat_col, mat_row, mat_add, mat_sub, mat_mul, mat_tra
from src.errors import ShapeMismatchedError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
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
    result: tuple[int, ...]

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
class MatSubTestCase:
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

data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/matrix.json", 'r') as file:
        data = json.load(file)

def load_mat_ide():
    load_data()
    return [MatIdeTestCase(tc["size"], tc["result"]) for tc in data["mat_ide"]]

def load_mat_siz():
    load_data()
    return [MatSizTestCase(tc["a"], tuple(tc["result"])) for tc in data["mat_siz"]]

def load_mat_col():
    load_data()
    return [MatColTestCase(tc["a"], tc["col_index"], tc["result"]) for tc in data["mat_col"]]

def load_mat_row():
    load_data()
    return [MatRowTestCase(tc["a"], tc["row_index"], tc["result"]) for tc in data["mat_row"]]

def load_mat_add():
    load_data()
    return [MatAddTestCase(tc["a"], tc["b"], tc["result"]) for tc in data["mat_add"]]

def load_mat_sub():
    load_data()
    return [MatSubTestCase(tc["a"], tc["b"], tc["result"]) for tc in data["mat_sub"]]

def load_mat_mul():
    load_data()
    return [MatMulTestCase(tc["a"], tc["b"], tc["result"]) for tc in data["mat_mul"]]

def load_mat_scl():
    load_data()
    return [MatSclTestCase(tc["a"], tc["s"], tc["result"]) for tc in data["mat_scl"]]

def load_mat_trans():
    load_data()
    return [MatTransTestCase(tc["a"], tc["result"]) for tc in data["mat_tra"]]

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
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
            mat_add(test_case.a, test_case.b)
    else:
        result = mat_add(test_case.a, test_case.b)
        np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_sub())
def test_mat_sub(test_case: MatSubTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
            mat_sub(test_case.a, test_case.b)
    else:
        result = mat_sub(test_case.a, test_case.b)
        np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_mul())
def test_mat_mul(test_case: MatMulTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
            mat_mul(test_case.a, test_case.b)
    
    else:
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

