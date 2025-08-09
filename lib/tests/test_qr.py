from src.qr import vec_prj, mat_prj, ortho, ortho_base, qr
from src.errors import SingularError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json



@dataclass
class VecProjTestCase:
    a: vec
    b: vec
    result: vec

@dataclass
class MatProjTestCase:
    a: mat
    result: mat

@dataclass
class OrthoTestCase:
    existing: list[vec]
    new: vec
    result: list[list[float]]

@dataclass
class OrthoBaseTestCase:
    vectors: list[vec]
    result: list[vec]

@dataclass
class QRTestCase:
    a: mat
    result: list[mat]


data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/qr.json", 'r') as file:
        data = json.load(file)

def load_vec_proj():
    load_data()
    return [VecProjTestCase(tc["a"], tc["b"], tc["result"]) for tc in data["vec_prj"]]

def load_mat_proj():
    load_data()
    return [MatProjTestCase(tc["a"], tc["result"]) for tc in data["mat_prj"]]

def load_ortho():
    load_data()
    return [OrthoTestCase(tc["existing"], tc["new"], tc["result"]) for tc in data["ortho"]]

def load_ortho_base():
    load_data()
    return [OrthoBaseTestCase(tc["vectors"], tc["result"]) for tc in data["ortho_base"]]

def load_qr():
    load_data()
    return [QRTestCase(tc["a"], tc["result"]) for tc in data["qr"]]

@pytest.mark.parametrize("test_case", load_vec_proj())
def test_vec_prj(test_case: VecProjTestCase):
    result = vec_prj(test_case.a, test_case.b)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_proj())
def test_mat_prj(test_case: MatProjTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(SingularError):
            mat_prj(test_case.a)

    else:
        result = mat_prj(test_case.a)
        np.testing.assert_allclose(result, test_case.result, atol=1e-14)

@pytest.mark.parametrize("test_case", load_ortho())
def test_ortho(test_case: OrthoTestCase):
    orthogonalized, factors = ortho(test_case.existing, test_case.new, True)
    np.testing.assert_allclose(orthogonalized, test_case.result[0])
    np.testing.assert_allclose(factors, test_case.result[1])

@pytest.mark.parametrize("test_case", load_ortho_base())
def test_ortho_base(test_case: OrthoBaseTestCase):
    result = ortho_base(test_case.vectors)
    np.testing.assert_allclose(result, test_case.result, atol=1e-14)

@pytest.mark.parametrize("test_case", load_qr())
def test_qr(test_case: QRTestCase):
    Q, R = qr(test_case.a)
    np.testing.assert_allclose(Q, test_case.result[0], atol=1e-2)
    np.testing.assert_allclose(R, test_case.result[1], atol=1e-2)
