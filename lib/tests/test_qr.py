from src.qr import vec_prj, mat_prj, ortho, ortho_base, qr
from src.errors import SingularError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import * 
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


data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/qr.json", 'r') as file:
        data = json.load(file)

def load_vec_proj():
    cases = []
    for i in range(TEST_CASES):
        v = random_vector()
        u = random_vector(v.shape)

        cases.append(VecProjTestCase(u, v, (np.dot(v, u) / np.dot(u, u)) * u))

    return cases

def load_mat_proj():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()

        # Very rare
        ATA = A.T @ A
        if abs(np.linalg.det(ATA)) < ZERO:
            continue

        cases.append(MatProjTestCase(A, A @ np.linalg.inv(ATA) @ A.T))

    return cases

def load_ortho():
    load_data()
    return [OrthoTestCase(tc["existing"], tc["new"], tc["result"]) for tc in data["ortho"]]

def load_ortho_base():
    load_data()
    return [OrthoBaseTestCase(tc["vectors"], tc["result"]) for tc in data["ortho_base"]]

def load_qr():
    cases = []
    for i in range(TEST_CASES):
        A = random_matrix()
        cases.append(QRTestCase(A))

    return cases

@pytest.mark.parametrize("test_case", load_vec_proj())
def test_vec_prj(test_case: VecProjTestCase):
    result = vec_prj(test_case.a, test_case.b)
    np.testing.assert_allclose(result, test_case.result)

@pytest.mark.parametrize("test_case", load_mat_proj())
def test_mat_prj(test_case: MatProjTestCase):
    result = mat_prj(test_case.a)
    np.testing.assert_allclose(result, test_case.result, atol=UNSTABLE_ZERO)

@pytest.mark.parametrize("test_case", load_ortho())
def test_ortho(test_case: OrthoTestCase):
    orthogonalized, factors = ortho(test_case.existing, test_case.new, True)
    np.testing.assert_allclose(orthogonalized, test_case.result[0])
    np.testing.assert_allclose(factors, test_case.result[1])

@pytest.mark.parametrize("test_case", load_ortho_base())
def test_ortho_base(test_case: OrthoBaseTestCase):
    result = ortho_base(test_case.vectors)
    np.testing.assert_allclose(result, test_case.result, atol=ZERO)

@pytest.mark.parametrize("test_case", load_qr())
def test_qr(test_case: QRTestCase):
    Q, R = qr(test_case.a)

    QM = np.array(Q)
    RM = np.array(R)

    np.testing.assert_allclose(QM @ QM.T, np.eye(len(Q)), atol=UNSTABLE_ZERO)
    np.testing.assert_allclose(QM @ RM, test_case.a, atol=UNSTABLE_ZERO)

    m, n = np.array(test_case.a).shape
    assert QM.shape == (m, m), f"Q is not full: got {QM.shape}, expected ({m},{m})"
    assert RM.shape == (m, n), f"R has wrong shape for full QR: {RM.shape}, expected ({m},{n})"
    assert np.allclose(RM, np.triu(RM)), f"R is not upper triangular {RM}"

