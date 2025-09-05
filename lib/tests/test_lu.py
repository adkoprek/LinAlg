from src.lu import lu, solve
from src.errors import SingularError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json


@dataclass
class LUTestCase:
    A: mat
    result: tuple | str

@dataclass
class SolveTestCase:
    A: mat
    b: vec
    result: tuple | str 

data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/lu.json", 'r') as file:
        data = json.load(file)

def load_solve():
    load_data()
    return [SolveTestCase(tc["A"], tc["b"], tc["result"]) for tc in data["solve"]]

def load_lu():
    load_data()
    return [LUTestCase(tc["A"], tc["result"]) for tc in data["lu"]]

@pytest.mark.parametrize("test_case", load_lu())
def test_lu(test_case: LUTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(SingularError):
            lu(test_case.A)
    else:
        L, U, P = lu(test_case.A)
        expected_L, expected_U, expected_P = test_case.result
        np.testing.assert_allclose(L, expected_L)
        np.testing.assert_allclose(U, expected_U)
        np.testing.assert_allclose(P, expected_P)

@pytest.mark.parametrize("test_case", load_solve())
def test_solve(test_case: SolveTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(SingularError):
            solve(test_case.A, test_case.b)
    else:
        result = solve(test_case.A, test_case.b)
        np.testing.assert_allclose(result, test_case.result)
