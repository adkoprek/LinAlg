from src.mat_vec import mat_vec_mul
from src.errors import ShapeMismatchedError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json

@dataclass
class MatVecMulTestCase:
    a: mat
    v: vec
    result: vec | str

data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/mat_vec.json", 'r') as file:
        data = json.load(file)

def load_mat_vec_mul():
    load_data()
    return [MatVecMulTestCase(tc["a"], tc["v"], tc["result"]) for tc in data["mat_vec_mul"]]

@pytest.mark.parametrize("test_case", load_mat_vec_mul())
def test_mat_vec_mul(test_case: MatVecMulTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
            mat_vec_mul(test_case.a, test_case.v)
    else:
        result = mat_vec_mul(test_case.a, test_case.v)
        np.testing.assert_allclose(result, test_case.result)
