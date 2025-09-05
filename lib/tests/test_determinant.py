from src.determinant import det
from src.errors import ShapeMismatchedError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json


@dataclass
class DeterminantTestCase:
    a: mat
    result: float | str 

data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/determinant.json", 'r') as file:
        data = json.load(file)

def load_determinant():
    load_data()
    return [DeterminantTestCase(tc["a"], tc["result"]) for tc in data["determinant"]]

@pytest.mark.parametrize("test_case", load_determinant())
def test_determinant(test_case: DeterminantTestCase):
    if isinstance(test_case.result, str):
        with pytest.raises(ShapeMismatchedError):
            det(test_case.a)
    else:
        result = det(test_case.a)
        assert abs(result - test_case.result) < 1e-9
