from src.rref import rref
from src.types import mat
from dataclasses import dataclass
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json

@dataclass
class RREFTestCase:
    a: mat
    result: mat

data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/rref.json", 'r') as file:
        data = json.load(file)

def load_rref():
    load_data()
    return [RREFTestCase(tc["a"], tc["return"]) for tc in data]

@pytest.mark.parametrize("test_case", load_rref())
def test_rref(test_case: RREFTestCase):
    result = rref(test_case.a)
    np.testing.assert_allclose(result, test_case.result)
