from src.inverse import inv
from src.errors import ShapeMismatchedError, SingularError
from src.types import mat, vec
from dataclasses import dataclass, field
from tests.consts import DATA_PATH
import numpy as np
import pytest
import json


@dataclass
class MatInvTestCase:
    a: mat
    result: mat | str

data = None
def load_data():
    global data
    if data is not None:
        return

    with open(f"{DATA_PATH}/inverse.json", 'r') as file:
        data = json.load(file)

def load_inv():
    load_data()
    return [MatInvTestCase(tc["a"], tc["result"]) for tc in data["inv"]]

@pytest.mark.parametrize("test_case", load_inv())
def test_inv(test_case: MatInvTestCase):
    if test_case.result == "SingularError":
        with pytest.raises(SingularError):
            inv(test_case.a)

    elif test_case.result == "ShapeMismatchedError":
        with pytest.raises(ShapeMismatchedError):
            inv(test_case.a)
    
    else:
        result = inv(test_case.a)
        np.testing.assert_allclose(result, test_case.result)

