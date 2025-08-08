from src.matrix import mat_ide


def test_mat_ide():
    assert mat_ide(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
