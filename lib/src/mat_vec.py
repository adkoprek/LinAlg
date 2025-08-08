from src.types import mat, vec


def mat_vec_mul(a: list[list[float]], b: list[float]) -> list[flat]:
    result: vec = []

    for i, b_e in enumerate(b):
        temp = 0
        for row_e in a[i]:
            temp += row_e * b_e
        result.append(temp)

    return result

