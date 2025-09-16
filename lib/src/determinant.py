from src.errors import ShapeMismatchedError
from src.matrix import mat_siz, mat_row
from src.lu import lu
from src.types import mat

def det(a: mat) -> float:
    rows, cols = mat_siz(a)
    if rows != cols:
        raise ShapeMismatchedError(f"The number of cols ({cols}) and the number of rows ({rows})")

    _, U, P = lu(a)

    det = 1

    for i in range(rows):
        det *= U[i][i]

    p = []
    for i in range(rows):
         p.append(mat_row(P, i).index(1))

    visited = [False]*rows
    cycles = 0
    for i in range(rows):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = True
                j = p[j]

    sign = 1 if ((rows - cycles) % 2 == 0) else -1
    det *= sign

    return det
