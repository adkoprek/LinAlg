import numpy as np
from tabulate import tabulate
from src.lu import solve

A = 100 * np.random.random((8, 8))
b = 100 * np.random.random((8, 1))
print("Det", np.linalg.det(A))

x = solve(A, b)

print("Close? ->", np.allclose(A @ x, b))
