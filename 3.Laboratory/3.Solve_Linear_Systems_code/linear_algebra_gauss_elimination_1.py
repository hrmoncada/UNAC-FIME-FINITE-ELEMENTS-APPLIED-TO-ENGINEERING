import numpy as np

def print_matrix(A, b, title=""):
    print("\n" + title)
    for i in range(len(A)):
        row = ["{:.2f}".format(x) for x in A[i]]
        print(f"{row} | {b[i]:.2f}")
    print()

def gaussian_elimination(A, b):
    n = len(A)

    A = A.astype(float)
    b = b.astype(float)

    print_matrix(A, b, "Initial System:")

    # Forward elimination
    for i in range(n):
        print(f"--- Eliminating column {i} ---")

        # Pivoting (optional but safer)
        max_row = np.argmax(abs(A[i:, i])) + i
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            print_matrix(A, b, f"Swap row {i} with row {max_row}")

        # Make pivot = 1
        pivot = A[i][i]
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot
        print_matrix(A, b, f"Normalize row {i}")

        # Eliminate below
        for j in range(i + 1, n):
            factor = A[j][i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
            print_matrix(A, b, f"R{j} = R{j} - ({factor:.2f})R{i}")

    # Back substitution
    x = np.zeros(n)

    print("--- Back Substitution ---")
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i+1:], x[i+1:])
        print(f"x{i} = {x[i]:.4f}")

    return x

# 🔹 Test system (3 equations)
A = np.array([
    [1, -2, 6],
    [2,  2, 3],
    [-1, 0, 3]
], dtype=float)

b = np.array([0, 3, 2], dtype=float)
'''
# 🔹 Test system (4 equations)
A = np.array([
    [2, -1, 1, 2],
    [1,  1, -1, 1],
    [3, -1, 2, 3],
    [1,  2, 3, -1]
], dtype=float)

b = np.array([8, 2, 13, 4], dtype=float)
'''
solution = gaussian_elimination(A, b)

print("\nFinal Solution:")
for i, val in enumerate(solution):
    print(f"x{i} = {val:.4f}")
