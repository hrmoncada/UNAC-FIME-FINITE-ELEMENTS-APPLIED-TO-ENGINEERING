import numpy as np

def print_matrix(A, b, title=""):
    print("\n" + title)
    for i in range(len(A)):
        row = ["{:.3f}".format(x) for x in A[i]]
        print(f"{row} | {b[i]:.3f}")
    print()

def gauss_jordan(A, b):
    n = len(A)

    A = A.astype(float)
    b = b.astype(float)

    print_matrix(A, b, "Initial System:")

    for i in range(n):
        print(f"--- Working on column {i} ---")

        # Pivoting
        max_row = np.argmax(abs(A[i:, i])) + i
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
            print_matrix(A, b, f"Swap R{i} <-> R{max_row}")

        # Normalize pivot row
        pivot = A[i][i]
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot
        print_matrix(A, b, f"Normalize R{i}")

        # Eliminate ALL other rows (above and below)
        for j in range(n):
            if j != i:
                factor = A[j][i]
                A[j] = A[j] - factor * A[i]
                b[j] = b[j] - factor * b[i]
                print_matrix(A, b, f"R{j} = R{j} - ({factor:.3f})R{i}")

    print("--- Final Reduced System (RREF) ---")
    print_matrix(A, b)

    return b  # solution vector

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

solution = gauss_jordan(A, b)

print("Final Solution:")
for i, val in enumerate(solution):
    print(f"x{i} = {val:.4f}")
