import numpy as np
import matplotlib.pyplot as plt

def conjugate_gradient(A, b, tol=1e-8, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()

    errors = []
    rs_old = np.dot(r, r)

    for k in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)

        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = np.dot(r, r)
        error = np.sqrt(rs_new)
        errors.append(error)

        print(f"Iter {k+1}: error = {error:.6e}, x = {x}")

        if error < tol:
            print("Converged!")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, errors


# 🔹 Original system (NOT SPD)
A = np.array([
    [1, -2, 6],
    [2,  2, 3],
    [-1, 0, 3]
], dtype=float)

b = np.array([0, 3, 2], dtype=float)

# 🔹 Convert to SPD system
A_spd = A.T @ A
b_spd = A.T @ b

# Solve
x, errors = conjugate_gradient(A_spd, b_spd)

print("\nFinal Solution:")
for i, val in enumerate(x):
    print(f"x{i} = {val:.6f}")

# 🔹 Plot side-by-side
plt.figure(figsize=(10,4))

# Left: Matrix A heatmap
plt.subplot(1, 2, 1)
plt.imshow(A, aspect='auto')
plt.title("Matrix A")
plt.colorbar()
plt.xticks(range(A.shape[1]))
plt.yticks(range(A.shape[0]))

# Right: Error plot
plt.subplot(1, 2, 2)
plt.plot(errors, marker='o')
plt.title("Error vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.grid()

plt.tight_layout()

# Save PNG
plt.savefig("cg_matrix_error_subplot.png")

plt.show()
