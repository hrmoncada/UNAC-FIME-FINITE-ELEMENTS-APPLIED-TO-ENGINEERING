import numpy as np

# ==========================================
# Gauss-Seidel Method
# ==========================================

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)

    # Initial guess
    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.array(x0, dtype=float)

    print("Initial Guess:")
    print(x)
    print("-" * 50)

    for k in range(max_iter):
        x_old = x.copy()

        # Update each variable
        for i in range(n):

            sigma1 = np.dot(A[i, :i], x[:i])       # updated values
            sigma2 = np.dot(A[i, i+1:], x_old[i+1:])  # old values

            x[i] = (b[i] - sigma1 - sigma2) / A[i, i]

        # Compute error
        error = np.linalg.norm(x - x_old, ord=np.inf)

        print(f"Iteration {k+1}: x = {x}, error = {error:.6e}")

        # Convergence check
        if error < tol:
            print("\nConverged!")
            return x

    print("\nMaximum iterations reached.")
    return x


# ==========================================
# Test System
# ==========================================

A = np.array([
    [10, -1,  2],
    [-1, 11, -1],
    [ 2, -1, 10]
], dtype=float)

b = np.array([6, 25, -11], dtype=float)

# ==========================================
# Solve
# ==========================================

solution = gauss_seidel(A, b)

print("\nFinal Solution:")
print(solution)
