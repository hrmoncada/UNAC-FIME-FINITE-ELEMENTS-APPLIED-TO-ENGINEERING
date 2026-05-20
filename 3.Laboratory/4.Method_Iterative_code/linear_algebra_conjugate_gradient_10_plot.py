import numpy as np
import matplotlib.pyplot as plt

# Load matrix
A = np.loadtxt("matrix.txt")

# Load error data
data = np.loadtxt("cg_error.txt", skiprows=1)
iters = data[:,0]
errors = data[:,1]

plt.figure(figsize=(10,4))

# Matrix subplot
plt.subplot(1,2,1)
plt.imshow(A)
plt.title("Matrix A")
plt.colorbar()

# Error subplot
plt.subplot(1,2,2)
plt.plot(iters, errors, marker='o')
plt.title("Error vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.grid()

plt.tight_layout()
plt.savefig("result.png")
plt.show()
