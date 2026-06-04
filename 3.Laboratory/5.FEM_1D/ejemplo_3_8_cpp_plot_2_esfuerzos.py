import matplotlib.pyplot as plt

x_elem = [100,350]

sigma = [12.6,-240.3]

plt.figure(figsize=(8,4))

plt.bar(
    x_elem,
    sigma,
    width=120
)

plt.xlabel("Centro del elemento (mm)")
plt.ylabel("Esfuerzo (MPa)")
plt.title("Distribución de esfuerzos")

plt.grid(True)

plt.show()
