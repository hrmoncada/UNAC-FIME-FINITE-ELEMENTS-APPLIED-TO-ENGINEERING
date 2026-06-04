import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Resultados FEM del libro
# ----------------------------------------

x_fem = np.array([0, 10.5, 21, 31.5, 42])

sigma_fem = np.array([
    583,
    510,
    437,
    218,
    0
])

# ----------------------------------------
# Solución exacta
# ----------------------------------------

rho = 0.2836
omega = 30.0
g = 32.2 * 12.0
L = 42.0

x = np.linspace(0, L, 300)

sigma_exacta = (
    rho*omega**2/(2*g)
)*(L**2 - x**2)

# ----------------------------------------
# Gráfica
# ----------------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    x,
    sigma_exacta,
    '--',
    linewidth=2,
    label='Solución exacta'
)

plt.plot(
    x_fem,
    sigma_fem,
    '-o',
    linewidth=2,
    label='FEM (2 elementos cuadráticos)'
)

plt.xlabel('Posición x (in)')
plt.ylabel('Esfuerzo axial (psi)')
plt.title('Distribución de esfuerzos en barra rotatoria')
plt.grid(True)
plt.legend()

plt.show()
