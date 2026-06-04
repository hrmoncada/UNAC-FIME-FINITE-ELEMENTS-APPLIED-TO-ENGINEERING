import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------
# Leer resultados
#----------------------------------

data = pd.read_csv("resultados.csv")

x = data["X"]
u = data["Q"]

#----------------------------------
# Desplazamientos
#----------------------------------

plt.figure(figsize=(8,4))

plt.plot(
    x,
    u,
    '-o',
    linewidth=2
)

plt.xlabel("Posición x (mm)")
plt.ylabel("Desplazamiento (mm)")
plt.title("Desplazamiento axial FEM")
plt.grid(True)

plt.show()
