'''
A continuación se muestra un código completo en Python para resolver el problema mediante el Método de los Elementos Finitos (FEM). El programa:

*. Ensambla automáticamente la matriz global de rigidez.
*. Calcula las cargas térmicas equivalentes.
*. Aplica las condiciones de frontera.
*. Resuelve los desplazamientos nodales.
*. Calcula los esfuerzos en cada elemento.
*. Presenta resultados numéricos.
*. Genera gráficas de desplazamientos y esfuerzos.
'''
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# DATOS DEL PROBLEMA
# =====================================================

# Propiedades del elemento 1 (Aluminio)
E1 = 70e3          # MPa = N/mm^2
A1 = 900           # mm^2
L1 = 200           # mm
alpha1 = 23e-6     # 1/°C

# Propiedades del elemento 2 (Acero)
E2 = 200e3         # MPa = N/mm^2
A2 = 1200          # mm^2
L2 = 300           # mm
alpha2 = 11.7e-6   # 1/°C

# Temperatura
T0 = 20
Tf = 60
dT = Tf - T0

# Carga puntual
P = 300e3  # N

# =====================================================
# MATRICES DE RIGIDEZ ELEMENTALES
# =====================================================

k1 = (E1*A1/L1) * np.array([
    [1, -1],
    [-1, 1]
])

k2 = (E2*A2/L2) * np.array([
    [1, -1],
    [-1, 1]
])

print("\nMatriz k1 (N/mm)")
print(k1)

print("\nMatriz k2 (N/mm)")
print(k2)

# =====================================================
# ENSAMBLAJE DE MATRIZ GLOBAL
# =====================================================

K = np.zeros((3,3))

# Elemento 1: nodos (1-2)
g1 = [0,1]

for i in range(2):
    for j in range(2):
        K[g1[i],g1[j]] += k1[i,j]

# Elemento 2: nodos (2-3)
g2 = [1,2]

for i in range(2):
    for j in range(2):
        K[g2[i],g2[j]] += k2[i,j]

print("\nMatriz global K (N/mm)")
print(K)

# =====================================================
# FUERZAS TÉRMICAS EQUIVALENTES
# =====================================================

theta1 = E1*A1*alpha1*dT*np.array([-1,1])

theta2 = E2*A2*alpha2*dT*np.array([-1,1])

Fth = np.zeros(3)

Fth[g1] += theta1
Fth[g2] += theta2

print("\nFuerzas térmicas equivalentes (N)")
print(Fth)

# =====================================================
# VECTOR DE FUERZAS EXTERNAS
# =====================================================

Fext = np.array([
    0,
    P,
    0
])

F = Fext + Fth

print("\nVector global F (N)")
print(F)

# =====================================================
# CONDICIONES DE FRONTERA
# Q1 = 0
# Q3 = 0
# =====================================================

free = [1]

Kred = K[np.ix_(free,free)]
Fred = F[free]

# =====================================================
# SOLUCIÓN
# =====================================================

Qred = np.linalg.solve(Kred, Fred)

Q = np.zeros(3)
Q[free] = Qred

print("\nDesplazamientos nodales (mm)")
for i,q in enumerate(Q):
    print(f"Q{i+1} = {q:.6f}")

# =====================================================
# ESFUERZOS
# sigma = E/L (u2-u1) - E alpha dT
# =====================================================

sigma1 = E1/L1*(Q[1]-Q[0]) - E1*alpha1*dT

sigma2 = E2/L2*(Q[2]-Q[1]) - E2*alpha2*dT

print("\nEsfuerzos (MPa)")
print(f"sigma_1 = {sigma1:.3f}")
print(f"sigma_2 = {sigma2:.3f}")

# =====================================================
# REACCIONES
# =====================================================

R = K @ Q - F

print("\nReacciones (N)")
for i,r in enumerate(R):
    print(f"R{i+1} = {r:.2f}")

# =====================================================
# GRAFICA DE DESPLAZAMIENTOS
# =====================================================

x_nodes = np.array([0,200,500])

plt.figure(figsize=(7,4))
plt.plot(x_nodes,Q,'o-',linewidth=2)
plt.grid(True)
plt.xlabel('Posición (mm)')
plt.ylabel('Desplazamiento (mm)')
plt.title('Desplazamientos nodales')
plt.tight_layout()
plt.show()

# =====================================================
# GRAFICA DE ESFUERZOS
# =====================================================

x_stress = [100,350]
stress = [sigma1,sigma2]

plt.figure(figsize=(7,4))
plt.bar(['Elemento 1\nAluminio',
         'Elemento 2\nAcero'],
         stress)

plt.ylabel('Esfuerzo (MPa)')
plt.title('Esfuerzos axiales')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# =====================================================
# DIAGRAMA DE ESFUERZO SOBRE LA BARRA
# =====================================================

plt.figure(figsize=(8,4))

plt.plot([0,200],[sigma1,sigma1],
         linewidth=4,
         label='Elemento 1')

plt.plot([200,500],[sigma2,sigma2],
         linewidth=4,
         label='Elemento 2')

plt.grid(True)

plt.xlabel('Posición x (mm)')
plt.ylabel('Esfuerzo (MPa)')
plt.title('Distribución FEM de esfuerzos')
plt.legend()

plt.tight_layout()
plt.show()
