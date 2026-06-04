'''
A continuación se muestra un código completo en Python para resolver computacionalmente el Ejemplo 3.7 mediante el Método de los Elementos Finitos (FEM) usando dos elementos cuadráticos de tres nodos, calcular:

*. La matriz global de rigidez.
*. El vector global de cargas centrífugas.
*. Los desplazamientos nodales.
*. Los esfuerzos en los nodos de cada elemento.
*. La solución analítica exacta.
*. Una comparación gráfica FEM vs. solución exacta.
'''
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# DATOS DEL PROBLEMA
# =====================================================

E = 1.0e7           # psi
A = 0.6             # in^2
rho = 0.2836        # lb/in^3
omega = 30.0        # rad/s

g = 32.2 * 12       # in/s^2

L_total = 42.0      # in
Le = 21.0           # in

# =====================================================
# MATRIZ DE RIGIDEZ DEL ELEMENTO CUADRATICO
# =====================================================

ke = (E*A)/(3*Le) * np.array([
    [ 7,  1, -8],
    [ 1,  7, -8],
    [-8, -8, 16]
])

print("\nMatriz de rigidez del elemento:")
print(ke)

# =====================================================
# ENSAMBLAJE GLOBAL
# =====================================================

K = np.zeros((5,5))

# conectividades
conn1 = [0,2,1]     # elemento 1 : [1,3,2]
conn2 = [2,4,3]     # elemento 2 : [3,5,4]

# ensamblaje
for conn in [conn1, conn2]:
    for i in range(3):
        for j in range(3):
            K[conn[i], conn[j]] += ke[i,j]

print("\nMatriz global K:")
print(K)

# =====================================================
# CARGAS CENTRIFUGAS
# =====================================================

r1 = 10.5
r2 = 31.5

f1 = rho*r1*omega**2/g
f2 = rho*r2*omega**2/g

print("\nf1 =", f1)
print("f2 =", f2)

fe1 = A*Le*f1*np.array([1/6, 1/6, 2/3])
fe2 = A*Le*f2*np.array([1/6, 1/6, 2/3])

F = np.zeros(5)

for i in range(3):
    F[conn1[i]] += fe1[i]

for i in range(3):
    F[conn2[i]] += fe2[i]

print("\nVector global F:")
print(F)

# =====================================================
# CONDICION DE FRONTERA
# =====================================================

# Q1 = 0

free = [1,2,3,4]

Kred = K[np.ix_(free,free)]
Fred = F[free]

# =====================================================
# SOLUCION
# =====================================================

Qred = np.linalg.solve(Kred, Fred)

Q = np.zeros(5)
Q[free] = Qred

print("\nDesplazamientos nodales (in)")
for i,q in enumerate(Q):
    print(f"Q{i+1} = {q:.8e}")

# =====================================================
# ESFUERZOS FEM
# =====================================================

def sigma_element(qe, xi):
    """
    esfuerzo axial elemento cuadratico
    """
    B = (2/Le)*np.array([
        -(1-2*xi)/2,
         (1+2*xi)/2,
        -2*xi
    ])

    return E*(B @ qe)

# nodos naturales
xis = [-1,0,1]

# elemento 1
q1 = np.array([Q[0],Q[2],Q[1]])

# elemento 2
q2 = np.array([Q[2],Q[4],Q[3]])

sigma1 = [sigma_element(q1,xi) for xi in xis]
sigma2 = [sigma_element(q2,xi) for xi in xis]

print("\nEsfuerzos FEM (psi)")
print("--------------------------------")
print("Elemento 1")
print("Nodo local 1 =", sigma1[0])
print("Nodo local 2 =", sigma1[1])
print("Nodo local 3 =", sigma1[2])

print("\nElemento 2")
print("Nodo local 1 =", sigma2[0])
print("Nodo local 2 =", sigma2[1])
print("Nodo local 3 =", sigma2[2])

# =====================================================
# SOLUCION EXACTA
# =====================================================

x_exact = np.linspace(0,L_total,400)

sigma_exact = rho*omega**2/(2*g)*(L_total**2 - x_exact**2)

# =====================================================
# PUNTOS FEM PARA GRAFICA
# =====================================================

x_fem = np.array([0,10.5,21,31.5,42])

sigma_fem = np.array([
    sigma1[0],
    sigma1[1],
    sigma1[2],
    sigma2[1],
    sigma2[2]
])

# =====================================================
# RESULTADOS NUMERICOS
# =====================================================

print("\n--------------------------------")
print("RESUMEN")
print("--------------------------------")
print("Desplazamientos [in]")
print(Q)

print("\nEsfuerzos FEM [psi]")
for x,s in zip(x_fem,sigma_fem):
    print(f"x = {x:5.1f} in  -->  sigma = {s:8.2f}")

# =====================================================
# GRAFICA
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    x_exact,
    sigma_exact,
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
plt.title('Distribución de esfuerzo axial')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
