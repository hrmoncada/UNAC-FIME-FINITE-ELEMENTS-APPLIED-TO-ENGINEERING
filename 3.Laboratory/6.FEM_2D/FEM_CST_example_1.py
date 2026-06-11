
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Example 5.1 : Shape functions for a CST triangle
# ==========================================================

nodes = np.array([
    [1.5, 2.0],   # node 1
    [7.0, 3.5],   # node 2
    [4.0, 7.0]    # node 3
])

P = np.array([3.85,4.80])

A = np.array([
    [nodes[0,0], nodes[1,0], nodes[2,0]],
    [nodes[0,1], nodes[1,1], nodes[2,1]],
    [1,1,1]
])

b = np.array([P[0],P[1],1])

N = np.linalg.solve(A,b)

print("Shape functions")
print("N1,N2,N3 =",N)
print("Sum =",N.sum())

# Plot
fig,ax=plt.subplots(figsize=(6,5))
tri=np.vstack((nodes,nodes[0]))
ax.plot(tri[:,0],tri[:,1],'-k')
for i,(x,y) in enumerate(nodes):
    ax.plot(x,y,'o')
    ax.text(x+0.1,y+0.1,f'{i+1}')
ax.plot(P[0],P[1],'ro')
ax.text(P[0]+0.1,P[1],'P')
ax.set_aspect('equal')
ax.set_title('Example 5.1')
ax.grid(True)

# ==========================================================
# Example 5.2 : Jacobian
# ==========================================================

x1,y1=nodes[0]
x2,y2=nodes[1]
x3,y3=nodes[2]

J=np.array([
    [x1-x3,y1-y3],
    [x2-x3,y2-y3]
])

print("\nJacobian")
print(J)
print("det(J) =",np.linalg.det(J))
print("Area =",0.5*abs(np.linalg.det(J)))

# ==========================================================
# Example 5.3 : B matrices
# ==========================================================

def B_matrix(coords):
    x1,y1=coords[0]
    x2,y2=coords[1]
    x3,y3=coords[2]

    x21=x2-x1
    x32=x3-x2
    x13=x1-x3

    y12=y1-y2
    y23=y2-y3
    y31=y3-y1

    detJ=(x1-x3)*(y2-y3)-(x2-x3)*(y1-y3)

    B=(1/detJ)*np.array([
        [y23,0,y31,0,y12,0],
        [0,x32,0,x13,0,x21],
        [x32,y23,x13,y31,x21,y12]
    ])
    return B,detJ

# Mesh
global_nodes=np.array([
    [3,0], #1
    [3,2], #2
    [0,2], #3
    [0,0]  #4
])

elem1=global_nodes[[0,3,1]]
elem2=global_nodes[[2,1,3]]

B1,d1=B_matrix(elem1)
B2,d2=B_matrix(elem2)

print("\nElement 1")
print(B1)

print("\nElement 2")
print(B2)

fig2,ax2=plt.subplots(figsize=(6,4))
rect=np.array([[0,0],[3,0],[3,2],[0,2],[0,0]])
ax2.plot(rect[:,0],rect[:,1],'k')
ax2.plot([0,3],[0,2],'k')
ax2.text(2,0.8,'e=1')
ax2.text(0.8,1.3,'e=2')
ax2.set_aspect('equal')
ax2.set_title('Example 5.3')
ax2.grid(True)

plt.show()
