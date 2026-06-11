import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Material and geometry
# -------------------------------------------------

t = 10.0          # thickness (mm)

nodes = {
    7: np.array([100.,20.]),
    8: np.array([85.,40.]),
    9: np.array([70.,60.])
}

# -------------------------------------------------
# Equivalent nodal load for one edge
# -------------------------------------------------

def edge_load(x1,y1,x2,y2,p1,p2,t):

    L = np.sqrt((x2-x1)**2+(y2-y1)**2)

    c = (y2-y1)/L
    s = (x1-x2)/L

    Tx1 = -p1*c
    Ty1 = -p1*s

    Tx2 = -p2*c
    Ty2 = -p2*s

    T = t*L/6*np.array([
        2*Tx1+Tx2,
        2*Ty1+Ty2,
        Tx1+2*Tx2,
        Ty1+2*Ty2
    ])

    return L,c,s,T

# -------------------------------------------------
# Edge 7-8
# -------------------------------------------------

L1,c1,s1,T1 = edge_load(
    100,20,
    85,40,
    1,2,
    t
)

print()
print("Edge 7-8")
print(T1)

# -------------------------------------------------
# Edge 8-9
# -------------------------------------------------

L2,c2,s2,T2 = edge_load(
    85,40,
    70,60,
    2,3,
    t
)

print()
print("Edge 8-9")
print(T2)

# -------------------------------------------------
# Global force vector
# -------------------------------------------------

F = np.zeros(6)

# edge 7-8

F[0]+=T1[0]
F[1]+=T1[1]

F[2]+=T1[2]
F[3]+=T1[3]

# edge 8-9

F[2]+=T2[0]
F[3]+=T2[1]

F[4]+=T2[2]
F[5]+=T2[3]

print()
print("Equivalent nodal loads")
print()

print("F13 =",F[0])
print("F14 =",F[1])
print("F15 =",F[2])
print("F16 =",F[3])
print("F17 =",F[4])
print("F18 =",F[5])

# -------------------------------------------------
# Plot geometry
# -------------------------------------------------

fig,ax=plt.subplots(figsize=(6,6))

x=[100,85,70]
y=[20,40,60]

ax.plot(x,y,'k',lw=3)

for i,n in enumerate([7,8,9]):
    ax.plot(x[i],y[i],'ko',ms=8)
    ax.text(x[i]+1,y[i],str(n),fontsize=14)

ax.set_aspect('equal')
ax.grid(True)

ax.set_title("Loaded boundary 7-8-9")

# -------------------------------------------------
# Distributed pressure
# -------------------------------------------------

for i in range(11):

    s=i/10

    xx=100+s*(70-100)
    yy=20+s*(60-20)

    p=1+s*(3-1)

    nx=-0.8
    ny=-0.6

    ax.arrow(
        xx,
        yy,
        5*nx*p,
        5*ny*p,
        head_width=1,
        color='blue'
    )

# -------------------------------------------------
# Equivalent nodal forces
# -------------------------------------------------

scale=0.05

Fx=[F[0],F[2],F[4]]
Fy=[F[1],F[3],F[5]]

for i in range(3):

    ax.arrow(
        x[i],
        y[i],
        scale*Fx[i],
        scale*Fy[i],
        head_width=1.5,
        color='red',
        linewidth=2
    )

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

plt.show()
