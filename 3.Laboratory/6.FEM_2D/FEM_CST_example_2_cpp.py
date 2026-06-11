import numpy as np
import matplotlib.pyplot as plt

geo=np.loadtxt("geometry.dat")
F=np.loadtxt("forces.dat")

fig,ax=plt.subplots(figsize=(7,6))

# Geometry

ax.plot(geo[:,0],geo[:,1],'k',lw=3)

for i,node in enumerate([7,8,9]):
    ax.plot(geo[i,0],geo[i,1],'ko',ms=8)
    ax.text(
        geo[i,0]+1,
        geo[i,1]+1,
        str(node),
        fontsize=14
    )

# Distributed pressure

for s in np.linspace(0,1,11):

    x=100+s*(70-100)
    y=20+s*(60-20)

    p=1+s*(3-1)

    nx=-0.8
    ny=-0.6

    ax.arrow(
        x,
        y,
        5*p*nx,
        5*p*ny,
        head_width=1,
        color='blue'
    )

# Equivalent nodal forces

scale=0.05

for i in range(3):

    ax.arrow(
        F[i,0],
        F[i,1],
        scale*F[i,2],
        scale*F[i,3],
        head_width=1.5,
        linewidth=2,
        color='red'
    )

ax.grid()

ax.set_aspect('equal')

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")

ax.set_title("Example 5.4")

plt.savefig("Example54.png",dpi=300)

plt.show()
