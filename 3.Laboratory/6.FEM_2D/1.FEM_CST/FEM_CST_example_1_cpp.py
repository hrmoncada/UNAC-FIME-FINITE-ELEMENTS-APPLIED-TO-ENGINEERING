import numpy as np
import matplotlib.pyplot as plt

#-------------------------
# Triangle
#-------------------------

p=np.loadtxt("triangle.dat")

fig,ax=plt.subplots(figsize=(6,5))

ax.plot(p[:,0],p[:,1],'k-',lw=2)

nodes=np.array([
[1.5,2],
[7,3.5],
[4,7]
])

for i in range(3):
    ax.plot(nodes[i,0],nodes[i,1],'bo')
    ax.text(nodes[i,0]+0.1,
            nodes[i,1]+0.1,
            str(i+1))

P=np.array([3.85,4.8])

ax.plot(P[0],P[1],'ro')
ax.text(P[0]+0.1,P[1],"P")

ax.set_aspect('equal')
ax.grid()

plt.title("Example 5.1")

plt.savefig("triangle.png",dpi=300)

#-------------------------
# Mesh
#-------------------------

mesh=np.loadtxt("mesh.dat")

fig,ax=plt.subplots(figsize=(6,4))

ax.plot(mesh[:,0],mesh[:,1],'k')

ax.plot([0,3],[0,2],'k')

ax.text(2,0.8,"e=1",fontsize=14)
ax.text(0.8,1.3,"e=2",fontsize=14)

ax.set_aspect('equal')
ax.grid()

plt.title("Example 5.3")

plt.savefig("mesh.png",dpi=300)

plt.show()
