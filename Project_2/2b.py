#!/usr/bin/env python3

import numpy as np
from scipy.linalg import eigvalsh,eig,eigh
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Hamiltonian for the pairing model
def Hamiltonian(xi,g):
    """ Takes as input:
        xi - spacing between levels,
        g  - interraction strenght.
    """
    H = np.asarray(
        [[   2*xi-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,         0. ],
         [   -0.5*g,    4*xi-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ], 
         [   -0.5*g,    -0.5*g,     6*xi-g,         0.,    -0.5*g,     -0.5*g ], 
         [   -0.5*g,    -0.5*g,         0.,     6*xi-g,    -0.5*g,     -0.5*g ], 
         [   -0.5*g,        0.,     -0.5*g,     -0.5*g,    8*xi-g,     -0.5*g ], 
         [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,    10*xi-g ]]
     )
    return H


xi = 1

np.set_printoptions(linewidth=200)
eigenvalues_matrix = eigvalsh(Hamiltonian(xi, -1.0))

probability = []
eigval, eigvec = eigh(Hamiltonian(xi, -1.0))
probability.append(np.sum(np.multiply(eigvec[:1,:1],eigvec[:1,:1])))
GG = [-1.0]

for g in np.arange(-0.9, 1.1, 0.1):
    H = Hamiltonian(xi, g)
    eigenvalues = eigvalsh(H)
    eigenvalues_matrix = np.vstack((eigenvalues_matrix,eigenvalues))
    eigval, eigvec = eigh(H)
    probability.append(np.sum(np.multiply(eigvec[:1,:1],eigvec[:1,:1])))
    GG.append(g)


G=np.array([np.arange(-1.0, 1.1, 0.1)]).transpose()
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(G, eigenvalues_matrix[:,:1],  color="green",  label=r'$k=1$')
ax1.plot(G, eigenvalues_matrix[:,1:2], color="blue",   label=r'$k=2$')
ax1.plot(G, eigenvalues_matrix[:,2:3], color="black",  label=r'$k=3$')
ax1.plot(G, eigenvalues_matrix[:,3:4], color="red", linestyle="none", marker="^",  label=r'$k=4$')
ax1.plot(G, eigenvalues_matrix[:,4:5], color="black", linestyle="--", label=r'$k=5$')
ax1.plot(G, eigenvalues_matrix[:,5:6], color="magenta",label=r'$k=6$')
plt.grid()

plt.legend(loc="upper left", fontsize=12)

ax1.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$g$', fontsize=20)
plt.ylabel(r'$E_k$', fontsize=20)

plt.draw()
#plt.show()
plt.savefig("task2b1.pdf")

""" ------------------------------------------------------------------------------------ """

print(probability)
print(GG)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(GG, probability,  color="black", marker = "^",  label=r'$k=1$')
plt.grid()

#plt.legend(loc="upper left", fontsize=12)
#axes = plt.gca()
#axes.set_ylim([0,1.2])
ax1.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$g$', fontsize=20)
plt.ylabel(r'$f(g)$', fontsize=20)

plt.draw()
#plt.show()
plt.savefig("task2b2.pdf")




