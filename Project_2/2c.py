#!/usr/bin/env python3

import numpy as np
from scipy.linalg import eigvalsh,eig,eigh
import matplotlib.pyplot as plt
from numpy import linalg as LA

# FCI Hamiltonian for the pairing model
def HamiltonianFCI(xi,g):
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

# CI_D Hamiltonian for the pairing model
def HamiltonianCID(xi,g):
    """ Takes as input:
        xi - spacing between levels,
        g  - interraction strenght.
    """
    H = np.asarray(
        [[   2*xi-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,  ],
         [   -0.5*g,    4*xi-g,     -0.5*g,     -0.5*g,        0.,  ], 
         [   -0.5*g,    -0.5*g,     6*xi-g,         0.,    -0.5*g,  ], 
         [   -0.5*g,    -0.5*g,         0.,     6*xi-g,    -0.5*g,  ], 
         [   -0.5*g,        0.,     -0.5*g,     -0.5*g,    8*xi-g,  ]] 
     )
    return H





xi = 1

np.set_printoptions(linewidth=200)

eigenvalues_matrix_FCI = eigvalsh(HamiltonianFCI(xi, -1.0))
eigenvalues_matrix_CID = eigvalsh(HamiltonianCID(xi, -1.0))

probability_FCI = []
probability_CID = []
eigval_FCI, eigvec_FCI = eigh(HamiltonianFCI(xi, -1.0))
eigval_CID, eigvec_CID = eigh(HamiltonianCID(xi, -1.0))
probability_FCI.append(np.sum(np.multiply(eigvec_FCI[:1,:1],eigvec_FCI[:1,:1])))
probability_CID.append(np.sum(np.multiply(eigvec_CID[:1,:1],eigvec_CID[:1,:1])))
GG = [-1.0]

for g in np.arange(-0.9, 1.1, 0.1):
    H_FCI = HamiltonianFCI(xi, g)
    H_CID = HamiltonianCID(xi, g)
    eigenvalues_FCI = eigvalsh(H_FCI)
    eigenvalues_CID = eigvalsh(H_CID)
    eigenvalues_matrix_FCI = np.vstack((eigenvalues_matrix_FCI,eigenvalues_FCI))
    eigenvalues_matrix_CID = np.vstack((eigenvalues_matrix_CID,eigenvalues_CID))
    eigval_FCI, eigvec_FCI = eigh(H_FCI)
    eigval_CID, eigvec_CID = eigh(H_CID)
    probability_FCI.append(np.sum(np.multiply(eigvec_FCI[:1,:1],eigvec_FCI[:1,:1])))
    probability_CID.append(np.sum(np.multiply(eigvec_CID[:1,:1],eigvec_CID[:1,:1])))
    GG.append(g)





G=np.array([np.arange(-1.0, 1.1, 0.1)]).transpose()
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(G, eigenvalues_matrix_CID[:,:1],  color="green",  label=r'$k=1$')
ax1.plot(G, eigenvalues_matrix_CID[:,1:2], color="blue",   label=r'$k=2$')
ax1.plot(G, eigenvalues_matrix_CID[:,2:3], color="black",  label=r'$k=3$')
ax1.plot(G, eigenvalues_matrix_CID[:,3:4], color="red",  label=r'$k=4$')
ax1.plot(G, eigenvalues_matrix_CID[:,4:5], color="black", linestyle="--", label=r'$k=5$')
#ax1.plot(G, eigenvalues_matrix_CID[:,5:6], color="magenta",label=r'$k=6$')
plt.grid()

plt.legend(loc="upper left", fontsize=12)

ax1.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$g$', fontsize=20)
plt.ylabel(r'$E_k$', fontsize=20)

plt.draw()
#plt.show()
plt.savefig("task2c1.pdf")

""" ------------------------------------------------------------------------------------ """


fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(GG, probability_FCI,  color="black",   label=r'$FCI$')
ax2.plot(GG, probability_CID,  color="black", linestyle = "--",  label=r'$CID$')
plt.grid()

plt.legend(loc="lower right", fontsize=12)
#axes = plt.gca()
#axes.set_ylim([0,1.2])
ax1.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$g$', fontsize=20)
plt.ylabel(r'$f(g)$', fontsize=20)

plt.draw()
#plt.show()
plt.savefig("task2c2.pdf")




