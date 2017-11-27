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

def RSPT(eigval_FCI, g):
    L=[]
    L.append(0)
    eigval_FCI.tolist()
    L.append(1.0/(eigval_FCI[0]-eigval_FCI[1]))
    L.append(1.0/(eigval_FCI[0]-eigval_FCI[2]))
    L.append(1.0/(eigval_FCI[0]-eigval_FCI[3]))
    L.append(1.0/(eigval_FCI[0]-eigval_FCI[4]))
    L.append(1.0/(eigval_FCI[0]-eigval_FCI[5]))
    
    
    E0=2.0
    E1=-g
    E2=0.25*g*g*(L[1] + L[2] + L[3] + L[4])
    E3=-0.25*g*g*g*(2*L[1]*L[1] + 2*L[2]*L[2] + 2*L[3]*L[3] + 2*L[4]*L[4] + L[1]*L[2] + L[1]*L[3] + L[2]*L[4] + L[3]*L[4])
    
    return(E0 + E1 + E2 + E3)



xi = 1

np.set_printoptions(linewidth=200)

eigenvalues_matrix_FCI = eigvalsh(HamiltonianFCI(xi, -1.0))
eigenvalues_matrix_CID = eigvalsh(HamiltonianCID(xi, -1.0))

probability_FCI = []
probability_CID = []
ERSPT=[]
eigval_FCI, eigvec_FCI = eigh(HamiltonianFCI(xi, -1.0))
eigval_CID, eigvec_CID = eigh(HamiltonianCID(xi, -1.0))
ERSPT.append(RSPT(eigval_FCI, -1.0))
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
    ERSPT.append(RSPT(eigval_FCI, g))
    probability_FCI.append(np.sum(np.multiply(eigvec_FCI[:1,:1],eigvec_FCI[:1,:1])))
    probability_CID.append(np.sum(np.multiply(eigvec_CID[:1,:1],eigvec_CID[:1,:1])))
    GG.append(g)





#G=np.array([np.arange(-1.0, 1.1, 0.1)]).transpose()
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(G, eigenvalues_matrix_CID[:,:1],  color="green",  label=r'$k=1$')
#ax1.plot(G, eigenvalues_matrix_CID[:,1:2], color="blue",   label=r'$k=2$')
#ax1.plot(G, eigenvalues_matrix_CID[:,2:3], color="black",  label=r'$k=3$')
#ax1.plot(G, eigenvalues_matrix_CID[:,3:4], color="red",  label=r'$k=4$')
#ax1.plot(G, eigenvalues_matrix_CID[:,4:5], color="black", linestyle="--", label=r'$k=5$')
##ax1.plot(G, eigenvalues_matrix_CID[:,5:6], color="magenta",label=r'$k=6$')
#plt.grid()
#
#plt.legend(loc="upper left", fontsize=12)
#
#ax1.tick_params(axis='both', which='major', labelsize=14)
#plt.xlabel(r'$g$', fontsize=20)
#plt.ylabel(r'$E_k$', fontsize=20)
#
#plt.draw()
##plt.show()
#plt.savefig("task2c1.pdf")
#
#""" ------------------------------------------------------------------------------------ """
#
#
def F(p, epsilon):
    if (p == 1 or p == 2):
        return epsilon[p-1] - 0.5*g
    elif (p == 3 or p == 4):
        return epsilon[p-1]


def CCamplitudes(sigma, epsilon, T_old, g):

    T = [[0 for i in range(2)] for j in range(2)]
    
    #for i in range(2):
    #    for j in range(2):
    #        print(T[i][j])

    for i in range(1,3):
        for a in range(3,5):
            I=i-1
            A=a-3
            denom=2.0*(F(i,epsilon) - F(a,epsilon)) + sigma
            numerator = sigma*T_old[I][A]
            numerator -= 0.5*g
            for b in range(3,5):
                numerator -= 0.5*g*T_old[I][b-3]
            for j in range(1,3):
                numerator -= 0.5*g*T_old[j-1][A]
            numerator -= 0.5*g*T_old[0][0]*T_old[1][1]
            numerator -= 0.5*g*T_old[0][1]*T_old[1][0]
            summ = 0.0
            for b in range(3,5):
                for j in range(1,3):
                    summ += T_old[j-1][b-3]
            summ *= T_old[I][A]
            numerator += 0.5*g*summ
            T[I][A] = numerator/denom
    return T

epsilon=[]

for i in range(1,5):
    epsilon.append(xi*(i-1))

maxiter = 100
#g=0
sigma=-0.5
GGG=[]
ECC=[]
for g in np.arange(-1.0, 1.1, 0.1):
    T0 = [[0 for i in range(2)] for j in range(2)]
    T = [[0 for i in range(2)] for j in range(2)]
    
    for i in range(0,maxiter):
        T = CCamplitudes(sigma, epsilon, T0, g)
        T0 = T
    
    sumofall=0.0
    for i in range(2):
        for a in range(2):
            sumofall += T[i][a]
    ECC.append(2.0 - g -g*0.5*sumofall)
    GGG.append(g)


#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111)
#ax1.plot(GGG, ECC,  color="green",  label=r'$E_{cc}$')
#plt.grid()

#plt.legend(loc="upper left", fontsize=12)

#ax1.tick_params(axis='both', which='major', labelsize=14)
#plt.xlabel(r'$g$', fontsize=20)
#plt.ylabel(r'$E_k$', fontsize=20)

#plt.draw()
#plt.show()
#plt.savefig("task3g.pdf")

X=np.array([np.arange(-1.0, 1.1, 0.1)])

Y=(eigenvalues_matrix_FCI[:,:1].reshape((1,21))-(np.asarray(ECC)).reshape((1,21)))

print(Y)
print(X)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(XX,YY,  linestyle="-.", color="red",  label=r'$FCI$')
plt.grid()
plt.legend(loc="upper left", fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$g$', fontsize=20)
plt.ylabel(r'$E_k$', fontsize=20)
plt.draw()
plt.show()
#plt.savefig("task3g.pdf")






#G=np.array([np.arange(-1.0, 1.1, 0.1)]).transpose()
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#ax2.plot(GG, ERSPT,  color="black", linestyle = "--",  label=r'$RSPT$')
#ax2.plot(GGG, ECC,  color="green", linestyle="none", marker="^",  label=r'$E_{cc}$')
#ax2.plot(G, eigenvalues_matrix_FCI[:,:1],  linestyle="-.", color="red",  label=r'$FCI$')
#plt.grid()
#
#plt.legend(loc="lower left", fontsize=12)
##axes = plt.gca()
##axes.set_ylim([0,1.2])
#ax2.tick_params(axis='both', which='major', labelsize=14)
#plt.xlabel(r'$g$', fontsize=20)
#plt.ylabel(r'$E(g)$', fontsize=20)
#
#plt.draw()
##plt.show()
#plt.savefig("RSsk2c2.pdf")
#
##print((np.asarray(ECC)).reshape((1,21)))
##print(eigenvalues_matrix_FCI[:,:1].reshape((1,21))-np.asarray(ECC))
#
