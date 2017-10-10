#!/usr/bin/env python

from UHF_scf import call_psi4,UHF 
from RHF_scf import RHF
import os
import numpy as np
import matplotlib.pyplot as plt


R=[]
E_uHF=[]
E_rHF=[]
E_uHF_psi4=[]
E_rHF_psi4=[]


for i in np.arange(0.7, 7.0, 0.1):
    # for each new value of radius we create new geometry
    r=i
    h2 = """
        0 1
        H
        H 1 %f
        symmetry c1
        units bohr
        """ % (r)
    #call psi4 (rhf and uhf) to get matrix elements and benchmark energy
    H,W,S,nbf,nalpha,nbeta,Enuc_RHF,E_RHF_psi4 = call_psi4(h2, {'reference' : 'rhf'})
    #call RHF SCF function
    E_RHF,nloops=RHF(H,W,S,nbf,nalpha)
    H,W,S,nbf,nalpha,nbeta,Enuc_UHF,E_UHF_psi4 = call_psi4(h2, {'reference' : 'uhf'})
    #call UHF SCF function
    E_UHF,nloops=UHF(H,W,S,nbf,nalpha,nbeta)
    #make lists for matplotlib
    R.append(r)
    E_uHF.append(E_UHF+Enuc_UHF)
    E_rHF.append(E_RHF+Enuc_RHF)
    E_uHF_psi4.append(E_UHF_psi4)
    E_rHF_psi4.append(E_RHF_psi4)

#creates plot
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(R, E_uHF_psi4, color="green",marker="^",label=r'$UHF\ Psi4$')
ax1.plot(R, E_rHF_psi4, color="blue",marker="v",label=r'$RHF\ Psi4$')
ax1.plot(R, E_uHF, color="black",linestyle="-.",label=r'$UHF$')
ax1.plot(R, E_rHF, color="red",linestyle="--",label=r'$RHF$')
plt.grid()

plt.legend(loc="lower right", fontsize=18)

ax1.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel(r'$r,\ r_{Bohr}$', fontsize=20)
plt.ylabel(r'$E,\ Ha$', fontsize=20)

plt.draw()
plt.show()

