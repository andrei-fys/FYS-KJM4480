#!/usr/bin/env python

from UHF_scf import call_psi4,UHF 
from RHF_scf import RHF
import os
import numpy as np
import matplotlib.pyplot as plt


R=[]
E_uHF=[]
E_rHF=[]


for i in np.arange(0.7, 7.0, 0.1): 
    r=i 
    h2 = """
        0 1
        H
        H 1 %f
        symmetry c1
        units bohr
        """ % (r)

    H,W,S,nbf,nalpha,nbeta = call_psi4(h2, {'reference' : 'rhf'})
    E_RHF,nloops=RHF(H,W,S,nbf,nalpha)
    H,W,S,nbf,nalpha,nbeta = call_psi4(h2, {'reference' : 'uhf'})
    E_UHF,nloops=UHF(H,W,S,nbf,nalpha,nbeta)
    R.append(r)
    E_uHF.append(E_UHF)
    E_rHF.append(E_RHF)


fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(R, E_uHF, color="black",label=r'$UHF$')
ax1.plot(R, E_rHF, color="red",label=r'$RHF$')
plt.grid()

plt.legend(loc="lower right", fontsize=18)
#
plt.xlabel('r', fontsize=20)
plt.ylabel(r'$E$', fontsize=20)
#
plt.draw()
plt.show()

