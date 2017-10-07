#!/usr/bin/env python

import psi4
import sys
import numpy as np
import matplotlib.pyplot as plt

def call_psi4(mol_spec, extra_opts = {}):
    mol = psi4.geometry(mol_spec)

    # Set options to Psi4.
    opts = {'basis': 'cc-pVDZ',
                     'reference' : 'uhf',
                     'scf_type' :'direct',
                     'guess' : 'core',
                     'guess_mix' : 'true',
                     'e_convergence': 1e-7}

    opts.update(extra_opts)

    # If you uncomment this line, Psi4 will *not* write to an output file.
    #psi4.core.set_output_file('output.dat', False)
    
    psi4.set_options(opts)

    # Set up Psi4's wavefunction. Psi4 parses the molecule specification,
    # creates the basis based on the option 'basis' and the geometry,
    # and computes all necessary integrals.
    wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('BASIS'))
    
    # Get access to integral objects
    mints = psi4.core.MintsHelper(wfn.basisset())

    # Get the integrals, and store as numpy arrays
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = T + V
    W = np.asarray(mints.ao_eri())
    S = np.asarray(mints.ao_overlap())

    # Get number of basis functions and number of occupied orbitals of each type
    nbf = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta = wfn.nbeta()


    # Perform a SCF calculation based on the options.
    SCF_E_psi4 = psi4.energy('SCF')
    Enuc = mol.nuclear_repulsion_energy()

    # We're done, return.
#    return SCF_E_psi4, Enuc, H,W,S, nbf, nalpha, nbeta
    return H,W,S,nbf,nalpha,nbeta,Enuc,SCF_E_psi4

def geig(A,B):
    """
    Solve the generalized eigenvalue problem A*U = B*U*lambda,
    where B is symmetric, positive definite, and where A is symmetric.
    
    Turns out numpy has removed the generalize eigenvalue call to LAPACK ... 
    Why? Beats me.
    
    Returns lamb, U.
    lamb = vector of eigenvalues in ascending order
    U = matrix of eigenvectors.
    """
    # Compute Cholesky decomp B = L * L^T
    L = np.linalg.cholesky(B)
    
    # Compute new matrix C = L^{-1} A L^{-T}
    Linv = np.linalg.inv(L)
    LinvT = np.transpose(Linv)
    C = np.linalg.multi_dot([Linv, A, LinvT])

    lamb, V = np.linalg.eigh(C)
    U = np.dot(LinvT, V)
    return lamb, U

def RHF(H,W,S,nbf,nalpha):
    ## SCF loop ##
    
    EnergyDifference = 10.0   # dummy value for loop start
    maxHFiterations = 500     # Max number of HF iterations
    hf_counter = 0
    tolerance = 1e-7
    convergence = 1e-6
    theta = 0.3
    #make initial guess for U matrix(Identity + random Hermitian noise)
    U = np.identity(nbf)
    N=nalpha #+nbeta
    #compute initial density matrix with Hermitian noise
    D = 2.0*np.matmul(U[0:,0:N], np.transpose(U[0:,0:N]))
    X = np.random.rand(nbf, nbf)
    X = X + np.transpose(X)
    D = D + 0.001*X
    
    lambda_old = np.zeros(nbf)
    
    ## loop(max_iteration or energy_diff < 10^-7) ###
    while ((hf_counter < maxHFiterations) and (convergence > tolerance)):
       
        #compute density matrix D (eq. (28))
        if hf_counter != 0:
           D_new = 2.0*np.matmul(U[0:,0:N],np.transpose(U[0:,0:N]))
           D = D*theta + (1-theta)*D_new

        #for Fock operator we compute three components
        # 1. take h/one-body term from psi4
    
        # 2. compute J(D) (eq. (29b))
        JD = np.einsum('pqrs,sr->pq', W, D)
    
        # 3. compute K (eq. (29c))
        K = np.einsum('psrq,sr->pq', W, D)
    
        # compute Fock matrix F (eq. (30))
        Fock_matrix = H + JD - 0.5*K
    
        #Solve the eigenvalue problem Fock_matrix*U=S*U*E
        lambda_new, U = geig(Fock_matrix,S)
    
        #eigenvalue difference difference
        convergence = np.amax(np.absolute(lambda_new - lambda_old))
        
        #assign previous step eigenvalues
        lambda_old = lambda_new
    
        #print(hf_counter,convergence)
        hf_counter += 1
         
    OneBody=np.trace(np.matmul(D,H)) 
    Direct=0.5*np.trace(np.matmul(D,JD)) 
    Exchange=0.5*np.trace(np.matmul(D,K))  
    return (OneBody + Direct -0.5*Exchange), hf_counter

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if str(sys.argv[1]) == "H2":
            #########################  H2  ######################### 
            r = 5.0
            h2 = """
                0 1
                H
                H 1 %f
                symmetry c1
                units bohr
                """ % (r)

            H,W,S,nbf,nalpha,nbeta,Enuc,E_psi4 = call_psi4(h2, {'reference' : 'rhf'})

            E_HF,nloops=RHF(H,W,S,nbf,nalpha)

            print("############## H2 ################")
            print("Convergence on loop # ",nloops)
            print("HF energy without nuclear interaction: ",E_HF)
            print("HF energy with nuclear interaction: ",E_HF+Enuc)
            print("HF energy after Psi4 SCF calculation: ",E_psi4)
        elif str(sys.argv[1]) == "H2O":
            #########################  H2O  ######################### 
            r = 1.84
            h2o = """
                 O
                 H 1 r
                 H 1 r 2 104
                 symmetry c1
                 r = %f
                 units bohr
                 """ % (r)
            H,W,S,nbf,nalpha,nbeta,Enuc,E_psi4 = call_psi4(h2o, {'reference' : 'uhf'})
            E_HF,nloops=RHF(H,W,S,nbf,nalpha)

            print("############## H2O ################")
            print("Convergence on loop # ",nloops)
            print("HF energy without nuclear interaction: ",E_HF)
            print("HF energy with nuclear interaction: ",E_HF+Enuc)
            print("HF energy after Psi4 SCF calculation: ",E_psi4)
        else:
            print("Possible options are H2O and H2")
    else:
        print("wrong usage, try python RHF_scf.py H2 or python RHF_scf.py H2O")

