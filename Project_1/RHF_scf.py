#!/usr/bin/env python

import psi4
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
    return H,W,S,nbf,nalpha,nbeta

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
    maxHFiterations = 100     # Max number of HF iterations
    hf_counter = 0
    tolerance = 1e-7
    convergence = 1e-6
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
           D = 2.0*np.matmul(U[0:,0:N],np.transpose(U[0:,0:N]))
           
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
    ########### MAIN PART ############
    #r = 1.84 
    
    #h2o ="""
    #    O
    #    H 1 r
    #    H 1 r 2 104
    #    symmetry c1
    #    r = %f
    #    units bohr
    #""" % (r)
    
    r = 5.0
    
    h2 = """
        0 1
        H
        H 1 %f
        symmetry c1
        units bohr
        """ % (r)

    #H,W,S,nbf,nalpha,nbeta = call_psi4(h2o, {'reference' : 'rhf'})
    H,W,S,nbf,nalpha,nbeta = call_psi4(h2, {'reference' : 'rhf'})
    
    E_HF,nloops=RHF(H,W,S,nbf,nalpha)
    print("Convergence on loop # ",nloops)
    print(E_HF)
    
    



    
#E_RHF_psi4, Enuc, H, W, S, norb, nocc_up, nocc_dn = call_psi4(h2, {'reference' : 'rhf'})
#   
#print("RHF energy of H2 molecule at interatomic distance R = %f is E_RHF = %f" % (r, E_RHF_psi4))
#print("The nuclear term is Enuc = ", Enuc)
#print("Basis size = %d, N_up = %d, N_down = %d" % (norb, nocc_dn, nocc_up))
#
#E_UHF_psi4, Enuc, H, W, S, norb, nocc_up, nocc_dn = call_psi4(h2o, {'reference' : 'uhf'})
#   
#print("RHF energy of H2O molecule at interatomic distance R = %f is E_RHF = %f" % (r, E_UHF_psi4))
#print("The nuclear term is Enuc = ", Enuc)
#print("Basis size = %d, N_up = %d, N_down = %d" % (norb, nocc_dn, nocc_up))
#

#print(np.allclose(np.dot(S12,np.dot(S,S12)),np.eye(norb)))