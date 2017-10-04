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

def UHF(H,W,S,nbf,nalpha,nbeta):
    
    ## SCF loop ##
    
    EnergyDifference = 10.0   # dummy value for loop start
    maxHFiterations = 500     # Max number of HF iterations
    hf_counter = 0
    tolerance = 1e-7
    convergence = 1e-6
    theta = 0.3
    #make initial guess for U matrix(Identity + random Hermitian noise)
    U_up = np.identity(nbf)
    U_down = np.identity(nbf)
    
    #compute initial density matrix with Hermitian noise
    D_sigma_up = np.matmul(U_up[0:,0:nalpha], np.transpose(U_up[0:,0:nalpha]))
    X = np.random.rand(nbf, nbf)
    X = X + np.transpose(X)
    D_sigma_up = D_sigma_up + 0.001*X
    
    D_sigma_down = np.matmul(U_down[0:,0:nbeta],np.transpose(U_down[0:,0:nbeta]))
    X = np.random.rand(nbf, nbf)
    X = X + np.transpose(X)
    D_sigma_down = D_sigma_down + 0.001*X
    
    lambda_up_old = np.zeros(nbf)
    lambda_down_old = np.zeros(nbf)
    
    ## loop(max_iteration or energy_diff < 10^-7) ###
    while ((hf_counter < maxHFiterations) and (convergence > tolerance)):
       
        #compute density matrix D (eq. (28))
        if hf_counter != 0:
           D_sigma_up_new = np.matmul(U_up[0:,0:nalpha],np.transpose(U_up[0:,0:nalpha]))
           D_sigma_down_new = np.matmul(U_down[0:,0:nbeta],np.transpose(U_down[0:,0:nbeta]))
           
           D_sigma_up = theta*D_sigma_up + (1-theta)*D_sigma_up_new 
           D_sigma_down = theta*D_sigma_down + (1-theta)*D_sigma_down_new

        #for Fock operator we compute three components
        # 1. take h/one-body term from psi4
    
        # 2. compute J(D) (eq. (29b))
        JD_up =  np.einsum('pqrs,sr->pq', W, D_sigma_up)
        JD_down =  np.einsum('pqrs,sr->pq', W, D_sigma_down)
        JD = JD_up + JD_down
    
        # 3. compute K (eq. (29c))
        K_up = np.einsum('psrq,sr->pq', W, D_sigma_up)
        K_down = np.einsum('psrq,sr->pq', W, D_sigma_down)
    
        # compute Fock matrix F (eq. (30))
        Fock_matrix_up = H + JD - K_up
        Fock_matrix_down = H + JD - K_down 
    
        #Solve the eigenvalue problem Fock_matrix_up*U_up=S*U_up*E_up
        lambda_up, U_up = geig(Fock_matrix_up,S)
    
        #Solve the eigenvalue problem Fock_matrix_down*U_down=S*U_down*E_down
        lambda_down, U_down = geig(Fock_matrix_down,S)
        
        #eigenvalue difference difference
        convergence_down = np.amax(np.absolute(lambda_down - lambda_down_old))
        convergence_up = np.amax(np.absolute(lambda_up - lambda_up_old))
        convergence = max(convergence_down, convergence_up)
        
        #assign previous step eigenvalues
        lambda_up_old = lambda_up
        lambda_down_old = lambda_down
    
        #print(hf_counter,convergence)
        hf_counter += 1
         
    D=D_sigma_down + D_sigma_up
    OneBody=np.trace(np.matmul(D,H)) 
    Direct=0.5*np.trace(np.matmul(D,JD)) 
    Exchange=0.5*np.trace(np.matmul(D_sigma_up,K_up)) + 0.5*np.trace(np.matmul(D_sigma_down,K_down))  
    return (OneBody + Direct - Exchange), hf_counter

if __name__ == "__main__":
    ########### MAIN PART ############
    #r = 1.84 
    #
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
    
    #H,W,S,nbf,nalpha,nbeta = call_psi4(h2o, {'reference' : 'uhf'})
    H,W,S,nbf,nalpha,nbeta,Enuc,E_psi4 = call_psi4(h2, {'reference' : 'uhf'})
    
    E_HF,nloops=UHF(H,W,S,nbf,nalpha,nbeta)
    
    print("Convergence on loop # ",nloops)
    print("HF energy without nuclear interaction: ",E_HF)
    print("HF energy with nuclear interaction: ",E_HF+Enuc)
    print("HF energy after Psi4 SCF calculation: ",E_psi4)
