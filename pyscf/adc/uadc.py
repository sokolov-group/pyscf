# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Unrestricted algebraic diagrammatic construction
'''
import time
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf.adc.multiroot_davidson1 import eighg
from pyscf.adc.linalg_helper_beta import davidson1, pick_real_eigs, davidson_nosym1
from scipy.sparse.linalg import eigsh, LinearOperator
from pyscf.adc.lanczos_full_orth import compute_lanczos
from pyscf.adc.blanczos import compute_blanczos, mom_blanczos
from pyscf.adc.remove_mo_r2_pyscf import complex_shift
import sys
import copy 
np.set_printoptions(threshold=np. inf)

def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

    nroots_kernel = nroots
    print('# of cvs orbs: ', )
    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
       raise NotImplementedError(adc.method)

    if adc.cvs_type == 'cce' and adc.nfc_orb > 0:
        raise NotImplementedError('Frozen core is not implemented for cce')

    if adc.cvs_type == 'cve' and (adc.ncore_cvs != adc.nfc_orb) and adc.nfc_orb > 0:
        raise ValueError('Number of Frozen core orbitals and CVS orbitals need to be equal')

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = adc.transform_integrals()

    imds = adc.get_imds(eris)
    
    if adc.ncore_cvs > 0:
        cvs = True
    else:
        cvs = False
    
    
    results_out = {}
     
    if nroots == 'full':
        matvec, diag = adc.gen_matvec(imds, eris, cvs=cvs)
        identity = np.identity(diag.size)
        M = np.zeros((diag.size,diag.size),dtype=complex)
        print("building and diagonalizing dim(M) = ", diag.size)
        for i in range(diag.size):
            M[:,i] = matvec(identity[:,i])

        print("Symmetric check", np.linalg.norm(M - M.T))
        print("Number of non-zeros in M-M.T", np.count_nonzero(M - M.T))
        print("Sum of non-zeros in M-M.T", np.sum(M - M.T))
        print("Indices of non-zeros in M-M.T: ")
        #x_idx, y_idx = np.nonzero(M - M.T)
        #print(np.column_stack((x_idx.astype(float), y_idx.astype(float)))) 
        print("Hermiticity check", np.linalg.norm(M - M.conj().T))

        E, U = np.linalg.eig(M)
        T = adc.get_trans_moments()
        T_a = T[0]
        T_b = T[1]
        X_a = np.dot(T_a, U).reshape(-1, diag.size)
        X_b = np.dot(T_b, U).reshape(-1, diag.size)
        P = lib.einsum("pi,pi->i", X_a, X_a.conj())
        P += lib.einsum("pi,pi->i", X_b, X_b.conj())
        print("Dimmension of M: ", E.size) 
        #E_unique, E_idx = np.unique(np.around(E*27.2114, decimals=8), return_index=True)
        #P_unique, P_idx = np.unique(np.around(P, decimals=8), return_index=True)
        print("----------- Start of Eigenvales of M -------------------") 
        print(np.column_stack(np.unique(np.around(E*27.2114, decimals=8), return_counts=True))) 
        print("----------- End of Eigenvales of M ---------------------")
 
        #T = adc.get_trans_moments()
        #spec_factors = get_spec_factors(adc, T, U, diag.size)
        print("----------- Start of Spec factors -------------------") 
        #P_unique = np.unique(np.around(P, decimals=8))
        print(np.column_stack(np.unique(np.around(P, decimals=8), return_counts=True))) 
        print("----------- End of Spec factors ---------------------")
        exit()

    if nroots == 'imds':
        nocc_a = adc.nocc_a
        nocc_b = adc.nocc_b
        nvir_a = adc.nvir_a
        nvir_b = adc.nvir_b
        print("nocc_a",   nocc_a)  
        print("nocc_b",   nocc_b)
        print("nvir_a",   nvir_a)
        print("nvir_b",   nvir_b)
        n_singles_a = nocc_a
        n_singles_b = nocc_b
        s_a = 0
        f_a = n_singles_a
        s_b = f_a
        f_b = s_b + n_singles_b

        M_ij_a, M_ij_b = imds
        E_a, U_a = np.linalg.eig(M_ij_a)
        E_b, U_b = np.linalg.eig(M_ij_b)
        T = adc.get_trans_moments()
        T_a = T[0]
        T_b = T[1]
        T_a = np.array(T_a)
        T_b = np.array(T_b)
        T_a = T_a[:,s_a:f_a].copy() 
        T_b = T_b[:,s_b:f_b].copy() 
        X_a = np.dot(T_a, U_a).reshape(-1,diag.size)
        X_b = np.dot(T_b, U_b).reshape(-1,diag.size)
        P_a = lib.einsum("pi,pi->i", X_a, X_a.conj())
        P_b = lib.einsum("pi,pi->i", X_b, X_b.conj())
        idx_a = np.argsort(P_a) 
        idx_b = np.argsort(P_b) 
        P_a = P_a[idx_a]
        P_b = P_b[idx_b]

        print("----------------------------------")
        print("imds_alpha")
        print("Symmetric check", np.linalg.norm(M_ij_a - M_ij_a.T))
        print("Number of non-zeros in M_ij_a-M_ij_a.T", np.count_nonzero(M_ij_a - M_ij_a.T))
        print("Sum of non-zeros in M_ij_a-M_ij_a.T", np.sum(M_ij_a - M_ij_a.T))
        print("Indices of non-zeros in M_ij_a-M_ij_a.T: ")
        x_idx, y_idx = np.nonzero(M_ij_a - M_ij_a.T)
        print(np.column_stack((x_idx.astype(float), y_idx.astype(float)))) 
        print("Hermiticity check", np.linalg.norm(M_ij_a - M_ij_a.conj().T))
        print("----------------------------------")
        print("imds_beta")
        print("Symmetric check", np.linalg.norm(M_ij_b - M_ij_b.T))
        print("Number of non-zeros in M_ij_b-M_ij_b.T", np.count_nonzero(M_ij_b - M_ij_b.T))
        print("Sum of non-zeros in M_ij_b-M_ij_b.T", np.sum(M_ij_b - M_ij_b.T))
        print("Indices of non-zeros in M_ij_b-M_ij_b.T: ")
        x_idx, y_idx = np.nonzero(M_ij_b - M_ij_b.T)
        print(np.column_stack((x_idx.astype(float), y_idx.astype(float)))) 
        print("Hermiticity check", np.linalg.norm(M_ij_b - M_ij_b.conj().T))
        print("----------------------------------")

        print("Dimmension of imds_alpha: ", E_a.size) 
        print("----------- Start of alpha Eigenvales of imds -------------------") 
        print(np.column_stack(np.unique(np.around(E_a, decimals=8), return_counts=True))) 
        print("----------- End of alpha Eigenvales of imds ---------------------") 
        print("Dimmension of imds_beta: ", E_b.size) 
        print("----------- Start of beta Eigenvales of imds -------------------") 
        print(np.column_stack(np.unique(np.around(E_b, decimals=8), return_counts=True))) 
        print("----------- End of beta Eigenvales of imds ---------------------") 

        print("----------- Start of Spec alpha factors -------------------") 
        print(np.column_stack(np.unique(np.around(P_a, decimals=8), return_counts=True))) 
        print("----------- End of Spec alpha factors ---------------------") 
        print("----------- Start of Spec beta factors -------------------") 
        print(np.column_stack(np.unique(np.around(P_b, decimals=8), return_counts=True))) 
        print("----------- End of Spec beta factors ---------------------") 
    #exit()
    
    matvec, diag = adc.gen_matvec(imds, eris, cvs=cvs)

    def sort_complex_eigenvalues(w, v, nroots, envs):     
        idx = np.argsort(w)[:nroots]
        w = w[idx]
        v = v[:, idx]
        return w, v, idx

    #nroots = 4
    #nroots = adc.cvs_npick[-1]+1 # commented out for tdm/rdm implementation
      
    guess = adc.get_init_guess(nroots, diag, ascending = True)
    #conv, E, U = lib.linalg_helper.davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=1e-16, max_cycle=adc.max_cycle, max_space=adc.max_space, pick=sort_complex_eigenvalues)
    conv, E_cvs, U = lib.linalg_helper.davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space)
    #conv, E, U = lib.linalg_helper.davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=1e-12, max_cycle=adc.max_cycle, max_space=adc.max_space, pick=pick_real_eigs)
    guess = U
    #imds = adc.get_imds(eris, fc_bool=False)

    matvec, diag = adc.gen_matvec(imds, eris, cvs=False, fc_bool=False)

    def cvs_pick(cvs_npick,U):          
        nroots = len(cvs_npick)
        dim_guess = np.array(U).shape[1]
        guess = np.zeros((nroots, dim_guess))#, dtype=complex)
        for idx_guess, npick in enumerate(cvs_npick):
            U = np.array(U)
            guess[idx_guess,:] = U[npick,:]
        return guess, nroots
    
    cvs_npick = adc.cvs_npick
    if cvs_npick:
        guess,nroots = cvs_pick(cvs_npick,U)
        E_cvs = E_cvs[cvs_npick]


    #Eh2eV = 27.211396641308
    Eh2eV = 27.2114
     
    print("CVS type: {}".format(adc.cvs_type))
    if adc.nfc_orb > 0:
        print("Number of FC orbitals: {}".format(adc.nfc_orb))
    T = adc.get_trans_moments()
    U = np.array(U)
    P_cvs, X = adc.get_spec_factors(T, U, nroots)
    print("Energies (eV)   |  Spec factors")
    print("---------------------------------------------------")
    for i in range(E_cvs.size):
        logger.info(adc, ' %12.8f  %14.8f', E_cvs[i]*Eh2eV, P_cvs[i])
    print("---------------------------------------------------")
    #analyze_eigenvector_ip(adc, U, analyze_v_vve=True)  # commented out for rdm/tdm implementation
    U = np.array(U)
    total_tpdm_idx = []
    for i in range(nroots):
        for j in range(nroots):
            print('i = ', i, ' j = ', j) 
            tpdm_a, tpdm_b = compute_rdm_tdm(adc, U[i,:], U[j,:])
            total_tpdm_idx.append((tpdm_a,tpdm_b,i,j))

    
    nrooots = nroots_kernel
           
    #E, U, conv = compute_lanczos(matvec, nroots, guess)
    max_mom_iter = 100
    de_tol = 1e-6
    res_tol = 1e-3
    #ovlp_tol = 0.01
    ovlp_tol = adc.es_mom_thresh 
    max_v = 2
    #if adc.lanczos_space > 0:
    #    E, U, conv, nroots = mom_blanczos(matvec, guess, adc.lanczos_space, max_mom_iter, E, de_tol, res_tol,ovlp_tol=ovlp_tol,max_v=max_v,fixed_proj=True)
    def guess_temp_func(numroots, ad, ascending=True):
        return adc.get_init_guess(numroots,ad,ascending) 

    def matvec_idn1(idn1_vec):
        return idn1_vec
    max_nmom = adc.max_nmom 
    def eig_close_to_init_guess(w, v, nroots, envs):      
           x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])      
           #s = np.dot(np.asarray(U).conj(), np.asarray(x0).T)
           s = np.dot(np.asarray(guess), np.asarray(x0).conj().T)
           snorm = np.einsum('pi,pi->i', s.conj(), s)
           #idx = np.argsort(-snorm)[:nroots] 
           idx = np.argsort(-snorm)
           snorm = snorm[idx]
           #print("snorm: ", snorm[:30])
           idx = idx[snorm > ovlp_tol]
           idx = idx[:max_nmom]  
           nroots = idx.size
           #w, v, idx = lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_eigenvectors = True)
           w = w[idx]
           v = v[:, idx]
           return w, v, idx, nroots
       
    #conv, E, U , nroots= davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space, pick =eig_close_to_init_guess)
    

    big_thresh = 1e15
    nocc_a = adc.nocc_a 
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_a = np.unique(np.round(e_vir_a, 7))
    nvir = e_vir_a.size
    vve_mo_count = e_vir_a.size
    cvs_edge = adc.cvs_edge
    #if cvs_edge == '1s' or cvs_edge == '2p' or cvs_edge == '3p':
    #    e_vir_temp = np.zeros(nvir+1)
    #    e_vir_temp[1:] = e_vir_a[::-1]
    #    e_vir_temp[0] = big_thresh
    #    e_vir_a = e_vir_temp
   
    results_out['E_cvs'] =  E_cvs[0]*Eh2eV
    results_out['P_cvs'] =  P_cvs[0]
    results_out['E_v'] =    ''
    results_out['P_v'] =    ''
    results_out['norm_v'] = ''
    #results_out['nvir'] =   ''
    iter_i = 0
    #E_mom = np.zeros(2) 
    #iter_idx = []
    oneshot_mom = adc.oneshot_mom
    oneshot_cvs = adc.oneshot_cvs
    mo_counter = None
    e_vir_min = None
    #end_bool = False
    oneshot_no_vve = adc.oneshot_no_vve
    if adc.oneshot_mom is True and ovlp_tol > 0 and oneshot_cvs is False and oneshot_no_vve is False:
        conv, E, U , nroots= davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space, pick =eig_close_to_init_guess, tol_residual = 1e-3)
        T = adc.get_trans_moments()
        U = np.array(U)
        P, X = adc.get_spec_factors(T, U, nroots)
        #idx_sort = np.argsort(E)
        #E = E[idx_sort]
        #U = U[idx_sort,:]
        s = np.dot(U, np.asarray(guess).conj().T)
        snorm = np.einsum('pi,pi->p', s.conj(), s)
        print("MOM type: {}".format('one-shot'))
        print("Core excitations: {}".format(adc.cvs_type))
        if adc.nfc_orb > 0:
            print("Number of FC orbitals: {}".format(adc.nfc_orb))
        print("Energies (eV)   |  Spec factors   |  snorm")
        print("---------------------------------------------------")
        for i in range(E.size):
            logger.info(adc, ' %12.8f  %14.8f   %14.8f', E[i]*Eh2eV, P[i], snorm[i])
        print("---------------------------------------------------")
        norm_v, norm_evv, max_u2_e = analyze_eigenvector_ip(adc, U, analyze_v_vve=True)
        #analyze_eigenvector_ip(adc, U, analyze_v_vve=True)
        for key, value in max_u2_e.items():
            print(f'root {key}')   
            for key2, value2 in value.items():
                print(f'{key2} = {value2}')
    elif adc.oneshot_mom is False and ovlp_tol > 0 and oneshot_cvs is False and oneshot_no_vve is True:

        nocc_a = adc.nocc_a 
        energy_thresh = 0#e_vir_a[0]-1e-7
        print("Running no vve calc")
        
        matvec, diag = adc.gen_matvec(imds, eris, cvs=False, fc_bool=False, energy_thresh=0)

        conv, E, U , nroots= davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space, pick =eig_close_to_init_guess, tol_residual = 1e-3)
        T = adc.get_trans_moments()
        U = np.array(U)
        P, X = adc.get_spec_factors(T, U, nroots)
        s = np.dot(U, np.asarray(guess).conj().T)
        snorm = np.einsum('pi,pi->p', s.conj(), s)

        results_out['<CVS|MOM_v>'] = snorm[0]

        print("MOM type: {}".format('No vve included'))
        print("Core excitations: {}".format(adc.cvs_type))
        if adc.nfc_orb > 0:
            print("Number of FC orbitals: {}".format(adc.nfc_orb))
        print("Energies (eV)   |  Spec factors   |  snorm")
        print("---------------------------------------------------")
        for i in range(E.size):
            logger.info(adc, ' %12.8f  %14.8f   %14.8f', E[i]*Eh2eV, P[i], snorm[i])
        print("---------------------------------------------------")
        norm_v, norm_evv, max_u2_e = analyze_eigenvector_ip(adc, U, analyze_v_vve=True)
        #results_out['E_cvs'] = E_cvs[0]*Eh2eV
        #results_out['P_cvs'] = P_cvs[0]
        results_out['E_v'] = E[0]*Eh2eV
        results_out['P_v'] = P[0]
        results_out['norm_v'] = norm_v
        results_out['E_v'] =    E[0]*Eh2eV
        results_out['P_v'] =    P[0]
        results_out['norm_v'] = norm_v
        results_out['Davidson_conv'] = conv[0]
        

    #else: 
        #while iter_i < e_vir_a.size and ovlp_tol > 0 and oneshot_cvs is False and oneshot_no_vve is False, switch_off_while_loop:
        #    energy_thresh_i = e_vir_a[iter_i] - 1e-7
        #    matvec, diag = adc.gen_matvec(imds, eris, cvs=False, fc_bool=False, energy_thresh=energy_thresh_i)
        #    conv, E, U , nroots= davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space, pick =eig_close_to_init_guess, tol_residual = 1e-3)
        #    #conv, E, U = lib.linalg_helper.davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=1e-12, max_cycle=adc.max_cycle, max_space=adc.max_space, pick=eig_close_to_init_guess)
        #    
        #    U = np.array(U)
        #    #idx_sort = np.argsort(E)
        #    #E = E[idx_sort]
        #    #U = U[idx_sort,:]
        #    T = adc.get_trans_moments()
        #    P, X = adc.get_spec_factors(T, U, nroots)
        #    s = np.dot(U, np.asarray(guess).conj().T)
        #    snorm = np.einsum('pi,pi->p', s.conj(), s)

        #    if e_vir_a[0] == big_thresh:
        #        if iter_i == 0:
        #           vve_str = "All vve are incluced"
        #        if iter_i > 0:
        #           vve_str = "Some vve are included"
        #        if iter_i == e_vir_a.size - 1:
        #           vve_str = "No vve are included"
        #    if e_vir_a[0] != big_thresh:
        #        if iter_i == 0:
        #           vve_str = "No vve are incluced"
        #        if iter_i > 0:
        #           vve_str = "Some vve are included"
        #        if iter_i == e_vir_a.size - 1:
        #           vve_str = "All vve are included"

        #    print('cvs_npick: ', cvs_npick)
        #    print("MOM type: {}".format(vve_str))
        #    print("Core excitations: {}".format(adc.cvs_type))
        #    print("Orbital energy shift threshold = {}".format(energy_thresh_i))
        #    if adc.nfc_orb > 0:
        #        print("Number of FC orbitals: {}".format(adc.nfc_orb))
        #    print("Energies (eV)   |  Spec factors   |  snorm")
        #    print("---------------------------------------------------")
        #    for i in range(E.size):
        #        logger.info(adc, ' %12.8f  %14.8f   %14.8f', E[i]*Eh2eV, P[i], snorm[i])
        #    print("---------------------------------------------------")
        #    #print(np.column_stack((E*27.2114, spec_factors)))
        #    #analyze_eigenvector_ip(adc, U, analyze_v_vve=True)
        #    norm_v, norm_evv, max_u2_e = analyze_eigenvector_ip(adc, U, analyze_v_vve=True)
        #    for key, value in max_u2_e.items():
        #        for key2, value2 in value.items():
        #            print(f'{key2} = {value2}')
        #    
        #    if iter_i == 0:
        #        E_last = E
        #        P_last = P
        #        S_last = snorm
        #        max_u2_e_last = max_u2_e
 
        #    if e_vir_a[0] != big_thresh and iter_i == 0:

        #       results_out['pole_E_mom_v'] = E[0]*Eh2eV 
        #       results_out['pole_P_mom_v'] = P[0]
        #       results_out['pole_S_mom_v'] = snorm[0]
        #       for root_key, root_value in max_u2_e['0'].items():
        #           results_out[f'pole_v_{root_key}'] = root_value

        #    if e_vir_a[0] == big_thresh and iter_i == 0:

        #       results_out['twopole1_E_mom_vve'] = E[0]*Eh2eV
        #       results_out['twopole1_P_mom_vve'] = P[0]
        #       results_out['twopole1_S_mom_vve'] = snorm[0]
        #       results_out['twopole2_E_mom_vve'] = "" 
        #       results_out['twopole2_P_mom_vve'] = "" 
        #       results_out['twopole2_S_mom_vve'] = ""  

        #       for root_key, root_value in max_u2_e['0'].items():
        #           results_out[f'twopole1_{root_key}'] = root_value
        #           results_out[f'twopole2_{root_key}'] = ""

        #       if E.size > 2: 
        #           results_out['twopole2_E_mom_vve'] = E[2]*Eh2eV  
        #           results_out['twopole2_P_mom_vve'] = P[2]
        #           results_out['twopole2_S_mom_vve'] = snorm[2]
 
        #           for root_key, root_value in max_u2_e['2'].items():
        #               results_out[f'twopole2_{root_key}'] = root_value
        #    
        #    first_pass = False
        #    if e_vir_a[0] != big_thresh and E.size > 2:
        #       print("CVS pole structure is destroyed\n========================\nPrinting summary of results ...\n=================================")
        #       print("Ecvs: ", E_cvs*Eh2eV)
        #       print("Emom: ", E_last*Eh2eV)
        #       print("Total number of non-degenerate virtual MOs: ", e_vir_a.size)
        #       print("Total number of non-degenerate virtual MOs used in vve: ", iter_i)
        #       print("Highest vve energy threshold used (Eh): ", e_vir_a[iter_i-1])
        #       print("Highest vve energy threshold used (eV): ", e_vir_a[iter_i-1]*Eh2eV) 
        #        
        #       results_out['onepole_E_vve_thresh'] = e_vir_a[iter_i-1]*Eh2eV
        #       results_out['onepole_nvir_vve'] = iter_i
        #       iter_i = e_vir_a.size

        #       results_out['onepole_E_mom_vve'] = E_last[0]*Eh2eV
        #       results_out['onepole_P_mom_vve'] = P_last[0]
        #       results_out['onepole_S_mom_vve'] = S_last[0]
        #       for root_key, root_value in max_u2_e_last['0'].items():
        #           results_out[f'onepole_{root_key}'] = root_value

        #       results_out['twopole1_E_mom_vve'] = E[0]*Eh2eV
        #       results_out['twopole1_P_mom_vve'] = P[0]
        #       results_out['twopole1_S_mom_vve'] = snorm[0]
        #       for root_key, root_value in max_u2_e['0'].items():
        #           results_out[f'twopole1_{root_key}'] = root_value
        #       results_out['twopole2_E_mom_vve'] = E[2]*Eh2eV  
        #       results_out['twopole2_P_mom_vve'] = P[2]
        #       results_out['twopole2_S_mom_vve'] = snorm[2] 
        #       for root_key, root_value in max_u2_e['2'].items():
        #           results_out[f'twopole2_{root_key}'] = root_value
 

        #    elif e_vir_a[0] == big_thresh and iter_i < e_vir_a.size-1 and E.size == 2:
        #        print("-------------------------------------------------\nMOM pole structure is now equal to CVS pole structure \nRemoving all vve and leaving only v ...\n------------------------------")
        #        vve_mo_count = e_vir_a.size-iter_i
        #        idx_thresh = iter_i 
        #        results_out['onepole_E_vve_thresh'] = e_vir_a[idx_thresh]*Eh2eV
        #        results_out['onepole_nvir_vve'] = vve_mo_count
        #        results_out['onepole_E_mom_vve'] = E_last[0]*Eh2eV 
        #        results_out['onepole_P_mom_vve'] = P_last[0]
        #        results_out['onepole_S_mom_vve'] = S_last[0]
        #        for root_key, root_value in max_u2_e_last['0'].items():
        #           results_out[f'onepole_{root_key}'] = root_value
        #        iter_i = e_vir_a.size-1
        #        E_last = E
        #        first_pass = True

        #    elif e_vir_a[0] == big_thresh and iter_i == e_vir_a.size-1:
        #        print("==========================\nPrinting summary of results ...\n================================")
        #        print("Ecvs: ", E_cvs*Eh2eV)
        #        print("Emom: ", E_last*Eh2eV)
        #        print("Total number of non-degenerate virtual MOs: ", e_vir_a.size)
        #        print("Total number of non-degenerate virtual MOs used in vve: ", vve_mo_count)
        #        if first_pass is False:
        #            idx_thresh = iter_i
        #        print("Lowest vve energy threshold used (Eh): ", e_vir_a[idx_thresh])
        #        print("Lowest vve energy threshold used (eV): ", e_vir_a[idx_thresh]*Eh2eV)

        #        results_out['pole_E_mom_v'] = E[0]*Eh2eV
        #        results_out['pole_P_mom_v'] = P[0]
        #        results_out['pole_S_mom_v'] = snorm[0]
        #        #results_out['twopole2_E_mom_vve_2'] = "" 
        #        #results_out['twopole2_P_mom_vve_2'] = "" 
        #        #results_out['twopole2_S_mom_vve_2'] = ""  

        #        for root_key, root_value in max_u2_e['0'].items():
        #            results_out[f'pole_v_{root_key}'] = root_value

        #        #if E.size > 2: 
        #        #    results_out['twopole2_E_mom_vve_2'] = E[2]*Eh2eV  
        #        #    results_out['twopole2_P_mom_vve_2'] = P[2]
        #        #    results_out['twopole2_S_mom_vve_2'] = snorm[2]
 
        #        #    for root_key, root_value in max_u2_e['2'].items():
        #        #        results_out[f'twopole2_{root_key}'] = root_value
        #        iter_i = e_vir_a.size
        #            
        #    else:
        #       E_last = E
        #       P_last = P
        #       S_last = snorm
        #       max_u2_e_last = max_u2_e
        #       iter_i += 1

     #Retruning results for high throughput analysis. Implemented only for closed-shell systems


        #results_out['E_cvs'] = E_cvs[0]*Eh2eV
        #results_out['P_cvs'] = P_cvs[0]
        #results_out['nvir_full'] = e_vir_a.size
 
        #results_out['onepole_E_vve_HF'] = onepole_E_vve
        #results_out['onepole_vve_idx'] = onepole_vve_idx 
        #results_out['onepole_vve_U2'] = onepole_vve_U2
        #results_out['twopole1_E_vve_HF'] = twopole1_E_vve
        #results_out['twopole1_vve_idx'] = twopole1_vve_idx 
        #results_out['twopole1_vve_U2'] = twopole1_vve_U2
        #results_out['twopole2_E_vve_HF'] = twopole2_E_vve
        #results_out['twopole2_vve_idx'] = twopole2_vve_idx 
        #results_out['twopole2_vve_U2'] = twopole2_vve_U2
        
         
        #return (E_cvs, spec_factors_cvs, E_mom, spec_factors_mom) 
    #return results_out
    return (adc.e_corr, E_cvs, total_tpdm_idx)
    #return (E_cvs, total_tpdm_a, total_tpdm_b)
    #exit()
 
    nfalse = np.shape(conv)[0] - np.sum(conv)
    if nfalse >= 1:
        print ("*************************************************************")
        print (" WARNING : ", "CVS Davidson iterations for ",nfalse, "root(s) not converged")
        print ("*************************************************************")

    if adc.verbose >= logger.INFO:
        if nroots == 1:
            logger.info(adc, '%s root %d    Energy (Eh) = %.10f    Energy (eV) = %.8f    Spec factors = %.8f    conv = %s',
                         adc.method, 0, E, E*27.2114, spec_factors, conv)
        else :
            for n, en, pn, convn in zip(range(nroots), E, spec_factors, conv):
                logger.info(adc, '%s root %d    Energy (Eh) = %.10f    Energy (eV) = %.8f    Spec factors = %.8f    conv = %s',
                          adc.method, n, en, en*27.2114, pn, convn)

    log.timer('ADC', *cput0)

    #analyze_eigenvector_ip(adc, U)

    return E, U, spec_factors

def analyze_eigenvector_ip(adc, U, print_thresh = 0.05, energy_thresh=None, analyze_v_vve=False):

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncore_cvs = adc.ncore_cvs
    Eh2eV = 27.211396641308
    e_occ_a = adc.mo_energy_a[:nocc_a]*Eh2eV 
    e_occ_b = adc.mo_energy_b[:nocc_b]*Eh2eV
    e_vir_a = adc.mo_energy_a[nocc_a:]*Eh2eV
    e_vir_b = adc.mo_energy_b[nocc_b:]*Eh2eV

    logger.info(adc, "Number of alpha occupied orbitals = %d", nocc_a)
    logger.info(adc, "Number of beta occupied orbitals = %d", nocc_b)
    logger.info(adc, "Number of alpha virtual orbitals =  %d", nvir_a)
    logger.info(adc, "Number of beta virtual orbitals =  %d", nvir_b)
    logger.info(adc, "Print eigenvector elements > %f\n", print_thresh)

    ij_a = np.tril_indices(nocc_a, k=-1)
    ij_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a* nocc_b
    n_doubles_aba = nvir_a * nocc_b* nocc_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    max_u2_e = {}
    U_full = np.array(U)
    U = U_full.copy()
    for I in range(U.shape[0]):
        U1 = U[I,:f_b]
        U2 = U[I,f_b:]
        U1dotU1 = np.dot(U1, U1.conj()) 
        U2dotU2 = np.dot(U2, U2.conj())
        iter_f = copy.deepcopy(I) 
        evec_tols = (1e-3, print_thresh)
        for iter_i, evec_print_tol in enumerate(evec_tols):
            if iter_i == 1:
                U = U_full
            
            if iter_i == 0 and analyze_v_vve is True:
                U_vve, norm_v, norm_evv = vve_projector(adc, U[I, :], analyze_v_vve=True)
                U = U_vve.reshape((1,-1))
                I = 0

            #temp_aaa = np.zeros((nvir_a, nocc_a, nocc_a), dtype=complex)
            temp_aaa = np.zeros((nvir_a, nocc_a, nocc_a))
            temp_aaa[:,ij_a[0],ij_a[1]] =  U[I,s_aaa:f_aaa].reshape(nvir_a,-1).copy()
            temp_aaa[:,ij_a[1],ij_a[0]] = -U[I,s_aaa:f_aaa].reshape(nvir_a,-1).copy()
            U_aaa = temp_aaa.reshape(-1).copy()

            #temp_bbb = np.zeros((nvir_b, nocc_b, nocc_b), dtype=complex)
            temp_bbb = np.zeros((nvir_b, nocc_b, nocc_b))
            temp_bbb[:,ij_b[0],ij_b[1]] =  U[I,s_bbb:f_bbb].reshape(nvir_b,-1).copy()
            temp_bbb[:,ij_b[1],ij_b[0]] = -U[I,s_bbb:f_bbb].reshape(nvir_b,-1).copy()
            U_bbb = temp_bbb.reshape(-1).copy()

            #U_sq = U[I,:].copy()**2
            U_sq = np.multiply(U[I,:], U[I,:].conj())
            ind_idx = np.argsort(-U_sq)
            U_sq = U_sq[ind_idx] 
            U_sorted = U[I,ind_idx].copy()

            U_sq_aaa = np.multiply(U_aaa, U_aaa.conj())
            U_sq_bbb = np.multiply(U_bbb, U_bbb.conj())
            ind_idx_aaa = np.argsort(-U_sq_aaa)
            ind_idx_bbb = np.argsort(-U_sq_bbb)
            U_sq_aaa = U_sq_aaa[ind_idx_aaa]
            U_sq_bbb = U_sq_bbb[ind_idx_bbb]
            U_sorted_aaa = U_aaa[ind_idx_aaa].copy()
            U_sorted_bbb = U_bbb[ind_idx_bbb].copy()


            U_sorted = U_sorted[U_sq > evec_print_tol**2]
            ind_idx = ind_idx[U_sq > evec_print_tol**2]
            U_sorted_aaa = U_sorted_aaa[U_sq_aaa > evec_print_tol**2]
            U_sorted_bbb = U_sorted_bbb[U_sq_bbb > evec_print_tol**2]
            ind_idx_aaa = ind_idx_aaa[U_sq_aaa > evec_print_tol**2]
            ind_idx_bbb = ind_idx_bbb[U_sq_bbb > evec_print_tol**2]
            
            singles_a_idx = []
            singles_b_idx = []
            doubles_aaa_idx = []
            doubles_bab_idx = []
            doubles_aba_idx = []
            doubles_bbb_idx = []  
            singles_a_val = []
            singles_b_val = []
            doubles_bab_val = []
            doubles_aba_val = []  
            iter_idx = 0

            singles_a_energy = []
            singles_b_energy = []
            doubles_aaa_energy = []
            doubles_bab_energy = []
            doubles_aba_energy = []  
            doubles_bbb_energy = []
            for orb_idx in ind_idx:

                if orb_idx in range(s_a,f_a):
                    i_idx = orb_idx + 1
                    singles_a_idx.append(i_idx)
                    singles_a_val.append(U_sorted[iter_idx])
                    singles_a_energy.append(-e_occ_a[i_idx - 1])
                   
                if orb_idx in range(s_b,f_b):
                    i_idx = orb_idx - s_b + 1
                    singles_b_idx.append(i_idx)
                    singles_b_val.append(U_sorted[iter_idx])
                    singles_b_energy.append(-e_occ_b[i_idx - 1])

                if orb_idx in range(s_bab,f_bab):
                    aij_idx = orb_idx - s_bab       
                    ij_rem = aij_idx % (nocc_a*nocc_b)
                    a_idx = aij_idx//(nocc_a*nocc_b)
                    i_idx = ij_rem//nocc_a
                    j_idx = ij_rem % nocc_a
                    doubles_bab_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                    doubles_bab_val.append(U_sorted[iter_idx])
                    doubles_bab_energy.append(e_vir_b[a_idx] - e_occ_a[i_idx] - e_occ_b[j_idx])
              
                if orb_idx in range(s_aba,f_aba):
                    aij_idx = orb_idx - s_aba     
                    ij_rem = aij_idx % (nocc_b*nocc_a)
                    a_idx = aij_idx//(nocc_b*nocc_a)
                    i_idx = ij_rem//nocc_b
                    j_idx = ij_rem % nocc_b
                    doubles_aba_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                    doubles_aba_val.append(U_sorted[iter_idx])
                    doubles_aba_energy.append(e_vir_a[a_idx] - e_occ_b[i_idx] - e_occ_a[j_idx])

                iter_idx += 1
                 
            for orb_aaa in ind_idx_aaa:              
                ij_rem = orb_aaa % (nocc_a*nocc_a)
                a_idx = orb_aaa//(nocc_a*nocc_a)
                i_idx = ij_rem//nocc_a
                j_idx = ij_rem % nocc_a
                doubles_aaa_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                doubles_aaa_energy.append(e_vir_a[a_idx] - e_occ_a[i_idx] - e_occ_a[j_idx])

            for orb_bbb in ind_idx_bbb:                
                ij_rem = orb_bbb % (nocc_b*nocc_b)
                a_idx = orb_bbb//(nocc_b*nocc_b)
                i_idx = ij_rem//nocc_b
                j_idx = ij_rem % nocc_b
                doubles_bbb_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bbb_energy.append(e_vir_b[a_idx] - e_occ_b[i_idx] - e_occ_b[j_idx])

            doubles_aaa_val = list(U_sorted_aaa)
            doubles_bbb_val = list(U_sorted_bbb)

            avg_energy_a =   np.multiply(np.square(singles_a_val), singles_a_energy)
            avg_energy_b =   np.multiply(np.square(singles_b_val), singles_b_energy)
            avg_energy_aaa = np.multiply(np.square(doubles_aaa_val), doubles_aaa_energy)
            avg_energy_bab = np.multiply(np.square(doubles_bab_val), doubles_bab_energy)
            avg_energy_aba = np.multiply(np.square(doubles_aba_val), doubles_aba_energy)
            avg_energy_bbb = np.multiply(np.square(doubles_bbb_val), doubles_bbb_energy)

            if iter_i == 0:
                avg_energy_a_tot   = np.sum(avg_energy_a)              
                avg_energy_b_tot   = np.sum(avg_energy_b)
                avg_energy_aaa_tot = np.sum(avg_energy_aaa)/2 
                avg_energy_bab_tot = np.sum(avg_energy_bab) 
                avg_energy_aba_tot = np.sum(avg_energy_aba) 
                avg_energy_bbb_tot = np.sum(avg_energy_bbb)/2

                u2_a   = 0     
                e_a    = 0 
                u2_b   = 0
                e_b    = 0
                u2_aaa = 0
                e_aaa  = 0
                u2_bab = 0
                e_bab  = 0
                u2_aba = 0
                e_aba  = 0
                u2_bbb = 0
                e_bbb  = 0
                a_idx_v = 0
                b_idx_v = 0
                aaa_idx_v = 0
                bab_idx_v = 0
                aba_idx_v = 0
                bbb_idx_v = 0
                #if analyze_v_vve:
                #   if singles_a_val:
                #      singles_a_val_v = np.array(singles_a_val)
                #      singles_a_energy_v = np.array(singles_a_energy)
                #      u2_a = np.unique(singles_a_val_v[0]**2)
                #      e_a = np.unique(singles_a_energy_v[0])
                #      a_idx_v = np.array(singles_a_idx[0]) 
                #   if singles_b_val:
                #      singles_b_val_v = np.array(singles_b_val)
                #      singles_b_energy_v = np.array(singles_b_energy)
                #      u2_b = np.unique(singles_b_val_v[0]**2)
                #      e_b =  np.unique(singles_b_energy_v[0])
                #      b_idx_v = np.array(singles_b_idx[0]) 
                #   if doubles_aaa_val:
                #      doubles_aaa_val_v = np.array(doubles_aaa_val)
                #      doubles_aaa_energy_v = np.array(doubles_aaa_energy)
                #      u2_aaa = np.unique(doubles_aaa_val_v[0]**2)
                #      e_aaa = np.unique(doubles_aaa_energy_v[0])
                #      aaa_idx_v = np.array(doubles_aaa_idx[0]) 
                #   if doubles_bab_val:
                #      doubles_bab_val_v = np.array(doubles_bab_val)
                #      doubles_bab_energy_v = np.array(doubles_bab_energy)
                #      u2_bab = np.unique(doubles_bab_val_v[0]**2)
                #      e_bab = np.unique(doubles_bab_energy_v[0])
                #      bab_idx_v = np.array(doubles_bab_idx[0]) 
                #   if doubles_aba_val:
                #      doubles_aba_val_v = np.array(doubles_aba_val)
                #      doubles_aba_energy_v = np.array(doubles_aba_energy)
                #      u2_aba = np.unique(doubles_aba_val_v[0]**2)
                #      e_aba = np.unique(doubles_aba_energy_v[0])
                #      aba_idx_v = np.array(doubles_aba_idx[0]) 
                #   if doubles_bbb_val:
                #      doubles_bbb_val_v = np.array(doubles_bbb_val)
                #      doubles_bbb_energy_v = np.array(doubles_bbb_energy)
                #      u2_bbb = np.unique(doubles_bbb_val_v[0]**2)
                #      e_bbb = np.unique(doubles_bbb_energy_v[0])
                #      aba_idx_v = np.array(doubles_aba_idx[0])

                #   u2_singles = np.array([u2_a, u2_b], dtype=tuple)
                #   u2_doubles = np.array([u2_aaa, u2_bab, u2_aba, u2_bbb], tuple) 
                #   e_singles = {'0':e_a, '1':e_b}
                #   e_doubles = {'0':e_aaa, '1':e_bab, '2':e_aba, '3':e_bbb} 
                #   orb_singles = {'0':a_idx_v, '1':b_idx_v}
                #   orb_doubles = {'0':aaa_idx_v, '1':bab_idx_v, '2': aba_idx_v, '3': bbb_idx_v} 
                #   u2_singles_sorted_idx = np.argsort(-u2_singles) 
                #   u2_doubles_sorted_idx = np.argsort(-u2_doubles) 

                #   u2_singles_max = u2_singles[u2_singles_sorted_idx][0] 
                #   u2_doubles_max = u2_doubles[u2_doubles_sorted_idx][0] 
                #   e_singles_max = e_singles[f'{u2_singles_sorted_idx[0]}'] 
                #   e_doubles_max = e_doubles[f'{u2_doubles_sorted_idx[0]}'] 
                #   orb_singles_max = orb_singles[f'{u2_singles_sorted_idx[0]}'] 
                #   orb_doubles_max = orb_doubles[f'{u2_doubles_sorted_idx[0]}']
 

                   #max_u2_e[f'{iter_f}'] = {'u2_singles': u2_singles_max, 'u2_doubles': u2_doubles_max, 'e_singles': e_singles_max, 'e_doubles': e_doubles_max, 'orb_singles': orb_singles_max, 'orb_doubles': orb_doubles_max, 'norm_v_full':norm_v, 'norm_vve_full':norm_evv}

            if iter_i == 1:
        
                logger.info(adc, "----------------------------------------------------------------------------------------")
                logger.info(adc,'%s | root %d | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',adc.method ,I, U1dotU1, U2dotU2)
                logger.info(adc, "----------------------------------------------------------------------------------------")

                if singles_a_val:
                        logger.info(adc, "1h(alpha) | thresh_weighted_energy = %.4f | total_weighted_energy = %.4f ", avg_energy_a_tot, np.sum(avg_energy_a)) 
                        logger.info(adc, "----------------------------------------------")
                        logger.info(adc, "     i    U(i)    HF_energy(eV)   weighted(eV)")
                        logger.info(adc, "----------------------------------------------")
                        for idx, print_singles in enumerate(singles_a_idx):
                            logger.info(adc, '  %4d   %7.4f   %8.2f   %12.2f', print_singles, singles_a_val[idx], singles_a_energy[idx], avg_energy_a[idx])

                if singles_b_val:
                        logger.info(adc, "1h(beta) | thresh_weighted_energy = %.4f | total_weighted_energy = %.4f ", avg_energy_b_tot, np.sum(avg_energy_b)) 
                        logger.info(adc, "----------------------------------------------")
                        logger.info(adc, "     i    U(i)    HF_energy(eV)   weighted(eV)")
                        logger.info(adc, "----------------------------------------------")
                        for idx, print_singles in enumerate(singles_b_idx):
                            logger.info(adc, '  %4d   %7.4f   %8.2f   %12.2f', print_singles, singles_b_val[idx], singles_b_energy[idx], avg_energy_b[idx])

                if doubles_aaa_val:
                        logger.info(adc, "\n2h1p(alpha|alpha|alpha) | threshold_weighted_energy(eV) = %.4f | total_weighted_energy(eV) = %.4f", np.sum(avg_energy_aaa), avg_energy_aaa_tot) 
                        logger.info(adc, "-------------------------------------------------------------------")
                        logger.info(adc, "     i     j     a     U(i,j,a)   HF_energy(eV)    weighted(eV)")
                        logger.info(adc, "-------------------------------------------------------------------")
                        for idx, print_doubles in enumerate(doubles_aaa_idx):
                            logger.info(adc, '  %4d  %4d  %4d   %7.4f  %13.2f   %13.2f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_aaa_val[idx], doubles_aaa_energy[idx], avg_energy_aaa[idx])

                if doubles_bab_val:
                        logger.info(adc, "\n2h1p(beta|alpha|beta) | threshold_weighted_energy(eV) = %.4f | total_weighted_energy(eV) = %.4f", np.sum(avg_energy_bab), avg_energy_bab_tot) 
                        logger.info(adc, "-------------------------------------------------------------------")
                        logger.info(adc, "     i     j     a     U(i,j,a)   HF_energy(eV)    weighted(eV)")
                        logger.info(adc, "-------------------------------------------------------------------")
                        for idx, print_doubles in enumerate(doubles_bab_idx):
                            logger.info(adc, '  %4d  %4d  %4d   %7.4f  %13.2f   %13.2f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_bab_val[idx], doubles_bab_energy[idx], avg_energy_bab[idx])

                if doubles_aba_val:
                        logger.info(adc, "\n2h1p(alpha|beta|alpha) | threshold_weighted_energy(eV) = %.4f | total_weighted_energy(eV) = %.4f", np.sum(avg_energy_aba), avg_energy_aba_tot) 
                        logger.info(adc, "-------------------------------------------------------------------")
                        logger.info(adc, "     i     j     a     U(i,j,a)   HF_energy(eV)    weighted(eV)")
                        logger.info(adc, "-------------------------------------------------------------------")
                        for idx, print_doubles in enumerate(doubles_aba_idx):
                            logger.info(adc, '  %4d  %4d  %4d   %7.4f  %13.2f   %13.2f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_aba_val[idx], doubles_aba_energy[idx], avg_energy_aba[idx])

                if doubles_bbb_val:
                        logger.info(adc, "\n2h1p(beta|beta|beta) | threshold_weighted_energy(eV) = %.4f | total_weighted_energy(eV) = %.4f", np.sum(avg_energy_bbb), avg_energy_bbb_tot) 
                        logger.info(adc, "-------------------------------------------------------------------")
                        logger.info(adc, "     i     j     a     U(i,j,a)   HF_energy(eV)    weighted(eV)")
                        logger.info(adc, "-------------------------------------------------------------------")
                        for idx, print_doubles in enumerate(doubles_bbb_idx):
                            logger.info(adc, '  %4d  %4d  %4d   %7.4f  %13.2f   %13.2f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_bbb_val[idx], doubles_bbb_energy[idx], avg_energy_bbb[idx])

                logger.info(adc, "\n*************************************************************\n")
                sys.stdout.flush()

    if analyze_v_vve:
        return norm_v, norm_evv, max_u2_e 
                 


def compute_amplitudes_energy(myadc, eris, verbose=None, fc_bool=True):

    t1, t2 = myadc.compute_amplitudes(eris, fc_bool=True) 
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris, fc_bool=True):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    if fc_bool is False:
        nfc_orb = 0
    else:
        nfc_orb = myadc.nfc_orb

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_oovv = eris.oovv
    eris_OOVV = eris.OOVV
    eris_ooVV = eris.ooVV
    eris_OOvv = eris.OOvv
    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov
    eris_ovoo = eris.ovoo
    eris_OVoo = eris.OVoo
    eris_ovOO = eris.ovOO
    eris_OVOO = eris.OVOO

    e_a = myadc.mo_energy_a
    e_b = myadc.mo_energy_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris_ovov[:].transpose(0,2,1,3).copy()
    v2e_oovv -= eris_ovov[:].transpose(0,2,3,1).copy()

    d_ij_a = e_a[:nocc_a][:,None] + e_a[:nocc_a]
    d_ab_a = e_a[nocc_a:][:,None] + e_a[nocc_a:]

    D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
    D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))

    t2_1_a = v2e_oovv/D2_a
    del (v2e_oovv)
    del (D2_a)

    v2e_OOVV = eris_OVOV[:].transpose(0,2,1,3).copy()
    v2e_OOVV -= eris_OVOV[:].transpose(0,2,3,1).copy()

    d_ij_b = e_b[:nocc_b][:,None] + e_b[:nocc_b]
    d_ab_b = e_b[nocc_b:][:,None] + e_b[nocc_b:]

    D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
    D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))

    t2_1_b = v2e_OOVV/D2_b
    del (v2e_OOVV)
    del (D2_b)

    v2e_oOvV = eris_ovOV[:].transpose(0,2,1,3).copy()

    d_ij_ab = e_a[:nocc_a][:,None] + e_b[:nocc_b]
    d_ab_ab = e_a[nocc_a:][:,None] + e_b[nocc_b:]

    D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
    D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))

    t2_1_ab = v2e_oOvV/D2_ab
    del (v2e_oOvV)
    del (D2_ab)

    # Frozen core
    t2_1_a[:nfc_orb,:,:,:] = 0
    t2_1_a[:,:nfc_orb,:,:] = 0
    t2_1_a[:nfc_orb,:nfc_orb,:,:] = 0
    t2_1_ab[:nfc_orb,:,:,:] = 0
    t2_1_ab[:,:nfc_orb,:,:] = 0
    t2_1_ab[:nfc_orb,:nfc_orb,:,:] = 0
    t2_1_b[:nfc_orb,:,:,:] = 0
    t2_1_b[:,:nfc_orb,:,:] = 0
    t2_1_b[:nfc_orb,:nfc_orb,:,:] = 0

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)

    D1_a = e_a[:nocc_a][:None].reshape(-1,1) - e_a[nocc_a:].reshape(-1)
    D1_b = e_b[:nocc_b][:None].reshape(-1,1) - e_b[nocc_b:].reshape(-1)
    D1_a = D1_a.reshape((nocc_a,nvir_a))
    D1_b = D1_b.reshape((nocc_b,nvir_b))

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    # Compute second-order singles t1 (tij)

    t1_2_a = np.zeros((nocc_a,nvir_a))
    t1_2_b = np.zeros((nocc_b,nvir_b))

    if isinstance(eris.ovvv, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_a

    a = 0
    for p in range(0,nocc_a,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_ovvv = dfadc.get_ovvv_a_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
        k = eris_ovvv.shape[0]

        t1_2_a += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
        t1_2_a -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
        del eris_ovvv
        a += k

    if isinstance(eris.OVvv, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_b

    a = 0
    for p in range(0,nocc_b,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_OVvv = dfadc.get_ovvv_a_df(myadc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
        k = eris_OVvv.shape[0]
        t1_2_a += lib.einsum('kdac,ikcd->ia',eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
        del eris_OVvv
        a += k

    if isinstance(eris.OVVV, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_b
    a = 0
    for p in range(0,nocc_b,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_OVVV = dfadc.get_ovvv_a_df(myadc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
        k = eris_OVVV.shape[0]
        t1_2_b += 0.5*lib.einsum('kdac,ikcd->ia',eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
        t1_2_b -= 0.5*lib.einsum('kcad,ikcd->ia',eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
        del eris_OVVV
        a += k

    if isinstance(eris.ovVV, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_a
    a = 0
    for p in range(0,nocc_a,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_ovVV = dfadc.get_ovvv_a_df(myadc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
        k = eris_ovVV.shape[0]
        t1_2_b += lib.einsum('kdac,kidc->ia',eris_ovVV,t2_1_ab[a:a+k],optimize=True)
        del eris_ovVV
        a += k

    t1_2_a -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1_a,optimize=True)
    t1_2_a += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1_a,optimize=True)
    t1_2_a -= lib.einsum('lcki,klac->ia',eris_OVoo,t2_1_ab,optimize=True)
    t1_2_b -= 0.5*lib.einsum('lcki,klac->ia',eris_OVOO,t2_1_b,optimize=True)
    t1_2_b += 0.5*lib.einsum('kcli,klac->ia',eris_OVOO,t2_1_b,optimize=True)
    t1_2_b -= lib.einsum('lcki,lkca->ia',eris_ovOO,t2_1_ab)

    t1_2_a = t1_2_a/D1_a
    t1_2_b = t1_2_b/D1_b

    cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    # Frozen core
    t1_2_a[:nfc_orb,:] = 0
    t1_2_b[:nfc_orb,:] = 0

    t1_2 = (t1_2_a , t1_2_b)
    t2_2 = (None,)
    t1_3 = (None,)

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

    # Compute second-order doubles t2 (tijab)

        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO

        t2_2_temp = None
        if isinstance(eris.vvvv_p, np.ndarray):
            eris_vvvv = eris.vvvv_p
            temp = np.ascontiguousarray(t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]])
            t2_2_temp = np.dot(temp,eris_vvvv.T)
            del eris_vvvv
        elif isinstance(eris.vvvv_p, list): 
            t2_2_temp = contract_ladder_antisym(myadc,t2_1_a, eris.vvvv_p)
        else:
            t2_2_temp = contract_ladder_antisym(myadc,t2_1_a, eris.Lvv)

        t2_2_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))   
        t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]] = t2_2_temp    
        t2_2_a[:,:,ab_ind_a[1],ab_ind_a[0]] = -t2_2_temp    
        t2_2_a += 0.5*lib.einsum('kilj,klab->ijab', eris_oooo, t2_1_a,optimize=True)
        t2_2_a -= 0.5*lib.einsum('kjli,klab->ijab', eris_oooo, t2_1_a,optimize=True)

        temp = lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1_a,optimize=True)
        temp -= lib.einsum('kjbc,kica->ijab',eris_oovv,t2_1_a,optimize=True)
        temp_1 = lib.einsum('kcbj,ikac->ijab',eris_OVvo,t2_1_ab,optimize=True)

        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        del (temp)
        del (temp_1)

        t2_2_temp = None
        if isinstance(eris.VVVV_p, np.ndarray):
            eris_VVVV = eris.VVVV_p
            temp = np.ascontiguousarray(t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]])
            t2_2_temp = np.dot(temp,eris_VVVV.T)
            del eris_VVVV
        elif isinstance(eris.VVVV_p, list) : 
            t2_2_temp = contract_ladder_antisym(myadc,t2_1_b,eris.VVVV_p)
        else :
            t2_2_temp = contract_ladder_antisym(myadc,t2_1_b,eris.LVV)

        t2_2_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))   
        t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]] = t2_2_temp    
        t2_2_b[:,:,ab_ind_b[1],ab_ind_b[0]] = -t2_2_temp    
        t2_2_b += 0.5*lib.einsum('kilj,klab->ijab', eris_OOOO, t2_1_b,optimize=True)
        t2_2_b -= 0.5*lib.einsum('kjli,klab->ijab', eris_OOOO, t2_1_b,optimize=True)
    
        del (t2_2_temp)

        temp = lib.einsum('kcbj,kica->ijab',eris_OVVO,t2_1_b,optimize=True)
        temp -= lib.einsum('kjbc,kica->ijab',eris_OOVV,t2_1_b,optimize=True)
        temp_1 = lib.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_ab,optimize=True)

        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        del (temp)
        del (temp_1)

        if isinstance(eris.vVvV_p, np.ndarray):
            temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
            eris_vVvV = eris.vVvV_p
            t2_2_ab = np.dot(temp,eris_vVvV.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        elif isinstance(eris.vVvV_p, list):
            t2_2_ab = contract_ladder(myadc,t2_1_ab,eris.vVvV_p)
        else :
            t2_2_ab = contract_ladder(myadc,t2_1_ab,(eris.Lvv,eris.LVV))

        t2_2_ab += lib.einsum('kilj,klab->ijab',eris_ooOO,t2_1_ab,optimize=True)
        t2_2_ab += lib.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_a,optimize=True)
        t2_2_ab += lib.einsum('kcbj,ikac->ijab',eris_OVVO,t2_1_ab,optimize=True)
        t2_2_ab -= lib.einsum('kjbc,ikac->ijab',eris_OOVV,t2_1_ab,optimize=True)
        t2_2_ab -= lib.einsum('kibc,kjac->ijab',eris_ooVV,t2_1_ab,optimize=True)
        t2_2_ab -= lib.einsum('kjac,ikcb->ijab',eris_OOvv,t2_1_ab,optimize=True)
        t2_2_ab += lib.einsum('kcai,kjcb->ijab',eris_OVvo,t2_1_b,optimize=True)
        t2_2_ab += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1_ab,optimize=True)
        t2_2_ab -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1_ab,optimize=True)

        D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
        D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
        t2_2_a = t2_2_a/D2_a
        del (D2_a)

        D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
        D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))
        t2_2_b = t2_2_b/D2_b
        del (D2_b)

        D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
        D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))
        t2_2_ab = t2_2_ab/D2_ab
        del (D2_ab)

        # Frozen core
        t2_2_a[:nfc_orb,:,:,:] = 0
        t2_2_a[:,:nfc_orb,:,:] = 0      
        t2_2_a[:nfc_orb,:nfc_orb,:,:] = 0
        t2_2_ab[:nfc_orb,:,:,:] = 0
        t2_2_ab[:,:nfc_orb,:,:] = 0      
        t2_2_ab[:nfc_orb,:nfc_orb,:,:] = 0
        t2_2_b[:nfc_orb,:,:,:] = 0
        t2_2_b[:,:nfc_orb,:,:] = 0      
        t2_2_b[:nfc_orb,:nfc_orb,:,:] = 0

        t2_2 = (t2_2_a , t2_2_ab, t2_2_b)
    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

    if (myadc.method == "adc(3)"):
    # Compute third-order singles (tij)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

        t1_3 = (None,)

        t1_3_a = lib.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_a,t1_2_a,optimize=True)
        t1_3_a += lib.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_b,optimize=True)
 
        t1_3_b  = lib.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_b, t1_2_b,optimize=True)
        t1_3_b += lib.einsum('d,lida,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_a,optimize=True)
 
        t1_3_a -= lib.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a, t1_2_a,optimize=True)
        t1_3_a -= lib.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_b,optimize=True)
 
        t1_3_b -= lib.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_b, t1_2_b,optimize=True)
        t1_3_b -= lib.einsum('l,lida,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_a,optimize=True)
 
        t1_3_a += 0.5*lib.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a, t1_2_a,optimize=True)
        t1_3_a += 0.5*lib.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_b,optimize=True)
 
        t1_3_b += 0.5*lib.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_b, t1_2_b,optimize=True)
        t1_3_b += 0.5*lib.einsum('a,lida,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_a,optimize=True)
 
        t1_3_a -= 0.5*lib.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a, t1_2_a,optimize=True)
        t1_3_a -= 0.5*lib.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_b,optimize=True)
 
        t1_3_b -= 0.5*lib.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_b, t1_2_b,optimize=True)
        t1_3_b -= 0.5*lib.einsum('i,lida,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_a,optimize=True)
 
        t1_3_a += lib.einsum('ld,iald->ia',t1_2_a,eris_ovov,optimize=True)
        t1_3_a -= lib.einsum('ld,laid->ia',t1_2_a,eris_ovov,optimize=True)
        t1_3_a += lib.einsum('ld,iald->ia',t1_2_b,eris_ovOV,optimize=True)
 
        t1_3_b += lib.einsum('ld,iald->ia',t1_2_b,eris_OVOV ,optimize=True)
        t1_3_b -= lib.einsum('ld,laid->ia',t1_2_b,eris_OVOV ,optimize=True)
        t1_3_b += lib.einsum('ld,ldia->ia',t1_2_a,eris_ovOV,optimize=True)
 
        t1_3_a += lib.einsum('ld,ldai->ia',t1_2_a,eris_ovvo ,optimize=True)
        t1_3_a -= lib.einsum('ld,liad->ia',t1_2_a,eris_oovv ,optimize=True)
        t1_3_a += lib.einsum('ld,ldai->ia',t1_2_b,eris_OVvo,optimize=True)
 
        t1_3_b += lib.einsum('ld,ldai->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b -= lib.einsum('ld,liad->ia',t1_2_b,eris_OOVV ,optimize=True)
        t1_3_b += lib.einsum('ld,ldai->ia',t1_2_a,eris_ovVO,optimize=True)
 
        t1_3_a -= 0.5*lib.einsum('lmad,mdli->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3_a += 0.5*lib.einsum('lmad,ldmi->ia',t2_2_a,eris_ovoo,optimize=True)
        t1_3_a -=     lib.einsum('lmad,mdli->ia',t2_2_ab,eris_OVoo,optimize=True)
 
        t1_3_b -= 0.5*lib.einsum('lmad,mdli->ia',t2_2_b,eris_OVOO,optimize=True)
        t1_3_b += 0.5*lib.einsum('lmad,ldmi->ia',t2_2_b,eris_OVOO,optimize=True)
        t1_3_b -=     lib.einsum('mlda,mdli->ia',t2_2_ab,eris_ovOO,optimize=True)
 
        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_a

        a = 0
        for p in range(0,nocc_a,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_a_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            k = eris_ovvv.shape[0]
            t1_3_a += 0.5*lib.einsum('ilde,lead->ia',t2_2_a[:,a:a+k],eris_ovvv,optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_a[:,a:a+k],eris_ovvv,optimize=True)
            t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a, eris_ovvv,  t2_1_a[:,a:a+k] ,optimize=True)
            t1_3_a += lib.einsum('ildf,mafe,lmde->ia',t2_1_a, eris_ovvv,  t2_1_a[:,a:a+k] ,optimize=True)
            t1_3_a += lib.einsum('ilfd,mefa,mled->ia',t2_1_ab,eris_ovvv, t2_1_ab[a:a+k],optimize=True)
            t1_3_a -= lib.einsum('ilfd,mafe,mled->ia',t2_1_ab,eris_ovvv, t2_1_ab[a:a+k],optimize=True)
            t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_a,eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_a,eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_b += 0.5*lib.einsum('lifa,mefd,lmde->ia',t2_1_ab,eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_b -= 0.5*lib.einsum('lifa,mdfe,lmde->ia',t2_1_ab,eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_a[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
            t1_3_a[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
            t1_3_a[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab,eris_ovvv,t2_1_ab,optimize=True)
            t1_3_a[a:a+k] -= lib.einsum('mlfd,ifea,mled->ia',t2_1_ab,eris_ovvv,t2_1_ab,optimize=True)
            t1_3_a[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
            t1_3_a[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1_a,eris_ovvv,t2_1_a,optimize=True)
            del eris_ovvv
            a += k

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_b
        a = 0
        for p in range(0,nocc_b,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_OVVV = dfadc.get_ovvv_a_df(myadc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
            else :
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            k = eris_OVVV.shape[0]
            t1_3_b += 0.5*lib.einsum('ilde,lead->ia',t2_2_b[:,a:a+k],eris_OVVV,optimize=True)
            t1_3_b -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_b[:,a:a+k],eris_OVVV,optimize=True)
            t1_3_b -= lib.einsum('ildf,mefa,lmde->ia',t2_1_b,eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b += lib.einsum('ildf,mafe,lmde->ia',t2_1_b,eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b += lib.einsum('lidf,mefa,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_b -= lib.einsum('lidf,mafe,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_ab,eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_ab,eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_b,eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_b,eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
            t1_3_b[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
            t1_3_b[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab,optimize=True)
            t1_3_b[a:a+k] -= lib.einsum('lmdf,ifea,lmde->ia',t2_1_ab,eris_OVVV,t2_1_ab,optimize=True)
            t1_3_b[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
            t1_3_b[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1_b,eris_OVVV,t2_1_b,optimize=True)
            del eris_OVVV
            a += k

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_a
        a = 0
        for p in range(0,nocc_a,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_ovVV = dfadc.get_ovvv_a_df(myadc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            k = eris_ovVV.shape[0]
            t1_3_b += lib.einsum('lied,lead->ia',t2_2_ab[a:a+k],eris_ovVV,optimize=True)
            t1_3_a -= lib.einsum('ildf,mafe,mlde->ia',t2_1_ab,eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_b -= lib.einsum('ildf,mefa,mled->ia',t2_1_b,eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_b += lib.einsum('lidf,mefa,lmde->ia',t2_1_ab,eris_ovVV,t2_1_a[:,a:a+k],optimize=True)
            t1_3_a += lib.einsum('ilaf,mefd,mled->ia',t2_1_ab,eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_b += lib.einsum('ilaf,mefd,mled->ia',t2_1_b,eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_a[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_b,eris_ovVV,t2_1_b,optimize=True)
            t1_3_a[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab,eris_ovVV,t2_1_ab,optimize=True)
            t1_3_a[a:a+k] -= lib.einsum('lmef,iedf,lmad->ia',t2_1_ab,eris_ovVV,t2_1_ab,optimize=True)
            del eris_ovVV
            a += k

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_b
        a = 0
        for p in range(0,nocc_b,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_OVvv = dfadc.get_ovvv_a_df(myadc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
            else :
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            k = eris_OVvv.shape[0]
            t1_3_a += lib.einsum('ilde,lead->ia',t2_2_ab[:,a:a+k],eris_OVvv,optimize=True)
            t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a,eris_OVvv, t2_1_ab[:,a:a+k],optimize=True)
            t1_3_a += lib.einsum('ilfd,mefa,lmde->ia',t2_1_ab,eris_OVvv,t2_1_b[:,a:a+k] ,optimize=True)
            t1_3_b -= lib.einsum('lifd,mafe,lmed->ia',t2_1_ab,eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_a += lib.einsum('ilaf,mefd,lmde->ia',t2_1_a,eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_b += lib.einsum('lifa,mefd,lmde->ia',t2_1_ab,eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_b[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_a,eris_OVvv,t2_1_a,optimize=True)
            t1_3_b[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab,eris_OVvv,t2_1_ab,optimize=True)
            t1_3_b[a:a+k] -= lib.einsum('mlfe,iedf,mlda->ia',t2_1_ab,eris_OVvv,t2_1_ab,optimize=True)

            del eris_OVvv
            a += k
 
        t1_3_a += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += lib.einsum('inde,lamn,lmde->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
 
        t1_3_b += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += lib.einsum('nied,lamn,mled->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
 
        t1_3_a += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1_a,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1_a,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 *lib.einsum('inad,lemn,lmed->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,mled->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_a += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_ab,eris_OVOO,t2_1_b,optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_ab,eris_OVOO,t2_1_b,optimize=True)
 
        t1_3_b += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= 0.5 * lib.einsum('inad,meln,mled->ia',t2_1_b,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 * lib.einsum('inad,lemn,lmed->ia',t2_1_b,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 *lib.einsum('nida,meln,lmde->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*lib.einsum('nida,lemn,mlde->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_b += 0.5*lib.einsum('nida,lemn,lmde->ia',t2_1_ab,eris_ovoo,t2_1_a,optimize=True)
        t1_3_b -= 0.5*lib.einsum('nida,meln,lmde->ia',t2_1_ab,eris_ovoo,t2_1_a,optimize=True)
 
        t1_3_a -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a -= lib.einsum('nled,ianm,mled->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a += lib.einsum('nled,naim,mled->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_b,eris_ovOO,t2_1_b,optimize=True)
        t1_3_a -= lib.einsum('lnde,ianm,lmde->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
 
        t1_3_b -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b -= lib.einsum('lnde,ianm,lmde->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b += lib.einsum('lnde,naim,lmde->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a,eris_OVoo,t2_1_a,optimize=True)
        t1_3_b -= lib.einsum('nled,ianm,mled->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
 
        t1_3_a -= lib.einsum('lnde,ienm,lmad->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += lib.einsum('lnde,neim,lmad->ia',t2_1_a,eris_ovoo,t2_1_a,optimize=True)
        t1_3_a += lib.einsum('lnde,neim,lmad->ia',t2_1_ab,eris_OVoo,t2_1_a,optimize=True)
        t1_3_a += lib.einsum('nled,ienm,mlad->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a -= lib.einsum('nled,neim,mlad->ia',t2_1_ab,eris_ovoo,t2_1_ab,optimize=True)
        t1_3_a += lib.einsum('lned,ienm,lmad->ia',t2_1_ab,eris_ovOO,t2_1_ab,optimize=True)
        t1_3_a -= lib.einsum('lnde,neim,mlad->ia',t2_1_b,eris_OVoo,t2_1_ab,optimize=True)
 
        t1_3_b -= lib.einsum('lnde,ienm,lmad->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += lib.einsum('lnde,neim,lmad->ia',t2_1_b,eris_OVOO,t2_1_b,optimize=True)
        t1_3_b += lib.einsum('nled,neim,lmad->ia',t2_1_ab,eris_ovOO,t2_1_b,optimize=True)
        t1_3_b += lib.einsum('lnde,ienm,lmda->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b -= lib.einsum('lnde,neim,lmda->ia',t2_1_ab,eris_OVOO,t2_1_ab,optimize=True)
        t1_3_b += lib.einsum('nlde,ienm,mlda->ia',t2_1_ab,eris_OVoo,t2_1_ab,optimize=True)
        t1_3_b -= lib.einsum('lnde,neim,lmda->ia',t2_1_a,eris_ovOO,t2_1_ab,optimize=True)
 
        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b

        # Frozen core
        t1_3_a[:nfc_orb,:] = 0
        t1_3_b[:nfc_orb,:] = 0

        t1_3 = (t1_3_a, t1_3_b)

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)
    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)

    del (D1_a, D1_b) 

    return t1, t2


def compute_energy(myadc, t1, t2, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV

    t2_1_a, t2_1_ab, t2_1_b  = t2[0]

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    #Compute MP2 correlation energy
    e_mp2 = 0.25 * lib.einsum('ijab,iajb', t2_1_a, eris_ovov)
    e_mp2 -= 0.25 * lib.einsum('ijab,ibja', t2_1_a, eris_ovov)
    e_mp2 += lib.einsum('ijab,iajb', t2_1_ab, eris_ovOV)
    e_mp2 += 0.25 * lib.einsum('ijab,iajb', t2_1_b, eris_OVOV)
    e_mp2 -= 0.25 * lib.einsum('ijab,ibja', t2_1_b, eris_OVOV)

    e_corr = e_mp2

    if (myadc.method == "adc(3)"):

        #Compute MP3 correlation energy
        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_OOvv = eris.OOvv
        eris_ooVV = eris.ooVV
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        t2_1_a_tril = np.ascontiguousarray(t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]])
        t2_1_b_tril = np.ascontiguousarray(t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]])

        if isinstance(eris.vvvv_p, np.ndarray):
            eris_vvvv = eris.vvvv_p
            temp_1_a = np.dot(t2_1_a_tril,eris_vvvv.T)
            del eris_vvvv
        elif isinstance(eris.vvvv_p, list) : 
            temp_1_a = contract_ladder_antisym(myadc,t2_1_a,eris.vvvv_p)
        else :
            temp_1_a = contract_ladder_antisym(myadc,t2_1_a,eris.Lvv)

        e_mp3 = 0.5 * lib.einsum('ijp,ijp',temp_1_a, t2_1_a_tril)
        del temp_1_a

        if isinstance(eris.VVVV_p, np.ndarray):
            eris_VVVV = eris.VVVV_p
            temp_1_b = np.dot(t2_1_b_tril,eris_VVVV.T)
            del eris_VVVV
        elif isinstance(eris.VVVV_p, list):
            temp_1_b = contract_ladder_antisym(myadc,t2_1_b,eris.VVVV_p)
        else : 
            temp_1_b = contract_ladder_antisym(myadc,t2_1_b,eris.LVV)

        e_mp3 += 0.5 * lib.einsum('ijp,ijp',temp_1_b, t2_1_b_tril)
        del temp_1_b

        if isinstance(eris.vVvV_p, np.ndarray):
            temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
            eris_vVvV = eris.vVvV_p
            temp_1_ab = np.dot(temp,eris_vVvV.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        elif isinstance(eris.vVvV_p, list):
            temp_1_ab = contract_ladder(myadc,t2_1_ab,eris.vVvV_p)
        else : 
            temp_1_ab = contract_ladder(myadc,t2_1_ab,(eris.Lvv,eris.LVV))

        e_mp3 +=  lib.einsum('ijcd,ijcd',temp_1_ab, t2_1_ab)
        del temp_1_ab

        temp_2_a =  lib.einsum('ijab,klab', t2_1_a, t2_1_a,optimize=True)
        e_mp3 += 0.125 * lib.einsum('ijkl,ikjl',temp_2_a, eris_oooo,optimize=True)
        e_mp3 -= 0.125 * lib.einsum('ijkl,iljk',temp_2_a, eris_oooo,optimize=True)
        del temp_2_a

        temp_2_b =  lib.einsum('ijab,klab', t2_1_b, t2_1_b,optimize=True)
        e_mp3 += 0.125 * lib.einsum('ijkl,ikjl',temp_2_b, eris_OOOO,optimize=True)
        e_mp3 -= 0.125 * lib.einsum('ijkl,iljk',temp_2_b, eris_OOOO,optimize=True)
        del temp_2_b

        temp_2_ab_1 =  lib.einsum('ijab,klab', t2_1_ab, t2_1_ab,optimize=True)
        e_mp3 +=  lib.einsum('ijkl,ikjl',temp_2_ab_1, eris_ooOO,optimize=True)
        del temp_2_ab_1

        temp_3_a = lib.einsum('ijab,ikcb->akcj', t2_1_a, t2_1_a,optimize=True)
        temp_3_a += lib.einsum('jiab,kicb->akcj', t2_1_ab, t2_1_ab,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_3_a, eris_oovv,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_3_a, eris_ovvo,optimize=True)
        del temp_3_a

        temp_3_b = lib.einsum('ijab,ikcb->akcj', t2_1_b, t2_1_b,optimize=True)
        temp_3_b += lib.einsum('ijba,ikbc->akcj', t2_1_ab, t2_1_ab,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_3_b, eris_OOVV,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_3_b, eris_OVVO,optimize=True)
        del temp_3_b

        temp_3_ab_1 = lib.einsum('ijab,ikcb->akcj', t2_1_ab, t2_1_ab,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_3_ab_1, eris_OOvv,optimize=True)
        del temp_3_ab_1
   
        temp_3_ab_2 = lib.einsum('jiba,kibc->akcj', t2_1_ab, t2_1_ab,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_3_ab_2, eris_ooVV,optimize=True)
        del temp_3_ab_2

        temp_3_ab_3 = -lib.einsum('ijab,ikbc->akcj', t2_1_a, t2_1_ab,optimize=True)
        temp_3_ab_3 -= lib.einsum('jiab,ikcb->akcj', t2_1_ab, t2_1_b,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_3_ab_3, eris_OVvo,optimize=True)
        del temp_3_ab_3

        temp_3_ab_4 = -lib.einsum('ijba,ikcb->akcj', t2_1_ab, t2_1_a,optimize=True)
        temp_3_ab_4 -= lib.einsum('ijab,kicb->akcj', t2_1_b, t2_1_ab,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_3_ab_4, eris_ovVO,optimize=True)
        del temp_3_ab_4
    
        e_corr += e_mp3
    
    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)

    return e_corr

def contract_ladder_antisym(myadc,t_amp,vvvv_d):

    nocc = t_amp.shape[0]
    nvir = t_amp.shape[2]

    nv_pair = nvir  *  (nvir - 1) // 2
    tril_idx = np.tril_indices(nvir, k=-1)               

    t_amp = t_amp[:,:,tril_idx[0],tril_idx[1]]
    t_amp_t = np.ascontiguousarray(t_amp.reshape(nocc*nocc,-1).T)

    t = np.zeros((nvir,nvir, nocc*nocc))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv_d, list):
        for dataset in vvvv_d:
             k = dataset.shape[0]
             dataset = dataset[:].reshape(-1,nv_pair)
             t[a:a+k] = np.dot(dataset,t_amp_t).reshape(-1,nvir,nocc*nocc)
             a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv = dfadc.get_vvvv_antisym_df(myadc, vvvv_d, p, chnk_size)
            k = vvvv.shape[0]
            vvvv = vvvv.reshape(-1,nv_pair)
            t[a:a+k] = np.dot(vvvv,t_amp_t).reshape(-1,nvir,nocc*nocc)
            del (vvvv)
            a += k
    else :
        raise Exception("Unknown vvvv type") 

    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc, nocc, nvir, nvir)
    t = t[:, :, tril_idx[0], tril_idx[1]]
    
    return t

def contract_ladder(myadc,t_amp,vvvv_p):

    nocc_a = t_amp.shape[0]
    nocc_b = t_amp.shape[1]
    nvir_a = t_amp.shape[2]
    nvir_b = t_amp.shape[3]

    t_amp_t = np.ascontiguousarray(t_amp.reshape(nocc_a*nocc_b,-1).T)
    t = np.zeros((nvir_a,nvir_b, nocc_a*nocc_b))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv_p, list):
        for dataset in vvvv_p:
             k = dataset.shape[0]
             dataset = dataset[:].reshape(-1,nvir_a * nvir_b)
             t[a:a+k] = np.dot(dataset,t_amp_t).reshape(-1,nvir_b,nocc_a*nocc_b)
             a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir_a,chnk_size):
            Lvv = vvvv_p[0]
            LVV = vvvv_p[1]
            vvvv = dfadc.get_vVvV_df(myadc, Lvv, LVV, p, chnk_size)
            k = vvvv.shape[0]
            vvvv = vvvv.reshape(-1,nvir_a*nvir_b)
            t[a:a+k] = np.dot(vvvv,t_amp_t).reshape(-1,nvir_b,nocc_a*nocc_b)
            del (vvvv)
            a += k
    else :
        raise Exception("Unknown vvvv type") 

    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc_a, nocc_b, nvir_a, nvir_b)

    return t

def density_matrix_so(myadc, T=None):

    if T is None:
        T = UADCIP(myadc).get_trans_moments()

    T_a = T[0]
    T_b = T[1]

    T_a = np.array(T_a)
    T_b = np.array(T_b)

    dm_a = np.dot(T_a, T_a.T)
    dm_b = np.dot(T_b, T_b.T)

    dm = (dm_a, dm_b)

    return dm


class UADC(lib.StreamObject):
    '''Ground state calculations

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).

            >>> mol = gto.M(atom = 'H 0 0 0; F 0 0 1.1', basis = 'ccpvdz')
            >>> mf = scf.RHF(mol).run()
            >>> myadc = adc.UADC(mf).run()

    Saved results

        e_corr : float
            MPn correlation correction
        e_tot : float
            Total energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    incore_complete = getattr(__config__, 'adc_uadc_UADC_incore_complete', False)
    
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        from pyscf import gto
        
        if 'dft' in str(mf.__module__):
            raise NotImplementedError('DFT reference for UADC')
        
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ
         
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory
        
        self.max_space = getattr(__config__, 'adc_uadc_UADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_uadc_UADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_uadc_UADC_conv_tol', 1e-12)

        self.scf_energy = mf.e_tot 
        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway
        
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self._nocc = mf.nelec
        self._nmo = (mo_coeff[0].shape[1], mo_coeff[1].shape[1])
        self._nvir = (self._nmo[0] - self._nocc[0], self._nmo[1] - self._nocc[1])
        self.mo_energy_a = mf.mo_energy[0] 
        self.mo_energy_b = mf.mo_energy[1]
        self.chkfile = mf.chkfile
        self.method = "adc(2)"
        self.method_type = "ip"
        self.with_df = None
        self.compute_mpn_energy = True
        self.fc_bool = True
        self.nfc_orb = 0
        self.ncore_cvs = 0
        self.cvs_npick = []
        self.lanczos_space = 0
        self.imaginary_shift = 0
        self.energy_thresh = 1000
        self.es_mom_thresh = 0.03
        self.cvs_type = 'cce'
        self.oneshot_mom = False
        self.oneshot_cvs = True
        self.cvs_edge = '1s'
        self.oneshot_no_vve = False
        self.max_nmom = 2
 
        keys = set(('conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mol', 'mo_energy_b', 'max_memory', 'scf_energy', 'e_tot', 't1', 'frozen', 'mo_energy_a', 'chkfile', 'max_space', 't2', 'mo_occ', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)
    
    compute_amplitudes = compute_amplitudes
    compute_energy = compute_energy
    transform_integrals = uadc_ao2mo.transform_integrals_incore
    make_rdm1s = density_matrix_so
    
    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self
    
    def dump_flags_gs(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        return self
    
    def kernel_gs(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
        
        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)
    
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()
    
        nmo_a, nmo_b = self._nmo
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmo_a * (nmo_a+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo_a**4) + nmo_pair**2) * 2 * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
           if getattr(self, 'with_df', None): 
               self.with_df = self.with_df
           else :
               self.with_df = self._scf.with_df

           def df_transform():
               return uadc_ao2mo.transform_integrals_df(self)
           self.transform_integrals = df_transform
        elif (self._scf._eri is None or
            (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return uadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals() 

        if self.compute_mpn_energy == True:
            self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose, fc_bool=True)
            self._finalize()
        else:
            self.t1, self.t2 = compute_amplitudes(self, eris=eris, fc_bool=True)
            self.e_corr = None

        return self.e_corr, self.t1, self.t2

    def kernel(self, nroots=1, guess=None, eris=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
    
        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)
    
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        nmo_a, nmo_b = self._nmo
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmo_a * (nmo_a+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo_a**4) + nmo_pair**2) * 2 * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
           if getattr(self, 'with_df', None): 
               self.with_df = self.with_df
           else :
               self.with_df = self._scf.with_df

           def df_transform():
               return uadc_ao2mo.transform_integrals_df(self)
           self.transform_integrals = df_transform
        elif (self._scf._eri is None or
            (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
            def outcore_transform():
                return uadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals() 

        if self.compute_mpn_energy == True:
            self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose, fc_bool=True)
            self._finalize()
        else:
            self.t1, self.t2 = compute_amplitudes(self, eris=eris, fc_bool=True)
            self.e_corr = None

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
        #    e_exc, v_exc, spec_fac = self.ip_adc(nroots=nroots, guess=guess, eris=eris)
            temp = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)

        #return e_exc, v_exc, spec_fac
        return temp

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E_corr = %.8f',
                    self.e_corr)
        return self
    
    def ea_adc(self, nroots=1, guess=None, eris=None):
        return UADCEA(self).kernel(nroots, guess, eris)
    
    def ip_adc(self, nroots=1, guess=None, eris=None):
        return UADCIP(self).kernel(nroots, guess, eris)

    def density_fit(self, auxbasis=None, with_df=None):
        if with_df is None:
            self.with_df = df.DF(self._scf.mol)
            self.with_df.max_memory = self.max_memory
            self.with_df.stdout = self.stdout
            self.with_df.verbose = self.verbose
            if auxbasis is None:
                self.with_df.auxbasis = self._scf.with_df.auxbasis
            else :
                self.with_df.auxbasis = auxbasis
        else :
            self.with_df = with_df
        return self


def get_imds_ea(adc, eris=None):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2_a, t1_2_b = t1[0]
    t2_1_a, t2_1_ab, t2_1_b = t2[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov

    # a-b block
    # Zeroth-order terms

    M_ab_a = lib.einsum('ab,a->ab', idn_vir_a, e_vir_a)
    M_ab_b = lib.einsum('ab,a->ab', idn_vir_b, e_vir_b)

   # Second-order terms

    M_ab_a +=  lib.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a +=  lib.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_a +=  lib.einsum('l,mlad,mlbd->ab',e_occ_b,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_b +=  lib.einsum('l,lmad,lmbd->ab',e_occ_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b +=  lib.einsum('l,mlda,mldb->ab',e_occ_b,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_b +=  lib.einsum('l,lmda,lmdb->ab',e_occ_a,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_a -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_b -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('d,mlda,mldb->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('d,lmda,lmdb->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_a -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('a,mlad,mlbd->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_b -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('a,mlda,mldb->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('a,lmda,lmdb->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_a -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('b,mlad,mlbd->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_b -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('b,mlda,mldb->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('b,lmda,lmdb->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_a -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab_a += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab_a -=        lib.einsum('lmad,lbmd->ab',t2_1_ab, eris_ovOV,optimize=True)

    M_ab_b -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_1_b, eris_OVOV,optimize=True)
    M_ab_b += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_1_b, eris_OVOV,optimize=True)
    M_ab_b -=        lib.einsum('mlda,mdlb->ab',t2_1_ab, eris_ovOV,optimize=True)

    M_ab_a -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_1_a,eris_ovov,optimize=True)
    M_ab_a += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_1_a, eris_ovov,optimize=True)
    M_ab_a -=        lib.einsum('lmbd,lamd->ab',t2_1_ab, eris_ovOV,optimize=True)

    M_ab_b -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_1_b, eris_OVOV,optimize=True)
    M_ab_b += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_1_b, eris_OVOV,optimize=True)
    M_ab_b -=        lib.einsum('mldb,mdla->ab',t2_1_ab, eris_ovOV,optimize=True)

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    #Third-order terms

    if(method =='adc(3)'):

        t2_2_a, t2_2_ab, t2_2_b = t2[1]

        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_OOvv = eris.OOvv
        eris_ooVV = eris.ooVV
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_OVvo = eris.OVvo
        eris_ovVO = eris.ovVO
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO
        
        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_a

        a = 0
        for p in range(0,nocc_a,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            k = eris_ovvv.shape[0]
            M_ab_a +=  lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
            M_ab_a -=  lib.einsum('ld,lbad->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
            M_ab_a += lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
            M_ab_a -= lib.einsum('ld,ladb->ab',t1_2_a[a:a+k], eris_ovvv,optimize=True)
            del eris_ovvv
            a += k

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_b

        a = 0
        for p in range(0,nocc_b,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_OVvv = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
            else :
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            k = eris_OVvv.shape[0]
            M_ab_a +=  lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVvv,optimize=True)
            M_ab_a += lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVvv,optimize=True)
            del eris_OVvv
            a += k

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_b
        a = 0
        for p in range(0,nocc_b,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_OVVV = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
            else :
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            k = eris_OVVV.shape[0]
            M_ab_b +=  lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
            M_ab_b -=  lib.einsum('ld,lbad->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
            M_ab_b -= lib.einsum('ld,ladb->ab',t1_2_b[a:a+k], eris_OVVV,optimize=True)
            del eris_OVVV
            a += k

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_a
        a = 0
        for p in range(0,nocc_a,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovVV = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            k = eris_ovVV.shape[0]
            M_ab_b +=  lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovVV,optimize=True)
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovVV,optimize=True)
            del eris_ovVV
            a += k

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

        M_ab_a -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_2_a, eris_ovov,optimize=True)
        M_ab_a += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_2_a, eris_ovov,optimize=True)
        M_ab_a -=        lib.einsum('lmad,lbmd->ab',t2_2_ab, eris_ovOV,optimize=True)

        M_ab_b -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_2_b, eris_OVOV,optimize=True)
        M_ab_b += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_2_b, eris_OVOV,optimize=True)
        M_ab_b -=        lib.einsum('mlda,mdlb->ab',t2_2_ab, eris_ovOV,optimize=True)

        M_ab_a -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_2_a,eris_ovov,optimize=True)
        M_ab_a += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_2_a, eris_ovov,optimize=True)
        M_ab_a -=        lib.einsum('lmbd,lamd->ab',t2_2_ab, eris_ovOV,optimize=True)

        M_ab_b -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_2_b, eris_OVOV,optimize=True)
        M_ab_b += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_2_b, eris_OVOV,optimize=True)
        M_ab_b -=        lib.einsum('mldb,mdla->ab',t2_2_ab, eris_ovOV,optimize=True)

        M_ab_a += lib.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += lib.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += lib.einsum('l,mlbd,mlad->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += lib.einsum('l,lmbd,lmad->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += lib.einsum('l,mldb,mlda->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += lib.einsum('l,lmdb,lmda->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a += lib.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += lib.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += lib.einsum('l,mlad,mlbd->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += lib.einsum('l,lmad,lmbd->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += lib.einsum('l,mlda,mldb->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += lib.einsum('l,lmda,lmdb->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir_a, t2_1_a ,t2_2_a, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_b -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_b ,t2_2_b, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,mldb,mlda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,lmdb,lmda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_a -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,mlda,mldb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,lmda,lmdb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*lib.einsum('a,mlbd,mlad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*lib.einsum('a,mldb,mlda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*lib.einsum('a,lmdb,lmda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*lib.einsum('a,mlad,mlbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*lib.einsum('a,mlda,mldb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*lib.einsum('a,lmda,lmdb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*lib.einsum('b,mlbd,mlad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*lib.einsum('b,mldb,mlda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*lib.einsum('b,lmdb,lmda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*lib.einsum('b,mlad,mlbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*lib.einsum('b,mlda,mldb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*lib.einsum('b,lmda,lmdb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= lib.einsum('lned,mlbd,nmae->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a += lib.einsum('lned,mlbd,mane->ab',t2_1_a, t2_1_a, eris_ovov, optimize=True)
        M_ab_a += lib.einsum('nled,mlbd,nmae->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a -= lib.einsum('nled,mlbd,mane->ab',t2_1_ab, t2_1_ab, eris_ovov, optimize=True)
        M_ab_a -= lib.einsum('lnde,mlbd,neam->ab',t2_1_ab, t2_1_a, eris_OVvo, optimize=True)
        M_ab_a += lib.einsum('lned,mlbd,neam->ab',t2_1_b, t2_1_ab, eris_OVvo, optimize=True)
        M_ab_a += lib.einsum('lned,lmbd,nmae->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_b -= lib.einsum('lned,mlbd,nmae->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('lned,mlbd,mane->ab',t2_1_b, t2_1_b, eris_OVOV, optimize=True)
        M_ab_b += lib.einsum('lnde,lmdb,nmae->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b -= lib.einsum('lnde,lmdb,mane->ab',t2_1_ab, t2_1_ab, eris_OVOV, optimize=True)
        M_ab_b -= lib.einsum('nled,mlbd,neam->ab',t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
        M_ab_b += lib.einsum('lned,lmdb,neam->ab',t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
        M_ab_b += lib.einsum('nlde,mldb,nmae->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_a -= lib.einsum('mled,lnad,nmeb->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a += lib.einsum('mled,lnad,nbem->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab_a += lib.einsum('mled,nlad,nmeb->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a -= lib.einsum('mled,nlad,nbem->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_a += lib.einsum('lmed,lnad,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)
        M_ab_a -= lib.einsum('mled,nlad,nbem->ab',t2_1_b, t2_1_ab, eris_ovVO, optimize=True)
        M_ab_a += lib.einsum('lmde,lnad,nbem->ab',t2_1_ab, t2_1_a, eris_ovVO, optimize=True)

        M_ab_b -= lib.einsum('mled,lnad,nmeb->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('mled,lnad,nbem->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b += lib.einsum('lmde,lnda,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b -= lib.einsum('lmde,lnda,nbem->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_b += lib.einsum('mlde,nlda,nmeb->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)
        M_ab_b -= lib.einsum('mled,lnda,nbem->ab',t2_1_a, t2_1_ab, eris_OVvo, optimize=True)
        M_ab_b += lib.einsum('mled,lnad,nbem->ab',t2_1_ab, t2_1_b, eris_OVvo, optimize=True)

        M_ab_a -= lib.einsum('mlbd,lnae,nmde->ab',t2_1_a, t2_1_a,   eris_oovv, optimize=True)
        M_ab_a += lib.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_a,   eris_ovvo, optimize=True)
        M_ab_a += lib.einsum('lmbd,lnae,nmde->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_a -= lib.einsum('lmbd,lnae,nedm->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_a += lib.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_ab,  eris_OVvo, optimize=True)
        M_ab_a -= lib.einsum('lmbd,lnae,nedm->ab',t2_1_ab, t2_1_a,  eris_ovVO, optimize=True)
        M_ab_a += lib.einsum('mlbd,nlae,nmde->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_b -= lib.einsum('mlbd,lnae,nmde->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('mlbd,lnae,nedm->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b += lib.einsum('mldb,nlea,nmde->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_b -= lib.einsum('mldb,nlea,nedm->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_b += lib.einsum('mlbd,nlea,nedm->ab',t2_1_b, t2_1_ab,  eris_ovVO, optimize=True)
        M_ab_b -= lib.einsum('mldb,lnae,nedm->ab',t2_1_ab, t2_1_b,  eris_OVvo, optimize=True)
        M_ab_b += lib.einsum('lmdb,lnea,nmed->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_a += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a -= 0.5*lib.einsum('lned,mled,nbam->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab_a -= lib.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a += lib.einsum('nled,mled,nbam->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_a += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOvv, optimize=True)
        M_ab_a -= lib.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_b += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b -= 0.5*lib.einsum('lned,mled,nbam->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b -= lib.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('lned,lmed,nbam->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_b += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_ooVV, optimize=True)
        M_ab_b -= lib.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_a -= 0.25*lib.einsum('mlbd,noad,nmol->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab_a += 0.25*lib.einsum('mlbd,noad,nlom->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab_a -= lib.einsum('mlbd,noad,nmol->ab',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)

        M_ab_b -= 0.25*lib.einsum('mlbd,noad,nmol->ab',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
        M_ab_b += 0.25*lib.einsum('mlbd,noad,nlom->ab',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
        M_ab_b -= lib.einsum('lmdb,onda,olnm->ab',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)

        log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

        if isinstance(eris.vvvv_p,np.ndarray):
            eris_vvvv = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)
            M_ab_a -= 0.25*lib.einsum('mlef,mlbd,adef->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
            M_ab_a -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
            M_ab_a += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
            temp_a = t2_1_a.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
            eris_vvvv = eris_vvvv.reshape(nvir_a*nvir_a,nvir_a*nvir_a)
            temp = np.dot(temp_a,eris_vvvv)
            del eris_vvvv
            temp = temp.reshape(nocc_a,nocc_a,nvir_a,nvir_a)
            M_ab_a -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_a, temp, optimize=True)

        else :
            if isinstance(eris.vvvv_p, list) :
                t2a_vvvv = contract_ladder_antisym(adc,t2_1_a,eris.vvvv_p)
            else :
                t2a_vvvv = contract_ladder_antisym(adc,t2_1_a,eris.Lvv)

            temp_t2a_vvvv = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))   
            temp_t2a_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = t2a_vvvv    
            temp_t2a_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -t2a_vvvv 
            del (t2a_vvvv)

            M_ab_a -= 2*0.25*lib.einsum('mlad,mlbd->ab',  temp_t2a_vvvv, t2_1_a, optimize=True)
            M_ab_a -= 2*0.25*lib.einsum('mlaf,mlbf->ab', t2_1_a, temp_t2a_vvvv, optimize=True)
            del (temp_t2a_vvvv)
            

        if isinstance(eris.VVVV_p,np.ndarray):
            eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
            M_ab_b -= 0.25*lib.einsum('mlef,mlbd,adef->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
            M_ab_b -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
            M_ab_b += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
            temp_b = t2_1_b.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
            eris_VVVV = eris_VVVV.reshape(nvir_b*nvir_b,nvir_b*nvir_b)
            temp = np.dot(temp_b,eris_VVVV)
            del eris_VVVV
            temp = temp.reshape(nocc_b,nocc_b,nvir_b,nvir_b)
            M_ab_b -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_b, temp, optimize=True)

        else:
            if isinstance(eris.VVVV_p, list):
                t2b_VVVV = contract_ladder_antisym(adc,t2_1_b,eris.VVVV_p)
            else:
                t2b_VVVV = contract_ladder_antisym(adc,t2_1_b,eris.LVV)

            temp_t2b_VVVV = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))   
            temp_t2b_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = t2b_VVVV 
            temp_t2b_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -t2b_VVVV 
            del (t2b_VVVV)        

            M_ab_b -= 2 * 0.25*lib.einsum('mlad,mlbd->ab',  temp_t2b_VVVV, t2_1_b, optimize=True)
            M_ab_b -= 2 * 0.25*lib.einsum('mlaf,mlbf->ab', t2_1_b, temp_t2b_VVVV, optimize=True)
            del (temp_t2b_VVVV)

        if isinstance(eris.vVvV_p,np.ndarray):

            eris_vVvV = eris.vVvV_p
            eris_vVvV = eris_vVvV.reshape(nvir_a,nvir_b,nvir_a,nvir_b)
            M_ab_a -= lib.einsum('mlef,mlbd,adef->ab',t2_1_ab, t2_1_ab,   eris_vVvV, optimize=True)
            M_ab_a -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_vVvV, optimize=True)
            M_ab_a += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab,   eris_vVvV, optimize=True)

            M_ab_b -= lib.einsum('mlef,mldb,daef->ab',t2_1_ab, t2_1_ab,   eris_vVvV, optimize=True)
            M_ab_b -= 0.5*lib.einsum('mldf,mled,eafb->ab',t2_1_a, t2_1_a, eris_vVvV, optimize=True)
            M_ab_b += lib.einsum('mlfd,mled,eafb->ab',t2_1_ab, t2_1_ab,   eris_vVvV, optimize=True)

            eris_vVvV = eris_vVvV.reshape(nvir_a*nvir_b,nvir_a*nvir_b)
            temp_ab = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
            temp = np.dot(temp_ab,eris_vVvV)
            temp = temp.reshape(nocc_a,nocc_b,nvir_a,nvir_b)
            M_ab_a -= lib.einsum('mlaf,mlbf->ab',t2_1_ab, temp, optimize=True)
            M_ab_b -= lib.einsum('mlfa,mlfb->ab',t2_1_ab, temp, optimize=True)

        else: 
           if isinstance(eris.vVvV_p, list):
               t2_vVvV = contract_ladder(adc,t2_1_ab,eris.vVvV_p)
           else:
               t2_vVvV = contract_ladder(adc,t2_1_ab,(eris.Lvv,eris.LVV))

           M_ab_a -= lib.einsum('mlad,mlbd->ab', t2_vVvV, t2_1_ab, optimize=True)
           M_ab_b -= lib.einsum('mlda,mldb->ab', t2_vVvV, t2_1_ab, optimize=True)
           M_ab_a -= lib.einsum('mlaf,mlbf->ab',t2_1_ab, t2_vVvV, optimize=True)
           M_ab_b -= lib.einsum('mlfa,mlfb->ab',t2_1_ab, t2_vVvV, optimize=True)
           del (t2_vVvV)

        if isinstance(eris.vvvv_p, list):

            a = 0
            temp = np.zeros((nvir_a,nvir_a))
            for dataset in eris.vvvv_p:
                k = dataset.shape[0]
                vvvv = dataset[:]
                eris_vvvv = np.zeros((k,nvir_a,nvir_a,nvir_a))   
                eris_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv    
                eris_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv 
                
                temp[a:a+k]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a,  eris_vvvv, optimize=True)
                temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
                del eris_vvvv
                a += k
            M_ab_a  += temp            

            a = 0
            temp = np.zeros((nvir_b,nvir_b))
            for dataset in eris.VVVV_p:
                k = dataset.shape[0]
                VVVV = dataset[:]
                eris_VVVV = np.zeros((k,nvir_b,nvir_b,nvir_b))   
                eris_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV   
                eris_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV 
                
                temp[a:a+k]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b,  eris_VVVV, optimize=True)
                temp[a:a+k]  += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
                del eris_VVVV
                a += k
            M_ab_b  += temp            

            a = 0
            temp = np.zeros((nvir_a,nvir_a))
            for dataset in eris.vVvV_p:
                k = dataset.shape[0]
                eris_vVvV = dataset[:].reshape(-1,nvir_b,nvir_a,nvir_b)
                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_vVvV, optimize=True)
                temp[a:a+k] += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_vVvV, optimize=True)
                a += k
            M_ab_a  += temp    
            a = 0

            temp = np.zeros((nvir_b,nvir_b))
            for dataset in eris.VvVv_p:
                k = dataset.shape[0]
                eris_VvVv = dataset[:].reshape(-1,nvir_a,nvir_b,nvir_a)
                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_VvVv, optimize=True)
                temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_VvVv, optimize=True)
                a += k
            M_ab_b  += temp    

        elif isinstance(eris.vvvv_p, type(None)):

            a = 0
            temp = np.zeros((nvir_a,nvir_a))
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for p in range(0,nvir_a,chnk_size):
                vvvv = dfadc.get_vvvv_antisym_df(adc, eris.Lvv, p, chnk_size) 
                k = vvvv.shape[0]

                eris_vvvv = np.zeros((k,nvir_a,nvir_a,nvir_a))   
                eris_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = vvvv    
                eris_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -vvvv 
                
                temp[a:a+k]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a,  eris_vvvv, optimize=True)
                temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
                del eris_vvvv
                a += k
            M_ab_a  += temp
            del (temp)            

            a = 0
            temp = np.zeros((nvir_b,nvir_b))
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for p in range(0,nvir_b,chnk_size):
                VVVV = dfadc.get_vvvv_antisym_df(adc, eris.LVV, p, chnk_size) 
                k = VVVV.shape[0]

                eris_VVVV = np.zeros((k,nvir_b,nvir_b,nvir_b))   
                eris_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = VVVV   
                eris_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -VVVV 
                
                temp[a:a+k]  -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b,  eris_VVVV, optimize=True)
                temp[a:a+k]  += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
                del eris_VVVV
                a += k
            M_ab_b  += temp            
            del (temp)            

            a = 0
            temp = np.zeros((nvir_a,nvir_a))
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
            for p in range(0,nvir_a,chnk_size):
                eris_vVvV = dfadc.get_vVvV_df(adc, eris.Lvv, eris.LVV, p, chnk_size) 
                k = eris_vVvV.shape[0]
            
                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_vVvV, optimize=True)
                temp[a:a+k] += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_vVvV, optimize=True)
                a += k
            M_ab_a  += temp    
            del (temp)            

            a = 0
            temp = np.zeros((nvir_b,nvir_b))
            for p in range(0,nvir_b,chnk_size):
                eris_VvVv = dfadc.get_vVvV_df(adc, eris.LVV, eris.Lvv, p, chnk_size) 
                k = eris_VvVv.shape[0]

                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_VvVv, optimize=True)
                temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_VvVv, optimize=True)
                a += k
            M_ab_b  += temp    
            del (temp)            

    M_ab = (M_ab_a, M_ab_b)
    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
    return M_ab

def get_imds_ip(adc, eris=None):

    #cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_2_a, t1_2_b = t1[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO
    eris_OVvo = eris.OVvo

    # i-j block
    # Zeroth-order terms

    M_ij_a = lib.einsum('ij,j->ij', idn_occ_a ,e_occ_a)
    M_ij_b = lib.einsum('ij,j->ij', idn_occ_b ,e_occ_b)

    # Second-order terms

    t2_1_a = t2[0][0][:]
    M_ij_a += 0.5 * 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a -= 0.5 * 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a += 0.5 * 0.5 *  lib.einsum('jlde,idel->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a -= 0.5 * 0.5 *  lib.einsum('jlde,ldei->ij',t2_1_a, eris_ovvo, optimize=True)

    t2_1_b = t2[0][2][:]
    M_ij_b += 0.5 * 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b -= 0.5 * 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b += 0.5 * 0.5 *  lib.einsum('jlde,idel->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b -= 0.5 * 0.5 *  lib.einsum('jlde,ldei->ij',t2_1_b, eris_OVVO, optimize=True)

    t2_1_ab = t2[0][1][:]
    M_ij_a += 0.5 * lib.einsum('ilde,jdel->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_b += 0.5 * lib.einsum('lied,ledj->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_a += 0.5 * lib.einsum('jlde,idel->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_b += 0.5 * lib.einsum('ljed,ledi->ij',t2_1_ab, eris_ovVO, optimize=True)
    del t2_1_ab

    # Third-order terms

    if (method == "adc(3)"):

        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_ooVV = eris.ooVV
        eris_OOvv = eris.OOvv
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_ovVO = eris.ovVO
        eris_OVvo = eris.OVvo
        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_ovOO = eris.ovOO
        eris_OVoo = eris.OVoo
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a -= lib.einsum('ld,jdli->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_b, eris_OVoo, optimize=True)

        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b -= lib.einsum('ld,jdli->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_a, eris_ovOO, optimize=True)

        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a -= lib.einsum('ld,idlj->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_b, eris_OVoo, optimize=True)

        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b -= lib.einsum('ld,idlj->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_a, eris_ovOO, optimize=True)

        t2_1_a = t2[0][0][:]
        t2_2_a = t2[1][0][:]

        M_ij_a += 0.5 * 0.5* lib.einsum('ilde,jdel->ij',t2_2_a, eris_ovvo, optimize=True)
        M_ij_a -= 0.5 * 0.5* lib.einsum('ilde,jedl->ij',t2_2_a, eris_ovvo, optimize=True)

        M_ij_a += 0.5 * 0.5* lib.einsum('jlde,ledi->ij',t2_2_a, eris_ovvo, optimize=True)
        M_ij_a -= 0.5 * 0.5* lib.einsum('jlde,iedl->ij',t2_2_a, eris_ovvo, optimize=True)

        t2_1_b = t2[0][2][:]
        t2_2_b = t2[1][2][:]

        M_ij_b += 0.5 * 0.5* lib.einsum('ilde,jdel->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b -= 0.5 * 0.5* lib.einsum('ilde,jedl->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b += 0.5 * 0.5* lib.einsum('jlde,ledi->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b -= 0.5 * 0.5* lib.einsum('jlde,iedl->ij',t2_2_b, eris_OVVO, optimize=True)

        t2_1_ab = t2[0][1][:]
        t2_2_ab = t2[1][1][:]

        M_ij_a += 0.5 * lib.einsum('ilde,jdel->ij',t2_2_ab, eris_ovVO, optimize=True)
        M_ij_b += 0.5 * lib.einsum('lied,ledj->ij',t2_2_ab, eris_ovVO, optimize=True)
        M_ij_a += 0.5 * lib.einsum('jlde,ledi->ij',t2_2_ab, eris_OVvo, optimize=True)
        M_ij_b += 0.5 * lib.einsum('ljed,ledi->ij',t2_2_ab, eris_ovVO, optimize=True)

        t2_1_a = t2[0][0][:]
        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_a, t2_1_a, optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mejf,mefi->ij',M_ij_t, eris_ovvo, optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mejf,mefi->ji',M_ij_t, eris_ovvo, optimize=True)
        M_ij_a += 0.5 * lib.einsum('mejf,mife->ij',M_ij_t, eris_oovv, optimize=True)
        M_ij_a += 0.5 * lib.einsum('mejf,mife->ji',M_ij_t, eris_oovv, optimize=True)
        del M_ij_t

        M_ij_a += 0.5 * 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.5 * 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)

        M_ij_a += 0.5 *0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.5 *0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_OOvv , optimize = True)

        M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooOO, optimize = True)

        t2_1_ab = t2[0][1][:]
        M_ij_a -= 0.5 * lib.einsum('lmde,jldf,mefi->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_b += 0.5 * lib.einsum('lmde,ljdf,mefi->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('lmde,ildf,mefj->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_b += 0.5 * lib.einsum('lmde,lidf,mefj->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)
        del t2_1_a

        t2_1_b = t2[0][2][:]
        M_ij_a += 0.5 * lib.einsum('lmde,jlfd,mefi->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('mled,jldf,mefi->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_a += 0.5 * lib.einsum('lmde,ilfd,mefj->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('mled,ildf,mefj->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_b, t2_1_b, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mejf,mefi->ij',M_ij_t, eris_OVVO, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mejf,mefi->ji',M_ij_t, eris_OVVO, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mejf,mife->ij',M_ij_t, eris_OOVV, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mejf,mife->ji',M_ij_t, eris_OOVV, optimize=True)
        del M_ij_t

        M_ij_b += 0.5 * 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.5 * 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)

        M_ij_b += 0.5 * 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.5 * 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ooVV , optimize = True)
        M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_OOVV , optimize = True)
        M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_b, t2_1_b, eris_OVVO , optimize = True)

        M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ooOO, optimize = True)
        M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        del t2_1_b

        t2_1_ab = t2[0][1][:]

        M_ij_a += 0.5 * lib.einsum('mled,jlfd,mefi->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mled,jlfd,mife->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mlde,jldf,mife->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)

        M_ij_b += 0.5 * lib.einsum('lmde,ljdf,mefi->ij',t2_1_ab, t2_1_ab, eris_OVVO,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmde,ljdf,mife->ij',t2_1_ab, t2_1_ab, eris_OOVV,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmed,ljfd,mife->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)

        M_ij_a += 0.5 * lib.einsum('mled,ilfd,mefj->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mled,ilfd,mjfe->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mlde,ildf,mjfe->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)

        M_ij_b += 0.5 * lib.einsum('lmde,lidf,mefj->ij',t2_1_ab, t2_1_ab, eris_OVVO ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmde,lidf,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOVV ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmed,lifd,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)

        M_ij_a += 0.5 * lib.einsum('lmde,jnde,limn->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)
        M_ij_b += 0.5 * lib.einsum('mled,njed,mnli->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)

        M_ij_a +=0.5 * lib.einsum('inde,lmde,jlnm->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b +=0.5 * lib.einsum('nied,mled,nmjl->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_oovv , optimize = True)
        M_ij_a -=lib.einsum('mlfd,mled,jfei->ij',t2_1_ab, t2_1_ab, eris_ovvo , optimize = True)
        M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ooVV , optimize = True)
        M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_OOVV , optimize = True)
        M_ij_b -=lib.einsum('lmdf,lmde,jfei->ij',t2_1_ab, t2_1_ab, eris_OVVO , optimize = True)
        M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_OOvv , optimize = True)

        M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a += lib.einsum('nled,mled,jmni->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b += lib.einsum('lnde,lmde,jmni->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)

        del t2_1_ab

    M_ij = (M_ij_a, M_ij_b)
    #cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

    return M_ij

def get_imds_ip_off(adc, eris=None, fc_bool=True):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    #t1 = adc.t1
    #t2 = adc.t2
    t1, t2 = adc.compute_amplitudes(eris, fc_bool)   

    t1_2_a, t1_2_b = t1[0]
    t2_1_a, t2_1_ab, t2_1_b = t2[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    imaginary_shift = adc.imaginary_shift
    energy_thresh = adc.energy_thresh

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]
    #e_vir_a = complex_shift(e_vir_a, energy_thresh, imaginary_shift)
    #e_vir_b = complex_shift(e_vir_b, energy_thresh, imaginary_shift)

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov

    # i-j block
    # Zeroth-order terms

    M_ij_a = lib.einsum('ij,j->ij', idn_occ_a ,e_occ_a)#.astype(complex)
    M_ij_b = lib.einsum('ij,j->ij', idn_occ_b ,e_occ_b)#.astype(complex)

    # Second-order terms

    M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_1_a, optimize=True)
    M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a +=  lib.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_b +=  lib.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_1_b, optimize=True)
    M_ij_b +=  lib.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b +=  lib.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_a -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a, optimize=True)
    M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_b -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b, optimize=True)
    M_ij_b -= 0.5*lib.einsum('l,lide,ljde->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_a -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a, optimize=True)
    M_ij_a -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_b -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b, optimize=True)
    M_ij_b -= 0.25 *  lib.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b -= 0.25 *  lib.einsum('i,lide,ljde->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_a -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a, optimize=True)
    M_ij_a -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_b -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b, optimize=True)
    M_ij_b -= 0.25 *  lib.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b -= 0.25 *  lib.einsum('j,lide,ljde->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_a += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_a, eris_ovov, optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_a, eris_ovov, optimize=True)
    M_ij_a += lib.einsum('ilde,jdle->ij',t2_1_ab, eris_ovOV, optimize=True)

    M_ij_b += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_b, eris_OVOV, optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_b, eris_OVOV, optimize=True)
    M_ij_b += lib.einsum('lied,lejd->ij',t2_1_ab, eris_ovOV, optimize=True)

    M_ij_a += 0.5 *  lib.einsum('jlde,idle->ij',t2_1_a, eris_ovov, optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1_a, eris_ovov, optimize=True)
    M_ij_a += lib.einsum('jlde,idle->ij',t2_1_ab, eris_ovOV, optimize=True)

    M_ij_b += 0.5 *  lib.einsum('jlde,idle->ij',t2_1_b, eris_OVOV, optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1_b, eris_OVOV, optimize=True)
    M_ij_b += lib.einsum('ljed,leid->ij',t2_1_ab, eris_ovOV, optimize=True)

    #e_a, _ = np.linalg.eigh(M_ij_a[:2,:2])
    #e_b, _ = np.linalg.eigh(M_ij_b[:2,:2])
    #print("e_a: ", e_a)
    #print("e_b: ", e_b)
    #exit()

    # Third-order terms

    if (method == "adc(3)"):

        t2_2_a, t2_2_ab, t2_2_b = t2[1]
        eris_oovv = eris.oovv
        eris_OOVV = eris.OOVV
        eris_ooVV = eris.ooVV
        eris_OOvv = eris.OOvv
        eris_ovvo = eris.ovvo
        eris_OVVO = eris.OVVO
        eris_ovVO = eris.ovVO
        eris_OVvo = eris.OVvo
        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_ovOO = eris.ovOO
        eris_OVoo = eris.OVoo
        eris_oooo = eris.oooo
        eris_OOOO = eris.OOOO
        eris_ooOO = eris.ooOO

        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a -= lib.einsum('ld,jdli->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_b, eris_OVoo, optimize=True)

        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b -= lib.einsum('ld,jdli->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_a, eris_ovOO, optimize=True)

        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a -= lib.einsum('ld,idlj->ij',t1_2_a, eris_ovoo, optimize=True)
        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_b, eris_OVoo, optimize=True)

        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b -= lib.einsum('ld,idlj->ij',t1_2_b, eris_OVOO, optimize=True)
        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_a, eris_ovOO, optimize=True)

        M_ij_a += 0.5* lib.einsum('ilde,jdle->ij',t2_2_a, eris_ovov, optimize=True)
        M_ij_a -= 0.5* lib.einsum('ilde,jeld->ij',t2_2_a, eris_ovov, optimize=True)
        M_ij_a += lib.einsum('ilde,jdle->ij',t2_2_ab, eris_ovOV, optimize=True)

        M_ij_b += 0.5* lib.einsum('ilde,jdle->ij',t2_2_b, eris_OVOV, optimize=True)
        M_ij_b -= 0.5* lib.einsum('ilde,jeld->ij',t2_2_b, eris_OVOV, optimize=True)
        M_ij_b += lib.einsum('lied,lejd->ij',t2_2_ab, eris_ovOV, optimize=True)

        M_ij_a += 0.5* lib.einsum('jlde,leid->ij',t2_2_a, eris_ovov, optimize=True)
        M_ij_a -= 0.5* lib.einsum('jlde,ield->ij',t2_2_a, eris_ovov, optimize=True)
        M_ij_a += lib.einsum('jlde,leid->ij',t2_2_ab, eris_OVov, optimize=True)

        M_ij_b += 0.5* lib.einsum('jlde,leid->ij',t2_2_b, eris_OVOV, optimize=True)
        M_ij_b -= 0.5* lib.einsum('jlde,ield->ij',t2_2_b, eris_OVOV, optimize=True)
        M_ij_b += lib.einsum('ljed,leid->ij',t2_2_ab, eris_ovOV, optimize=True)

        M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  lib.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  lib.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  lib.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  lib.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a +=  lib.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  lib.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  lib.einsum('d,jled,iled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  lib.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  lib.einsum('d,ljde,lide->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  lib.einsum('d,ljed,lied->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('i,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('i,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('j,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('j,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= lib.einsum('lmde,jldf,mefi->ij',t2_1_a, t2_1_a, eris_ovvo,optimize = True)
        M_ij_a += lib.einsum('lmde,jldf,mife->ij',t2_1_a, t2_1_a, eris_oovv,optimize = True)
        M_ij_a += lib.einsum('mled,jlfd,mefi->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= lib.einsum('mled,jlfd,mife->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= lib.einsum('lmde,jldf,mefi->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_a -= lib.einsum('mlde,jldf,mife->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)
        M_ij_a += lib.einsum('lmde,jlfd,mefi->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)

        M_ij_b -= lib.einsum('lmde,jldf,mefi->ij',t2_1_b, t2_1_b, eris_OVVO,optimize = True)
        M_ij_b += lib.einsum('lmde,jldf,mife->ij',t2_1_b, t2_1_b, eris_OOVV,optimize = True)
        M_ij_b += lib.einsum('lmde,ljdf,mefi->ij',t2_1_ab, t2_1_ab, eris_OVVO,optimize = True)
        M_ij_b -= lib.einsum('lmde,ljdf,mife->ij',t2_1_ab, t2_1_ab, eris_OOVV,optimize = True)
        M_ij_b -= lib.einsum('mled,jldf,mefi->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_b -= lib.einsum('lmed,ljfd,mife->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)
        M_ij_b += lib.einsum('lmde,ljdf,mefi->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)

        M_ij_a -= lib.einsum('lmde,ildf,mefj->ij',t2_1_a, t2_1_a, eris_ovvo ,optimize = True)
        M_ij_a += lib.einsum('lmde,ildf,mjfe->ij',t2_1_a, t2_1_a, eris_oovv ,optimize = True)
        M_ij_a += lib.einsum('mled,ilfd,mefj->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= lib.einsum('mled,ilfd,mjfe->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= lib.einsum('lmde,ildf,mefj->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_a -= lib.einsum('mlde,ildf,mjfe->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)
        M_ij_a += lib.einsum('lmde,ilfd,mefj->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)

        M_ij_b -= lib.einsum('lmde,ildf,mefj->ij',t2_1_b, t2_1_b, eris_OVVO ,optimize = True)
        M_ij_b += lib.einsum('lmde,ildf,mjfe->ij',t2_1_b, t2_1_b, eris_OOVV ,optimize = True)
        M_ij_b += lib.einsum('lmde,lidf,mefj->ij',t2_1_ab, t2_1_ab, eris_OVVO ,optimize = True)
        M_ij_b -= lib.einsum('lmde,lidf,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOVV ,optimize = True)
        M_ij_b -= lib.einsum('mled,ildf,mefj->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_b -= lib.einsum('lmed,lifd,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)
        M_ij_b += lib.einsum('lmde,lidf,mefj->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)

        M_ij_a += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a += lib.einsum('lmde,jnde,limn->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b += lib.einsum('mled,njed,mnli->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)


        if isinstance(eris.vvvv_p,np.ndarray):
            eris_vvvv = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)
            M_ij_a += 0.25 * lib.einsum('ilde,jlgf,gfde->ij',t2_1_a, t2_1_a, eris_vvvv, optimize = True)
            del eris_vvvv

        else:
            if isinstance(eris.vvvv_p,list):
                t2a_vvvv = contract_ladder_antisym(adc,t2_1_a,eris.vvvv_p)
            else :
                t2a_vvvv = contract_ladder_antisym(adc,t2_1_a,eris.Lvv)

            temp_t2a_vvvv = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))   
            temp_t2a_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = t2a_vvvv    
            temp_t2a_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -t2a_vvvv 

            M_ij_a += 2*0.25 * lib.einsum('ilgf,jlgf->ij', temp_t2a_vvvv, t2_1_a, optimize = True)
        
        if isinstance(eris.VVVV_p,np.ndarray):
            eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
            M_ij_b += 0.25 * lib.einsum('ilde,jlgf,gfde->ij',t2_1_b, t2_1_b, eris_VVVV, optimize = True)
            del eris_VVVV

        else:
            if isinstance(eris.VVVV_p,list) :
                t2b_VVVV = contract_ladder_antisym(adc,t2_1_b,eris.VVVV_p)
            else :
                t2b_VVVV = contract_ladder_antisym(adc,t2_1_b,eris.LVV)

            temp_t2b_VVVV = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))   
            temp_t2b_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = t2b_VVVV 
            temp_t2b_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -t2b_VVVV 

            M_ij_b += 2*0.25 * lib.einsum('ilgf,jlgf->ij', temp_t2b_VVVV, t2_1_b, optimize = True)

        if isinstance(eris.vVvV_p,np.ndarray):
            eris_vVvV = eris.vVvV_p
            eris_vVvV = eris_vVvV.reshape(nvir_a,nvir_b,nvir_a,nvir_b)
            M_ij_a +=lib.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,eris_vVvV, optimize = True)
            temp = lib.einsum('ljfg,fged->ljed',t2_1_ab,eris_vVvV, optimize = True)
            M_ij_b +=lib.einsum('lied,ljed->ij',t2_1_ab, temp, optimize = True)
            eris_vVvV = eris_vVvV.reshape(nvir_a*nvir_b,nvir_a*nvir_b)

        else:
            if isinstance(eris.vVvV_p,list):
                t2_vVvV = contract_ladder(adc,t2_1_ab,eris.vVvV_p)
            else:
                t2_vVvV = contract_ladder(adc,t2_1_ab,(eris.Lvv,eris.LVV))

            M_ij_a +=lib.einsum('ilgf,jlgf->ij', t2_vVvV, t2_1_ab, optimize = True)
            M_ij_b +=lib.einsum('lied,ljed->ij',t2_1_ab, t2_vVvV, optimize = True)

        M_ij_a += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a +=lib.einsum('inde,lmde,jlnm->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b +=lib.einsum('nied,mled,nmjl->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_oovv , optimize = True)
        M_ij_a -=lib.einsum('mlfd,mled,jfei->ij',t2_1_ab, t2_1_ab, eris_ovvo , optimize = True)
        M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ooVV , optimize = True)
        M_ij_a +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ooVV , optimize = True)

        M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_OOVV , optimize = True)
        M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_b, t2_1_b, eris_OVVO , optimize = True)
        M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_OOVV , optimize = True)
        M_ij_b -=lib.einsum('lmdf,lmde,jfei->ij',t2_1_ab, t2_1_ab, eris_OVVO , optimize = True)
        M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_OOvv , optimize = True)
        M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_OOvv , optimize = True)

        M_ij_a -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_ab, eris_ovVO, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_a, eris_OVvo, optimize = True)
        M_ij_a -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOVV, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_ab, eris_OVVO, optimize = True)
        M_ij_a -= lib.einsum('iled,jmfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOvv, optimize = True)

        M_ij_b -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_b, t2_1_b, eris_OOVV, optimize = True)
        M_ij_b += lib.einsum('ilde,jmdf,lefm->ij',t2_1_b, t2_1_b, eris_OVVO, optimize = True)
        M_ij_b += lib.einsum('ilde,mjfd,lefm->ij',t2_1_b, t2_1_ab, eris_OVvo, optimize = True)
        M_ij_b += lib.einsum('lied,jmdf,lefm->ij',t2_1_ab, t2_1_b, eris_ovVO, optimize = True)
        M_ij_b -= lib.einsum('lied,mjfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_oovv, optimize = True)
        M_ij_b += lib.einsum('lied,mjfd,lefm->ij',t2_1_ab, t2_1_ab, eris_ovvo, optimize = True)
        M_ij_b -= lib.einsum('lide,mjdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_ooVV, optimize = True)

        M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a += lib.einsum('nled,mled,jmni->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ooOO, optimize = True)

        M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b += lib.einsum('lnde,lmde,jmni->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooOO, optimize = True)
    """
    E_a, _ = np.linalg.eigh(M_ij_a[:5,:5])
    E_b, _ = np.linalg.eigh(M_ij_b[:5,:5])
    print("E_a: ", E_a)
    print("E_b: ", E_b)
    exit()
    """
    
    M_ij = (M_ij_a, M_ij_b)
    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

    return M_ij


def ea_adc_diag(adc,M_ab=None,eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ab is None:
        M_ab = adc.get_imds()

    M_ab_a, M_ab_b = M_ab[0], M_ab[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    d_i_a = e_occ_a[:,None]
    d_ab_a = e_vir_a[:,None] + e_vir_a
    D_n_a = -d_i_a + d_ab_a.reshape(-1)
    D_n_a = D_n_a.reshape((nocc_a,nvir_a,nvir_a))
    D_iab_a = D_n_a.copy()[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

    d_i_b = e_occ_b[:,None]
    d_ab_b = e_vir_b[:,None] + e_vir_b
    D_n_b = -d_i_b + d_ab_b.reshape(-1)
    D_n_b = D_n_b.reshape((nocc_b,nvir_b,nvir_b))
    D_iab_b = D_n_b.copy()[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

    d_ab_ab = e_vir_a[:,None] + e_vir_b
    d_i_b = e_occ_b[:,None]
    D_n_bab = -d_i_b + d_ab_ab.reshape(-1)
    D_iab_bab = D_n_bab.reshape(-1)

    d_ab_ab = e_vir_b[:,None] + e_vir_a
    d_i_a = e_occ_a[:,None]
    D_n_aba = -d_i_a + d_ab_ab.reshape(-1)
    D_iab_aba = D_n_aba.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in p1-p1 block

    M_ab_a_diag = np.diagonal(M_ab_a)
    M_ab_b_diag = np.diagonal(M_ab_b)

    diag[s_a:f_a] = M_ab_a_diag.copy()
    diag[s_b:f_b] = M_ab_b_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s_aaa:f_aaa] = D_iab_a
    diag[s_bab:f_bab] = D_iab_bab
    diag[s_aba:f_aba] = D_iab_aba
    diag[s_bbb:f_bbb] = D_iab_b

#    ###### Additional terms for the preconditioner ####
    if (method == "adc(2)-x" or method == "adc(3)"):

        if eris is None:
            eris = adc.transform_integrals()

            if isinstance(eris.vvvv, np.ndarray): 

                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo
                eris_OOVV = eris.OOVV
                eris_OVVO = eris.OVVO
                eris_OOvv = eris.OOvv
                eris_ooVV = eris.ooVV

                eris_vvvv = eris.vvvv_p
                temp = np.zeros((nocc_a, eris_vvvv.shape[0]))
                temp[:] += np.diag(eris_vvvv)
                diag[s_aaa:f_aaa] += temp.reshape(-1)

                eris_VVVV = eris.VVVV_p
                temp = np.zeros((nocc_b, eris_VVVV.shape[0]))
                temp[:] += np.diag(eris_VVVV)
                diag[s_bbb:f_bbb] += temp.reshape(-1)

                eris_vVvV = eris.vVvV_p
                temp = np.zeros((nocc_b, eris_vVvV.shape[0]))
                temp[:] += np.diag(eris_vVvV)
                diag[s_bab:f_bab] += temp.reshape(-1)
                
                temp = np.zeros((nocc_a, nvir_a, nvir_b))
                temp[:] += np.diag(eris_vVvV).reshape(nvir_a,nvir_b)
                temp = np.ascontiguousarray(temp.transpose(0,2,1))
                diag[s_aba:f_aba] += temp.reshape(-1)
                    
                eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
                eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
                eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
  
                temp = np.zeros((eris_ovov_p.shape[0],nvir_a))
                temp.T[:] += np.diagonal(eris_ovov_p)
                temp = temp.reshape(nocc_a, nvir_a, nvir_a)
                diag[s_aaa:f_aaa] += -temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

                temp = np.ascontiguousarray(temp.transpose(0,2,1))
                diag[s_aaa:f_aaa] += -temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

                eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
                eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
                eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
  
                temp = np.zeros((eris_OVOV_p.shape[0],nvir_b))
                temp.T[:] += np.diagonal(eris_OVOV_p)
                temp = temp.reshape(nocc_b, nvir_b, nvir_b)
                diag[s_bbb:f_bbb] += -temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

                temp = np.ascontiguousarray(temp.transpose(0,2,1))
                diag[s_bbb:f_bbb] += -temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

                temp = np.zeros((nvir_a, nocc_b, nvir_b))
                temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
                temp = np.ascontiguousarray(temp.transpose(1,0,2))
                diag[s_bab:f_bab] += -temp.reshape(-1)

                temp = np.zeros((nvir_b, nocc_a, nvir_a))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
                temp = np.ascontiguousarray(temp.transpose(1,0,2))
                diag[s_aba:f_aba] += -temp.reshape(-1)

                eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
                eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)
  
                temp = np.zeros((nvir_b, nocc_b, nvir_a))
                temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
                temp = np.ascontiguousarray(temp.transpose(1,2,0))
                diag[s_bab:f_bab] += -temp.reshape(-1)

                eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
                eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)

                temp = np.zeros((nvir_a, nocc_a, nvir_b))
                temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
                temp = np.ascontiguousarray(temp.transpose(1,2,0))
                diag[s_aba:f_aba] += -temp.reshape(-1)

    return diag


def ip_adc_diag(adc,M_ij=None,eris=None,cvs=False, fc_bool=True):
   
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    if M_ij is None:
        M_ij = adc.get_imds(fc_bool)

    M_ij_a, M_ij_b = M_ij[0], M_ij[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    imaginary_shift = adc.imaginary_shift
    energy_thresh = adc.energy_thresh
    ncore_cvs = adc.ncore_cvs

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]#.astype(complex)
    e_occ_b = adc.mo_energy_b[:nocc_b]#.astype(complex)
    e_vir_a = adc.mo_energy_a[nocc_a:]#.astype(complex)
    e_vir_b = adc.mo_energy_b[nocc_b:]#.astype(complex)
    #if cvs is False:
    #e_vir_a = complex_shift(e_vir_a, energy_thresh, imaginary_shift)
    #e_vir_b = complex_shift(e_vir_b, energy_thresh, imaginary_shift)
    idx_a_thresh = np.argwhere(e_vir_a >= energy_thresh)  
    idx_b_thresh = np.argwhere(e_vir_b >= energy_thresh)  

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    #d_ij_a = e_occ_a[:,None] + e_occ_a
    #d_a_a = e_vir_a[:,None]
    #D_n_a = -d_a_a + d_ij_a.reshape(-1)
    #D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    #D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    #d_ij_b = e_occ_b[:,None] + e_occ_b
    #d_a_b = e_vir_b[:,None]
    #D_n_b = -d_a_b + d_ij_b.reshape(-1)
    #D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    #D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    #d_ij_ab = e_occ_b[:,None] + e_occ_a
    #d_a_b = e_vir_b[:,None]
    #D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    #D_aij_bab = D_n_bab.reshape(-1)

    #d_ij_ab = e_occ_a[:,None] + e_occ_b
    #d_a_a = e_vir_a[:,None]
    #D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    #D_aij_aba = D_n_aba.reshape(-1)

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    ### applying imaginary shift to vve elemts ###
    #D_n_a[idx_a_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    ### applying imaginary shift to vve elemts ###
    #D_n_b[idx_b_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_n_bab = D_n_bab.reshape((nvir_b,nocc_b,nocc_a))
    ### applying imaginary shift to vve elemts ###
    #D_n_bab[idx_b_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_bab = D_n_bab.reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_n_aba = D_n_aba.reshape((nvir_a,nocc_a,nocc_b))
    ### applying imaginary shift to vve elemts ###
    #D_n_aba[idx_a_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_aba = D_n_aba.reshape(-1)

    diag = np.zeros(dim)#.astype(complex)

    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    diag[s_a:f_a] = M_ij_a_diag.copy()
    diag[s_b:f_b] = M_ij_b_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s_aaa:f_aaa] = D_aij_a.copy()
    diag[s_bab:f_bab] = D_aij_bab.copy()
    diag[s_aba:f_aba] = D_aij_aba.copy()
    diag[s_bbb:f_bbb] = D_aij_b.copy()

    ###### Additional terms for the preconditioner ####
     
    if (method == "adc(2)-x" or method == "adc(3)"):
        if eris is None:
            eris = adc.transform_integrals()

            if isinstance(eris.vvvv, np.ndarray): 

                eris_oooo = eris.oooo
                eris_OOOO = eris.OOOO
                eris_ooOO = eris.ooOO
                eris_oovv = eris.oovv
                eris_OOVV = eris.OOVV
                eris_ooVV = eris.ooVV
                eris_OOvv = eris.OOvv
                eris_ovvo = eris.ovvo
                eris_OVVO = eris.OVVO

                eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
                eris_oooo_p -= np.ascontiguousarray(eris_oooo_p.transpose(0,1,3,2))
                eris_oooo_p = eris_oooo_p.reshape(nocc_a*nocc_a, nocc_a*nocc_a)
  
                temp = np.zeros((nvir_a,eris_oooo_p.shape[0]))
                temp[:] += np.diagonal(eris_oooo_p)
                temp = temp.reshape(nvir_a, nocc_a, nocc_a)
                diag[s_aaa:f_aaa] += -temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

                eris_OOOO_p = np.ascontiguousarray(eris_OOOO.transpose(0,2,1,3))
                eris_OOOO_p -= np.ascontiguousarray(eris_OOOO_p.transpose(0,1,3,2))
                eris_OOOO_p = eris_OOOO_p.reshape(nocc_b*nocc_b, nocc_b*nocc_b)
  
                temp = np.zeros((nvir_b,eris_OOOO_p.shape[0]))
                temp[:] += np.diagonal(eris_OOOO_p)
                temp = temp.reshape(nvir_b, nocc_b, nocc_b)
                diag[s_bbb:f_bbb] += -temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

                eris_oOoO_p = np.ascontiguousarray(eris_ooOO.transpose(0,2,1,3))
                eris_oOoO_p = eris_oOoO_p.reshape(nocc_a*nocc_b, nocc_a*nocc_b)
  
                temp = np.zeros((nvir_b, eris_oOoO_p.shape[0]))
                temp[:] += np.diag(eris_oOoO_p)
                diag[s_bab:f_bab] += -temp.reshape(-1)
                
                temp = np.zeros((nvir_a, eris_oOoO_p.shape[0]))
                temp[:] += np.diag(eris_oOoO_p.T)
                diag[s_aba:f_aba] += -temp.reshape(-1)
                
                eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
                eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
                eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
  
                temp = np.zeros((nocc_a,nocc_a,nvir_a))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a,nvir_a)
                temp = np.ascontiguousarray(temp.T)
                diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

                temp = np.ascontiguousarray(temp.transpose(0,2,1))
                diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

                eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
                eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
                eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
  
                temp = np.zeros((nocc_b,nocc_b,nvir_b))
                temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b,nvir_b)
                temp = np.ascontiguousarray(temp.T)
                diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

                temp = np.ascontiguousarray(temp.transpose(0,2,1))
                diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

                temp = np.zeros((nocc_a, nocc_b, nvir_b))
                temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
                temp = np.ascontiguousarray(temp.transpose(2,0,1))
                diag[s_bab:f_bab] += temp.reshape(-1)

                temp = np.zeros((nocc_b, nocc_a, nvir_a))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
                temp = np.ascontiguousarray(temp.transpose(2,0,1))
                diag[s_aba:f_aba] += temp.reshape(-1)

                eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
                eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)
  
                temp = np.zeros((nocc_b, nocc_a, nvir_b))
                temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
                temp = np.ascontiguousarray(temp.transpose(2,1,0))
                diag[s_bab:f_bab] += temp.reshape(-1)

                eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
                eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)

                temp = np.zeros((nocc_a, nocc_b, nvir_a))
                temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
                temp = np.ascontiguousarray(temp.transpose(2,1,0))
                diag[s_aba:f_aba] += temp.reshape(-1)
    
    if cvs is True:
        
        shift = -100000000000.0 
        #if np.imag(imaginary_shift) != 0:
        #    shift += 0*1j
        
        ncore = adc.ncore_cvs
        cvs_type = adc.cvs_type
        diag[(s_a+ncore):f_a] += shift
        diag[(s_b+ncore):f_b] += shift

        temp = np.zeros((nvir_a, nocc_a, nocc_a))#.astype(complex)
        temp[:,ij_ind_a[0],ij_ind_a[1]] = diag[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
        temp[:,ij_ind_a[1],ij_ind_a[0]] = -diag[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
        temp[:,ncore:,ncore:] += shift
        if cvs_type == 'cve':
            temp[:,:ncore,:ncore] += shift
        #temp[:,ncore:,:ncore] = shift
        #temp[:,:ncore,:ncore] += shift

        diag[s_aaa:f_aaa] = temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1).copy()

        temp = diag[s_bab:f_bab].copy()
        temp = temp.reshape((nvir_b, nocc_b, nocc_a))
        temp[:,ncore:,ncore:] += shift
        if cvs_type == 'cve':
            temp[:,:ncore,:ncore] += shift
        #temp[:,ncore:,:ncore] = shift
        #temp[:,:ncore,:ncore] += shift
        diag[s_bab:f_bab] = temp.reshape(-1).copy()

        temp = diag[s_aba:f_aba].copy()
        temp = temp.reshape((nvir_a, nocc_a, nocc_b))
        temp[:,ncore:,ncore:] += shift
        if cvs_type == 'cve':
            temp[:,:ncore,:ncore] += shift
        #temp[:,ncore:,:ncore] = shift
        #temp[:,:ncore,:ncore] += shift

        diag[s_aba:f_aba] = temp.reshape(-1).copy()

        temp = np.zeros((nvir_b, nocc_b, nocc_b))#.astype(complex)
        temp[:,ij_ind_b[0],ij_ind_b[1]] = diag[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
        temp[:,ij_ind_b[1],ij_ind_b[0]] = -diag[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

        temp[:,ncore:,ncore:] += shift
        if cvs_type == 'cve':
            temp[:,:ncore,:ncore] += shift
        #temp[:,ncore:,:ncore] = shift
        #temp[:,:ncore,:ncore] += shift

        diag[s_bbb:f_bbb] = temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1).copy()
        
        
    diag = -diag

    return diag

def cvs_projector(adc, r,):

    ncore = adc.ncore_cvs
    cvs_type = adc.cvs_type

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    imaginary_shift = adc.imaginary_shift

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    ij_a = np.tril_indices(nocc_a, k=-1)
    ij_b = np.tril_indices(nocc_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    Pr = r#.astype(complex)

    Pr[(s_a+ncore):f_a] = 0.0
    Pr[(s_b+ncore):f_b] = 0.0

    temp = np.zeros((nvir_a, nocc_a, nocc_a))#.astype(complex)
    temp[:,ij_a[0],ij_a[1]] = Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
    temp[:,ij_a[1],ij_a[0]] = -Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    if cvs_type == 'cve':
        temp[:,:ncore,:ncore] = 0.0

    Pr[s_aaa:f_aaa] = temp[:,ij_a[0],ij_a[1]].reshape(-1).copy()

    temp = Pr[s_bab:f_bab].copy()
    temp = temp.reshape((nvir_b, nocc_b, nocc_a))

    temp[:,ncore:,ncore:] = 0.0
    if cvs_type == 'cve':
        temp[:,:ncore,:ncore] = 0.0

    Pr[s_bab:f_bab] = temp.reshape(-1).copy()

    temp = Pr[s_aba:f_aba].copy()
    temp = temp.reshape((nvir_a, nocc_a, nocc_b))

    temp[:,ncore:,ncore:] = 0.0
    if cvs_type == 'cve':
        temp[:,:ncore,:ncore] = 0.0

    Pr[s_aba:f_aba] = temp.reshape(-1).copy()

    temp = np.zeros((nvir_b, nocc_b, nocc_b))#.astype(complex)
    temp[:,ij_b[0],ij_b[1]] = Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
    temp[:,ij_b[1],ij_b[0]] = -Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    if cvs_type == 'cve':
        temp[:,:ncore,:ncore] = 0.0

    Pr[s_bbb:f_bbb] = temp[:,ij_b[0],ij_b[1]].reshape(-1).copy()

    return Pr

def vve_projector(adc, r, cvs_composition=False, energy_thresh=None, analyze_v_vve=False):

    ncore = adc.ncore_cvs
    cvs_type = adc.cvs_type

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    e_vir_a = adc.mo_energy_a[nocc_a:]#.astype(complex)
    e_vir_b = adc.mo_energy_b[nocc_b:]#.astype(complex)
    if energy_thresh is None:
        energy_thresh = adc.energy_thresh
    idx_a_thresh = np.argwhere(e_vir_a >= energy_thresh)  
    idx_b_thresh = np.argwhere(e_vir_b >= energy_thresh)  

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    ij_a = np.tril_indices(nocc_a, k=-1)
    ij_b = np.tril_indices(nocc_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    Pr = r#.astype(complex)

    if analyze_v_vve:
        Pr[s_a:ncore] = 0.0
        Pr[s_b:(s_b+ncore)] = 0.0

    norm_ecc = 0
    norm_ecv = 0
    norm_evc = 0
    norm_evv = 0
 
    if cvs_composition or analyze_v_vve:
        norm_c = np.dot(Pr[s_a:(s_a+ncore)], Pr[s_a:(s_a+ncore)])  
        norm_c += np.dot(Pr[s_b:(s_b+ncore)], Pr[s_b:(s_b+ncore)]) 
        norm_v = np.dot(Pr[(s_a+ncore):f_a], Pr[(s_a+ncore):f_a])  
        norm_v += np.dot(Pr[(s_b+ncore):f_b], Pr[(s_b+ncore):f_b]) 

    temp = np.zeros((nvir_a, nocc_a, nocc_a))#.astype(complex)
    temp[:,ij_a[0],ij_a[1]] = Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
    temp[:,ij_a[1],ij_a[0]] = -Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()

    if cvs_composition or analyze_v_vve:
        norm_ecc += np.dot(temp[:, :ncore, :ncore].ravel(), temp[:, :ncore, :ncore].ravel())/2
        norm_ecv += np.dot(temp[:, :ncore, ncore:].ravel(), temp[:, :ncore, ncore:].ravel())/2
        norm_evc += np.dot(temp[:, ncore:, :ncore].ravel(), temp[:, ncore:, :ncore].ravel())/2
        norm_evv += np.dot(temp[:, ncore:, ncore:].ravel(), temp[:, ncore:, ncore:].ravel())/2

    if analyze_v_vve:
        temp[:,:ncore,:ncore] = 0.0
        temp[:,:ncore,:] = 0.0
        temp[:,:,:ncore] = 0.0
    else:
        temp[idx_a_thresh,ncore:,ncore:] = 0.0
    Pr[s_aaa:f_aaa] = temp[:,ij_a[0],ij_a[1]].reshape(-1).copy()

    temp = Pr[s_bab:f_bab].copy()
    temp = temp.reshape((nvir_b, nocc_b, nocc_a))

    if cvs_composition or analyze_v_vve:
        norm_ecc += np.dot(temp[:, :ncore, :ncore].ravel(), temp[:, :ncore, :ncore].ravel())
        norm_ecv += np.dot(temp[:, :ncore, ncore:].ravel(), temp[:, :ncore, ncore:].ravel())
        norm_evc += np.dot(temp[:, ncore:, :ncore].ravel(), temp[:, ncore:, :ncore].ravel())
        norm_evv += np.dot(temp[:, ncore:, ncore:].ravel(), temp[:, ncore:, ncore:].ravel())

    if analyze_v_vve:
        temp[:,:ncore,:ncore] = 0.0
        temp[:,:ncore,:] = 0.0
        temp[:,:,:ncore] = 0.0
    else:
        temp[idx_a_thresh,ncore:,ncore:] = 0.0
    Pr[s_bab:f_bab] = temp.reshape(-1).copy()

    temp = Pr[s_aba:f_aba].copy()
    temp = temp.reshape((nvir_a, nocc_a, nocc_b))

    if cvs_composition or analyze_v_vve:
        norm_ecc += np.dot(temp[:, :ncore, :ncore].ravel(), temp[:, :ncore, :ncore].ravel())
        norm_ecv += np.dot(temp[:, :ncore, ncore:].ravel(), temp[:, :ncore, ncore:].ravel())
        norm_evc += np.dot(temp[:, ncore:, :ncore].ravel(), temp[:, ncore:, :ncore].ravel())
        norm_evv += np.dot(temp[:, ncore:, ncore:].ravel(), temp[:, ncore:, ncore:].ravel())

    if analyze_v_vve:
        temp[:,:ncore,:ncore] = 0.0
        temp[:,:ncore,:] = 0.0
        temp[:,:,:ncore] = 0.0
    else:
        temp[idx_a_thresh,ncore:,ncore:] = 0.0
    Pr[s_aba:f_aba] = temp.reshape(-1).copy()

    temp = np.zeros((nvir_b, nocc_b, nocc_b))#.astype(complex)
    temp[:,ij_b[0],ij_b[1]] = Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
    temp[:,ij_b[1],ij_b[0]] = -Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

    if cvs_composition or analyze_v_vve:
        norm_ecc += np.dot(temp[:, :ncore, :ncore].ravel(), temp[:, :ncore, :ncore].ravel())/2
        norm_ecv += np.dot(temp[:, :ncore, ncore:].ravel(), temp[:, :ncore, ncore:].ravel())/2
        norm_evc += np.dot(temp[:, ncore:, :ncore].ravel(), temp[:, ncore:, :ncore].ravel())/2
        norm_evv += np.dot(temp[:, ncore:, ncore:].ravel(), temp[:, ncore:, ncore:].ravel())/2


        logger.info(adc, "-----composition by cvs components------------")
        print("norm_c   = ", np.round(norm_c, 6))
        print("nrom_v   = ", np.round(norm_v, 6))
        print("norm_ecc = ", np.round(norm_ecc, 6))
        print("norm_ecv = ", np.round(norm_ecv, 6))
        print("norm_evc = ", np.round(norm_evc, 6))
        print("norm_evv = ", np.round(norm_evv, 6))
        logger.info(adc, "----------------------------------------------")

    if analyze_v_vve:
        temp[:,:ncore,:ncore] = 0.0
        temp[:,:ncore,:] = 0.0
        temp[:,:,:ncore] = 0.0
    else:
        temp[idx_a_thresh,ncore:,ncore:] = 0.0

    Pr[s_bbb:f_bbb] = temp[:,ij_b[0],ij_b[1]].reshape(-1).copy()

    if cvs_composition or analyze_v_vve:
        return Pr, norm_v, norm_evv
    else:
        return Pr

def ea_contract_r_vvvv_antisym(myadc,r2,vvvv_d):

    nocc = r2.shape[0]
    nvir = r2.shape[1] 

    nv_pair = nvir  *  (nvir - 1) // 2
    tril_idx = np.tril_indices(nvir, k=-1)               

    r2 = r2[:,tril_idx[0],tril_idx[1]]
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))

    r2_vvvv = np.zeros((nocc,nvir,nvir))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    a = 0
    if isinstance(vvvv_d,list):
        for dataset in vvvv_d:
             k = dataset.shape[0]
             dataset = dataset[:].reshape(-1,nv_pair)
             r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
             a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv = dfadc.get_vvvv_antisym_df(myadc, vvvv_d, p, chnk_size)
            k = vvvv.shape[0]
            vvvv = vvvv.reshape(-1,nv_pair)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv.T).reshape(nocc,-1,nvir)
            del (vvvv)
            a += k
    else :
        raise Exception("Unknown vvvv type") 
    return r2_vvvv


def ea_contract_r_vvvv(myadc,r2,vvvv_d):

    nocc_1 = r2.shape[0]
    nvir_1 = r2.shape[1] 
    nvir_2 = r2.shape[2] 

    r2 = r2.reshape(-1,nvir_1*nvir_2)
    r2_vvvv = np.zeros((nocc_1,nvir_1,nvir_2))
    chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv_d, list):
        for dataset in vvvv_d:
             k = dataset.shape[0]
             dataset = dataset[:].reshape(-1,nvir_1*nvir_2)
             r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc_1,-1,nvir_2)
             a += k
    elif getattr(myadc, 'with_df', None):
        Lvv = vvvv_d[0]
        LVV = vvvv_d[1]
        for p in range(0,nvir_1,chnk_size):
            vvvv = dfadc.get_vVvV_df(myadc, Lvv, LVV, p, chnk_size)
            k = vvvv.shape[0]
            vvvv = vvvv.reshape(-1,nvir_1*nvir_2)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv.T).reshape(nocc_1,-1,nvir_2)
            del (vvvv)
            a += k
    else :
        raise Exception("Unknown vvvv type") 

    return r2_vvvv

def compute_rdm_tdm(adc, U1, U2):
    
    U1 = np.array(U1)
    U2 = np.array(U2)
    #U1 = U1[0]
    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]
    t1_2_a = adc.t1[0][0][:]
    t1_2_b = adc.t1[0][1][:]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    nmo_a = nocc_a + nvir_a
    nmo_b = nocc_b + nvir_b

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb
    
    total_opdm_a  = np.zeros((nmo_a,nmo_a))
    total_opdm_b  = np.zeros((nmo_b,nmo_b))
    kd_oc_a = np.identity(nocc_a)
    kd_oc_b = np.identity(nocc_b)

    U1_a = U1[s_a:f_a]
    U1_b = U1[s_b:f_b]
    U1_aaa = U1[s_aaa:f_aaa]
    U1_bab = U1[s_bab:f_bab]
    U1_aba = U1[s_aba:f_aba]
    U1_bbb = U1[s_bbb:f_bbb]

    U2_a = U2[s_a:f_a]
    U2_b = U2[s_b:f_b]
    U2_aaa = U2[s_aaa:f_aaa]
    U2_bab = U2[s_bab:f_bab]
    U2_aba = U2[s_aba:f_aba]
    U2_bbb = U2[s_bbb:f_bbb]

    U1_aaa = U1_aaa.reshape(nvir_a,-1)
    U1_bbb = U1_bbb.reshape(nvir_b,-1)
    U1_aaa_u = None
    U1_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
    U1_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= U1_aaa.copy()
    U1_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -U1_aaa.copy()

    U1_bbb_u = None
    U1_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
    U1_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= U1_bbb.copy()
    U1_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -U1_bbb.copy()

    U1_aba = U1_aba.reshape(nvir_a,nocc_a,nocc_b)
    U1_bab = U1_bab.reshape(nvir_b,nocc_b,nocc_a)


    U2_aaa = U2_aaa.reshape(nvir_a,-1)
    U2_bbb = U2_bbb.reshape(nvir_b,-1)
    U2_aaa_u = None
    U2_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
    U2_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= U2_aaa.copy()
    U2_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -U2_aaa.copy()

    U2_bbb_u = None
    U2_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
    U2_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= U2_bbb.copy()
    U2_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -U2_bbb.copy()

    U2_aba = U2_aba.reshape(nvir_a,nocc_a,nocc_b)
    U2_bab = U2_bab.reshape(nvir_b,nocc_b,nocc_a)

################TOTAL OPDMS OPDMS OPDMS OPDMS####################################

#####G^000#### block- ij
    total_opdm_a[:nocc_a,:nocc_a] =  np.einsum('ij,m,m->ij',kd_oc_a,U1_a,U2_a,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] -= np.einsum('i,j->ij',U1_a,U2_a,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] +=  np.einsum('ij,m,m->ij',kd_oc_a,U1_b,U2_b,optimize=True)

    total_opdm_b[:nocc_b,:nocc_b] =  np.einsum('ij,m,m->ij',kd_oc_b,U1_b,U2_b,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] -= np.einsum('i,j->ij',U1_b,U2_b,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] += np.einsum('ij,m,m->ij',kd_oc_b,U1_a,U2_a,optimize=True) 

####G^101#### block- ij

    total_opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_a,U1_aaa_u,U2_aaa_u,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] += np.einsum('ij,etu,etu->ij',kd_oc_a,U1_bab,U2_bab,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] += np.einsum('ij,etu,etu->ij',kd_oc_a,U1_aba,U2_aba,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_a,U1_bbb_u,U2_bbb_u,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] -= np.einsum('eti,etj->ij',U1_aaa_u,U2_aaa_u,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] -= np.einsum('eti,etj->ij',U1_bab,U2_bab,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] -= np.einsum('eit,ejt->ij',U1_aba,U2_aba,optimize=True)

    total_opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_b,U1_aaa_u,U2_aaa_u,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] += np.einsum('ij,etu,etu->ij',kd_oc_b,U1_bab,U2_bab,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] += np.einsum('ij,etu,etu->ij',kd_oc_b,U1_aba,U2_aba,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_b,U1_bbb_u,U2_bbb_u,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] -= np.einsum('eti,etj->ij',U1_bbb_u,U2_bbb_u,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] -= np.einsum('eti,etj->ij',U1_aba,U2_aba,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] -= np.einsum('eit,ejt->ij',U1_bab,U2_bab,optimize=True)

######G^020#### block- ij

    total_opdm_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', U1_a,U2_a,t2_1_a,t2_1_a,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', U1_b,U2_b,t2_1_a,t2_1_a,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] -= np.einsum('g,g,jhcd,ihcd->ij', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] -= np.einsum('g,g,jhcd,ihcd->ij', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True) 
    total_opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('g,h,jgcd,ihcd->ij', U1_a,U2_a,t2_1_a,t2_1_a,optimize=True)  
    total_opdm_a[:nocc_a,:nocc_a] += np.einsum('g,h,jhcd,ihcd->ij', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)  
    total_opdm_a[:nocc_a,:nocc_a] += 0.25*np.einsum('g,j,ghcd,ihcd->ij',U1_a,U2_a,t2_1_a,t2_1_a,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('g,j,ghcd,ihcd->ij',U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] += 0.25*np.einsum('g,i,jhcd,ghcd->ij',U1_a,U2_a,t2_1_a,t2_1_a,optimize=True)
    total_opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('g,i,jhcd,ghcd->ij',U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)


    total_opdm_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', U1_b,U2_b,t2_1_b,t2_1_b,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', U1_a,U2_a,t2_1_b,t2_1_b,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] -= np.einsum('g,g,hjcd,hicd->ij', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] -= np.einsum('g,g,hjcd,hicd->ij', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True) 
    total_opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('g,h,jgcd,ihcd->ij', U1_b,U2_b,t2_1_b,t2_1_b,optimize=True)  
    total_opdm_b[:nocc_b,:nocc_b] += np.einsum('g,h,gjcd,hicd->ij', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)  
    total_opdm_b[:nocc_b,:nocc_b] += 0.25*np.einsum('g,j,ghcd,ihcd->ij',U1_b,U2_b,t2_1_b,t2_1_b,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('g,j,hgcd,hicd->ij',U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] += 0.25*np.einsum('g,i,jhcd,ghcd->ij',U1_b,U2_b,t2_1_b,t2_1_b,optimize=True)
    total_opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('g,i,hjcd,hgcd->ij',U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)
        
#####G^101#### block- ab

    total_opdm_a[nocc_a:,nocc_a:] = 0.5*np.einsum('atu,btu->ab', U1_aaa_u,U2_aaa_u,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] += np.einsum('atu,btu->ab', U1_aba,U2_aba,optimize=True)

    total_opdm_b[nocc_b:,nocc_b:] = 0.5*np.einsum('atu,btu->ab', U1_bbb_u,U2_bbb_u,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] += np.einsum('atu,btu->ab', U1_bab,U2_bab,optimize=True)

#####G^020#### block- ab
    total_opdm_a[nocc_a:,nocc_a:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', U1_a,U2_a,t2_1_a,t2_1_a,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', U1_b,U2_b,t2_1_a,t2_1_a,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] += np.einsum('g,g,hmbc,hmac->ab', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] += np.einsum('g,g,hmbc,hmac->ab', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] -= np.einsum('g,h,hmbc,gmac->ab', U1_a,U2_a,t2_1_a,t2_1_a,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] -= np.einsum('g,h,hmbc,gmac->ab', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_a[nocc_a:,nocc_a:] -= np.einsum('g,h,mhbc,mgac->ab', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)

    total_opdm_b[nocc_b:,nocc_b:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', U1_b,U2_b,t2_1_b,t2_1_b,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', U1_a,U2_a,t2_1_b,t2_1_b,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] += np.einsum('g,g,hmcb,hmca->ab', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] += np.einsum('g,g,hmcb,hmca->ab', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] -= np.einsum('g,h,hmbc,gmac->ab', U1_b,U2_b,t2_1_b,t2_1_b,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] -= np.einsum('g,h,mhcb,mgca->ab', U1_b,U2_b,t2_1_ab,t2_1_ab,optimize=True)
    total_opdm_b[nocc_b:,nocc_b:] -= np.einsum('g,h,hmcb,gmca->ab', U1_a,U2_a,t2_1_ab,t2_1_ab,optimize=True)

#####G^100#### block- ia
    total_opdm_a[:nocc_a,nocc_a:] = -np.einsum('n,ani->ia', U1_a,U2_aaa_u,optimize=True)
    total_opdm_a[:nocc_a,nocc_a:] += np.einsum('n,ain->ia', U1_b,U2_aba,optimize=True)

    total_opdm_b[:nocc_b,nocc_b:] = -np.einsum('n,ani->ia', U1_b,U2_bbb_u,optimize=True)
    total_opdm_b[:nocc_b,nocc_b:] += np.einsum('n,ain->ia', U1_a,U2_bab,optimize=True)
#####G^110#### block- ia

    total_opdm_a[:nocc_a,nocc_a:] -= np.einsum('g,cgh,ihac->ia', U1_a,U2_aaa_u,t2_1_a,optimize=True) 
    total_opdm_a[:nocc_a,nocc_a:] += np.einsum('g,chg,ihac->ia', U1_a,U2_bab,t2_1_ab,optimize=True) 
    total_opdm_a[:nocc_a,nocc_a:] += np.einsum('g,chg,ihac->ia', U1_b,U2_aba,t2_1_a,optimize=True) 
    total_opdm_a[:nocc_a,nocc_a:] -= np.einsum('g,cgh,ihac->ia', U1_b,U2_bbb_u,t2_1_ab,optimize=True) 
    total_opdm_a[:nocc_a,nocc_a:] += 0.5*np.einsum('i,cgh,ghac->ia', U1_a,U2_aaa_u,t2_1_a,optimize=True)
    total_opdm_a[:nocc_a,nocc_a:] -= np.einsum('i,chg,ghac->ia', U1_a,U2_bab,t2_1_ab,optimize=True)

    total_opdm_b[:nocc_b,nocc_b:] -= np.einsum('g,cgh,ihac->ia', U1_b,U2_bbb_u,t2_1_b,optimize=True) 
    total_opdm_b[:nocc_b,nocc_b:] += np.einsum('g,chg,hica->ia', U1_b,U2_aba,t2_1_ab,optimize=True) 
    total_opdm_b[:nocc_b,nocc_b:] += np.einsum('g,chg,hica->ia', U1_a,U2_bab,t2_1_b,optimize=True) 
    total_opdm_b[:nocc_b,nocc_b:] -= np.einsum('g,cgh,hica->ia', U1_a,U2_aaa_u,t2_1_ab,optimize=True) 
    total_opdm_b[:nocc_b,nocc_b:] += 0.5*np.einsum('i,cgh,ghac->ia', U1_b,U2_bbb_u,t2_1_b,optimize=True)
    total_opdm_b[:nocc_b,nocc_b:] -= np.einsum('i,chg,hgca->ia', U1_b,U2_aba,t2_1_ab,optimize=True)

######G^020#### block- ia

    total_opdm_a[:nocc_a,nocc_a:] += np.einsum('g,g,ia->ia', U1_a,U2_a,t1_2_a,optimize=True) 
    total_opdm_a[:nocc_a,nocc_a:] += np.einsum('g,g,ia->ia', U1_b,U2_b,t1_2_a,optimize=True) 
    total_opdm_a[:nocc_a,nocc_a:] -= np.einsum('g,i,ga->ia', U1_a,U2_a,t1_2_a,optimize=True)

    total_opdm_b[:nocc_b,nocc_b:] += np.einsum('g,g,ia->ia', U1_b,U2_b,t1_2_b,optimize=True) 
    total_opdm_b[:nocc_b,nocc_b:] += np.einsum('g,g,ia->ia', U1_a,U2_a,t1_2_b,optimize=True) 
    total_opdm_b[:nocc_b,nocc_b:] -= np.einsum('g,i,ga->ia', U1_b,U2_b,t1_2_b,optimize=True)

#######G^001#### block- ai

    total_opdm_a[nocc_a:,:nocc_a] = -np.einsum('m,ami->ai', U1_a,U2_aaa_u,optimize=True)
    total_opdm_a[nocc_a:,:nocc_a] += np.einsum('m,aim->ai', U1_b,U2_aba,optimize=True)

    total_opdm_b[nocc_b:,:nocc_b] = -np.einsum('m,ami->ai', U1_b,U2_bbb_u,optimize=True)
    total_opdm_b[nocc_b:,:nocc_b] += np.einsum('m,aim->ai', U1_a,U2_bab,optimize=True)

#######G^110#### block- ai
    total_opdm_a[nocc_a:,:nocc_a] -= np.einsum('g,cgh,ihac->ai', U1_a,U2_aaa_u,t2_1_a,optimize=True) 
    total_opdm_a[nocc_a:,:nocc_a] += np.einsum('g,chg,ihac->ai', U1_a,U2_bab,t2_1_ab,optimize=True) 
    total_opdm_a[nocc_a:,:nocc_a] += np.einsum('g,chg,ihac->ai', U1_b,U2_aba,t2_1_a,optimize=True) 
    total_opdm_a[nocc_a:,:nocc_a] -= np.einsum('g,cgh,ihac->ai', U1_b,U2_bbb_u,t2_1_ab,optimize=True) 
    total_opdm_a[nocc_a:,:nocc_a] += 0.5*np.einsum('i,cgh,ghac->ai', U1_a,U2_aaa_u,t2_1_a,optimize=True)
    total_opdm_a[nocc_a:,:nocc_a] -= np.einsum('i,chg,ghac->ai', U1_a,U2_bab,t2_1_ab,optimize=True)

    total_opdm_b[nocc_b:,:nocc_b] -= np.einsum('g,cgh,ihac->ai', U1_b,U2_bbb_u,t2_1_b,optimize=True) 
    total_opdm_b[nocc_b:,:nocc_b] += np.einsum('g,chg,hica->ai', U1_b,U2_aba,t2_1_ab,optimize=True) 
    total_opdm_b[nocc_b:,:nocc_b] += np.einsum('g,chg,ihac->ai', U1_a,U2_bab,t2_1_b,optimize=True) 
    total_opdm_b[nocc_b:,:nocc_b] -= np.einsum('g,cgh,hica->ai', U1_a,U2_aaa_u,t2_1_ab,optimize=True) 
    total_opdm_b[nocc_b:,:nocc_b] += 0.5*np.einsum('i,cgh,ghac->ai', U1_b,U2_bbb_u,t2_1_b,optimize=True)
    total_opdm_b[nocc_b:,:nocc_b] -= np.einsum('i,chg,hgca->ai', U1_b,U2_aba,t2_1_ab,optimize=True)
#######G^020#### block- ai
    total_opdm_a[nocc_a:,:nocc_a] += np.einsum('g,g,ia->ai', U1_a,U2_a,t1_2_a,optimize=True) 
    total_opdm_a[nocc_a:,:nocc_a] += np.einsum('g,g,ia->ai', U1_b,U2_b,t1_2_a,optimize=True) 
    total_opdm_a[nocc_a:,:nocc_a] -= np.einsum('g,i,ga->ai', U1_a,U2_a,t1_2_a,optimize=True)

    total_opdm_b[nocc_b:,:nocc_b] += np.einsum('g,g,ia->ai', U1_b,U2_b,t1_2_b,optimize=True) 
    total_opdm_b[nocc_b:,:nocc_b] += np.einsum('g,g,ia->ai', U1_a,U2_a,t1_2_b,optimize=True) 
    total_opdm_b[nocc_b:,:nocc_b] -= np.einsum('g,i,ga->ai', U1_b,U2_b,t1_2_b,optimize=True)
#########################
    herm_a = np.linalg.norm(total_opdm_a - total_opdm_a.transpose(1,0))
    herm_b = np.linalg.norm(total_opdm_b - total_opdm_b.transpose(1,0))
    norm_a = np.linalg.norm(total_opdm_a)
    norm_b = np.linalg.norm(total_opdm_b)

    print('norm_a: ', norm_a)
    print('norm_b: ', norm_b)
    print("total OPDM singles norm for Hermiticity (alpha)",herm_a)
    print("total OPDM singles norm for Hermiticity (beta)",herm_b)
    print("alpha opdm trace",np.einsum('pp',total_opdm_a))
    print("beta opdm trace",np.einsum('pp',total_opdm_b))
    
    return total_opdm_a, total_opdm_b

def ea_adc_matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov
    eris_OVOV = eris.OVOV
    eris_ovOV = eris.ovOV
    eris_OVov = eris.OVov

    d_i_a = e_occ_a[:,None]
    d_ab_a = e_vir_a[:,None] + e_vir_a
    D_n_a = -d_i_a + d_ab_a.reshape(-1)
    D_n_a = D_n_a.reshape((nocc_a,nvir_a,nvir_a))
    D_iab_a = D_n_a.copy()[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

    d_i_b = e_occ_b[:,None]
    d_ab_b = e_vir_b[:,None] + e_vir_b
    D_n_b = -d_i_b + d_ab_b.reshape(-1)
    D_n_b = D_n_b.reshape((nocc_b,nvir_b,nvir_b))
    D_iab_b = D_n_b.copy()[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

    d_ab_ab = e_vir_a[:,None] + e_vir_b
    d_i_b = e_occ_b[:,None]
    D_n_bab = -d_i_b + d_ab_ab.reshape(-1)
    D_iab_bab = D_n_bab.reshape(-1)

    d_ab_ab = e_vir_b[:,None] + e_vir_a
    d_i_a = e_occ_a[:,None]
    D_n_aba = -d_i_a + d_ab_ab.reshape(-1)
    D_iab_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    if M_ab is None:
        M_ab = adc.get_imds()
    M_ab_a, M_ab_b = M_ab
    
    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]

        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_aaa_ = np.zeros((nocc_a, nvir_a, nvir_a))
        r_aaa_[:, ab_ind_a[0], ab_ind_a[1]] = r_aaa.reshape(nocc_a, -1)
        r_aaa_[:, ab_ind_a[1], ab_ind_a[0]] = -r_aaa.reshape(nocc_a, -1)
        r_bbb_ = np.zeros((nocc_b, nvir_b, nvir_b))
        r_bbb_[:, ab_ind_b[0], ab_ind_b[1]] = r_bbb.reshape(nocc_b, -1)
        r_bbb_[:, ab_ind_b[1], ab_ind_b[0]] = -r_bbb.reshape(nocc_b, -1)
     
        r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)
        r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)

############ ADC(2) ab block ############################

        s[s_a:f_a] = lib.einsum('ab,b->a',M_ab_a,r_a)
        s[s_b:f_b] = lib.einsum('ab,b->a',M_ab_b,r_b)

############ ADC(2) a - ibc and ibc - a coupling blocks #########################


        if isinstance(eris.ovvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_a

        a = 0
        temp = np.zeros((nocc_a, nvir_a, nvir_a))
        for p in range(0,nocc_a,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            k = eris_ovvv.shape[0]

            s[s_a:f_a] += 0.5*lib.einsum('icab,ibc->a',eris_ovvv, r_aaa_[a:a+k], optimize = True)
            s[s_a:f_a] -= 0.5*lib.einsum('ibac,ibc->a',eris_ovvv, r_aaa_[a:a+k], optimize = True)
            temp[a:a+k] += lib.einsum('icab,a->ibc', eris_ovvv, r_a, optimize = True)
            temp[a:a+k] -= lib.einsum('ibac,a->ibc', eris_ovvv, r_a, optimize = True)
            del eris_ovvv
            a += k

        s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
        del (temp)

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_b

        a = 0
        temp = np.zeros((nocc_b, nvir_a, nvir_b))
        for p in range(0,nocc_b,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_OVvv = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
            else :
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            k = eris_OVvv.shape[0]
            s[s_a:f_a] += lib.einsum('icab,ibc->a', eris_OVvv, r_bab[a:a+k], optimize = True)
            temp[a:a+k] += lib.einsum('icab,a->ibc', eris_OVvv, r_a, optimize = True)
            del eris_OVvv
            a += k
 
        s[s_bab:f_bab] += temp.reshape(-1)
        del temp

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_b
        a = 0
        temp = np.zeros((nocc_b, nvir_b, nvir_b))
        for p in range(0,nocc_b,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_OVVV = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
            else :
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            k = eris_OVVV.shape[0]
            s[s_b:f_b] += 0.5*lib.einsum('icab,ibc->a',eris_OVVV, r_bbb_[a:a+k], optimize = True)
            s[s_b:f_b] -= 0.5*lib.einsum('ibac,ibc->a',eris_OVVV, r_bbb_[a:a+k], optimize = True)
            temp[a:a+k] += lib.einsum('icab,a->ibc', eris_OVVV, r_b, optimize = True)
            temp[a:a+k] -= lib.einsum('ibac,a->ibc', eris_OVVV, r_b, optimize = True)
            del eris_OVVV
            a += k

        s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
        del temp

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_a
        a = 0
        temp = np.zeros((nocc_a, nvir_b, nvir_a))
        for p in range(0,nocc_a,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovVV = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            k = eris_ovVV.shape[0]
            s[s_b:f_b] += lib.einsum('icab,ibc->a', eris_ovVV, r_aba[a:a+k], optimize = True)
            temp[a:a+k] += lib.einsum('icab,a->ibc', eris_ovVV, r_b, optimize = True)
            del eris_ovVV
            a += k
        
        s[s_aba:f_aba] += temp.reshape(-1)
        del temp


############### ADC(2) iab - jcd block ############################

        s[s_aaa:f_aaa] += D_iab_a * r_aaa
        s[s_bab:f_bab] += D_iab_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_iab_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_iab_b * r_bbb

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

               eris_oovv = eris.oovv
               eris_OOVV = eris.OOVV
               eris_ooVV = eris.ooVV
               eris_OOvv = eris.OOvv
               eris_ovvo = eris.ovvo
               eris_OVVO = eris.OVVO
               eris_ovVO = eris.ovVO
               eris_OVvo = eris.OVvo

               r_aaa = r_aaa.reshape(nocc_a,-1)
               r_bbb = r_bbb.reshape(nocc_b,-1)

               r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = None
               r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               if isinstance(eris.vvvv_p, np.ndarray):
                   eris_vvvv = eris.vvvv_p
                   temp_1 = np.dot(r_aaa,eris_vvvv.T)
                   del eris_vvvv
               elif isinstance(eris.vvvv_p, list):
                   temp_1 = ea_contract_r_vvvv_antisym(adc,r_aaa_u,eris.vvvv_p)
                   temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]  
               else:
                   temp_1 = ea_contract_r_vvvv_antisym(adc,r_aaa_u,eris.Lvv)
                   temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]  

               s[s_aaa:f_aaa] += temp_1.reshape(-1)

               if isinstance(eris.VVVV_p, np.ndarray):
                   eris_VVVV = eris.VVVV_p 
                   temp_1 = np.dot(r_bbb,eris_VVVV.T)
                   del eris_VVVV
               elif isinstance(eris.VVVV_p, list):
                   temp_1 = ea_contract_r_vvvv_antisym(adc,r_bbb_u,eris.VVVV_p)
                   temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]
               else:
                   temp_1 = ea_contract_r_vvvv_antisym(adc,r_bbb_u,eris.LVV)
                   temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]

               s[s_bbb:f_bbb] += temp_1.reshape(-1)

               if isinstance(eris.vVvV_p, np.ndarray):
                   r_bab_t = r_bab.reshape(nocc_b,-1)
                   r_aba_t = r_aba.transpose(0,2,1).reshape(nocc_a,-1)
                   eris_vVvV = eris.vVvV_p
                   s[s_bab:f_bab] += np.dot(r_bab_t,eris_vVvV.T).reshape(-1)
                   temp_1 = np.dot(r_aba_t,eris_vVvV.T).reshape(nocc_a, nvir_a,nvir_b)
                   s[s_aba:f_aba] += temp_1.transpose(0,2,1).copy().reshape(-1)
               elif isinstance(eris.vVvV_p, list):
                   temp_1 = ea_contract_r_vvvv(adc,r_bab,eris.vVvV_p)
                   temp_2 = ea_contract_r_vvvv(adc,r_aba,eris.VvVv_p)

                   s[s_bab:f_bab] += temp_1.reshape(-1)
                   s[s_aba:f_aba] += temp_2.reshape(-1)
               else :
                   temp_1 = ea_contract_r_vvvv(adc,r_bab,(eris.Lvv,eris.LVV))
                   temp_2 = ea_contract_r_vvvv(adc,r_aba,(eris.LVV,eris.Lvv))

                   s[s_bab:f_bab] += temp_1.reshape(-1)
                   s[s_aba:f_aba] += temp_2.reshape(-1)

               temp = 0.5*lib.einsum('jiyz,jzx->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp +=0.5*lib.einsum('jzyi,jxz->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovVO,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_OVVO,r_bab,optimize = True).reshape(-1)

               temp = 0.5*lib.einsum('jiyz,jzx->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp +=0.5* lib.einsum('jzyi,jxz->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_OVvo,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('jzxi,jzy->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jzxi,jyz->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_OOvv,r_bab,optimize = True).reshape(-1)

               temp = -0.5*lib.einsum('jixz,jzy->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('jzxi,jzy->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jzxi,jyz->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*lib.einsum('jixz,jzy->ixy',eris_ooVV,r_aba,optimize = True).reshape(-1)

               temp = 0.5*lib.einsum('jixw,jyw->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*lib.einsum('jixw,jwy->ixy',eris_OOvv,r_bab,optimize = True).reshape(-1)

               temp = 0.5*lib.einsum('jixw,jyw->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jwxi,jyw->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*lib.einsum('jixw,jwy->ixy',eris_ooVV,r_aba,optimize = True).reshape(-1)

               temp = -0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVvo,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVVO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovVO,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVvo,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*lib.einsum('jiyw,jxw->ixy',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_OVVO,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovVO,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

            #print("Calculating additional terms for adc(3)")

               eris_ovoo = eris.ovoo
               eris_OVOO = eris.OVOO
               eris_ovOO = eris.ovOO
               eris_OVoo = eris.OVoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               r_aaa = r_aaa.reshape(nocc_a,-1)
               temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_a_t,r_aaa)
               del (t2_1_a_t)
               s[s_a:f_a] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
               s[s_a:f_a] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)
               del (temp)

               temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1_ab,r_bab)
               s[s_a:f_a] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovOO, optimize=True)
               del (temp_1)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               r_bbb = r_bbb.reshape(nocc_b,-1)
               temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
               del (t2_1_b_t)
               s[s_b:f_b] += lib.einsum('lmj,lamj->a',temp, eris_OVOO, optimize=True)
               s[s_b:f_b] -= lib.einsum('lmj,malj->a',temp, eris_OVOO, optimize=True)
               del (temp)

               temp_1 = -lib.einsum('mlwz,jzw->jlm',t2_1_ab,r_aba)
               s[s_b:f_b] -= lib.einsum('jlm,lamj->a',temp_1, eris_OVoo, optimize=True)
               del (temp_1)

               r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)
               r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a

               a = 0
               temp_s_a = np.zeros_like(r_bab)
               temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1_a,r_aaa_u,optimize=True)
               temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1_ab,r_bab,optimize=True)

               temp_s_a_1 = np.zeros_like(r_bab)
               temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',t2_1_a,r_aaa_u,optimize=True)
               temp_s_a_1 += -lib.einsum('ljdz,jwz->lwd',t2_1_ab,r_bab,optimize=True)

               temp_1_1 = np.zeros((nocc_a,nvir_a,nvir_a))
               temp_1_2 = np.zeros((nocc_a,nvir_a,nvir_a))
               for p in range(0,nocc_a,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                   k = eris_ovvv.shape[0]
                   s[s_a:f_a] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a[a:a+k],eris_ovvv,optimize=True)
                   s[s_a:f_a] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a[a:a+k],eris_ovvv,optimize=True)
                   s[s_a:f_a] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1[a:a+k],eris_ovvv,optimize=True)
                   s[s_a:f_a] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1[a:a+k],eris_ovvv,optimize=True)

                   temp_1_1[a:a+k] += lib.einsum('ldxb,b->lxd', eris_ovvv,r_a,optimize=True)
                   temp_1_1[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r_a,optimize=True)

                   temp_1_2[a:a+k] += lib.einsum('ldyb,b->lyd', eris_ovvv,r_a,optimize=True)
                   temp_1_2[a:a+k] -= lib.einsum('lbyd,b->lyd', eris_ovvv,r_a,optimize=True)
                   del eris_ovvv
                   a += k

               del temp_s_a
               del temp_s_a_1

               if isinstance(eris.OVVV, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_b
               a = 0
                 
               temp_s_b = np.zeros_like(r_aba)
               temp_s_b = lib.einsum('jlwd,jzw->lzd',t2_1_b,r_bbb_u,optimize=True)
               temp_s_b += lib.einsum('jlwd,jzw->lzd',t2_1_ab,r_aba,optimize=True)

               temp_s_b_1 = np.zeros_like(r_aba)
               temp_s_b_1 = -lib.einsum('jlzd,jwz->lwd',t2_1_b,r_bbb_u,optimize=True)
               temp_s_b_1 += -lib.einsum('jlzd,jwz->lwd',t2_1_ab,r_aba,optimize=True)

               temp_1_3 = np.zeros((nocc_b,nvir_b,nvir_b))
               temp_1_4 = np.zeros((nocc_b,nvir_b,nvir_b))
               for p in range(0,nocc_b,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_OVVV = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                   else :
                       eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                   k = eris_OVVV.shape[0]
                   s[s_b:f_b] += 0.5*lib.einsum('lzd,ldza->a',temp_s_b[a:a+k],eris_OVVV,optimize=True)
                   s[s_b:f_b] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_b[a:a+k],eris_OVVV,optimize=True)
                   s[s_b:f_b] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_b_1[a:a+k],eris_OVVV,optimize=True)
                   s[s_b:f_b] += 0.5*lib.einsum('lwd,lawd->a',temp_s_b_1[a:a+k],eris_OVVV,optimize=True)

                   temp_1_3[a:a+k] += lib.einsum('ldxb,b->lxd', eris_OVVV,r_b,optimize=True)
                   temp_1_3[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_OVVV,r_b,optimize=True)

                   temp_1_4[a:a+k] += lib.einsum('ldyb,b->lyd', eris_OVVV,r_b,optimize=True)
                   temp_1_4[a:a+k] -= lib.einsum('lbyd,b->lyd', eris_OVVV,r_b,optimize=True)
                   del eris_OVVV
                   a += k

               del temp_s_b
               del temp_s_b_1

               if isinstance(eris.OVvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_b

               a = 0
               temp_1 = np.zeros_like(r_bab)
               temp_1= lib.einsum('jlwd,jzw->lzd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += lib.einsum('jlwd,jzw->lzd',t2_1_b,r_bab,optimize=True)
               temp_2 = lib.einsum('jldw,jwz->lzd',t2_1_ab,r_aba,optimize=True)
               temp_1_new = np.zeros_like(r_bab)
               temp_1_new = -lib.einsum('jlzd,jwz->lwd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1_new += -lib.einsum('jlzd,jwz->lwd',t2_1_b,r_bab,optimize=True)
               temp_2_new = -lib.einsum('jldz,jzw->lwd',t2_1_ab,r_aba,optimize=True)
               temp_2_1 = np.zeros((nocc_b,nvir_a,nvir_b))
               temp_2_2 = np.zeros((nocc_b,nvir_a,nvir_b))
               temp = np.zeros((nocc_b,nvir_a,nvir_a))
               for p in range(0,nocc_b,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_OVvv = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                   else :
                       eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                   k = eris_OVvv.shape[0]
                   s[s_a:f_a] += 0.5*lib.einsum('lzd,ldza->a',temp_1[a:a+k],eris_OVvv,optimize=True)

                   s[s_b:f_b] -= 0.5*lib.einsum('lzd,lazd->a',temp_2[a:a+k],eris_OVvv,optimize=True)

                   s[s_a:f_a] -= 0.5*lib.einsum('lwd,ldwa->a',temp_1_new[a:a+k],eris_OVvv,optimize=True)

                   s[s_b:f_b] += 0.5*lib.einsum('lwd,lawd->a',temp_2_new[a:a+k],eris_OVvv,optimize=True)

                   temp_2_1[a:a+k] += lib.einsum('ldxb,b->lxd', eris_OVvv,r_a,optimize=True)
                   temp_2_2[a:a+k] += lib.einsum('ldyb,b->lyd', eris_OVvv,r_a,optimize=True)

                   temp[a:a+k]  -= lib.einsum('lbyd,b->lyd',eris_OVvv,r_b,optimize=True)
                   del eris_OVvv
                   a += k

               temp_new = -lib.einsum('lyd,ildx->ixy',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_new.reshape(-1)
               del temp
               del temp_new
               del temp_1
               del temp_1_new
               del temp_2
               del temp_2_new

               if isinstance(eris.ovVV, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a
               a = 0
               r_bab_t = r_bab.reshape(nocc_b*nvir_a,-1)
               temp = np.ascontiguousarray(t2_1_ab.transpose(0,3,1,2)).reshape(nocc_a*nvir_b,nocc_b*nvir_a)
               temp_2 = np.dot(temp,r_bab_t).reshape(nocc_a,nvir_b,nvir_b)
               del temp 
               temp_2 = np.ascontiguousarray(temp_2.transpose(0,2,1))
               temp_2_new = -lib.einsum('ljzd,jzw->lwd',t2_1_ab,r_bab,optimize=True)


               temp_new_1 = np.zeros_like(r_aba)
               temp_new_1 = lib.einsum('ljdw,jzw->ldz',t2_1_ab,r_bbb_u,optimize=True)
               temp_new_1 += lib.einsum('jlwd,jzw->ldz',t2_1_a,r_aba,optimize=True)

               temp_new_2 = np.zeros_like(r_bab)
               temp_new_2 = -lib.einsum('ljdz,jwz->lwd',t2_1_ab,r_bbb_u,optimize=True)
               temp_new_2 += -lib.einsum('jlzd,jwz->lwd',t2_1_a,r_aba,optimize=True)

               temp_2_3 = np.zeros((nocc_a,nvir_b,nvir_a))
               temp_2_4 = np.zeros((nocc_a,nvir_b,nvir_a))

               temp = np.zeros((nocc_a,nvir_b,nvir_b))
               for p in range(0,nocc_a,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovVV = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                   else :
                       eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                   k = eris_ovVV.shape[0]
                   s[s_a:f_a] -= 0.5*lib.einsum('lzd,lazd->a',temp_2[a:a+k],eris_ovVV,optimize=True)

                   s[s_a:f_a] += 0.5*lib.einsum('lwd,lawd->a',temp_2_new[a:a+k],eris_ovVV,optimize=True)

                   s[s_b:f_b] += 0.5*np.einsum('ldz,ldza->a',temp_new_1[a:a+k],eris_ovVV)
                   s[s_b:f_b] -= 0.5*np.einsum('lwd,ldwa->a',temp_new_2[a:a+k],eris_ovVV)

                   eris_ovVV = eris_ovVV.reshape(-1, nvir_a, nvir_b, nvir_b)

                   temp_2_3[a:a+k] += lib.einsum('ldxb,b->lxd', eris_ovVV,r_b,optimize=True)
                   temp_2_4[a:a+k] += lib.einsum('ldyb,b->lyd', eris_ovVV,r_b,optimize=True)

                   temp[a:a+k]  -= lib.einsum('lbyd,b->lyd',eris_ovVV,r_a,optimize=True)
                   del eris_ovVV
                   a += k

               temp = -lib.einsum('lyd,lixd->ixy',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp.reshape(-1)
               del temp
               del temp_2
               del temp_2_new
               del temp_new_1
               del temp_new_2

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               temp = lib.einsum('b,lbmi->lmi',r_a,eris_ovoo)
               temp -= lib.einsum('b,mbli->lmi',r_a,eris_ovoo)
               s[s_aaa:f_aaa] += 0.5*lib.einsum('lmi,lmp->ip',temp, t2_1_a_t, optimize=True).reshape(-1)

               temp_1 = lib.einsum('b,lbmi->lmi',r_a,eris_ovOO)
               s[s_bab:f_bab] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               temp = lib.einsum('b,lbmi->lmi',r_b,eris_OVOO)
               temp -= lib.einsum('b,mbli->lmi',r_b,eris_OVOO)
               s[s_bbb:f_bbb] += 0.5*lib.einsum('lmi,lmp->ip',temp, t2_1_b_t, optimize=True).reshape(-1)

               temp_1 = lib.einsum('b,lbmi->mli',r_b,eris_OVoo)
               s[s_aba:f_aba] += lib.einsum('mli,mlyx->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_1_1,t2_1_a,optimize=True)
               temp += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = lib.einsum('lyd,ilxd->ixy',temp_1_2,t2_1_a,optimize=True)
               temp += lib.einsum('lyd,ilxd->ixy',temp_2_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1_ab,optimize=True)
               temp  += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_b,optimize=True)
               temp += lib.einsum('lxd,lidy->ixy',temp_2_3,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = lib.einsum('lyd,ilxd->ixy',temp_1_4,t2_1_b,optimize=True)
               temp += lib.einsum('lyd,lidx->ixy',temp_2_4,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_ab,optimize=True)
               temp  += lib.einsum('lxd,ilyd->ixy',temp_2_3,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)
               cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)

        return s
        del temp_1_1
        del temp_1_2
        del temp_1_3
        del temp_1_4
        del temp_2_1
        del temp_2_2
        del temp_2_3
        del temp_2_4

    return sigma_

def ip_adc_matvec(adc, M_ij=None, eris=None, cvs=False, fc_bool=True, energy_thresh=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    imaginary_shift = adc.imaginary_shift
    if energy_thresh is None:
        energy_thresh = adc.energy_thresh
    ncore_cvs = adc.ncore_cvs
 
    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]#.astype(complex) 
    e_occ_b = adc.mo_energy_b[:nocc_b]#.astype(complex)
    e_vir_a = adc.mo_energy_a[nocc_a:]#.astype(complex)
    e_vir_b = adc.mo_energy_b[nocc_b:]#.astype(complex)
    
    #if cvs is False:
    #e_vir_a = complex_shift(e_vir_a, energy_thresh, imaginary_shift)
    #e_vir_b = complex_shift(e_vir_b, energy_thresh, imaginary_shift)
    idx_a_thresh = np.argwhere(e_vir_a >= energy_thresh)  
    idx_b_thresh = np.argwhere(e_vir_b >= energy_thresh)  
    
    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    ### applying imaginary shift to vve elements ###
    #D_n_a[idx_a_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    ### applying imaginary shift to vve elements ###
    #D_n_b[idx_b_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_n_bab = D_n_bab.reshape((nvir_b,nocc_b,nocc_a))
    ### applying imaginary shift to vve elements ###
    #D_n_bab[idx_b_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_bab = D_n_bab.reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_n_aba = D_n_aba.reshape((nvir_a,nocc_a,nocc_b))
    ### applying imaginary shift to vve elements ###
    #D_n_aba[idx_a_thresh,ncore_cvs:,ncore_cvs:] -= imaginary_shift
    #################
    D_aij_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    if M_ij is None:
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij
    M_ij_a = M_ij_a#.astype(complex)
    M_ij_b = M_ij_b#.astype(complex)
    #@profile
    #Calculate sigma vector
    def sigma_(r):
        #r = np.ones(dim)
 
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        if cvs is True:
            r = cvs_projector(adc, r)
        else:
            print("DEBUG vve energy_thresh: ", energy_thresh)
            #exit()
            r = vve_projector(adc, r, energy_thresh=energy_thresh)

        #s = np.zeros((dim), dtype=complex)
        s = np.zeros(dim)

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_aaa = r_aaa.reshape(nvir_a,-1)
        r_bbb = r_bbb.reshape(nvir_b,-1)

        r_aaa_u = None
        #r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a), dtype=complex)
        r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
        r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
        r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

        r_bbb_u = None
        #r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b), dtype=complex)
        r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
        r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
        r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

        r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
        r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

############ ADC(2) ij block ############################
         
        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)

############# ADC(2) i - kja block #########################
        s[s_a:f_a] += 0.5*lib.einsum('jaki,ajk->i', eris_ovoo, r_aaa_u, optimize = True)
        s[s_a:f_a] -= 0.5*lib.einsum('kaji,ajk->i', eris_ovoo, r_aaa_u, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo, r_bab, optimize = True)

        s[s_b:f_b] += 0.5*lib.einsum('jaki,ajk->i', eris_OVOO, r_bbb_u, optimize = True)
        s[s_b:f_b] -= 0.5*lib.einsum('kaji,ajk->i', eris_OVOO, r_bbb_u, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO, r_aba, optimize = True)

############## ADC(2) ajk - i block ############################

        temp = lib.einsum('jaki,i->ajk', eris_ovoo, r_a, optimize = True)
        temp -= lib.einsum('kaji,i->ajk', eris_ovoo, r_a, optimize = True)
        s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
        s[s_bab:f_bab] += lib.einsum('jaik,i->ajk', eris_OVoo, r_a, optimize = True).reshape(-1)
        s[s_aba:f_aba] += lib.einsum('jaki,i->ajk', eris_ovOO, r_b, optimize = True).reshape(-1)
        temp = lib.einsum('jaki,i->ajk', eris_OVOO, r_b, optimize = True)
        temp -= lib.einsum('kaji,i->ajk', eris_OVOO, r_b, optimize = True)
        s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
        
############ ADC(2) ajk - bil block ############################
        r_aaa = r_aaa.reshape(-1)
        r_bbb = r_bbb.reshape(-1)
        
        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb
       
############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               eris_oooo = eris.oooo
               eris_OOOO = eris.OOOO
               eris_ooOO = eris.ooOO
               eris_oovv = eris.oovv
               eris_OOVV = eris.OOVV
               eris_ooVV = eris.ooVV
               eris_OOvv = eris.OOvv
               eris_ovvo = eris.ovvo
               eris_OVVO = eris.OVVO
               eris_ovVO = eris.ovVO
               eris_OVvo = eris.OVvo

               #r_aaa = r_aaa.reshape(nvir_a,-1)
               #r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               #r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
               #r_bbb = r_bbb.reshape(nvir_b,-1)
               #
               #r_aaa_u = None
               #r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               #r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               #r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()
               #
               #r_bbb_u = None
               #r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               #r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               #r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()
               
               temp = 0.5*lib.einsum('jlki,ail->ajk',eris_oooo,r_aaa_u ,optimize = True)
               temp -= 0.5*lib.einsum('jikl,ail->ajk',eris_oooo,r_aaa_u ,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
             
               temp = 0.5*lib.einsum('jlki,ail->ajk',eris_OOOO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jikl,ail->ajk',eris_OOOO,r_bbb_u,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
                
               #s[s_bab:f_bab] -= 0.5*lib.einsum('kijl,ali->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= lib.einsum('klji,ail->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
                
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jlki,ali->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= lib.einsum('jikl,ail->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               """
               # Einsum paths testing 
               path_oovv = np.einsum_path('klba,bjl->ajk',eris_oovv,r_aaa_u,optimize = True)
               path_ovvo = np.einsum_path('kabl,bjl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               print("path info for: ", adc.method_type)
               print("path info for 'klba,bjl->ajk',eris_oovv,r_aaa_u")
               print(path_oovv[0])
               print(path_oovv[1])
               print("path info for 'kabl,bjl->ajk',eris_ovvo,r_aaa_u")
               print(path_ovvo[0])
               print(path_ovvo[1])
               """ 
               temp = 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('kabl,bjl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5* lib.einsum('kabl,blj->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
                
               s[s_bab:f_bab] += 0.5*lib.einsum('klba,bjl->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)
               
               temp_1 = 0.5*lib.einsum('klba,bjl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp_1 -= 0.5*lib.einsum('kabl,bjl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp_1 += 0.5*lib.einsum('kabl,blj->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('klba,bjl->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)
                
               temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] +=  0.5*lib.einsum('jlba,blk->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -=  0.5*lib.einsum('jabl,blk->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)
               
               temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
                
               s[s_aba:f_aba] += 0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)
                
               temp = -0.5*lib.einsum('kiba,bij->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] += 0.5*lib.einsum('kiba,bji->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)
               
               temp = -0.5*lib.einsum('kiba,bij->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('kiba,bji->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)
               
               temp = 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] += 0.5*lib.einsum('jiba,bik->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)
               
               temp = 0.5*lib.einsum('jiba,bik->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo
               eris_OVOO = eris.OVOO
               eris_ovOO = eris.ovOO
               eris_OVoo = eris.OVoo

################ ADC(3) i - kja and ajk - i block ############################
               t2_1_a = adc.t2[0][0][:]
               t2_1_ab = adc.t2[0][1][:]
               
               if isinstance(eris.ovvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a

               a = 0
               #temp_singles = np.zeros((nocc_a), dtype=complex)
               #temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a), dtype=complex)
               temp_singles = np.zeros((nocc_a))
               temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a))
               r_aaa = r_aaa.reshape(nvir_a,-1)
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               temp_1 = lib.einsum('pbc,ap->abc',t2_1_a_t,r_aaa, optimize=True)
               for p in range(0,nocc_a,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                   k = eris_ovvv.shape[0]
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,ibac->i',temp_1, eris_ovvv, optimize=True)

                   temp_doubles += lib.einsum('i,icab->bca',r_a[a:a+k],eris_ovvv,optimize=True)
                   temp_doubles -= lib.einsum('i,ibac->bca',r_a[a:a+k],eris_ovvv,optimize=True)
                   del eris_ovvv
                   a += k
               
               s[s_a:f_a] += temp_singles
               s[s_aaa:f_aaa] += 0.5*lib.einsum('bca,pbc->ap',temp_doubles,t2_1_a_t,optimize=True).reshape(-1)
               del temp_singles
               del temp_doubles
                 
               #r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               #r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               #r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

               #r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
               # Bug is probably here?
               temp = lib.einsum('jlab,ajk->blk',t2_1_a,r_aaa_u,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
                
               temp_1 = lib.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovOO,optimize=True)
                
               temp = -lib.einsum('klab,akj->blj',t2_1_a,r_aaa_u,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)

               temp_1 = -lib.einsum('klab,akj->blj',t2_1_a,r_aba,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovOO,optimize=True)

               temp_1 = lib.einsum('i,lbik->kbl',r_a, eris_ovoo)
               temp_1 -= lib.einsum('i,iblk->kbl',r_a, eris_ovoo)

               temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_a,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp_2  = lib.einsum('i,lbik->kbl',r_b,eris_ovOO)
               temp = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = lib.einsum('i,lbij->jbl',r_a, eris_ovoo)
               temp_1 -= lib.einsum('i,iblj->jbl',r_a, eris_ovoo)

               temp  = lib.einsum('jbl,klab->ajk',temp_1,t2_1_a,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               del t2_1_a
                
               t2_1_b = adc.t2[0][2][:]
                
               if isinstance(eris.OVVV, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_b
               a = 0
               #temp_singles = np.zeros((nocc_b), dtype=complex)
               #temp_doubles = np.zeros((nvir_b, nvir_b, nvir_b), dtype=complex)
               temp_singles = np.zeros((nocc_b))
               temp_doubles = np.zeros((nvir_b, nvir_b, nvir_b))
               r_bbb = r_bbb.reshape(nvir_b,-1)
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               temp_1 = lib.einsum('pbc,ap->abc',t2_1_b_t,r_bbb, optimize=True)
               for p in range(0,nocc_b,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                   else :
                       eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                   k = eris_OVVV.shape[0]
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,icab->i',temp_1, eris_OVVV, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,ibac->i',temp_1, eris_OVVV, optimize=True)

                   temp_doubles += lib.einsum('i,icab->bca',r_b[a:a+k],eris_OVVV,optimize=True)
                   temp_doubles -= lib.einsum('i,ibac->bca',r_b[a:a+k],eris_OVVV,optimize=True)
                   del eris_OVVV
                   a += k

               s[s_b:f_b] += temp_singles
               s[s_bbb:f_bbb] += 0.5*lib.einsum('bca,pbc->ap',temp_doubles,t2_1_b_t,optimize=True).reshape(-1)
               del temp_singles
               del temp_doubles

               #r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               #r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               #r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

               #r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

               temp_1 = lib.einsum('jlab,ajk->blk',t2_1_b,r_bab,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_OVoo,optimize=True)

               temp = lib.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)

               s[s_b:f_b] += 0.5*lib.einsum('blk,lbik->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_OVOO,optimize=True)

               temp_1 = -lib.einsum('klab,akj->blj',t2_1_b,r_bab,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_OVoo,optimize=True)

               temp = -lib.einsum('klab,akj->blj',t2_1_b,r_bbb_u,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,iblj->i',temp,eris_OVOO,optimize=True)

               temp_2  = lib.einsum('i,lbik->kbl',r_a,eris_OVoo)
               temp = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = lib.einsum('i,lbik->kbl',r_b, eris_OVOO)
               temp_1 -= lib.einsum('i,iblk->kbl',r_b, eris_OVOO)

               temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_b,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp_1 = lib.einsum('i,lbij->jbl',r_b, eris_OVOO)
               temp_1 -= lib.einsum('i,iblj->jbl',r_b, eris_OVOO)

               temp  = lib.einsum('jbl,klab->ajk',temp_1,t2_1_b,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)
               del t2_1_b
                 
               if isinstance(eris.ovVV, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a
               a = 0
               temp_1 = lib.einsum('kjcb,ajk->abc',t2_1_ab,r_bab, optimize=True)
               #temp_2 = np.zeros((nvir_a, nvir_b, nvir_b), dtype=complex)
               temp_2 = np.zeros((nvir_a, nvir_b, nvir_b))
               for p in range(0,nocc_a,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                   else :
                       eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                   k = eris_ovVV.shape[0]

                   s[s_a:f_a][a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovVV, optimize=True)

                   temp_2 += lib.einsum('i,icab->cba',r_a[a:a+k],eris_ovVV,optimize=True)
                   del eris_ovVV
                   a += k

               s[s_bab:f_bab] += lib.einsum('cba,kjcb->ajk',temp_2, t2_1_ab, optimize=True).reshape(-1)
               del temp_1
               del temp_2
               
               if isinstance(eris.OVvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_b

               a = 0
               temp_1 = lib.einsum('jkbc,ajk->abc',t2_1_ab,r_aba, optimize=True)
               #temp_2 = np.zeros((nvir_a, nvir_b, nvir_a), dtype=complex)
               temp_2 = np.zeros((nvir_a, nvir_b, nvir_a))
               for p in range(0,nocc_b,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                   else :
                       eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                   k = eris_OVvv.shape[0]
                   s[s_b:f_b][a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_OVvv, optimize=True)

                   temp_2 += lib.einsum('i,icab->bca',r_b[a:a+k],eris_OVvv,optimize=True)
                   del eris_OVvv
                   a += k

               s[s_aba:f_aba] += lib.einsum('bca,jkbc->ajk',temp_2, t2_1_ab, optimize=True).reshape(-1)
               del temp_1
               del temp_2
               
               temp = lib.einsum('ljba,ajk->blk',t2_1_ab,r_bab,optimize=True)
               temp_1 = lib.einsum('jlab,ajk->blk',t2_1_ab,r_aaa_u,optimize=True)
               temp_2 = lib.einsum('jlba,akj->blk',t2_1_ab,r_bab, optimize=True)

               s[s_a:f_a] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_OVoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovOO,optimize=True)

               temp = lib.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)
               temp_1 = lib.einsum('ljba,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
               temp_2 = lib.einsum('ljab,akj->blk',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] += 0.5*lib.einsum('blk,lbik->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_OVoo,optimize=True)

               temp = -lib.einsum('lkba,akj->blj',t2_1_ab,r_bab,optimize=True)
               temp_1 = -lib.einsum('klab,akj->blj',t2_1_ab,r_aaa_u,optimize=True)
               temp_2 = -lib.einsum('klba,ajk->blj',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_OVoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovOO,optimize=True)

               temp = -lib.einsum('klab,akj->blj',t2_1_ab,r_aba,optimize=True)
               temp_1 = -lib.einsum('lkba,akj->blj',t2_1_ab,r_bbb_u,optimize=True)
               temp_2 = -lib.einsum('lkab,ajk->blj',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,iblj->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_OVoo,optimize=True)
               
               temp_2 = lib.einsum('i,lbik->kbl',r_a, eris_OVoo)
               temp = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp_1  = lib.einsum('i,lbik->kbl',r_a,eris_ovoo)
               temp_1  -= lib.einsum('i,iblk->kbl',r_a,eris_ovoo)

               temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1_ab,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_2 = lib.einsum('i,lbik->kbl',r_b, eris_ovOO)

               temp = lib.einsum('kbl,ljba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp_1  = lib.einsum('i,lbik->kbl',r_b,eris_OVOO)
               temp_1  -= lib.einsum('i,iblk->kbl',r_b,eris_OVOO)

               temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_ab,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_2 = lib.einsum('i,lbij->jbl',r_a, eris_OVoo)

               temp = lib.einsum('jbl,klab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp  = -lib.einsum('i,iblj->jbl',r_a,eris_ovOO,optimize=True)
               temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_2 = lib.einsum('i,lbij->jbl',r_b, eris_ovOO)
               temp = lib.einsum('jbl,lkba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp  = -lib.einsum('i,iblj->jbl',r_b,eris_OVoo,optimize=True)
               temp_1 = -lib.einsum('jbl,lkab->ajk',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)
               
               del t2_1_ab
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        if cvs is True:
            s = cvs_projector(adc, s)
        else:
            s = vve_projector(adc, s, energy_thresh=energy_thresh)
        """
        print('s[s_a:f_a]',             np.linalg.norm(s[s_a:f_a]))
        print(np.column_stack(np.unique(np.around(s[s_a:f_a], decimals=8), return_counts=True))) 
        print('s[s_b:f_b]' ,            np.linalg.norm(s[s_b:f_b]))
        print(np.column_stack(np.unique(np.around(s[s_b:f_b], decimals=8), return_counts=True))) 
        print('s[s_aaa:f_aaa]', np.linalg.norm(s[s_aaa:f_aaa]))      
        print(np.column_stack(np.unique(np.around(s[s_aaa:f_aaa], decimals=8), return_counts=True))) 
        print('s[s_bab:f_bab]', np.linalg.norm(s[s_bab:f_bab]))
        print(np.column_stack(np.unique(np.around(s[s_bab:f_bab], decimals=8), return_counts=True))) 
        print('s[s_aba:f_aba]', np.linalg.norm(s[s_aba:f_aba]))
        print(np.column_stack(np.unique(np.around(s[s_aba:f_aba], decimals=8), return_counts=True))) 
        print('s[s_bbb:f_bbb]', np.linalg.norm(s[s_bbb:f_bbb]))
        print(np.column_stack(np.unique(np.around(s[s_bbb:f_bbb], decimals=8), return_counts=True))) 
        temp_aaa = s[s_aaa:f_aaa] 
        temp_bbb = s[s_bbb:f_bbb] 
        print("# of non zeros in aaa: ", temp_aaa[temp_aaa != 0].size)
        print("# of non zeros in bbb: ", temp_bbb[temp_bbb != 0].size)
        """
        return s

    return sigma_


def ip_adc_matvec_off(adc, M_ij=None, eris=None, cvs=False, fc_bool=True):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

    t2_1_a  = t2_1_a.astype(complex)  
    t2_1_ab = t2_1_a.astype(complex)
    t2_1_b  = t2_1_b.astype(complex)
    t1_2_a  = t1_2_a.astype(complex)
    t1_2_b  = t1_2_b.astype(complex)

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    imaginary_shift = adc.imaginary_shift
    energy_thresh = adc.energy_thresh

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a].astype(complex)
    e_occ_b = adc.mo_energy_b[:nocc_b].astype(complex)
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]
    #e_vir_a = complex_shift(e_vir_a, energy_thresh, imaginary_shift)
    #e_vir_b = complex_shift(e_vir_b, energy_thresh, imaginary_shift)

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_aij_bab = D_n_bab.reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_aij_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    if M_ij is None:
        M_ij = adc.get_imds(fc_bool)
    M_ij_a, M_ij_b = M_ij

    M_ij_a  = M_ij_a.astype(complex) 
    M_ij_b  = M_ij_b.astype(complex)

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)
        #print("input trial vec shape: ", np.array(r).shape)
        #print("expected size: ", dim)
        if cvs is True:
            #print("CVS Matvec")
            r = cvs_projector(adc, r)

        s = np.zeros((dim)).astype(complex)

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_aaa = r_aaa.reshape(nvir_a,-1)
        r_bbb = r_bbb.reshape(nvir_b,-1)

        r_aaa_u = None
        r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a)).astype(complex)
        r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
        r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

        r_bbb_u = None
        r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b)).astype(complex)
        r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
        r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

        r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
        r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

############ ADC(2) ij block ############################

        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)

############# ADC(2) i - kja block #########################
       
        s[s_a:f_a] += 0.5*lib.einsum('jaki,ajk->i', eris_ovoo, r_aaa_u, optimize = True)
        s[s_a:f_a] -= 0.5*lib.einsum('kaji,ajk->i', eris_ovoo, r_aaa_u, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo, r_bab, optimize = True)

        s[s_b:f_b] += 0.5*lib.einsum('jaki,ajk->i', eris_OVOO, r_bbb_u, optimize = True)
        s[s_b:f_b] -= 0.5*lib.einsum('kaji,ajk->i', eris_OVOO, r_bbb_u, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO, r_aba, optimize = True)

############## ADC(2) ajk - i block ############################
        
        temp = lib.einsum('jaki,i->ajk', eris_ovoo, r_a, optimize = True)
        temp -= lib.einsum('kaji,i->ajk', eris_ovoo, r_a, optimize = True)
        s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
        s[s_bab:f_bab] += lib.einsum('jaik,i->ajk', eris_OVoo, r_a, optimize = True).reshape(-1)
        s[s_aba:f_aba] += lib.einsum('jaki,i->ajk', eris_ovOO, r_b, optimize = True).reshape(-1)
        temp = lib.einsum('jaki,i->ajk', eris_OVOO, r_b, optimize = True)
        temp -= lib.einsum('kaji,i->ajk', eris_OVOO, r_b, optimize = True)
        s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
        
############ ADC(2) ajk - bil block ############################
        
        r_aaa = r_aaa.reshape(-1)
        r_bbb = r_bbb.reshape(-1)

        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb
        

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

               eris_oooo = eris.oooo
               eris_OOOO = eris.OOOO
               eris_ooOO = eris.ooOO
               eris_oovv = eris.oovv
               eris_OOVV = eris.OOVV
               eris_ooVV = eris.ooVV
               eris_OOvv = eris.OOvv
               eris_ovvo = eris.ovvo
               eris_OVVO = eris.OVVO
               eris_ovVO = eris.ovVO
               eris_OVvo = eris.OVvo
               
               #r_aaa = r_aaa.reshape(nvir_a,-1)
               #r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               #r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
               #r_bbb = r_bbb.reshape(nvir_b,-1)
               #
               #r_aaa_u = None
               #r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               #r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               #r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()
               #
               #r_bbb_u = None
               #r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               #r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               #r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()
               
               temp = 0.5*lib.einsum('jlki,ail->ajk',eris_oooo,r_aaa_u ,optimize = True)
               temp -= 0.5*lib.einsum('jikl,ail->ajk',eris_oooo,r_aaa_u ,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               temp = 0.5*lib.einsum('jlki,ail->ajk',eris_OOOO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jikl,ail->ajk',eris_OOOO,r_bbb_u,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
               s[s_bab:f_bab] -= 0.5*lib.einsum('kijl,ali->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*lib.einsum('klji,ail->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               
               s[s_aba:f_aba] -= 0.5*lib.einsum('jlki,ali->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jikl,ail->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               
               temp = 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('kabl,bjl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5* lib.einsum('kabl,blj->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] += 0.5*lib.einsum('klba,bjl->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)
               
               temp_1 = 0.5*lib.einsum('klba,bjl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp_1 -= 0.5*lib.einsum('kabl,bjl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp_1 += 0.5*lib.einsum('kabl,blj->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('klba,bjl->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)
               
               temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] +=  0.5*lib.einsum('jlba,blk->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -=  0.5*lib.einsum('jabl,blk->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)
               
               temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)
               
               temp = -0.5*lib.einsum('kiba,bij->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] += 0.5*lib.einsum('kiba,bji->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)
               
               temp = -0.5*lib.einsum('kiba,bij->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp += 0.5*lib.einsum('kabi,bij->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('kiba,bji->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)
               
               temp = 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r_aaa_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovVO,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               s[s_bab:f_bab] += 0.5*lib.einsum('jiba,bik->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)
               
               s[s_aba:f_aba] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)
               
               temp = 0.5*lib.einsum('jiba,bik->ajk',eris_OOVV,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVVO,r_bbb_u,optimize = True)
               temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVvo,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo
               eris_OVOO = eris.OVOO
               eris_ovOO = eris.ovOO
               eris_OVoo = eris.OVoo

################ ADC(3) i - kja and ajk - i block ############################


               if isinstance(eris.ovvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a

               a = 0
               temp_singles = np.zeros((nocc_a), dtype=complex)
               temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a), dtype=complex)
               r_aaa = r_aaa.reshape(nvir_a,-1)
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               temp_1 = lib.einsum('pbc,ap->abc',t2_1_a_t,r_aaa, optimize=True)
               for p in range(0,nocc_a,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
                   k = eris_ovvv.shape[0]
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,ibac->i',temp_1, eris_ovvv, optimize=True)

                   temp_doubles += lib.einsum('i,icab->bca',r_a[a:a+k],eris_ovvv,optimize=True)
                   temp_doubles -= lib.einsum('i,ibac->bca',r_a[a:a+k],eris_ovvv,optimize=True)
                   del eris_ovvv
                   a += k

               s[s_a:f_a] += temp_singles
               s[s_aaa:f_aaa] += 0.5*lib.einsum('bca,pbc->ap',temp_doubles,t2_1_a_t,optimize=True).reshape(-1)
               del temp_singles
               del temp_doubles

               if isinstance(eris.OVVV, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_b
               a = 0
               temp_singles = np.zeros((nocc_b), dtype=complex)
               temp_doubles = np.zeros((nvir_b, nvir_b, nvir_b), dtype=complex)
               r_bbb = r_bbb.reshape(nvir_b,-1)
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               temp_1 = lib.einsum('pbc,ap->abc',t2_1_b_t,r_bbb, optimize=True)
               for p in range(0,nocc_b,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_OVVV = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                   else :
                       eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
                   k = eris_OVVV.shape[0]
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,icab->i',temp_1, eris_OVVV, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,ibac->i',temp_1, eris_OVVV, optimize=True)

                   temp_doubles += lib.einsum('i,icab->bca',r_b[a:a+k],eris_OVVV,optimize=True)
                   temp_doubles -= lib.einsum('i,ibac->bca',r_b[a:a+k],eris_OVVV,optimize=True)
                   del eris_OVVV
                   a += k

               s[s_b:f_b] += temp_singles
               s[s_bbb:f_bbb] += 0.5*lib.einsum('bca,pbc->ap',temp_doubles,t2_1_b_t,optimize=True).reshape(-1)
               del temp_singles
               del temp_doubles

               if isinstance(eris.ovVV, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a
               a = 0
               temp_1 = lib.einsum('kjcb,ajk->abc',t2_1_ab,r_bab, optimize=True)
               temp_2 = np.zeros((nvir_a, nvir_b, nvir_b), dtype=complex)
               for p in range(0,nocc_a,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovVV = dfadc.get_ovvv_a_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                   else :
                       eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
                   k = eris_ovVV.shape[0]

                   s[s_a:f_a][a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovVV, optimize=True)

                   temp_2 += lib.einsum('i,icab->cba',r_a[a:a+k],eris_ovVV,optimize=True)
                   del eris_ovVV
                   a += k

               s[s_bab:f_bab] += lib.einsum('cba,kjcb->ajk',temp_2, t2_1_ab, optimize=True).reshape(-1)
               del temp_1
               del temp_2

               if isinstance(eris.OVvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_b

               a = 0
               temp_1 = lib.einsum('jkbc,ajk->abc',t2_1_ab,r_aba, optimize=True)
               temp_2 = np.zeros((nvir_a, nvir_b, nvir_a), dtype=complex)
               for p in range(0,nocc_b,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_OVvv = dfadc.get_ovvv_a_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                   else :
                       eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
                   k = eris_OVvv.shape[0]
                   s[s_b:f_b][a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_OVvv, optimize=True)

                   temp_2 += lib.einsum('i,icab->bca',r_b[a:a+k],eris_OVvv,optimize=True)
                   del eris_OVvv
                   a += k

               s[s_aba:f_aba] += lib.einsum('bca,jkbc->ajk',temp_2, t2_1_ab, optimize=True).reshape(-1)
               del temp_1
               del temp_2

               #r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a), dtype=complex)
               #r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               #r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

               #r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b), dtype=complex)
               #r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               #r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

               #r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               #r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

               temp = np.zeros_like(r_bab, dtype=complex)
               temp = lib.einsum('jlab,ajk->blk',t2_1_a,r_aaa_u,optimize=True)
               temp += lib.einsum('ljba,ajk->blk',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab, dtype=complex)
               temp_1 = lib.einsum('jlab,ajk->blk',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += lib.einsum('jlab,ajk->blk',t2_1_b,r_bab,optimize=True)

               temp_2 = lib.einsum('jlba,akj->blk',t2_1_ab,r_bab, optimize=True)

               s[s_a:f_a] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_OVoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovOO,optimize=True)

               temp = np.zeros_like(r_aba, dtype=complex)
               temp = lib.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)
               temp += lib.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_aba, dtype=complex)
               temp_1 = lib.einsum('ljba,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += lib.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)

               temp_2 = lib.einsum('ljab,akj->blk',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] += 0.5*lib.einsum('blk,lbik->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_OVoo,optimize=True)

               temp = np.zeros_like(r_bab, dtype=complex)
               temp = -lib.einsum('klab,akj->blj',t2_1_a,r_aaa_u,optimize=True)
               temp -= lib.einsum('lkba,akj->blj',t2_1_ab,r_bab,optimize=True)

               temp_1 = np.zeros_like(r_bab, dtype=complex)
               temp_1 = -lib.einsum('klab,akj->blj',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 -= lib.einsum('klab,akj->blj',t2_1_b,r_bab,optimize=True)

               temp_2 = -lib.einsum('klba,ajk->blj',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_OVoo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovOO,optimize=True)

               temp = np.zeros_like(r_aba, dtype=complex)
               temp = -lib.einsum('klab,akj->blj',t2_1_b,r_bbb_u,optimize=True)
               temp -= lib.einsum('klab,akj->blj',t2_1_ab,r_aba,optimize=True)

               temp_1 = np.zeros_like(r_bab, dtype=complex)
               temp_1 = -lib.einsum('lkba,akj->blj',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 -= lib.einsum('klab,akj->blj',t2_1_a,r_aba,optimize=True)

               temp_2 = -lib.einsum('lkab,ajk->blj',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,iblj->i',temp,eris_OVOO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovOO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_OVoo,optimize=True)

               temp_1 = lib.einsum('i,lbik->kbl',r_a, eris_ovoo)
               temp_1 -= lib.einsum('i,iblk->kbl',r_a, eris_ovoo)
               temp_2 = lib.einsum('i,lbik->kbl',r_a, eris_OVoo)

               temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_a,optimize=True)
               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp_1  = lib.einsum('i,lbik->kbl',r_a,eris_ovoo)
               temp_1  -= lib.einsum('i,iblk->kbl',r_a,eris_ovoo)
               temp_2  = lib.einsum('i,lbik->kbl',r_a,eris_OVoo)

               temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1_ab,optimize=True)
               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = lib.einsum('i,lbik->kbl',r_b, eris_OVOO)
               temp_1 -= lib.einsum('i,iblk->kbl',r_b, eris_OVOO)
               temp_2 = lib.einsum('i,lbik->kbl',r_b, eris_ovOO)

               temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_b,optimize=True)
               temp += lib.einsum('kbl,ljba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp_1  = lib.einsum('i,lbik->kbl',r_b,eris_OVOO)
               temp_1  -= lib.einsum('i,iblk->kbl',r_b,eris_OVOO)
               temp_2  = lib.einsum('i,lbik->kbl',r_b,eris_ovOO)

               temp  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_ab,optimize=True)
               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = lib.einsum('i,lbij->jbl',r_a, eris_ovoo)
               temp_1 -= lib.einsum('i,iblj->jbl',r_a, eris_ovoo)
               temp_2 = lib.einsum('i,lbij->jbl',r_a, eris_OVoo)

               temp  = lib.einsum('jbl,klab->ajk',temp_1,t2_1_a,optimize=True)
               temp += lib.einsum('jbl,klab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)

               temp  = -lib.einsum('i,iblj->jbl',r_a,eris_ovOO,optimize=True)
               temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_1 = lib.einsum('i,lbij->jbl',r_b, eris_OVOO)
               temp_1 -= lib.einsum('i,iblj->jbl',r_b, eris_OVOO)
               temp_2 = lib.einsum('i,lbij->jbl',r_b, eris_ovOO)

               temp  = lib.einsum('jbl,klab->ajk',temp_1,t2_1_b,optimize=True)
               temp += lib.einsum('jbl,lkba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)

               temp  = -lib.einsum('i,iblj->jbl',r_b,eris_OVoo,optimize=True)
               temp_1 = -lib.einsum('jbl,lkab->ajk',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)
               cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)

        s *= -1.0

        if cvs is True:
            #print("CVS Matvec-sigma")
            s = cvs_projector(adc, s)

        return s

    return sigma_


def ea_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a* (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a* nvir_b
    n_doubles_aba = nocc_a * nvir_b* nvir_a
    n_doubles_bbb = nvir_b* (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    T = np.zeros((dim))

######## spin = alpha  ############################################
    if spin=="alpha":
######## ADC(2) part  ############################################

        if orb < nocc_a:

            T[s_a:f_a] = -t1_2_a[orb,:]

            t2_1_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(1,0,2,3).copy()

            T[s_aaa:f_aaa] += t2_1_t[:,orb,:].reshape(-1)
            T[s_bab:f_bab] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :

            T[s_a:f_a] += idn_vir_a[(orb-nocc_a), :]
            T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)
######## ADC(3) 2p-1h  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb < nocc_a:

                t2_2_t = t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(1,0,2,3).copy()

                T[s_aaa:f_aaa] += t2_2_t[:,orb,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(3) 1p part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:

                T[s_a:f_a] += 0.5*lib.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_2_a.T,optimize = True)
                T[s_a:f_a] -= 0.5*lib.einsum('kac,ck->a',t2_1_ab[orb,:,:,:], t1_2_b.T,optimize = True)

                T[s_a:f_a] -= t1_3_a[orb,:]

            else:

                T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize = True)

                T[s_a:f_a] -= 0.25*lib.einsum('klac,klc->a',t2_1_a, t2_2_a[:,:,(orb-nocc_a),:],optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('klac,klc->a',t2_1_ab, t2_2_ab[:,:,(orb-nocc_a),:],optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('lkac,lkc->a',t2_1_ab, t2_2_ab[:,:,(orb-nocc_a),:],optimize = True)

######### spin = beta  ############################################
    else:
######## ADC(2) part  ############################################


        if orb < nocc_b:

            T[s_b:f_b] = -t1_2_b[orb,:]

            t2_1_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(0,1,3,2).copy()

            T[s_bbb:f_bbb] += t2_1_t[:,orb,:].reshape(-1)
            T[s_aba:f_aba] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :

            T[s_b:f_b] += idn_vir_b[(orb-nocc_b), :]
            T[s_b:f_b] -= 0.25*lib.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_1_b, optimize = True)
            T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)
            T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)

######### ADC(3) 2p-1h part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb < nocc_b:

                t2_2_t = t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(0,1,3,2).copy()

                T[s_bbb:f_bbb] += t2_2_t[:,orb,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(2) 1p part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                T[s_b:f_b] += 0.5*lib.einsum('kac,ck->a',t2_1_b[:,orb,:,:], t1_2_b.T,optimize = True)
                T[s_b:f_b] -= 0.5*lib.einsum('kca,ck->a',t2_1_ab[:,orb,:,:], t1_2_a.T,optimize = True)

                T[s_b:f_b] -= t1_3_b[orb,:]

            else:

                T[s_b:f_b] -= 0.25*lib.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize = True)

                T[s_b:f_b] -= 0.25*lib.einsum('klac,klc->a',t2_1_b, t2_2_b[:,:,(orb-nocc_b),:],optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('lkca,lkc->a',t2_1_ab, t2_2_ab[:,:,:,(orb-nocc_b)],optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('klca,klc->a',t2_1_ab, t2_2_ab[:,:,:,(orb-nocc_b)],optimize = True)

    return T


def ip_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1_a, t2_1_ab, t2_1_b = adc.t2[0]
    t1_2_a, t1_2_b = adc.t1[0]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    imaginary_shift = adc.imaginary_shift
    energy_thresh = adc.energy_thresh

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a* nocc_b
    n_doubles_aba = nvir_a * nocc_b* nocc_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    #T = np.zeros((dim), dtype=complex)
    T = np.zeros((dim))

######## spin = alpha  ############################################
    if spin=="alpha":
######## ADC(2) 1h part  ############################################

        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :]
            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
        else :
            T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]

######## ADC(2) 2h-1p  part  ############################################

            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
            t2_1_t_a = t2_1_t.transpose(2,1,0).copy()
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0).copy()

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
                t2_2_t_a = t2_2_t.transpose(2,1,0).copy()
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0).copy()

                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:
                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)

                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a, t2_2_a[:,orb,:,:],optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True)
            else:
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a[:,:,(orb-nocc_a),:], t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab[:,:,(orb-nocc_a),:], t1_2_b,optimize = True)
                T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]

######## spin = beta  ############################################
    else:
######## ADC(2) 1h part  ############################################

        if orb < nocc_b:
            T[s_b:f_b] = idn_occ_b[orb, :]
            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_1_b, optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
        else :
            T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]

######## ADC(2) 2h-1p part  ############################################

            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
            t2_1_t_b = t2_1_t.transpose(2,1,0).copy()
            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1).copy()

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = adc.t2[1]

            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
                t2_2_t_b = t2_2_t.transpose(2,1,0).copy()

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1).copy()

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:
                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_2_ab, optimize = True)

                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b, t2_2_b[:,orb,:,:],optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab, t2_2_ab[:,orb,:,:],optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab, t2_2_ab[:,orb,:,:],optimize = True)
            else:
                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b[:,:,(orb-nocc_b),:], t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab[:,:,:,(orb-nocc_b)], t1_2_a,optimize = True)
                T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]

    return T


def get_trans_moments(adc):

    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b

    T_a = []
    T_b = []

    for orb in range(nmo_a):

            T_aa = adc.compute_trans_moments(orb, spin = "alpha")
            T_a.append(T_aa)

    for orb in range(nmo_b):

            T_bb = adc.compute_trans_moments(orb, spin = "beta")
            T_b.append(T_bb)
    
    return (T_a, T_b)

   
def get_spec_factors(adc, T, U, nroots=1):

    T_a = T[0]
    T_b = T[1]

    #T_a = np.array(T_a, dtype=complex)
    X_a = np.dot(T_a, U.T).reshape(-1,nroots) # changed U.T to U.transpose()
    del T_a
    #T_b = np.array(T_b, dtype=complex)
    X_b = np.dot(T_b, U.T).reshape(-1,nroots)
    del T_b

    #P = lib.einsum("pi,pi->i", X_a, X_a.conj())
    #P += lib.einsum("pi,pi->i", X_b, X_b.conj())
    P = lib.einsum("pi,pi->i", X_a, X_a.conj())
    P += lib.einsum("pi,pi->i", X_b, X_b.conj())
    X = (X_a, X_b)
    
    # return P      # Originally only P was returned, not P and X 
    return P, X



class UADCEA(UADC):
    '''unrestricted ADC for EA energies and spectroscopic amplitudes

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).
        conv_tol : float
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
	nroots : int
	    Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.UADC(mf).run()
            >>> myadcea = adc.UADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''
    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self.nocc_a = adc._nocc[0]
        self.nocc_b = adc._nocc[1]
        self.nvir_a = adc._nvir[0]
        self.nvir_b = adc._nvir[1]
        self.mo_coeff = adc.mo_coeff
        self.mo_energy_a = adc.mo_energy_a
        self.mo_energy_b = adc.mo_energy_b
        self.nmo_a = adc._nmo[0]
        self.nmo_b = adc._nmo[1]
        self.mol = adc.mol
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_mpn_energy = True
        self.fc_bool = adc.fc_bool
        self.nfc_orb = adc.nfc_orb
        self.ncore_cvs = adc.ncore_cvs
        self.cvs_npick = adc.cvs_npick
        self.lanczos_space = adc.lanczos_space
        self.imaginary_shift = adc.imaginary_shift
        self.energy_thresh = adc.energy_thresh
        self.es_mom_thresh = adc.es_mom_thresh
        self.cvs_type = adc.cvs_type
        self.oneshot_mom = adc.oneshot_mom
        self.oneshot_cvs = adc.oneshot_cvs
        self.cvs_edge = adc.cvs_edge
        self.oneshot_no_vve = adc.oneshot_no_vve
        self.max_nmom = adc.max_nmom

        keys = set(('conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)
    
    kernel = kernel
    get_imds = get_imds_ea
    matvec = ea_adc_matvec
    get_diag = ea_adc_diag
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors
    
    def get_init_guess(self, nroots=1, diag=None, ascending = True):
       if diag is None :
           diag = self.ea_adc_diag()
       idx = None
       if ascending:
           idx = np.argsort(diag)
       else:
           idx = np.argsort(diag)[::-1]
       guess = np.zeros((diag.shape[0], nroots))
       min_shape = min(diag.shape[0], nroots)
       guess[:min_shape,:min_shape] = np.identity(min_shape)
       g = np.zeros((diag.shape[0], nroots))
       g[idx] = guess.copy()
       guess = []
       for p in range(g.shape[1]):
           guess.append(g[:,p])
       return guess
    
    def gen_matvec(self, imds=None, eris=None, fc_bool=True, cvs=True):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag


class UADCIP(UADC):
    '''unrestricted ADC for IP energies and spectroscopic amplitudes

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).
        conv_tol : float
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
	nroots : int
	    Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.UADC(mf).run()
            >>> myadcip = adc.UADC(myadc).run()

    Saved results

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''
    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self._nocc = adc._nocc
        self._nvir = adc._nvir
        self.nocc_a = adc._nocc[0]
        self.nocc_b = adc._nocc[1]
        self.nvir_a = adc._nvir[0]
        self.nvir_b = adc._nvir[1]
        self.mo_coeff = adc.mo_coeff
        self.mo_energy_a = adc.mo_energy_a
        self.mo_energy_b = adc.mo_energy_b
        self.nmo_a = adc._nmo[0]
        self.nmo_b = adc._nmo[1]
        self.mol = adc.mol
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_mpn_energy = True
        self.fc_bool = adc.fc_bool
        self.nfc_orb = adc.nfc_orb
        self.ncore_cvs = adc.ncore_cvs
        self.cvs_npick = adc.cvs_npick
        self.lanczos_space = adc.lanczos_space
        self.imaginary_shift = adc.imaginary_shift
        self.energy_thresh = adc.energy_thresh
        self.es_mom_thresh = adc.es_mom_thresh
        self.cvs_type = adc.cvs_type
        self.oneshot_mom = adc.oneshot_mom
        self.oneshot_cvs = adc.oneshot_cvs
        self.cvs_edge = adc.cvs_edge
        self.oneshot_no_vve = adc.oneshot_no_vve
        self.max_nmom = adc.max_nmom

        keys = set(('conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.ip_adc_diag()
        idx = None
        if ascending:
            #idx = np.argsort(diag**2)
            diag_r_sq = np.real(diag)**2
            diag_im_sq = np.imag(diag)**2
            diag_sq = diag_r_sq + 1j*diag_im_sq
            idx = np.argsort(diag)
        else:
            idx = np.argsort(diag)[::-1]
        #print("idx: ", idx)
        guess = np.zeros((diag.shape[0], nroots))
        min_shape = min(diag.shape[0], nroots)
        guess[:min_shape,:min_shape] = np.identity(min_shape)
        g = np.zeros((diag.shape[0], nroots))
        g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess

    def gen_matvec(self, imds=None, eris=None, fc_bool=True, cvs=False, energy_thresh=None):
        if imds is None: imds = self.get_imds(eris, fc_bool)
        diag = self.get_diag(imds,eris, cvs, fc_bool)
        matvec = self.matvec(imds, eris, cvs, fc_bool, energy_thresh=energy_thresh)
        #matvec = lambda x: self.matvec()
        return matvec, diag

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import adc

    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', ( 0., 0.    , -r/2   )],
        ['N', ( 0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr -  -0.32201692499346535)

    myadcip = UADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(2) IP energies")
    print (e[0] - 0.5434389897908212)
    print (e[1] - 0.5434389942222756)
    print (e[2] - 0.6240296265084732)

    print("ADC(2) IP spectroscopic factors")
    print (p[0] - 0.884404855445607)
    print (p[1] - 0.8844048539643351)
    print (p[2] - 0.9096460559671828)

    myadcea = UADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)
    print("ADC(2) EA energies")
    print (e[0] - 0.09617819143037348)
    print (e[1] - 0.09617819161265123)
    print (e[2] - 0.12583269048810924)

    print("ADC(2) EA spectroscopic factors")
    print (p[0] - 0.991642716974455)
    print (p[1] - 0.9916427170555298)
    print (p[2] - 0.9817184409336244)

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr - -0.31694173142858517)

    myadcip = UADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(3) IP energies")
    print (e[0] - 0.5667526838174817)
    print (e[1] - 0.5667526888293601)
    print (e[2] - 0.6099995181296374)

    print("ADC(3) IP spectroscopic factors")
    print (p[0] - 0.9086596203469742)
    print (p[1] - 0.9086596190173993)
    print (p[2] - 0.9214613318791076)

    myadcea = UADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)

    print("ADC(3) EA energies")
    print (e[0] - 0.09836545519235675)
    print (e[1] - 0.09836545535587536)
    print (e[2] - 0.12957093060942082)

    print("ADC(3) EA spectroscopic factors")
    print (p[0] - 0.9920495578633931)
    print (p[1] - 0.992049557938337)
    print (p[2] - 0.9819274864738444)

    myadc.method = "adc(2)-x"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x IP energies")
    print (e[0] - 0.5405255355249104)
    print (e[1] - 0.5405255399061982)
    print (e[2] - 0.62080267098272)
    print (e[3] - 0.620802670982715)

    myadc.method_type = "ea"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x EA energies")
    print (e[0] - 0.09530653292650725)
    print (e[1] - 0.09530653311305577)
    print (e[2] - 0.1238833077840878)
    print (e[3] - 0.12388330873739162)
