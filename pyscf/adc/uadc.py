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
from pyscf import lib, symm
from pyscf.lib import logger
from pyscf.adc import uadc_ao2mo
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
import pandas as pd

def kernel(adc, nroots=1, guess=None, eris=None, verbose=None):

    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
       raise NotImplementedError(adc.method)

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = adc.transform_integrals()

    imds = adc.get_imds(eris)
    matvec, diag = adc.gen_matvec(imds, eris)

    guess = adc.get_init_guess(nroots, diag, ascending = True)

   
    r = np.identity(diag.size)
    M = np.zeros((diag.size, diag.size))
    for i in range(diag.size):
         print("Iteration #", i)
         M[:,i] = matvec(r[:,i])
    
    print("ADC Method Type: ", adc.method_type)
    print("Hermiticity: ")
    print("Full M: ", np.linalg.norm(M - M.T))
    
    if adc.method_type == 'ip':
        nocc_a = adc._nocc[0]
        nocc_b = adc._nocc[1]
        singles = nocc_a + nocc_b
 
    if adc.method_type == 'ip-cvs':
        ncvs = adc.ncvs
        singles = 2*ncvs 

    M_i_j = M[:singles,:singles]
    M_i_ajk = M[:singles,singles:]   
    M_ajk_i = M[singles:,:singles]
    M_ajk_bli = M[singles:,singles:]
    print('M_i_j - M_i_j.T: ', np.linalg.norm(M_i_j - M_i_j.T))
    print('M_ajk_bli - M_ajk_bli.T: ', np.linalg.norm(M_ajk_bli - M_ajk_bli))
    print('M_ajk_i - M_i_ajk.T: ', np.linalg.norm(M_ajk_i - M_i_ajk.T))
    print('M norm: ', np.linalg.norm(M))
    print('M_i_j norm: ', np.linalg.norm(M_i_j))
    print('M_i_ajk norm: ', np.linalg.norm(M_i_ajk))
    print('M_ajk_i norm: ', np.linalg.norm(M_ajk_i))
    print('M_ajk_bli norm: ', np.linalg.norm(M_ajk_bli))
    print("s = M*r: ")
    print("ADC Method Type: ", adc.method_type)
    r = np.ones(diag.size)
    if adc.method_type == 'ip':
        s_matvec = matvec(r)
    else:
        s_matvec = matvec(r)
        s = np.dot(M, r)
     
    print('s_matvec norm: ', np.linalg.norm(s_matvec))
    E,eig_vec = np.linalg.eig(M)
    
    
    print("ADC Method Type: ", adc.method_type)
    print(np.column_stack(np.unique(np.around(E, decimals=8), return_counts=True))) 
    eig_vec = eig_vec[:, E != 0]
    E = E[E != 0]
    idx_E_sort = np.argsort(E)
    E = E[idx_E_sort]
    eig_vec[:, idx_E_sort]
    P, X = get_properties(adc, E.size ,eig_vec)
    #P = P[idx_E_sort]
    print("ADC Method Type: ", adc.method_type)
    print("shape of P: ", P.shape)
    print("norm of P: ", np.linalg.norm(P))
    idx_P_sort = np.argsort(P)
    P = P[idx_P_sort]
    print(np.column_stack(np.unique(np.around(P, decimals=8), return_counts=True))) 
    '''
    U = np.zeros((E.size, E.size))
    for i in range(E.size):
        idx_phase = np.argsort(-eig_vec[:,i])
        eig_vec[:,i] = eig_vec[idx_phase, i]
        if eig_vec[0,i] > 0: 
            eig_vec[:,i] *= 1
        if eig_vec[0,i] < 0: 
            eig_vec[:,i] *= -1
    '''
    E_P = (E, P)
    return E_P  

    #count = 0
    #indicies = []
    #values = []
    #for i in range(diag.size):
    #    for j in range(diag.size):
    #        if i != j and M[i,j] != 0:
    #            count += 1
    #            indicies.append([i,j])
    #            values.append(M[i,j])
    #print("number of non-zero off-diagonal elments: ", count)
    #print("indicies of non-zero off-diagonal elments: ", indicies)
    #print("non-zero off-diagonal elments: ", values)
    #return M
    #exit()
    conv, adc.E, U = lib.linalg_helper.davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space,tol_residual=adc.tol_residual)
    print("Davidson CVS energies: ")
    print("ADC Method Type: ", adc.method_type)
    print(np.column_stack(np.unique(np.around(adc.E, decimals=8), return_counts=True))) 
    exit()
    adc.U = np.array(U).T.copy()

    if adc.compute_properties:
        adc.P,adc.X = adc.get_properties(nroots)

    nfalse = np.shape(conv)[0] - np.sum(conv)

    str = ("\n*************************************************************"
           "\n            ADC calculation summary"
           "\n*************************************************************")
    logger.info(adc, str)

    if nfalse >= 1:
        logger.warn(adc, "Davidson iterations for " + str(nfalse) + " root(s) not converged\n")

    for n in range(nroots):
        print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' % (adc.method, n, adc.E[n], adc.E[n]*27.2114))
        if adc.compute_properties:
            print_string += ("|  Spec factors = %10.8f  " % adc.P[n])
        print_string += ("|  conv = %s" % conv[n])
        logger.info(adc, print_string)

    log.timer('ADC', *cput0)

    #return adc.E, adc.U, adc.P, adc.X

def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1, t2, myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_oovv = eris.oovv
    eris_OOVV = eris.OOVV
    eris_ooVV = eris.ooVV
    eris_OOvv = eris.OOvv
    eris_ovoo = eris.ovoo
    eris_OVoo = eris.OVoo
    eris_ovOO = eris.ovOO
    eris_OVOO = eris.OVOO

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO

    e_a = myadc.mo_energy_a
    e_b = myadc.mo_energy_b

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris_ovvo[:].transpose(0,3,1,2).copy()
    v2e_oovv -= eris_ovvo[:].transpose(0,3,2,1).copy()

    d_ij_a = e_a[:nocc_a][:,None] + e_a[:nocc_a]
    d_ab_a = e_a[nocc_a:][:,None] + e_a[nocc_a:]

    D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
    D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))

    t2_1_a = v2e_oovv/D2_a
    if not isinstance(eris.oooo, np.ndarray):
        t2_1_a = radc_ao2mo.write_dataset(t2_1_a)

    del v2e_oovv
    del D2_a

    v2e_OOVV = eris_OVVO[:].transpose(0,3,1,2).copy()
    v2e_OOVV -= eris_OVVO[:].transpose(0,3,2,1).copy()

    d_ij_b = e_b[:nocc_b][:,None] + e_b[:nocc_b]
    d_ab_b = e_b[nocc_b:][:,None] + e_b[nocc_b:]

    D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
    D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))

    t2_1_b = v2e_OOVV/D2_b
    if not isinstance(eris.oooo, np.ndarray):
        t2_1_b = radc_ao2mo.write_dataset(t2_1_b)
    del v2e_OOVV
    del D2_b

    v2e_oOvV = eris_ovVO[:].transpose(0,3,1,2).copy()

    d_ij_ab = e_a[:nocc_a][:,None] + e_b[:nocc_b]
    d_ab_ab = e_a[nocc_a:][:,None] + e_b[nocc_b:]

    D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
    D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))

    t2_1_ab = v2e_oOvV/D2_ab
    if not isinstance(eris.oooo, np.ndarray):
        t2_1_ab = radc_ao2mo.write_dataset(t2_1_ab)
    del v2e_oOvV
    del D2_ab

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
            eris_ovvv = dfadc.get_ovvv_spin_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
        k = eris_ovvv.shape[0]
        t1_2_a += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
        t1_2_a -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
        del eris_ovvv
        a += k

    t1_2_a -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1_a[:],optimize=True)
    t1_2_a += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1_a[:],optimize=True)

    if isinstance(eris.OVvv, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_b

    a = 0
    for p in range(0,nocc_b,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_OVvv = dfadc.get_ovvv_spin_df(myadc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
        else :
            eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
        k = eris_OVvv.shape[0]
        t1_2_a += lib.einsum('kdac,ikcd->ia',eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
        del eris_OVvv
        a += k

    if isinstance(eris.ovVV, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_a
    a = 0
    for p in range(0,nocc_a,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_ovVV = dfadc.get_ovvv_spin_df(myadc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
        else :
            eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
        k = eris_ovVV.shape[0]
        t1_2_b += lib.einsum('kdac,kidc->ia',eris_ovVV,t2_1_ab[a:a+k],optimize=True)
        del eris_ovVV
        a += k

    t1_2_a -= lib.einsum('lcki,klac->ia',eris_OVoo,t2_1_ab[:],optimize=True)
    t1_2_b -= lib.einsum('lcki,lkca->ia',eris_ovOO,t2_1_ab[:])

    if isinstance(eris.OVVV, type(None)):
        chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc_b

    a = 0
    for p in range(0,nocc_b,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_OVVV = dfadc.get_ovvv_spin_df(myadc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
        else :
            eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
        k = eris_OVVV.shape[0]
        t1_2_b += 0.5*lib.einsum('kdac,ikcd->ia',eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
        t1_2_b -= 0.5*lib.einsum('kcad,ikcd->ia',eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
        del eris_OVVV
        a += k

    t1_2_b -= 0.5*lib.einsum('lcki,klac->ia',eris_OVOO,t2_1_b[:],optimize=True)
    t1_2_b += 0.5*lib.einsum('kcli,klac->ia',eris_OVOO,t2_1_b[:],optimize=True)

    t1_2_a = t1_2_a/D1_a
    t1_2_b = t1_2_b/D1_b

    cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    t1_2 = (t1_2_a , t1_2_b)
    t2_2 = (None,)
    t1_3 = (None,)
    t2_1_vvvv = (None,)

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
            temp = np.ascontiguousarray(t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]).reshape(nocc_a*nocc_a,-1)
            t2_1_vvvv_a = np.dot(temp,eris_vvvv.T).reshape(nocc_a, nocc_a, -1)
            del eris_vvvv
        elif isinstance(eris.vvvv_p, list):
            t2_1_vvvv_a = contract_ladder_antisym(myadc,t2_1_a[:], eris.vvvv_p)
        else:
            t2_1_vvvv_a = contract_ladder_antisym(myadc,t2_1_a[:], eris.Lvv)

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv_a = radc_ao2mo.write_dataset(t2_1_vvvv_a)

        t2_2_a = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
        t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]] = t2_1_vvvv_a[:]
        t2_2_a[:,:,ab_ind_a[1],ab_ind_a[0]] = -t2_1_vvvv_a[:]

        t2_2_a += 0.5*lib.einsum('kilj,klab->ijab', eris_oooo, t2_1_a[:],optimize=True)
        t2_2_a -= 0.5*lib.einsum('kjli,klab->ijab', eris_oooo, t2_1_a[:],optimize=True)

        temp = lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1_a[:],optimize=True)
        temp -= lib.einsum('kjbc,kica->ijab',eris_oovv,t2_1_a[:],optimize=True)
        temp_1 = lib.einsum('kcbj,ikac->ijab',eris_OVvo,t2_1_ab[:],optimize=True)

        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        del temp
        del temp_1

        if isinstance(eris.VVVV_p, np.ndarray):
            eris_VVVV = eris.VVVV_p
            temp = np.ascontiguousarray(t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]).reshape(nocc_b*nocc_b,-1)
            t2_1_vvvv_b = np.dot(temp,eris_VVVV.T).reshape(nocc_b, nocc_b, -1)
            del eris_VVVV
        elif isinstance(eris.VVVV_p, list) : 
            t2_1_vvvv_b = contract_ladder_antisym(myadc,t2_1_b[:],eris.VVVV_p)
        else :
            t2_1_vvvv_b = contract_ladder_antisym(myadc,t2_1_b[:],eris.LVV)

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv_b = radc_ao2mo.write_dataset(t2_1_vvvv_b)

        t2_2_b = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
        t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]] = t2_1_vvvv_b[:]
        t2_2_b[:,:,ab_ind_b[1],ab_ind_b[0]] = -t2_1_vvvv_b[:]

        t2_2_b += 0.5*lib.einsum('kilj,klab->ijab', eris_OOOO, t2_1_b[:],optimize=True)
        t2_2_b -= 0.5*lib.einsum('kjli,klab->ijab', eris_OOOO, t2_1_b[:],optimize=True)

        temp = lib.einsum('kcbj,kica->ijab',eris_OVVO,t2_1_b[:],optimize=True)
        temp -= lib.einsum('kjbc,kica->ijab',eris_OOVV,t2_1_b[:],optimize=True)
        temp_1 = lib.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_ab[:],optimize=True)

        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)
        del temp
        del temp_1

        if isinstance(eris.vVvV_p, np.ndarray):
            temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
            eris_vVvV = eris.vVvV_p
            t2_1_vvvv_ab = np.dot(temp,eris_vVvV.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        elif isinstance(eris.vVvV_p, list):
            t2_1_vvvv_ab = contract_ladder(myadc,t2_1_ab[:],eris.vVvV_p)
        else :
            t2_1_vvvv_ab = contract_ladder(myadc,t2_1_ab[:],(eris.Lvv,eris.LVV))

        if not isinstance(eris.oooo, np.ndarray):
            t2_1_vvvv_ab = radc_ao2mo.write_dataset(t2_1_vvvv_ab)

        t2_2_ab = t2_1_vvvv_ab[:].copy()

        t2_2_ab += lib.einsum('kilj,klab->ijab',eris_ooOO,t2_1_ab[:],optimize=True)
        t2_2_ab += lib.einsum('kcbj,kica->ijab',eris_ovVO,t2_1_a[:],optimize=True)
        t2_2_ab += lib.einsum('kcbj,ikac->ijab',eris_OVVO,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kjbc,ikac->ijab',eris_OOVV,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kibc,kjac->ijab',eris_ooVV,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kjac,ikcb->ijab',eris_OOvv,t2_1_ab[:],optimize=True)
        t2_2_ab += lib.einsum('kcai,kjcb->ijab',eris_OVvo,t2_1_b[:],optimize=True)
        t2_2_ab += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1_ab[:],optimize=True)
        t2_2_ab -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1_ab[:],optimize=True)

        D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
        D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
        t2_2_a = t2_2_a/D2_a
        if not isinstance(eris.oooo, np.ndarray):
            t2_2_a = radc_ao2mo.write_dataset(t2_2_a)
        del D2_a

        D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
        D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))
        t2_2_b = t2_2_b/D2_b
        if not isinstance(eris.oooo, np.ndarray):
            t2_2_b = radc_ao2mo.write_dataset(t2_2_b)
        del D2_b

        D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
        D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))
        t2_2_ab = t2_2_ab/D2_ab
        if not isinstance(eris.oooo, np.ndarray):
            t2_2_ab = radc_ao2mo.write_dataset(t2_2_ab)
        del D2_ab

    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)

    if (myadc.method == "adc(3)"):
    # Compute third-order singles (tij)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

        t1_3 = (None,)

        t1_3_a = lib.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_a[:],t1_2_a,optimize=True)
        t1_3_a += lib.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_ab[:],t1_2_b,optimize=True)

        t1_3_b  = lib.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b += lib.einsum('d,lida,ld->ia',e_a[nocc_a:],t2_1_ab[:],t1_2_a,optimize=True)
 
        t1_3_a -= lib.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a[:], t1_2_a,optimize=True)
        t1_3_a -= lib.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab[:],t1_2_b,optimize=True)
 
        t1_3_b -= lib.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b -= lib.einsum('l,lida,ld->ia',e_a[:nocc_a],t2_1_ab[:],t1_2_a,optimize=True)
 
        t1_3_a += 0.5*lib.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a[:], t1_2_a,optimize=True)
        t1_3_a += 0.5*lib.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab[:],t1_2_b,optimize=True)
 
        t1_3_b += 0.5*lib.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b += 0.5*lib.einsum('a,lida,ld->ia',e_b[nocc_b:],t2_1_ab[:],t1_2_a,optimize=True)
 
        t1_3_a -= 0.5*lib.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a[:], t1_2_a,optimize=True)
        t1_3_a -= 0.5*lib.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab[:],t1_2_b,optimize=True)
 
        t1_3_b -= 0.5*lib.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_b[:], t1_2_b,optimize=True)
        t1_3_b -= 0.5*lib.einsum('i,lida,ld->ia',e_b[:nocc_b],t2_1_ab[:],t1_2_a,optimize=True)
 
        t1_3_a += lib.einsum('ld,iadl->ia',t1_2_a,eris_ovvo,optimize=True)
        t1_3_a -= lib.einsum('ld,ladi->ia',t1_2_a,eris_ovvo,optimize=True)
        t1_3_a += lib.einsum('ld,iadl->ia',t1_2_b,eris_ovVO,optimize=True)
 
        t1_3_b += lib.einsum('ld,iadl->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b -= lib.einsum('ld,ladi->ia',t1_2_b,eris_OVVO ,optimize=True)
        t1_3_b += lib.einsum('ld,ldai->ia',t1_2_a,eris_ovVO,optimize=True)
 
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
                eris_ovvv = dfadc.get_ovvv_spin_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir_a)
            k = eris_ovvv.shape[0]
            t1_3_a += 0.5*lib.einsum('ilde,lead->ia',t2_2_a[:,a:a+k],eris_ovvv,optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_a[:,a:a+k],eris_ovvv,optimize=True)
            t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a[:], eris_ovvv,  t2_1_a[:,a:a+k] ,optimize=True)
            t1_3_a += lib.einsum('ildf,mafe,lmde->ia',t2_1_a[:], eris_ovvv,  t2_1_a[:,a:a+k] ,optimize=True)
            t1_3_a += lib.einsum('ilfd,mefa,mled->ia',t2_1_ab[:],eris_ovvv, t2_1_ab[a:a+k],optimize=True)
            t1_3_a -= lib.einsum('ilfd,mafe,mled->ia',t2_1_ab[:],eris_ovvv, t2_1_ab[a:a+k],optimize=True)
            t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_a[:],eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_a[:],eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_b += 0.5*lib.einsum('lifa,mefd,lmde->ia',t2_1_ab[:],eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_b -= 0.5*lib.einsum('lifa,mdfe,lmde->ia',t2_1_ab[:],eris_ovvv,t2_1_a[:,a:a+k],optimize=True)
            t1_3_a[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab[:],eris_ovvv,t2_1_ab[:],optimize=True)
            t1_3_a[a:a+k] -= lib.einsum('mlfd,ifea,mled->ia',t2_1_ab[:],eris_ovvv,t2_1_ab[:],optimize=True)
            t1_3_a[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            t1_3_a[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1_a[:],eris_ovvv,t2_1_a[:],optimize=True)
            del eris_ovvv
            a += k

        if isinstance(eris.OVVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_b
        a = 0
        for p in range(0,nocc_b,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_OVVV = dfadc.get_ovvv_spin_df(myadc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
            else :
                eris_OVVV = radc_ao2mo.unpack_eri_1(eris.OVVV, nvir_b)
            k = eris_OVVV.shape[0]
            t1_3_b += 0.5*lib.einsum('ilde,lead->ia',t2_2_b[:,a:a+k],eris_OVVV,optimize=True)
            t1_3_b -= 0.5*lib.einsum('ilde,ldae->ia',t2_2_b[:,a:a+k],eris_OVVV,optimize=True)
            t1_3_b -= lib.einsum('ildf,mefa,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b += lib.einsum('ildf,mafe,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b += lib.einsum('lidf,mefa,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_b -= lib.einsum('lidf,mafe,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_a += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_a -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:,a:a+k],optimize=True)
            t1_3_b[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:],optimize=True)
            t1_3_b[a:a+k] -= lib.einsum('lmdf,ifea,lmde->ia',t2_1_ab[:],eris_OVVV,t2_1_ab[:],optimize=True)
            t1_3_b[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            t1_3_b[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1_b[:],eris_OVVV,t2_1_b[:],optimize=True)
            del eris_OVVV
            a += k

        if isinstance(eris.ovVV, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_a
        a = 0
        for p in range(0,nocc_a,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_ovVV = dfadc.get_ovvv_spin_df(myadc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            k = eris_ovVV.shape[0]
            t1_3_b += lib.einsum('lied,lead->ia',t2_2_ab[a:a+k],eris_ovVV,optimize=True)
            t1_3_a -= lib.einsum('ildf,mafe,mlde->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_b -= lib.einsum('ildf,mefa,mled->ia',t2_1_b[:],eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_b += lib.einsum('lidf,mefa,lmde->ia',t2_1_ab[:],eris_ovVV,t2_1_a[:,a:a+k],optimize=True)
            t1_3_a += lib.einsum('ilaf,mefd,mled->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_b += lib.einsum('ilaf,mefd,mled->ia',t2_1_b[:],eris_ovVV,t2_1_ab[a:a+k],optimize=True)
            t1_3_a[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_b[:],eris_ovVV,t2_1_b[:],optimize=True)
            t1_3_a[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[:],optimize=True)
            t1_3_a[a:a+k] -= lib.einsum('lmef,iedf,lmad->ia',t2_1_ab[:],eris_ovVV,t2_1_ab[:],optimize=True)
            del eris_ovVV
            a += k

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc_b
        a = 0
        for p in range(0,nocc_b,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_OVvv = dfadc.get_ovvv_spin_df(myadc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
            else :
                eris_OVvv = radc_ao2mo.unpack_eri_1(eris.OVvv, nvir_a)
            k = eris_OVvv.shape[0]
            t1_3_a += lib.einsum('ilde,lead->ia',t2_2_ab[:,a:a+k],eris_OVvv,optimize=True)
            t1_3_a -= lib.einsum('ildf,mefa,lmde->ia',t2_1_a[:],eris_OVvv, t2_1_ab[:,a:a+k],optimize=True)
            t1_3_a += lib.einsum('ilfd,mefa,lmde->ia',t2_1_ab[:],eris_OVvv,t2_1_b[:,a:a+k] ,optimize=True)
            t1_3_b -= lib.einsum('lifd,mafe,lmed->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_a += lib.einsum('ilaf,mefd,lmde->ia',t2_1_a[:],eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_b += lib.einsum('lifa,mefd,lmde->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:,a:a+k],optimize=True)
            t1_3_b[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1_a[:],eris_OVvv,t2_1_a[:],optimize=True)
            t1_3_b[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:],optimize=True)
            t1_3_b[a:a+k] -= lib.einsum('mlfe,iedf,mlda->ia',t2_1_ab[:],eris_OVvv,t2_1_ab[:],optimize=True)

            del eris_OVvv
            a += k
 
        t1_3_a += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('inde,lamn,lmde->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
 
        t1_3_b += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('nied,lamn,mled->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
 
        t1_3_a += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1_a[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1_a[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5 *lib.einsum('inad,lemn,lmed->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,mled->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_a += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_b[:],optimize=True)
 
        t1_3_b += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= 0.5 * lib.einsum('inad,meln,mled->ia',t2_1_b[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5 * lib.einsum('inad,lemn,lmed->ia',t2_1_b[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5 *lib.einsum('nida,meln,lmde->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('nida,lemn,mlde->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_b += 0.5*lib.einsum('nida,lemn,lmde->ia',t2_1_ab[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('nida,meln,lmde->ia',t2_1_ab[:],eris_ovoo,t2_1_a[:],optimize=True)
 
        t1_3_a -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a -= lib.einsum('nled,ianm,mled->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a += lib.einsum('nled,naim,mled->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_b[:],eris_ovOO,t2_1_b[:],optimize=True)
        t1_3_a -= lib.einsum('lnde,ianm,lmde->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
 
        t1_3_b -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b -= lib.einsum('lnde,ianm,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b += lib.einsum('lnde,naim,lmde->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1_a[:],eris_OVoo,t2_1_a[:],optimize=True)
        t1_3_b -= lib.einsum('nled,ianm,mled->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
 
        t1_3_a -= lib.einsum('lnde,ienm,lmad->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('lnde,neim,lmad->ia',t2_1_a[:],eris_ovoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('lnde,neim,lmad->ia',t2_1_ab[:],eris_OVoo,t2_1_a[:],optimize=True)
        t1_3_a += lib.einsum('nled,ienm,mlad->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a -= lib.einsum('nled,neim,mlad->ia',t2_1_ab[:],eris_ovoo,t2_1_ab[:],optimize=True)
        t1_3_a += lib.einsum('lned,ienm,lmad->ia',t2_1_ab[:],eris_ovOO,t2_1_ab[:],optimize=True)
        t1_3_a -= lib.einsum('lnde,neim,mlad->ia',t2_1_b[:],eris_OVoo,t2_1_ab[:],optimize=True)
 
        t1_3_b -= lib.einsum('lnde,ienm,lmad->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('lnde,neim,lmad->ia',t2_1_b[:],eris_OVOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('nled,neim,lmad->ia',t2_1_ab[:],eris_ovOO,t2_1_b[:],optimize=True)
        t1_3_b += lib.einsum('lnde,ienm,lmda->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b -= lib.einsum('lnde,neim,lmda->ia',t2_1_ab[:],eris_OVOO,t2_1_ab[:],optimize=True)
        t1_3_b += lib.einsum('nlde,ienm,mlda->ia',t2_1_ab[:],eris_OVoo,t2_1_ab[:],optimize=True)
        t1_3_b -= lib.einsum('lnde,neim,lmda->ia',t2_1_a[:],eris_ovOO,t2_1_ab[:],optimize=True)
 
        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b

        t1_3 = (t1_3_a, t1_3_b)

    del D1_a, D1_b 

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
       t2_2 = (t2_2_a , t2_2_ab, t2_2_b)
       t2_1_vvvv = (t2_1_vvvv_a, t2_1_vvvv_ab, t2_1_vvvv_b)

    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)

    return t1, t2, t2_1_vvvv


def compute_energy(myadc, t1, t2, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc_a = myadc._nocc[0]
    nocc_b = myadc._nocc[1]
    nvir_a = myadc._nvir[0]
    nvir_b = myadc._nvir[1]

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO

    #Compute MPn correlation energy
    t2_a  = t2[0][0][:].copy()
    if (myadc.method == "adc(3)"):
       t2_a += t2[1][0][:]

    e_mp = 0.25 * lib.einsum('ijab,iabj', t2_a, eris_ovvo)
    e_mp -= 0.25 * lib.einsum('ijab,ibaj', t2_a, eris_ovvo)
    del t2_a

    t2_ab  = t2[0][1][:].copy()
    if (myadc.method == "adc(3)"):
       t2_ab += t2[1][1][:]

    e_mp += lib.einsum('ijab,iabj', t2_ab, eris_ovVO)
    del t2_ab

    t2_b  = t2[0][2][:].copy()
    if (myadc.method == "adc(3)"):
       t2_b += t2[1][2][:]

    e_mp += 0.25 * lib.einsum('ijab,iabj', t2_b, eris_OVVO)
    e_mp -= 0.25 * lib.einsum('ijab,ibaj', t2_b, eris_OVVO)
    del t2_b

    cput0 = log.timer_debug1("Completed energy calculation", *cput0)

    return e_mp

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
            del vvvv
            a += k
    else :
        raise Exception("Unknown vvvv type")

    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc_a, nocc_b, nvir_a, nvir_b)

    return t


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
            del vvvv
            a += k
    else :
        raise Exception("Unknown vvvv type") 

    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc, nocc, nvir, nvir)
    t = t[:, :, tril_idx[0], tril_idx[1]]

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


def analyze(myadc):

    str = ("\n*************************************************************"
          "\n           Eigenvector analysis summary"                    
          "\n*************************************************************")
    logger.info(myadc, str)

    myadc.analyze_eigenvector()
 
    if myadc.compute_properties:

        str = ("\n*************************************************************"
               "\n            Spectroscopic factors analysis summary"
               "\n*************************************************************")
        logger.info(myadc, str)

        myadc.analyze_spec_factor()


def compute_dyson_mo(myadc):
     
    X_a = myadc.X[0]
    X_b = myadc.X[1]

    if X_a is None:
        nroots = myadc.U.shape[1]
        P,X_a,X_b = myadc.get_properties(nroots)

    nroots = X_a.shape[1]
    dyson_mo_a = np.dot(myadc.mo_coeff[0],X_a)
    dyson_mo_b = np.dot(myadc.mo_coeff[1],X_b)

    dyson_mo = (dyson_mo_a,dyson_mo_b)

    return dyson_mo


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
        self.tol_residual = getattr(__config__, 'adc_uadc_UADC_tol_res', 1e-6)

        self.scf_energy = mf.e_tot
        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway
        
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.imds = lambda:None
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
        self.compute_spec = True
        self.compute_properties = True
        self.evec_print_tol = 0.1
        self.spec_factor_print_tol = 0.1
        self.ncvs = 1
        self.E = None
        self.U = None
        self.P = None
        self.X = (None,)
        
        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mol', 'mo_energy_b', 'max_memory', 'scf_energy', 'e_tot', 't1', 'frozen', 'mo_energy_a', 'chkfile', 'max_space', 't2', 'mo_occ', 'max_cycle'))

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
        logger.info(self, 'tol_residual = %s', self.tol_residual)
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

        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self._finalize()

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

        self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris, verbose=self.verbose)
        self._finalize()

        self.method_type = self.method_type.lower()
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac, X, adc_es = self.ea_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip"):
            #e_exc, v_exc, spec_fac, X, adc_es = self.ip_adc(nroots=nroots, guess=guess, eris=eris)
            r = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip-cvs"):
            #e_exc, v_exc, spec_fac, x, adc_es = self.ip_cvs_adc(nroots=nroots, guess=guess, eris=eris)
            r = self.ip_cvs_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)

        #self._adc_es = adc_es
        #return e_exc, v_exc, spec_fac, X
        return r

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E_corr = %.8f',
                    self.e_corr)
        return self
    
    def ea_adc(self, nroots=1, guess=None, eris=None):
        adc_es = UADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_adc(self, nroots=1, guess=None, eris=None):
        adc_es = UADCIP(self)
        #e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        r = adc_es.kernel(nroots, guess, eris)
        #return e_exc, v_exc, spec_fac, x, adc_es
        return r

    def ip_cvs_adc(self, nroots=1, guess=None, eris=None):
        adc_es = UADCIPCVS(self)
        #e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        r = adc_es.kernel(nroots, guess, eris)
        return r

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

    def analyze(self):
        self._adc_es.analyze()

    def compute_dyson_mo(self):   
        return self._adc_es.compute_dyson_mo() 

def get_imds_ea(adc, eris=None):

    cput0 = (time.clock(), time.time())
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
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO
    eris_OVvo = eris.OVvo

    # a-b block
    # Zeroth-order terms

    M_ab_a = lib.einsum('ab,a->ab', idn_vir_a, e_vir_a)
    M_ab_b = lib.einsum('ab,a->ab', idn_vir_b, e_vir_b)

   # Second-order terms

    t2_1_a = t2[0][0][:]
    M_ab_a +=  lib.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1_a, eris_ovvo,optimize=True)
    M_ab_a += 0.5 *  lib.einsum('lmad,ldbm->ab',t2_1_a, eris_ovvo,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1_a,eris_ovvo,optimize=True)
    M_ab_a += 0.5 *  lib.einsum('lmbd,ldam->ab',t2_1_a, eris_ovvo,optimize=True)
    del t2_1_a

    t2_1_b = t2[0][2][:]
    M_ab_b +=  lib.einsum('l,lmad,lmbd->ab',e_occ_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b += 0.5 *  lib.einsum('lmad,ldbm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_1_b, eris_OVVO,optimize=True)
    M_ab_b += 0.5 *  lib.einsum('lmbd,ldam->ab',t2_1_b, eris_OVVO,optimize=True)
    del t2_1_b

    t2_1_ab = t2[0][1][:]
    M_ab_a +=  lib.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_a +=  lib.einsum('l,mlad,mlbd->ab',e_occ_b,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_b +=  lib.einsum('l,mlda,mldb->ab',e_occ_b,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_b +=  lib.einsum('l,lmda,lmdb->ab',e_occ_a,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_a -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_a -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir_b,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_b -= 0.5 *  lib.einsum('d,mlda,mldb->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)
    M_ab_b -= 0.5 *  lib.einsum('d,lmda,lmdb->ab',e_vir_a,t2_1_ab, t2_1_ab,optimize=True)

    M_ab_t = lib.einsum('lmad,lmbd->ab', t2_1_ab,t2_1_ab, optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('a,ab->ab',e_vir_a,M_ab_t,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('b,ab->ab',e_vir_a,M_ab_t,optimize=True)
    del M_ab_t

    M_ab_t = lib.einsum('mlda,mldb->ab', t2_1_ab,t2_1_ab, optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('a,ab->ab',e_vir_b,M_ab_t,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('b,ab->ab',e_vir_b,M_ab_t,optimize=True)
    del M_ab_t

    M_ab_t = lib.einsum('lmda,lmdb->ab', t2_1_ab,t2_1_ab, optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('a,ab->ab',e_vir_b,M_ab_t,optimize=True)
    M_ab_b -= 0.25 *  lib.einsum('b,ab->ab',e_vir_b,M_ab_t,optimize=True)
    del M_ab_t

    M_ab_t = lib.einsum('mlad,mlbd->ab', t2_1_ab,t2_1_ab, optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('a,ab->ab',e_vir_a,M_ab_t,optimize=True)
    M_ab_a -= 0.25 *  lib.einsum('b,ab->ab',e_vir_a,M_ab_t,optimize=True)
    del M_ab_t

    M_ab_a -=        lib.einsum('lmad,lbdm->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_b -=        lib.einsum('mlda,mdbl->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_a -=        lib.einsum('lmbd,ladm->ab',t2_1_ab, eris_ovVO,optimize=True)
    M_ab_b -=        lib.einsum('mldb,mdal->ab',t2_1_ab, eris_ovVO,optimize=True)
    del t2_1_ab

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    #Third-order terms
    if(method =='adc(3)'):

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
                eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
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
                eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
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
                eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
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
                eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
            else :
                eris_ovVV = radc_ao2mo.unpack_eri_1(eris.ovVV, nvir_b)
            k = eris_ovVV.shape[0]
            M_ab_b +=  lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovVV,optimize=True)
            M_ab_b += lib.einsum('ld,ldab->ab',t1_2_a[a:a+k], eris_ovVV,optimize=True)
            del eris_ovVV
            a += k

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

        t2_2_a = t2[1][0][:]
        M_ab_a -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2_a, eris_ovvo,optimize=True)
        M_ab_a += 0.5 *  lib.einsum('lmad,ldbm->ab',t2_2_a, eris_ovvo,optimize=True)
        M_ab_a -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_2_a,eris_ovvo,optimize=True)
        M_ab_a += 0.5 *  lib.einsum('lmbd,ldam->ab',t2_2_a, eris_ovvo,optimize=True)

        t2_1_a = t2[0][0][:]
        M_ab_a += lib.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += lib.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir_a, t2_1_a ,t2_2_a, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir_a, t2_1_a, t2_2_a, optimize=True)

        M_ab_t = lib.einsum('lmbd,lmad->ab', t2_1_a,t2_2_a, optimize=True)
        M_ab_a -= 0.25*lib.einsum('a,ab->ab',e_vir_a, M_ab_t, optimize=True)
        M_ab_a -= 0.25*lib.einsum('a,ba->ab',e_vir_a, M_ab_t, optimize=True)
        M_ab_a -= 0.25*lib.einsum('b,ab->ab',e_vir_a, M_ab_t, optimize=True)
        M_ab_a -= 0.25*lib.einsum('b,ba->ab',e_vir_a, M_ab_t, optimize=True)
        del M_ab_t
        del t2_1_a
        del t2_2_a

        t2_2_b = t2[1][2][:]
        M_ab_b -= 0.5 *  lib.einsum('lmad,lbdm->ab',t2_2_b, eris_OVVO,optimize=True)
        M_ab_b += 0.5 *  lib.einsum('lmad,ldbm->ab',t2_2_b, eris_OVVO,optimize=True)
        M_ab_b -= 0.5 *  lib.einsum('lmbd,ladm->ab',t2_2_b, eris_OVVO,optimize=True)
        M_ab_b += 0.5 *  lib.einsum('lmbd,ldam->ab',t2_2_b, eris_OVVO,optimize=True)

        t2_1_b = t2[0][2][:]
        M_ab_b += lib.einsum('l,lmbd,lmad->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += lib.einsum('l,lmad,lmbd->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_b ,t2_2_b, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_b, t2_2_b, optimize=True)

        M_ab_t = lib.einsum('lmbd,lmad->ab', t2_1_b,t2_2_b, optimize=True)
        M_ab_b -= 0.25*lib.einsum('a,ab->ab',e_vir_b, M_ab_t, optimize=True)
        M_ab_b -= 0.25*lib.einsum('a,ba->ab',e_vir_b, M_ab_t, optimize=True)
        M_ab_b -= 0.25*lib.einsum('b,ab->ab',e_vir_b, M_ab_t, optimize=True)
        M_ab_b -= 0.25*lib.einsum('b,ba->ab',e_vir_b, M_ab_t, optimize=True)
        del M_ab_t
        del t2_1_b
        del t2_2_b

        t2_2_ab = t2[1][1][:]
        M_ab_a -=        lib.einsum('lmad,lbdm->ab',t2_2_ab, eris_ovVO,optimize=True)
        M_ab_b -=        lib.einsum('mlda,mdbl->ab',t2_2_ab, eris_ovVO,optimize=True)
        M_ab_a -=        lib.einsum('lmbd,ladm->ab',t2_2_ab, eris_ovVO,optimize=True)
        M_ab_b -=        lib.einsum('mldb,mdal->ab',t2_2_ab, eris_ovVO,optimize=True)

        t2_1_ab = t2[0][1][:]
        M_ab_a += lib.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += lib.einsum('l,mlbd,mlad->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += lib.einsum('l,mldb,mlda->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += lib.einsum('l,lmdb,lmda->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a += lib.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += lib.einsum('l,mlad,mlbd->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += lib.einsum('l,mlda,mldb->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += lib.einsum('l,lmda,lmdb->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_b -= 0.5*lib.einsum('d,mldb,mlda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,lmdb,lmda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_a -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.5*lib.einsum('d,mlda,mldb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.5*lib.einsum('d,lmda,lmdb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_t = lib.einsum('lmbd,lmad->ab', t2_1_ab,t2_2_ab, optimize=True)
        M_ab_a -= 0.5*lib.einsum('a,ab->ab',e_vir_a, M_ab_t, optimize=True)
        M_ab_a -= 0.5*lib.einsum('a,ba->ab',e_vir_a, M_ab_t, optimize=True)
        M_ab_a -= 0.5*lib.einsum('b,ab->ab',e_vir_a, M_ab_t, optimize=True)
        M_ab_a -= 0.5*lib.einsum('b,ba->ab',e_vir_a, M_ab_t, optimize=True)
        del M_ab_t

        M_ab_t = lib.einsum('mldb,mlda->ab', t2_1_ab,t2_2_ab, optimize=True)
        M_ab_b -= 0.5*lib.einsum('a,ab->ab',e_vir_b, M_ab_t, optimize=True)
        M_ab_b -= 0.5*lib.einsum('a,ba->ab',e_vir_b, M_ab_t, optimize=True)
        M_ab_b -= 0.5*lib.einsum('b,ab->ab',e_vir_b, M_ab_t, optimize=True)
        M_ab_b -= 0.5*lib.einsum('b,ba->ab',e_vir_b, M_ab_t, optimize=True)
        del M_ab_t
        del t2_1_ab
        del t2_2_ab

        t2_1_a = t2[0][0][:]
        t2_1_ab = t2[0][1][:]

        M_ab_a -= lib.einsum('lnde,mlbd,neam->ab',t2_1_ab, t2_1_a, eris_OVvo, optimize=True)
        M_ab_a += lib.einsum('lned,lmbd,nmae->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_t = lib.einsum('lned,mlbd->nemb', t2_1_a,t2_1_a, optimize=True)
        M_ab_a -= lib.einsum('nemb,nmae->ab',M_ab_t, eris_oovv, optimize=True)
        M_ab_a += lib.einsum('nemb,maen->ab',M_ab_t, eris_ovvo, optimize=True)
        M_ab_a -= lib.einsum('name,nmeb->ab',M_ab_t, eris_oovv, optimize=True)
        M_ab_a += lib.einsum('name,nbem->ab',M_ab_t, eris_ovvo, optimize=True)
        del M_ab_t

        M_ab_t = lib.einsum('nled,mlbd->nemb', t2_1_ab,t2_1_ab, optimize=True)
        M_ab_a += lib.einsum('nemb,nmae->ab',M_ab_t, eris_oovv, optimize=True)
        M_ab_a -= lib.einsum('nemb,maen->ab',M_ab_t, eris_ovvo, optimize=True)
        del M_ab_t

        M_ab_t = lib.einsum('lnde,lmdb->nemb', t2_1_ab,t2_1_ab, optimize=True)
        M_ab_b += lib.einsum('nemb,nmae->ab',M_ab_t, eris_OOVV, optimize=True)
        M_ab_b -= lib.einsum('nemb,maen->ab',M_ab_t, eris_OVVO, optimize=True)
        del M_ab_t

        M_ab_b += lib.einsum('lned,lmdb,neam->ab',t2_1_a, t2_1_ab, eris_ovVO, optimize=True)
        M_ab_b += lib.einsum('nlde,mldb,nmae->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_a += lib.einsum('mled,nlad,nmeb->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a -= lib.einsum('mled,nlad,nbem->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)
        M_ab_a += lib.einsum('lmed,lnad,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)
        M_ab_a += lib.einsum('lmde,lnad,nbem->ab',t2_1_ab, t2_1_a, eris_ovVO, optimize=True)

        M_ab_b += lib.einsum('lmde,lnda,nmeb->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b -= lib.einsum('lmde,lnda,nbem->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_b += lib.einsum('mlde,nlda,nmeb->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)
        M_ab_b -= lib.einsum('mled,lnda,nbem->ab',t2_1_a, t2_1_ab, eris_OVvo, optimize=True)

        M_ab_a -= lib.einsum('mlbd,lnae,nmde->ab',t2_1_a, t2_1_a,   eris_oovv, optimize=True)
        M_ab_a += lib.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_a,   eris_ovvo, optimize=True)
        M_ab_a += lib.einsum('lmbd,lnae,nmde->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_a -= lib.einsum('lmbd,lnae,nedm->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_a += lib.einsum('mlbd,lnae,nedm->ab',t2_1_a, t2_1_ab,  eris_OVvo, optimize=True)
        M_ab_a -= lib.einsum('lmbd,lnae,nedm->ab',t2_1_ab, t2_1_a,  eris_ovVO, optimize=True)
        M_ab_a += lib.einsum('mlbd,nlae,nmde->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_b += lib.einsum('mldb,nlea,nmde->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_b -= lib.einsum('mldb,nlea,nedm->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)

        M_ab_b += lib.einsum('lmdb,lnea,nmed->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)

        M_ab_a += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_oovv, optimize=True)
        M_ab_a -= 0.5*lib.einsum('lned,mled,nbam->ab',t2_1_a, t2_1_a, eris_ovvo, optimize=True)
        M_ab_a -= lib.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_oovv, optimize=True)
        M_ab_a += lib.einsum('nled,mled,nbam->ab',t2_1_ab, t2_1_ab, eris_ovvo, optimize=True)

        M_ab_a -= lib.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOvv, optimize=True)
        M_ab_b -= lib.einsum('lned,lmed,nmab->ab',t2_1_ab, t2_1_ab, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('lned,lmed,nbam->ab',t2_1_ab, t2_1_ab, eris_OVVO, optimize=True)
        M_ab_b += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_a, t2_1_a, eris_ooVV, optimize=True)
        M_ab_b -= lib.einsum('nled,mled,nmab->ab',t2_1_ab, t2_1_ab, eris_ooVV, optimize=True)

        M_ab_a -= 0.25*lib.einsum('mlbd,noad,nmol->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab_a += 0.25*lib.einsum('mlbd,noad,nlom->ab',t2_1_a, t2_1_a, eris_oooo, optimize=True)
        M_ab_a -= lib.einsum('mlbd,noad,nmol->ab',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)
        M_ab_b -= lib.einsum('lmdb,onda,olnm->ab',t2_1_ab, t2_1_ab, eris_ooOO, optimize=True)
        del t2_1_a

        t2_1_b = t2[0][2][:]
        M_ab_a += lib.einsum('lned,mlbd,neam->ab',t2_1_b, t2_1_ab, eris_OVvo, optimize=True)

        M_ab_t = lib.einsum('lned,mlbd->nemb', t2_1_b,t2_1_b, optimize=True)
        M_ab_b -= lib.einsum('nemb,nmae->ab',M_ab_t, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('nemb,maen->ab',M_ab_t, eris_OVVO, optimize=True)
        M_ab_b -= lib.einsum('name,nmeb->ab',M_ab_t, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('name,nbem->ab',M_ab_t, eris_OVVO, optimize=True)
        del M_ab_t

        M_ab_b -= lib.einsum('nled,mlbd,neam->ab',t2_1_ab, t2_1_b, eris_ovVO, optimize=True)
        M_ab_a -= lib.einsum('mled,nlad,nbem->ab',t2_1_b, t2_1_ab, eris_ovVO, optimize=True)
        M_ab_b += lib.einsum('mled,lnad,nbem->ab',t2_1_ab, t2_1_b, eris_OVvo, optimize=True)

        M_ab_b -= lib.einsum('mlbd,lnae,nmde->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b += lib.einsum('mlbd,lnae,nedm->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)
        M_ab_b += lib.einsum('mlbd,nlea,nedm->ab',t2_1_b, t2_1_ab,  eris_ovVO, optimize=True)
        M_ab_b -= lib.einsum('mldb,lnae,nedm->ab',t2_1_ab, t2_1_b,  eris_OVvo, optimize=True)
        M_ab_a += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOvv, optimize=True)

        M_ab_b += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1_b, t2_1_b, eris_OOVV, optimize=True)
        M_ab_b -= 0.5*lib.einsum('lned,mled,nbam->ab',t2_1_b, t2_1_b, eris_OVVO, optimize=True)

        M_ab_b -= 0.25*lib.einsum('mlbd,noad,nmol->ab',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
        M_ab_b += 0.25*lib.einsum('mlbd,noad,nlom->ab',t2_1_b, t2_1_b, eris_OOOO, optimize=True)
        del t2_1_b

        log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

        t2_1_a = t2[0][0][:]
        t2_1_ab = t2[0][1][:]

        if isinstance(eris.vvvv_p,np.ndarray):
            eris_vvvv = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)
            M_ab_a -= 0.25*lib.einsum('mlef,mlbd,adef->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
            M_ab_a -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_vvvv, optimize=True)
            M_ab_a += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_vvvv, optimize=True)
            del eris_vvvv

            temp = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))
            temp[:,:,ab_ind_a[0],ab_ind_a[1]] =  adc.imds.t2_1_vvvv[0]
            temp[:,:,ab_ind_a[1],ab_ind_a[0]] = -adc.imds.t2_1_vvvv[0]

            M_ab_a -= 2 * 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_a, temp, optimize=True)
            del temp
            
        else :

            temp_t2a_vvvv = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))   
            temp_t2a_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = adc.imds.t2_1_vvvv[0][:]   
            temp_t2a_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -adc.imds.t2_1_vvvv[0][:]

            M_ab_a -= 2*0.25*lib.einsum('mlad,mlbd->ab',  temp_t2a_vvvv, t2_1_a, optimize=True)
            M_ab_a -= 2*0.25*lib.einsum('mlaf,mlbf->ab', t2_1_a, temp_t2a_vvvv, optimize=True)
            del temp_t2a_vvvv

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
            del temp            

            a = 0
            temp = np.zeros((nvir_b,nvir_b))
            for p in range(0,nvir_b,chnk_size):
                eris_VvVv = dfadc.get_vVvV_df(adc, eris.LVV, eris.Lvv, p, chnk_size) 
                k = eris_VvVv.shape[0]

                temp[a:a+k] -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, eris_VvVv, optimize=True)
                temp[a:a+k] += lib.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, eris_VvVv, optimize=True)
                a += k
            M_ab_b  += temp    
            del temp            

        t2_1_b = t2[0][2][:]
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
            temp = adc.imds.t2_1_vvvv[1]
            M_ab_a -= lib.einsum('mlaf,mlbf->ab',t2_1_ab, temp, optimize=True)
            M_ab_b -= lib.einsum('mlfa,mlfb->ab',t2_1_ab, temp, optimize=True)
            del temp
        else: 
           t2_vVvV = adc.imds.t2_1_vvvv[1][:]

           M_ab_a -= lib.einsum('mlad,mlbd->ab', t2_vVvV, t2_1_ab, optimize=True)
           M_ab_b -= lib.einsum('mlda,mldb->ab', t2_vVvV, t2_1_ab, optimize=True)
           M_ab_a -= lib.einsum('mlaf,mlbf->ab',t2_1_ab, t2_vVvV, optimize=True)
           M_ab_b -= lib.einsum('mlfa,mlfb->ab',t2_1_ab, t2_vVvV, optimize=True)
           del t2_vVvV
        del t2_1_a

        if isinstance(eris.VVVV_p,np.ndarray):
            eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
            M_ab_b -= 0.25*lib.einsum('mlef,mlbd,adef->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
            M_ab_b -= 0.5*lib.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, eris_VVVV, optimize=True)
            M_ab_b += lib.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, eris_VVVV, optimize=True)
            del eris_VVVV

            temp = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))
            temp[:,:,ab_ind_b[0],ab_ind_b[1]] =  adc.imds.t2_1_vvvv[2]
            temp[:,:,ab_ind_b[1],ab_ind_b[0]] = -adc.imds.t2_1_vvvv[2]
            M_ab_b -= 2 * 0.25*lib.einsum('mlaf,mlbf->ab',t2_1_b, temp, optimize=True)
            del temp
        else:

            temp_t2b_VVVV = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))   
            temp_t2b_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = adc.imds.t2_1_vvvv[2][:] 
            temp_t2b_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -adc.imds.t2_1_vvvv[2][:]

            M_ab_b -= 2 * 0.25*lib.einsum('mlad,mlbd->ab',  temp_t2b_VVVV, t2_1_b, optimize=True)
            M_ab_b -= 2 * 0.25*lib.einsum('mlaf,mlbf->ab', t2_1_b, temp_t2b_VVVV, optimize=True)
            del temp_t2b_VVVV

        if isinstance(eris.vvvv_p, list):

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

        elif isinstance(eris.vvvv_p, type(None)):

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
            del temp            

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
            del temp            

        del t2_1_ab, t2_1_b

    M_ab = (M_ab_a, M_ab_b)
     
    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
    return M_ab


def get_imds_ip(adc, eris=None):

    cput0 = (time.clock(), time.time())
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
    ncvs = adc.ncvs

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

    M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_1_a, optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a, optimize=True)
    M_ij_a += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a += 0.5 *  lib.einsum('jlde,idel->ij',t2_1_a, eris_ovvo, optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('jlde,ldei->ij',t2_1_a, eris_ovvo, optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_a,t2_1_a, optimize=True)
    M_ij_a -= 0.25 *  lib.einsum('i,ij->ij',e_occ_a, M_ij_t, optimize=True) 
    M_ij_a -= 0.25 *  lib.einsum('j,ij->ij',e_occ_a, M_ij_t, optimize=True) 
    del M_ij_t
    del t2_1_a

    t2_1_b = t2[0][2][:]

    M_ij_b +=  lib.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_1_b, optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b, optimize=True)
    M_ij_b += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b += 0.5 *  lib.einsum('jlde,idel->ij',t2_1_b, eris_OVVO, optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('jlde,ldei->ij',t2_1_b, eris_OVVO, optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_b, t2_1_b, optimize=True)
    M_ij_b -= 0.25 *  lib.einsum('i,ij->ij',e_occ_b, M_ij_t, optimize=True) 
    M_ij_b -= 0.25 *  lib.einsum('j,ij->ij',e_occ_b, M_ij_t, optimize=True) 
    del M_ij_t
    del t2_1_b

    t2_1_ab = t2[0][1][:]
    M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a +=  lib.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_b +=  lib.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b +=  lib.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_b -= 0.5*lib.einsum('l,lide,ljde->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_1_ab, optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_ab, t2_1_ab, optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('i,ij->ij',e_occ_a, M_ij_t, optimize=True) 
    M_ij_a -= 0.5 *  lib.einsum('j,ij->ij',e_occ_a, M_ij_t, optimize=True) 
    del M_ij_t

    M_ij_t = lib.einsum('lied,ljed->ij', t2_1_ab, t2_1_ab, optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('i,ij->ij',e_occ_b, M_ij_t, optimize=True) 
    M_ij_b -= 0.5 *  lib.einsum('j,ij->ij',e_occ_b, M_ij_t, optimize=True) 
    del M_ij_t

    M_ij_a += lib.einsum('ilde,jdel->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_b += lib.einsum('lied,ledj->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_a += lib.einsum('jlde,idel->ij',t2_1_ab, eris_ovVO, optimize=True)
    M_ij_b += lib.einsum('ljed,ledi->ij',t2_1_ab, eris_ovVO, optimize=True)
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

        M_ij_a += 0.5* lib.einsum('ilde,jdel->ij',t2_2_a, eris_ovvo, optimize=True)
        M_ij_a -= 0.5* lib.einsum('ilde,jedl->ij',t2_2_a, eris_ovvo, optimize=True)

        M_ij_a += 0.5* lib.einsum('jlde,ledi->ij',t2_2_a, eris_ovvo, optimize=True)
        M_ij_a -= 0.5* lib.einsum('jlde,iedl->ij',t2_2_a, eris_ovvo, optimize=True)

        M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  lib.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)

        M_ij_a -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_a,t2_2_a, optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('i,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.25 *  lib.einsum('i,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.25 *  lib.einsum('j,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.25 *  lib.einsum('j,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        del M_ij_t
        del t2_2_a
        del t2_1_a

        t2_1_b = t2[0][2][:]
        t2_2_b = t2[1][2][:]

        M_ij_b += 0.5* lib.einsum('ilde,jdel->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b -= 0.5* lib.einsum('ilde,jedl->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b += 0.5* lib.einsum('jlde,ledi->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b -= 0.5* lib.einsum('jlde,iedl->ij',t2_2_b, eris_OVVO, optimize=True)

        M_ij_b +=  lib.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  lib.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_b, t2_2_b, optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('i,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.25 *  lib.einsum('i,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.25 *  lib.einsum('j,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.25 *  lib.einsum('j,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        del M_ij_t
        del t2_2_b
        del t2_1_b

        t2_1_ab = t2[0][1][:]
        t2_2_ab = t2[1][1][:]

        M_ij_a += lib.einsum('ilde,jdel->ij',t2_2_ab, eris_ovVO, optimize=True)
        M_ij_b += lib.einsum('lied,ledj->ij',t2_2_ab, eris_ovVO, optimize=True)
        M_ij_a += lib.einsum('jlde,ledi->ij',t2_2_ab, eris_OVvo, optimize=True)
        M_ij_b += lib.einsum('ljed,ledi->ij',t2_2_ab, eris_ovVO, optimize=True)

        M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  lib.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  lib.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  lib.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a +=  lib.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  lib.einsum('d,jled,iled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  lib.einsum('d,ljde,lide->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  lib.einsum('d,ljed,lied->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5*lib.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_ab, t2_2_ab, optimize=True)
        M_ij_a -= 0.5 *  lib.einsum('i,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.5 *  lib.einsum('i,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.5 *  lib.einsum('j,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.5 *  lib.einsum('j,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        del M_ij_t

        M_ij_t = lib.einsum('lied,ljed->ij', t2_1_ab, t2_2_ab, optimize=True)
        M_ij_b -= 0.5 *  lib.einsum('i,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.5 *  lib.einsum('i,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.5 *  lib.einsum('j,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.5 *  lib.einsum('j,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        del M_ij_t
        del t2_1_ab
        del t2_2_ab

        t2_1_a = t2[0][0][:]
        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_a, t2_1_a, optimize=True)
        M_ij_a -= lib.einsum('mejf,mefi->ij',M_ij_t, eris_ovvo, optimize=True) 
        M_ij_a -= lib.einsum('mejf,mefi->ji',M_ij_t, eris_ovvo, optimize=True) 
        M_ij_a += lib.einsum('mejf,mife->ij',M_ij_t, eris_oovv, optimize=True) 
        M_ij_a += lib.einsum('mejf,mife->ji',M_ij_t, eris_oovv, optimize=True) 
        del M_ij_t

        M_ij_a += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)

        if isinstance(eris.vvvv_p,np.ndarray):
            eris_vvvv = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)
            M_ij_a += 0.25 * lib.einsum('ilde,jlgf,gfde->ij',t2_1_a, t2_1_a, eris_vvvv, optimize = True)
            del eris_vvvv

        else:

            temp_t2a_vvvv = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))   
            temp_t2a_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = adc.imds.t2_1_vvvv[0][:]    
            temp_t2a_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -adc.imds.t2_1_vvvv[0][:]

            M_ij_a += 2*0.25 * lib.einsum('ilgf,jlgf->ij', temp_t2a_vvvv, t2_1_a, optimize = True)
            del temp_t2a_vvvv 

        M_ij_a += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_OOvv , optimize = True)

        M_ij_a -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)

        M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooOO, optimize = True)

        t2_1_ab = t2[0][1][:]
        M_ij_a -= lib.einsum('lmde,jldf,mefi->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_b += lib.einsum('lmde,ljdf,mefi->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)
        M_ij_a -= lib.einsum('lmde,ildf,mefj->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_b += lib.einsum('lmde,lidf,mefj->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_ab, eris_ovVO, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_a, eris_OVvo, optimize = True)
        del t2_1_a

        t2_1_b = t2[0][2][:]
        M_ij_a += lib.einsum('lmde,jlfd,mefi->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)
        M_ij_b -= lib.einsum('mled,jldf,mefi->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_a += lib.einsum('lmde,ilfd,mefj->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)
        M_ij_b -= lib.einsum('mled,ildf,mefj->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_b += lib.einsum('ilde,mjfd,lefm->ij',t2_1_b, t2_1_ab, eris_OVvo, optimize = True)
        M_ij_b += lib.einsum('lied,jmdf,lefm->ij',t2_1_ab, t2_1_b, eris_ovVO, optimize = True)
        del t2_1_ab

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_b, t2_1_b, optimize=True)
        M_ij_b -= lib.einsum('mejf,mefi->ij',M_ij_t, eris_OVVO, optimize=True) 
        M_ij_b -= lib.einsum('mejf,mefi->ji',M_ij_t, eris_OVVO, optimize=True) 
        M_ij_b += lib.einsum('mejf,mife->ij',M_ij_t, eris_OOVV, optimize=True) 
        M_ij_b += lib.einsum('mejf,mife->ji',M_ij_t, eris_OOVV, optimize=True) 
        del M_ij_t

        M_ij_b += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)

        if isinstance(eris.VVVV_p,np.ndarray):
            eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
            M_ij_b += 0.25 * lib.einsum('ilde,jlgf,gfde->ij',t2_1_b, t2_1_b, eris_VVVV, optimize = True)
            del eris_VVVV

        else:

            temp_t2b_VVVV = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))   
            temp_t2b_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = adc.imds.t2_1_vvvv[2][:] 
            temp_t2b_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -adc.imds.t2_1_vvvv[2][:]

            M_ij_b += 2*0.25 * lib.einsum('ilgf,jlgf->ij', temp_t2b_VVVV, t2_1_b, optimize = True)
            del temp_t2b_VVVV

        M_ij_b += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)

        M_ij_a +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ooVV , optimize = True)
        M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_OOVV , optimize = True)
        M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_b, t2_1_b, eris_OVVO , optimize = True)

        M_ij_b -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_b, t2_1_b, eris_OOVV, optimize = True)
        M_ij_b += lib.einsum('ilde,jmdf,lefm->ij',t2_1_b, t2_1_b, eris_OVVO, optimize = True)

        M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ooOO, optimize = True)
        M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        del t2_1_b

        t2_1_ab = t2[0][1][:]

        M_ij_a += lib.einsum('mled,jlfd,mefi->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= lib.einsum('mled,jlfd,mife->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= lib.einsum('mlde,jldf,mife->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)

        M_ij_b += lib.einsum('lmde,ljdf,mefi->ij',t2_1_ab, t2_1_ab, eris_OVVO,optimize = True)
        M_ij_b -= lib.einsum('lmde,ljdf,mife->ij',t2_1_ab, t2_1_ab, eris_OOVV,optimize = True)
        M_ij_b -= lib.einsum('lmed,ljfd,mife->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)

        M_ij_a += lib.einsum('mled,ilfd,mefj->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= lib.einsum('mled,ilfd,mjfe->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= lib.einsum('mlde,ildf,mjfe->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)

        M_ij_b += lib.einsum('lmde,lidf,mefj->ij',t2_1_ab, t2_1_ab, eris_OVVO ,optimize = True)
        M_ij_b -= lib.einsum('lmde,lidf,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOVV ,optimize = True)
        M_ij_b -= lib.einsum('lmed,lifd,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)

        M_ij_a += lib.einsum('lmde,jnde,limn->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)
        M_ij_b += lib.einsum('mled,njed,mnli->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)
        
        if isinstance(eris.vVvV_p,np.ndarray):
            eris_vVvV = eris.vVvV_p
            eris_vVvV = eris_vVvV.reshape(nvir_a,nvir_b,nvir_a,nvir_b)
            M_ij_a +=lib.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,eris_vVvV, optimize = True)
            temp = lib.einsum('ljfg,fged->ljed',t2_1_ab,eris_vVvV, optimize = True)
            M_ij_b +=lib.einsum('lied,ljed->ij',t2_1_ab, temp, optimize = True)
            eris_vVvV = eris_vVvV.reshape(nvir_a*nvir_b,nvir_a*nvir_b)

        else:

            t2_vVvV = adc.imds.t2_1_vvvv[1][:]

            M_ij_a +=lib.einsum('ilgf,jlgf->ij', t2_vVvV, t2_1_ab, optimize = True)
            M_ij_b +=lib.einsum('lied,ljed->ij',t2_1_ab, t2_vVvV, optimize = True)
            del t2_vVvV

        M_ij_a +=lib.einsum('inde,lmde,jlnm->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b +=lib.einsum('nied,mled,nmjl->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_oovv , optimize = True)
        M_ij_a -=lib.einsum('mlfd,mled,jfei->ij',t2_1_ab, t2_1_ab, eris_ovvo , optimize = True)
        M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ooVV , optimize = True)
        M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_OOVV , optimize = True)
        M_ij_b -=lib.einsum('lmdf,lmde,jfei->ij',t2_1_ab, t2_1_ab, eris_OVVO , optimize = True)
        M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_OOvv , optimize = True)

        M_ij_a -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOVV, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_ab, eris_OVVO, optimize = True)
        M_ij_a -= lib.einsum('iled,jmfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOvv, optimize = True)

        M_ij_b -= lib.einsum('lied,mjfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_oovv, optimize = True)
        M_ij_b += lib.einsum('lied,mjfd,lefm->ij',t2_1_ab, t2_1_ab, eris_ovvo, optimize = True)
        M_ij_b -= lib.einsum('lide,mjdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_ooVV, optimize = True)

        M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a += lib.einsum('nled,mled,jmni->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b += lib.einsum('lnde,lmde,jmni->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)

        del t2_1_ab

    if adc.method_type == 'ip-cvs':
        M_ij_a = M_ij_a[:ncvs,:ncvs]
        M_ij_b = M_ij_b[:ncvs,:ncvs]
    
    M_ij = (M_ij_a, M_ij_b)
    cput0 = log.timer_debug1("Completed M_ij ADC(3) calculation", *cput0)
    
    '''
    E_a,_ = np.linalg.eigh(M_ij_a[:2,:2])
    E_b,_ = np.linalg.eigh(M_ij_b[:2,:2])
    print("Mij_a: ", E_a)
    print("Mij_b: ", E_b)
    exit()
    '''
    
    return M_ij

def get_imds_ip_cvs(adc, eris=None):

    cput0 = (time.clock(), time.time())
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
    ncvs = adc.ncvs

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

    eris_ovvo = eris.ovvo
    eris_OVVO = eris.OVVO
    eris_ovVO = eris.ovVO
    eris_OVvo = eris.OVvo

    # i-j block
    # Zeroth-order terms

    M_ij_a = lib.einsum('ij,j->ij', idn_occ_a[:ncvs,:ncvs] ,e_occ_a[:ncvs])
    M_ij_b = lib.einsum('ij,j->ij', idn_occ_b[:ncvs,:ncvs] ,e_occ_b[:ncvs])

    # Second-order terms

    t2_1_a = t2[0][0][:]

    M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a[:ncvs,:,:,:], t2_1_a[:ncvs,:,:,:], optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a[:ncvs,:,:,:], t2_1_a[:ncvs,:,:,:], optimize=True)
    M_ij_a += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_a[:ncvs,:,:,:], eris_ovvo[:ncvs,:,:,:], optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_a[:ncvs,:,:,:], eris_ovvo[:ncvs,:,:,:], optimize=True)
    M_ij_a += 0.5 *  lib.einsum('jlde,idel->ij',t2_1_a[:ncvs,:,:,:], eris_ovvo[:ncvs,:,:,:], optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('jlde,ldei->ij',t2_1_a[:ncvs,:,:,:], eris_ovvo[:,:,:,:ncvs], optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_a[:ncvs,:,:,:],t2_1_a[:ncvs,:,:,:], optimize=True)
    M_ij_a -= 0.25 *  lib.einsum('i,ij->ij',e_occ_a[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    M_ij_a -= 0.25 *  lib.einsum('j,ij->ij',e_occ_a[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    del M_ij_t
    del t2_1_a

    t2_1_b = t2[0][2][:]

    M_ij_b +=  lib.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b[:ncvs,:,:,:], t2_1_b[:ncvs,:,:,:], optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b[:ncvs,:,:,:], t2_1_b[:ncvs,:,:,:], optimize=True)
    M_ij_b += 0.5 *  lib.einsum('ilde,jdel->ij',t2_1_b[:ncvs,:,:,:], eris_OVVO[:ncvs,:,:,:], optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('ilde,jedl->ij',t2_1_b[:ncvs,:,:,:], eris_OVVO[:ncvs,:,:,:], optimize=True)
    M_ij_b += 0.5 *  lib.einsum('jlde,idel->ij',t2_1_b[:ncvs,:,:,:], eris_OVVO[:ncvs,:,:,:], optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('jlde,ldei->ij',t2_1_b[:ncvs,:,:,:], eris_OVVO[:,:,:,:ncvs], optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_b[:ncvs,:,:,:], t2_1_b[:ncvs,:,:,:], optimize=True)
    M_ij_b -= 0.25 *  lib.einsum('i,ij->ij',e_occ_b[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    M_ij_b -= 0.25 *  lib.einsum('j,ij->ij',e_occ_b[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    del M_ij_t
    del t2_1_b

    t2_1_ab = t2[0][1][:]
    M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab[:ncvs,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize=True)
    M_ij_a +=  lib.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab[:ncvs,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize=True)

    M_ij_b +=  lib.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab[:,:ncvs,:,:], t2_1_ab[:,:ncvs,:,:], optimize=True)
    M_ij_b +=  lib.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab[:,:ncvs,:,:], t2_1_ab[:,:ncvs,:,:], optimize=True)

    M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab[:ncvs,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize=True)
    M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab[:ncvs,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize=True)

    M_ij_b -= 0.5*lib.einsum('l,lide,ljde->ij',e_occ_a,t2_1_ab[:,:ncvs,:,:], t2_1_ab[:,:ncvs,:,:], optimize=True)
    M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab[:,:ncvs,:,:], t2_1_ab[:,:ncvs,:,:], optimize=True)

    M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_ab[:ncvs,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize=True)
    M_ij_a -= 0.5 *  lib.einsum('i,ij->ij',e_occ_a[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    M_ij_a -= 0.5 *  lib.einsum('j,ij->ij',e_occ_a[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    del M_ij_t

    M_ij_t = lib.einsum('lied,ljed->ij', t2_1_ab[:,:ncvs,:,:], t2_1_ab[:,:ncvs,:,:], optimize=True)
    M_ij_b -= 0.5 *  lib.einsum('i,ij->ij',e_occ_b[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    M_ij_b -= 0.5 *  lib.einsum('j,ij->ij',e_occ_b[:ncvs], M_ij_t[:ncvs,:ncvs], optimize=True) 
    del M_ij_t

    M_ij_a += lib.einsum('ilde,jdel->ij',t2_1_ab[:ncvs,:,:,:], eris_ovVO[:ncvs,:,:,:], optimize=True)
    M_ij_b += lib.einsum('lied,ledj->ij',t2_1_ab[:,:ncvs,:,:], eris_ovVO[:,:,:,:ncvs], optimize=True)
    M_ij_a += lib.einsum('jlde,idel->ij',t2_1_ab[:ncvs,:,:,:], eris_ovVO[:ncvs,:,:,:], optimize=True)
    M_ij_b += lib.einsum('ljed,ledi->ij',t2_1_ab[:,:ncvs,:,:], eris_ovVO[:,:,:,:ncvs], optimize=True)
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

        M_ij_a += 0.5* lib.einsum('ilde,jdel->ij',t2_2_a, eris_ovvo, optimize=True)
        M_ij_a -= 0.5* lib.einsum('ilde,jedl->ij',t2_2_a, eris_ovvo, optimize=True)

        M_ij_a += 0.5* lib.einsum('jlde,ledi->ij',t2_2_a, eris_ovvo, optimize=True)
        M_ij_a -= 0.5* lib.einsum('jlde,iedl->ij',t2_2_a, eris_ovvo, optimize=True)

        M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  lib.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)

        M_ij_a -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_a,t2_2_a, optimize=True)
        M_ij_a -= 0.25 *  lib.einsum('i,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.25 *  lib.einsum('i,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.25 *  lib.einsum('j,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.25 *  lib.einsum('j,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        del M_ij_t
        del t2_2_a
        del t2_1_a

        t2_1_b = t2[0][2][:]
        t2_2_b = t2[1][2][:]

        M_ij_b += 0.5* lib.einsum('ilde,jdel->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b -= 0.5* lib.einsum('ilde,jedl->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b += 0.5* lib.einsum('jlde,ledi->ij',t2_2_b, eris_OVVO, optimize=True)
        M_ij_b -= 0.5* lib.einsum('jlde,iedl->ij',t2_2_b, eris_OVVO, optimize=True)

        M_ij_b +=  lib.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  lib.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_b, t2_2_b, optimize=True)
        M_ij_b -= 0.25 *  lib.einsum('i,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.25 *  lib.einsum('i,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.25 *  lib.einsum('j,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.25 *  lib.einsum('j,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        del M_ij_t
        del t2_2_b
        del t2_1_b

        t2_1_ab = t2[0][1][:]
        t2_2_ab = t2[1][1][:]

        M_ij_a += lib.einsum('ilde,jdel->ij',t2_2_ab, eris_ovVO, optimize=True)
        M_ij_b += lib.einsum('lied,ledj->ij',t2_2_ab, eris_ovVO, optimize=True)
        M_ij_a += lib.einsum('jlde,ledi->ij',t2_2_ab, eris_OVvo, optimize=True)
        M_ij_b += lib.einsum('ljed,ledi->ij',t2_2_ab, eris_ovVO, optimize=True)

        M_ij_a +=  lib.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  lib.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  lib.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  lib.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a +=  lib.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  lib.einsum('d,jled,iled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b +=  lib.einsum('d,ljde,lide->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  lib.einsum('d,ljed,lied->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5*lib.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*lib.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_t = lib.einsum('ilde,jlde->ij', t2_1_ab, t2_2_ab, optimize=True)
        M_ij_a -= 0.5 *  lib.einsum('i,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.5 *  lib.einsum('i,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.5 *  lib.einsum('j,ij->ij',e_occ_a, M_ij_t, optimize=True) 
        M_ij_a -= 0.5 *  lib.einsum('j,ji->ij',e_occ_a, M_ij_t, optimize=True) 
        del M_ij_t

        M_ij_t = lib.einsum('lied,ljed->ij', t2_1_ab, t2_2_ab, optimize=True)
        M_ij_b -= 0.5 *  lib.einsum('i,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.5 *  lib.einsum('i,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.5 *  lib.einsum('j,ij->ij',e_occ_b, M_ij_t, optimize=True) 
        M_ij_b -= 0.5 *  lib.einsum('j,ji->ij',e_occ_b, M_ij_t, optimize=True) 
        del M_ij_t
        del t2_1_ab
        del t2_2_ab

        t2_1_a = t2[0][0][:]
        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_a, t2_1_a, optimize=True)
        M_ij_a -= lib.einsum('mejf,mefi->ij',M_ij_t, eris_ovvo, optimize=True) 
        M_ij_a -= lib.einsum('mejf,mefi->ji',M_ij_t, eris_ovvo, optimize=True) 
        M_ij_a += lib.einsum('mejf,mife->ij',M_ij_t, eris_oovv, optimize=True) 
        M_ij_a += lib.einsum('mejf,mife->ji',M_ij_t, eris_oovv, optimize=True) 
        del M_ij_t

        M_ij_a += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)

        if isinstance(eris.vvvv_p,np.ndarray):
            eris_vvvv = radc_ao2mo.unpack_eri_2(eris.vvvv_p, nvir_a)
            M_ij_a += 0.25 * lib.einsum('ilde,jlgf,gfde->ij',t2_1_a, t2_1_a, eris_vvvv, optimize = True)
            del eris_vvvv

        else:

            temp_t2a_vvvv = np.zeros((nocc_a,nocc_a,nvir_a,nvir_a))   
            temp_t2a_vvvv[:,:,ab_ind_a[0],ab_ind_a[1]] = adc.imds.t2_1_vvvv[0][:]    
            temp_t2a_vvvv[:,:,ab_ind_a[1],ab_ind_a[0]] = -adc.imds.t2_1_vvvv[0][:]

            M_ij_a += 2*0.25 * lib.einsum('ilgf,jlgf->ij', temp_t2a_vvvv, t2_1_a, optimize = True)
            del temp_t2a_vvvv 

        M_ij_a += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)
        M_ij_a -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a, t2_1_a,eris_oooo, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)
        M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_OOvv , optimize = True)

        M_ij_a -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_a, t2_1_a, eris_oovv, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_a, eris_ovvo, optimize = True)

        M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_a += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_a, t2_1_a, eris_oooo, optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooOO, optimize = True)

        t2_1_ab = t2[0][1][:]
        M_ij_a -= lib.einsum('lmde,jldf,mefi->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_b += lib.einsum('lmde,ljdf,mefi->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)
        M_ij_a -= lib.einsum('lmde,ildf,mefj->ij',t2_1_ab, t2_1_a, eris_OVvo,optimize = True)
        M_ij_b += lib.einsum('lmde,lidf,mefj->ij',t2_1_a, t2_1_ab, eris_ovVO ,optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_a, t2_1_ab, eris_ovVO, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_a, eris_OVvo, optimize = True)
        del t2_1_a

        t2_1_b = t2[0][2][:]
        M_ij_a += lib.einsum('lmde,jlfd,mefi->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)
        M_ij_b -= lib.einsum('mled,jldf,mefi->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_a += lib.einsum('lmde,ilfd,mefj->ij',t2_1_b, t2_1_ab, eris_OVvo ,optimize = True)
        M_ij_b -= lib.einsum('mled,ildf,mefj->ij',t2_1_ab, t2_1_b, eris_ovVO,optimize = True)
        M_ij_b += lib.einsum('ilde,mjfd,lefm->ij',t2_1_b, t2_1_ab, eris_OVvo, optimize = True)
        M_ij_b += lib.einsum('lied,jmdf,lefm->ij',t2_1_ab, t2_1_b, eris_ovVO, optimize = True)
        del t2_1_ab

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_b, t2_1_b, optimize=True)
        M_ij_b -= lib.einsum('mejf,mefi->ij',M_ij_t, eris_OVVO, optimize=True) 
        M_ij_b -= lib.einsum('mejf,mefi->ji',M_ij_t, eris_OVVO, optimize=True) 
        M_ij_b += lib.einsum('mejf,mife->ij',M_ij_t, eris_OOVV, optimize=True) 
        M_ij_b += lib.einsum('mejf,mife->ji',M_ij_t, eris_OOVV, optimize=True) 
        del M_ij_t

        M_ij_b += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)

        if isinstance(eris.VVVV_p,np.ndarray):
            eris_VVVV = radc_ao2mo.unpack_eri_2(eris.VVVV_p, nvir_b)
            M_ij_b += 0.25 * lib.einsum('ilde,jlgf,gfde->ij',t2_1_b, t2_1_b, eris_VVVV, optimize = True)
            del eris_VVVV

        else:

            temp_t2b_VVVV = np.zeros((nocc_b,nocc_b,nvir_b,nvir_b))   
            temp_t2b_VVVV[:,:,ab_ind_b[0],ab_ind_b[1]] = adc.imds.t2_1_vvvv[2][:] 
            temp_t2b_VVVV[:,:,ab_ind_b[1],ab_ind_b[0]] = -adc.imds.t2_1_vvvv[2][:]

            M_ij_b += 2*0.25 * lib.einsum('ilgf,jlgf->ij', temp_t2b_VVVV, t2_1_b, optimize = True)
            del temp_t2b_VVVV

        M_ij_b += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)
        M_ij_b -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_b, t2_1_b,eris_OOOO, optimize = True)

        M_ij_a +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ooVV , optimize = True)
        M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_OOVV , optimize = True)
        M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1_b, t2_1_b, eris_OVVO , optimize = True)

        M_ij_b -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_b, t2_1_b, eris_OOVV, optimize = True)
        M_ij_b += lib.einsum('ilde,jmdf,lefm->ij',t2_1_b, t2_1_b, eris_OVVO, optimize = True)

        M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ooOO, optimize = True)
        M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        M_ij_b += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1_b, t2_1_b, eris_OOOO, optimize = True)
        del t2_1_b

        t2_1_ab = t2[0][1][:]

        M_ij_a += lib.einsum('mled,jlfd,mefi->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= lib.einsum('mled,jlfd,mife->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= lib.einsum('mlde,jldf,mife->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)

        M_ij_b += lib.einsum('lmde,ljdf,mefi->ij',t2_1_ab, t2_1_ab, eris_OVVO,optimize = True)
        M_ij_b -= lib.einsum('lmde,ljdf,mife->ij',t2_1_ab, t2_1_ab, eris_OOVV,optimize = True)
        M_ij_b -= lib.einsum('lmed,ljfd,mife->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)

        M_ij_a += lib.einsum('mled,ilfd,mefj->ij',t2_1_ab, t2_1_ab, eris_ovvo ,optimize = True)
        M_ij_a -= lib.einsum('mled,ilfd,mjfe->ij',t2_1_ab, t2_1_ab, eris_oovv ,optimize = True)
        M_ij_a -= lib.einsum('mlde,ildf,mjfe->ij',t2_1_ab, t2_1_ab, eris_ooVV ,optimize = True)

        M_ij_b += lib.einsum('lmde,lidf,mefj->ij',t2_1_ab, t2_1_ab, eris_OVVO ,optimize = True)
        M_ij_b -= lib.einsum('lmde,lidf,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOVV ,optimize = True)
        M_ij_b -= lib.einsum('lmed,lifd,mjfe->ij',t2_1_ab, t2_1_ab, eris_OOvv ,optimize = True)

        M_ij_a += lib.einsum('lmde,jnde,limn->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)
        M_ij_b += lib.einsum('mled,njed,mnli->ij',t2_1_ab ,t2_1_ab,eris_ooOO, optimize = True)
        
        if isinstance(eris.vVvV_p,np.ndarray):
            eris_vVvV = eris.vVvV_p
            eris_vVvV = eris_vVvV.reshape(nvir_a,nvir_b,nvir_a,nvir_b)
            M_ij_a +=lib.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,eris_vVvV, optimize = True)
            temp = lib.einsum('ljfg,fged->ljed',t2_1_ab,eris_vVvV, optimize = True)
            M_ij_b +=lib.einsum('lied,ljed->ij',t2_1_ab, temp, optimize = True)
            eris_vVvV = eris_vVvV.reshape(nvir_a*nvir_b,nvir_a*nvir_b)

        else:

            t2_vVvV = adc.imds.t2_1_vvvv[1][:]

            M_ij_a +=lib.einsum('ilgf,jlgf->ij', t2_vVvV, t2_1_ab, optimize = True)
            M_ij_b +=lib.einsum('lied,ljed->ij',t2_1_ab, t2_vVvV, optimize = True)
            del t2_vVvV

        M_ij_a +=lib.einsum('inde,lmde,jlnm->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_b +=lib.einsum('nied,mled,nmjl->ij',t2_1_ab, t2_1_ab,eris_ooOO, optimize = True)

        M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_oovv , optimize = True)
        M_ij_a -=lib.einsum('mlfd,mled,jfei->ij',t2_1_ab, t2_1_ab, eris_ovvo , optimize = True)
        M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ooVV , optimize = True)
        M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_OOVV , optimize = True)
        M_ij_b -=lib.einsum('lmdf,lmde,jfei->ij',t2_1_ab, t2_1_ab, eris_OVVO , optimize = True)
        M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_OOvv , optimize = True)

        M_ij_a -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOVV, optimize = True)
        M_ij_a += lib.einsum('ilde,jmdf,lefm->ij',t2_1_ab, t2_1_ab, eris_OVVO, optimize = True)
        M_ij_a -= lib.einsum('iled,jmfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_OOvv, optimize = True)

        M_ij_b -= lib.einsum('lied,mjfd,lmfe->ij',t2_1_ab, t2_1_ab, eris_oovv, optimize = True)
        M_ij_b += lib.einsum('lied,mjfd,lefm->ij',t2_1_ab, t2_1_ab, eris_ovvo, optimize = True)
        M_ij_b -= lib.einsum('lide,mjdf,lmfe->ij',t2_1_ab, t2_1_ab, eris_ooVV, optimize = True)

        M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a += lib.einsum('nled,mled,jmni->ij',t2_1_ab, t2_1_ab, eris_oooo, optimize = True)
        M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)
        M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b += lib.einsum('lnde,lmde,jmni->ij',t2_1_ab, t2_1_ab, eris_OOOO, optimize = True)
        M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooOO, optimize = True)

        del t2_1_ab

    M_ij = (M_ij_a, M_ij_b)
    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

    """ 
    E_a,_ = np.linalg.eigh(M_ij_a)
    E_b,_ = np.linalg.eigh(M_ij_b)
    print("Mij_a: ", E_a)
    print("Mij_b: ", E_b)
    exit()
    """

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
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv_p, np.ndarray): 
#
#            eris_oovv = eris.oovv
#            eris_ovvo = eris.ovvo
#            eris_OOVV = eris.OOVV
#            eris_OVVO = eris.OVVO
#            eris_OOvv = eris.OOvv
#            eris_ooVV = eris.ooVV
#
#            eris_vvvv = eris.vvvv_p
#            temp = np.zeros((nocc_a, eris_vvvv.shape[0]))
#            temp[:] += np.diag(eris_vvvv)
#            diag[s_aaa:f_aaa] += temp.reshape(-1)
#
#            eris_VVVV = eris.VVVV_p
#            temp = np.zeros((nocc_b, eris_VVVV.shape[0]))
#            temp[:] += np.diag(eris_VVVV)
#            diag[s_bbb:f_bbb] += temp.reshape(-1)
#
#            eris_vVvV = eris.vVvV_p
#            temp = np.zeros((nocc_b, eris_vVvV.shape[0]))
#            temp[:] += np.diag(eris_vVvV)
#            diag[s_bab:f_bab] += temp.reshape(-1)
#            
#            temp = np.zeros((nocc_a, nvir_a, nvir_b))
#            temp[:] += np.diag(eris_vVvV).reshape(nvir_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#                
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
#            eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
#  
#            temp = np.zeros((eris_ovov_p.shape[0],nvir_a))
#            temp.T[:] += np.diagonal(eris_ovov_p)
#            temp = temp.reshape(nocc_a, nvir_a, nvir_a)
#            diag[s_aaa:f_aaa] += -temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aaa:f_aaa] += -temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)
#
#            eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
#            eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
#            eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
#  
#            temp = np.zeros((eris_OVOV_p.shape[0],nvir_b))
#            temp.T[:] += np.diagonal(eris_OVOV_p)
#            temp = temp.reshape(nocc_b, nvir_b, nvir_b)
#            diag[s_bbb:f_bbb] += -temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_bbb:f_bbb] += -temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)
#
#            temp = np.zeros((nvir_a, nocc_b, nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(1,0,2))
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#
#            temp = np.zeros((nvir_b, nocc_a, nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(1,0,2))
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#
#            eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
#            eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)
#  
#            temp = np.zeros((nvir_b, nocc_b, nvir_a))
#            temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(1,2,0))
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#
#            eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
#            eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)
#
#            temp = np.zeros((nvir_a, nocc_a, nvir_b))
#            temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(1,2,0))
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#        else :
#           raise Exception("Precond not available for out-of-core and density-fitted algo")

    return diag


def ip_adc_diag(adc,M_ij=None,eris=None):
   
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    if M_ij is None:
        M_ij = adc.get_imds()

    M_ij_a, M_ij_b = M_ij[0], M_ij[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

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

    diag = np.zeros(dim)

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

    #diag[s_aaa:f_bbb] = 5

    ###### Additional terms for the preconditioner ####
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv_p, np.ndarray): 
#
#            eris_oooo = eris.oooo
#            eris_OOOO = eris.OOOO
#            eris_ooOO = eris.ooOO
#            eris_oovv = eris.oovv
#            eris_OOVV = eris.OOVV
#            eris_ooVV = eris.ooVV
#            eris_OOvv = eris.OOvv
#            eris_ovvo = eris.ovvo
#            eris_OVVO = eris.OVVO
#
#            eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
#            eris_oooo_p -= np.ascontiguousarray(eris_oooo_p.transpose(0,1,3,2))
#            eris_oooo_p = eris_oooo_p.reshape(nocc_a*nocc_a, nocc_a*nocc_a)
#  
#            temp = np.zeros((nvir_a,eris_oooo_p.shape[0]))
#            temp[:] += np.diagonal(eris_oooo_p)
#            temp = temp.reshape(nvir_a, nocc_a, nocc_a)
#            diag[s_aaa:f_aaa] += -temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            eris_OOOO_p = np.ascontiguousarray(eris_OOOO.transpose(0,2,1,3))
#            eris_OOOO_p -= np.ascontiguousarray(eris_OOOO_p.transpose(0,1,3,2))
#            eris_OOOO_p = eris_OOOO_p.reshape(nocc_b*nocc_b, nocc_b*nocc_b)
#  
#            temp = np.zeros((nvir_b,eris_OOOO_p.shape[0]))
#            temp[:] += np.diagonal(eris_OOOO_p)
#            temp = temp.reshape(nvir_b, nocc_b, nocc_b)
#            diag[s_bbb:f_bbb] += -temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            eris_oOoO_p = np.ascontiguousarray(eris_ooOO.transpose(0,2,1,3))
#            eris_oOoO_p = eris_oOoO_p.reshape(nocc_a*nocc_b, nocc_a*nocc_b)
#  
#            temp = np.zeros((nvir_b, eris_oOoO_p.shape[0]))
#            temp[:] += np.diag(eris_oOoO_p)
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#            
#            temp = np.zeros((nvir_a, eris_oOoO_p.shape[0]))
#            temp[:] += np.diag(eris_oOoO_p.T)
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#            
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
#            eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
#  
#            temp = np.zeros((nocc_a,nocc_a,nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a,nvir_a)
#            temp = np.ascontiguousarray(temp.T)
#            diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
#            eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
#            eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
#  
#            temp = np.zeros((nocc_b,nocc_b,nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b,nvir_b)
#            temp = np.ascontiguousarray(temp.T)
#            diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            temp = np.zeros((nocc_a, nocc_b, nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            temp = np.zeros((nocc_b, nocc_a, nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#
#            eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
#            eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)
#  
#            temp = np.zeros((nocc_b, nocc_a, nvir_b))
#            temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
#            eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)
#
#            temp = np.zeros((nocc_a, nocc_b, nvir_a))
#            temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#        else :
#           raise Exception("Precond not available for out-of-core and density-fitted algo")

    if adc.ncvs > 0:
        
        shift = -100000000.0
        #shift = 0
        ncore = adc.ncvs
      
        diag[(s_a+ncore):f_a] += shift
        diag[(s_b+ncore):f_b] += shift
        
        temp = np.zeros((nvir_a, nocc_a, nocc_a))
        temp[:,ij_ind_a[0],ij_ind_a[1]] = diag[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
        temp[:,ij_ind_a[1],ij_ind_a[0]] = -diag[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        #temp[:,:ncore,:ncore] += shift

        diag[s_aaa:f_aaa] = temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1).copy()

        temp = diag[s_bab:f_bab].copy()
        temp = temp.reshape((nvir_b, nocc_a, nocc_b))
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        #temp[:,:ncore,:ncore] += shift

        diag[s_bab:f_bab] = temp.reshape(-1).copy()

        temp = diag[s_aba:f_aba].copy()
        temp = temp.reshape((nvir_a, nocc_b, nocc_a))
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        #temp[:,:ncore,:ncore] += shift

        diag[s_aba:f_aba] = temp.reshape(-1).copy()

        temp = np.zeros((nvir_b, nocc_b, nocc_b))
        temp[:,ij_ind_b[0],ij_ind_b[1]] = diag[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
        temp[:,ij_ind_b[1],ij_ind_b[0]] = -diag[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
        
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        #temp[:,:ncore,:ncore] = shift
        
        diag[s_bbb:f_bbb] = temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1).copy()

        """
        print('diag[s_a:f_a]',             np.linalg.norm(diag[s_a:f_a]))
        print(np.column_stack(np.unique(np.around(diag[s_a:f_a], decimals=8), return_counts=True))) 
        print('diag[s_b:f_b]' ,            np.linalg.norm(diag[s_b:f_b]))
        print(np.column_stack(np.unique(np.around(diag[s_b:f_b], decimals=8), return_counts=True))) 
        print('diag[s_aaa:f_aaa]', np.linalg.norm(diag[s_aaa:f_aaa]))      
        print(np.column_stack(np.unique(np.around(diag[s_aaa:f_aaa], decimals=8), return_counts=True))) 
        print('diag[s_bab:f_bab]', np.linalg.norm(diag[s_bab:f_bab]))
        print(np.column_stack(np.unique(np.around(diag[s_bab:f_bab], decimals=8), return_counts=True))) 
        print('diag[s_aba:f_aba]', np.linalg.norm(diag[s_aba:f_aba]))
        print(np.column_stack(np.unique(np.around(diag[s_aba:f_aba], decimals=8), return_counts=True))) 
        print('diag[s_bbb:f_bbb]', np.linalg.norm(diag[s_bbb:f_bbb]))
        print(np.column_stack(np.unique(np.around(diag[s_bbb:f_bbb], decimals=8), return_counts=True))) 
        exit()
        """
    diag = -diag
    return diag


def ip_cvs_adc_diag(adc,M_ij=None,eris=None):
   
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    if M_ij is None:
        M_ij = adc.get_imds()

    M_ij_a, M_ij_b = M_ij[0], M_ij[1]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = nvir_a * ncvs * (ncvs - 1) // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a  
    n_doubles_bab_ecc = nvir_b * ncvs * ncvs
    n_doubles_bab_ecv = nvir_b * ncvs * nval_b
    n_doubles_bab_evc = nvir_b * nval_a * ncvs
    n_doubles_aba_ecc = nvir_a * ncvs * ncvs
    n_doubles_aba_ecv = nvir_a * ncvs * nval_a
    n_doubles_aba_evc = nvir_a * nval_b * ncvs
    n_doubles_bbb_ecc = nvir_b * ncvs * (ncvs - 1) // 2
    n_doubles_bbb_ecv = nvir_b * ncvs * nval_b

    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_aaa_ecv + n_doubles_bab_ecc + n_doubles_bab_ecv + n_doubles_bab_evc + n_doubles_aba_ecc + n_doubles_aba_ecv + n_doubles_aba_evc + n_doubles_bbb_ecc + n_doubles_bbb_ecv

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bab_ecc = f_aaa_ecv
    f_bab_ecc = s_bab_ecc + n_doubles_bab_ecc
    s_bab_ecv = f_bab_ecc
    f_bab_ecv = s_bab_ecv + n_doubles_bab_ecv
    s_bab_evc = f_bab_ecv
    f_bab_evc = s_bab_evc + n_doubles_bab_evc
    s_aba_ecc = f_bab_evc
    f_aba_ecc = s_aba_ecc + n_doubles_aba_ecc
    s_aba_ecv = f_aba_ecc
    f_aba_ecv = s_aba_ecv + n_doubles_aba_ecv
    s_aba_evc = f_aba_ecv
    f_aba_evc = s_aba_evc + n_doubles_aba_evc
    s_bbb_ecc = f_aba_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_n_a_ecc = D_n_a[:, :ncvs, :ncvs].copy()
    D_aij_a_ecv = D_n_a[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_a_ecc = D_n_a_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_n_b_ecc = D_n_b[:, :ncvs, :ncvs].copy()
    D_aij_b_ecv = D_n_b[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_b_ecc = D_n_b_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_n_bab = D_n_bab.reshape((nvir_b,nocc_a,nocc_b))
    D_aij_bab_ecc = D_n_bab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_bab_ecv = D_n_bab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_bab_evc = D_n_bab[:, ncvs:, :ncvs].reshape(-1)

    d_ij_ba = e_occ_b[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ba.reshape(-1)
    D_n_aba = D_n_aba.reshape(-1)
    D_n_aba = D_n_aba.reshape((nvir_a,nocc_b,nocc_a))
    D_aij_aba_ecc = D_n_aba[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aba_ecv = D_n_aba[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aba_evc = D_n_aba[:, ncvs:, :ncvs].reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    diag[s_a:f_a] = M_ij_a_diag.copy()
    diag[s_b:f_b] = M_ij_b_diag.copy()

    # Compute precond in 2p1h-2p1h block
    
    diag[s_aaa_ecc:f_aaa_ecc] = D_aij_a_ecc.copy()
    diag[s_aaa_ecv:f_aaa_ecv] = D_aij_a_ecv.copy()
    diag[s_bab_ecc:f_bab_ecc] = D_aij_bab_ecc.copy()
    diag[s_bab_ecv:f_bab_ecv] = D_aij_bab_ecv.copy()
    diag[s_bab_evc:f_bab_evc] = D_aij_bab_evc.copy()
    diag[s_aba_ecc:f_aba_ecc] = D_aij_aba_ecc.copy()
    diag[s_aba_ecv:f_aba_ecv] = D_aij_aba_ecv.copy()
    diag[s_aba_evc:f_aba_evc] = D_aij_aba_evc.copy()
    diag[s_bbb_ecc:f_bbb_ecc] = D_aij_b_ecc.copy()
    diag[s_bbb_ecv:f_bbb_ecv] = D_aij_b_ecv.copy()
    
    #diag[s_aaa_ecc:f_bbb_ecv] = 5

    ###### Additional terms for the preconditioner ####
#    if (method == "adc(2)-x" or method == "adc(3)"):
#
#        if eris is None:
#            eris = adc.transform_integrals()
#
#        if isinstance(eris.vvvv_p, np.ndarray): 
#
#            eris_oooo = eris.oooo
#            eris_OOOO = eris.OOOO
#            eris_ooOO = eris.ooOO
#            eris_oovv = eris.oovv
#            eris_OOVV = eris.OOVV
#            eris_ooVV = eris.ooVV
#            eris_OOvv = eris.OOvv
#            eris_ovvo = eris.ovvo
#            eris_OVVO = eris.OVVO
#
#            eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
#            eris_oooo_p -= np.ascontiguousarray(eris_oooo_p.transpose(0,1,3,2))
#            eris_oooo_p = eris_oooo_p.reshape(nocc_a*nocc_a, nocc_a*nocc_a)
#  
#            temp = np.zeros((nvir_a,eris_oooo_p.shape[0]))
#            temp[:] += np.diagonal(eris_oooo_p)
#            temp = temp.reshape(nvir_a, nocc_a, nocc_a)
#            diag[s_aaa:f_aaa] += -temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            eris_OOOO_p = np.ascontiguousarray(eris_OOOO.transpose(0,2,1,3))
#            eris_OOOO_p -= np.ascontiguousarray(eris_OOOO_p.transpose(0,1,3,2))
#            eris_OOOO_p = eris_OOOO_p.reshape(nocc_b*nocc_b, nocc_b*nocc_b)
#  
#            temp = np.zeros((nvir_b,eris_OOOO_p.shape[0]))
#            temp[:] += np.diagonal(eris_OOOO_p)
#            temp = temp.reshape(nvir_b, nocc_b, nocc_b)
#            diag[s_bbb:f_bbb] += -temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            eris_oOoO_p = np.ascontiguousarray(eris_ooOO.transpose(0,2,1,3))
#            eris_oOoO_p = eris_oOoO_p.reshape(nocc_a*nocc_b, nocc_a*nocc_b)
#  
#            temp = np.zeros((nvir_b, eris_oOoO_p.shape[0]))
#            temp[:] += np.diag(eris_oOoO_p)
#            diag[s_bab:f_bab] += -temp.reshape(-1)
#            
#            temp = np.zeros((nvir_a, eris_oOoO_p.shape[0]))
#            temp[:] += np.diag(eris_oOoO_p.T)
#            diag[s_aba:f_aba] += -temp.reshape(-1)
#            
#            eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3))
#            eris_ovov_p -= np.ascontiguousarray(eris_ovvo.transpose(0,2,3,1))
#            eris_ovov_p = eris_ovov_p.reshape(nocc_a*nvir_a, nocc_a*nvir_a)
#  
#            temp = np.zeros((nocc_a,nocc_a,nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a,nvir_a)
#            temp = np.ascontiguousarray(temp.T)
#            diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
#
#            eris_OVOV_p = np.ascontiguousarray(eris_OOVV.transpose(0,2,1,3))
#            eris_OVOV_p -= np.ascontiguousarray(eris_OVVO.transpose(0,2,3,1))
#            eris_OVOV_p = eris_OVOV_p.reshape(nocc_b*nvir_b, nocc_b*nvir_b)
#  
#            temp = np.zeros((nocc_b,nocc_b,nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b,nvir_b)
#            temp = np.ascontiguousarray(temp.T)
#            diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            temp = np.ascontiguousarray(temp.transpose(0,2,1))
#            diag[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
#
#            temp = np.zeros((nocc_a, nocc_b, nvir_b))
#            temp[:] += np.diagonal(eris_OVOV_p).reshape(nocc_b, nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            temp = np.zeros((nocc_b, nocc_a, nvir_a))
#            temp[:] += np.diagonal(eris_ovov_p).reshape(nocc_a, nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(2,0,1))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#
#            eris_oVoV_p = np.ascontiguousarray(eris_ooVV.transpose(0,2,1,3))
#            eris_oVoV_p = eris_oVoV_p.reshape(nocc_a*nvir_b, nocc_a*nvir_b)
#  
#            temp = np.zeros((nocc_b, nocc_a, nvir_b))
#            temp[:] += np.diagonal(eris_oVoV_p).reshape(nocc_a,nvir_b)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s_bab:f_bab] += temp.reshape(-1)
#
#            eris_OvOv_p = np.ascontiguousarray(eris_OOvv.transpose(0,2,1,3))
#            eris_OvOv_p = eris_OvOv_p.reshape(nocc_b*nvir_a, nocc_b*nvir_a)
#
#            temp = np.zeros((nocc_a, nocc_b, nvir_a))
#            temp[:] += np.diagonal(eris_OvOv_p).reshape(nocc_b,nvir_a)
#            temp = np.ascontiguousarray(temp.transpose(2,1,0))
#            diag[s_aba:f_aba] += temp.reshape(-1)
#        else :
#           raise Exception("Precond not available for out-of-core and density-fitted algo")
    """
    print('M_CVS diagonal shape: ', diag.shape)
    print('M_CVS diagonal norm: ', np.linalg.norm(diag))
    unique_elements, counts_elements = np.unique(np.around(diag, decimals=8), return_counts=True)
    print("M_CVS unique diagonal elements and degeneracies:")
    print(np.column_stack((unique_elements, counts_elements)))
    exit()
    """
    """ 
    print('diag[s_a:f_a]',             np.linalg.norm(diag[s_a:f_a]))
    print(np.column_stack(np.unique(np.around(diag[s_a:f_a], decimals=8), return_counts=True))) 
    print('diag[s_b:f_b]' ,            np.linalg.norm(diag[s_b:f_b]))
    print(np.column_stack(np.unique(np.around(diag[s_b:f_b], decimals=8), return_counts=True))) 
    print('diag[s_aaa_ecc:f_aaa_ecv]', np.linalg.norm(diag[s_aaa_ecc:f_aaa_ecv]))      
    print(np.column_stack(np.unique(np.around(diag[s_aaa_ecc:f_aaa_ecv], decimals=8), return_counts=True))) 
    print('diag[s_bab_ecc:f_bab_evc]', np.linalg.norm(diag[s_bab_ecc:f_bab_evc]))
    print(np.column_stack(np.unique(np.around(diag[s_bab_ecc:f_bab_evc], decimals=8), return_counts=True))) 
    print('diag[s_aba_ecc:f_aba_evc]', np.linalg.norm(diag[s_aba_ecc:f_aba_evc]))
    print(np.column_stack(np.unique(np.around(diag[s_aba_ecc:f_aba_evc], decimals=8), return_counts=True))) 
    print('diag[s_bbb_ecc:f_bbb_ecv]', np.linalg.norm(diag[s_bbb_ecc:f_bbb_ecv]))
    print(np.column_stack(np.unique(np.around(diag[s_bbb_ecc:f_bbb_ecv], decimals=8), return_counts=True))) 
    exit()
    """
    """
    print("diag^-1",np.linalg.norm(diag**-1))
    exit()
    """
    diag = -diag
    return diag

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
            del vvvv
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
            del vvvv
            a += k
    else :
        raise Exception("Unknown vvvv type") 

    return r2_vvvv


def ea_adc_matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

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
                eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
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
        del temp

        if isinstance(eris.OVvv, type(None)):
            chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc_b

        a = 0
        temp = np.zeros((nocc_b, nvir_a, nvir_b))
        for p in range(0,nocc_b,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
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
                eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
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
                eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
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
               t2_1_a = adc.t2[0][0][:]
               t2_1_ab = adc.t2[0][1][:]

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]

               r_aaa = r_aaa.reshape(nocc_a,-1)
               temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_a_t,r_aaa)
               del t2_1_a_t
               s[s_a:f_a] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
               s[s_a:f_a] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)
               del temp

               temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1_ab,r_bab)
               s[s_a:f_a] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovOO, optimize=True)
               del temp_1

               temp_1 = -lib.einsum('mlwz,jzw->jlm',t2_1_ab,r_aba)
               s[s_b:f_b] -= lib.einsum('jlm,lamj->a',temp_1, eris_OVoo, optimize=True)
               del temp_1

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
                       eris_ovvv = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
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
                       eris_ovVV = dfadc.get_ovvv_spin_df(adc, eris.Lov, eris.LVV, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
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

               temp  = lib.einsum('lxd,ilyd->ixy',temp_1_1,t2_1_a,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = lib.einsum('lyd,ilxd->ixy',temp_1_2,t2_1_a,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_2_3,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)
               
               del t2_1_a

               t2_1_b = adc.t2[0][2][:]

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               r_bbb = r_bbb.reshape(nocc_b,-1)
               temp = 0.5*lib.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
               del t2_1_b_t
               s[s_b:f_b] += lib.einsum('lmj,lamj->a',temp, eris_OVOO, optimize=True)
               s[s_b:f_b] -= lib.einsum('lmj,malj->a',temp, eris_OVOO, optimize=True)
               del temp

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
                       eris_OVVV = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.LVV, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
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
                       eris_OVvv = dfadc.get_ovvv_spin_df(adc, eris.LOV, eris.Lvv, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
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

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               temp = lib.einsum('b,lbmi->lmi',r_b,eris_OVOO)
               temp -= lib.einsum('b,mbli->lmi',r_b,eris_OVOO)
               s[s_bbb:f_bbb] += 0.5*lib.einsum('lmi,lmp->ip',temp, t2_1_b_t, optimize=True).reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_b,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = lib.einsum('lyd,ilxd->ixy',temp_1_4,t2_1_b,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)
               del t2_1_b

               temp_1 = lib.einsum('b,lbmi->lmi',r_a,eris_ovOO)
               s[s_bab:f_bab] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp_1 = lib.einsum('b,lbmi->mli',r_b,eris_OVoo)
               s[s_aba:f_aba] += lib.einsum('mli,mlyx->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp = lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp = lib.einsum('lyd,ilxd->ixy',temp_2_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1_ab,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp = lib.einsum('lxd,lidy->ixy',temp_2_3,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp = lib.einsum('lyd,lidx->ixy',temp_2_4,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)

               temp  = lib.einsum('lxd,ilyd->ixy',temp_1_3,t2_1_ab,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               del t2_1_ab

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        return s

        del temp_2_1
        del temp_1_3
        del temp_1_4
        del temp_1_1
        del temp_1_2
        del temp_2_3

    return sigma_


def cvs_projector(adc, r):

    ncore = adc.ncvs

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

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

    Pr = r.copy()

    Pr[(s_a+ncore):f_a] = 0.0
    Pr[(s_b+ncore):f_b] = 0.0

    temp = np.zeros((nvir_a, nocc_a, nocc_a))
    temp[:,ij_a[0],ij_a[1]] = Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
    temp[:,ij_a[1],ij_a[0]] = -Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    #temp[:,:ncore,:ncore] = 0.0

    Pr[s_aaa:f_aaa] = temp[:,ij_a[0],ij_a[1]].reshape(-1).copy()

    temp = Pr[s_bab:f_bab].copy()
    temp = temp.reshape((nvir_b, nocc_b, nocc_a))
    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    #temp[:,:ncore,:ncore] = 0.0

    Pr[s_bab:f_bab] = temp.reshape(-1).copy()

    temp = Pr[s_aba:f_aba].copy()
    temp = temp.reshape((nvir_a, nocc_a, nocc_b))
    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    #temp[:,:ncore,:ncore] = 0.0

    Pr[s_aba:f_aba] = temp.reshape(-1).copy()

    temp = np.zeros((nvir_b, nocc_b, nocc_b))
    temp[:,ij_b[0],ij_b[1]] = Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
    temp[:,ij_b[1],ij_b[0]] = -Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    #temp[:,:ncore,:ncore] = 0.0

    Pr[s_bbb:f_bbb] = temp[:,ij_b[0],ij_b[1]].reshape(-1).copy()

    return Pr


def ip_adc_matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

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
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij

    #Calculate sigma vector
    def sigma_(r):
        #r = np.ones(dim)
 
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        if adc.ncvs > 0:
            r = cvs_projector(adc, r)

        s = np.zeros((dim))

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_aaa = r_aaa.reshape(nvir_a,-1)
        r_bbb = r_bbb.reshape(nvir_b,-1)

        r_aaa_u = None
        r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
        r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
        r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

        r_bbb_u = None
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
               
               r_aaa = r_aaa.reshape(nvir_a,-1)
               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
               r_bbb = r_bbb.reshape(nvir_b,-1)
               
               r_aaa_u = None
               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()
               
               r_bbb_u = None
               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()
               
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
               t2_1_a = adc.t2[0][0][:]
               t2_1_ab = adc.t2[0][1][:]

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a

               a = 0
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

               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

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

               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

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

        if adc.ncvs > 0:
            s = cvs_projector(adc, s)
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
        return s

    return sigma_

def ip_cvs_adc_matvec(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs
    print('orbitals sizes')
    print(nocc_a, nocc_b, nvir_a, nvir_b ,ncvs, nval_a, nval_b)
    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)
    ij_ind_a = np.tril_indices(nocc_a, k=-1)

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = nvir_a * ncvs * (ncvs - 1) // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a  
    n_doubles_bba_ecc = nvir_b * ncvs * ncvs
    n_doubles_bba_ecv = nvir_b * ncvs * nval_a
    n_doubles_bba_evc = nvir_b * nval_b * ncvs
    n_doubles_aab_ecc = nvir_a * ncvs * ncvs
    n_doubles_aab_ecv = nvir_a * ncvs * nval_b
    n_doubles_aab_evc = nvir_a * nval_a * ncvs
    n_doubles_bbb_ecc = nvir_b * ncvs * (ncvs - 1) // 2
    n_doubles_bbb_ecv = nvir_b * ncvs * nval_b
    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_aaa_ecv + n_doubles_bba_ecc + n_doubles_bba_ecv + n_doubles_bba_evc + n_doubles_aab_ecc + n_doubles_aab_ecv + n_doubles_aab_evc + n_doubles_bbb_ecc + n_doubles_bbb_ecv

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

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bba_ecc = f_aaa_ecv
    f_bba_ecc = s_bba_ecc + n_doubles_bba_ecc
    s_bba_ecv = f_bba_ecc
    f_bba_ecv = s_bba_ecv + n_doubles_bba_ecv
    s_bba_evc = f_bba_ecv
    f_bba_evc = s_bba_evc + n_doubles_bba_evc
    s_aab_ecc = f_bba_evc
    f_aab_ecc = s_aab_ecc + n_doubles_aab_ecc
    s_aab_ecv = f_aab_ecc
    f_aab_ecv = s_aab_ecv + n_doubles_aab_ecv
    s_aab_evc = f_aab_ecv
    f_aab_evc = s_aab_evc + n_doubles_aab_evc
    s_bbb_ecc = f_aab_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_n_a_ecc = D_n_a[:, :ncvs, :ncvs].copy()
    D_aij_a_ecv = D_n_a[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_a_ecc = D_n_a_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_n_b_ecc = D_n_b[:, :ncvs, :ncvs].copy()
    D_aij_b_ecv = D_n_b[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_b_ecc = D_n_b_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)

    d_ij_ba = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bba = -d_a_b + d_ij_ba.reshape(-1)
    D_n_bba = D_n_bba.reshape((nvir_b,nocc_b,nocc_a))
    D_aij_bba_ecc = D_n_bba[:, :ncvs, :ncvs].reshape(-1)
    D_aij_bba_ecv = D_n_bba[:, :ncvs, ncvs:].reshape(-1)
    D_aij_bba_evc = D_n_bba[:, ncvs:, :ncvs].reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aab = -d_a_a + d_ij_ab.reshape(-1)
    #D_n_aba = D_n_aba.reshape(-1)
    D_n_aab = D_n_aab.reshape((nvir_a,nocc_a,nocc_b))
    D_aij_aab_ecc = D_n_aab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aab_ecv = D_n_aab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aab_evc = D_n_aab[:, ncvs:, :ncvs].reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij

    #Calculate sigma vector
    def sigma_(r):
        #r = np.ones(dim)
 
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa_ecc = r[s_aaa_ecc:f_aaa_ecc]
        r_aaa_ecv = r[s_aaa_ecv:f_aaa_ecv]
        r_bba_ecc = r[s_bba_ecc:f_bba_ecc]
        r_bba_ecv = r[s_bba_ecv:f_bba_ecv]
        r_bba_evc = r[s_bba_evc:f_bba_evc]
        r_aab_ecc = r[s_aab_ecc:f_aab_ecc]
        r_aab_ecv = r[s_aab_ecv:f_aab_ecv]
        r_aab_evc = r[s_aab_evc:f_aab_evc]
        r_bbb_ecc = r[s_bbb_ecc:f_bbb_ecc]
        r_bbb_ecv = r[s_bbb_ecv:f_bbb_ecv]

       
        r_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)
        r_bbb_ecc = r_bbb_ecc.reshape(nvir_b,-1)

        r_aaa_ecc_u = None
        r_aaa_ecc_u = np.zeros((nvir_a,ncvs,ncvs))
        r_aaa_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_aaa_ecc.copy()
        r_aaa_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_aaa_ecc.copy()

        r_bbb_ecc_u = None
        r_bbb_ecc_u = np.zeros((nvir_b,ncvs,ncvs))
        r_bbb_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_bbb_ecc.copy()
        r_bbb_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_bbb_ecc.copy()
        

        r_aaa_ecv = r_aaa_ecv.reshape(nvir_a,ncvs,nval_a)
        r_bba_ecc = r_bba_ecc.reshape(nvir_b,ncvs,ncvs)
        r_bba_ecv = r_bba_ecv.reshape(nvir_b,ncvs,nval_a)
        r_bba_evc = r_bba_evc.reshape(nvir_b,nval_b,ncvs)
        r_aab_ecc = r_aab_ecc.reshape(nvir_a,ncvs,ncvs)
        r_aab_ecv = r_aab_ecv.reshape(nvir_a,ncvs,nval_b)
        r_aab_evc = r_aab_evc.reshape(nvir_a,nval_a,ncvs)
        r_bbb_ecv = r_bbb_ecv.reshape(nvir_b,ncvs,nval_b)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

############ ADC(2) ij block ############################
         
        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)

############# ADC(2) i - kja block #########################
        # ECC Block
         
        s[s_a:f_a] += 0.5*lib.einsum('jaki,ajk->i', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_aaa_ecc_u, optimize = True)
        s[s_a:f_a] -= 0.5*lib.einsum('kaji,ajk->i', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_aaa_ecc_u, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_ovoo[:ncvs,:,ncvs:,:ncvs], r_aaa_ecv, optimize = True)
        s[s_a:f_a] -= lib.einsum('kaji,ajk->i', eris_ovoo[ncvs:,:,:ncvs,:ncvs], r_aaa_ecv, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo[:ncvs,:,:ncvs,:ncvs], r_bba_ecc, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo[:ncvs,:,ncvs:,:ncvs], r_bba_ecv, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo[ncvs:,:,:ncvs,:ncvs], r_bba_evc, optimize = True)

        s[s_b:f_b] += 0.5*lib.einsum('jaki,ajk->i', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_bbb_ecc_u, optimize = True)
        s[s_b:f_b] -= 0.5*lib.einsum('kaji,ajk->i', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_bbb_ecc_u, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_OVOO[:ncvs,:,ncvs:,:ncvs], r_bbb_ecv, optimize = True)
        s[s_b:f_b] -= lib.einsum('kaji,ajk->i', eris_OVOO[ncvs:,:,:ncvs,:ncvs], r_bbb_ecv, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO[:ncvs,:,:ncvs,:ncvs], r_aab_ecc, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO[:ncvs,:,ncvs:,:ncvs], r_aab_ecv, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO[ncvs:,:,:ncvs,:ncvs], r_aab_evc, optimize = True)
        
############## ADC(2) ajk - i block ############################
         
        temp_aaa_ecc = lib.einsum('JaKI,I->aJK', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_a, optimize = True)
        temp_aaa_ecc -= lib.einsum('KaJI,I->aJK', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_a, optimize = True)
        s[s_aaa_ecc:f_aaa_ecc] += temp_aaa_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] += lib.einsum('JakI,I->aJk', eris_ovoo[:ncvs,:,ncvs:,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] -= lib.einsum('kaJI,I->aJk', eris_ovoo[ncvs:,:,:ncvs,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_bba_ecc:f_bba_ecc] += lib.einsum('jaik,i->ajk', eris_OVoo[:ncvs,:,:ncvs,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_bba_ecv:f_bba_ecv] += lib.einsum('jaik,i->ajk', eris_OVoo[:ncvs,:,:ncvs,ncvs:], r_a, optimize = True).reshape(-1)
        s[s_bba_evc:f_bba_evc] += lib.einsum('jaik,i->ajk', eris_OVoo[ncvs:,:,:ncvs,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_aab_ecc:f_aab_ecc] += lib.einsum('jaki,i->ajk', eris_ovOO[:ncvs,:,:ncvs,:ncvs], r_b, optimize = True).reshape(-1)
        s[s_aab_ecv:f_aab_ecv] += lib.einsum('jaki,i->ajk', eris_ovOO[:ncvs,:,ncvs:,:ncvs], r_b, optimize = True).reshape(-1)
        s[s_aab_evc:f_aab_evc] += lib.einsum('jaki,i->ajk', eris_ovOO[ncvs:,:,:ncvs,:ncvs], r_b, optimize = True).reshape(-1)
        temp_bbb_ecc = lib.einsum('jaki,i->ajk', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_b, optimize = True)
        temp_bbb_ecc -= lib.einsum('kaji,i->ajk', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_b, optimize = True)
        s[s_bbb_ecc:f_bbb_ecc] += temp_bbb_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] += lib.einsum('jaki,i->ajk', eris_OVOO[:ncvs,:,ncvs:,:ncvs], r_b, optimize = True).reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] -= lib.einsum('kaji,i->ajk', eris_OVOO[ncvs:,:,:ncvs,:ncvs], r_b, optimize = True).reshape(-1)
        
############ ADC(2) ajk - bil block ############################

        r_aaa_ecc = r_aaa_ecc.reshape(-1)
        r_bbb_ecc = r_bbb_ecc.reshape(-1)
        
        s[s_aaa_ecc:f_aaa_ecc] += D_aij_a_ecc * r_aaa_ecc
        s[s_aaa_ecv:f_aaa_ecv] += D_aij_a_ecv * r_aaa_ecv.reshape(-1)
        s[s_bba_ecc:f_bba_ecc] += D_aij_bba_ecc * r_bba_ecc.reshape(-1)
        s[s_bba_ecv:f_bba_ecv] += D_aij_bba_ecv * r_bba_ecv.reshape(-1)
        s[s_bba_evc:f_bba_evc] += D_aij_bba_evc * r_bba_evc.reshape(-1)
        s[s_aab_ecc:f_aab_ecc] += D_aij_aab_ecc * r_aab_ecc.reshape(-1)
        s[s_aab_ecv:f_aab_ecv] += D_aij_aab_ecv * r_aab_ecv.reshape(-1)
        s[s_aab_evc:f_aab_evc] += D_aij_aab_evc * r_aab_evc.reshape(-1)
        s[s_bbb_ecc:f_bbb_ecc] += D_aij_b_ecc * r_bbb_ecc
        s[s_bbb_ecv:f_bbb_ecv] += D_aij_b_ecv * r_bbb_ecv.reshape(-1)
       
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
               
               
               temp_ecc = 0.5*lib.einsum('JLKI,aIL->aJK',eris_oooo[:ncvs,:ncvs,:ncvs,:ncvs],r_aaa_ecc_u ,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_oooo[:ncvs,:ncvs,:ncvs,:ncvs],r_aaa_ecc_u ,optimize = True)
               temp_ecc += lib.einsum('JlKI,aIl->aJK',eris_oooo[:ncvs,ncvs:,:ncvs,:ncvs],r_aaa_ecv ,optimize = True)
               temp_ecc -= lib.einsum('JIKl,aIl->aJK',eris_oooo[:ncvs,:ncvs,:ncvs,ncvs:],r_aaa_ecv ,optimize = True)
               temp_ecv = 0.5*lib.einsum('JLkI,aIL->aJk',eris_oooo[:ncvs,:ncvs,ncvs:,:ncvs],r_aaa_ecc_u ,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JIkL,aIL->aJk',eris_oooo[:ncvs,:ncvs,ncvs:,:ncvs],r_aaa_ecc_u ,optimize = True)
               temp_ecv += lib.einsum('JlkI,aIl->aJk',eris_oooo[:ncvs,ncvs:,ncvs:,:ncvs],r_aaa_ecv ,optimize = True)
               temp_ecv -= lib.einsum('JIkl,aIl->aJk',eris_oooo[:ncvs,:ncvs,ncvs:,ncvs:],r_aaa_ecv ,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

               temp_ecc = 0.5*lib.einsum('JLKI,aIL->aJK',eris_OOOO[:ncvs,:ncvs,:ncvs,:ncvs],r_bbb_ecc_u ,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_OOOO[:ncvs,:ncvs,:ncvs,:ncvs],r_bbb_ecc_u ,optimize = True)
               temp_ecc += lib.einsum('JlKI,aIl->aJK',eris_OOOO[:ncvs,ncvs:,:ncvs,:ncvs],r_bbb_ecv ,optimize = True)
               temp_ecc -= lib.einsum('JIKl,aIl->aJK',eris_OOOO[:ncvs,:ncvs,:ncvs,ncvs:],r_bbb_ecv ,optimize = True)
               temp_ecv = 0.5*lib.einsum('JLkI,aIL->aJk',eris_OOOO[:ncvs,:ncvs,ncvs:,:ncvs],r_bbb_ecc_u ,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JIkL,aIL->aJk',eris_OOOO[:ncvs,:ncvs,ncvs:,:ncvs],r_bbb_ecc_u ,optimize = True)
               temp_ecv += lib.einsum('JlkI,aIl->aJk',eris_OOOO[:ncvs,ncvs:,ncvs:,:ncvs],r_bbb_ecv ,optimize = True)
               temp_ecv -= lib.einsum('JIkl,aIl->aJk',eris_OOOO[:ncvs,:ncvs,ncvs:,ncvs:],r_bbb_ecv ,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

               #s[s_bab:f_bab] -= 0.5*lib.einsum('kijl,ali->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               #s[s_bab:f_bab] -= 0.5*lib.einsum('klji,ail->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('KIJL,aLI->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('KLJI,aIL->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('KiJL,aLi->aJK',eris_ooOO[:ncvs,ncvs:,:ncvs,:ncvs],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('KLJi,aiL->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('KIJl,alI->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('KlJI,aIl->aJK',eris_ooOO[:ncvs,ncvs:,:ncvs,:ncvs],r_bba_ecv,optimize = True).reshape(-1)

               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('kIJL,aLI->aJk',eris_ooOO[ncvs:,:ncvs,:ncvs,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('kLJI,aIL->aJk',eris_ooOO[ncvs:,:ncvs,:ncvs,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('kiJL,aLi->aJk',eris_ooOO[ncvs:,ncvs:,:ncvs,:ncvs],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('kLJi,aiL->aJk',eris_ooOO[ncvs:,:ncvs,:ncvs,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('kIJl,alI->aJk',eris_ooOO[ncvs:,:ncvs,:ncvs,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('klJI,aIl->aJk',eris_ooOO[ncvs:,ncvs:,:ncvs,:ncvs],r_bba_ecv,optimize = True).reshape(-1)

               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('KIjL,aLI->ajK',eris_ooOO[:ncvs,:ncvs,ncvs:,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('KLjI,aIL->ajK',eris_ooOO[:ncvs,:ncvs,ncvs:,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('KijL,aLi->ajK',eris_ooOO[:ncvs,ncvs:,ncvs:,:ncvs],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('KLji,aiL->ajK',eris_ooOO[:ncvs,:ncvs,ncvs:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('KIjl,alI->ajK',eris_ooOO[:ncvs,:ncvs,ncvs:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('KljI,aIl->ajK',eris_ooOO[:ncvs,ncvs:,ncvs:,:ncvs],r_bba_ecv,optimize = True).reshape(-1)
                
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jlki,ali->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jikl,ail->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
              
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JLKI,aLI->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JLKi,aLi->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,ncvs:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JiKL,aiL->aJK',eris_ooOO[:ncvs,ncvs:,:ncvs,:ncvs],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JlKI,alI->aJK',eris_ooOO[:ncvs,ncvs:,:ncvs,:ncvs],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JIKl,aIl->aJK',eris_ooOO[:ncvs,:ncvs,:ncvs,ncvs:],r_aab_ecv,optimize = True).reshape(-1)

               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JLkI,aLI->aJk',eris_ooOO[:ncvs,:ncvs,ncvs:,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JIkL,aIL->aJk',eris_ooOO[:ncvs,:ncvs,ncvs:,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JLki,aLi->aJk',eris_ooOO[:ncvs,:ncvs,ncvs:,ncvs:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JikL,aiL->aJk',eris_ooOO[:ncvs,ncvs:,ncvs:,:ncvs],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JlkI,alI->aJk',eris_ooOO[:ncvs,ncvs:,ncvs:,:ncvs],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JIkl,aIl->aJk',eris_ooOO[:ncvs,:ncvs,ncvs:,ncvs:],r_aab_ecv,optimize = True).reshape(-1)

               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jLKI,aLI->ajK',eris_ooOO[ncvs:,:ncvs,:ncvs,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jIKL,aIL->ajK',eris_ooOO[ncvs:,:ncvs,:ncvs,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jLKi,aLi->ajK',eris_ooOO[ncvs:,:ncvs,:ncvs,ncvs:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jiKL,aiL->ajK',eris_ooOO[ncvs:,ncvs:,:ncvs:,:ncvs],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jlKI,alI->ajK',eris_ooOO[ncvs:,ncvs:,:ncvs,:ncvs],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jIKl,aIl->ajK',eris_ooOO[ncvs:,:ncvs,:ncvs,ncvs:],r_aab_ecv,optimize = True).reshape(-1)
                
               #temp = 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r_aaa_u,optimize = True)
               #temp -= 0.5*lib.einsum('kabl,bjl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               #temp += 0.5* lib.einsum('kabl,blj->ajk',eris_ovVO,r_bab,optimize = True)
               #s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
               
               temp_ecc = 0.5*lib.einsum('KLba,bJL->aJK',eris_oovv[:ncvs,:ncvs,:,:],r_aaa_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('KabL,bJL->aJK',eris_ovvo[:ncvs,:,:,:ncvs],r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5* lib.einsum('KabL,bLJ->aJK',eris_ovVO[:ncvs,:,:,:ncvs],r_bba_ecc,optimize = True)
               temp_ecc += 0.5*lib.einsum('Klba,bJl->aJK',eris_oovv[:ncvs,ncvs:,:,:],r_aaa_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('Kabl,bJl->aJK',eris_ovvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True)
               temp_ecc += 0.5* lib.einsum('Kabl,blJ->aJK',eris_ovVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True)
               temp_ecv = 0.5*lib.einsum('kLba,bJL->aJk',eris_oovv[ncvs:,:ncvs,:,:],r_aaa_ecc_u,optimize = True)
               temp_ecv -= 0.5*lib.einsum('kabL,bJL->aJk',eris_ovvo[ncvs:,:,:,:ncvs],r_aaa_ecc_u,optimize = True)
               temp_ecv += 0.5* lib.einsum('kabL,bLJ->aJk',eris_ovVO[ncvs:,:,:,:ncvs],r_bba_ecc,optimize = True)
               temp_ecv += 0.5*lib.einsum('klba,bJl->aJk',eris_oovv[ncvs:,ncvs:,:,:],r_aaa_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('kabl,bJl->aJk',eris_ovvo[ncvs:,:,:,ncvs:],r_aaa_ecv,optimize = True)
               temp_ecv += 0.5* lib.einsum('kabl,blJ->aJk',eris_ovVO[ncvs:,:,:,ncvs:],r_bba_evc,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
              
                
               #s[s_bab:f_bab] += 0.5*lib.einsum('klba,bjl->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)

               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('KLba,bJL->aJK',eris_ooVV[:ncvs,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('Klba,bJl->aJK',eris_ooVV[:ncvs,ncvs:,:,:],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('kLba,bJL->aJk',eris_ooVV[ncvs:,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('klba,bJl->aJk',eris_ooVV[ncvs:,ncvs:,:,:],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('KLba,bjL->ajK',eris_ooVV[:ncvs,:ncvs,:,:],r_bba_evc,optimize = True).reshape(-1)
               #s[s_bab_evc:f_bab_evc] += 0.5*lib.einsum('Klba,bjl->ajK',eris_ooVV[:ncore,ncore:,:,:],r_bba_ecv,optimize = True).reshape(-1)
               
               #temp_1 = 0.5*lib.einsum('klba,bjl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               #temp_1 -= 0.5*lib.einsum('kabl,bjl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               #temp_1 += 0.5*lib.einsum('kabl,blj->ajk',eris_OVvo,r_aba,optimize = True)
               #s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
              
               temp_1_ecc = 0.5*lib.einsum('KLba,bJL->aJK',eris_OOVV[:ncvs,:ncvs,:,:],r_bbb_ecc_u,optimize = True)
               temp_1_ecc -= 0.5*lib.einsum('KabL,bJL->aJK',eris_OVVO[:ncvs,:,:,:ncvs],r_bbb_ecc_u,optimize = True)
               temp_1_ecc += 0.5*lib.einsum('KabL,bLJ->aJK',eris_OVvo[:ncvs,:,:,:ncvs],r_aab_ecc,optimize = True)
               temp_1_ecc += 0.5*lib.einsum('Klba,bJl->aJK',eris_OOVV[:ncvs,ncvs:,:,:],r_bbb_ecv,optimize = True)
               temp_1_ecc -= 0.5*lib.einsum('Kabl,bJl->aJK',eris_OVVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True)
               temp_1_ecc += 0.5*lib.einsum('Kabl,blJ->aJK',eris_OVvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True)
               temp_1_ecv = 0.5*lib.einsum('kLba,bJL->aJk',eris_OOVV[ncvs:,:ncvs,:,:],r_bbb_ecc_u,optimize = True)
               temp_1_ecv -= 0.5*lib.einsum('kabL,bJL->aJk',eris_OVVO[ncvs:,:,:,:ncvs],r_bbb_ecc_u,optimize = True)
               temp_1_ecv += 0.5*lib.einsum('kabL,bLJ->aJk',eris_OVvo[ncvs:,:,:,:ncvs],r_aab_ecc,optimize = True)
               temp_1_ecv += 0.5*lib.einsum('klba,bJl->aJk',eris_OOVV[ncvs:,ncvs:,:,:],r_bbb_ecv,optimize = True)
               temp_1_ecv -= 0.5*lib.einsum('kabl,bJl->aJk',eris_OVVO[ncvs:,:,:,ncvs:],r_bbb_ecv,optimize = True)
               temp_1_ecv += 0.5*lib.einsum('kabl,blJ->aJk',eris_OVvo[ncvs:,:,:,ncvs:],r_aab_evc,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_1_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_1_ecv.reshape(-1)
               
                
               #s[s_aba:f_aba] += 0.5*lib.einsum('klba,bjl->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)

               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('KLba,bJL->aJK',eris_OOvv[:ncvs,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Klba,bJl->aJK',eris_OOvv[:ncvs,ncvs:,:,:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('kLba,bJL->aJk',eris_OOvv[ncvs:,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('klba,bJl->aJk',eris_OOvv[ncvs:,ncvs:,:,:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('KLba,bjL->ajK',eris_OOvv[:ncvs,:ncvs,:,:],r_aab_evc,optimize = True).reshape(-1)
               #s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('Klba,bjl->ajK',eris_OOvv[:ncvs,ncvs:,:,:],r_aab_ecv,optimize = True).reshape(-1)
               
               #temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_oovv,r_aaa_u,optimize = True)
               #temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r_aaa_u,optimize = True)
               #temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovVO,r_bab,optimize = True)
               #s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               temp_ecc = -0.5*lib.einsum('JLba,bKL->aJK',eris_oovv[:ncvs,:ncvs,:,:],r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('JabL,bKL->aJK',eris_ovvo[:ncvs,:,:,:ncvs],r_aaa_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JabL,bLK->aJK',eris_ovVO[:ncvs,:,:,:ncvs],r_bba_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jlba,bKl->aJK',eris_oovv[:ncvs,ncvs:,:,:],r_aaa_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('Jabl,bKl->aJK',eris_ovvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('Jabl,blK->aJK',eris_ovVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True)
               temp_ecv = 0.5*lib.einsum('JLba,bLk->aJk',eris_oovv[:ncvs,:ncvs,:,:],r_aaa_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabL,bLk->aJk',eris_ovvo[:ncvs,:,:,:ncvs],r_aaa_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabL,bLk->aJk',eris_ovVO[:ncvs,:,:,:ncvs],r_bba_ecv,optimize = True)
               #temp_ecv += -0.5*lib.einsum('Jlba,bkl->aJk',eris_oovv[:ncvs,ncvs:,:,:],r_aaa_ecv,optimize = True)
               #temp_ecv += 0.5*lib.einsum('Jabl,bkl->aJk',eris_ovvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True)
               #temp_ecv -= 0.5*lib.einsum('Jabl,blk->aJk',eris_ovVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
               
               #s[s_bab:f_bab] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)
               #s[s_bab:f_bab] +=  0.5*lib.einsum('jlba,blk->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               #s[s_bab:f_bab] -=  0.5*lib.einsum('jabl,blk->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)

               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('JabL,bKL->aJK',eris_OVvo[:ncvs,:,:,:ncvs],r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('JLba,bLK->aJK',eris_OOVV[:ncvs,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -=  0.5*lib.einsum('JabL,bLK->aJK',eris_OVVO[:ncvs,:,:,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('Jabl,bKl->aJK',eris_OVvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('Jlba,blK->aJK',eris_OOVV[:ncvs,ncvs:,:,:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -=  0.5*lib.einsum('Jabl,blK->aJK',eris_OVVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] +=  -0.5*lib.einsum('JabL,bLk->aJk',eris_OVvo[:ncvs,:,:,:ncvs],r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] +=  0.5*lib.einsum('JLba,bLk->aJk',eris_OOVV[:ncvs,:ncvs,:,:],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -=  0.5*lib.einsum('JabL,bLk->aJk',eris_OVVO[:ncvs,:,:,:ncvs],r_bba_ecv,optimize = True).reshape(-1)
               #s[s_bba_ecv:f_bba_ecv] +=  0.5*lib.einsum('Jabl,bkl->aJk',eris_OVvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True).reshape(-1)
               #s[s_bba_ecv:f_bba_ecv] +=  0.5*lib.einsum('Jlba,blk->aJk',eris_OOVV[:ncvs,ncvs:,:,:],r_bba_evc,optimize = True).reshape(-1)
               #s[s_bba_ecv:f_bba_ecv] -=  0.5*lib.einsum('Jabl,blk->aJk',eris_OVVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jabL,bKL->ajK',eris_OVvo[ncvs:,:,:,:ncvs],r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jLba,bLK->ajK',eris_OOVV[ncvs:,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -=  0.5*lib.einsum('jabL,bLK->ajK',eris_OVVO[ncvs:,:,:,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jabl,bKl->ajK',eris_OVvo[ncvs:,:,:,ncvs:],r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jlba,blK->ajK',eris_OOVV[ncvs:,ncvs:,:,:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -=  0.5*lib.einsum('jabl,blK->ajK',eris_OVVO[ncvs:,:,:,ncvs:],r_bba_evc,optimize = True).reshape(-1)

               #temp = -0.5*lib.einsum('jlba,bkl->ajk',eris_OOVV,r_bbb_u,optimize = True)
               #temp += 0.5*lib.einsum('jabl,bkl->ajk',eris_OVVO,r_bbb_u,optimize = True)
               #temp -= 0.5*lib.einsum('jabl,blk->ajk',eris_OVvo,r_aba,optimize = True)
               #s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               temp_ecc = -0.5*lib.einsum('JLba,bKL->aJK',eris_OOVV[:ncvs,:ncvs,:,:],r_bbb_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('JabL,bKL->aJK',eris_OVVO[:ncvs,:,:,:ncvs],r_bbb_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JabL,bLK->aJK',eris_OVvo[:ncvs,:,:,:ncvs],r_aab_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jlba,bKl->aJK',eris_OOVV[:ncvs,ncvs:,:,:],r_bbb_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('Jabl,bKl->aJK',eris_OVVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('Jabl,blK->aJK',eris_OVvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True)
               temp_ecv = 0.5*lib.einsum('JLba,bLk->aJk',eris_OOVV[:ncvs,:ncvs,:,:],r_bbb_ecv,optimize = True)
               temp_ecv += -0.5*lib.einsum('JabL,bLk->aJk',eris_OVVO[:ncvs,:,:,:ncvs],r_bbb_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabL,bLk->aJk',eris_OVvo[:ncvs,:,:,:ncvs],r_aab_ecv,optimize = True)
               #temp_ecv += -0.5*lib.einsum('Jlba,bkl->aJk',eris_OOVV[:ncvs,ncvs:,:,:],r_bbb_ecv,optimize = True)
               #temp_ecv += 0.5*lib.einsum('Jabl,bkl->aJk',eris_OVVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True)
               #temp_ecv -= 0.5*lib.einsum('Jabl,blk->aJk',eris_OVvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)
               
               #s[s_aba:f_aba] += 0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               #s[s_aba:f_aba] += 0.5*lib.einsum('jabl,bkl->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)

               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('JLba,bLK->aJK',eris_oovv[:ncvs,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JabL,bLK->aJK',eris_ovvo[:ncvs,:,:,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('JabL,bKL->aJK',eris_ovVO[:ncvs,:,:,:ncvs],r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Jlba,blK->aJK',eris_oovv[:ncvs,ncvs:,:,:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('Jabl,blK->aJK',eris_ovvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Jabl,bKl->aJK',eris_ovVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('JLba,bLk->aJk',eris_oovv[:ncvs,:ncvs,:,:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JabL,bLk->aJk',eris_ovvo[:ncvs,:,:,:ncvs],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += -0.5*lib.einsum('JabL,bLk->aJk',eris_ovVO[:ncvs,:,:,:ncvs],r_bbb_ecv,optimize = True).reshape(-1)
               #s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('Jlba,blk->aJk',eris_oovv[:ncvs,ncvs:,:,:],r_aab_evc,optimize = True).reshape(-1)
               #s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('Jabl,blk->aJk',eris_ovvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True).reshape(-1)
               #s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('Jabl,bkl->aJk',eris_ovVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jLba,bLK->ajK',eris_oovv[ncvs:,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jabL,bLK->ajK',eris_ovvo[ncvs:,:,:,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jabL,bKL->ajK',eris_ovVO[ncvs:,:,:,:ncvs],r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jlba,blK->ajK',eris_oovv[ncvs:,ncvs:,:,:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jabl,blK->ajK',eris_ovvo[ncvs:,:,:,ncvs:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jabl,bKl->ajK',eris_ovVO[ncvs:,:,:,ncvs:],r_bbb_ecv,optimize = True).reshape(-1)
                
               #temp = -0.5*lib.einsum('kiba,bij->ajk',eris_oovv,r_aaa_u,optimize = True)
               #temp += 0.5*lib.einsum('kabi,bij->ajk',eris_ovvo,r_aaa_u,optimize = True)
               #temp += 0.5*lib.einsum('kabi,bij->ajk',eris_ovVO,r_bab,optimize = True)
               #s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               temp_ecc = -0.5*lib.einsum('KIba,bIJ->aJK',eris_oovv[:ncvs,:ncvs,:,:],r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KabI,bIJ->aJK',eris_ovvo[:ncvs,:,:,:ncvs],r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KabI,bIJ->aJK',eris_ovVO[:ncvs,:,:,:ncvs],r_bba_ecc,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kiba,bJi->aJK',eris_oovv[:ncvs,ncvs:,:,:],r_aaa_ecv,optimize = True)
               temp_ecc += -0.5*lib.einsum('Kabi,bJi->aJK',eris_ovvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kabi,biJ->aJK',eris_ovVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True)
               temp_ecv = -0.5*lib.einsum('kIba,bIJ->aJk',eris_oovv[ncvs:,:ncvs,:,:],r_aaa_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kabI,bIJ->aJk',eris_ovvo[ncvs:,:,:,:ncvs],r_aaa_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kabI,bIJ->aJk',eris_ovVO[ncvs:,:,:,:ncvs],r_bba_ecc,optimize = True)
               temp_ecv += 0.5*lib.einsum('kiba,bJi->aJk',eris_oovv[ncvs:,ncvs:,:,:],r_aaa_ecv,optimize = True)
               temp_ecv += -0.5*lib.einsum('kabi,bJi->aJk',eris_ovvo[ncvs:,:,:,ncvs:],r_aaa_ecv,optimize = True)
               temp_ecv += 0.5*lib.einsum('kabi,biJ->aJk',eris_ovVO[ncvs:,:,:,ncvs:],r_bba_evc,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
               
               #s[s_bab:f_bab] += 0.5*lib.einsum('kiba,bji->ajk',eris_ooVV,r_bab,optimize = True).reshape(-1)

               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('KIba,bJI->aJK',eris_ooVV[:ncvs,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('Kiba,bJi->aJK',eris_ooVV[:ncvs,ncvs:,:,:],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('kIba,bJI->aJk',eris_ooVV[ncvs:,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('kiba,bJi->aJk',eris_ooVV[ncvs:,ncvs:,:,:],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('KIba,bjI->ajK',eris_ooVV[:ncvs,:ncvs,:,:],r_bba_evc,optimize = True).reshape(-1)
               #s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('Kiba,bji->ajK',eris_ooVV[:ncvs,ncvs:,:,:],r_bba_ecv,optimize = True).reshape(-1)
               
               #temp = -0.5*lib.einsum('kiba,bij->ajk',eris_OOVV,r_bbb_u,optimize = True)
               #temp += 0.5*lib.einsum('kabi,bij->ajk',eris_OVVO,r_bbb_u,optimize = True)
               #temp += 0.5*lib.einsum('kabi,bij->ajk',eris_OVvo,r_aba,optimize = True)
               #s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               temp_ecc = -0.5*lib.einsum('KIba,bIJ->aJK',eris_OOVV[:ncvs,:ncvs,:,:],r_bbb_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KabI,bIJ->aJK',eris_OVVO[:ncvs,:,:,:ncvs],r_bbb_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KabI,bIJ->aJK',eris_OVvo[:ncvs,:,:,:ncvs],r_aab_ecc,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kiba,bJi->aJK',eris_OOVV[:ncvs,ncvs:,:,:],r_bbb_ecv,optimize = True)
               temp_ecc += -0.5*lib.einsum('Kabi,bJi->aJK',eris_OVVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kabi,biJ->aJK',eris_OVvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True)
               temp_ecv = -0.5*lib.einsum('kIba,bIJ->aJk',eris_OOVV[ncvs:,:ncvs,:,:],r_bbb_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kabI,bIJ->aJk',eris_OVVO[ncvs:,:,:,:ncvs],r_bbb_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kabI,bIJ->aJk',eris_OVvo[ncvs:,:,:,:ncvs],r_aab_ecc,optimize = True)
               temp_ecv += 0.5*lib.einsum('kiba,bJi->aJk',eris_OOVV[ncvs:,ncvs:,:,:],r_bbb_ecv,optimize = True)
               temp_ecv += -0.5*lib.einsum('kabi,bJi->aJk',eris_OVVO[ncvs:,:,:,ncvs:],r_bbb_ecv,optimize = True)
               temp_ecv += 0.5*lib.einsum('kabi,biJ->aJk',eris_OVvo[ncvs:,:,:,ncvs:],r_aab_evc,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)
               
               #s[s_aba:f_aba] += 0.5*lib.einsum('kiba,bji->ajk',eris_OOvv,r_aba,optimize = True).reshape(-1)

               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('KIba,bJI->aJK',eris_OOvv[:ncvs,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Kiba,bJi->aJK',eris_OOvv[:ncvs,ncvs:,:,:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('kIba,bJI->aJk',eris_OOvv[ncvs:,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('kiba,bJi->aJk',eris_OOvv[ncvs:,ncvs:,:,:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('KIba,bjI->ajK',eris_OOvv[:ncvs,:ncvs,:,:],r_aab_evc,optimize = True).reshape(-1)
               #s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('Kiba,bji->ajK',eris_OOvv[:ncvs,ncvs:,:,:],r_aab_ecv,optimize = True).reshape(-1)

                
               #temp = 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r_aaa_u,optimize = True)
               #temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r_aaa_u,optimize = True)
               #temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovVO,r_bab,optimize = True)
               #s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               temp_ecc = 0.5*lib.einsum('JIba,bIK->aJK',eris_oovv[:ncvs,:ncvs,:,:],r_aaa_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JabI,bIK->aJK',eris_ovvo[:ncvs,:,:,:ncvs],r_aaa_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JabI,bIK->aJK',eris_ovVO[:ncvs,:,:,:ncvs],r_bba_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jiba,bKi->aJK',eris_oovv[:ncvs,ncvs:,:,:],r_aaa_ecv,optimize = True)
               temp_ecc -= -0.5*lib.einsum('Jabi,bKi->aJK',eris_ovvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('Jabi,biK->aJK',eris_ovVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True)
               temp_ecv = 0.5*lib.einsum('JIba,bIk->aJk',eris_oovv[:ncvs,:ncvs,:,:],r_aaa_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabI,bIk->aJk',eris_ovvo[:ncvs,:,:,:ncvs],r_aaa_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabI,bIk->aJk',eris_ovVO[:ncvs,:,:,:ncvs],r_bba_ecv,optimize = True)
               #temp_ecv += -0.5*lib.einsum('Jiba,bki->aJk',eris_oovv[:ncvs,ncvs:,:,:],r_aaa_ecv,optimize = True)
               #temp_ecv -= -0.5*lib.einsum('Jabi,bki->aJk',eris_ovvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True)
               #temp_ecv -= 0.5*lib.einsum('Jabi,bik->aJk',eris_ovVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
               
               #s[s_bab:f_bab] += 0.5*lib.einsum('jiba,bik->ajk',eris_OOVV,r_bab,optimize = True).reshape(-1)
               #s[s_bab:f_bab] -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVVO,r_bab,optimize = True).reshape(-1)
               #s[s_bab:f_bab] -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVvo,r_aaa_u,optimize = True).reshape(-1)

               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('JIba,bIK->aJK',eris_OOVV[:ncvs,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1) 
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('JabI,bIK->aJK',eris_OVVO[:ncvs,:,:,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('JabI,bIK->aJK',eris_OVvo[:ncvs,:,:,:ncvs],r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('Jiba,biK->aJK',eris_OOVV[:ncvs,ncvs:,:,:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('Jabi,biK->aJK',eris_OVVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= -0.5*lib.einsum('Jabi,bKi->aJK',eris_OVvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('JIba,bIk->aJk',eris_OOVV[:ncvs,:ncvs,:,:],r_bba_ecv,optimize = True).reshape(-1) 
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('JabI,bIk->aJk',eris_OVVO[:ncvs,:,:,:ncvs],r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('JabI,bIk->aJk',eris_OVvo[:ncvs,:,:,:ncvs],r_aaa_ecv,optimize = True).reshape(-1)
               #s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('Jiba,bik->aJk',eris_OOVV[:ncvs,ncvs:,:,:],r_bba_evc,optimize = True).reshape(-1)
               #s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('Jabi,bik->aJk',eris_OVVO[:ncvs,:,:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               #s[s_bba_ecv:f_bba_ecv] -= -0.5*lib.einsum('Jabi,bki->aJk',eris_OVvo[:ncvs,:,:,ncvs:],r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('jIba,bIK->ajK',eris_OOVV[ncvs:,:ncvs,:,:],r_bba_ecc,optimize = True).reshape(-1) 
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('jabI,bIK->ajK',eris_OVVO[ncvs:,:,:,:ncvs],r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('jabI,bIK->ajK',eris_OVvo[ncvs:,:,:,:ncvs],r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('jiba,biK->ajK',eris_OOVV[ncvs:,ncvs:,:,:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('jabi,biK->ajK',eris_OVVO[ncvs:,:,:,ncvs:],r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= -0.5*lib.einsum('jabi,bKi->ajK',eris_OVvo[ncvs:,:,:,ncvs:],r_aaa_ecv,optimize = True).reshape(-1)
               
               #s[s_aba:f_aba] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r_aba,optimize = True).reshape(-1)
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r_aba,optimize = True).reshape(-1)
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovVO,r_bbb_u,optimize = True).reshape(-1)

               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('JIba,bIK->aJK',eris_oovv[:ncvs,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JabI,bIK->aJK',eris_ovvo[:ncvs,:,:,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JabI,bIK->aJK',eris_ovVO[:ncvs,:,:,:ncvs],r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Jiba,biK->aJK',eris_oovv[:ncvs,ncvs:,:,:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('Jabi,biK->aJK',eris_ovvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= -0.5*lib.einsum('Jabi,bKi->aJK',eris_ovVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('JIba,bIk->aJk',eris_oovv[:ncvs,:ncvs,:,:],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JabI,bIk->aJk',eris_ovvo[:ncvs,:,:,:ncvs],r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JabI,bIk->aJk',eris_ovVO[:ncvs,:,:,:ncvs],r_bbb_ecv,optimize = True).reshape(-1)
               #s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('Jiba,bik->aJk',eris_oovv[:ncvs,ncvs:,:,:],r_aab_evc,optimize = True).reshape(-1)
               #s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('Jabi,bik->aJk',eris_ovvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True).reshape(-1)
               #s[s_aab_ecv:f_aab_ecv] -= -0.5*lib.einsum('Jabi,bki->aJk',eris_ovVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jIba,bIK->ajK',eris_oovv[ncvs:,:ncvs,:,:],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jabI,bIK->ajK',eris_ovvo[ncvs:,:,:,:ncvs],r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jabI,bIK->ajK',eris_ovVO[ncvs:,:,:,:ncvs],r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jiba,biK->ajK',eris_oovv[ncvs:,ncvs:,:,:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jabi,biK->ajK',eris_ovvo[ncvs:,:,:,ncvs:],r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= -0.5*lib.einsum('jabi,bKi->ajK',eris_ovVO[ncvs:,:,:,ncvs:],r_bbb_ecv,optimize = True).reshape(-1)
               
               #temp = 0.5*lib.einsum('jiba,bik->ajk',eris_OOVV,r_bbb_u,optimize = True)
               #temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVVO,r_bbb_u,optimize = True)
               #temp -= 0.5*lib.einsum('jabi,bik->ajk',eris_OVvo,r_aba,optimize = True)
               #s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               temp_ecc = 0.5*lib.einsum('JIba,bIK->aJK',eris_OOVV[:ncvs,:ncvs,:,:],r_bbb_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JabI,bIK->aJK',eris_OVVO[:ncvs,:,:,:ncvs],r_bbb_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JabI,bIK->aJK',eris_OVvo[:ncvs,:,:,:ncvs],r_aab_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jiba,bKi->aJK',eris_OOVV[:ncvs,ncvs:,:,:],r_bbb_ecv,optimize = True)
               temp_ecc -= -0.5*lib.einsum('Jabi,bKi->aJK',eris_OVVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('Jabi,biK->aJK',eris_OVvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True)
               temp_ecv = 0.5*lib.einsum('JIba,bIk->aJk',eris_OOVV[:ncvs,:ncvs,:,:],r_bbb_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabI,bIk->aJk',eris_OVVO[:ncvs,:,:,:ncvs],r_bbb_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JabI,bIk->aJk',eris_OVvo[:ncvs,:,:,:ncvs],r_aab_ecv,optimize = True)
               #temp_ecv += -0.5*lib.einsum('Jiba,bki->aJk',eris_OOVV[:ncvs,ncvs:,:,:],r_bbb_ecv,optimize = True)
               #temp_ecv -= -0.5*lib.einsum('Jabi,bki->aJk',eris_OVVO[:ncvs,:,:,ncvs:],r_bbb_ecv,optimize = True)
               #temp_ecv -= 0.5*lib.einsum('Jabi,bik->aJk',eris_OVvo[:ncvs,:,:,ncvs:],r_aab_evc,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)
               
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
               #temp_singles = np.zeros((nocc_a))
               #temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a))
               #r_aaa = r_aaa.reshape(nvir_a,-1)
               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]

               temp_singles = np.zeros((ncvs))
               temp_doubles_ecc = np.zeros((nvir_a, ncvs, ncvs))
               temp_doubles_ecv = np.zeros((nvir_a, ncvs, nval_a))
               r_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)
               
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

               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

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

               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

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
        
        print('s[s_a:f_a]',             np.linalg.norm(s[s_a:f_a]))
        print(np.column_stack(np.unique(np.around(s[s_a:f_a], decimals=8), return_counts=True))) 
        print('s[s_b:f_b]' ,            np.linalg.norm(s[s_b:f_b]))
        print(np.column_stack(np.unique(np.around(s[s_b:f_b], decimals=8), return_counts=True))) 
        print('s[s_aaa_ecc:f_aaa_ecv]', np.linalg.norm(s[s_aaa_ecc:f_aaa_ecv]))      
        print(np.column_stack(np.unique(np.around(s[s_aaa_ecc:f_aaa_ecv], decimals=8), return_counts=True))) 
        print('s[s_aaa_ecv:f_aaa_ecv]', np.linalg.norm(s[s_aaa_ecv:f_aaa_ecv]))      
        print(np.column_stack(np.unique(np.around(s[s_aaa_ecv:f_aaa_ecv], decimals=8), return_counts=True))) 
        print('s[s_bba_ecc:f_bba_evc]', np.linalg.norm(s[s_bba_ecc:f_bba_evc]))
        print(np.column_stack(np.unique(np.around(s[s_bba_ecc:f_bba_evc], decimals=8), return_counts=True))) 
        print('s[s_aab_ecc:f_aab_evc]', np.linalg.norm(s[s_aab_ecc:f_aab_evc]))
        print(np.column_stack(np.unique(np.around(s[s_aab_ecc:f_aab_evc], decimals=8), return_counts=True))) 
        print('s[s_bbb_ecc:f_bbb_ecv]', np.linalg.norm(s[s_bbb_ecc:f_bbb_ecv]))
        print(np.column_stack(np.unique(np.around(s[s_bbb_ecc:f_bbb_ecv], decimals=8), return_counts=True))) 
        
        temp_aaa = s[s_aaa_ecc:f_aaa_ecv] 
        temp_bbb = s[s_bbb_ecc:f_bbb_ecv] 
        print("# of non zeros in aaa: ", temp_aaa[temp_aaa != 0].size)
        print("# of non zeros in bbb: ", temp_bbb[temp_bbb != 0].size)
        return s

    return sigma_

def ip_cvs_adc_matvec_off(adc, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = ncvs * (ncvs - 1) * nvir_a // 2
    n_doubles_bab_ecc = nvir_b * ncvs * ncvs
    n_doubles_aba_ecc = nvir_a * ncvs * ncvs
    n_doubles_bbb_ecc = ncvs * (ncvs - 1) * nvir_b // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a
    n_doubles_bab_ecv = nvir_b * ncvs * nval_b
    n_doubles_bab_evc = nvir_b * nval_a * ncvs
    n_doubles_aba_ecv = nvir_a * ncvs * nval_a
    n_doubles_aba_evc = nvir_a * nval_b * ncvs
    n_doubles_bbb_ecv = ncvs * nval_b * nvir_b

    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_bab_ecc + n_doubles_aba_ecc + n_doubles_bbb_ecc + n_doubles_aaa_ecv + n_doubles_bab_ecv + n_doubles_aba_ecv + n_doubles_bbb_ecv + n_doubles_bab_evc + n_doubles_aba_evc

    e_occ_a = adc.mo_energy_a[:nocc_a]
    e_occ_b = adc.mo_energy_b[:nocc_b]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    if eris is None:
        eris = adc.transform_integrals()

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_n_a_ecc = D_n_a[:, :ncvs, :ncvs].copy()
    D_aij_a_ecv = D_n_a[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_a_ecc = D_n_a_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
    #D_aij_a_evc = D_n_a[:, ncvs:, :ncvs].copy().reshape(-1)


    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_n_b_ecc = D_n_b[:, :ncvs, :ncvs].copy()
    D_aij_b_ecv = D_n_b[:, :ncvs, ncvs:].copy().reshape(-1)
    D_aij_b_ecc = D_n_b_ecc.copy()[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
    #D_aij_b_evc = D_n_b[:, ncvs:, :ncvs].copy().reshape(-1)


    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_n_bab = D_n_bab.reshape((nvir_b,nocc_a,nocc_b))
    D_aij_bab_ecc = D_n_bab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_bab_ecv = D_n_bab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_bab_evc = D_n_bab[:, ncvs:, :ncvs].reshape(-1)

    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_n_aba = D_n_aba.reshape((nvir_a,nocc_b,nocc_a))
    D_aij_aba_ecc = D_n_aba[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aba_ecv = D_n_aba[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aba_evc = D_n_aba[:, ncvs:, :ncvs].reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    #s_aaa_evc = f_aaa_ecv
    #f_aaa_evc = s_aaa_evc + n_doubles_aaa_ecv
    s_bab_ecc = f_aaa_ecv
    f_bab_ecc = s_bab_ecc + n_doubles_bab_ecc
    s_bab_ecv = f_bab_ecc
    f_bab_ecv = s_bab_ecv + n_doubles_bab_ecv
    s_bab_evc = f_bab_ecv
    f_bab_evc = s_bab_evc + n_doubles_bab_evc
    s_aba_ecc = f_bab_evc
    f_aba_ecc = s_aba_ecc + n_doubles_aba_ecc
    s_aba_ecv = f_aba_ecc
    f_aba_ecv = s_aba_ecv + n_doubles_aba_ecv
    s_aba_evc = f_aba_ecv
    f_aba_evc = s_aba_evc + n_doubles_aba_evc
    s_bbb_ecc = f_aba_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv
    #s_bbb_evc = f_bbb_ecv
    #f_bbb_evc = s_bbb_evc + n_doubles_bbb_ecv

    if M_ij is None:
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij

    #Calculate sigma vector
    def sigma_(r):
        #r = np.ones(dim) 
        #r = np.linspace(1,dim,dim)

        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))
        

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa_ecc = r[s_aaa_ecc:f_aaa_ecc]
        r_aaa_ecv = r[s_aaa_ecv:f_aaa_ecv]
        r_bab_ecc = r[s_bab_ecc:f_bab_ecc]
        r_bab_ecv = r[s_bab_ecv:f_bab_ecv]
        r_bab_evc = r[s_bab_evc:f_bab_evc]
        r_aba_ecc = r[s_aba_ecc:f_aba_ecc]
        r_aba_ecv = r[s_aba_ecv:f_aba_ecv]
        r_aba_evc = r[s_aba_evc:f_aba_evc]
        r_bbb_ecc = r[s_bbb_ecc:f_bbb_ecc]
        r_bbb_ecv = r[s_bbb_ecv:f_bbb_ecv]
        
        print('s_aaa_ecc:f_aaa_ecc',  s_aaa_ecc,f_aaa_ecc)
        print('s_aaa_ecv:f_aaa_ecv',  s_aaa_ecv,f_aaa_ecv)
        print('s_bab_ecc:f_bab_ecc',  s_bab_ecc,f_bab_ecc)
        print('s_bab_ecv:f_bab_ecv',  s_bab_ecv,f_bab_ecv)
        print('s_bab_evc:f_bab_evc',  s_bab_evc,f_bab_evc)
        print('s_aba_ecc:f_aba_ecc',  s_aba_ecc,f_aba_ecc)
        print('s_aba_ecv:f_aba_ecv',  s_aba_ecv,f_aba_ecv)
        print('s_aba_evc:f_aba_evc',  s_aba_evc,f_aba_evc)
        print('s_bbb_ecc:f_bbb_ecc',  s_bbb_ecc,f_bbb_ecc)
        print('s_bbb_ecv:f_bbb_ecv',  s_bbb_ecv,f_bbb_ecv)

 



        r_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)
        r_bbb_ecc = r_bbb_ecc.reshape(nvir_b,-1)

        r_aaa_u_ecc = None
        r_aaa_u_ecc = np.zeros((nvir_a,ncvs,ncvs))
        r_aaa_u_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_aaa_ecc.copy()
        r_aaa_u_ecc[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_aaa_ecc.copy()

        r_bbb_u_ecc = None
        r_bbb_u_ecc = np.zeros((nvir_b,ncvs,ncvs))
        r_bbb_u_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_bbb_ecc.copy()
        r_bbb_u_ecc[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_bbb_ecc.copy()

        r_aaa_ecv = r_aaa_ecv.reshape(nvir_a,ncvs,nval_a)
        r_aba_ecc = r_aba_ecc.reshape(nvir_a,ncvs,ncvs)
        r_aba_ecv = r_aba_ecv.reshape(nvir_a,ncvs,nval_a)
        r_aba_evc = r_aba_ecv.reshape(nvir_a,nval_b,ncvs)
        r_bab_ecc = r_bab_ecc.reshape(nvir_b,ncvs,ncvs)
        r_bab_ecv = r_bab_ecv.reshape(nvir_b,ncvs,nval_b)
        r_bab_evc = r_bab_evc.reshape(nvir_b,nval_a,ncvs)
        r_bbb_ecv = r_bbb_ecv.reshape(nvir_b,ncvs,nval_b)

        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

############ ADC(2) ij block ############################
        # C-C (alpha) 
        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)

        # C-C (beta) 
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)
        
############# ADC(2) i - kja block #########################
        # ECC Block
        s[s_a:f_a] += 0.5*lib.einsum('jaki,ajk->i', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_aaa_u_ecc, optimize = True)
        s[s_a:f_a] -= 0.5*lib.einsum('kaji,ajk->i', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_aaa_u_ecc, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_ovoo[:ncvs,:,ncvs:,:ncvs], r_aaa_ecv, optimize = True)
        s[s_a:f_a] -= lib.einsum('kaji,ajk->i', eris_ovoo[ncvs:,:,:ncvs,:ncvs], r_aaa_ecv, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo[:ncvs,:,:ncvs,:ncvs], r_bab_ecc, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo[:ncvs,:,ncvs:,:ncvs], r_bab_ecv, optimize = True)
        s[s_a:f_a] += lib.einsum('jaki,ajk->i', eris_OVoo[ncvs:,:,:ncvs,:ncvs], r_bab_evc, optimize = True)

        s[s_b:f_b] += 0.5*lib.einsum('jaki,ajk->i', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_bbb_u_ecc, optimize = True)
        s[s_b:f_b] -= 0.5*lib.einsum('kaji,ajk->i', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_bbb_u_ecc, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_OVOO[:ncvs,:,ncvs:,:ncvs], r_bbb_ecv, optimize = True)
        s[s_b:f_b] -= lib.einsum('kaji,ajk->i', eris_OVOO[ncvs:,:,:ncvs,:ncvs], r_bbb_ecv, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO[:ncvs,:,:ncvs,:ncvs], r_aba_ecc, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO[:ncvs,:,ncvs:,:ncvs], r_aba_ecv, optimize = True)
        s[s_b:f_b] += lib.einsum('jaki,ajk->i', eris_ovOO[ncvs:,:,:ncvs,:ncvs], r_aba_evc, optimize = True)
        
############## ADC(2) ajk - i block ############################
         
        temp_aaa_ecc = lib.einsum('jaki,i->ajk', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_a, optimize = True)
        temp_aaa_ecc -= lib.einsum('kaji,i->ajk', eris_ovoo[:ncvs,:,:ncvs,:ncvs], r_a, optimize = True)
        s[s_aaa_ecc:f_aaa_ecc] += temp_aaa_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] += 2*lib.einsum('jaki,i->ajk', eris_ovoo[:ncvs,:,ncvs:,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] -= 2*lib.einsum('kaji,i->ajk', eris_ovoo[ncvs:,:,:ncvs,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_bab_ecc:f_bab_ecc] += lib.einsum('jaik,i->ajk', eris_OVoo[:ncvs,:,:ncvs,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_bab_ecv:f_bab_ecv] += lib.einsum('jaik,i->ajk', eris_OVoo[:ncvs,:,:ncvs,ncvs:], r_a, optimize = True).reshape(-1)
        s[s_bab_evc:f_bab_evc] += lib.einsum('jaik,i->ajk', eris_OVoo[ncvs:,:,:ncvs,:ncvs], r_a, optimize = True).reshape(-1)
        s[s_aba_ecc:f_aba_ecc] += lib.einsum('jaki,i->ajk', eris_ovOO[:ncvs,:,:ncvs,:ncvs], r_b, optimize = True).reshape(-1)
        s[s_aba_ecv:f_aba_ecv] += lib.einsum('jaki,i->ajk', eris_ovOO[:ncvs,:,ncvs:,:ncvs], r_b, optimize = True).reshape(-1)
        s[s_aba_evc:f_aba_evc] += lib.einsum('jaki,i->ajk', eris_ovOO[ncvs:,:,:ncvs,:ncvs], r_b, optimize = True).reshape(-1)
        temp_bbb_ecc = lib.einsum('jaki,i->ajk', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_b, optimize = True)
        temp_bbb_ecc -= lib.einsum('kaji,i->ajk', eris_OVOO[:ncvs,:,:ncvs,:ncvs], r_b, optimize = True)
        s[s_bbb_ecc:f_bbb_ecc] += temp_bbb_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] += 2*lib.einsum('jaki,i->ajk', eris_OVOO[:ncvs,:,ncvs:,:ncvs], r_b, optimize = True).reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] -= 2*lib.einsum('kaji,i->ajk', eris_OVOO[ncvs:,:,:ncvs,:ncvs], r_b, optimize = True).reshape(-1)
############ ADC(2) ajk - bil block ############################
        
        #temp_test = np.linspace(1,30,30)
         
        s[s_aaa_ecc:f_aaa_ecc] += D_aij_a_ecc * r_aaa_ecc.reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] += D_aij_a_ecv * r_aaa_ecv.reshape(-1)
        s[s_bab_ecc:f_bab_ecc] += D_aij_bab_ecc * r_bab_ecc.reshape(-1)
        s[s_bab_ecv:f_bab_ecv] += D_aij_bab_ecv * r_bab_ecv.reshape(-1)
        s[s_bab_evc:f_bab_evc] += D_aij_bab_evc * r_bab_evc.reshape(-1)
        s[s_aba_ecc:f_aba_ecc] += D_aij_aba_ecc * r_aba_ecc.reshape(-1)
        s[s_aba_ecv:f_aba_ecv] += D_aij_aba_ecv * r_aba_ecv.reshape(-1)
        s[s_aba_evc:f_aba_evc] += D_aij_aba_evc * r_aba_evc.reshape(-1)
        s[s_bbb_ecc:f_bbb_ecc] += D_aij_b_ecc * r_bbb_ecc.reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] += D_aij_b_ecv * r_bbb_ecv.reshape(-1)

        s[s_aaa_ecc:f_aaa_ecc] += 5 * r_aaa_ecc.reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] += 5 * r_aaa_ecv.reshape(-1)
        s[s_bab_ecc:f_bab_ecc] += 5 * r_bab_ecc.reshape(-1)
        s[s_bab_ecv:f_bab_ecv] += 5 * r_bab_ecv.reshape(-1)
        s[s_bab_evc:f_bab_evc] += 5 * r_bab_evc.reshape(-1)
        s[s_aba_ecc:f_aba_ecc] += 5 * r_aba_ecc.reshape(-1)
        s[s_aba_ecv:f_aba_ecv] += 5 * r_aba_ecv.reshape(-1)
        s[s_aba_evc:f_aba_evc] += 5 * r_aba_evc.reshape(-1)
        s[s_bbb_ecc:f_bbb_ecc] += 5 * r_bbb_ecc.reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] += 5 * r_bbb_ecv.reshape(-1)

        print('s_aaa_ecc:f_aaa_ecc', s_aaa_ecc,f_aaa_ecc)
        print('s_aaa_ecv:f_aaa_ecv', s_aaa_ecv,f_aaa_ecv)
        print('s_bab_ecc:f_bab_ecc', s_bab_ecc,f_bab_ecc)
        print('s_bab_ecv:f_bab_ecv', s_bab_ecv,f_bab_ecv)
        print('s_bab_evc:f_bab_evc', s_bab_evc,f_bab_evc)
        print('s_aba_ecc:f_aba_ecc', s_aba_ecc,f_aba_ecc)
        print('s_aba_ecv:f_aba_ecv', s_aba_ecv,f_aba_ecv)
        print('s_aba_evc:f_aba_evc', s_aba_evc,f_aba_evc)
        print('s_bbb_ecc:f_bbb_ecc', s_bbb_ecc,f_bbb_ecc)
        print('s_bbb_ecv:f_bbb_ecv', s_bbb_ecv,f_bbb_ecv)

        print('s[s_a:f_a]',             np.linalg.norm(s[s_a:f_a]))
        print(np.column_stack(np.unique(np.around(s[s_a:f_a], decimals=8), return_counts=True))) 
        print('s[s_b:f_b]' ,            np.linalg.norm(s[s_b:f_b]))
        print(np.column_stack(np.unique(np.around(s[s_b:f_b], decimals=8), return_counts=True))) 
        print('s[s_aaa_ecc:f_aaa_ecc]', np.linalg.norm(s[s_aaa_ecc:f_aaa_ecc]))      
        print(np.column_stack(np.unique(np.around(s[s_aaa_ecc:f_aaa_ecc], decimals=8), return_counts=True))) 
        print('[s_aaa_ecv:f_aaa_ecv]', np.linalg.norm(s[s_aaa_ecv:f_aaa_ecv]))
        print(np.column_stack(np.unique(np.around(s[s_aaa_ecv:f_aaa_ecv], decimals=8), return_counts=True))) 
        print('s[s_bab_ecc:f_bab_ecc]', np.linalg.norm(s[s_bab_ecc:f_bab_ecc]))
        print(np.column_stack(np.unique(np.around(s[s_bab_ecc:f_bab_ecc], decimals=8), return_counts=True))) 
        print('s[s_bab_ecv:f_bab_ecv]', np.linalg.norm(s[s_bab_ecv:f_bab_ecv]))
        print(np.column_stack(np.unique(np.around(s[s_bab_ecv:f_bab_ecv], decimals=8), return_counts=True))) 
        print('s[s_bab_evc:f_bab_evc]', np.linalg.norm(s[s_bab_evc:f_bab_evc]))
        print(np.column_stack(np.unique(np.around(s[s_bab_evc:f_bab_evc], decimals=8), return_counts=True))) 
        print('s[s_aba_ecc:f_aba_ecc]', np.linalg.norm(s[s_aba_ecc:f_aba_ecc]))
        print(np.column_stack(np.unique(np.around(s[s_aba_ecc:f_aba_ecc], decimals=8), return_counts=True))) 
        print('s[s_aba_ecv:f_aba_ecv]', np.linalg.norm(s[s_aba_ecv:f_aba_ecv]))
        print(np.column_stack(np.unique(np.around(s[s_aba_ecv:f_aba_ecv], decimals=8), return_counts=True))) 
        print('s[s_aba_evc:f_aba_evc]', np.linalg.norm(s[s_aba_evc:f_aba_evc]))
        print(np.column_stack(np.unique(np.around(s[s_aba_evc:f_aba_evc], decimals=8), return_counts=True))) 
        print('s[s_bbb_ecc:f_bbb_ecc]', np.linalg.norm(s[s_bbb_ecc:f_bbb_ecc]))
        print(np.column_stack(np.unique(np.around(s[s_bbb_ecc:f_bbb_ecc], decimals=8), return_counts=True))) 
        print('s[s_bbb_ecv:f_bbb_ecv]', np.linalg.norm(s[s_bbb_ecv:f_bbb_ecv]))
        print(np.column_stack(np.unique(np.around(s[s_bbb_ecv:f_bbb_ecv], decimals=8), return_counts=True)))

        print('s[s_a:f_a]',             np.linalg.norm(s[s_a:f_a]))
        print(np.column_stack(np.unique(np.around(s[s_a:f_a], decimals=8), return_counts=True))) 
        print('s[s_b:f_b]' ,            np.linalg.norm(s[s_b:f_b]))
        print(np.column_stack(np.unique(np.around(s[s_b:f_b], decimals=8), return_counts=True))) 
        print('s[s_aaa_ecc:f_aaa_ecv]', np.linalg.norm(s[s_aaa_ecc:f_aaa_ecv]))      
        print(np.column_stack(np.unique(np.around(s[s_aaa_ecc:f_aaa_ecv], decimals=8), return_counts=True))) 
        print('s[s_bab_ecc:f_bab_evc]', np.linalg.norm(s[s_bab_ecc:f_bab_evc]))
        print(np.column_stack(np.unique(np.around(s[s_bab_ecc:f_bab_evc], decimals=8), return_counts=True))) 
        print('s[s_aba_ecc:f_aba_evc]', np.linalg.norm(s[s_aba_ecc:f_aba_evc]))
        print(np.column_stack(np.unique(np.around(s[s_aba_ecc:f_aba_evc], decimals=8), return_counts=True))) 
        print('s[s_bbb_ecc:f_bbb_ecv]', np.linalg.norm(s[s_bbb_ecc:f_bbb_ecv]))
        print(np.column_stack(np.unique(np.around(s[s_bbb_ecc:f_bbb_ecv], decimals=8), return_counts=True))) 
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
               
               r_aaa = r_aaa.reshape(nvir_a,-1)
               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
               r_bbb = r_bbb.reshape(nvir_b,-1)
               
               r_aaa_u = None
               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()
               
               r_bbb_u = None
               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()
               
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
               t2_1_a = adc.t2[0][0][:]
               t2_1_ab = adc.t2[0][1][:]

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc_a

               a = 0
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

               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()

               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

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

               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)

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
        #s *= -1.0

        #return s
        
        
    return sigma_


def ea_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

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

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_a:

            T[s_a:f_a] = -t1_2_a[orb,:]

            t2_1_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(1,0,2,3)

            T[s_aaa:f_aaa] += t2_1_t[:,orb,:].reshape(-1)
            T[s_bab:f_bab] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :
            T[s_a:f_a] += idn_vir_a[(orb-nocc_a), :]
            T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)
######## ADC(3) 2p-1h  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb < nocc_a:

                t2_2_t = t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(1,0,2,3)

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

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab

######### spin = beta  ############################################
    else:
######## ADC(2) part  ############################################

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_b:

            T[s_b:f_b] = -t1_2_b[orb,:]

            t2_1_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(0,1,3,2)

            T[s_bbb:f_bbb] += t2_1_t[:,orb,:].reshape(-1)
            T[s_aba:f_aba] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :

            T[s_b:f_b] += idn_vir_b[(orb-nocc_b), :]
            T[s_b:f_b] -= 0.25*lib.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_1_b, optimize = True)
            T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)
            T[s_b:f_b] -= 0.25*lib.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)

######### ADC(3) 2p-1h part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb < nocc_b:

                t2_2_t = t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(0,1,3,2)

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

                del t2_2_b
                del t2_2_ab

        del t2_1_b
        del t2_1_ab

    return T


def ip_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1_2_a, t1_2_b = adc.t1[0]
    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b

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

    T = np.zeros((dim))

######## spin = alpha  ############################################
    if spin=="alpha":
######## ADC(2) 1h part  ############################################

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :]
            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
        else :
            T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]

######## ADC(2) 2h-1p  part  ############################################
            '''
            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
            t2_1_t_a = t2_1_t.transpose(2,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0)

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
            '''
######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            '''
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:]
                t2_2_t_a = t2_2_t.transpose(2,1,0)
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0)

                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
            '''
######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:
                     
                t2_1_a_tmp = np.ascontiguousarray(t2_1_a[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a_tmp, t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
       
                del t2_1_a_tmp, t2_1_ab_tmp

                t2_2_a_tmp = np.ascontiguousarray(t2_2_a[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a,  t2_2_a_tmp,optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab_tmp,optimize = True)

                del t2_2_a_tmp, t2_2_ab_tmp
            else:
                t2_1_a_tmp =  np.ascontiguousarray(t2_1_a[:,:,(orb-nocc_a),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,(orb-nocc_a),:])

                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a_tmp, t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab_tmp, t1_2_b,optimize = True)
                T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]
                del t2_1_a_tmp, t2_1_ab_tmp

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab
######## spin = beta  ############################################
    else:
######## ADC(2) 1h part  ############################################

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_b:
            
            t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
            t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

            T[s_b:f_b] = idn_occ_b[orb, :]
            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_1_b, optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_1_ab, optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_1_ab, optimize = True)
            del t2_1_b_tmp, t2_1_ab_tmp
        else :
            T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]

######## ADC(2) 2h-1p part  ############################################
            '''
            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
            t2_1_t_b = t2_1_t.transpose(2,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1)

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
            '''
######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            '''
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:]
                t2_2_t_b = t2_2_t.transpose(2,1,0)

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1)

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
             '''
######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_2_ab, optimize = True)

                del t2_1_b_tmp, t2_1_ab_tmp

                t2_2_b_tmp = np.ascontiguousarray(t2_2_b[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b,  t2_2_b_tmp ,optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab, t2_2_ab_tmp,optimize = True)
                
                del t2_2_b_tmp, t2_2_ab_tmp

            else:
                t2_1_b_tmp  = np.ascontiguousarray(t2_1_b[:,:,(orb-nocc_b),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,:,(orb-nocc_b)])

                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b_tmp, t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab_tmp, t1_2_a,optimize = True)
                T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]
                del t2_1_b_tmp, t2_1_ab_tmp
                del t2_2_b
                del t2_2_ab

        del t2_1_b
        del t2_1_ab

    return T

def ip_cvs_compute_trans_moments(adc, orb, spin="alpha"):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1_2_a, t1_2_b = adc.t1[0]
    t2_1_a = adc.t2[0][0][:]
    t2_1_ab = adc.t2[0][1][:]
    t2_1_b = adc.t2[0][2][:]

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs

    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    n_singles_a = ncvs
    n_singles_b = ncvs
    n_doubles_aaa_ecc = nvir_a * ncvs * (ncvs - 1) // 2
    n_doubles_aaa_ecv = nvir_a * ncvs * nval_a  
    n_doubles_bba_ecc = nvir_b * ncvs * ncvs
    n_doubles_bba_ecv = nvir_b * ncvs * nval_a
    n_doubles_bba_evc = nvir_b * nval_b * ncvs
    n_doubles_aab_ecc = nvir_a * ncvs * ncvs
    n_doubles_aab_ecv = nvir_a * ncvs * nval_b
    n_doubles_aab_evc = nvir_a * nval_a * ncvs
    n_doubles_bbb_ecc = nvir_b * ncvs * (ncvs - 1) // 2
    n_doubles_bbb_ecv = nvir_b * ncvs * nval_b
    dim = n_singles_a + n_singles_b + n_doubles_aaa_ecc + n_doubles_aaa_ecv + n_doubles_bba_ecc + n_doubles_bba_ecv + n_doubles_bba_evc + n_doubles_aab_ecc + n_doubles_aab_ecv + n_doubles_aab_evc + n_doubles_bbb_ecc + n_doubles_bbb_ecv

    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    '''   
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
    '''

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bba_ecc = f_aaa_ecv
    f_bba_ecc = s_bba_ecc + n_doubles_bba_ecc
    s_bba_ecv = f_bba_ecc
    f_bba_ecv = s_bba_ecv + n_doubles_bba_ecv
    s_bba_evc = f_bba_ecv
    f_bba_evc = s_bba_evc + n_doubles_bba_evc
    s_aab_ecc = f_bba_evc
    f_aab_ecc = s_aab_ecc + n_doubles_aab_ecc
    s_aab_ecv = f_aab_ecc
    f_aab_ecv = s_aab_ecv + n_doubles_aab_ecv
    s_aab_evc = f_aab_ecv
    f_aab_evc = s_aab_evc + n_doubles_aab_evc
    s_bbb_ecc = f_aab_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

    T = np.zeros((dim))

######## spin = alpha  ############################################
    if spin=="alpha":
######## ADC(2) 1h part  ############################################

        t2_1_a = adc.t2[0][0][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :ncvs]
            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a[:ncvs,:,:,:], optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab[:ncvs,:,:,:], optimize = True)
        else :
            T[s_a:f_a] += t1_2_a[:ncvs,(orb-nocc_a)]

######## ADC(2) 2h-1p  part  ############################################
            ''' 
            #t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
            t2_1_t_ecc = t2_1_a[:ncvs,:ncvs,:,:].copy()
            t2_1_t_ecc = t2_1_t_ecc[ij_ind_ncvs[0],ij_ind_ncvs[1],:,:]
            t2_1_t_a_ecc = t2_1_t_ecc.transpose(2,1,0)
             
            t2_1_t_a_ecv = t2_1_a.transpose(2,3,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0)

            print('t2_1_a shape: ',t2_1_a.shape )
            print('T[s_bba_ecc:f_bba_ecc]: ', T[s_bba_ecc:f_bba_ecc].shape)
             
            T[s_aaa_ecc:f_aaa_ecc] = t2_1_t_a_ecc[(orb-nocc_a),:,:].reshape(-1)
            T[s_aaa_ecv:f_aaa_ecv] = t2_1_t_a_ecv[(orb-nocc_a),:,:ncvs,ncvs:].reshape(-1)
            T[s_bba_ecc:f_bba_ecc] = t2_1_t_ab[(orb-nocc_a),:,:ncvs,:ncvs].reshape(-1)
            T[s_bba_ecv:f_bba_ecv] = t2_1_t_ab[(orb-nocc_a),:,:ncvs,ncvs:].reshape(-1)
            T[s_bba_evc:f_bba_evc] = t2_1_t_ab[(orb-nocc_a),:,:ncvs,ncvs:].reshape(-1)
            '''
######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            '''
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]

            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:]
                t2_2_t_a = t2_2_t.transpose(2,1,0)
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0)

                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
           '''
######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:
                     
                t2_1_a_tmp = np.ascontiguousarray(t2_1_a[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a_tmp, t2_2_a[:ncvs,:,:,:], optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab_tmp, t2_2_ab[:ncvs,:,:,:], optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab_tmp, t2_2_ab[:ncvs,:,:,:], optimize = True)
       
                del t2_1_a_tmp, t2_1_ab_tmp

                t2_2_a_tmp = np.ascontiguousarray(t2_2_a[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a[:ncvs,:,:,:],  t2_2_a_tmp,optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab[:ncvs,:,:,:], t2_2_ab_tmp,optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab[:ncvs,:,:,:], t2_2_ab_tmp,optimize = True)

                del t2_2_a_tmp, t2_2_ab_tmp
            else:
                t2_1_a_tmp =  np.ascontiguousarray(t2_1_a[:,:,(orb-nocc_a),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,(orb-nocc_a),:])

                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a_tmp[:ncvs,:,:], t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab_tmp[:ncvs,:,:], t1_2_b,optimize = True)
                T[s_a:f_a] += t1_3_a[:ncvs,(orb-nocc_a)]
                del t2_1_a_tmp, t2_1_ab_tmp

                del t2_2_a
                del t2_2_ab

        del t2_1_a
        del t2_1_ab
######## spin = beta  ############################################
    else:
######## ADC(2) 1h part  ############################################

        t2_1_b = adc.t2[0][2][:]
        t2_1_ab = adc.t2[0][1][:]
        if orb < nocc_b:
            
            t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
            t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

            T[s_b:f_b] = idn_occ_b[orb, :ncvs]
            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_1_b[:ncvs,:,:,:], optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_1_ab[:,:ncvs,:,:], optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_1_ab[:,:ncvs,:,:], optimize = True)
            del t2_1_b_tmp, t2_1_ab_tmp
        else :
            T[s_b:f_b] += t1_2_b[:ncvs,(orb-nocc_b)]

######## ADC(2) 2h-1p part  ############################################

            #t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
            #t2_1_t_b = t2_1_t.transpose(2,1,0)
            #t2_1_t_ab = t2_1_ab.transpose(2,3,0,1)

            #T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            #T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
            '''
            t2_1_t_ecc = t2_1_b[:ncvs,:ncvs,:,:].copy()
            t2_1_t_ecc = t2_1_t_ecc[ij_ind_ncvs[0],ij_ind_ncvs[1],:,:]
            t2_1_t_b_ecc = t2_1_t_ecc.transpose(2,1,0)
            t2_1_t_b_ecv = t2_1_b.transpose(2,3,0,1)
            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1)
             
            T[s_bbb_ecc:f_bbb_ecc] = t2_1_t_b_ecc[(orb-nocc_a),:,:].reshape(-1)
            T[s_bbb_ecv:f_bbb_ecv] = t2_1_t_b_ecv[:,(orb-nocc_a),:ncvs,ncvs:].reshape(-1)
            T[s_bba_ecc:f_bba_ecc] = t2_1_t_ab[(orb-nocc_a),:,:ncvs,:ncvs].reshape(-1)
            T[s_bba_ecv:f_bba_ecv] = t2_1_t_ab[(orb-nocc_a),:,:ncvs,ncvs:].reshape(-1)
            T[s_bba_evc:f_bba_evc] = t2_1_t_ab[(orb-nocc_a),:,ncvs:,:ncvs].reshape(-1)
            '''
######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            '''
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]

            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:]
                t2_2_t_b = t2_2_t.transpose(2,1,0)

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1)

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
            '''
######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_2_b[:ncvs,:,:,:], optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_2_ab[:,:ncvs,:,:], optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_2_ab[:,:ncvs,:,:], optimize = True)

                del t2_1_b_tmp, t2_1_ab_tmp

                t2_2_b_tmp = np.ascontiguousarray(t2_2_b[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b[:ncvs,:,:,:],  t2_2_b_tmp ,optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab[:,:ncvs,:,:], t2_2_ab_tmp,optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab[:,:ncvs,:,:], t2_2_ab_tmp,optimize = True)
                
                del t2_2_b_tmp, t2_2_ab_tmp

            else:
                t2_1_b_tmp  = np.ascontiguousarray(t2_1_b[:,:,(orb-nocc_b),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:,:,(orb-nocc_b)])

                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b_tmp[:ncvs,:,:], t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab_tmp[:,:ncvs,:], t1_2_a,optimize = True)
                T[s_b:f_b] += t1_3_b[:ncvs,(orb-nocc_b)]
                del t2_1_b_tmp, t2_1_ab_tmp
                del t2_2_b
                del t2_2_ab

        del t2_1_b
        del t2_1_ab

    return T

def get_trans_moments(adc):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b
    ncvs = adc.ncvs

    T_a = []
    T_b = []

    for orb in range(nmo_a):

            T_aa = adc.compute_trans_moments(orb, spin = "alpha")
            T_a.append(T_aa)

    for orb in range(nmo_b):

            T_bb = adc.compute_trans_moments(orb, spin = "beta")
            T_b.append(T_bb)
    
    cput0 = log.timer_debug1("completed spec vactor calc in ADC(3) calculation", *cput0)
    return (T_a, T_b)
'''
def temp_spec_factors(adc, U):

    T = adc.get_trans_moments()
    
    T_a = T[0]
    T_b = T[1]

    T_a = np.array(T_a)
    T_b = np.array(T_b)
    nmo_a  = adc.nmo_a
    nmo_b  = adc.nmo_b
    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs
    ij_ind_ncvs = np.tril_indices(ncvs, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa_ecc = f_b
    f_aaa_ecc = s_aaa_ecc + n_doubles_aaa_ecc
    s_aaa_ecv = f_aaa_ecc
    f_aaa_ecv = s_aaa_ecv + n_doubles_aaa_ecv
    s_bba_ecc = f_aaa_ecv
    f_bba_ecc = s_bba_ecc + n_doubles_bba_ecc
    s_bba_ecv = f_bba_ecc
    f_bba_ecv = s_bba_ecv + n_doubles_bba_ecv
    s_bba_evc = f_bba_ecv
    f_bba_evc = s_bba_evc + n_doubles_bba_evc
    s_aab_ecc = f_bba_evc
    f_aab_ecc = s_aab_ecc + n_doubles_aab_ecc
    s_aab_ecv = f_aab_ecc
    f_aab_ecv = s_aab_ecv + n_doubles_aab_ecv
    s_aab_evc = f_aab_ecv
    f_aab_evc = s_aab_evc + n_doubles_aab_evc
    s_bbb_ecc = f_aab_evc
    f_bbb_ecc = s_bbb_ecc + n_doubles_bbb_ecc
    s_bbb_ecv = f_bbb_ecc
    f_bbb_ecv = s_bbb_ecv + n_doubles_bbb_ecv

     Ta_a = T_a[:, s_a:f_a]
     Ta_b = T_a[:, s_b:f_b]
     Ta_aaa_ecc = T_a[:, s_aaa_ecc:f_aaa_ecc]
     Ta_aaa_ecv = T_a[:, s_aaa_ecv:f_aaa_ecv].reshape(nmo_a,nvir_a,ncvs,nval_a)
     Ta_bba_ecc = T_a[:, s_bba_ecc:f_bba_ecc].reshape(nmo_a,nvir_b,ncvs,ncvs)
     Ta_bba_ecv = T_a[:, s_bba_ecv:f_bba_ecv].reshape(nmo_a,nvir_b,ncvs,nval_a)
     Ta_bba_evc = T_a[:, s_bba_evc:f_bba_evc].reshape(nmo_a,nvir_b,nval_b,ncvs)
     Ta_aab_ecc = T_a[:, s_aab_ecc:f_aab_ecc].reshape(nmo_a,nvir_a,ncvs,ncvs)
     Ta_aab_ecv = T_a[:, s_aab_ecv:f_aab_ecv].reshape(nmo_a,nvir_a,ncvs,nval_b)
     Ta_aab_evc = T_a[:, s_aab_evc:f_aab_evc].reshape(nmo_a,nvir_a,nval_a,ncvs)
     Ta_bbb_ecc = T_a[:, s_bbb_ecc:f_bbb_ecc]
     Ta_bbb_ecv = T_a[:, s_bbb_ecv:f_bbb_ecv].reshape(nmo_a,nvir_b,ncvs,nval_b)

       
     Ta_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)
     Ta_bbb_ecc = r_bbb_ecc.reshape(nvir_b,-1)

     Ta_aaa_ecc_u = None
     Ta_aaa_ecc_u = np.zeros((nvir_a,ncvs,ncvs))
     Ta_aaa_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= Ta_aaa_ecc.copy()
     Ta_aaa_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -Ta_aaa_ecc.copy()

     Ta_bbb_ecc_u = None
     Ta_bbb_ecc_u = np.zeros((nvir_b,ncvs,ncvs))
     Ta_bbb_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= Ta_bbb_ecc.copy()
     Ta_bbb_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -Ta_bbb_ecc.copy()
        

     Ta_aaa_ecv = Ta_aaa_ecv.reshape(nvir_a,ncvs,nval_a)
     Ta_bba_ecc = Ta_bba_ecc.reshape(nvir_b,ncvs,ncvs)
     Ta_bba_ecv = Ta_bba_ecv.reshape(nvir_b,ncvs,nval_a)
     Ta_bba_evc = Ta_bba_evc.reshape(nvir_b,nval_b,ncvs)
     Ta_aab_ecc = Ta_aab_ecc.reshape(nvir_a,ncvs,ncvs)
     Ta_aab_ecv = Ta_aab_ecv.reshape(nvir_a,ncvs,nval_b)
     Ta_aab_evc = Ta_aab_evc.reshape(nvir_a,nval_a,ncvs)
     Ta_bbb_ecv = Ta_bbb_ecv.reshape(nvir_b,ncvs,nval_b)

    T_a_cvs = np.zeros((nmo_a, dim))
    T_b_cvs = np.zeros((nmo_b, dim))
     
    T_a_cvs[:]Ta_a 
    Ta_b  
    Ta_aaa_ecc
    Ta_aaa_ecv 
    Ta_bba_ecc 
    Ta_bba_ecv 
    Ta_bba_evc 
    Ta_aab_ecc 
    Ta_aab_ecv 
    Ta_aab_evc 
    Ta_bbb_ecc 
    Ta_bbb_ecv 
    #Spectroscopic amplitudes
    X_a = np.dot(T_a, U).reshape(-1,nroots)
    X_b = np.dot(T_b, U).reshape(-1,nroots)

    X = (X_a,X_b)

    #Spectroscopic factors
    P = lib.einsum("pi,pi->i", X_a, X_a)
    P += lib.einsum("pi,pi->i", X_b, X_b)

    return P, X
'''   
def analyze_spec_factor(adc):

    X_a = adc.X[0]
    X_b = adc.X[1]

    logger.info(adc, "Print spectroscopic factors > %E\n", adc.spec_factor_print_tol)

    X_tot = (X_a, X_b)

    for iter_idx, X in enumerate(X_tot):
        if iter_idx == 0:
            spin = "alpha"
        else:
            spin = "beta"

        X_2 = (X.copy()**2)

        thresh = adc.spec_factor_print_tol

        for i in range(X_2.shape[1]):

            sort = np.argsort(-X_2[:,i])
            X_2_row = X_2[:,i]

            X_2_row = X_2_row[sort]

            if adc.mol.symmetry == False:
                sym = np.repeat(['A'], X_2_row.shape[0])
            else:
                if spin == "alpha":
                    sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff[0].orbsym]
                    sym = np.array(sym)
                else:
                    sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff[1].orbsym]
                    sym = np.array(sym)

                sym = sym[sort]

            spec_Contribution = X_2_row[X_2_row > thresh]
            index_mo = sort[X_2_row > thresh]+1

            if np.sum(spec_Contribution) == 0.0:
                continue

            logger.info(adc, '%s | root %d %s\n', adc.method, i, spin)
            logger.info(adc, "     HF MO     Spec. Contribution     Orbital symmetry")
            logger.info(adc, "-----------------------------------------------------------")

            for c in range(index_mo.shape[0]):
                logger.info(adc, '     %3.d          %10.8f                %s', index_mo[c], spec_Contribution[c], sym[c])

            logger.info(adc, '\nPartial spec. factor sum = %10.8f', np.sum(spec_Contribution))
            logger.info(adc, "\n*************************************************************\n")


def analyze_eigenvector_ea(adc):

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    evec_print_tol = adc.evec_print_tol

    logger.info(adc, "Number of alpha occupied orbitals = %d", nocc_a)
    logger.info(adc, "Number of beta occupied orbitals = %d", nocc_b)
    logger.info(adc, "Number of alpha virtual orbitals =  %d", nvir_a)
    logger.info(adc, "Number of beta virtual orbitals =  %d", nvir_b)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)
    ab_a = np.tril_indices(nvir_a, k=-1)
    ab_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a* (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a* nvir_b
    n_doubles_aba = nocc_a * nvir_b* nvir_a
    n_doubles_bbb = nvir_b* (nvir_b - 1) * nocc_b // 2

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

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:f_b, I]
        U2 = U[f_b:, I]
        U1dotU1 = np.dot(U1, U1) 
        U2dotU2 = np.dot(U2, U2) 
           
        temp_aaa = np.zeros((nocc_a, nvir_a, nvir_a))
        temp_aaa[:,ab_a[0],ab_a[1]] =  U[s_aaa:f_aaa,I].reshape(nocc_a,-1).copy()
        temp_aaa[:,ab_a[1],ab_a[0]] = -U[s_aaa:f_aaa,I].reshape(nocc_a,-1).copy()
        U_aaa = temp_aaa.reshape(-1).copy()

        temp_bbb = np.zeros((nocc_b, nvir_b, nvir_b))
        temp_bbb[:,ab_b[0],ab_b[1]] =  U[s_bbb:f_bbb,I].reshape(nocc_b,-1).copy()
        temp_bbb[:,ab_b[1],ab_b[0]] = -U[s_bbb:f_bbb,I].reshape(nocc_b,-1).copy()
        U_bbb = temp_bbb.reshape(-1).copy()

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx] 
        U_sorted = U[ind_idx,I].copy()

        U_sq_aaa = U_aaa.copy()**2
        U_sq_bbb = U_bbb.copy()**2
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
        for orb_idx in ind_idx:

            if orb_idx in range(s_a,f_a):
                a_idx = orb_idx + 1 + nocc_a
                singles_a_idx.append(a_idx)
                singles_a_val.append(U_sorted[iter_idx])
               
            if orb_idx in range(s_b,f_b):
                a_idx = orb_idx - s_b + 1 + nocc_b
                singles_b_idx.append(a_idx)
                singles_b_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bab,f_bab):
                iab_idx = orb_idx - s_bab       
                ab_rem = iab_idx % (nvir_a*nvir_b)
                i_idx = iab_idx//(nvir_a*nvir_b)
                a_idx = ab_rem//nvir_b
                b_idx = ab_rem % nvir_b
                doubles_bab_idx.append((i_idx + 1, a_idx + 1 + nocc_a, b_idx + 1 + nocc_b))
                doubles_bab_val.append(U_sorted[iter_idx])
          
            if orb_idx in range(s_aba,f_aba):
                iab_idx = orb_idx - s_aba     
                ab_rem = iab_idx % (nvir_b*nvir_a)
                i_idx = iab_idx//(nvir_b*nvir_a)
                a_idx = ab_rem//nvir_a
                b_idx = ab_rem % nvir_a
                doubles_aba_idx.append((i_idx + 1, a_idx + 1 + nocc_b, b_idx + 1 + nocc_a))
                doubles_aba_val.append(U_sorted[iter_idx])

            iter_idx += 1
             
        for orb_aaa in ind_idx_aaa:              
            ab_rem = orb_aaa % (nvir_a*nvir_a)
            i_idx = orb_aaa//(nvir_a*nvir_a)
            a_idx = ab_rem//nvir_a
            b_idx = ab_rem % nvir_a
            doubles_aaa_idx.append((i_idx + 1, a_idx + 1 + nocc_a, b_idx + 1 + nocc_a))

        for orb_bbb in ind_idx_bbb:                
            ab_rem = orb_bbb % (nvir_b*nvir_b)
            i_idx = orb_bbb//(nvir_b*nvir_b)
            a_idx = ab_rem//nvir_b
            b_idx = ab_rem % nvir_b
            doubles_bbb_idx.append((i_idx + 1, a_idx + 1 + nocc_b, b_idx + 1 + nocc_b))
        
        doubles_aaa_val = list(U_sorted_aaa)
        doubles_bbb_val = list(U_sorted_bbb)
        
        logger.info(adc,'%s | root %d | norm(1p)  = %6.4f | norm(1h2p) = %6.4f ',adc.method ,I, U1dotU1, U2dotU2)

        if singles_a_val:
            logger.info(adc, "\n1p(alpha) block: ") 
            logger.info(adc, "     a     U(a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_a_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_a_val[idx])

        if singles_b_val:
            logger.info(adc, "\n1p(beta) block: ") 
            logger.info(adc, "     a     U(a)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_b_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_b_val[idx])

        if doubles_aaa_val:
            logger.info(adc, "\n1h2p(alpha|alpha|alpha) block: ") 
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aaa_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[0], print_doubles[1], print_doubles[2], doubles_aaa_val[idx])

        if doubles_bab_val:
            logger.info(adc, "\n1h2p(beta|alpha|beta) block: ") 
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bab_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[0], print_doubles[1], print_doubles[2], doubles_bab_val[idx])

        if doubles_aba_val:
            logger.info(adc, "\n1h2p(alpha|beta|alpha) block: ") 
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aba_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[0], print_doubles[1], print_doubles[2], doubles_aba_val[idx])

        if doubles_bbb_val:
            logger.info(adc, "\n1h2p(beta|beta|beta) block: ") 
            logger.info(adc, "     i     a     b     U(i,a,b)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bbb_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[0], print_doubles[1], print_doubles[2], doubles_bbb_val[idx])
                
        logger.info(adc, "\n*************************************************************\n")


def analyze_eigenvector_ip(adc):

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    evec_print_tol = adc.evec_print_tol

    logger.info(adc, "Number of alpha occupied orbitals = %d", nocc_a)
    logger.info(adc, "Number of beta occupied orbitals = %d", nocc_b)
    logger.info(adc, "Number of alpha virtual orbitals =  %d", nvir_a)
    logger.info(adc, "Number of beta virtual orbitals =  %d", nvir_b)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

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

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:f_b,I]
        U2 = U[f_b:,I]
        U1dotU1 = np.dot(U1, U1) 
        U2dotU2 = np.dot(U2, U2) 

        temp_aaa = np.zeros((nvir_a, nocc_a, nocc_a))
        temp_aaa[:,ij_a[0],ij_a[1]] =  U[s_aaa:f_aaa,I].reshape(nvir_a,-1).copy()
        temp_aaa[:,ij_a[1],ij_a[0]] = -U[s_aaa:f_aaa,I].reshape(nvir_a,-1).copy()
        U_aaa = temp_aaa.reshape(-1).copy()

        temp_bbb = np.zeros((nvir_b, nocc_b, nocc_b))
        temp_bbb[:,ij_b[0],ij_b[1]] =  U[s_bbb:f_bbb,I].reshape(nvir_b,-1).copy()
        temp_bbb[:,ij_b[1],ij_b[0]] = -U[s_bbb:f_bbb,I].reshape(nvir_b,-1).copy()
        U_bbb = temp_bbb.reshape(-1).copy()

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx] 
        U_sorted = U[ind_idx,I].copy()

        U_sq_aaa = U_aaa.copy()**2
        U_sq_bbb = U_bbb.copy()**2
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
        for orb_idx in ind_idx:

            if orb_idx in range(s_a,f_a):
                i_idx = orb_idx + 1
                singles_a_idx.append(i_idx)
                singles_a_val.append(U_sorted[iter_idx])
               
            if orb_idx in range(s_b,f_b):
                i_idx = orb_idx - s_b + 1
                singles_b_idx.append(i_idx)
                singles_b_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bab,f_bab):
                aij_idx = orb_idx - s_bab       
                ij_rem = aij_idx % (nocc_a*nocc_b)
                a_idx = aij_idx//(nocc_a*nocc_b)
                i_idx = ij_rem//nocc_a
                j_idx = ij_rem % nocc_a
                doubles_bab_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bab_val.append(U_sorted[iter_idx])
          
            if orb_idx in range(s_aba,f_aba):
                aij_idx = orb_idx - s_aba     
                ij_rem = aij_idx % (nocc_b*nocc_a)
                a_idx = aij_idx//(nocc_b*nocc_a)
                i_idx = ij_rem//nocc_b
                j_idx = ij_rem % nocc_b
                doubles_aba_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                doubles_aba_val.append(U_sorted[iter_idx])

            iter_idx += 1
             
        for orb_aaa in ind_idx_aaa:              
            ij_rem = orb_aaa % (nocc_a*nocc_a)
            a_idx = orb_aaa//(nocc_a*nocc_a)
            i_idx = ij_rem//nocc_a
            j_idx = ij_rem % nocc_a
            doubles_aaa_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))

        for orb_bbb in ind_idx_bbb:                
            ij_rem = orb_bbb % (nocc_b*nocc_b)
            a_idx = orb_bbb//(nocc_b*nocc_b)
            i_idx = ij_rem//nocc_b
            j_idx = ij_rem % nocc_b
            doubles_bbb_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))

        doubles_aaa_val = list(U_sorted_aaa)
        doubles_bbb_val = list(U_sorted_bbb)
        
        logger.info(adc,'%s | root %d | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',adc.method ,I, U1dotU1, U2dotU2)

        if singles_a_val:
            logger.info(adc, "\n1h(alpha) block: ") 
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_a_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_a_val[idx])

        if singles_b_val:
            logger.info(adc, "\n1h(beta) block: ") 
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_b_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_b_val[idx])

        if doubles_aaa_val:
            logger.info(adc, "\n2h1p(alpha|alpha|alpha) block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aaa_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_aaa_val[idx])

        if doubles_bab_val:
            logger.info(adc, "\n2h1p(beta|alpha|beta) block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bab_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_bab_val[idx])

        if doubles_aba_val:
            logger.info(adc, "\n2h1p(alpha|beta|alpha) block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aba_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_aba_val[idx])

        if doubles_bbb_val:
            logger.info(adc, "\n2h1p(beta|beta|beta) block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bbb_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_bbb_val[idx])

        logger.info(adc, "\n*************************************************************\n")


def get_properties(adc, nroots=1, U=None):

    #Transition moments
    T = adc.get_trans_moments()
    
    T_a = T[0]
    T_b = T[1]

    T_a = np.array(T_a)
    T_b = np.array(T_b)

    #U = adc.U

    #Spectroscopic amplitudes
    X_a = np.dot(T_a, U).reshape(-1,nroots)
    X_b = np.dot(T_b, U).reshape(-1,nroots)

    X = (X_a,X_b)

    #Spectroscopic factors
    P = lib.einsum("pi,pi->i", X_a, X_a)
    P += lib.einsum("pi,pi->i", X_b, X_b)

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
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.imds = adc.imds
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
        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol

        self.compute_properties = adc.compute_properties
        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)
    
    kernel = kernel
    get_imds = get_imds_ea
    matvec = ea_adc_matvec
    get_diag = ea_adc_diag
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    analyze_spec_factor = analyze_spec_factor
    get_properties = get_properties
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo
    
    analyze_eigenvector = analyze_eigenvector_ea

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
    
    def gen_matvec(self, imds=None, eris=None):
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
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.imds = adc.imds
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
        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol
        self.ncvs= adc.ncvs

        self.compute_properties = adc.compute_properties
        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_properties = get_properties

    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector_ip
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.ip_adc_diag()
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

    def gen_matvec(self, imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
        #matvec = lambda x: self.matvec()
        return matvec, diag

class UADCIPCVS(UADC):
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
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.imds = adc.imds
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
        self.spec_factor_print_tol = adc.spec_factor_print_tol
        self.evec_print_tol = adc.evec_print_tol

        self.ncvs= adc.ncvs
        self.compute_properties = adc.compute_properties
        self.E = adc.E
        self.U = adc.U
        self.P = adc.P
        self.X = adc.X

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'method_type', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    #get_imds = get_imds_ip_cvs
    get_imds = get_imds_ip
    get_diag = ip_cvs_adc_diag
    matvec = ip_cvs_adc_matvec
    compute_trans_moments = ip_cvs_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_properties = get_properties

    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector_ip
    analyze = analyze
    compute_dyson_mo = compute_dyson_mo

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.ip_cvs_adc_diag()
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

    def gen_matvec(self, imds=None, eris=None):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds,eris)
        matvec = self.matvec(imds, eris)
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
