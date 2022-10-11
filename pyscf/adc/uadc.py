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
    conv, adc.E, U = lib.linalg_helper.davidson1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space,tol_residual=adc.tol_residual)
    U = adc.U = np.array(U).T.copy()
    
    if adc.method_type == 'ip': 
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
        total_tpdm_a = ()
        total_tpdm_b = ()
        #U = U[:,::2]
        for i in range(nroots):
            for j in range(nroots):
                #norm_a_i = np.linalg.norm(U[s_a:f_a,i]) > np.linalg.norm(U[s_b:f_b,i])
                #norm_a_j = np.linalg.norm(U[s_a:f_a,j]) > np.linalg.norm(U[s_b:f_b,j])
                #norm_aaa_i = np.linalg.norm(U[s_aaa:f_aaa,i]) > np.linalg.norm(U[s_bbb:f_bbb,i])
                #norm_aaa_j = np.linalg.norm(U[s_aaa:f_aaa,j]) > np.linalg.norm(U[s_bbb:f_bbb,j])
                #if i == j and ((norm_a_i and norm_a_j) or (norm_aaa_i and norm_aaa_j)):
                #    print('i = ', i, ' j = ', j) 
                #    tpdm_a, tpdm_b = compute_rdm_tdm(adc, U[:,i], U[:,j])
                #    total_tpdm_a += (tpdm_a,)
                #if i == j and ((norm_a_i is False and norm_a_j is False) or (norm_aaa_i is False and norm_aaa_j is False)):
                #    print('i = ', i, ' j = ', j) 
                #    tpdm_a, tpdm_b = compute_rdm_tdm(adc, U[:,i], U[:,j])
                #    total_tpdm_b += (tpdm_b,)
                if i == j:# or i != j :
                    print('i = ', i, ' j = ', j) 
                    tpdm_a, tpdm_b = compute_rdm_tdm(adc, U[:,i], U[:,j])
                    total_tpdm_a += (tpdm_a,)
                    total_tpdm_b += (tpdm_b,)

        nmo = nocc_a + nvir_a
        print('total tpdm_a tuple length',len(total_tpdm_a))
        #big_tensor_opdm = np.zeros((nmo, len(total_tpdm_a)))
        big_tensor_opdm = np.zeros(len(total_tpdm_a))
        opdm_trace = np.zeros(len(total_tpdm_a))
        idx = 0
        Herm_sum = 0
        for opdm in zip(total_tpdm_a, total_tpdm_b):
            print('density matrix idx = ', idx)
            opdm_a, opdm_b = opdm
            #opdm_a, opdm_b = np.rot90(opdm_a), np.rot90(opdm_b)
            e_a, v = np.linalg.eigh(opdm_a)
            e_b, v = np.linalg.eigh(opdm_b)
            e_a_idx = np.argsort(-e_a)
            e_b_idx = np.argsort(-e_b)
            #big_tensor_opdm[:,idx] += e_a[e_a_idx] + e_b[e_b_idx]
            big_tensor_opdm[idx] += np.sum(e_a) + np.sum(e_b)
            opdm_trace[idx] += np.einsum('pp', opdm_a + opdm_b, optimize=True)
            Herm_sum += np.linalg.norm(opdm_a - opdm_a.T)
            Herm_sum += np.linalg.norm(opdm_b - opdm_b.T)
            idx += 1


        #print('Herm sum UADC: ', Herm_sum) 
        #print('eigs sum: ', big_tensor_opdm)
        #print('traces: ', opdm_trace)
        #eigs_sum_idx = np.argsort(-big_tensor_opdm)
        #traces_idx = np.argsort(-opdm_trace)
        #big_tensor_opdm = big_tensor_opdm[eigs_sum_idx][::2]
        #opdm_trace = opdm_trace[eigs_sum_idx][::2] 
        ##big_tensor_opdm = big_tensor_opdm[:,::2]
        ##opdm_trace = opdm_trace[::2]
        ##opdm_trace = np.abs(opdm_trace)
        #opdm_trace = np.abs(opdm_trace)
        #trace_sort_idx = np.argsort(-opdm_trace)
        #opdm_trace = opdm_trace[trace_sort_idx]
        #big_tensor_opdm = np.abs(big_tensor_opdm[trace_sort_idx])
        #print('opdm_trace = ', opdm_trace)
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
        print_string = ('%s root %d  |  Energy (Eh) = %14.14f  |  Energy (eV) = %12.8f  ' % (adc.method, n, adc.E[n], adc.E[n]*27.2114))
        if adc.compute_properties:
            print_string += ("|  Spec factors = %10.14f  " % adc.P[n])
        print_string += ("|  conv = %s" % conv[n])
        logger.info(adc, print_string)

    log.timer('ADC', *cput0)

    #return adc.E, (big_tensor_opdm, opdm_trace), adc.P, adc.X
    #return adc.E, adc.U, adc.P, adc.X
    return adc.E, adc.P

def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1, t2, myadc.imds.t2_1_vvvv = myadc.compute_amplitudes(eris)
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def freezingcore(t,nfc):
    t = np.array(t)
    if t.ndim == 2:
        t[:nfc,:] = 0 
    if t.ndim == 4:
        t[:nfc,:nfc,:,:] = 0
        t[:nfc,:,:,:] = 0
        t[:,:nfc,:,:] = 0
    return t

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

    nfc = 1
    t1_2_a = freezingcore(t1_2_a,nfc)
    t1_2_b = freezingcore(t1_2_b,nfc)

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

    t2_1_a = freezingcore(t2_1_a,nfc)
    t2_1_ab = freezingcore(t2_1_ab,nfc)
    t2_1_b = freezingcore(t2_1_b,nfc)

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

       t2_2_a = freezingcore(t2_2_a,nfc)
       t2_2_ab = freezingcore(t2_2_ab,nfc)
       t2_2_b = freezingcore(t2_2_b,nfc)

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
        self.mo_energy_a = mf.mo_energy[0] + 1e-10
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
            e_exc, v_exc, spec_fac, X, adc_es = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        elif(self.method_type == "ip-cvs"):
            e_exc, v_exc, spec_fac, X, adc_es = self.ip_cvs_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)

        self._adc_es = adc_es
        return e_exc, v_exc, spec_fac, X

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
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_cvs_adc(self, nroots=1, guess=None, eris=None):
        adc_es = UADCIPCVS(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris)
        return e_exc, v_exc, spec_fac, x, adc_es

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


#@profile
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
    
    return M_ij

def get_imds_ip_cvs(adc, eris=None):

    #cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2 = adc.t2

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs

    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)
    
    # The number of ionized alpha and beta electrons is assumed to be the same 
    e_occ_a = adc.mo_energy_a[:ncvs]
    e_occ_b = adc.mo_energy_b[:ncvs]
    e_vir_a = adc.mo_energy_a[nocc_a:]
    e_vir_b = adc.mo_energy_b[nocc_b:]

    idn_occ = np.identity(ncvs)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ceoe = eris.ceoe
    eris_CEOE = eris.CEOE
    eris_ceOE = eris.ceOE
    eris_oeCE = eris.oeCE
    
    # i-j block
    # Zeroth-order terms

    M_ij_a = lib.einsum('ij,j->ij', idn_occ ,e_occ_a)
    M_ij_b = lib.einsum('ij,j->ij', idn_occ ,e_occ_b)

    # Second-order terms

    t2_1_a = t2[0][0][:]
    t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
    M_ij_a += 0.5 * 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_a_coee, eris_ceoe, optimize=True)
    M_ij_a -= 0.5 * 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_a_coee, eris_ceoe, optimize=True)
    M_ij_a += 0.5 * 0.5 *  lib.einsum('jlde,idle->ij',t2_1_a_coee, eris_ceoe, optimize=True)
    M_ij_a -= 0.5 * 0.5 *  lib.einsum('jlde,ield->ij',t2_1_a_coee, eris_ceoe, optimize=True)

    t2_1_b = t2[0][2][:]
    t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
    M_ij_b += 0.5 * 0.5 *  lib.einsum('ilde,jdle->ij',t2_1_b_coee, eris_CEOE, optimize=True)
    M_ij_b -= 0.5 * 0.5 *  lib.einsum('ilde,jeld->ij',t2_1_b_coee, eris_CEOE, optimize=True)
    M_ij_b += 0.5 * 0.5 *  lib.einsum('jlde,idle->ij',t2_1_b_coee, eris_CEOE, optimize=True)
    M_ij_b -= 0.5 * 0.5 *  lib.einsum('jlde,ield->ij',t2_1_b_coee, eris_CEOE, optimize=True)

    t2_1_ab = t2[0][1][:]
    t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
    t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
    M_ij_a += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ab_coee, eris_ceOE, optimize=True)
    M_ij_b += 0.5 * lib.einsum('lied,lejd->ij',t2_1_ab_ocee, eris_oeCE, optimize=True)
    M_ij_a += 0.5 * lib.einsum('jlde,idle->ij',t2_1_ab_coee, eris_ceOE, optimize=True)
    M_ij_b += 0.5 * lib.einsum('ljed,leid->ij',t2_1_ab_ocee, eris_oeCE, optimize=True)

    del t2_1_a
    del t2_1_a_coee
    del t2_1_b
    del t2_1_b_coee
    del t2_1_ab
    del t2_1_ab_coee
    del t2_1_ab_ocee

    # Third-order terms

    if (method == "adc(3)"):

        eris_ccee = eris.ccee
        eris_CCEE = eris.CCEE
        eris_ccEE = eris.ccEE
        eris_CCee = eris.CCee
        eris_oecc = eris.oecc
        eris_OECC = eris.OECC
        eris_oeCC = eris.oeCC
        eris_OEcc = eris.OEcc
        eris_cooo = eris.cooo
        eris_ccoo = eris.ccoo
        eris_COOO = eris.COOO
        eris_CCOO = eris.CCOO
        eris_ccOO = eris.ccOO
        eris_ooCC = eris.ooCC
        eris_coOO = eris.coOO

        eris_cece = eris.cece
        eris_CECE = eris.CECE
        eris_coee = eris.coee
        eris_COEE = eris.COEE
        eris_coEE = eris.coEE
        eris_COee = eris.COee
        eris_ceco = eris.ceco
        eris_CECO = eris.CECO
        eris_coco = eris.coco
        eris_COCO = eris.COCO
        eris_ooCO = eris.ooCO

        t1 = adc.t1
        t1_2_a, t1_2_b = t1[0]

        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_a, eris_oecc, optimize=True)
        M_ij_a -= lib.einsum('ld,jdil->ij',t1_2_a, eris_ceco, optimize=True)
        M_ij_a += lib.einsum('ld,ldji->ij',t1_2_b, eris_OEcc, optimize=True)

        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_b, eris_OECC, optimize=True)
        M_ij_b -= lib.einsum('ld,jdil->ij',t1_2_b, eris_CECO, optimize=True)
        M_ij_b += lib.einsum('ld,ldji->ij',t1_2_a, eris_oeCC, optimize=True)

        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_a, eris_oecc, optimize=True)
        M_ij_a -= lib.einsum('ld,idjl->ij',t1_2_a, eris_ceco, optimize=True)
        M_ij_a += lib.einsum('ld,ldij->ij',t1_2_b, eris_OEcc, optimize=True)

        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_b, eris_OECC, optimize=True)
        M_ij_b -= lib.einsum('ld,idjl->ij',t1_2_b, eris_CECO, optimize=True)
        M_ij_b += lib.einsum('ld,ldij->ij',t1_2_a, eris_oeCC, optimize=True)
       
        del t1_2_a
        del t1_2_b

        t2_1_a = t2[0][0][:]
        t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
        t2_2_a = t2[1][0][:]
        t2_2_a_coee = t2_2_a[:ncvs,:,:,:].copy()

        M_ij_a += 0.5 * 0.5* lib.einsum('ilde,jdle->ij',t2_2_a_coee, eris_ceoe, optimize=True)
        M_ij_a -= 0.5 * 0.5* lib.einsum('ilde,jeld->ij',t2_2_a_coee, eris_ceoe, optimize=True)

        M_ij_a += 0.5 * 0.5* lib.einsum('jlde,idle->ij',t2_2_a_coee, eris_ceoe, optimize=True)
        M_ij_a -= 0.5 * 0.5* lib.einsum('jlde,ield->ij',t2_2_a_coee, eris_ceoe, optimize=True)

        t2_1_b = t2[0][2][:]
        t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
        t2_2_b = t2[1][2][:]
        t2_2_b_coee = t2_2_b[:ncvs,:,:,:].copy()

        M_ij_b += 0.5 * 0.5* lib.einsum('ilde,jdle->ij',t2_2_b_coee, eris_CEOE, optimize=True)
        M_ij_b -= 0.5 * 0.5* lib.einsum('ilde,jeld->ij',t2_2_b_coee, eris_CEOE, optimize=True)
        M_ij_b += 0.5 * 0.5* lib.einsum('jlde,idle->ij',t2_2_b_coee, eris_CEOE, optimize=True)
        M_ij_b -= 0.5 * 0.5* lib.einsum('jlde,ield->ij',t2_2_b_coee, eris_CEOE, optimize=True)

        t2_1_ab = t2[0][1][:]
        t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
        t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
        t2_2_ab = t2[1][1][:]
        t2_2_ab_coee = t2_2_ab[:ncvs,:,:,:].copy()
        t2_2_ab_ocee = t2_2_ab[:,:ncvs,:,:].copy()

        M_ij_a += 0.5 * lib.einsum('ilde,jdle->ij',t2_2_ab_coee, eris_ceOE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('lied,lejd->ij',t2_2_ab_ocee, eris_oeCE, optimize=True)
        M_ij_a += 0.5 * lib.einsum('jlde,idle->ij',t2_2_ab_coee, eris_ceOE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('ljed,leid->ij',t2_2_ab_ocee, eris_oeCE, optimize=True)

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_a, t2_1_a_coee, optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mejf,ifme->ij',M_ij_t, eris_ceoe, optimize=True)
        M_ij_a -= 0.5 * lib.einsum('mejf,ifme->ji',M_ij_t, eris_ceoe, optimize=True)
        M_ij_a += 0.5 * lib.einsum('mejf,imfe->ij',M_ij_t, eris_coee, optimize=True)
        M_ij_a += 0.5 * lib.einsum('mejf,imfe->ji',M_ij_t, eris_coee, optimize=True)
        del M_ij_t

        M_ij_a += 0.5 * 0.25*lib.einsum('lmde,jnde,ilmn->ij',t2_1_a, t2_1_a_coee,eris_cooo, optimize = True)
        M_ij_a -= 0.5 * 0.25*lib.einsum('lmde,jnde,imnl->ij',t2_1_a, t2_1_a_coee,eris_cooo, optimize = True)

        M_ij_a += 0.5 *0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_a_coee, t2_1_a,eris_cooo, optimize = True)
        M_ij_a -= 0.5 *0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_a_coee, t2_1_a,eris_cooo, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_ccee, optimize = True)
        M_ij_a -= 0.5*lib.einsum('lmdf,lmde,jfie->ij',t2_1_a, t2_1_a, eris_cece, optimize = True)
        M_ij_b +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_a, t2_1_a, eris_CCee , optimize = True)

        M_ij_a -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_a, t2_1_a, eris_ccoo, optimize = True)
        M_ij_a += 0.5*lib.einsum('lnde,lmde,jmin->ij',t2_1_a, t2_1_a, eris_coco, optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lnde,lmde,nmji->ij',t2_1_a, t2_1_a, eris_ooCC, optimize = True)

        t2_1_ab = t2[0][1][:]
        t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
        t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
        M_ij_a -= 0.5 * lib.einsum('lmde,jldf,ifme->ij',t2_1_ab, t2_1_a_coee, eris_ceOE,optimize = True)
        M_ij_b += 0.5 * lib.einsum('lmde,ljdf,meif->ij',t2_1_a, t2_1_ab_ocee, eris_oeCE ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('lmde,ildf,jfme->ij',t2_1_ab, t2_1_a_coee, eris_ceOE,optimize = True)
        M_ij_b += 0.5 * lib.einsum('lmde,lidf,mejf->ij',t2_1_a, t2_1_ab_ocee, eris_oeCE ,optimize = True)
        del t2_1_a
        del t2_1_a_coee

        M_ij_a += 0.5 * lib.einsum('lmde,jlfd,ifme->ij',t2_1_b, t2_1_ab_coee, eris_ceOE ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('mled,jldf,meif->ij',t2_1_ab, t2_1_b_coee, eris_oeCE,optimize = True)
        M_ij_a += 0.5 * lib.einsum('lmde,ilfd,jfme->ij',t2_1_b, t2_1_ab_coee, eris_ceOE ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('mled,ildf,mejf->ij',t2_1_ab, t2_1_b_coee, eris_oeCE,optimize = True)

        M_ij_t = lib.einsum('lmde,jldf->mejf', t2_1_b, t2_1_b_coee, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mejf,ifme->ij',M_ij_t, eris_CEOE, optimize=True)
        M_ij_b -= 0.5 * lib.einsum('mejf,ifme->ji',M_ij_t, eris_CEOE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mejf,imfe->ij',M_ij_t, eris_COEE, optimize=True)
        M_ij_b += 0.5 * lib.einsum('mejf,imfe->ji',M_ij_t, eris_COEE, optimize=True)
        del M_ij_t

        M_ij_b += 0.5 * 0.25*lib.einsum('lmde,jnde,ilmn->ij',t2_1_b, t2_1_b_coee,eris_COOO, optimize = True)
        M_ij_b -= 0.5 * 0.25*lib.einsum('lmde,jnde,imnl->ij',t2_1_b, t2_1_b_coee,eris_COOO, optimize = True)

        M_ij_b += 0.5 * 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1_b_coee, t2_1_b,eris_COOO, optimize = True)
        M_ij_b -= 0.5 * 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1_b_coee, t2_1_b,eris_COOO, optimize = True)

        M_ij_a += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_ccEE , optimize = True)
        M_ij_b += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1_b, t2_1_b, eris_CCEE , optimize = True)
        M_ij_b -= 0.5*lib.einsum('lmdf,lmde,jfie->ij',t2_1_b, t2_1_b, eris_CECE , optimize = True)

        M_ij_a -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_ccOO, optimize = True)
        M_ij_b -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1_b, t2_1_b, eris_CCOO, optimize = True)
        M_ij_b += 0.5*lib.einsum('lnde,lmde,jmin->ij',t2_1_b, t2_1_b, eris_COCO, optimize = True)
        del t2_1_b
        del t2_1_b_coee

        M_ij_a += 0.5 * lib.einsum('mled,jlfd,ifme->ij',t2_1_ab, t2_1_ab_coee, eris_ceoe ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mled,jlfd,imfe->ij',t2_1_ab, t2_1_ab_coee, eris_coee ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mlde,jldf,imfe->ij',t2_1_ab, t2_1_ab_coee, eris_coEE ,optimize = True)

        M_ij_b += 0.5 * lib.einsum('lmde,ljdf,ifme->ij',t2_1_ab, t2_1_ab_ocee, eris_CEOE,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmde,ljdf,imfe->ij',t2_1_ab, t2_1_ab_ocee, eris_COEE,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmed,ljfd,imfe->ij',t2_1_ab, t2_1_ab_ocee, eris_COee ,optimize = True)

        M_ij_a += 0.5 * lib.einsum('mled,ilfd,jfme->ij',t2_1_ab, t2_1_ab_coee, eris_ceoe ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mled,ilfd,jmfe->ij',t2_1_ab, t2_1_ab_coee, eris_coee ,optimize = True)
        M_ij_a -= 0.5 * lib.einsum('mlde,ildf,jmfe->ij',t2_1_ab, t2_1_ab_coee, eris_coEE ,optimize = True)

        M_ij_b += 0.5 * lib.einsum('lmde,lidf,jfme->ij',t2_1_ab, t2_1_ab_ocee, eris_CEOE ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmde,lidf,jmfe->ij',t2_1_ab, t2_1_ab_ocee, eris_COEE ,optimize = True)
        M_ij_b -= 0.5 * lib.einsum('lmed,lifd,jmfe->ij',t2_1_ab, t2_1_ab_ocee, eris_COee ,optimize = True)

        M_ij_a += 0.5 * lib.einsum('lmde,jnde,ilmn->ij',t2_1_ab ,t2_1_ab_coee,eris_coOO, optimize = True)
        M_ij_b += 0.5 * lib.einsum('mled,njed,mnil->ij',t2_1_ab ,t2_1_ab_ocee,eris_ooCO, optimize = True)

        M_ij_a +=0.5 * lib.einsum('inde,lmde,jlnm->ij',t2_1_ab_coee, t2_1_ab,eris_coOO, optimize = True)

        M_ij_b +=0.5 * lib.einsum('nied,mled,nmjl->ij',t2_1_ab_ocee, t2_1_ab,eris_ooCO, optimize = True)

        M_ij_a +=lib.einsum('mlfd,mled,jief->ij',t2_1_ab, t2_1_ab, eris_ccee , optimize = True)
        M_ij_a -=lib.einsum('mlfd,mled,jfie->ij',t2_1_ab, t2_1_ab, eris_cece , optimize = True)
        M_ij_a +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_ccEE , optimize = True)
        M_ij_b +=lib.einsum('lmdf,lmde,jief->ij',t2_1_ab, t2_1_ab, eris_CCEE , optimize = True)
        M_ij_b -=lib.einsum('lmdf,lmde,jfie->ij',t2_1_ab, t2_1_ab, eris_CECE , optimize = True)
        M_ij_b +=lib.einsum('lmfd,lmed,jief->ij',t2_1_ab, t2_1_ab, eris_CCee , optimize = True)

        M_ij_a -= lib.einsum('nled,mled,jinm->ij',t2_1_ab, t2_1_ab, eris_ccoo, optimize = True)
        M_ij_a += lib.einsum('nled,mled,jmin->ij',t2_1_ab, t2_1_ab, eris_coco, optimize = True)
        M_ij_a -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_ccOO, optimize = True)
        M_ij_b -= lib.einsum('lnde,lmde,jinm->ij',t2_1_ab, t2_1_ab, eris_CCOO, optimize = True)
        M_ij_b += lib.einsum('lnde,lmde,jmin->ij',t2_1_ab, t2_1_ab, eris_COCO, optimize = True)
        M_ij_b -= lib.einsum('nled,mled,nmji->ij',t2_1_ab, t2_1_ab, eris_ooCC, optimize = True)

        del t2_1_ab
        del t2_1_ab_coee
        del t2_1_ab_ocee

    M_ij = (M_ij_a, M_ij_b)
    #cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

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
        temp[:,:ncore,:ncore] += shift

        diag[s_aaa:f_aaa] = temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1).copy()

        temp = diag[s_bab:f_bab].copy()
        temp = temp.reshape((nvir_b, nocc_b, nocc_a))
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        temp[:,:ncore,:ncore] += shift

        diag[s_bab:f_bab] = temp.reshape(-1).copy()

        temp = diag[s_aba:f_aba].copy()
        temp = temp.reshape((nvir_a, nocc_a, nocc_b))
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        temp[:,:ncore,:ncore] += shift

        diag[s_aba:f_aba] = temp.reshape(-1).copy()

        temp = np.zeros((nvir_b, nocc_b, nocc_b))
        temp[:,ij_ind_b[0],ij_ind_b[1]] = diag[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
        temp[:,ij_ind_b[1],ij_ind_b[0]] = -diag[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
        
        temp[:,ncore:,ncore:] += shift
        #temp[:,:ncore,ncore:] += shift
        #temp[:,ncore:,:ncore] += shift
        temp[:,:ncore,:ncore] = shift
        
        diag[s_bbb:f_bbb] = temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1).copy()
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
    D_n_aab = D_n_aab.reshape((nvir_a,nocc_a,nocc_b))
    D_aij_aab_ecc = D_n_aab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aab_ecv = D_n_aab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aab_evc = D_n_aab[:, ncvs:, :ncvs].reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    diag[s_a:f_a] = M_ij_a_diag.copy()
    diag[s_b:f_b] = M_ij_b_diag.copy()

    # Compute precond in 2p1h-2p1h block
    
    diag[s_aaa_ecc:f_aaa_ecc] = D_aij_a_ecc.copy()
    diag[s_aaa_ecv:f_aaa_ecv] = D_aij_a_ecv.copy()
    diag[s_bba_ecc:f_bba_ecc] = D_aij_bba_ecc.copy()
    diag[s_bba_ecv:f_bba_ecv] = D_aij_bba_ecv.copy()
    diag[s_bba_evc:f_bba_evc] = D_aij_bba_evc.copy()
    diag[s_aab_ecc:f_aab_ecc] = D_aij_aab_ecc.copy()
    diag[s_aab_ecv:f_aab_ecv] = D_aij_aab_ecv.copy()
    diag[s_aab_evc:f_aab_evc] = D_aij_aab_evc.copy()
    diag[s_bbb_ecc:f_bbb_ecc] = D_aij_b_ecc.copy()
    diag[s_bbb_ecv:f_bbb_ecv] = D_aij_b_ecv.copy()

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
                
               #s[s_bab:f_bab] -= 0.5*lib.einsum('kijl,ali->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= lib.einsum('klji,ail->ajk',eris_ooOO,r_bab,optimize = True).reshape(-1)
                
               #s[s_aba:f_aba] -= 0.5*lib.einsum('jlki,ali->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= lib.einsum('jikl,ail->ajk',eris_ooOO,r_aba,optimize = True).reshape(-1)

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

        return s

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
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_aaa:f_aaa] = temp[:,ij_a[0],ij_a[1]].reshape(-1).copy()

    temp = Pr[s_bab:f_bab].copy()
    temp = temp.reshape((nvir_b, nocc_b, nocc_a))
    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_bab:f_bab] = temp.reshape(-1).copy()

    temp = Pr[s_aba:f_aba].copy()
    temp = temp.reshape((nvir_a, nocc_a, nocc_b))
    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_aba:f_aba] = temp.reshape(-1).copy()

    temp = np.zeros((nvir_b, nocc_b, nocc_b))
    temp[:,ij_b[0],ij_b[1]] = Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
    temp[:,ij_b[1],ij_b[0]] = -Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    #temp[:,:ncore,ncore:] = 0.0
    #temp[:,ncore:,:ncore] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_bbb:f_bbb] = temp[:,ij_b[0],ij_b[1]].reshape(-1).copy()

    return Pr
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
    D_n_aab = D_n_aab.reshape((nvir_a,nocc_a,nocc_b))
    D_aij_aab_ecc = D_n_aab[:, :ncvs, :ncvs].reshape(-1)
    D_aij_aab_ecv = D_n_aab[:, :ncvs, ncvs:].reshape(-1)
    D_aij_aab_evc = D_n_aab[:, ncvs:, :ncvs].reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds()
    M_ij_a, M_ij_b = M_ij
    
    #Calculate sigma vector
    def sigma_(r):
 
        cput0 = (time.clock(), time.time())
        cput0i = np.array([time.clock(), time.time()])
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
        r_aaa_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_aaa_ecc
        r_aaa_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_aaa_ecc

        r_bbb_ecc_u = None
        r_bbb_ecc_u = np.zeros((nvir_b,ncvs,ncvs))
        r_bbb_ecc_u[:,ij_ind_ncvs[0],ij_ind_ncvs[1]]= r_bbb_ecc
        r_bbb_ecc_u[:,ij_ind_ncvs[1],ij_ind_ncvs[0]]= -r_bbb_ecc
        

        r_aaa_ecv = r_aaa_ecv.reshape(nvir_a,ncvs,nval_a) 
        r_bba_ecc = r_bba_ecc.reshape(nvir_b,ncvs,ncvs)  
        r_bba_ecv = r_bba_ecv.reshape(nvir_b,ncvs,nval_a)
        r_bba_evc = r_bba_evc.reshape(nvir_b,nval_b,ncvs)
        r_aab_ecc = r_aab_ecc.reshape(nvir_a,ncvs,ncvs)  
        r_aab_ecv = r_aab_ecv.reshape(nvir_a,ncvs,nval_b)
        r_aab_evc = r_aab_evc.reshape(nvir_a,nval_a,ncvs)
        r_bbb_ecv = r_bbb_ecv.reshape(nvir_b,ncvs,nval_b)

        #eris_cecc = eris.cecc     eris_cecc = eris.ovoo[:ncvs,:,:ncvs,:ncvs] 
        #eris_vecc = eris.vecc     eris_vecc = eris.ovoo[ncvs:,:,:ncvs,:ncvs]
        #eris_CECC = eris.CECC     eris_CECC = eris.OVOO[:ncvs,:,:ncvs,:ncvs]
        #eris_VECC = eris.VECC     eris_VECC = eris.OVOO[ncvs:,:,:ncvs,:ncvs]
        #eris_ceCC = eris.ceCC     eris_ceCC = eris.ovOO[:ncvs,:,:ncvs,:ncvs]
        #eris_veCC = eris.veCC     eris_veCC = eris.ovOO[ncvs:,:,:ncvs,:ncvs]
        #eris_CEcc = eris.CEcc     eris_CEcc = eris.OVoo[:ncvs,:,:ncvs,:ncvs]
        #eris_VEcc = eris.VEcc     eris_VEcc = eris.OVoo[ncvs:,:,:ncvs,:ncvs]

        #eris_cecv = eris.cecv     eris_cecv = eris.ovoo[:ncvs,:,:ncvs,ncvs:]
        #eris_CECV = eris.CECV     eris_CECV = eris.OVOO[:ncvs,:,:ncvs,ncvs:]
        #eris_ceCV = eris.ceCV     eris_ceCV = eris.ovOO[:ncvs,:,:ncvs,ncvs:]
        #eris_CEcv = eris.CEcv     eris_CEcv = eris.OVoo[:ncvs,:,:ncvs,ncvs:]
       
        eris_ovoo = eris.ovoo
        eris_OVOO = eris.OVOO
        eris_OVoo = eris.OVoo
        eris_ovOO = eris.ovOO

        eris_cecc = eris.ovoo[:ncvs,:,:ncvs,:ncvs].copy() 
        eris_vecc = eris.ovoo[ncvs:,:,:ncvs,:ncvs].copy()
        eris_CECC = eris.OVOO[:ncvs,:,:ncvs,:ncvs].copy()
        eris_VECC = eris.OVOO[ncvs:,:,:ncvs,:ncvs].copy()
        eris_ceCC = eris.ovOO[:ncvs,:,:ncvs,:ncvs].copy()
        eris_veCC = eris.ovOO[ncvs:,:,:ncvs,:ncvs].copy()
        eris_CEcc = eris.OVoo[:ncvs,:,:ncvs,:ncvs].copy()
        eris_VEcc = eris.OVoo[ncvs:,:,:ncvs,:ncvs].copy()
                                                  
        eris_cecv = eris.ovoo[:ncvs,:,:ncvs,ncvs:].copy()
        eris_CECV = eris.OVOO[:ncvs,:,:ncvs,ncvs:].copy()
        eris_ceCV = eris.ovOO[:ncvs,:,:ncvs,ncvs:].copy()
        eris_CEcv = eris.OVoo[:ncvs,:,:ncvs,ncvs:].copy()

############ ADC(2) ij block ############################
         
        s[s_a:f_a] = lib.einsum('ij,j->i',M_ij_a,r_a)
        s[s_b:f_b] = lib.einsum('ij,j->i',M_ij_b,r_b)

############# ADC(2) i - kja block #########################
         
        s[s_a:f_a] += 0.5*lib.einsum('JaKI,aJK->I', eris_cecc, r_aaa_ecc_u, optimize = True)
        s[s_a:f_a] -= 0.5*lib.einsum('KaJI,aJK->I', eris_cecc, r_aaa_ecc_u, optimize = True)
        s[s_a:f_a] += lib.einsum('JaIk,aJk->I', eris_cecv, r_aaa_ecv, optimize = True)
        s[s_a:f_a] -= lib.einsum('kaJI,aJk->I', eris_vecc, r_aaa_ecv, optimize = True)
        s[s_a:f_a] += lib.einsum('JaKI,aJK->I', eris_CEcc, r_bba_ecc, optimize = True)
        s[s_a:f_a] += lib.einsum('JaIk,aJk->I', eris_CEcv, r_bba_ecv, optimize = True)
        s[s_a:f_a] += lib.einsum('jaKI,ajK->I', eris_VEcc, r_bba_evc, optimize = True)

        s[s_b:f_b] += 0.5*lib.einsum('JaKI,aJK->I', eris_CECC, r_bbb_ecc_u, optimize = True)
        s[s_b:f_b] -= 0.5*lib.einsum('KaJI,aJK->I', eris_CECC, r_bbb_ecc_u, optimize = True)
        s[s_b:f_b] += lib.einsum('JaIk,aJk->I', eris_CECV, r_bbb_ecv, optimize = True)
        s[s_b:f_b] -= lib.einsum('kaJI,aJk->I', eris_VECC, r_bbb_ecv, optimize = True)
        s[s_b:f_b] += lib.einsum('JaKI,aJK->I', eris_ceCC, r_aab_ecc, optimize = True)
        s[s_b:f_b] += lib.einsum('JaIk,aJk->I', eris_ceCV, r_aab_ecv, optimize = True)
        s[s_b:f_b] += lib.einsum('jaKI,ajK->I', eris_veCC, r_aab_evc, optimize = True)
        
############## ADC(2) ajk - i block ############################
         
        temp_aaa_ecc = lib.einsum('JaKI,I->aJK', eris_cecc, r_a, optimize = True)
        temp_aaa_ecc -= lib.einsum('KaJI,I->aJK', eris_cecc, r_a, optimize = True)
        s[s_aaa_ecc:f_aaa_ecc] += temp_aaa_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] += lib.einsum('JaIk,I->aJk', eris_cecv, r_a, optimize = True).reshape(-1)
        s[s_aaa_ecv:f_aaa_ecv] -= lib.einsum('kaJI,I->aJk', eris_vecc, r_a, optimize = True).reshape(-1)
        s[s_bba_ecc:f_bba_ecc] += lib.einsum('JaIK,I->aJK', eris_CEcc, r_a, optimize = True).reshape(-1)
        s[s_bba_ecv:f_bba_ecv] += lib.einsum('JaIk,I->aJk', eris_CEcv, r_a, optimize = True).reshape(-1)
        s[s_bba_evc:f_bba_evc] += lib.einsum('jaIK,I->ajK', eris_VEcc, r_a, optimize = True).reshape(-1)
        s[s_aab_ecc:f_aab_ecc] += lib.einsum('JaKI,I->aJK', eris_ceCC, r_b, optimize = True).reshape(-1)
        s[s_aab_ecv:f_aab_ecv] += lib.einsum('JaIk,I->aJk', eris_ceCV, r_b, optimize = True).reshape(-1)
        s[s_aab_evc:f_aab_evc] += lib.einsum('jaKI,I->ajK', eris_veCC, r_b, optimize = True).reshape(-1)
        temp_bbb_ecc = lib.einsum('JaKI,I->aJK', eris_CECC, r_b, optimize = True)
        temp_bbb_ecc -= lib.einsum('KaJI,I->aJK', eris_CECC, r_b, optimize = True)
        s[s_bbb_ecc:f_bbb_ecc] += temp_bbb_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] += lib.einsum('JaIk,I->aJk', eris_CECV, r_b, optimize = True).reshape(-1)
        s[s_bbb_ecv:f_bbb_ecv] -= lib.einsum('kaJI,I->aJk', eris_VECC, r_b, optimize = True).reshape(-1)
        
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

               #eris_cccc = eris.cccc
               #eris_cccv = eris.cccv
               #eris_ccvv = eris.ccvv
               #eris_CCCC = eris.CCCC
               #eris_CCCV = eris.CCCV
               #eris_CCVV = eris.CCVV
               #eris_ccCC = eris.ccCC
               #eris_ccCV = eris.ccCV
               #eris_vvCC = eris.vvCC
               #eris_ccVV = eris.ccVV
               #eris_ccee = eris.ccee
               #eris_vvee = eris.vvee
               #eris_CCEE = eris.CCEE
               #eris_VVEE = eris.VVEE
               #eris_ccEE = eris.ccEE
               #eris_vvEE = eris.vvEE
               #eris_CCee = eris.CCee
               #eris_VVee = eris.VVee

               #eris_cvcv = eris.cvcv  
               #eris_CVCV = eris.CVCV 
               #eris_cvCC = eris.cvCC 
               #eris_cvCV = eris.cvCV 
               #eris_cece = eris.cece 
               #eris_vece = eris.vece 
               #eris_veve = eris.veve 
               #eris_CECE = eris.CECE 
               #eris_VECE = eris.VECE 
               #eris_VEVE = eris.VEVE 
               #eris_CEce = eris.CEce 
               #eris_VEce = eris.VEce 
               #eris_CEve = eris.CEve 
               #eris_VEve = eris.VEve 
               #eris_cvee = eris.cvee 
               #eris_CVEE = eris.CVEE 
               #eris_cvEE = eris.cvEE 
               #eris_CVee = eris.CVee 
               
               eris_ovov = eris.ovvo.transpose(0,1,3,2)
               eris_OVOV = eris.OVVO.transpose(0,1,3,2)
               eris_OVov = eris.OVvo.transpose(0,1,3,2)

               eris_cccc = eris.oooo[:ncvs,:ncvs,:ncvs,:ncvs] 
               eris_cccv = eris.oooo[:ncvs,:ncvs,:ncvs,ncvs:]
               eris_ccvv = eris.oooo[:ncvs,:ncvs,ncvs:,ncvs:]
               eris_CCCC = eris.OOOO[:ncvs,:ncvs,:ncvs,:ncvs]
               eris_CCCV = eris.OOOO[:ncvs,:ncvs,:ncvs,ncvs:]
               eris_CCVV = eris.OOOO[:ncvs,:ncvs,ncvs:,ncvs:]
               eris_ccCC = eris.ooOO[:ncvs,:ncvs,:ncvs,:ncvs]
               eris_ccCV = eris.ooOO[:ncvs,:ncvs,:ncvs,ncvs:]
               eris_vvCC = eris.ooOO[ncvs:,ncvs:,:ncvs,:ncvs]
               eris_ccVV = eris.ooOO[:ncvs,:ncvs,ncvs:,ncvs:]
               eris_ccee = eris.oovv[:ncvs,:ncvs,:,:]
               eris_vvee = eris.oovv[ncvs:,ncvs:,:,:]
               eris_CCEE = eris.OOVV[:ncvs,:ncvs,:,:]
               eris_VVEE = eris.OOVV[ncvs:,ncvs:,:,:]
               eris_ccEE = eris.ooVV[:ncvs,:ncvs,:,:]
               eris_vvEE = eris.ooVV[ncvs:,ncvs:,:,:]
               eris_CCee = eris.OOvv[:ncvs,:ncvs,:,:]
               eris_VVee = eris.OOvv[ncvs:,ncvs:,:,:]

               eris_cvcv = eris.oooo[:ncvs,ncvs:,:ncvs,ncvs:].copy()   
               eris_CVCV = eris.OOOO[:ncvs,ncvs:,:ncvs,ncvs:].copy()  
               eris_cvCC = eris.ooOO[:ncvs,ncvs:,:ncvs,:ncvs].copy()  
               eris_cvCV = eris.ooOO[:ncvs,ncvs:,:ncvs,ncvs:].copy()  
               eris_cece = eris_ovov[:ncvs,:,:ncvs,:].copy()  
               eris_vece = eris_ovov[ncvs:,:,:ncvs,:].copy()  
               eris_veve = eris_ovov[ncvs:,:,ncvs:,:].copy()  
               eris_CECE = eris_OVOV[:ncvs,:,:ncvs,:].copy()  
               eris_VECE = eris_OVOV[ncvs:,:,:ncvs,:].copy()  
               eris_VEVE = eris_OVOV[ncvs:,:,ncvs:,:].copy()  
               eris_CEce = eris_OVov[:ncvs,:,:ncvs,:].copy()  
               eris_VEce = eris_OVov[ncvs:,:,:ncvs,:].copy()  
               eris_CEve = eris_OVov[:ncvs,:,ncvs:,:].copy()  
               eris_VEve = eris_OVov[ncvs:,:,ncvs:,:].copy()  
               eris_cvee = eris.oovv[:ncvs,ncvs:,:,:].copy()  
               eris_CVEE = eris.OOVV[:ncvs,ncvs:,:,:].copy()  
               eris_cvEE = eris.ooVV[:ncvs,ncvs:,:,:].copy()  
               eris_CVee = eris.OOvv[:ncvs,ncvs:,:,:].copy() 
 
               temp_ecc =  0.5*lib.einsum('JLKI,aIL->aJK',eris_cccc,r_aaa_ecc_u ,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_cccc,r_aaa_ecc_u ,optimize = True)
               temp_ecc +=     lib.einsum('KIJl,aIl->aJK',eris_cccv,r_aaa_ecv ,optimize = True)
               temp_ecc -=     lib.einsum('JIKl,aIl->aJK',eris_cccv,r_aaa_ecv ,optimize = True)
               temp_ecv =  0.5*lib.einsum('JLIk,aIL->aJk',eris_cccv,r_aaa_ecc_u ,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JILk,aIL->aJk',eris_cccv,r_aaa_ecc_u ,optimize = True)
               temp_ecv +=     lib.einsum('JlIk,aIl->aJk',eris_cvcv,r_aaa_ecv ,optimize = True)
               temp_ecv -=     lib.einsum('JIkl,aIl->aJk',eris_ccvv,r_aaa_ecv ,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

               temp_ecc =  0.5*lib.einsum('JLKI,aIL->aJK',eris_CCCC,r_bbb_ecc_u ,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JIKL,aIL->aJK',eris_CCCC,r_bbb_ecc_u ,optimize = True)
               temp_ecc +=     lib.einsum('KIJl,aIl->aJK',eris_CCCV,r_bbb_ecv ,optimize = True)
               temp_ecc -=     lib.einsum('JIKl,aIl->aJK',eris_CCCV,r_bbb_ecv ,optimize = True)
               temp_ecv =  0.5*lib.einsum('JLIk,aIL->aJk',eris_CCCV,r_bbb_ecc_u ,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JILk,aIL->aJk',eris_CCCV,r_bbb_ecc_u ,optimize = True)
               temp_ecv +=     lib.einsum('JlIk,aIl->aJk',eris_CVCV,r_bbb_ecv ,optimize = True)
               temp_ecv -=     lib.einsum('JIkl,aIl->aJk',eris_CCVV,r_bbb_ecv ,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

               s[s_bba_ecc:f_bba_ecc] -= lib.einsum('KLJI,aIL->aJK',eris_ccCC,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= lib.einsum('KIJl,alI->aJK',eris_ccCV,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= lib.einsum('KlJI,aIl->aJK',eris_cvCC,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= lib.einsum('LkJI,aIL->aJk',eris_cvCC,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= lib.einsum('IkJl,alI->aJk',eris_cvCV,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= lib.einsum('klJI,aIl->aJk',eris_vvCC,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= lib.einsum('KLIj,aIL->ajK',eris_ccCV,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= lib.einsum('KIjl,alI->ajK',eris_ccVV,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= lib.einsum('KlIj,aIl->ajK',eris_cvCV,r_bba_ecv,optimize = True).reshape(-1)
                
               s[s_aab_ecc:f_aab_ecc] -= lib.einsum('JIKL,aIL->aJK',eris_ccCC,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= lib.einsum('JlKI,alI->aJK',eris_cvCC,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= lib.einsum('JIKl,aIl->aJK',eris_ccCV,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= lib.einsum('JILk,aIL->aJk',eris_ccCV,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= lib.einsum('JlIk,alI->aJk',eris_cvCV,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= lib.einsum('JIkl,aIl->aJk',eris_ccVV,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= lib.einsum('IjKL,aIL->ajK',eris_cvCC,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= lib.einsum('jlKI,alI->ajK',eris_vvCC,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= lib.einsum('IjKl,aIl->ajK',eris_cvCV,r_aab_ecv,optimize = True).reshape(-1)
               
               temp_ecc =  0.5*lib.einsum('KLba,bJL->aJK',eris_ccee,r_aaa_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('KaLb,bJL->aJK',eris_cece,r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('LbKa,bLJ->aJK',eris_CEce,r_bba_ecc,optimize = True)
               temp_ecc += 0.5*lib.einsum('Klba,bJl->aJK',eris_cvee,r_aaa_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('lbKa,bJl->aJK',eris_vece,r_aaa_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('lbKa,blJ->aJK',eris_VEce,r_bba_evc,optimize = True)
               temp_ecv =  0.5*lib.einsum('Lkba,bJL->aJk',eris_cvee,r_aaa_ecc_u,optimize = True)
               temp_ecv -= 0.5*lib.einsum('kaLb,bJL->aJk',eris_vece,r_aaa_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('Lbka,bLJ->aJk',eris_CEve,r_bba_ecc,optimize = True)
               temp_ecv += 0.5*lib.einsum('klba,bJl->aJk',eris_vvee,r_aaa_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('kalb,bJl->aJk',eris_veve,r_aaa_ecv,optimize = True)
               temp_ecv += 0.5*lib.einsum('lbka,blJ->aJk',eris_VEve,r_bba_evc,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
              
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('KLba,bJL->aJK',eris_ccEE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('Klba,bJl->aJK',eris_cvEE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('Lkba,bJL->aJk',eris_cvEE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('klba,bJl->aJk',eris_vvEE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('KLba,bjL->ajK',eris_ccEE,r_bba_evc,optimize = True).reshape(-1)
               
               temp_1_ecc =  0.5*lib.einsum('KLba,bJL->aJK',eris_CCEE,r_bbb_ecc_u,optimize = True)
               temp_1_ecc -= 0.5*lib.einsum('KaLb,bJL->aJK',eris_CECE,r_bbb_ecc_u,optimize = True)
               temp_1_ecc += 0.5*lib.einsum('KaLb,bLJ->aJK',eris_CEce,r_aab_ecc,optimize = True)
               temp_1_ecc += 0.5*lib.einsum('Klba,bJl->aJK',eris_CVEE,r_bbb_ecv,optimize = True)
               temp_1_ecc -= 0.5*lib.einsum('lbKa,bJl->aJK',eris_VECE,r_bbb_ecv,optimize = True)
               temp_1_ecc += 0.5*lib.einsum('Kalb,blJ->aJK',eris_CEve,r_aab_evc,optimize = True)
               temp_1_ecv =  0.5*lib.einsum('Lkba,bJL->aJk',eris_CVEE,r_bbb_ecc_u,optimize = True)
               temp_1_ecv -= 0.5*lib.einsum('kaLb,bJL->aJk',eris_VECE,r_bbb_ecc_u,optimize = True)
               temp_1_ecv += 0.5*lib.einsum('kaLb,bLJ->aJk',eris_VEce,r_aab_ecc,optimize = True)
               temp_1_ecv += 0.5*lib.einsum('klba,bJl->aJk',eris_VVEE,r_bbb_ecv,optimize = True)
               temp_1_ecv -= 0.5*lib.einsum('kalb,bJl->aJk',eris_VEVE,r_bbb_ecv,optimize = True)
               temp_1_ecv += 0.5*lib.einsum('kalb,blJ->aJk',eris_VEve,r_aab_evc,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_1_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_1_ecv.reshape(-1)
               
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('KLba,bJL->aJK',eris_CCee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Klba,bJl->aJK',eris_CVee,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('Lkba,bJL->aJk',eris_CVee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('klba,bJl->aJk',eris_VVee,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('KLba,bjL->ajK',eris_CCee,r_aab_evc,optimize = True).reshape(-1)

               temp_ecc =  -0.5*lib.einsum('JLba,bKL->aJK',eris_ccee,r_aaa_ecc_u,optimize = True)
               temp_ecc +=  0.5*lib.einsum('JaLb,bKL->aJK',eris_cece,r_aaa_ecc_u,optimize = True)
               temp_ecc -=  0.5*lib.einsum('LbJa,bLK->aJK',eris_CEce,r_bba_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jlba,bKl->aJK',eris_cvee,r_aaa_ecv,optimize = True)
               temp_ecc +=  0.5*lib.einsum('lbJa,bKl->aJK',eris_vece,r_aaa_ecv,optimize = True)
               temp_ecc -=  0.5*lib.einsum('lbJa,blK->aJK',eris_VEce,r_bba_evc,optimize = True)
               temp_ecv =   0.5*lib.einsum('JLba,bLk->aJk',eris_ccee,r_aaa_ecv,optimize = True)
               temp_ecv -=  0.5*lib.einsum('JaLb,bLk->aJk',eris_cece,r_aaa_ecv,optimize = True)
               temp_ecv -=  0.5*lib.einsum('LbJa,bLk->aJk',eris_CEce,r_bba_ecv,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('JaLb,bKL->aJK',eris_CEce,r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('JLba,bLK->aJK',eris_CCEE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -=  0.5*lib.einsum('JaLb,bLK->aJK',eris_CECE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('Jalb,bKl->aJK',eris_CEve,r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] +=  0.5*lib.einsum('Jlba,blK->aJK',eris_CVEE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -=  0.5*lib.einsum('lbJa,blK->aJK',eris_VECE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += -0.5*lib.einsum('JaLb,bLk->aJk',eris_CEce,r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] +=  0.5*lib.einsum('JLba,bLk->aJk',eris_CCEE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -=  0.5*lib.einsum('JaLb,bLk->aJk',eris_CECE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jaLb,bKL->ajK',eris_VEce,r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('Ljba,bLK->ajK',eris_CVEE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -=  0.5*lib.einsum('jaLb,bLK->ajK',eris_VECE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jalb,bKl->ajK',eris_VEve,r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] +=  0.5*lib.einsum('jlba,blK->ajK',eris_VVEE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -=  0.5*lib.einsum('jalb,blK->ajK',eris_VEVE,r_bba_evc,optimize = True).reshape(-1)

               temp_ecc = -0.5*lib.einsum('JLba,bKL->aJK',eris_CCEE,r_bbb_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('JaLb,bKL->aJK',eris_CECE,r_bbb_ecc_u,optimize = True)
               temp_ecc -= 0.5*lib.einsum('JaLb,bLK->aJK',eris_CEce,r_aab_ecc,optimize = True)
               temp_ecc +=-0.5*lib.einsum('Jlba,bKl->aJK',eris_CVEE,r_bbb_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('lbJa,bKl->aJK',eris_VECE,r_bbb_ecv,optimize = True)
               temp_ecc -= 0.5*lib.einsum('Jalb,blK->aJK',eris_CEve,r_aab_evc,optimize = True)
               temp_ecv =  0.5*lib.einsum('JLba,bLk->aJk',eris_CCEE,r_bbb_ecv,optimize = True)
               temp_ecv +=-0.5*lib.einsum('JaLb,bLk->aJk',eris_CECE,r_bbb_ecv,optimize = True)
               temp_ecv -= 0.5*lib.einsum('JaLb,bLk->aJk',eris_CEce,r_aab_ecv,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)
               
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('JLba,bLK->aJK',eris_ccee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JaLb,bLK->aJK',eris_cece,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('LbJa,bKL->aJK',eris_CEce,r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Jlba,blK->aJK',eris_cvee,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('lbJa,blK->aJK',eris_vece,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('lbJa,bKl->aJK',eris_VEce,r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('JLba,bLk->aJk',eris_ccee,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JaLb,bLk->aJk',eris_cece,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] +=-0.5*lib.einsum('LbJa,bLk->aJk',eris_CEce,r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('Ljba,bLK->ajK',eris_cvee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jaLb,bLK->ajK',eris_vece,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('Lbja,bKL->ajK',eris_CEve,r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jlba,blK->ajK',eris_vvee,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jalb,blK->ajK',eris_veve,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('lbja,bKl->ajK',eris_VEve,r_bbb_ecv,optimize = True).reshape(-1)
                
               temp_ecc = -0.5*lib.einsum('KIba,bIJ->aJK',eris_ccee,r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KaIb,bIJ->aJK',eris_cece,r_aaa_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('IbKa,bIJ->aJK',eris_CEce,r_bba_ecc,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kiba,bJi->aJK',eris_cvee,r_aaa_ecv,optimize = True)
               temp_ecc +=-0.5*lib.einsum('ibKa,bJi->aJK',eris_vece,r_aaa_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('ibKa,biJ->aJK',eris_VEce,r_bba_evc,optimize = True)
               temp_ecv = -0.5*lib.einsum('Ikba,bIJ->aJk',eris_cvee,r_aaa_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kaIb,bIJ->aJk',eris_vece,r_aaa_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('Ibka,bIJ->aJk',eris_CEve,r_bba_ecc,optimize = True)
               temp_ecv += 0.5*lib.einsum('kiba,bJi->aJk',eris_vvee,r_aaa_ecv,optimize = True)
               temp_ecv +=-0.5*lib.einsum('kaib,bJi->aJk',eris_veve,r_aaa_ecv,optimize = True)
               temp_ecv += 0.5*lib.einsum('ibka,biJ->aJk',eris_VEve,r_bba_evc,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
               
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('KIba,bJI->aJK',eris_ccEE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('Kiba,bJi->aJK',eris_cvEE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('Ikba,bJI->aJk',eris_cvEE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('kiba,bJi->aJk',eris_vvEE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('KIba,bjI->ajK',eris_ccEE,r_bba_evc,optimize = True).reshape(-1)
               
               temp_ecc = -0.5*lib.einsum('KIba,bIJ->aJK',eris_CCEE,r_bbb_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KaIb,bIJ->aJK',eris_CECE,r_bbb_ecc_u,optimize = True)
               temp_ecc += 0.5*lib.einsum('KaIb,bIJ->aJK',eris_CEce,r_aab_ecc,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kiba,bJi->aJK',eris_CVEE,r_bbb_ecv,optimize = True)
               temp_ecc +=-0.5*lib.einsum('ibKa,bJi->aJK',eris_VECE,r_bbb_ecv,optimize = True)
               temp_ecc += 0.5*lib.einsum('Kaib,biJ->aJK',eris_CEve,r_aab_evc,optimize = True)
               temp_ecv = -0.5*lib.einsum('Ikba,bIJ->aJk',eris_CVEE,r_bbb_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kaIb,bIJ->aJk',eris_VECE,r_bbb_ecc_u,optimize = True)
               temp_ecv += 0.5*lib.einsum('kaIb,bIJ->aJk',eris_VEce,r_aab_ecc,optimize = True)
               temp_ecv += 0.5*lib.einsum('kiba,bJi->aJk',eris_VVEE,r_bbb_ecv,optimize = True)
               temp_ecv +=-0.5*lib.einsum('kaib,bJi->aJk',eris_VEVE,r_bbb_ecv,optimize = True)
               temp_ecv += 0.5*lib.einsum('kaib,biJ->aJk',eris_VEve,r_aab_evc,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)
               
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('KIba,bJI->aJK',eris_CCee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Kiba,bJi->aJK',eris_CVee,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('Ikba,bJI->aJk',eris_CVee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('kiba,bJi->aJk',eris_VVee,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('KIba,bjI->ajK',eris_CCee,r_aab_evc,optimize = True).reshape(-1)

               temp_ecc =   0.5*lib.einsum('JIba,bIK->aJK',eris_ccee,r_aaa_ecc_u,optimize = True)
               temp_ecc -=  0.5*lib.einsum('JaIb,bIK->aJK',eris_cece,r_aaa_ecc_u,optimize = True)
               temp_ecc -=  0.5*lib.einsum('IbJa,bIK->aJK',eris_CEce,r_bba_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jiba,bKi->aJK',eris_cvee,r_aaa_ecv,optimize = True)
               temp_ecc -= -0.5*lib.einsum('ibJa,bKi->aJK',eris_vece,r_aaa_ecv,optimize = True)
               temp_ecc -=  0.5*lib.einsum('ibJa,biK->aJK',eris_VEce,r_bba_evc,optimize = True)
               temp_ecv =   0.5*lib.einsum('JIba,bIk->aJk',eris_ccee,r_aaa_ecv,optimize = True)
               temp_ecv -=  0.5*lib.einsum('JaIb,bIk->aJk',eris_cece,r_aaa_ecv,optimize = True)
               temp_ecv -=  0.5*lib.einsum('IbJa,bIk->aJk',eris_CEce,r_bba_ecv,optimize = True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)
               
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('JIba,bIK->aJK',eris_CCEE,r_bba_ecc,optimize = True).reshape(-1) 
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('JaIb,bIK->aJK',eris_CECE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('JaIb,bIK->aJK',eris_CEce,r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] += 0.5*lib.einsum('Jiba,biK->aJK',eris_CVEE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -= 0.5*lib.einsum('ibJa,biK->aJK',eris_VECE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_ecc:f_bba_ecc] -=-0.5*lib.einsum('Jaib,bKi->aJK',eris_CEve,r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += 0.5*lib.einsum('JIba,bIk->aJk',eris_CCEE,r_bba_ecv,optimize = True).reshape(-1) 
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('JaIb,bIk->aJk',eris_CECE,r_bba_ecv,optimize = True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= 0.5*lib.einsum('JaIb,bIk->aJk',eris_CEce,r_aaa_ecv,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('Ijba,bIK->ajK',eris_CVEE,r_bba_ecc,optimize = True).reshape(-1) 
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('jaIb,bIK->ajK',eris_VECE,r_bba_ecc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('jaIb,bIK->ajK',eris_VEce,r_aaa_ecc_u,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += 0.5*lib.einsum('jiba,biK->ajK',eris_VVEE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -= 0.5*lib.einsum('jaib,biK->ajK',eris_VEVE,r_bba_evc,optimize = True).reshape(-1)
               s[s_bba_evc:f_bba_evc] -=-0.5*lib.einsum('jaib,bKi->ajK',eris_VEve,r_aaa_ecv,optimize = True).reshape(-1)
               
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('JIba,bIK->aJK',eris_ccee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('JaIb,bIK->aJK',eris_cece,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('IbJa,bIK->aJK',eris_CEce,r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] += 0.5*lib.einsum('Jiba,biK->aJK',eris_cvee,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -= 0.5*lib.einsum('ibJa,biK->aJK',eris_vece,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_ecc:f_aab_ecc] -=-0.5*lib.einsum('ibJa,bKi->aJK',eris_VEce,r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += 0.5*lib.einsum('JIba,bIk->aJk',eris_ccee,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('JaIb,bIk->aJk',eris_cece,r_aab_ecv,optimize = True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= 0.5*lib.einsum('IbJa,bIk->aJk',eris_CEce,r_bbb_ecv,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('Ijba,bIK->ajK',eris_cvee,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jaIb,bIK->ajK',eris_vece,r_aab_ecc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('Ibja,bIK->ajK',eris_CEve,r_bbb_ecc_u,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += 0.5*lib.einsum('jiba,biK->ajK',eris_vvee,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -= 0.5*lib.einsum('jaib,biK->ajK',eris_veve,r_aab_evc,optimize = True).reshape(-1)
               s[s_aab_evc:f_aab_evc] -=-0.5*lib.einsum('ibja,bKi->ajK',eris_VEve,r_bbb_ecv,optimize = True).reshape(-1)

               temp_ecc =   0.5*lib.einsum('JIba,bIK->aJK',eris_CCEE,r_bbb_ecc_u,optimize = True)
               temp_ecc -=  0.5*lib.einsum('JaIb,bIK->aJK',eris_CECE,r_bbb_ecc_u,optimize = True)
               temp_ecc -=  0.5*lib.einsum('JaIb,bIK->aJK',eris_CEce,r_aab_ecc,optimize = True)
               temp_ecc += -0.5*lib.einsum('Jiba,bKi->aJK',eris_CVEE,r_bbb_ecv,optimize = True)
               temp_ecc -= -0.5*lib.einsum('ibJa,bKi->aJK',eris_VECE,r_bbb_ecv,optimize = True)
               temp_ecc -=  0.5*lib.einsum('Jaib,biK->aJK',eris_CEve,r_aab_evc,optimize = True)
               temp_ecv =   0.5*lib.einsum('JIba,bIk->aJk',eris_CCEE,r_bbb_ecv,optimize = True)
               temp_ecv -=  0.5*lib.einsum('JaIb,bIk->aJk',eris_CECE,r_bbb_ecv,optimize = True)
               temp_ecv -=  0.5*lib.einsum('JaIb,bIk->aJk',eris_CEce,r_aab_ecv,optimize = True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1]].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)
               
        if (method == "adc(3)"):

               #eris_oecc = eris.oecc
               #eris_oecv = eris.oecv
               #eris_OECC = eris.OECC
               #eris_OECV = eris.OECV
               #eris_OEcc = eris.OEcc
               #eris_OEcv = eris.OEcv
               #eris_oeCC = eris.oeCC
               #eris_oeCV = eris.oeCV

               #eris_ceco = eris.ceco
               #eris_cevo = eris.cevo
               #eris_CECO = eris.CECO
               #eris_CEVO = eris.CEVO
               #eris_CEco = eris.CEco
               #eris_CEvo = eris.CEvo
               #eris_ceCO = eris.ceCO
               #eris_ceVO = eris.ceVO

               eris_oecc = eris.ovoo[:,:,:ncvs,:ncvs].copy() 
               eris_oecv = eris.ovoo[:,:,:ncvs,ncvs:].copy()
               eris_OECC = eris.OVOO[:,:,:ncvs,:ncvs].copy()
               eris_OECV = eris.OVOO[:,:,:ncvs,ncvs:].copy()
               eris_OEcc = eris.OVoo[:,:,:ncvs,:ncvs].copy()
               eris_OEcv = eris.OVoo[:,:,:ncvs,ncvs:].copy()
               eris_oeCC = eris.ovOO[:,:,:ncvs,:ncvs].copy()
               eris_oeCV = eris.ovOO[:,:,:ncvs,ncvs:].copy()

               eris_ceco = eris.ovoo[:ncvs,:,:ncvs,:].copy()
               eris_cevo = eris.ovoo[:ncvs,:,ncvs:,:].copy()
               eris_CECO = eris.OVOO[:ncvs,:,:ncvs,:].copy()
               eris_CEVO = eris.OVOO[:ncvs,:,ncvs:,:].copy()
               eris_CEco = eris.OVoo[:ncvs,:,ncvs:,:].copy()
               eris_CEvo = eris.OVoo[:ncvs,:,ncvs:,:].copy()
               eris_ceCO = eris.ovOO[:ncvs,:,:ncvs,:].copy()
               eris_ceVO = eris.ovOO[:ncvs,:,ncvs:,:].copy()

################ ADC(3) i - kja and ajk - i block ############################
               t2_1_a = adc.t2[0][0][:]
               t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy() 
               t2_1_a_voee = t2_1_a[ncvs:,:,:,:].copy()
               t2_1_a_cvee = t2_1_a[:ncvs,ncvs:,:,:].copy()
               t2_1_a_ccee = t2_1_a[:ncvs,:ncvs,:,:].copy()
               t2_1_a_ccee_t = t2_1_a_ccee[ij_ind_ncvs[0],ij_ind_ncvs[1],:,:] 
                
               if isinstance(eris.ceee, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = ncvs

               a = 0
               temp_singles = np.zeros((ncvs))
               temp_doubles = np.zeros((nvir_a, nvir_a, nvir_a))
               r_aaa_ecc = r_aaa_ecc.reshape(nvir_a,-1)
               
               temp_1_ecc = lib.einsum('Pbc,aP->abc',t2_1_a_ccee_t,r_aaa_ecc, optimize=True)
               temp_1_ecv = lib.einsum('Pqbc,aPq->abc',t2_1_a_cvee,r_aaa_ecv, optimize=True)
               for p in range(0,ncvs,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ceee = dfadc.get_ceee_spin_df(adc, eris.L_ce_t, eris.L_ee_t, p, chnk_size).reshape(-1,nvir_a,nvir_a,nvir_a)
                   else :       
                       eris_ceee = radc_ao2mo.unpack_eri_1(eris.ceee, nvir_a)
                   k = eris_ceee.shape[0]
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,Icab->I',temp_1_ecc, eris_ceee, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,Ibac->I',temp_1_ecc, eris_ceee, optimize=True)
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,Icab->I',temp_1_ecv, eris_ceee, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,Ibac->I',temp_1_ecv, eris_ceee, optimize=True)

                   temp_doubles += lib.einsum('I,Icab->bca',r_a[a:a+k],eris_ceee,optimize=True)
                   temp_doubles -= lib.einsum('I,Ibac->bca',r_a[a:a+k],eris_ceee,optimize=True)
                   del eris_ceee
                   a += k

               s[s_a:f_a] += temp_singles
               s[s_aaa_ecc:f_aaa_ecc] += 0.5*lib.einsum('bca,Pbc->aP',temp_doubles,t2_1_a_ccee_t,optimize=True).reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += 0.5*lib.einsum('bca,Pqbc->aPq',temp_doubles,t2_1_a_cvee,optimize=True).reshape(-1)
               del temp_singles
               del temp_doubles

               temp_ecc = lib.einsum('Jlab,aJK->blK',t2_1_a_coee,r_aaa_ecc_u,optimize=True)
               temp_ecv_1 = lib.einsum('Jlab,aJk->blk',t2_1_a_coee,r_aaa_ecv,optimize=True)
               temp_ecv_2 = -lib.einsum('jlab,aKj->blK',t2_1_a_voee,r_aaa_ecv,optimize=True)

               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_oecc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_ceco,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv_1,eris_oecv,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv_1,eris_cevo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_ecv_2,eris_oecc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecv_2,eris_ceco,optimize=True)
                
               temp_1_ecc = lib.einsum('Jlab,aJK->blK',t2_1_a_coee,r_aab_ecc,optimize=True)
               temp_1_ecv = lib.einsum('Jlab,aJk->blk',t2_1_a_coee,r_aab_ecv,optimize=True)
               temp_1_evc = lib.einsum('jlab,ajK->blK',t2_1_a_voee,r_aab_evc,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_oeCC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_1_ecv,eris_oeCV,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_evc,eris_oeCC,optimize=True)

               temp_ecc = -lib.einsum('klab,akj->blj',t2_1_a_coee,r_aaa_ecc_u,optimize=True)
               temp_ecv_1 = -lib.einsum('klab,akj->blj',t2_1_a_coee,r_aaa_ecv,optimize=True)
               temp_ecv_2 = lib.einsum('klab,ajk->blj',t2_1_a_voee,r_aaa_ecv,optimize=True)

               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_oecc,optimize=True)
               s[s_a:f_a] +=0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_ceco,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv_1,eris_oecv,optimize=True)
               s[s_a:f_a] +=0.5*lib.einsum('blj,Ibjl->I',temp_ecv_1,eris_cevo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecv_2,eris_oecc,optimize=True)
               s[s_a:f_a] +=0.5*lib.einsum('blJ,IbJl->I',temp_ecv_2,eris_ceco,optimize=True)

               temp_1_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_a_coee,r_aab_ecc,optimize=True)
               temp_1_ecv = -lib.einsum('Klab,aKj->blj',t2_1_a_coee,r_aab_ecv,optimize=True)
               temp_1_evc = -lib.einsum('klab,akJ->blJ',t2_1_a_voee,r_aab_evc,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_oeCC,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv,eris_oeCV,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_evc,eris_oeCC,optimize=True)

               temp_1 = lib.einsum('I,lbIK->Kbl',r_a, eris_oecc)
               temp_1 -= lib.einsum('I,IbKl->Kbl',r_a, eris_ceco)
               temp_2 = lib.einsum('I,lbIk->kbl',r_a, eris_oecv)
               temp_2 -= lib.einsum('I,Ibkl->kbl',r_a, eris_cevo)

               temp_ecc  = lib.einsum('kbl,jlab->ajk',temp_1,t2_1_a_coee,optimize=True)
               temp_ecv  = lib.einsum('kbl,jlab->ajk',temp_2,t2_1_a_coee,optimize=True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

               temp_3  = lib.einsum('I,lbIK->Kbl',r_b,eris_oeCC)
               temp_4  = lib.einsum('I,lbIk->kbl',r_b,eris_oeCV)
               temp_ecc = lib.einsum('Kbl,Jlab->aJK',temp_3,t2_1_a_coee,optimize=True)
               temp_ecv = lib.einsum('kbl,Jlab->aJk',temp_4,t2_1_a_coee,optimize=True)
               temp_evc = lib.einsum('Kbl,jlab->ajK',temp_3,t2_1_a_voee,optimize=True)
               s[s_aab_ecc:f_aab_ecc] += temp_ecc.reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += temp_ecv.reshape(-1)
               s[s_aab_evc:f_aab_evc] += temp_evc.reshape(-1)

               temp_1 = lib.einsum('I,lbIJ->Jbl',r_a, eris_oecc)
               temp_1 -= lib.einsum('I,IbJl->Jbl',r_a, eris_ceco)

               temp_ecc  = lib.einsum('Jbl,Klab->aJK',temp_1,t2_1_a_coee,optimize=True)
               temp_ecv  = lib.einsum('Jbl,klab->aJk',temp_1,t2_1_a_voee,optimize=True)
               s[s_aaa_ecc:f_aaa_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] -= temp_ecv.reshape(-1)

               del t2_1_a_coee
               del t2_1_a_voee
                
               t2_1_b = adc.t2[0][2][:]
               t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy() 
               t2_1_b_voee = t2_1_b[ncvs:,:,:,:].copy()
               t2_1_b_cvee = t2_1_b[:ncvs,ncvs:,:,:].copy() 
               t2_1_b_ccee = t2_1_b[:ncvs,:ncvs,:,:].copy()
               t2_1_b_ccee_t = t2_1_b_ccee[ij_ind_ncvs[0],ij_ind_ncvs[1],:,:]
               t2_1_ab = adc.t2[0][1][:]
               t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
               t2_1_ab_voee = t2_1_ab[ncvs:,:,:,:].copy()
               t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
               t2_1_ab_ovee = t2_1_ab[:,ncvs:,:,:].copy()
               t2_1_ab_ccee = t2_1_ab[:ncvs,:ncvs,:,:].copy()
               t2_1_ab_cvee = t2_1_ab[:ncvs,ncvs:,:,:].copy()
               t2_1_ab_vcee = t2_1_ab[ncvs:,:ncvs,:,:].copy()
               
               if isinstance(eris.CEEE, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = ncvs
               a = 0
               temp_singles = np.zeros((ncvs))
               temp_doubles = np.zeros((nvir_b, nvir_b, nvir_b))
               r_bbb_ecc = r_bbb_ecc.reshape(nvir_b,-1)

               temp_1_ecc = lib.einsum('Pbc,aP->abc',t2_1_b_ccee_t,r_bbb_ecc, optimize=True)
               temp_1_ecv = lib.einsum('Pqbc,aPq->abc',t2_1_b_cvee,r_bbb_ecv, optimize=True)
               for p in range(0,ncvs,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_CEEE = dfadc.get_ceee_spin_df(adc, eris.L_CE_t, eris.L_EE_t, p, chnk_size).reshape(-1,nvir_b,nvir_b,nvir_b)
                   else :
                       eris_CEEE = radc_ao2mo.unpack_eri_1(eris.CEEE, nvir_b)
                   k = eris_CEEE.shape[0]
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,Icab->I',temp_1_ecc, eris_CEEE, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,Ibac->I',temp_1_ecc, eris_CEEE, optimize=True)
                   temp_singles[a:a+k] += 0.5*lib.einsum('abc,Icab->I',temp_1_ecv, eris_CEEE, optimize=True)
                   temp_singles[a:a+k] -= 0.5*lib.einsum('abc,Ibac->I',temp_1_ecv, eris_CEEE, optimize=True)

                   temp_doubles += lib.einsum('I,Icab->bca',r_b[a:a+k],eris_CEEE,optimize=True)
                   temp_doubles -= lib.einsum('I,Ibac->bca',r_b[a:a+k],eris_CEEE,optimize=True)
                   del eris_CEEE
                   a += k

               s[s_b:f_b] += temp_singles
               s[s_bbb_ecc:f_bbb_ecc] += 0.5*lib.einsum('bca,Pbc->aP',temp_doubles,t2_1_b_ccee_t,optimize=True).reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += 0.5*lib.einsum('bca,Pqbc->aPq',temp_doubles,t2_1_b_cvee,optimize=True).reshape(-1)
               del temp_singles
               del temp_doubles

               temp_1_ecc = lib.einsum('Jlab,aJK->blK',t2_1_b_coee,r_bba_ecc,optimize=True)
               temp_1_ecv = lib.einsum('Jlab,aJk->blk',t2_1_b_coee,r_bba_ecv,optimize=True)
               temp_1_evc = lib.einsum('jlab,ajK->blK',t2_1_b_voee,r_bba_evc,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_OEcc,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_1_ecv,eris_OEcv,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_evc,eris_OEcc,optimize=True)

               temp_ecc = lib.einsum('Jlab,aJK->blK',t2_1_b_coee,r_bbb_ecc_u,optimize=True)
               temp_ecv_1 = lib.einsum('Jlab,aJk->blk',t2_1_b_coee,r_bbb_ecv,optimize=True)
               temp_ecv_2 = -lib.einsum('jlab,aKj->blK',t2_1_b_voee,r_bbb_ecv,optimize=True)

               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_OECC,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_CECO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv_1,eris_OECV,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv_1,eris_CEVO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_ecv_2,eris_OECC,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecv_2,eris_CECO,optimize=True)

               temp_1_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_b_coee,r_bba_ecc,optimize=True)
               temp_1_ecv = -lib.einsum('Klab,aKj->blj',t2_1_b_coee,r_bba_ecv,optimize=True)
               temp_1_evc = -lib.einsum('klab,akJ->blJ',t2_1_b_voee,r_bba_evc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_OEcc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv,eris_OEcv,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_evc,eris_OEcc,optimize=True)

               temp_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_b_coee,r_bbb_ecc_u,optimize=True)
               temp_ecv_1 = -lib.einsum('Klab,aKj->blj',t2_1_b_coee,r_bbb_ecv,optimize=True)
               temp_ecv_2 = lib.einsum('klab,aJk->blJ',t2_1_b_voee,r_bbb_ecv,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_OECC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_CECO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv_1,eris_OECV,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,Ibjl->I',temp_ecv_1,eris_CEVO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecv_2,eris_OECC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecv_2,eris_CECO,optimize=True)

               temp_2  = lib.einsum('I,lbIK->Kbl',r_a,eris_OEcc)
               temp_3  = lib.einsum('I,lbIk->kbl',r_a,eris_OEcv)
               temp_ecc = lib.einsum('Kbl,Jlab->aJK',temp_2,t2_1_b_coee,optimize=True)
               temp_ecv = lib.einsum('kbl,Jlab->aJk',temp_3,t2_1_b_coee,optimize=True)
               temp_evc = lib.einsum('Kbl,jlab->ajK',temp_2,t2_1_b_voee,optimize=True)
               s[s_bba_ecc:f_bba_ecc] += temp_ecc.reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += temp_ecv.reshape(-1)
               s[s_bba_evc:f_bba_evc] += temp_evc.reshape(-1)

               temp_1 = lib.einsum('I,lbIK->Kbl',r_b, eris_OECC)
               temp_1 -= lib.einsum('I,IbKl->Kbl',r_b, eris_CECO)
               temp_2 = lib.einsum('I,lbIk->kbl',r_b, eris_OECV)
               temp_2 -= lib.einsum('I,Ibkl->kbl',r_b, eris_CEVO)

               temp_ecc  = lib.einsum('Kbl,Jlab->aJK',temp_1,t2_1_b_coee,optimize=True)
               temp_ecv  = lib.einsum('kbl,Jlab->aJk',temp_2,t2_1_b_coee,optimize=True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

               temp_1 = lib.einsum('I,lbIJ->Jbl',r_b, eris_OECC)
               temp_1 -= lib.einsum('I,IbJl->Jbl',r_b, eris_CECO)

               temp_ecc  = lib.einsum('Jbl,Klab->aJK',temp_1,t2_1_b_coee,optimize=True)
               temp_ecv  = lib.einsum('Jbl,klab->aJk',temp_1,t2_1_b_voee,optimize=True)
               s[s_bbb_ecc:f_bbb_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] -= temp_ecv.reshape(-1)
               del t2_1_b_coee
               del t2_1_b_voee
                 
               if isinstance(eris.ceEE, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = ncvs
               a = 0
               temp_1_ecc = lib.einsum('KJcb,aJK->abc',t2_1_ab_ccee,r_bba_ecc, optimize=True)
               temp_1_ecv = lib.einsum('kJcb,aJk->abc',t2_1_ab_vcee,r_bba_ecv, optimize=True)
               temp_1_evc = lib.einsum('Kjcb,ajK->abc',t2_1_ab_cvee,r_bba_evc, optimize=True)
               temp_2 = np.zeros((nvir_a, nvir_b, nvir_b))
               for p in range(0,ncvs,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ceEE = dfadc.get_ceee_spin_df(adc, eris.L_ce_t, eris.L_EE_t, p, chnk_size).reshape(-1,nvir_a,nvir_b,nvir_b)
                   else :
                       eris_ceEE = radc_ao2mo.unpack_eri_1(eris.ceEE, nvir_b)
                   k = eris_ceEE.shape[0]

                   s[s_a:f_a][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecc, eris_ceEE, optimize=True)
                   s[s_a:f_a][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecv, eris_ceEE, optimize=True)
                   s[s_a:f_a][a:a+k] += lib.einsum('abc,Icab->I',temp_1_evc, eris_ceEE, optimize=True)

                   temp_2 += lib.einsum('I,Icab->cba',r_a[a:a+k],eris_ceEE,optimize=True)
                   del eris_ceEE
                   a += k

               s[s_bba_ecc:f_bba_ecc] += lib.einsum('cba,KJcb->aJK',temp_2, t2_1_ab_ccee, optimize=True).reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += lib.einsum('cba,kJcb->aJk',temp_2, t2_1_ab_vcee, optimize=True).reshape(-1)
               s[s_bba_evc:f_bba_evc] += lib.einsum('cba,Kjcb->ajK',temp_2, t2_1_ab_cvee, optimize=True).reshape(-1)
               del temp_1_ecc
               del temp_1_ecv
               del temp_1_evc
               del temp_2
                
               if isinstance(eris.CEee, type(None)):
                   chnk_size = uadc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = ncvs

               a = 0
               temp_1_ecc = lib.einsum('JKbc,aJK->abc',t2_1_ab_ccee,r_aab_ecc, optimize=True)
               temp_1_ecv = lib.einsum('Jkbc,aJk->abc',t2_1_ab_cvee,r_aab_ecv, optimize=True)
               temp_1_evc = lib.einsum('jKbc,ajK->abc',t2_1_ab_vcee,r_aab_evc, optimize=True)
               temp_2 = np.zeros((nvir_a, nvir_b, nvir_a))
               for p in range(0,ncvs,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_CEee = dfadc.get_ceee_spin_df(adc, eris.L_CE_t, eris.L_ee_t, p, chnk_size).reshape(-1,nvir_b,nvir_a,nvir_a)
                   else :
                       eris_CEee = radc_ao2mo.unpack_eri_1(eris.CEee, nvir_a)
                   k = eris_CEee.shape[0]
                   s[s_b:f_b][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecc, eris_CEee, optimize=True)
                   s[s_b:f_b][a:a+k] += lib.einsum('abc,Icab->I',temp_1_ecv, eris_CEee, optimize=True)
                   s[s_b:f_b][a:a+k] += lib.einsum('abc,Icab->I',temp_1_evc, eris_CEee, optimize=True)

                   temp_2 += lib.einsum('I,Icab->bca',r_b[a:a+k],eris_CEee,optimize=True)
                   del eris_CEee
                   a += k

               s[s_aab_ecc:f_aab_ecc] += lib.einsum('bca,JKbc->aJK',temp_2, t2_1_ab_ccee, optimize=True).reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += lib.einsum('bca,Jkbc->aJk',temp_2, t2_1_ab_cvee, optimize=True).reshape(-1)
               s[s_aab_evc:f_aab_evc] += lib.einsum('bca,jKbc->ajK',temp_2, t2_1_ab_vcee, optimize=True).reshape(-1)
               del temp_1_ecc
               del temp_1_ecv
               del temp_2

               temp_ecc = lib.einsum('lJba,aJK->blK',t2_1_ab_ocee,r_bba_ecc,optimize=True)
               temp_ecv = lib.einsum('lJba,aJk->blk',t2_1_ab_ocee,r_bba_ecv,optimize=True)
               temp_evc = lib.einsum('ljba,ajK->blK',t2_1_ab_ovee,r_bba_evc,optimize=True)
               temp_1_ecc = lib.einsum('Jlab,aJK->blK',t2_1_ab_coee,r_aaa_ecc_u,optimize=True)
               temp_1_ecv = lib.einsum('Jlab,aJk->blk',t2_1_ab_coee,r_aaa_ecv,optimize=True)
               temp_1_evc = -lib.einsum('jlab,aKj->blK',t2_1_ab_voee,r_aaa_ecv,optimize=True)
               temp_2_ecc = lib.einsum('Jlba,aKJ->blK',t2_1_ab_coee,r_bba_ecc, optimize=True)
               temp_2_ecv = lib.einsum('jlba,aKj->blK',t2_1_ab_voee,r_bba_ecv, optimize=True)
               temp_2_evc = lib.einsum('Jlba,akJ->blk',t2_1_ab_coee,r_bba_evc,optimize=True)

               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_oecc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_ceco,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv,eris_oecv,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv,eris_cevo,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_evc,eris_oecc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_evc,eris_ceco,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_OEcc,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blk,lbIk->I',temp_1_ecv,eris_OEcv,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blK,lbIK->I',temp_1_evc,eris_OEcc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecc,eris_ceCO,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecv,eris_ceCO,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_evc,eris_ceVO,optimize=True)

               temp_ecc = lib.einsum('Jlab,aJK->blK',t2_1_ab_coee,r_aab_ecc,optimize=True)
               temp_ecv = lib.einsum('Jlab,aJk->blk',t2_1_ab_coee,r_aab_ecv,optimize=True)
               temp_evc = lib.einsum('jlab,ajK->blK',t2_1_ab_voee,r_aab_evc,optimize=True)
               temp_1_ecc = lib.einsum('lJba,aJK->blK',t2_1_ab_ocee,r_bbb_ecc_u,optimize=True)
               temp_1_evc = lib.einsum('lJba,aJk->blk',t2_1_ab_ocee,r_bbb_ecv,optimize=True)
               temp_1_ecv = -lib.einsum('ljba,aKj->blK',t2_1_ab_ovee,r_bbb_ecv,optimize=True)
               temp_2_ecc = lib.einsum('lJab,aKJ->blK',t2_1_ab_ocee,r_aab_ecc,optimize=True)
               temp_2_ecv = lib.einsum('ljab,aKj->blK',t2_1_ab_ovee,r_aab_ecv,optimize=True)
               temp_2_evc = lib.einsum('lJab,akJ->blk',t2_1_ab_ocee,r_aab_evc,optimize=True)

               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_ecc,eris_OECC,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_ecc,eris_CECO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_ecv,eris_OECV,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,Ibkl->I',temp_ecv,eris_CEVO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_evc,eris_OECC,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_evc,eris_CECO,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecc,eris_oeCC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blK,lbIK->I',temp_1_ecv,eris_oeCC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blk,lbIk->I',temp_1_evc,eris_oeCV,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecc,eris_CEco,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blK,IbKl->I',temp_2_ecv,eris_CEco,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blk,Ibkl->I',temp_2_evc,eris_CEvo,optimize=True)

               temp_ecc = -lib.einsum('lKba,aKJ->blJ',t2_1_ab_ocee,r_bba_ecc,optimize=True)
               temp_ecv = -lib.einsum('lKba,aKj->blj',t2_1_ab_ocee,r_bba_ecv,optimize=True)
               temp_evc = -lib.einsum('lkba,akJ->blJ',t2_1_ab_ovee,r_bba_evc,optimize=True)
               temp_1_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_ab_coee,r_aaa_ecc_u,optimize=True)
               temp_1_ecv_1 = -lib.einsum('Klab,aKj->blj',t2_1_ab_coee,r_aaa_ecv,optimize=True)
               temp_1_ecv_2 = lib.einsum('klab,aJk->blJ',t2_1_ab_voee,r_aaa_ecv,optimize=True)
               temp_2_ecc = -lib.einsum('Klba,aJK->blJ',t2_1_ab_coee,r_bba_ecc,optimize=True)
               temp_2_ecv = -lib.einsum('klba,aJk->blJ',t2_1_ab_voee,r_bba_ecv,optimize=True)
               temp_2_evc = -lib.einsum('Klba,ajK->blj',t2_1_ab_coee,r_bba_evc,optimize=True)

               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_oecc,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_ceco,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv,eris_oecv,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,Ibjl->I',temp_ecv,eris_cevo,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_evc,eris_oecc,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_evc,eris_ceco,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_OEcc,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv_1,eris_OEcv,optimize=True)
               s[s_a:f_a] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecv_2,eris_OEcc,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecc,eris_ceCO,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecv,eris_ceCO,optimize=True)
               s[s_a:f_a] += 0.5*lib.einsum('blj,Ibjl->I',temp_2_evc,eris_ceVO,optimize=True)

               temp_ecc = -lib.einsum('Klab,aKJ->blJ',t2_1_ab_coee,r_aab_ecc,optimize=True)
               temp_ecv = -lib.einsum('Klab,aKj->blj',t2_1_ab_coee,r_aab_ecv,optimize=True)
               temp_evc = -lib.einsum('klab,akJ->blJ',t2_1_ab_voee,r_aab_evc,optimize=True)
               temp_1_ecc = -lib.einsum('lKba,aKJ->blJ',t2_1_ab_ocee,r_bbb_ecc_u,optimize=True)
               temp_1_ecv_1 = -lib.einsum('lKba,aKj->blj',t2_1_ab_ocee,r_bbb_ecv,optimize=True)
               temp_1_ecv_2 = lib.einsum('lkba,aJk->blJ',t2_1_ab_ovee,r_bbb_ecv,optimize=True)
               temp_2_ecc = -lib.einsum('lKab,aJK->blJ',t2_1_ab_ocee,r_aab_ecc,optimize=True)
               temp_2_ecv = -lib.einsum('lkab,aJk->blJ',t2_1_ab_ovee,r_aab_ecv,optimize=True)
               temp_2_evc = -lib.einsum('lKab,ajK->blj',t2_1_ab_ocee,r_aab_evc,optimize=True)

               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_ecc,eris_OECC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_ecc,eris_CECO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_ecv,eris_OECV,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,Ibjl->I',temp_ecv,eris_CEVO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_evc,eris_OECC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_evc,eris_CECO,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecc,eris_oeCC,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blj,lbIj->I',temp_1_ecv_1,eris_oeCV,optimize=True)
               s[s_b:f_b] -= 0.5*lib.einsum('blJ,lbIJ->I',temp_1_ecv_2,eris_oeCC,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecc,eris_CEco,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blJ,IbJl->I',temp_2_ecv,eris_CEco,optimize=True)
               s[s_b:f_b] += 0.5*lib.einsum('blj,Ibjl->I',temp_2_evc,eris_CEvo,optimize=True)
               
               temp_2 = lib.einsum('I,lbIK->Kbl',r_a, eris_OEcc)
               temp_3 = lib.einsum('I,lbIk->kbl',r_a, eris_OEcv)
               temp_ecc = lib.einsum('Kbl,Jlab->aJK',temp_2, t2_1_ab_coee,optimize=True)
               temp_ecv = lib.einsum('kbl,Jlab->aJk',temp_3, t2_1_ab_coee,optimize=True)
               s[s_aaa_ecc:f_aaa_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] += temp_ecv.reshape(-1)

               temp_2  = lib.einsum('I,lbIK->Kbl',r_a,eris_oecc)
               temp_2  -= lib.einsum('I,IbKl->Kbl',r_a,eris_ceco)
               temp_3  = lib.einsum('I,lbIk->kbl',r_a,eris_oecv)
               temp_3  -= lib.einsum('I,Ibkl->kbl',r_a,eris_cevo)

               temp_ecc  = lib.einsum('Kbl,lJba->aJK',temp_2,t2_1_ab_ocee,optimize=True)
               temp_ecv  = lib.einsum('kbl,lJba->aJk',temp_3,t2_1_ab_ocee,optimize=True)
               temp_evc  = lib.einsum('Kbl,ljba->ajK',temp_2,t2_1_ab_ovee,optimize=True)
               s[s_bba_ecc:f_bba_ecc] += temp_ecc.reshape(-1)
               s[s_bba_ecv:f_bba_ecv] += temp_ecv.reshape(-1)
               s[s_bba_evc:f_bba_evc] += temp_evc.reshape(-1)

               temp_4 = lib.einsum('I,lbIK->Kbl',r_b, eris_oeCC)
               temp_5 = lib.einsum('I,lbIk->kbl',r_b, eris_oeCV)

               temp_ecc = lib.einsum('Kbl,lJba->aJK',temp_4,t2_1_ab_ocee,optimize=True)
               temp_ecv = lib.einsum('kbl,lJba->aJk',temp_5,t2_1_ab_ocee,optimize=True)
               s[s_bbb_ecc:f_bbb_ecc] += temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] += temp_ecv.reshape(-1)

               temp_1  = lib.einsum('I,lbIK->Kbl',r_b,eris_OECC)
               temp_1  -= lib.einsum('I,IbKl->Kbl',r_b,eris_CECO)
               temp_2  = lib.einsum('I,lbIk->kbl',r_b,eris_OECV)
               temp_2  -= lib.einsum('I,Ibkl->kbl',r_b,eris_CEVO)

               temp_ecc  = lib.einsum('Kbl,Jlab->aJK',temp_1,t2_1_ab_coee,optimize=True)
               temp_ecv  = lib.einsum('kbl,Jlab->aJk',temp_2,t2_1_ab_coee,optimize=True)
               temp_evc  = lib.einsum('Kbl,jlab->ajK',temp_1,t2_1_ab_voee,optimize=True)
               s[s_aab_ecc:f_aab_ecc] += temp_ecc.reshape(-1)
               s[s_aab_ecv:f_aab_ecv] += temp_ecv.reshape(-1)
               s[s_aab_evc:f_aab_evc] += temp_evc.reshape(-1)

               temp_2 = lib.einsum('I,lbIJ->Jbl',r_a, eris_OEcc)

               temp_ecc = lib.einsum('Jbl,Klab->aJK',temp_2,t2_1_ab_coee,optimize=True)
               temp_ecv = lib.einsum('Jbl,klab->aJk',temp_2,t2_1_ab_voee,optimize=True)
               s[s_aaa_ecc:f_aaa_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_aaa_ecv:f_aaa_ecv] -= temp_ecv.reshape(-1)

               temp_1  = -lib.einsum('I,IbJl->Jbl',r_a,eris_ceCO,optimize=True)
               temp_2  = -lib.einsum('I,Ibjl->jbl',r_a,eris_ceVO,optimize=True)
               temp_1_ecc = -lib.einsum('Jbl,Klba->aJK',temp_1,t2_1_ab_coee,optimize=True)
               temp_1_ecv = -lib.einsum('Jbl,klba->aJk',temp_1,t2_1_ab_voee,optimize=True)
               temp_1_evc = -lib.einsum('jbl,Klba->ajK',temp_2,t2_1_ab_coee,optimize=True)
               s[s_bba_ecc:f_bba_ecc] -= temp_1_ecc.reshape(-1)
               s[s_bba_ecv:f_bba_ecv] -= temp_1_ecv.reshape(-1)
               s[s_bba_evc:f_bba_evc] -= temp_1_evc.reshape(-1)

               temp_2 = lib.einsum('I,lbIJ->Jbl',r_b, eris_oeCC)
               temp_ecc = lib.einsum('Jbl,lKba->aJK',temp_2,t2_1_ab_ocee,optimize=True)
               temp_ecv = lib.einsum('Jbl,lkba->aJk',temp_2,t2_1_ab_ovee,optimize=True)
               s[s_bbb_ecc:f_bbb_ecc] -= temp_ecc[:,ij_ind_ncvs[0],ij_ind_ncvs[1] ].reshape(-1)
               s[s_bbb_ecv:f_bbb_ecv] -= temp_ecv.reshape(-1)

               temp_3  = -lib.einsum('I,IbJl->Jbl',r_b,eris_CEco,optimize=True)
               temp_4  = -lib.einsum('I,Ibjl->jbl',r_b,eris_CEvo,optimize=True)
               temp_1_ecc = -lib.einsum('Jbl,lKab->aJK',temp_3,t2_1_ab_ocee,optimize=True)
               temp_1_ecv = -lib.einsum('Jbl,lkab->aJk',temp_3,t2_1_ab_ovee,optimize=True)
               temp_1_evc = -lib.einsum('jbl,lKab->ajK',temp_4,t2_1_ab_ocee,optimize=True)
               s[s_aab_ecc:f_aab_ecc] -= temp_1_ecc.reshape(-1)
               s[s_aab_ecv:f_aab_ecv] -= temp_1_ecv.reshape(-1)
               s[s_aab_evc:f_aab_evc] -= temp_1_evc.reshape(-1)

               del  t2_1_ab_coee
               del  t2_1_ab_voee              
               del  t2_1_ab_ocee
               del  t2_1_ab_ovee
               del  t2_1_ab_ccee
               del  t2_1_ab_cvee
               del  t2_1_ab_vcee
               
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        return s

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
            
            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
            t2_1_t_a = t2_1_t.transpose(2,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0)

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
           
######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
          
            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:]
                t2_2_t_a = t2_2_t.transpose(2,1,0)
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0)

                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
            
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
            
            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
            t2_1_t_b = t2_1_t.transpose(2,1,0)
            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1)

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
            
######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]
            
            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:]
                t2_2_t_b = t2_2_t.transpose(2,1,0)

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1)

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
             
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

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

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
        t2_1_a_coee = t2_1_a[:ncvs,:,:,:].copy()
        t2_1_ab_coee = t2_1_ab[:ncvs,:,:,:].copy()
        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :ncvs]
            T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a_coee, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab_coee, optimize = True)
            T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab_coee, optimize = True)
        else :
            T[s_a:f_a] += t1_2_a[:ncvs,(orb-nocc_a)]

######## ADC(2) 2h-1p  part  ############################################

            t2_1_a_t = t2_1_a.transpose(2,3,1,0)
            t2_1_ab_t = t2_1_ab.transpose(2,3,1,0)
            t2_1_a_eecc = t2_1_a_t[:,:,:ncvs,:ncvs].copy()
            t2_1_a_eecv = t2_1_a_t[:,:,:ncvs,ncvs:].copy()
            t2_1_a_ecc = t2_1_a_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
            t2_1_ab_eecc = t2_1_ab_t[:,:,:ncvs,:ncvs].copy()
            t2_1_ab_eecv = t2_1_ab_t[:,:,:ncvs,ncvs:].copy()
            t2_1_ab_eevc = t2_1_ab_t[:,:,ncvs:,:ncvs].copy()

            T[s_aaa_ecc:f_aaa_ecc] = t2_1_a_ecc[(orb-nocc_a),:,:].reshape(-1)
            T[s_aaa_ecv:f_aaa_ecv] = t2_1_a_eecv[(orb-nocc_a),:,:,:].reshape(-1)
            T[s_bba_ecc:f_bba_ecc] = t2_1_ab_eecc[(orb-nocc_a),:,:,:].reshape(-1)
            T[s_bba_ecv:f_bba_ecv] = t2_1_ab_eecv[(orb-nocc_a),:,:,:].reshape(-1)
            T[s_bba_evc:f_bba_evc] = t2_1_ab_eevc[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):
            
            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            
            if orb >= nocc_a:

                t2_2_a_t = t2_2_a.transpose(2,3,1,0)
                t2_2_ab_t = t2_2_ab.transpose(2,3,1,0)
                t2_2_a_eecc = t2_2_a_t[:,:,:ncvs,:ncvs].copy()
                t2_2_a_eecv = t2_2_a_t[:,:,:ncvs,ncvs:].copy()
                t2_2_a_ecc = t2_2_a_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
                t2_2_ab_eecc = t2_2_ab_t[:,:,:ncvs,:ncvs].copy()
                t2_2_ab_eecv = t2_2_ab_t[:,:,:ncvs,ncvs:].copy()
                t2_2_ab_eevc = t2_2_ab_t[:,:,ncvs:,:ncvs].copy()

                T[s_aaa_ecc:f_aaa_ecc] += t2_2_a_ecc[(orb-nocc_a),:,:].reshape(-1)
                T[s_aaa_ecv:f_aaa_ecv] += t2_2_a_eecv[(orb-nocc_a),:,:,:].reshape(-1)
                T[s_bba_ecc:f_bba_ecc] += t2_2_ab_eecc[(orb-nocc_a),:,:,:].reshape(-1)
                T[s_bba_ecv:f_bba_ecv] += t2_2_ab_eecv[(orb-nocc_a),:,:,:].reshape(-1)
                T[s_bba_evc:f_bba_evc] += t2_2_ab_eevc[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_a:
                     
                t2_1_a_tmp = np.ascontiguousarray(t2_1_a[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[orb,:,:,:])
                t2_2_a_coee = t2_2_a[:ncvs,:,:,:].copy()
                t2_2_ab_coee = t2_2_ab[:ncvs,:,:,:].copy()

                T[s_a:f_a] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a_tmp, t2_2_a_coee, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1_ab_tmp, t2_2_ab_coee, optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1_ab_tmp, t2_2_ab_coee, optimize = True)
       
                del t2_1_a_tmp, t2_1_ab_tmp

                t2_2_a_tmp = np.ascontiguousarray(t2_2_a[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[orb,:,:,:])

                T[s_a:f_a] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a_coee,  t2_2_a_tmp,optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1_ab_coee, t2_2_ab_tmp,optimize = True)
                T[s_a:f_a] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1_ab_coee, t2_2_ab_tmp,optimize = True)

                del t2_2_a_tmp, t2_2_ab_tmp
            else:
                t2_1_a_tmp =  np.ascontiguousarray(t2_1_a[:ncvs,:,(orb-nocc_a),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:ncvs,:,(orb-nocc_a),:])

                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_a_tmp, t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*lib.einsum('ikc,kc->i',t2_1_ab_tmp, t1_2_b,optimize = True)
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
        t2_1_b_coee = t2_1_b[:ncvs,:,:,:].copy()
        t2_1_ab_ocee = t2_1_ab[:,:ncvs,:,:].copy()
        if orb < nocc_b:
            
            t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
            t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])

            T[s_b:f_b] = idn_occ_b[orb, :ncvs]
            T[s_b:f_b]+= 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_1_b_coee, optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_1_ab_ocee, optimize = True)
            T[s_b:f_b]-= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_1_ab_ocee, optimize = True)
            del t2_1_b_tmp, t2_1_ab_tmp
        else :
            T[s_b:f_b] += t1_2_b[:ncvs,(orb-nocc_b)]

######## ADC(2) 2h-1p part  ############################################

            t2_1_b_t = t2_1_b.transpose(2,3,1,0)
            t2_1_ab_t = t2_1_ab.transpose(2,3,0,1)
            t2_1_b_eecc = t2_1_b_t[:,:,:ncvs,:ncvs].copy()
            t2_1_b_eecv = t2_1_b_t[:,:,:ncvs,ncvs:].copy()
            t2_1_b_ecc = t2_1_b_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
            t2_1_ab_eecc = t2_1_ab_t[:,:,:ncvs,:ncvs].copy()
            t2_1_ab_eecv = t2_1_ab_t[:,:,:ncvs,ncvs:].copy()
            t2_1_ab_eevc = t2_1_ab_t[:,:,ncvs:,:ncvs].copy()

            T[s_bbb_ecc:f_bbb_ecc] = t2_1_b_ecc[(orb-nocc_b),:,:].reshape(-1)
            T[s_bbb_ecv:f_bbb_ecv] = t2_1_b_eecv[(orb-nocc_b),:,:,:].reshape(-1)
            T[s_aab_ecc:f_aab_ecc] = t2_1_ab_eecc[:,(orb-nocc_b),:,:].reshape(-1)
            T[s_aab_ecv:f_aab_ecv] = t2_1_ab_eecv[:,(orb-nocc_b),:,:].reshape(-1)
            T[s_aab_evc:f_aab_evc] = t2_1_ab_eevc[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-x'or method=='adc(3)'):

            t2_2_a = adc.t2[1][0][:]
            t2_2_ab = adc.t2[1][1][:]
            t2_2_b = adc.t2[1][2][:]
            
            if orb >= nocc_b:

                t2_2_b_t = t2_2_b.transpose(2,3,1,0)
                t2_2_ab_t = t2_2_ab.transpose(2,3,0,1)
                t2_2_b_eecc = t2_2_b_t[:,:,:ncvs,:ncvs].copy()
                t2_2_b_eecv = t2_2_b_t[:,:,:ncvs,ncvs:].copy()
                t2_2_b_ecc = t2_2_b_eecc[:,:,ij_ind_ncvs[0],ij_ind_ncvs[1]]
                t2_2_ab_eecc = t2_2_ab_t[:,:,:ncvs,:ncvs].copy()
                t2_2_ab_eecv = t2_2_ab_t[:,:,:ncvs,ncvs:].copy()
                t2_2_ab_eevc = t2_2_ab_t[:,:,ncvs:,:ncvs].copy()

                T[s_bbb_ecc:f_bbb_ecc] += t2_2_b_ecc[(orb-nocc_b),:,:].reshape(-1)
                T[s_bbb_ecv:f_bbb_ecv] += t2_2_b_eecv[(orb-nocc_b),:,:,:].reshape(-1)
                T[s_aab_ecc:f_aab_ecc] += t2_2_ab_eecc[:,(orb-nocc_b),:,:].reshape(-1)
                T[s_aab_ecv:f_aab_ecv] += t2_2_ab_eecv[:,(orb-nocc_b),:,:].reshape(-1)
                T[s_aab_evc:f_aab_evc] += t2_2_ab_eevc[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):

            t1_3_a, t1_3_b = adc.t1[1]

            if orb < nocc_b:

                t2_1_b_tmp = np.ascontiguousarray(t2_1_b[:,orb,:,:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,orb,:,:])
                t2_2_b_coee = t2_2_b[:ncvs,:,:,:].copy()
                t2_2_ab_ocee = t2_2_ab[:,:ncvs,:,:].copy()
                

                T[s_b:f_b] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_b_tmp, t2_2_b_coee, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kdc,kidc->i',t2_1_ab_tmp, t2_2_ab_ocee, optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kcd,kicd->i',t2_1_ab_tmp, t2_2_ab_ocee, optimize = True)

                del t2_1_b_tmp, t2_1_ab_tmp

                t2_2_b_tmp = np.ascontiguousarray(t2_2_b[:,orb,:,:])
                t2_2_ab_tmp = np.ascontiguousarray(t2_2_ab[:,orb,:,:])

                T[s_b:f_b] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_b_coee,  t2_2_b_tmp ,optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kicd,kcd->i',t2_1_ab_ocee, t2_2_ab_tmp,optimize = True)
                T[s_b:f_b] -= 0.25*lib.einsum('kidc,kdc->i',t2_1_ab_ocee, t2_2_ab_tmp,optimize = True)
                
                del t2_2_b_tmp, t2_2_ab_tmp

            else:
                t2_1_b_tmp  = np.ascontiguousarray(t2_1_b[:ncvs,:,(orb-nocc_b),:])
                t2_1_ab_tmp = np.ascontiguousarray(t2_1_ab[:,:ncvs,:,(orb-nocc_b)])

                T[s_b:f_b] += 0.5*lib.einsum('ikc,kc->i',t2_1_b_tmp, t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*lib.einsum('kic,kc->i',t2_1_ab_tmp, t1_2_a,optimize = True)
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
        
        print("U_sorted.size: ", U_sorted.size)
        print("U_sorted: ", U_sorted)
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
                print("a_idx", a_idx)                
                print("i_idx", i_idx)
                print("j_idx", j_idx)
                doubles_bab_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bab_val.append(U_sorted[iter_idx])
          
            if orb_idx in range(s_aba,f_aba):
                aij_idx = orb_idx - s_aba    
                ij_rem = aij_idx % (nocc_b*nocc_a)
                a_idx = aij_idx//(nocc_b*nocc_a)
                i_idx = ij_rem//nocc_b
                j_idx = ij_rem % nocc_b
                print("a_idx", a_idx)                
                print("i_idx", i_idx)
                print("j_idx", j_idx)
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

def analyze_eigenvector_ip_cvs(adc):

    nocc_a = adc.nocc_a
    nocc_b = adc.nocc_b
    nvir_a = adc.nvir_a
    nvir_b = adc.nvir_b
    ncvs = adc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs
    evec_print_tol = adc.evec_print_tol

    logger.info(adc, "Number of core orbitals = %d", ncvs)
    logger.info(adc, "Number of alpha occupied orbitals = %d", nocc_a)
    logger.info(adc, "Number of beta occupied orbitals = %d", nocc_b)
    logger.info(adc, "Number of alpha virtual orbitals =  %d", nvir_a)
    logger.info(adc, "Number of beta virtual orbitals =  %d", nvir_b)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)

    ij_ncvs = np.tril_indices(ncvs, k=-1)

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

    U = adc.U

    for I in range(U.shape[1]):
        U1 = U[:f_b,I]
        U2 = U[f_b:,I]
        U1dotU1 = np.dot(U1, U1) 
        U2dotU2 = np.dot(U2, U2) 

        temp_aaa_ecc = np.zeros((nvir_a, ncvs, ncvs))
        temp_aaa_ecc[:,ij_ncvs[0],ij_ncvs[1]] =  U[s_aaa_ecc:f_aaa_ecc,I].reshape(nvir_a,-1).copy()
        temp_aaa_ecc[:,ij_ncvs[1],ij_ncvs[0]] = -U[s_aaa_ecc:f_aaa_ecc,I].reshape(nvir_a,-1).copy()
        U_aaa_ecc = temp_aaa_ecc.reshape(-1).copy()

        temp_bbb_ecc = np.zeros((nvir_b, ncvs, ncvs))
        temp_bbb_ecc[:,ij_ncvs[0],ij_ncvs[1]] =  U[s_bbb_ecc:f_bbb_ecc,I].reshape(nvir_b,-1).copy()
        temp_bbb_ecc[:,ij_ncvs[1],ij_ncvs[0]] = -U[s_bbb_ecc:f_bbb_ecc,I].reshape(nvir_b,-1).copy()
        U_bbb_ecc = temp_bbb_ecc.reshape(-1).copy()

        U_sq = U[:,I].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx] 
        U_sorted = U[ind_idx,I].copy()

        U_sq_aaa_ecc = U_aaa_ecc.copy()**2
        U_sq_bbb_ecc = U_bbb_ecc.copy()**2
        ind_idx_aaa_ecc = np.argsort(-U_sq_aaa_ecc)
        ind_idx_bbb_ecc = np.argsort(-U_sq_bbb_ecc)
        U_sq_aaa_ecc = U_sq_aaa_ecc[ind_idx_aaa_ecc]
        U_sq_bbb_ecc = U_sq_bbb_ecc[ind_idx_bbb_ecc]
        U_sorted_aaa_ecc = U_aaa_ecc[ind_idx_aaa_ecc].copy()
        U_sorted_bbb_ecc = U_bbb_ecc[ind_idx_bbb_ecc].copy()

        U_sorted = U_sorted[U_sq > evec_print_tol**2]
        ind_idx = ind_idx[U_sq > evec_print_tol**2]
        U_sorted_aaa_ecc = U_sorted_aaa_ecc[U_sq_aaa_ecc > evec_print_tol**2]
        U_sorted_bbb_ecc = U_sorted_bbb_ecc[U_sq_bbb_ecc > evec_print_tol**2]
        ind_idx_aaa_ecc = ind_idx_aaa_ecc[U_sq_aaa_ecc > evec_print_tol**2]
        ind_idx_bbb_ecc = ind_idx_bbb_ecc[U_sq_bbb_ecc > evec_print_tol**2]
        
        print("U_sorted.size: ", U_sorted.size)
        print("U_sorted: ", U_sorted)
        print("idx_idx: ", ind_idx)
        print(" s_aaa_ecc", s_aaa_ecc ) 
        print(" f_aaa_ecc", f_aaa_ecc ) 
        print(" s_aaa_ecv", s_aaa_ecv ) 
        print(" f_aaa_ecv", f_aaa_ecv ) 
        print(" s_bba_ecc", s_bba_ecc ) 
        print(" f_bba_ecc", f_bba_ecc ) 
        print(" s_bba_ecv", s_bba_ecv ) 
        print(" f_bba_ecv", f_bba_ecv ) 
        print(" s_bba_evc", s_bba_evc ) 
        print(" f_bba_evc", f_bba_evc ) 
        print(" s_aab_ecc", s_aab_ecc ) 
        print(" f_aab_ecc", f_aab_ecc ) 
        print(" s_aab_ecv", s_aab_ecv ) 
        print(" f_aab_ecv", f_aab_ecv ) 
        print(" s_aab_evc", s_aab_evc ) 
        print(" f_aab_evc", f_aab_evc ) 
        print(" s_bbb_ecc", s_bbb_ecc ) 
        print(" f_bbb_ecc", f_bbb_ecc ) 
        print(" s_bbb_ecv", s_bbb_ecv ) 
        print(" f_bbb_ecv", f_bbb_ecv ) 
        singles_a_idx = []
        singles_b_idx = []
        doubles_aaa_idx = []
        doubles_bba_idx = []
        doubles_aab_idx = []
        doubles_bbb_idx = []  
        singles_a_val = []
        singles_b_val = []
        doubles_aaa_val = []
        doubles_bba_val = []
        doubles_aab_val = []  
        doubles_bbb_val = []
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

            if orb_idx in range(s_aaa_ecv,f_aaa_ecv):
                aij_idx = orb_idx - s_aaa_ecv       
                ij_rem = aij_idx % (ncvs*nval_a)
                a_idx = aij_idx//(ncvs*nval_a)
                i_idx = ij_rem//nval_a
                j_idx = ij_rem % nval_a 
                doubles_aaa_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                doubles_aaa_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bba_ecv,f_bba_ecv):
                aij_idx = orb_idx - s_bba_ecv       
                ij_rem = aij_idx % (ncvs*nval_a)
                a_idx = aij_idx//(ncvs*nval_a)
                i_idx = ij_rem//ncvs
                j_idx = ij_rem % ncvs
                print("a_idx_ecv", a_idx)                
                print("i_idx_ecv", i_idx)
                print("j_idx_ecv", j_idx)
 
                doubles_bba_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bba_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bba_evc,f_bba_evc):
                aij_idx = orb_idx - s_bba_evc       
                ij_rem = aij_idx % (nval_b*ncvs)
                a_idx = aij_idx//(nval_b*ncvs)
                i_idx = ij_rem//nval_b
                j_idx = ij_rem % nval_b
                print("a_idx_evc", a_idx)                
                print("i_idx_evc", i_idx)
                print("j_idx_evc", j_idx)
                doubles_bba_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bba_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_aab_ecv,f_aab_ecv):
                aij_idx = orb_idx - s_aab_ecv    
                ij_rem = aij_idx % (ncvs*nval_b)
                a_idx = aij_idx//(ncvs*nval_b)
                i_idx = ij_rem//nval_b
                j_idx = ij_rem % nval_b
                print("a_idx_ecv", a_idx)                
                print("i_idx_ecv", i_idx)
                print("j_idx_ecv", j_idx)
                doubles_aab_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                doubles_aab_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_aab_evc,f_aab_evc):
                aij_idx = orb_idx - s_aab_evc   
                ij_rem = aij_idx % (ncvs*nval_b)
                a_idx = aij_idx//(ncvs*nval_b)
                i_idx = ij_rem//ncvs
                j_idx = ij_rem % ncvs 
                print("a_idx_evc", a_idx)                
                print("i_idx_evc", i_idx)
                print("j_idx_evc", j_idx)
                doubles_aab_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
                doubles_aab_val.append(U_sorted[iter_idx])

            if orb_idx in range(s_bbb_ecv,f_bbb_ecv):
                aij_idx = orb_idx - s_bbb_ecv     
                ij_rem = aij_idx % (ncvs*nval_b)
                a_idx = aij_idx//(ncvs*nval_b)
                i_idx = ij_rem//nval_b
                j_idx = ij_rem % nval_b
                doubles_bbb_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
                doubles_bbb_val.append(U_sorted[iter_idx])
            iter_idx += 1
             
        for iter_idx, orb_aaa_ecc in enumerate(ind_idx_aaa_ecc):              
            ij_rem = orb_aaa_ecc % (ncvs*ncvs)
            a_idx = orb_aaa_ecc//(ncvs*ncvs)
            i_idx = ij_rem//ncvs
            j_idx = ij_rem % ncvs
            doubles_aaa_idx.append((a_idx + 1 + nocc_a, i_idx + 1, j_idx + 1))
            doubles_aaa_val.append(U_sorted_aaa_ecc[iter_idx])

        for iter_idx, orb_bbb_ecc in enumerate(ind_idx_bbb_ecc):                
            ij_rem = orb_bbb_ecc % (ncvs*ncvs) 
            a_idx = orb_bbb_ecc//(ncvs*ncvs)
            i_idx = ij_rem//ncvs
            j_idx = ij_rem %ncvs
            doubles_bbb_idx.append((a_idx + 1 + nocc_b, i_idx + 1, j_idx + 1))
            doubles_bbb_val.append(U_sorted_bbb_ecc[iter_idx])

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

        if doubles_bba_val:
            logger.info(adc, "\n2h1p(beta|alpha|beta) block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_bba_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_bba_val[idx])

        if doubles_aab_val:
            logger.info(adc, "\n2h1p(alpha|beta|alpha) block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_aab_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_aab_val[idx])

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

    U = adc.U
    print('debugging # of roots specfactors = ', nroots)
    #Spectroscopic amplitudes
    X_a = np.dot(T_a, U).reshape(-1,nroots)
    X_b = np.dot(T_b, U).reshape(-1,nroots)

    X = (X_a,X_b)

    #Spectroscopic factors
    P = lib.einsum("pi,pi->i", X_a, X_a)
    P += lib.einsum("pi,pi->i", X_b, X_b)

    return P, X

def compute_rdm_tdm(adc, L, R):
    
    L = np.array(L).ravel() 
    R = np.array(R).ravel()

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
    
    opdm_a  = np.zeros((nmo_a,nmo_a))
    opdm_b  = np.zeros((nmo_b,nmo_b))
    kd_oc_a = np.identity(nocc_a)
    kd_oc_b = np.identity(nocc_b)

    L_a = L[s_a:f_a]
    L_b = L[s_b:f_b]
    L_aaa = L[s_aaa:f_aaa]
    L_bab = L[s_bab:f_bab]
    L_aba = L[s_aba:f_aba]
    L_bbb = L[s_bbb:f_bbb]

    R_a = R[s_a:f_a]
    R_b = R[s_b:f_b]
    R_aaa = R[s_aaa:f_aaa]
    R_bab = R[s_bab:f_bab]
    R_aba = R[s_aba:f_aba]
    R_bbb = R[s_bbb:f_bbb]

    L_aaa = L_aaa.reshape(nvir_a,-1)
    print('L_aaa shape = ', L_aaa.shape)
    L_bbb = L_bbb.reshape(nvir_b,-1)
    L_aaa_u = None
    L_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
    L_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= L_aaa.copy()
    L_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -L_aaa.copy()

    L_bbb_u = None
    L_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
    L_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= L_bbb.copy()
    L_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -L_bbb.copy()

    L_aba = L_aba.reshape(nvir_a,nocc_a,nocc_b)
    L_bab = L_bab.reshape(nvir_b,nocc_b,nocc_a)


    R_aaa = R_aaa.reshape(nvir_a,-1)
    R_bbb = R_bbb.reshape(nvir_b,-1)
    R_aaa_u = None
    R_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
    R_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= R_aaa.copy()
    R_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -R_aaa.copy()

    R_bbb_u = None
    R_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
    R_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= R_bbb.copy()
    R_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -R_bbb.copy()

    R_aba = R_aba.reshape(nvir_a,nocc_a,nocc_b)
    R_bab = R_bab.reshape(nvir_b,nocc_b,nocc_a)
################TOTAL OPDMS OPDMS OPDMS OPDMS####################################

#########################
#####G^000#### block- ij
    opdm_a[:nocc_a,:nocc_a] =  np.einsum('ij,m,m->ij',kd_oc_a,L_a,R_a,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] -= np.einsum('i,j->ij',L_a,R_a,optimize=True)
    opdm_a[:nocc_a,:nocc_a] +=  np.einsum('ij,m,m->ij',kd_oc_a,L_b,R_b,optimize=True)

    opdm_b[:nocc_b,:nocc_b] =  np.einsum('ij,m,m->ij',kd_oc_b,L_b,R_b,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] -= np.einsum('i,j->ij',L_b,R_b,optimize=True)
    opdm_b[:nocc_b,:nocc_b] += np.einsum('ij,m,m->ij',kd_oc_b,L_a,R_a,optimize=True) 

####G^101#### block- ij

    opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_a,L_aaa_u,R_aaa_u,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] += np.einsum('ij,etu,etu->ij',kd_oc_a,L_bab,R_bab,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] += np.einsum('ij,etu,etu->ij',kd_oc_a,L_aba,R_aba,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_a,L_bbb_u,R_bbb_u,optimize=True)
    opdm_a[:nocc_a,:nocc_a] -= np.einsum('eti,etj->ij',L_aaa_u,R_aaa_u,optimize=True)
    opdm_a[:nocc_a,:nocc_a] -= np.einsum('eti,etj->ij',L_bab,R_bab,optimize=True)
    opdm_a[:nocc_a,:nocc_a] -= np.einsum('eit,ejt->ij',L_aba,R_aba,optimize=True)

    opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_b,L_aaa_u,R_aaa_u,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] += np.einsum('ij,etu,etu->ij',kd_oc_b,L_bab,R_bab,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] += np.einsum('ij,etu,etu->ij',kd_oc_b,L_aba,R_aba,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('ij,etu,etu->ij',kd_oc_b,L_bbb_u,R_bbb_u,optimize=True)
    opdm_b[:nocc_b,:nocc_b] -= np.einsum('eti,etj->ij',L_bbb_u,R_bbb_u,optimize=True)
    opdm_b[:nocc_b,:nocc_b] -= np.einsum('eti,etj->ij',L_aba,R_aba,optimize=True)
    opdm_b[:nocc_b,:nocc_b] -= np.einsum('eit,ejt->ij',L_bab,R_bab,optimize=True)

######G^020#### block- ij

    opdm_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', L_a,R_a,t2_1_a,t2_1_a,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', L_b,R_b,t2_1_a,t2_1_a,optimize=True)
 
    opdm_a[:nocc_a,:nocc_a] -= np.einsum('g,g,jhcd,ihcd->ij', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] -= np.einsum('g,g,jhcd,ihcd->ij', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True) 
    opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('g,h,jgcd,ihcd->ij', L_a,R_a,t2_1_a,t2_1_a,optimize=True)  
    opdm_a[:nocc_a,:nocc_a] += np.einsum('g,h,jgcd,ihcd->ij', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)  
    opdm_a[:nocc_a,:nocc_a] += 0.25*np.einsum('g,j,ghcd,ihcd->ij',L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('g,j,ghcd,ihcd->ij',L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    opdm_a[:nocc_a,:nocc_a] += 0.25*np.einsum('g,i,jhcd,ghcd->ij',R_a,L_a,t2_1_a,t2_1_a,optimize=True)
    opdm_a[:nocc_a,:nocc_a] += 0.5*np.einsum('g,i,jhcd,ghcd->ij',R_a,L_a,t2_1_ab,t2_1_ab,optimize=True)


    opdm_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', L_b,R_b,t2_1_b,t2_1_b,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] -= 0.5*np.einsum('g,g,hjcd,hicd->ij', L_a,R_a,t2_1_b,t2_1_b,optimize=True)
 
    opdm_b[:nocc_b,:nocc_b] -= np.einsum('g,g,hjcd,hicd->ij', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] -= np.einsum('g,g,hjcd,hicd->ij', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True) 
    opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('g,h,jgcd,ihcd->ij', L_b,R_b,t2_1_b,t2_1_b,optimize=True)  
    opdm_b[:nocc_b,:nocc_b] += np.einsum('g,h,gjcd,hicd->ij', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)  
    opdm_b[:nocc_b,:nocc_b] += 0.25*np.einsum('g,j,ghcd,ihcd->ij',L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('g,j,hgcd,hicd->ij',L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    opdm_b[:nocc_b,:nocc_b] += 0.25*np.einsum('g,i,jhcd,ghcd->ij',R_b,L_b,t2_1_b,t2_1_b,optimize=True)
    opdm_b[:nocc_b,:nocc_b] += 0.5*np.einsum('g,i,hjcd,hgcd->ij',R_b,L_b,t2_1_ab,t2_1_ab,optimize=True)
       
######G^101#### block- ab

    opdm_a[nocc_a:,nocc_a:] = 0.5*np.einsum('atu,btu->ab', L_aaa_u,R_aaa_u,optimize=True)
    opdm_a[nocc_a:,nocc_a:] += np.einsum('atu,btu->ab', L_aba,R_aba,optimize=True)

    opdm_b[nocc_b:,nocc_b:] = 0.5*np.einsum('atu,btu->ab', L_bbb_u,R_bbb_u,optimize=True)
    opdm_b[nocc_b:,nocc_b:] += np.einsum('atu,btu->ab', L_bab,R_bab,optimize=True)

#######G^020#### block- ab
    opdm_a[nocc_a:,nocc_a:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    opdm_a[nocc_a:,nocc_a:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', L_b,R_b,t2_1_a,t2_1_a,optimize=True)
    opdm_a[nocc_a:,nocc_a:] += np.einsum('g,g,hmbc,hmac->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    opdm_a[nocc_a:,nocc_a:] += np.einsum('g,g,hmbc,hmac->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    opdm_a[nocc_a:,nocc_a:] -= np.einsum('g,h,hmbc,gmac->ab', L_a,R_a,t2_1_a,t2_1_a,optimize=True)
    opdm_a[nocc_a:,nocc_a:] -= np.einsum('g,h,hmbc,gmac->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    opdm_a[nocc_a:,nocc_a:] -= np.einsum('g,h,mhbc,mgac->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)

    opdm_b[nocc_b:,nocc_b:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    opdm_b[nocc_b:,nocc_b:] += 0.5*np.einsum('g,g,hmbc,hmac->ab', L_a,R_a,t2_1_b,t2_1_b,optimize=True)
    opdm_b[nocc_b:,nocc_b:] += np.einsum('g,g,hmcb,hmca->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    opdm_b[nocc_b:,nocc_b:] += np.einsum('g,g,hmcb,hmca->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)
    opdm_b[nocc_b:,nocc_b:] -= np.einsum('g,h,hmbc,gmac->ab', L_b,R_b,t2_1_b,t2_1_b,optimize=True)
    opdm_b[nocc_b:,nocc_b:] -= np.einsum('g,h,mhcb,mgca->ab', L_b,R_b,t2_1_ab,t2_1_ab,optimize=True)
    opdm_b[nocc_b:,nocc_b:] -= np.einsum('g,h,hmcb,gmca->ab', L_a,R_a,t2_1_ab,t2_1_ab,optimize=True)

#######G^100#### block- ia
    opdm_a[:nocc_a,nocc_a:] = -np.einsum('n,ani->ia', R_a,L_aaa_u,optimize=True)
    opdm_a[:nocc_a,nocc_a:] += np.einsum('n,ain->ia', R_b,L_aba,optimize=True)

    opdm_b[:nocc_b,nocc_b:] = -np.einsum('n,ani->ia', R_b,L_bbb_u,optimize=True)
    opdm_b[:nocc_b,nocc_b:] += np.einsum('n,ain->ia', R_a,L_bab,optimize=True)
######G^110#### block- ia

    opdm_a[:nocc_a,nocc_a:] -= np.einsum('g,cgh,ihac->ia', L_a,R_aaa_u,t2_1_a,optimize=True) 
    opdm_a[:nocc_a,nocc_a:] += np.einsum('g,chg,ihac->ia', L_a,R_bab,t2_1_ab,optimize=True) 
    opdm_a[:nocc_a,nocc_a:] += np.einsum('g,chg,ihac->ia', L_b,R_aba,t2_1_a,optimize=True) 
    opdm_a[:nocc_a,nocc_a:] -= np.einsum('g,cgh,ihac->ia', L_b,R_bbb_u,t2_1_ab,optimize=True) 
    opdm_a[:nocc_a,nocc_a:] += 0.5*np.einsum('i,cgh,ghac->ia', L_a,R_aaa_u,t2_1_a,optimize=True)
    opdm_a[:nocc_a,nocc_a:] -= np.einsum('i,chg,ghac->ia', L_a,R_bab,t2_1_ab,optimize=True)

    opdm_b[:nocc_b,nocc_b:] -= np.einsum('g,cgh,ihac->ia', L_b,R_bbb_u,t2_1_b,optimize=True) 
    opdm_b[:nocc_b,nocc_b:] += np.einsum('g,chg,hica->ia', L_b,R_aba,t2_1_ab,optimize=True) 
    opdm_b[:nocc_b,nocc_b:] += np.einsum('g,chg,hica->ia', L_a,R_bab,t2_1_b,optimize=True) 
    opdm_b[:nocc_b,nocc_b:] -= np.einsum('g,cgh,hica->ia', L_a,R_aaa_u,t2_1_ab,optimize=True) 
    opdm_b[:nocc_b,nocc_b:] += 0.5*np.einsum('i,cgh,ghac->ia', L_b,R_bbb_u,t2_1_b,optimize=True)
    opdm_b[:nocc_b,nocc_b:] -= np.einsum('i,chg,hgca->ia', L_b,R_aba,t2_1_ab,optimize=True)

#######G^020#### block- ia

    opdm_a[:nocc_a,nocc_a:] += np.einsum('g,g,ia->ia', L_a,R_a,t1_2_a,optimize=True) 
    opdm_a[:nocc_a,nocc_a:] += np.einsum('g,g,ia->ia', L_b,R_b,t1_2_a,optimize=True) 
    opdm_a[:nocc_a,nocc_a:] -= np.einsum('g,i,ga->ia', R_a,L_a,t1_2_a,optimize=True)

    opdm_b[:nocc_b,nocc_b:] += np.einsum('g,g,ia->ia', L_b,R_b,t1_2_b,optimize=True) 
    opdm_b[:nocc_b,nocc_b:] += np.einsum('g,g,ia->ia', L_a,R_a,t1_2_b,optimize=True) 
    opdm_b[:nocc_b,nocc_b:] -= np.einsum('g,i,ga->ia', R_b,L_b,t1_2_b,optimize=True)

############ block- ai
    opdm_a[nocc_a:,:nocc_a] = opdm_a[:nocc_a,nocc_a:].T 
    opdm_b[nocc_b:,:nocc_b] = opdm_b[:nocc_b,nocc_b:].T
########################
    herm_a = np.linalg.norm(opdm_a - opdm_a.transpose(1,0))
    herm_b = np.linalg.norm(opdm_b - opdm_b.transpose(1,0))
    norm_a = np.linalg.norm(opdm_a)
    norm_b = np.linalg.norm(opdm_b)

    print('norm_a: ', norm_a)
    print('norm_b: ', norm_b)
    print("total OPDM singles norm for Hermiticity (alpha)",herm_a)
    print("total OPDM singles norm for Hermiticity (beta)",herm_b)
    print("alpha opdm trace",np.einsum('pp',opdm_a))
    print("beta opdm trace",np.einsum('pp',opdm_b))
    print("alpha+beta opdm trace",np.einsum('pp',opdm_a+opdm_b))
    
    return opdm_a, opdm_b

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
    '''unrestricted ADC for IP-CVS energies and spectroscopic amplitudes

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
            IP-CVS energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP-CVS transition.
        p_ip : float
            Spectroscopic amplitudes for each IP-CVS transition.
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
    get_imds = get_imds_ip_cvs
    get_imds = get_imds_ip
    get_diag = ip_cvs_adc_diag
    matvec = ip_cvs_adc_matvec
    compute_trans_moments = ip_cvs_compute_trans_moments
    #compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_properties = get_properties


    analyze_spec_factor = analyze_spec_factor
    analyze_eigenvector = analyze_eigenvector_ip_cvs
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
