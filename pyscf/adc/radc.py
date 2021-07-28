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
Restricted algebraic diagrammatic construction
'''
import time
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
import pandas as pd
from linalg_helper_beta import davidson_nosym1
from multiroot_davidson import eighg

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
      
    nfc_orb = adc.nfc_orb
    nkop_chk = adc.nkop_chk
    cvs_npick = adc.cvs_npick
    kop_npick = adc.kop_npick
    ncore_proj = adc.ncore_proj
    ncore_proj_valence = adc.ncore_proj_valence
    Eh2ev = 27.211386245988
    alpha_proj = adc.alpha_proj
    mom_skd_iter = adc.mom_skd_iter 

    if ncore_proj > 0:
        cvs = True
    else:
        cvs = False
 
    if ((ncore_proj > 0) and (kop_npick is not False)) or ((cvs_npick is not False) and (kop_npick is not False)):
        raise Exception("Cannot calculate CVS and Koopman's excitations simultaneously")

    if  ((ncore_proj > 0) or (cvs_npick is not False) or (kop_npick is not False)) and (adc.method_type == "ea"):
       raise Exception("CVS and Koopman's aren't not implemented for EA")



    imds = adc.get_imds(eris)
    if (mom_skd_iter == False) and (ncore_proj > 0):
        matvec, diag = adc.gen_matvec(imds, eris, cvs)
        guess = adc.get_init_guess(nroots, diag, ascending = True)
        conv, E, U = davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=1e-14, max_cycle=adc.max_cycle, max_space=adc.max_space)

    elif (mom_skd_iter == True) and (ncore_proj > 0):
    
        matvec, diag = adc.gen_matvec(imds, eris, cvs=True, alpha_proj=0)
        guess = adc.get_init_guess(nroots, diag, ascending = True)
        conv, E, U = lib.linalg_helper.davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space)
        guess = U
        imds = adc.get_imds(eris,fc_bool=False)
        matvec, diag = adc.gen_matvec(imds, eris, cvs=False, fc_bool=False, alpha_proj=1)
            
            
        def cvs_pick(cvs_npick,U):          
            len_cvs_npick = len(cvs_npick)
            nroots = len_cvs_npick
            dim_guess = np.array(U).shape[1]
            guess = np.zeros((len_cvs_npick, dim_guess))
            for idx_guess, npick in enumerate(cvs_npick):
                U = np.array(U)
                guess[idx_guess,:] = U[npick,:]
            return guess, nroots
           
        if cvs_npick:
            guess,nroots = cvs_pick(cvs_npick,U)
           
        def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum('pi,pi->i', s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            w, v, idx = lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_eigenvectors = True)
            return w, v, idx

        conv, E, U = lib.linalg_helper.davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, pick=eig_close_to_init_guess, nroots=nroots, verbose=log, tol=adc.conv_tol, max_cycle=adc.max_cycle, max_space=adc.max_space)
        #conv, E, U = davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, pick=eig_close_to_init_guess, nroots=nroots, verbose=log, tol=skd[2], max_cycle=skd[1], max_space=adc.max_space)
    elif (mom_skd_iter == False) and (ncore_proj == 0):
        matvec, diag = adc.gen_matvec(imds, eris, cvs)
        guess = adc.get_init_guess(nroots, diag, ascending = True)
        conv, E, U = davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=1e-14, max_cycle=adc.max_cycle, max_space=adc.max_space)
           
    elif (ncore_proj_valence > 0):
        matvec, diag = adc.gen_matvec(imds, eris, cvs)
        guess = adc.get_init_guess(nroots, diag, ascending = True)
        conv, E, U = davidson_nosym1(lambda xs : [matvec(x) for x in xs], guess, diag, nroots=nroots, verbose=log, tol=1e-14, max_cycle=adc.max_cycle, max_space=adc.max_space)
     
############################################

    U = np.array(U)
    for i in range(U.shape[0]):
        print("CVS/MOM overlap: ", np.dot(np.array(guess)[i,:], U[i,:].T))
    #adc.analyze_eigenvector_ip(U)
    T = adc.get_trans_moments()

    spec_factors = adc.get_spec_factors(T, U, nroots)
    nfalse = np.shape(conv)[0] - np.sum(conv)
    if nfalse >= 1:
        print ("*************************************************************")
        print (" WARNING : ", "Davidson iterations for ",nfalse, "root(s) not converged")
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

    return E, U, spec_factors


def compute_amplitudes_energy(myadc, eris, verbose=None, fc_bool=True):

    t1, t2 = myadc.compute_amplitudes(eris, fc_bool=True)
    e_corr = myadc.compute_energy(t1, t2, eris)

    return e_corr, t1, t2


def compute_amplitudes(myadc, eris, fc_bool=True):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)
   
    nocc = myadc._nocc
    nvir = myadc._nvir
    
    if fc_bool is False:
        nfc_orb = 0
    else:
        nfc_orb = myadc.nfc_orb
    
        
    eris_oooo = eris.oooo
    eris_ovoo = eris.ovoo
    eris_ovov = eris.ovov
    eris_oovv = eris.oovv
    eris_ovvo = eris.ovvo
  
    # Compute first-order doubles t2 (tijab)

    v2e_oovv = eris.ovov[:].transpose(0,2,1,3).copy()

    e = myadc.mo_energy
    d_ij = e[:nocc][:,None] + e[:nocc]
    d_ab = e[nocc:][:,None] + e[nocc:]

    D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
    D1 = e[:nocc][:None].reshape(-1,1) - e[nocc:].reshape(-1)

    D2 = D2.reshape((nocc,nocc,nvir,nvir))
    D1 = D1.reshape((nocc,nvir))

    t2_1 = v2e_oovv/D2

    # Frozen core
    t2_1[:nfc_orb,:,:,:] = 0
    t2_1[:,:nfc_orb,:,:] = 0
    t2_1[:nfc_orb,:nfc_orb,:,:] = 0

    del (v2e_oovv)
    del (D2)

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    # Compute second-order singles t1 (tij)
 
    if isinstance(eris.ovvv, type(None)):
        chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
    else :
        chnk_size = nocc
    a = 0
    t1_2 = np.zeros((nocc,nvir))
    
    for p in range(0,nocc,chnk_size):
        if getattr(myadc, 'with_df', None):
            eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
        else :
            eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
        k = eris_ovvv.shape[0]
 
        t1_2 += 0.5*lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
        t1_2 -= 0.5*lib.einsum('kdac,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)
        t1_2 -= 0.5*lib.einsum('kcad,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
        t1_2 += 0.5*lib.einsum('kcad,kicd->ia',eris_ovvv,t2_1[a:a+k,:],optimize=True)

        t1_2 += lib.einsum('kdac,ikcd->ia',eris_ovvv,t2_1[:,a:a+k],optimize=True)
        del eris_ovvv
        a += k

    t1_2 -= 0.5*lib.einsum('lcki,klac->ia',eris_ovoo,t2_1,optimize=True)
    t1_2 += 0.5*lib.einsum('lcki,lkac->ia',eris_ovoo,t2_1,optimize=True)
    t1_2 -= 0.5*lib.einsum('kcli,lkac->ia',eris_ovoo,t2_1,optimize=True)
    t1_2 += 0.5*lib.einsum('kcli,klac->ia',eris_ovoo,t2_1,optimize=True)
    t1_2 -= lib.einsum('lcki,klac->ia',eris_ovoo,t2_1,optimize=True)
    
    # Frozen core
    t1_2[:nfc_orb,:] = 0

    t1_2 = t1_2/D1

    cput0 = log.timer_debug1("Completed t1_2 amplitude calculation", *cput0)

    t2_2 = None
    t1_3 = None

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

    # Compute second-order doubles t2 (tijab)

        eris_oooo = eris.oooo 
        eris_ovvo = eris.ovvo

        if isinstance(eris.vvvv, np.ndarray):
            eris_vvvv = eris.vvvv
            temp = t2_1.reshape(nocc*nocc,nvir*nvir)
            t2_2 = np.dot(temp,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            t2_2 = contract_ladder(myadc,t2_1,eris.vvvv)
        else:
            t2_2 = contract_ladder(myadc,t2_1,eris.Lvv)

        t2_2 += lib.einsum('kilj,klab->ijab',eris_oooo,t2_1,optimize=True)
        t2_2 += lib.einsum('kcbj,kica->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 -= lib.einsum('kcbj,ikca->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 += lib.einsum('kcbj,ikac->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 -= lib.einsum('kjbc,ikac->ijab',eris_oovv,t2_1,optimize=True)
        t2_2 -= lib.einsum('kibc,kjac->ijab',eris_oovv,t2_1,optimize=True)
        t2_2 -= lib.einsum('kjac,ikcb->ijab',eris_oovv,t2_1,optimize=True)
        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 -= lib.einsum('kcai,jkcb->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 += lib.einsum('kcai,kjcb->ijab',eris_ovvo,t2_1,optimize=True)
        t2_2 -= lib.einsum('kiac,kjcb->ijab',eris_oovv,t2_1,optimize=True)
        
       
        D2 = d_ij.reshape(-1,1) - d_ab.reshape(-1)
        D2 = D2.reshape((nocc,nocc,nvir,nvir))
        
        # Frozen core
        t2_2[:nfc_orb,:,:,:] = 0
        t2_2[:,:nfc_orb,:,:] = 0      
        t2_2[:nfc_orb,:nfc_orb,:,:] = 0

        t2_2 = t2_2/D2
        del (D2)

    cput0 = log.timer_debug1("Completed t2_2 amplitude calculation", *cput0)
        
    if (myadc.method == "adc(3)"):
    # Compute third-order singles (tij)

        eris_ovoo = eris.ovoo
                
        t1_3 = lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1,t1_2,optimize=True)
        t1_3 -= lib.einsum('d,liad,ld->ia',e[nocc:],t2_1,t1_2,optimize=True)
        t1_3 += lib.einsum('d,ilad,ld->ia',e[nocc:],t2_1,t1_2,optimize=True)
 
        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1, t1_2,optimize=True)
        t1_3 += lib.einsum('l,liad,ld->ia',e[:nocc],t2_1, t1_2,optimize=True)
        t1_3 -= lib.einsum('l,ilad,ld->ia',e[:nocc],t2_1,t1_2,optimize=True)
 
        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1, t1_2,optimize=True)
        t1_3 -= 0.5*lib.einsum('a,liad,ld->ia',e[nocc:],t2_1, t1_2,optimize=True)
        t1_3 += 0.5*lib.einsum('a,ilad,ld->ia',e[nocc:],t2_1,t1_2,optimize=True)
 
        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1, t1_2,optimize=True)
        t1_3 += 0.5*lib.einsum('i,liad,ld->ia',e[:nocc],t2_1, t1_2,optimize=True)
        t1_3 -= 0.5*lib.einsum('i,ilad,ld->ia',e[:nocc],t2_1,t1_2,optimize=True)
 
        t1_3 += lib.einsum('ld,iald->ia',t1_2,eris_ovov,optimize=True)
        t1_3 -= lib.einsum('ld,laid->ia',t1_2,eris_ovov,optimize=True)
        t1_3 += lib.einsum('ld,iald->ia',t1_2,eris_ovov,optimize=True)
 
        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo ,optimize=True)
        t1_3 -= lib.einsum('ld,liad->ia',t1_2,eris_oovv ,optimize=True)
        t1_3 += lib.einsum('ld,ldai->ia',t1_2,eris_ovvo,optimize=True)
 
        t1_3 -= 0.5*lib.einsum('lmad,mdli->ia',t2_2,eris_ovoo,optimize=True)
        t1_3 += 0.5*lib.einsum('mlad,mdli->ia',t2_2,eris_ovoo,optimize=True)
        t1_3 += 0.5*lib.einsum('lmad,ldmi->ia',t2_2,eris_ovoo,optimize=True)
        t1_3 -= 0.5*lib.einsum('mlad,ldmi->ia',t2_2,eris_ovoo,optimize=True)
        t1_3 -=     lib.einsum('lmad,mdli->ia',t2_2,eris_ovoo,optimize=True)
        
        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(myadc)
        else :
            chnk_size = nocc
        a = 0
      

        for p in range(0,nocc,chnk_size):
            if getattr(myadc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_df(myadc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            k = eris_ovvv.shape[0]

            t1_3 += 0.5*lib.einsum('ilde,lead->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
            t1_3 -= 0.5*lib.einsum('lide,lead->ia', t2_2[a:a+k],eris_ovvv,optimize=True)

            t1_3 -= 0.5*lib.einsum('ilde,ldae->ia', t2_2[:,a:a+k],eris_ovvv,optimize=True)
            t1_3 += 0.5*lib.einsum('lide,ldae->ia', t2_2[a:a+k],eris_ovvv,optimize=True)

            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1, eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 += lib.einsum('ildf,mefa,mlde->ia',t2_1, eris_ovvv,  t2_1[a:a+k] ,optimize=True)
            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1, eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('lidf,mefa,mlde->ia',t2_1, eris_ovvv,  t2_1[a:a+k] ,optimize=True)

            t1_3 += lib.einsum('ildf,mafe,lmde->ia',t2_1, eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1, eris_ovvv,  t2_1[a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('lidf,mafe,lmde->ia',t2_1, eris_ovvv,  t2_1[:,a:a+k] ,optimize=True)
            t1_3 += lib.einsum('lidf,mafe,mlde->ia',t2_1, eris_ovvv,  t2_1[a:a+k] ,optimize=True)

            t1_3 += lib.einsum('ilfd,mefa,mled->ia',  t2_1,eris_ovvv, t2_1[a:a+k],optimize=True)
            t1_3 -= lib.einsum('ilfd,mafe,mled->ia',  t2_1,eris_ovvv, t2_1[a:a+k],optimize=True)

            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('liaf,mefd,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('liaf,mefd,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('liaf,mdfe,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('liaf,mdfe,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,ifea,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,ifea,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('mldf,ifea,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,ifea,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3[a:a+k] += lib.einsum('mlfd,iaef,mled->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= lib.einsum('mlfd,ifea,mled->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3[a:a+k] -= 0.25*lib.einsum('lmef,iedf,lmad->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.25*lib.einsum('lmef,iedf,mlad->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.25*lib.einsum('mlef,iedf,lmad->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.25*lib.einsum('mlef,iedf,mlad->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3[a:a+k] += 0.25*lib.einsum('lmef,ifde,lmad->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.25*lib.einsum('lmef,ifde,mlad->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.25*lib.einsum('mlef,ifde,lmad->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.25*lib.einsum('mlef,ifde,mlad->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3 += 0.5*lib.einsum('ilaf,mefd,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= 0.5*lib.einsum('ilaf,mefd,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3 -= 0.5*lib.einsum('ilaf,mdfe,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 += 0.5*lib.einsum('ilaf,mdfe,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3 -= lib.einsum('ildf,mafe,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)
            t1_3 += lib.einsum('ilaf,mefd,mled->ia',t2_1,eris_ovvv,t2_1[a:a+k],optimize=True)

            t1_3[a:a+k] += 0.5*lib.einsum('lmdf,iaef,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('lmdf,iaef,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= 0.5*lib.einsum('mldf,iaef,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] += 0.5*lib.einsum('mldf,iaef,mlde->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3[a:a+k] += lib.einsum('lmdf,iaef,lmde->ia',t2_1,eris_ovvv,t2_1,optimize=True)
            t1_3[a:a+k] -= lib.einsum('lmef,iedf,lmad->ia',t2_1,eris_ovvv,t2_1,optimize=True)

            t1_3 += lib.einsum('ilde,lead->ia',t2_2[:,a:a+k],eris_ovvv,optimize=True)

            t1_3 -= lib.einsum('ildf,mefa,lmde->ia',t2_1,eris_ovvv, t2_1[:,a:a+k],optimize=True)
            t1_3 += lib.einsum('lidf,mefa,lmde->ia',t2_1,eris_ovvv, t2_1[:,a:a+k],optimize=True)

            t1_3 += lib.einsum('ilfd,mefa,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k] ,optimize=True)
            t1_3 -= lib.einsum('ilfd,mefa,mlde->ia',t2_1,eris_ovvv,t2_1[a:a+k] ,optimize=True)

            t1_3 += lib.einsum('ilaf,mefd,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)
            t1_3 -= lib.einsum('liaf,mefd,lmde->ia',t2_1,eris_ovvv,t2_1[:,a:a+k],optimize=True)

            del eris_ovvv
            a += k
        t1_3 += 0.25*lib.einsum('inde,lamn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.25*lib.einsum('inde,lamn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.25*lib.einsum('nide,lamn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.25*lib.einsum('nide,lamn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.25*lib.einsum('inde,maln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.25*lib.einsum('inde,maln,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.25*lib.einsum('nide,maln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.25*lib.einsum('nide,maln,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('inde,lamn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
 
        t1_3 += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('inad,lemn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('niad,lemn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('niad,lemn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('inad,meln,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('niad,meln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('niad,meln,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,lemn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5 * lib.einsum('niad,lemn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5 * lib.einsum('inad,meln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5 * lib.einsum('niad,meln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5 *lib.einsum('inad,lemn,lmed->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('inad,meln,mled->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 += 0.5*lib.einsum('inad,lemn,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('inad,lemn,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5*lib.einsum('inad,meln,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('inad,meln,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('lnde,ianm,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('nlde,ianm,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('nlde,ianm,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 += 0.5*lib.einsum('lnde,naim,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('lnde,naim,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('nlde,naim,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('nlde,naim,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= lib.einsum('nled,ianm,mled->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('nled,naim,mled->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= 0.5*lib.einsum('lnde,ianm,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('lnde,ianm,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += 0.5*lib.einsum('nlde,ianm,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= 0.5*lib.einsum('nlde,ianm,mlde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= lib.einsum('lnde,ianm,lmde->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= lib.einsum('lnde,ienm,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('lnde,ienm,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('nlde,ienm,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('nlde,ienm,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('nlde,neim,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('nlde,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 += lib.einsum('lnde,neim,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 += lib.einsum('nled,ienm,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 -= lib.einsum('nled,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('lned,ienm,lmad->ia',t2_1,eris_ovoo,t2_1,optimize=True)

        t1_3 -= lib.einsum('lnde,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        t1_3 += lib.einsum('nlde,neim,mlad->ia',t2_1,eris_ovoo,t2_1,optimize=True)
        
        # Frozen core
        t1_3[:nfc_orb,:] = 0

        t1_3 = t1_3/D1
      
    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    cput0 = log.timer_debug1("Completed amplitude calculation", *cput0)

    return t1, t2


def compute_energy(myadc, t1, t2, eris):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)
    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nocc = myadc._nocc
    nvir = myadc._nvir
    

    eris_ovov = eris.ovov

    t2_1  = t2[0]

    #Compute MP2 correlation energy

    e_mp2 = 0.5 * lib.einsum('ijab,iajb', t2_1, eris_ovov,optimize=True)
    e_mp2 -= 0.5 * lib.einsum('ijab,ibja', t2_1, eris_ovov,optimize=True)
    e_mp2 -= 0.5 * lib.einsum('jiab,iajb', t2_1, eris_ovov,optimize=True)
    e_mp2 += 0.5 * lib.einsum('jiab,ibja', t2_1, eris_ovov,optimize=True)
    e_mp2 += lib.einsum('ijab,iajb', t2_1, eris_ovov,optimize=True)

    e_corr = e_mp2

    if (myadc.method == "adc(3)"):

        #Compute MP3 correlation energy
        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo
        eris_oooo = eris.oooo

        temp_t2_a = None
        temp_t2_ab = None
        temp_t2_t2 = None
        temp_t2a_t2a = None
        temp_t2_a_vvvv = None
        temp_t2_ab_vvvv = None

        eris_vvvv = eris.vvvv

        if isinstance(eris.vvvv, np.ndarray):
            temp_t2 = t2_1.reshape(nocc*nocc,nvir*nvir)
            temp_t2_vvvv = np.dot(temp_t2,eris_vvvv.T).reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list): 
            temp_t2_vvvv = contract_ladder(myadc,t2_1,eris.vvvv)
        else : 
            temp_t2_vvvv = contract_ladder(myadc,t2_1,eris.Lvv)

        e_mp3 =  lib.einsum('ijcd,ijcd',temp_t2_vvvv, t2_1,optimize=True)

        e_mp3 += 0.25 * lib.einsum('ijcd,ijcd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 -= 0.25 * lib.einsum('ijdc,ijcd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 -= 0.25 * lib.einsum('ijcd,jicd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 += 0.25 * lib.einsum('ijdc,jicd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 -= 0.25 * lib.einsum('jicd,ijcd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 += 0.25 * lib.einsum('jidc,ijcd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 += 0.25 * lib.einsum('jicd,jicd',temp_t2_vvvv, t2_1,optimize=True)
        e_mp3 -= 0.25 * lib.einsum('jidc,jicd',temp_t2_vvvv, t2_1,optimize=True)
        del temp_t2_vvvv       

        temp_t2_t2  =  lib.einsum('ijab,klab', t2_1, t2_1,optimize=True)
        temp_t2_t2 -=  lib.einsum('ijab,lkab', t2_1, t2_1,optimize=True)
        temp_t2_t2 -=  lib.einsum('jiab,klab', t2_1, t2_1,optimize=True)
        temp_t2_t2 +=  lib.einsum('jiab,lkab', t2_1, t2_1,optimize=True)
        e_mp3 += 0.25 * lib.einsum('ijkl,ikjl',temp_t2_t2, eris_oooo,optimize=True)
        e_mp3 -= 0.25 * lib.einsum('ijkl,iljk',temp_t2_t2, eris_oooo,optimize=True)
        del temp_t2_t2

        temp_t2_t2 =  lib.einsum('ijab,klab', t2_1, t2_1,optimize=True)
        e_mp3 +=  lib.einsum('ijkl,ikjl',temp_t2_t2, eris_oooo,optimize=True)
        del temp_t2_t2

        temp_t2_t2  = lib.einsum('ijab,ikcb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 -= lib.einsum('ijab,kicb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 -= lib.einsum('jiab,ikcb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 += lib.einsum('jiab,kicb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 += lib.einsum('jiab,kicb->akcj', t2_1, t2_1,optimize=True)
        e_mp3 -= 2 * lib.einsum('akcj,kjac',temp_t2_t2, eris_oovv,optimize=True)
        e_mp3 += 2 * lib.einsum('akcj,kcaj',temp_t2_t2, eris_ovvo,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = lib.einsum('ijab,ikcb->akcj', t2_1, t2_1,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_t2_t2, eris_oovv,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = lib.einsum('jiba,kibc->akcj', t2_1, t2_1,optimize=True)
        e_mp3 -= lib.einsum('akcj,kjac',temp_t2_t2, eris_oovv,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = -lib.einsum('ijab,ikbc->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 += lib.einsum('jiab,ikbc->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 -= lib.einsum('jiab,ikcb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 += lib.einsum('jiab,kicb->akcj', t2_1, t2_1,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_t2_t2, eris_ovvo,optimize=True)
        del temp_t2_t2

        temp_t2_t2 = -lib.einsum('ijba,ikcb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 += lib.einsum('ijba,kicb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 -= lib.einsum('ijab,kicb->akcj', t2_1, t2_1,optimize=True)
        temp_t2_t2 += lib.einsum('jiab,kicb->akcj', t2_1, t2_1,optimize=True)
        e_mp3 += lib.einsum('akcj,kcaj',temp_t2_t2, eris_ovvo,optimize=True)
        del temp_t2_t2
    
        e_corr += e_mp3
      
    cput0 = log.timer_debug1("Completed mp energy calculation", *cput0)

    log.timer_debug1("Completed mpn energy calculation")

    return e_corr

def contract_ladder(myadc,t_amp,vvvv):

    log = logger.Logger(myadc.stdout, myadc.verbose)
    nocc = myadc._nocc
    nvir = myadc._nvir

    t_amp_t = np.ascontiguousarray(t_amp.reshape(nocc*nocc,nvir*nvir).T)
    t = np.zeros((nvir,nvir, nocc*nocc))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
             k = dataset.shape[0]
             dataset = dataset[:].reshape(-1,nvir*nvir)
             t[a:a+k] = np.dot(dataset,t_amp_t).reshape(-1,nvir,nocc*nocc)
             a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            t[a:a+k] = np.dot(vvvv_p,t_amp_t).reshape(-1,nvir,nocc*nocc)
            del (vvvv_p)
            a += k
            #print("Buffer number", a, flush=True)
    else :
        raise Exception("Unknown vvvv type") 

    del (t_amp_t)
    t = np.ascontiguousarray(t.transpose(2,0,1)).reshape(nocc, nocc, nvir, nvir)

    return t

def cvs_projector(myadc, r, alpha_proj=0):
    
    ncore_proj = myadc.ncore_proj
    
    #if alpha_proj != 0:
    #    alpha_proj = myadc.alpha_proj 
    
    nocc = myadc._nocc
    nvir = myadc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc
    
    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles 
    
    Pr = r.copy()
    
    Pr[ncore_proj:f1] *= alpha_proj    
    
    temp = np.zeros((nvir, nocc, nocc))
    temp = Pr[s2:f2].reshape((nvir, nocc, nocc)).copy()
    
    temp[:,ncore_proj:,ncore_proj:] *= alpha_proj
    #temp[:,:ncore_proj,:ncore_proj] *= alpha_proj
    
    Pr[s2:f2] = temp.reshape(-1).copy()
    
    return Pr

def cvs_proj_valence(myadc, r):
    
    ncore_proj = myadc.ncore_proj_valence
    
    nocc = myadc._nocc
    nvir = myadc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc
    
    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles 
    
    Pr = r.copy()
    
    Pr[s1:ncore_proj] = 0    
    
    temp = np.zeros((nvir, nocc, nocc))
    temp = Pr[s2:f2].reshape((nvir, nocc, nocc)).copy()
    
    temp[:,:ncore_proj:,:ncore_proj] = 0
    temp[:,:ncore_proj,ncore_proj:] = 0
    temp[:,ncore_proj:,:ncore_proj] = 0
    
    Pr[s2:f2] = temp.reshape(-1).copy()
    
    return Pr

def density_matrix(myadc, T=None):

    if T is None:
        T = RADCIP(myadc).get_trans_moments()

    nocc = myadc._nocc
    nvir = myadc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc
    ij_ind = np.tril_indices(nocc, k=-1)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T_doubles = T[:,n_singles:]
    T_doubles = T_doubles.reshape(-1,nvir,nocc,nocc)
    T_doubles_transpose = T_doubles.transpose(0,1,3,2).copy()
    T_bab = (2/3)*T_doubles + (1/3)*T_doubles_transpose

    T_aaa = T_bab - T_bab.transpose(0,1,3,2)

    T_a = T[:,s1:f1]
    T_bab = T_bab.reshape(-1,n_doubles)
    T_aaa = T_aaa.reshape(-1,n_doubles)

    dm = 2 * np.dot(T_a,T_a.T) + np.dot(T_aaa, T_aaa.T) + 2 * np.dot(T_bab, T_bab.T)

    return dm


class RADC(lib.StreamObject):
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
            >>> myadc = adc.RADC(mf).run()

    Saved results

        e_corr : float
            MPn correlation correction
        e_tot : float
            Total energy (HF + correlation)
        t1, t2 :
            T amplitudes t1[i,a], t2[i,j,a,b]  (i,j in occ, a,b in virt)
    '''
    incore_complete = getattr(__config__, 'adc_radc_RADC_incore_complete', False)
    async_io = getattr(__config__, 'adc_radc_RADC_async_io', True)
    blkmin = getattr(__config__, 'adc_radc_RADC_blkmin', 4)
    memorymin = getattr(__config__, 'adc_radc_RADC_memorymin', 2000)
    
    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
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

        self.max_space = getattr(__config__, 'adc_radc_RADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_radc_RADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_radc_RADC_conv_tol', 1e-12)
        self.scf_energy = mf.e_tot
        
        self.frozen = frozen
        self.incore_complete = self.incore_complete or self.mol.incore_anyway
        
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self._nocc = mf.mol.nelectron//2
        self._nmo = mo_coeff.shape[1]
        self._nvir = self._nmo - self._nocc
        self.mo_energy = mf.mo_energy
        self.chkfile = mf.chkfile
        self.method = "adc(2)"
        self.method_type = "ip"
        self.with_df = None
        self.compute_mpn_energy = True
        self.nfc_orb = 0 
        self.nkop_chk = False
        self.cvs_npick = False
        self.kop_npick = False
        self.fc_bool = True
        self.ncore_proj = 0
        self.ncore_proj_valence = 0
        self.alpha_proj = 0
        self.mom_skd_iter = []
        keys = set(('conv_tol', 'e_corr', 'method', 'mo_coeff', 'mol', 'mo_energy', 'max_memory', 'incore_complete', 'scf_energy', 'e_tot', 't1', 'frozen', 'chkfile', 'max_space', 't2', 'mo_occ', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)
    
    compute_amplitudes = compute_amplitudes
    compute_energy = compute_energy
    transform_integrals = radc_ao2mo.transform_integrals_incore
    make_rdm1 = density_matrix 
    
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
    
    def kernel_gs(self,adc):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)
    
        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)
    
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()
    
        nmo = self._nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
           if getattr(self, 'with_df', None): 
               self.with_df = self.with_df
           else :
               self.with_df = self._scf.with_df

           def df_transform():
               return radc_ao2mo.transform_integrals_df(self)
           self.transform_integrals = df_transform
        elif (self._scf._eri is None or
            (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
           def outcore_transform():
               return radc_ao2mo.transform_integrals_outcore(self)
           self.transform_integrals = outcore_transform

        eris = self.transform_integrals()
        
        if self.compute_mpn_energy == True:
            self.e_corr, self.t1, self.t2 = compute_amplitudes_energy(self, eris=eris,verbose=self.verbose, fc_bool=True)
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
    
        nmo = self._nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):  
           if getattr(self, 'with_df', None): 
               self.with_df = self.with_df
           else :
               self.with_df = self._scf.with_df

           def df_transform():
              return radc_ao2mo.transform_integrals_df(self)
           self.transform_integrals = df_transform
        elif (self._scf._eri is None or
            (mem_incore+mem_now >= self.max_memory and not self.incore_complete)):
           def outcore_transform():
               return radc_ao2mo.transform_integrals_outcore(self)
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
            e_exc, v_exc, spec_fac = self.ip_adc(nroots=nroots, guess=guess, eris=eris)

        else:
            raise NotImplementedError(self.method_type)

        return e_exc, v_exc, spec_fac

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        logger.note(self, 'E_corr = %.8f',
                    self.e_corr)
        return self
    
    def ea_adc(self, nroots=1, guess=None, eris=None):
        return RADCEA(self).kernel(nroots, guess, eris)
    
    def ip_adc(self, nroots=1, guess=None, eris=None):
        return RADCIP(self).kernel(nroots, guess, eris)

    def density_fit(self, auxbasis=None, with_df = None):
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

    t1_2 = t1[0]
    t2_1 = t2[0]

    nocc = adc._nocc
    nvir = adc._nvir

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov

    # a-b block
    # Zeroth-order terms

    M_ab = lib.einsum('ab,a->ab', idn_vir, e_vir)

   # Second-order terms

    M_ab +=  lib.einsum('l,lmad,lmbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab -=  lib.einsum('l,lmad,mlbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab -=  lib.einsum('l,mlad,lmbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,mlad,mlbd->ab',e_occ ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,lmad,lmbd->ab',e_occ,t2_1, t2_1,optimize=True)
    M_ab +=  lib.einsum('l,mlad,mlbd->ab',e_occ,t2_1, t2_1,optimize=True)

    M_ab -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.5 *  lib.einsum('d,lmad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.5 *  lib.einsum('d,mlad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.5 *  lib.einsum('d,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.25 *  lib.einsum('a,lmad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.25 *  lib.einsum('a,mlad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('a,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('a,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('a,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.25 *  lib.einsum('b,lmad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab += 0.25 *  lib.einsum('b,mlad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('b,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('b,lmad,lmbd->ab',e_vir,t2_1, t2_1,optimize=True)
    M_ab -= 0.25 *  lib.einsum('b,mlad,mlbd->ab',e_vir,t2_1, t2_1,optimize=True)

    M_ab -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_1, eris_ovov,optimize=True)
    M_ab += 0.5 *  lib.einsum('mlad,lbmd->ab',t2_1, eris_ovov,optimize=True)
    M_ab += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_1, eris_ovov,optimize=True)
    M_ab -= 0.5 *  lib.einsum('mlad,ldmb->ab',t2_1, eris_ovov,optimize=True)
    M_ab -=        lib.einsum('lmad,lbmd->ab',t2_1, eris_ovov,optimize=True)

    M_ab -= 0.5 *  lib.einsum('lmbd,lamd->ab',t2_1, eris_ovov,optimize=True)
    M_ab += 0.5 *  lib.einsum('mlbd,lamd->ab',t2_1, eris_ovov,optimize=True)
    M_ab += 0.5 *  lib.einsum('lmbd,ldma->ab',t2_1, eris_ovov,optimize=True)
    M_ab -= 0.5 *  lib.einsum('mlbd,ldma->ab',t2_1, eris_ovov,optimize=True)
    M_ab -=        lib.einsum('lmbd,lamd->ab',t2_1, eris_ovov,optimize=True)

    cput0 = log.timer_debug1("Completed M_ab second-order terms ADC(2) calculation", *cput0)

    #Third-order terms

    if(method =='adc(3)'):

        t2_2 = t2[1]

        eris_oovv = eris.oovv[:]
        eris_ovvo = eris.ovvo[:]
        eris_oooo = eris.oooo[:]

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc
        a = 0
        for p in range(0,nocc,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            k = eris_ovvv.shape[0]
            M_ab += 4. * lib.einsum('ld,ldab->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
            M_ab -=  lib.einsum('ld,lbad->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
            M_ab -= lib.einsum('ld,ladb->ab',t1_2[a:a+k], eris_ovvv,optimize=True)
            del eris_ovvv
            a += k

        cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation", *cput0)

        M_ab -= 0.5 *  lib.einsum('lmad,lbmd->ab',t2_2, eris_ovov,optimize=True)
        M_ab += 0.5 *  lib.einsum('mlad,lbmd->ab',t2_2, eris_ovov,optimize=True)
        M_ab += 0.5 *  lib.einsum('lmad,ldmb->ab',t2_2, eris_ovov,optimize=True)
        M_ab -= 0.5 *  lib.einsum('mlad,ldmb->ab',t2_2, eris_ovov,optimize=True)
        M_ab -=        lib.einsum('lmad,lbmd->ab',t2_2, eris_ovov,optimize=True)

        M_ab -= 0.5 * lib.einsum('lmbd,lamd->ab',t2_2,eris_ovov,optimize=True)
        M_ab += 0.5 * lib.einsum('mlbd,lamd->ab',t2_2,eris_ovov,optimize=True)
        M_ab += 0.5 * lib.einsum('lmbd,ldma->ab',t2_2, eris_ovov,optimize=True)
        M_ab -= 0.5 * lib.einsum('mlbd,ldma->ab',t2_2, eris_ovov,optimize=True)
        M_ab -=       lib.einsum('lmbd,lamd->ab',t2_2,eris_ovov,optimize=True)

        M_ab += lib.einsum('l,lmbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,lmbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,mlbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,lmbd,lmad->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlbd,mlad->ab',e_occ, t2_1, t2_2, optimize=True)

        M_ab += lib.einsum('l,lmad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,lmad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab -= lib.einsum('l,mlad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,lmad,lmbd->ab',e_occ, t2_1, t2_2, optimize=True)
        M_ab += lib.einsum('l,mlad,mlbd->ab',e_occ, t2_1, t2_2, optimize=True)

        M_ab -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,lmbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,mlbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,lmbd,lmad->ab', e_vir, t2_1 ,t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlbd,mlad->ab', e_vir, t2_1 ,t2_2, optimize=True)

        M_ab -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,lmad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.5*lib.einsum('d,mlad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,lmad,lmbd->ab', e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.5*lib.einsum('d,mlad,mlbd->ab', e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('a,lmbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('a,mlbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,mlbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,lmbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,mlbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('a,lmad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('a,mlad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,mlad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,lmad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('a,mlad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('b,lmbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('b,mlbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,mlbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,lmbd,lmad->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,mlbd,mlad->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('b,lmad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab += 0.25*lib.einsum('b,mlad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,mlad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,lmad,lmbd->ab',e_vir, t2_1, t2_2, optimize=True)
        M_ab -= 0.25*lib.einsum('b,mlad,mlbd->ab',e_vir, t2_1, t2_2, optimize=True)

        M_ab -= lib.einsum('lned,mlbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('lned,lmbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('nled,mlbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('nled,lmbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab += lib.einsum('lned,mlbd,mane->ab',t2_1, t2_1, eris_ovov, optimize=True)
        M_ab -= lib.einsum('lned,lmbd,mane->ab',t2_1, t2_1, eris_ovov, optimize=True)
        M_ab -= lib.einsum('nled,mlbd,mane->ab',t2_1, t2_1, eris_ovov, optimize=True)
        M_ab += lib.einsum('nled,lmbd,mane->ab',t2_1, t2_1, eris_ovov, optimize=True)

        M_ab += lib.einsum('nled,mlbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('nled,mlbd,mane->ab',t2_1, t2_1, eris_ovov, optimize=True)

        M_ab -= lib.einsum('lnde,mlbd,neam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lnde,lmbd,neam->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab += lib.einsum('lned,mlbd,neam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('nled,mlbd,neam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lned,lmbd,nmae->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= lib.einsum('mled,lnad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('mled,nlad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('lmed,lnad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('lmed,nlad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab += lib.einsum('mled,lnad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('mled,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('lmed,lnad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmed,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab += lib.einsum('mled,nlad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('mled,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmed,lnad,nmeb->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= lib.einsum('mled,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmed,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab += lib.einsum('lmde,lnad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab -= lib.einsum('lmde,nlad,nbem->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab -= lib.einsum('mlbd,lnae,nmde->ab',t2_1, t2_1,   eris_oovv, optimize=True)
        M_ab += lib.einsum('mlbd,nlae,nmde->ab',t2_1, t2_1,   eris_oovv, optimize=True)
        M_ab += lib.einsum('lmbd,lnae,nmde->ab',t2_1, t2_1,   eris_oovv, optimize=True)
        M_ab -= lib.einsum('lmbd,nlae,nmde->ab',t2_1, t2_1,   eris_oovv, optimize=True)

        M_ab += lib.einsum('mlbd,lnae,nedm->ab',t2_1, t2_1,   eris_ovvo, optimize=True)
        M_ab -= lib.einsum('mlbd,nlae,nedm->ab',t2_1, t2_1,   eris_ovvo, optimize=True)
        M_ab -= lib.einsum('lmbd,lnae,nedm->ab',t2_1, t2_1,   eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmbd,nlae,nedm->ab',t2_1, t2_1,   eris_ovvo, optimize=True)

        M_ab += lib.einsum('lmbd,lnae,nmde->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= lib.einsum('lmbd,lnae,nedm->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab += lib.einsum('mlbd,lnae,nedm->ab',t2_1, t2_1,  eris_ovvo, optimize=True)
        M_ab -= lib.einsum('lmbd,lnae,nedm->ab',t2_1, t2_1,  eris_ovvo, optimize=True)

        M_ab -= lib.einsum('lmbd,lnae,nedm->ab',t2_1, t2_1,  eris_ovvo, optimize=True)
        M_ab += lib.einsum('lmbd,nlae,nedm->ab',t2_1, t2_1,  eris_ovvo, optimize=True)

        M_ab += lib.einsum('mlbd,nlae,nmde->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= 0.5*lib.einsum('lned,lmed,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= 0.5*lib.einsum('nled,mled,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += 0.5*lib.einsum('nled,lmed,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= 0.5*lib.einsum('lned,mled,nbam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += 0.5*lib.einsum('lned,lmed,nbam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab += 0.5*lib.einsum('nled,mled,nbam->ab',t2_1, t2_1, eris_ovvo, optimize=True)
        M_ab -= 0.5*lib.einsum('nled,lmed,nbam->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab -= lib.einsum('nled,mled,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += lib.einsum('nled,mled,nbam->ab',t2_1, t2_1, eris_ovvo, optimize=True)

        M_ab += 0.5*lib.einsum('lned,mled,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= 0.5*lib.einsum('lned,lmed,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab -= 0.5*lib.einsum('nled,mled,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)
        M_ab += 0.5*lib.einsum('nled,lmed,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= lib.einsum('lned,lmed,nmab->ab',t2_1, t2_1, eris_oovv, optimize=True)

        M_ab -= 0.25*lib.einsum('mlbd,noad,nmol->ab',t2_1, t2_1, eris_oooo, optimize=True)
        M_ab += 0.25*lib.einsum('mlbd,onad,nmol->ab',t2_1, t2_1, eris_oooo, optimize=True)
        M_ab += 0.25*lib.einsum('lmbd,noad,nmol->ab',t2_1, t2_1, eris_oooo, optimize=True)
        M_ab -= 0.25*lib.einsum('lmbd,onad,nmol->ab',t2_1, t2_1, eris_oooo, optimize=True)

        M_ab += 0.25*lib.einsum('mlbd,noad,nlom->ab',t2_1, t2_1, eris_oooo, optimize=True)
        M_ab -= 0.25*lib.einsum('mlbd,onad,nlom->ab',t2_1, t2_1, eris_oooo, optimize=True)
        M_ab -= 0.25*lib.einsum('lmbd,noad,nlom->ab',t2_1, t2_1, eris_oooo, optimize=True)
        M_ab += 0.25*lib.einsum('lmbd,onad,nlom->ab',t2_1, t2_1, eris_oooo, optimize=True)

        M_ab -= lib.einsum('mlbd,noad,nmol->ab',t2_1, t2_1, eris_oooo, optimize=True)

        log.timer_debug1("Completed M_ab ADC(3) small integrals calculation")

        log.timer_debug1("Starting M_ab vvvv ADC(3) calculation")

        if isinstance(eris.vvvv, np.ndarray):
            t2_1_r = t2_1.reshape(nocc*nocc,nvir*nvir)
            eris_vvvv = eris.vvvv
            temp_t2 = np.dot(t2_1_r,eris_vvvv)
            temp_t2 = temp_t2.reshape(nocc,nocc,nvir,nvir)
 
            M_ab -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,lmbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,mlbf->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,lmbf->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.25*lib.einsum('mlaf,lmfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,mlfb->ab',t2_1, temp_t2, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,lmfb->ab',t2_1, temp_t2, optimize=True)
            M_ab -= lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2, optimize=True)

            temp_vvvv_t2 = np.dot(eris_vvvv,t2_1_r.T)
            temp_vvvv_t2 = temp_vvvv_t2.reshape(nvir,nvir,nocc,nocc)
            M_ab += 0.25*lib.einsum('adlm,mlbd->ab',temp_vvvv_t2, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('adlm,lmbd->ab',temp_vvvv_t2, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('adml,mlbd->ab',temp_vvvv_t2, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('adml,lmbd->ab',temp_vvvv_t2, t2_1, optimize=True)

            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab', temp_t2, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('lmad,mlbd->ab', temp_t2, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab', temp_t2, t2_1, optimize=True)
            M_ab -= lib.einsum('mlad,mlbd->ab', temp_t2, t2_1, optimize=True)

            eris_vvvv = eris_vvvv.reshape(nvir,nvir,nvir,nvir)
            M_ab -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            M_ab -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
            eris_vvvv = eris_vvvv.reshape(nvir*nvir,nvir*nvir)

        else:
            if isinstance(eris.vvvv, list):
                temp_t2_vvvv = contract_ladder(adc,t2_1,eris.vvvv)
            else :
                temp_t2_vvvv = contract_ladder(adc,t2_1,eris.Lvv)

            M_ab -= 0.25*lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('mlaf,lmbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,lmbf->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab += 0.25*lib.einsum('mlaf,mlfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.25*lib.einsum('mlaf,lmfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab -= 0.25*lib.einsum('lmaf,mlfb->ab',t2_1, temp_t2_vvvv, optimize=True)
            M_ab += 0.25*lib.einsum('lmaf,lmfb->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab -= lib.einsum('mlaf,mlbf->ab',t2_1, temp_t2_vvvv, optimize=True)

            M_ab += 0.25*lib.einsum('lmad,mlbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab',temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab',temp_t2_vvvv, t2_1, optimize=True)

            M_ab -= 0.25*lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('mlad,lmbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab += 0.25*lib.einsum('lmad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= 0.25*lib.einsum('lmad,lmbd->ab', temp_t2_vvvv, t2_1, optimize=True)
            M_ab -= lib.einsum('mlad,mlbd->ab', temp_t2_vvvv, t2_1, optimize=True)
          
            del (temp_t2_vvvv)

            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0 
            temp = np.zeros((nvir,nvir))

            if isinstance(eris.vvvv, list):
                for dataset in eris.vvvv:
                    k = dataset.shape[0]
                    eris_vvvv = dataset[:].reshape(-1,nvir,nvir,nvir)
                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1,  eris_vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, eris_vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, eris_vvvv, optimize=True)
                    del eris_vvvv
                    a += k
            else :

                for p in range(0,nvir,chnk_size):

                    vvvv = dfadc.get_vvvv_df(adc, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                    k = vvvv.shape[0]

                    temp[a:a+k] -= lib.einsum('mldf,mled,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('mldf,lmed,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += lib.einsum('lmdf,mled,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('lmdf,lmed,aebf->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('mldf,mled,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('mldf,lmed,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] -= 0.5*lib.einsum('lmdf,mled,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 0.5*lib.einsum('lmdf,lmed,aefb->ab',t2_1, t2_1,  vvvv, optimize=True)
                    temp[a:a+k] += 2.*lib.einsum('mlfd,mled,aebf->ab',t2_1, t2_1, vvvv, optimize=True)
                    temp[a:a+k] -= lib.einsum('mlfd,mled,aefb->ab',t2_1, t2_1, vvvv, optimize=True)
                    del (vvvv)
                    a += k

            M_ab += temp
            del (temp)

    cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)
        
    return M_ab


def get_imds_ip(adc, eris=None, fc_bool=True):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    t1, t2 = adc.compute_amplitudes(eris, fc_bool)   
    #t1 = adc.t1
    #t2 = adc.t2

    t1_2 = t1[0]
    t2_1 = t2[0]

    nocc = adc._nocc
    nvir = adc._nvir

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    eris_ovov = eris.ovov

    # i-j block
    # Zeroth-order terms
    
    """FC = True
    nfc_orb = 2
    if FC is True:
        e_occ = e_occ[nfc_orb:]
        idn_occ = np.identity(nocc-nfc_orb)"""

    M_ij = lib.einsum('ij,j->ij', idn_occ ,e_occ)

    # Second-order terms

    M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij -=  lib.einsum('d,ilde,ljde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij -=  lib.einsum('d,lide,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,lide,ljde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_1, optimize=True)
    M_ij +=  lib.einsum('d,iled,jled->ij',e_vir,t2_1, t2_1, optimize=True)

    M_ij -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)

    M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.25 *  lib.einsum('i,ilde,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.25 *  lib.einsum('i,lide,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('i,lide,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)

    M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.25 *  lib.einsum('j,ilde,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij += 0.25 *  lib.einsum('j,lide,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('j,lide,ljde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_1, optimize=True)
    
    """if FC is True:
        M_ivjv_FC = eris_ovov[:nfc_orb,:,:nfc_orb,:].copy()
        eris_ovov = eris_ovov[nfc_orb:,:,nfc_orb:,:].copy()"""
        
    M_ij += 0.5 *  lib.einsum('ilde,jdle->ij',t2_1, eris_ovov,optimize=True)
    M_ij -= 0.5 *  lib.einsum('lide,jdle->ij',t2_1, eris_ovov,optimize=True)
    M_ij -= 0.5 *  lib.einsum('ilde,jeld->ij',t2_1, eris_ovov,optimize=True)
    M_ij += 0.5 *  lib.einsum('lide,jeld->ij',t2_1, eris_ovov,optimize=True)
    M_ij += lib.einsum('ilde,jdle->ij',t2_1, eris_ovov,optimize=True)

    M_ij += 0.5 *  lib.einsum('jlde,idle->ij',t2_1, eris_ovov,optimize=True)
    M_ij -= 0.5 *  lib.einsum('ljde,idle->ij',t2_1, eris_ovov,optimize=True)
    M_ij -= 0.5 *  lib.einsum('jlde,ldie->ij',t2_1, eris_ovov,optimize=True)
    M_ij += 0.5 *  lib.einsum('ljde,ldie->ij',t2_1, eris_ovov,optimize=True)
    M_ij += lib.einsum('jlde,idle->ij',t2_1, eris_ovov,optimize=True)
    
    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    # Third-order terms

    if (method == "adc(3)"):

        t2_2 = t2[1]

        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo
        eris_ovoo = eris.ovoo
        eris_oooo = eris.oooo

        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,jdli->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldji->ij',t1_2, eris_ovoo,optimize=True)

        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)
        M_ij -= lib.einsum('ld,idlj->ij',t1_2, eris_ovoo,optimize=True)
        M_ij += lib.einsum('ld,ldij->ij',t1_2, eris_ovoo,optimize=True)

        M_ij += 0.5* lib.einsum('ilde,jdle->ij',t2_2, eris_ovov,optimize=True)
        M_ij -= 0.5* lib.einsum('lide,jdle->ij',t2_2, eris_ovov,optimize=True)
        M_ij -= 0.5* lib.einsum('ilde,jeld->ij',t2_2, eris_ovov,optimize=True)
        M_ij += 0.5* lib.einsum('lide,jeld->ij',t2_2, eris_ovov,optimize=True)
        M_ij += lib.einsum('ilde,jdle->ij',t2_2, eris_ovov,optimize=True)

        M_ij += 0.5* lib.einsum('jlde,leid->ij',t2_2, eris_ovov,optimize=True)
        M_ij -= 0.5* lib.einsum('ljde,leid->ij',t2_2, eris_ovov,optimize=True)
        M_ij -= 0.5* lib.einsum('jlde,ield->ij',t2_2, eris_ovov,optimize=True)
        M_ij += 0.5* lib.einsum('ljde,ield->ij',t2_2, eris_ovov,optimize=True)
        M_ij += lib.einsum('jlde,leid->ij',t2_2, eris_ovov,optimize=True)

        M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,ilde,ljde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,lide,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,lide,ljde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,ilde,jlde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,iled,jled->ij',e_vir,t2_1, t2_2,optimize=True)

        M_ij +=  lib.einsum('d,jlde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,jlde,lide->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij -=  lib.einsum('d,ljde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,ljde,lide->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,jlde,ilde->ij',e_vir,t2_1, t2_2,optimize=True)
        M_ij +=  lib.einsum('d,jled,iled->ij',e_vir,t2_1, t2_2,optimize=True)

        M_ij -= 0.5 *  lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,ilde,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,lide,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5 *  lib.einsum('l,lide,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.5 *  lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,jlde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.5 *  lib.einsum('l,ljde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5 *  lib.einsum('l,ljde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.5*lib.einsum('l,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('i,ilde,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('i,lide,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,lide,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('i,jlde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('i,ljde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,ljde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('i,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('j,jlde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('j,ljde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,ljde,lide->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,jlde,ilde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('j,ilde,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij += 0.25 *  lib.einsum('j,lide,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,lide,ljde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)
        M_ij -= 0.25 *  lib.einsum('j,ilde,jlde->ij',e_occ,t2_1, t2_2,optimize=True)

        M_ij -= lib.einsum('lmde,jldf,mefi->ij',t2_1, t2_1, eris_ovvo,optimize = True)
        M_ij += lib.einsum('lmde,ljdf,mefi->ij',t2_1, t2_1, eris_ovvo,optimize = True)
        M_ij += lib.einsum('mlde,jldf,mefi->ij',t2_1, t2_1, eris_ovvo,optimize = True)
        M_ij -= lib.einsum('mlde,ljdf,mefi->ij',t2_1, t2_1, eris_ovvo,optimize = True)

        M_ij += lib.einsum('lmde,jldf,mife->ij',t2_1, t2_1, eris_oovv,optimize = True)
        M_ij -= lib.einsum('lmde,ljdf,mife->ij',t2_1, t2_1, eris_oovv,optimize = True)
        M_ij -= lib.einsum('mlde,jldf,mife->ij',t2_1, t2_1, eris_oovv,optimize = True)
        M_ij += lib.einsum('mlde,ljdf,mife->ij',t2_1, t2_1, eris_oovv,optimize = True)

        M_ij += lib.einsum('mled,jlfd,mefi->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mled,jlfd,mife->ij',t2_1, t2_1, eris_oovv ,optimize = True)

        M_ij -= lib.einsum('lmde,jldf,mefi->ij',t2_1, t2_1, eris_ovvo,optimize = True)
        M_ij += lib.einsum('lmde,ljdf,mefi->ij',t2_1, t2_1, eris_ovvo,optimize = True)

        M_ij -= lib.einsum('mlde,jldf,mife->ij',t2_1, t2_1, eris_oovv ,optimize = True)

        M_ij += lib.einsum('lmde,jlfd,mefi->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mlde,jlfd,mefi->ij',t2_1, t2_1, eris_ovvo ,optimize = True)

        M_ij -= lib.einsum('lmde,ildf,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij += lib.einsum('lmde,lidf,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij += lib.einsum('mlde,ildf,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mlde,lidf,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)

        M_ij += lib.einsum('lmde,ildf,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij -= lib.einsum('lmde,lidf,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij -= lib.einsum('mlde,ildf,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij += lib.einsum('mlde,lidf,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)

        M_ij += lib.einsum('mled,ilfd,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mled,ilfd,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij -= lib.einsum('lmde,ildf,mefj->ij',t2_1, t2_1, eris_ovvo,optimize = True)
        M_ij += lib.einsum('lmde,lidf,mefj->ij',t2_1, t2_1, eris_ovvo,optimize = True)
        M_ij -= lib.einsum('mlde,ildf,mjfe->ij',t2_1, t2_1, eris_oovv ,optimize = True)
        M_ij += lib.einsum('lmde,ilfd,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)
        M_ij -= lib.einsum('mlde,ilfd,mefj->ij',t2_1, t2_1, eris_ovvo ,optimize = True)

        M_ij += 0.25*lib.einsum('lmde,jnde,limn->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('lmde,njde,limn->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('mlde,jnde,limn->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij += 0.25*lib.einsum('mlde,njde,limn->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('lmde,jnde,lnmi->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij += 0.25*lib.einsum('lmde,njde,lnmi->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij += 0.25*lib.einsum('mlde,jnde,lnmi->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('mlde,njde,lnmi->ij',t2_1, t2_1,eris_oooo, optimize = True)
        M_ij += lib.einsum('lmde,jnde,limn->ij',t2_1 ,t2_1, eris_oooo, optimize = True)

        if isinstance(eris.vvvv, np.ndarray):
            eris_vvvv = eris.vvvv
            t2_1_r = t2_1.reshape(nocc*nocc,nvir*nvir)
            temp_t2_vvvv = np.dot(t2_1_r,eris_vvvv)
            temp_t2_vvvv = temp_t2_vvvv.reshape(nocc,nocc,nvir,nvir)
        elif isinstance(eris.vvvv, list):
            temp_t2_vvvv = contract_ladder(adc,t2_1,eris.vvvv)
        else :
            temp_t2_vvvv = contract_ladder(adc,t2_1,eris.Lvv)

        M_ij += 0.25*lib.einsum('ilde,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('ilde,ljde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('lide,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij += 0.25*lib.einsum('lide,ljde->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('ilde,jled->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij += 0.25*lib.einsum('ilde,ljed->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij += 0.25*lib.einsum('lide,jled->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij -= 0.25*lib.einsum('lide,ljed->ij',t2_1, temp_t2_vvvv, optimize = True)
        M_ij +=lib.einsum('ilde,jlde->ij',t2_1, temp_t2_vvvv, optimize = True)

        M_ij += 0.25*lib.einsum('inde,lmde,jlnm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('inde,mlde,jlnm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('nide,lmde,jlnm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.25*lib.einsum('nide,mlde,jlnm->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij -= 0.25*lib.einsum('inde,lmde,jmnl->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.25*lib.einsum('inde,mlde,jmnl->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.25*lib.einsum('nide,lmde,jmnl->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.25*lib.einsum('nide,mlde,jmnl->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij +=lib.einsum('inde,lmde,jlnm->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij += 0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij -= 0.5*lib.einsum('lmdf,mlde,jief->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij -= 0.5*lib.einsum('mldf,lmde,jief->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij += 0.5*lib.einsum('mldf,mlde,jief->ij',t2_1, t2_1, eris_oovv, optimize = True)

        M_ij -= 0.5*lib.einsum('lmdf,lmde,jfei->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij += 0.5*lib.einsum('lmdf,mlde,jfei->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij += 0.5*lib.einsum('mldf,lmde,jfei->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= 0.5*lib.einsum('mldf,mlde,jfei->ij',t2_1, t2_1, eris_ovvo, optimize = True)

        M_ij +=lib.einsum('mlfd,mled,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)
        M_ij -=lib.einsum('mlfd,mled,jfei->ij',t2_1, t2_1, eris_ovvo , optimize = True)
        M_ij +=lib.einsum('lmdf,lmde,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)

        M_ij +=0.5*lib.einsum('lmdf,lmde,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)
        M_ij -=0.5*lib.einsum('lmdf,mlde,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)
        M_ij -=0.5*lib.einsum('mldf,lmde,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)
        M_ij +=0.5*lib.einsum('mldf,mlde,jief->ij',t2_1, t2_1, eris_oovv , optimize = True)

        M_ij -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij += lib.einsum('ilde,mjdf,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij += lib.einsum('lide,jmdf,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij -= lib.einsum('lide,mjdf,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)

        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('ilde,mjdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('lide,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij += lib.einsum('lide,mjdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)

        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('lide,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)

        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('ilde,mjdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)

        M_ij -= lib.einsum('ilde,jmdf,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)
        M_ij += lib.einsum('ilde,jmdf,lefm->ij',t2_1, t2_1, eris_ovvo, optimize = True)
        M_ij -= lib.einsum('iled,jmfd,lmfe->ij',t2_1, t2_1, eris_oovv, optimize = True)

        M_ij -= 0.5*lib.einsum('lnde,lmde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.5*lib.einsum('lnde,mlde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.5*lib.einsum('nlde,lmde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.5*lib.einsum('nlde,mlde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij += 0.5*lib.einsum('lnde,lmde,jmni->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.5*lib.einsum('lnde,mlde,jmni->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.5*lib.einsum('nlde,lmde,jmni->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.5*lib.einsum('nlde,mlde,jmni->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij -= lib.einsum('nled,mled,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += lib.einsum('nled,mled,jmni->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= lib.einsum('lnde,lmde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)

        M_ij -= 0.5 * lib.einsum('lnde,lmde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.5 * lib.einsum('lnde,mlde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij += 0.5 * lib.einsum('nlde,lmde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)
        M_ij -= 0.5 * lib.einsum('nlde,mlde,jinm->ij',t2_1, t2_1, eris_oooo, optimize = True)

    cput0 = log.timer_debug1("Completed M_ij ADC(n) calculation", *cput0)
    return M_ij


def ea_adc_diag(adc,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ab is None:
        M_ab = adc.get_imds()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ab = e_vir[:,None] + e_vir
    d_i = e_occ[:,None]
    D_n = -d_i + d_ab.reshape(-1)
    D_iab = D_n.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in p1-p1 block

    M_ab_diag = np.diagonal(M_ab)

    diag[s1:f1] = M_ab_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] = D_iab
    del (D_iab)

    ###### Additional terms for the preconditioner ####

    if (method == "adc(2)-x" or method == "adc(3)"):

        if eris is None:
            eris = adc.transform_integrals()

            if isinstance(eris.vvvv, np.ndarray):

                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo
                eris_vvvv = eris.vvvv

                temp = np.zeros((nocc, eris_vvvv.shape[0]))
                temp[:] += np.diag(eris_vvvv)
                diag[s2:f2] += temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nvir, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(1,0,2))
                diag[s2:f2] += -temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv[:].transpose(0,2,1,3))
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nvir, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(1,2,0))
                diag[s2:f2] += -temp.reshape(-1)
                
    log.timer_debug1("Completed ea_diag calculation")
    return diag


def ip_adc_diag(adc,M_ij=None,eris=None,cvs=True, fc_bool=True, mom_skd=False, alpha_proj=0):
   
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    if M_ij is None:
        M_ij = adc.get_imds(fc_bool)

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    diag = np.zeros(dim)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij)

    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    diag[s2:f2] = D_aij.copy()

    ###### Additional terms for the preconditioner ####
    """
    if (method == "adc(2)-x" or method == "adc(3)"):

        if eris is None:
            eris = adc.transform_integrals()

            if isinstance(eris.vvvv, np.ndarray):

                eris_oooo = eris.oooo
                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo

                eris_oooo_p = np.ascontiguousarray(eris_oooo.transpose(0,2,1,3))
                eris_oooo_p = eris_oooo_p.reshape(nocc*nocc, nocc*nocc)
  
                temp = np.zeros((nvir, eris_oooo_p.shape[0]))
                temp[:] += np.diag(eris_oooo_p)
                diag[s2:f2] += -temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3)) 
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nocc, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(2,1,0))
                diag[s2:f2] += temp.reshape(-1)

                eris_ovov_p = np.ascontiguousarray(eris_oovv.transpose(0,2,1,3)) 
                eris_ovov_p = eris_ovov_p.reshape(nocc*nvir, nocc*nvir)

                temp = np.zeros((nocc, nocc, nvir))
                temp[:] += np.diagonal(eris_ovov_p).reshape(nocc, nvir)
                temp = np.ascontiguousarray(temp.transpose(2,0,1))
                diag[s2:f2] += temp.reshape(-1)
    """
    if (cvs is True) or ((mom_skd is True) and (alpha_proj==0)):

        shift = -100000.0
        ncore_proj = adc.ncore_proj
        diag[ncore_proj:f1] += shift

        temp = np.zeros((nvir,nocc,nocc))
        temp[:,ncore_proj:,ncore_proj:] += shift
        #temp[:,:ncore_proj,:ncore_proj] += shift

        diag[s2:f2] += temp.reshape(-1).copy()

    diag = -diag

    return diag

def ea_contract_r_vvvv(myadc,r2,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    r2_vvvv = np.zeros((nocc,nvir,nvir))
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
             k = dataset.shape[0]
             dataset = dataset[:].reshape(-1,nvir*nvir)
             r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
             del (dataset)
             a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc,-1,nvir)
            del (vvvv_p)
            a += k
    else :
        raise Exception("Unknown vvvv type") 

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv


def ea_adc_matvec(adc, M_ab=None, eris=None):


    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]

    nocc = adc._nocc
    nvir = adc._nvir

    ab_ind = np.tril_indices(nvir, k=-1)

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ab = e_vir[:,None] + e_vir
    d_i = e_occ[:,None]
    D_n = -d_i + d_ab.reshape(-1)
    D_iab = D_n.reshape(-1)

    if M_ab is None:
        M_ab = adc.get_imds()
    
    time0 = time.time()
    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        
        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nocc,nvir,nvir)

############ ADC(2) ab block ############################

        s[s1:f1] = lib.einsum('ab,b->a',M_ab,r1)

############# ADC(2) a - ibc and ibc - a coupling blocks #########################

        if isinstance(eris.ovvv, type(None)):
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
        else :
            chnk_size = nocc
        a = 0
        temp_doubles = np.zeros((nocc,nvir,nvir))
        for p in range(0,nocc,chnk_size):
            if getattr(adc, 'with_df', None):
                eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
            else :
                eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            k = eris_ovvv.shape[0]

            s[s1:f1] +=  2. * lib.einsum('icab,ibc->a', eris_ovvv, r2[a:a+k], optimize = True)
            s[s1:f1] -=  lib.einsum('ibac,ibc->a',   eris_ovvv, r2[a:a+k], optimize = True)
   
            temp_doubles[a:a+k] += lib.einsum('icab,a->ibc', eris_ovvv, r1, optimize = True)
            del eris_ovvv
            a += k

        s[s2:f2] +=  temp_doubles.reshape(-1)
################ ADC(2) iab - jcd block ############################

        s[s2:f2] +=  D_iab * r2.reshape(-1)

        cput0 = log.timer_debug1("completed sigma vector ADC(2) calculation", *cput0)

############### ADC(3) iab - jcd block ############################


        if (method == "adc(2)-x" or method == "adc(3)"):

               t2_2 = adc.t2[1]

               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo

               r2 = r2.reshape(nocc, nvir, nvir)

               if isinstance(eris.vvvv, np.ndarray):
                   r_bab_t = r2.reshape(nocc,-1)
                   eris_vvvv = eris.vvvv
                   s[s2:f2] += np.dot(r_bab_t,eris_vvvv.T).reshape(-1)
               elif isinstance(eris.vvvv, list):
                   s[s2:f2] += ea_contract_r_vvvv(adc,r2,eris.vvvv)
               else :
                   s[s2:f2] += ea_contract_r_vvvv(adc,r2,eris.Lvv)

               s[s2:f2] -= 0.5*lib.einsum('jzyi,jzx->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,r2,optimize = True).reshape(-1)

               s[s2:f2] -= 0.5*lib.einsum('jiyz,jxz->ixy',eris_oovv,r2,optimize = True).reshape(-1)
               s[s2:f2] += 0.5*lib.einsum('jzyi,jxz->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] -=  0.5*lib.einsum('jixz,jzy->ixy',eris_oovv,r2,optimize = True).reshape(-1)
               s[s2:f2] -=  0.5*lib.einsum('jixw,jwy->ixy',eris_oovv,r2,optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('jiyw,jxw->ixy',eris_oovv,r2,optimize = True).reshape(-1)
               s[s2:f2] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] += 0.5*lib.einsum('jwyi,jxw->ixy',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('jwyi,jwx->ixy',eris_ovvo,r2,optimize = True).reshape(-1)

            #print("Calculating additional terms for adc(3)")
        cput0 = log.timer_debug1("completed sigma vector ADC(2)-x calculation", *cput0)

        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo

############### ADC(3) a - ibc block and ibc-a coupling blocks ########################

               temp =   0.25 * lib.einsum('lmab,jab->lmj',t2_1,r2)
               temp -=  0.25 * lib.einsum('lmab,jba->lmj',t2_1,r2)
               temp -=  0.25 * lib.einsum('mlab,jab->lmj',t2_1,r2)
               temp +=  0.25 * lib.einsum('mlab,jba->lmj',t2_1,r2)

               s[s1:f1] += lib.einsum('lmj,lamj->a',temp, eris_ovoo, optimize=True)
               s[s1:f1] -= lib.einsum('lmj,malj->a',temp, eris_ovoo, optimize=True)
               del (temp)

               temp_1 = -lib.einsum('lmzw,jzw->jlm',t2_1,r2)
               s[s1:f1] -= lib.einsum('jlm,lamj->a',temp_1, eris_ovoo, optimize=True)

               temp_s_a = lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
               temp_s_a -= lib.einsum('jlwd,jwz->lzd',t2_1,r2,optimize=True)
               temp_s_a -= lib.einsum('ljwd,jzw->lzd',t2_1,r2,optimize=True)
               temp_s_a += lib.einsum('ljwd,jwz->lzd',t2_1,r2,optimize=True)
               temp_s_a += lib.einsum('ljdw,jzw->lzd',t2_1,r2,optimize=True)

               temp_s_a_1 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
               temp_s_a_1 += lib.einsum('jlzd,jzw->lwd',t2_1,r2,optimize=True)
               temp_s_a_1 += lib.einsum('ljzd,jwz->lwd',t2_1,r2,optimize=True)
               temp_s_a_1 -= lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)
               temp_s_a_1 += -lib.einsum('ljdz,jwz->lwd',t2_1,r2,optimize=True)

#########################################################################################
               temp_t2_r2_1 = lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
               temp_t2_r2_1 -= lib.einsum('jlwd,jwz->lzd',t2_1,r2,optimize=True)
               temp_t2_r2_1 += lib.einsum('jlwd,jzw->lzd',t2_1,r2,optimize=True)
               temp_t2_r2_1 -= lib.einsum('ljwd,jzw->lzd',t2_1,r2,optimize=True)

               temp_t2_r2_2 = -lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
               temp_t2_r2_2 += lib.einsum('jlzd,jzw->lwd',t2_1,r2,optimize=True)
               temp_t2_r2_2 -= lib.einsum('jlzd,jwz->lwd',t2_1,r2,optimize=True)
               temp_t2_r2_2 += lib.einsum('ljzd,jwz->lwd',t2_1,r2,optimize=True)

               temp_t2_r2_3 = -lib.einsum('ljzd,jzw->lwd',t2_1,r2,optimize=True)

               temp_a = t2_1.transpose(0,3,1,2).copy()
               temp_b = temp_a.reshape(nocc*nvir,nocc*nvir)
               r2_t = r2.reshape(nocc*nvir,-1)
               temp_c = np.dot(temp_b,r2_t).reshape(nocc,nvir,nvir)
               temp_t2_r2_4 = temp_c.transpose(0,2,1).copy()

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = radc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc
               a = 0
               temp = np.zeros((nocc,nvir,nvir))
               temp_1_1 = np.zeros((nocc,nvir,nvir))
               temp_2_1 = np.zeros((nocc,nvir,nvir))
               for p in range(0,nocc,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                   k = eris_ovvv.shape[0]

                   temp_1_1[a:a+k] = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)
                   temp_1_1[a:a+k] -= lib.einsum('lbxd,b->lxd', eris_ovvv,r1,optimize=True)
                   temp_2_1[a:a+k] = lib.einsum('ldxb,b->lxd', eris_ovvv,r1,optimize=True)

                   s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_s_a[a:a+k],eris_ovvv,optimize=True)
                   s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_s_a[a:a+k],eris_ovvv,optimize=True)
                   s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_s_a_1[a:a+k],eris_ovvv,optimize=True)
                   s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_s_a_1[a:a+k],eris_ovvv,optimize=True)

                   s[s1:f1] += 0.5*lib.einsum('lzd,ldza->a',temp_t2_r2_1[a:a+k],eris_ovvv,optimize=True)

                   s[s1:f1] -= 0.5*lib.einsum('lwd,ldwa->a',temp_t2_r2_2[a:a+k],eris_ovvv,optimize=True)

                   s[s1:f1] += 0.5*lib.einsum('lwd,lawd->a',temp_t2_r2_3[a:a+k],eris_ovvv,optimize=True)

                   s[s1:f1] -= 0.5*lib.einsum('lzd,lazd->a',temp_t2_r2_4[a:a+k],eris_ovvv,optimize=True)

                   temp[a:a+k]  = -lib.einsum('lbyd,b->lyd',eris_ovvv,r1,optimize=True)
                   temp_1= -lib.einsum('lyd,lixd->ixy',temp[a:a+k],t2_1[a:a+k],optimize=True)
                   s[s2:f2] -= temp_1.reshape(-1)
                   del eris_ovvv
                   a += k

               del (temp_s_a)
               del (temp_s_a_1)
               del (temp_t2_r2_1)
               del (temp_t2_r2_2)
               del (temp_t2_r2_3)
               del (temp_t2_r2_4)
#########################################################################################

               temp_1 = lib.einsum('b,lbmi->lmi',r1,eris_ovoo)
               s[s2:f2] += lib.einsum('lmi,lmxy->ixy',temp_1, t2_1, optimize=True).reshape(-1)

               temp  = lib.einsum('lxd,lidy->ixy',temp_1_1,t2_1,optimize=True)
               temp  += lib.einsum('lxd,ilyd->ixy',temp_2_1,t2_1,optimize=True)
               temp  -= lib.einsum('lxd,ildy->ixy',temp_2_1,t2_1,optimize=True)
               s[s2:f2] += temp.reshape(-1)
               cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)
              
               del (temp)
               del (temp_1)
               del (temp_1_1)
               del (temp_2_1)

        return s
        cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)

    return sigma_

def ip_adc_matvec_off(adc,M_ij=None, eris=None, cvs=False, fc_bool=True, mom_skd=False, alpha_proj=0):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    
    t1, t2 = adc.compute_amplitudes(eris, fc_bool)
    
    t2_1 = t2[0]
    t1_2 = t1[0]
    
    #t2_1 = adc.t2[0]
    #t1_2 = adc.t1[0]

    nocc = adc._nocc
    nvir = adc._nvir

    ij_ind = np.tril_indices(nocc, k=-1)

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds(fc_bool)

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)
        
        #if cvs is True:
        #    r = cvs_projector(adc, r)
         

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]

        r2 = r2.reshape(nvir,nocc,nocc)

        eris_ovoo = eris.ovoo
        
        # Scaling off-diagonal (w.r.t. CVS) elements by alpha 
        #alpha_proj = adc.alpha_proj
        ncore = adc.ncore_proj
        
        # Scaling i-j block
        alpha_ij = np.ones((n_singles,n_singles))
        #A
        alpha_ij[:ncore, ncore:] *= alpha_proj**3
        #B
        alpha_ij[ncore:, :ncore] *= alpha_proj**3
        
        # Scaling aij-bkl block
        alpha_aij_bkl = np.ones((nvir,nocc,nocc,nvir,nocc,nocc))
        #A
        """ 
        alpha_aij_bkl[:,:ncore,:ncore,:,:ncore,ncore:] *= alpha_proj
        alpha_aij_bkl[:,:ncore,:ncore,:,ncore:,:ncore] *= alpha_proj
        #B
        #alpha_aij_bkl[:,:ncore,:ncore,:,ncore:,ncore:] *= alpha_proj
        #C
        #alpha_aij_bkl[:,ncore:,ncore:,:,:ncore,:ncore] *= alpha_proj
        #D
        alpha_aij_bkl[:,ncore:,ncore:,:,:ncore,ncore:] *= alpha_proj
        alpha_aij_bkl[:,ncore:,ncore:,:,ncore:,:ncore] *= alpha_proj
        #E
        alpha_aij_bkl[:,:ncore,ncore:,:,:ncore,:ncore] *= alpha_proj
        alpha_aij_bkl[:,ncore:,:ncore,:,:ncore,:ncore] *= alpha_proj
        #F
        alpha_aij_bkl[:,:ncore,ncore:,:,ncore:,ncore:] *= alpha_proj
        alpha_aij_bkl[:,ncore:,:ncore,:,ncore:,ncore:] *= alpha_proj
        
        #A-b
        """ 
        alpha_aij_bkl[:,:ncore,:ncore,:,ncore:,ncore:] *= alpha_proj
        #B-b
        alpha_aij_bkl[:,:ncore,ncore:,:,ncore:,ncore:] *= alpha_proj
        alpha_aij_bkl[:,ncore:,:ncore,:,ncore:,ncore:] *= alpha_proj
        #C-b
        alpha_aij_bkl[:,ncore:,ncore:,:,:ncore,ncore:] *= alpha_proj
        alpha_aij_bkl[:,ncore:,ncore:,:,ncore:,:ncore] *= alpha_proj
        #D-b 
        alpha_aij_bkl[:,ncore:,ncore:,:,:ncore,:ncore] *= alpha_proj
        
        # Scaling i-ajk block
        alpha_i_ajk = np.ones((n_singles,nvir,nocc,nocc))
        #A
        alpha_i_ajk[:ncore,:,ncore:,ncore:] *= alpha_proj**2
        #B
        alpha_i_ajk[ncore:,:,ncore:,:ncore] *= alpha_proj**2
        alpha_i_ajk[ncore:,:,:ncore,ncore:] *= alpha_proj**2
        #CVS-a
        #alpha_i_ajk[:ncore,:,:ncore,:ncore] *= alpha_proj
        #CVS-b
        alpha_i_ajk[ncore:,:,:ncore,:ncore] *= alpha_proj**2
        
        
        # Scaling ajk-i block
        alpha_ajk_i = alpha_i_ajk.transpose(1,2,3,0)
        
         
############ ADC(2) ij block ############################
        
        r1_ss = np.einsum('ij,j->ij', alpha_ij, r1, optimize = True)
               
        s[s1:f1] = lib.einsum('ij,j,ij->i',M_ij,r1,alpha_ij, optimize = True)

############ ADC(2) i - kja block #########################
        r2_sd = lib.einsum('iajk,ajk->iajk', alpha_i_ajk, r2, optimize=True)
                         
        s[s1:f1] += 2. * lib.einsum('jaki,ajk,iajk->i', eris_ovoo, r2, alpha_i_ajk, optimize = True)
        s[s1:f1] -= lib.einsum('kaji,ajk,iajk->i', eris_ovoo, r2,alpha_i_ajk, optimize = True)

############## ADC(2) ajk - i block ############################
        
        r1_ds = lib.einsum('ajki,i->ajki', alpha_ajk_i, r1, optimize=True) 
        
        temp = lib.einsum('jaki,i,iajk->ajk', eris_ovoo, r1,alpha_i_ajk, optimize = True).reshape(-1)
        s[s2:f2] += temp.reshape(-1)

################ ADC(2) ajk - bil block ############################
         
        s[s2:f2] += D_aij * r2.reshape(-1)
        cput0 = log.timer_debug1("completed sigma vector ADC(2) calculation", *cput0)

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):
        
               t2_2 = adc.t2[1]

               eris_oooo = eris.oooo
               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo
               
               #r2_dd = np.einsum('aijbkl,bkl->aijbkl', alpha_aij_bkl, r2, optimize=True)
               #r2_dd = r2  
               s[s2:f2] -= 0.5*lib.einsum('kijl,ali,ajkali->ajk',eris_oooo, r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('klji,ail,ajkail->ajk',eris_oooo ,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               
               s[s2:f2] += 0.5*lib.einsum('klba,bjl,ajkbjl->ajk',eris_oovv,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               
               s[s2:f2] +=  0.5*lib.einsum('jabl,bkl,ajkbkl->ajk',eris_ovvo,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] -=  0.5*lib.einsum('jabl,blk,ajkblk->ajk',eris_ovvo,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] +=  0.5*lib.einsum('jlba,blk,ajkblk->ajk',eris_oovv,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] -=  0.5*lib.einsum('jabl,blk,ajkblk->ajk',eris_ovvo,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               
               s[s2:f2] += 0.5*lib.einsum('kiba,bji,ajkbji->ajk',eris_oovv,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               
               s[s2:f2] += 0.5*lib.einsum('jiba,bik,ajkbik->ajk',eris_oovv,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('jabi,bik,ajkbik->ajk',eris_ovvo,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('jabi,bik,ajkbik->ajk',eris_ovvo,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               s[s2:f2] += 0.5*lib.einsum('jabi,bki,ajkbki->ajk',eris_ovvo,r2,alpha_aij_bkl, optimize = True).reshape(-1)
               
        cput0 = log.timer_debug1("completed sigma vector ADC(2)-x calculation", *cput0)
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo

################ ADC(3) i - kja block and ajk - i ############################
               
               temp =  0.25 * lib.einsum('ijbc,faij->fabc',t2_1, r2_sd, optimize=True)
               temp -= 0.25 * lib.einsum('ijbc,faji->fabc',t2_1, r2_sd, optimize=True)
               temp -= 0.25 * lib.einsum('jibc,faij->fabc',t2_1, r2_sd, optimize=True)
               temp += 0.25 * lib.einsum('jibc,faji->fabc',t2_1, r2_sd, optimize=True)
                 
               temp_1 = lib.einsum('kjcb,fajk->fabc',t2_1,r2_sd, optimize=True)

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = radc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc
               a = 0
               temp_singles = np.zeros((nocc))
               temp_doubles = np.zeros((nvir,nvir,nvir))
               for p in range(0,nocc,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                   k = eris_ovvv.shape[0]

                   temp_singles[a:a+k] += lib.einsum('iabc,icab->i',temp, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] -= lib.einsum('iabc,ibac->i',temp, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] += lib.einsum('iabc,icab->i',temp_1, eris_ovvv, optimize=True)
                   temp_doubles = lib.einsum('ajki,icab->jkcba',r1_ds,eris_ovvv, optimize=True)
                   s[s2:f2] += lib.einsum('jkcba,kjcb->ajk',temp_doubles, t2_1, optimize=True).reshape(-1)
                   del eris_ovvv
                   del temp_doubles
                   a += k

               s[s1:f1] += temp_singles
               temp = np.zeros_like(r2)
               temp =  lib.einsum('jlab,fajk->fblk',t2_1,r2_sd,optimize=True)
               temp -= lib.einsum('jlab,fakj->fblk',t2_1,r2_sd,optimize=True)
               temp -= lib.einsum('ljab,fajk->fblk',t2_1,r2_sd,optimize=True)
               temp += lib.einsum('ljab,fakj->fblk',t2_1,r2_sd,optimize=True)
               temp += lib.einsum('ljba,fajk->fblk',t2_1,r2_sd,optimize=True)

               temp_1 = np.zeros_like(r2)
               temp_1 =  lib.einsum('jlab,fajk->fblk',t2_1,r2_sd,optimize=True)
               temp_1 -= lib.einsum('jlab,fakj->fblk',t2_1,r2_sd,optimize=True)
               temp_1 += lib.einsum('jlab,fajk->fblk',t2_1,r2_sd,optimize=True)
               temp_1 -= lib.einsum('ljab,fajk->fblk',t2_1,r2_sd,optimize=True)

               temp_2 = lib.einsum('jlba,fakj->fblk',t2_1,r2_sd, optimize=True)

               s[s1:f1] += 0.5*lib.einsum('iblk,lbik->i',temp,eris_ovoo, optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('iblk,iblk->i',temp,eris_ovoo, optimize=True)
               s[s1:f1] += 0.5*lib.einsum('iblk,lbik->i',temp_1,eris_ovoo, optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('iblk,iblk->i',temp_2,eris_ovoo, optimize=True)
               del temp
               del temp_1
               del temp_2

               temp = np.zeros_like(r2)
               temp = -lib.einsum('klab,fakj->fblj',t2_1,r2_sd,optimize=True)
               temp += lib.einsum('klab,fajk->fblj',t2_1,r2_sd,optimize=True)
               temp += lib.einsum('lkab,fakj->fblj',t2_1,r2_sd,optimize=True)
               temp -= lib.einsum('lkab,fajk->fblj',t2_1,r2_sd,optimize=True)
               temp -= lib.einsum('lkba,fakj->fblj',t2_1,r2_sd,optimize=True)

               temp_1 = np.zeros_like(r2)
               temp_1  = -lib.einsum('klab,fakj->fblj',t2_1,r2_sd,optimize=True)
               temp_1 += lib.einsum('klab,fajk->fblj',t2_1,r2_sd,optimize=True)
               temp_1 -= lib.einsum('klab,fakj->fblj',t2_1,r2_sd,optimize=True)
               temp_1 += lib.einsum('lkab,fakj->fblj',t2_1,r2_sd,optimize=True)

               temp_2 = -lib.einsum('klba,fajk->fblj',t2_1,r2_sd,optimize=True)

               s[s1:f1] -= 0.5*lib.einsum('iblj,lbij->i',temp,eris_ovoo, optimize=True)
               s[s1:f1] += 0.5*lib.einsum('iblj,iblj->i',temp,eris_ovoo, optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('iblj,lbij->i',temp_1,eris_ovoo, optimize=True)
               s[s1:f1] += 0.5*lib.einsum('iblj,iblj->i',temp_2,eris_ovoo, optimize=True)
               
               del temp
               del temp_1
               del temp_2

               temp_1  = lib.einsum('ajki,lbik->ajkbl',r1_ds,eris_ovoo)
               temp_1  -= lib.einsum('ajki,iblk->ajkbl',r1_ds,eris_ovoo)
               temp_2  = lib.einsum('ajki,lbik->ajkbl',r1_ds,eris_ovoo)

               temp  = lib.einsum('ajkbl,ljba->ajk',temp_1,t2_1, optimize=True)
               temp += lib.einsum('ajkbl,jlab->ajk',temp_2,t2_1, optimize=True)
               temp -= lib.einsum('ajkbl,ljab->ajk',temp_2,t2_1, optimize=True)
               s[s2:f2] += temp.reshape(-1)

               temp  = -lib.einsum('ajki,iblj->akjbl',r1_ds,eris_ovoo,optimize=True)
               temp_1 = -lib.einsum('akjbl,klba->ajk',temp,t2_1, optimize=True)
               s[s2:f2] -= temp_1.reshape(-1)
               cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)
               del temp
               del temp_1
               del temp_2

        s *= -1.0
        
        #if cvs is True:
        #    s = cvs_projector(adc, s)

        return s

    return sigma_

def ip_adc_matvec(adc,M_ij=None, eris=None, cvs=False, fc_bool=True, mom_skd=False, alpha_proj=0):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    #t1, t2 = adc.compute_amplitudes(eris, fc_bool)
    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]

    nocc = adc._nocc
    nvir = adc._nvir

    ij_ind = np.tril_indices(nocc, k=-1)

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds(fc_bool)

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        if cvs is True:
            r = cvs_projector(adc, r)

        if adc.ncore_proj_valence > 0:
            r = cvs_proj_valence(adc, r)

        s = np.zeros((dim))

        r1 = r[s1:f1]
        r2 = r[s2:f2]
        #r1 = np.ravel(r1)
        r2 = r2.reshape(nvir,nocc,nocc)

        eris_ovoo = eris.ovoo

############ ADC(2) ij block ############################
        
        s[s1:f1] = lib.einsum('ij,j->i',M_ij,r1)

############ ADC(2) i - kja block #########################

        s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo, r2, optimize = True)
        s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo, r2, optimize = True)

############## ADC(2) ajk - i block ############################

        temp = lib.einsum('jaki,i->ajk', eris_ovoo, r1, optimize = True).reshape(-1)
        s[s2:f2] += temp.reshape(-1)

################ ADC(2) ajk - bil block ############################

        s[s2:f2] += D_aij * r2.reshape(-1)
        cput0 = log.timer_debug1("completed sigma vector ADC(2) calculation", *cput0)

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):
        
               t2_2 = adc.t2[1]

               eris_oooo = eris.oooo
               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo
               
               s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)
               
               s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               
               s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               
               s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               
               s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               
        cput0 = log.timer_debug1("completed sigma vector ADC(2)-x calculation", *cput0)
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo

################ ADC(3) i - kja block and ajk - i ############################

               temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
               temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
               temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
               temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)

               temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = radc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc
               a = 0
               temp_singles = np.zeros((nocc))
               temp_doubles = np.zeros((nvir,nvir,nvir))
               for p in range(0,nocc,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                   k = eris_ovvv.shape[0]

                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                   temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv,optimize=True)
                   s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles, t2_1, optimize=True).reshape(-1)
                   del eris_ovvv
                   del temp_doubles
                   a += k

               s[s1:f1] += temp_singles
               temp = np.zeros_like(r2)
               temp =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
               temp -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
               temp -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
               temp += lib.einsum('ljab,akj->blk',t2_1,r2,optimize=True)
               temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)

               temp_1 = np.zeros_like(r2)
               temp_1 =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
               temp_1 -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
               temp_1 += lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
               temp_1 -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)

               temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)

               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)
               del temp
               del temp_1
               del temp_2

               temp = np.zeros_like(r2)
               temp = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
               temp += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
               temp += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
               temp -= lib.einsum('lkab,ajk->blj',t2_1,r2,optimize=True)
               temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)

               temp_1 = np.zeros_like(r2)
               temp_1  = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
               temp_1 += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
               temp_1 -= lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
               temp_1 += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)

               temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)

               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)
               
               del temp
               del temp_1
               del temp_2

               temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
               temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
               temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)

               temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1,optimize=True)
               temp -= lib.einsum('kbl,ljab->ajk',temp_2,t2_1,optimize=True)
               s[s2:f2] += temp.reshape(-1)

               temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
               temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
               s[s2:f2] -= temp_1.reshape(-1)
               cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)
               del temp
               del temp_1
               del temp_2

        s *= -1.0

        if cvs is True:
            s = cvs_projector(adc, s)


        if adc.ncore_proj_valence > 0:
            s = cvs_proj_valence(adc, s)

        return s

    return sigma_

def ip_adc_matvec_off(adc,M_ij=None, eris=None, cvs=False, fc_bool=True, mom_skd=False, alpha_proj=0):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]

    nocc = adc._nocc
    nvir = adc._nvir

    ij_ind = np.tril_indices(nocc, k=-1)

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    if eris is None:
        eris = adc.transform_integrals()
   
    ncore = adc.ncore_proj
  
    sc = 0
    fc = ncore
    sv = fc
    fv = n_singles
    s2 = fv
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None] + e_occ
    d_a = e_vir[:,None]
    D_n = -d_a + d_ij.reshape(-1)
    D_aij = D_n.reshape(-1)

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        s = np.zeros((dim))

        s1 = s[:n_singles]
        s2 = s[n_singles:].reshape(nvir,nocc,nocc)

        r1_c = r[sc:fc]
        r1_v = r[sv:fv]
        r2 = r[n_singles:]
        r2 = r2.reshape(nvir,nocc,nocc)
        r2_ecc = r2[:,:ncore,:ncore]
        r2_ecv = r2[:,:ncore,ncore:]
        r2_evc = r2[:,ncore:,:ncore] 
        r2_evv = r2[:,ncore:,ncore:]

        eris_ovoo = eris.ovoo

############ ADC(2) ij block ############################

        ###s[s1:f1] = lib.einsum('ij,j->i',M_ij,r1)
        
        M_cc = M_ij[:ncore,:ncore].copy()
        M_cv = M_ij[:ncore,ncore:].copy()
        M_vc = M_ij[ncore:,:ncore].copy()
        M_vv = M_ij[ncore:,ncore:].copy()
   
        s1[sc:fc] +=              lib.einsum('ij,j->i',M_cc,r1_c, optimize=True)
        s1[sc:fc] += alpha_proj * lib.einsum('ij,j->i',M_cv,r1_v, optimize=True)
        s1[sv:fv] += alpha_proj * lib.einsum('ij,j->i',M_vc,r1_c, optimize=True)
        s1[sv:fv] +=              lib.einsum('ij,j->i',M_vv,r1_v, optimize=True)


############ ADC(2) i - kja block #########################

        ###s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_ovoo, r2, optimize = True)
        ###s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_ovoo, r2, optimize = True)
        
        s1[sc:fc] += alpha_proj * 2. * lib.einsum('jaki,ajk->i', eris_ovoo[:ncore,:,:ncore,:ncore], r2_ecc, opitimize = True)
        s1[sc:fc] -= alpha_proj * lib.einsum('kaji,ajk->i', eris_ovoo[:ncore,:,:ncore,:ncore], r2_ecc, optimize = True)
        s1[sc:fc] +=              2. * lib.einsum('jaki,ajk->i', eris_ovoo[:ncore,:,ncore:,:ncore], r2_ecv, optimize = True)
        s1[sc:fc] -=              lib.einsum('kaji,ajk->i', eris_ovoo[ncore:,:,:ncore,:ncore], r2_ecv, optimize = True)
        s1[sc:fc] +=              2. * lib.einsum('jaki,ajk->i', eris_ovoo[ncore:,:,:ncore,:ncore], r2_evc, optimize = True)
        s1[sc:fc] -=              lib.einsum('kaji,ajk->i', eris_ovoo[:ncore,:,ncore:,:ncore], r2_evc, optimize = True)
        s1[sc:fc] += alpha_proj * 2. * lib.einsum('jaki,ajk->i', eris_ovoo[ncore:,:,ncore:,:ncore], r2_evv, optimize = True)
        s1[sc:fc] -= alpha_proj * lib.einsum('kaji,ajk->i', eris_ovoo[ncore:,:,ncore:,:ncore], r2_evv, optimize = True)

        s1[sv:fv] +=              2. * lib.einsum('jaki,ajk->i', eris_ovoo[:ncore,:,:ncore,ncore:], r2_ecc, opitimize = True)
        s1[sv:fv] -=              lib.einsum('kaji,ajk->i', eris_ovoo[:ncore,:,:ncore,ncore:], r2_ecc, optimize = True)
        s1[sv:fv] += alpha_proj * 2. * lib.einsum('jaki,ajk->i', eris_ovoo[:ncore,:,ncore:,ncore:], r2_ecv, optimize = True)
        s1[sv:fv] -= alpha_proj * lib.einsum('kaji,ajk->i', eris_ovoo[ncore:,:,:ncore,ncore:], r2_ecv, optimize = True)
        s1[sv:fv] += alpha_proj * 2. * lib.einsum('jaki,ajk->i', eris_ovoo[ncore:,:,:ncore,ncore:], r2_evc, optimize = True)
        s1[sv:fv] -= alpha_proj * lib.einsum('kaji,ajk->i', eris_ovoo[:ncore,:,ncore:,ncore:], r2_evc, optimize = True)
        s1[sv:fv] +=              2. * lib.einsum('jaki,ajk->i', eris_ovoo[ncore:,:,ncore:,ncore:], r2_evv, optimize = True)
        s1[sv:fv] -=              lib.einsum('kaji,ajk->i', eris_ovoo[ncore:,:,ncore:,ncore:], r2_evv, optimize = True)


############## ADC(2) ajk - i block ############################

        ###temp = lib.einsum('jaki,i->ajk', eris_ovoo, r1, optimize = True).reshape(-1)
        ###s[s2:f2] += temp.reshape(-1)

        s2[:,:ncore,:ncore] += alpha_proj * lib.einsum('jaki,i->ajk', eris_ovoo[:ncore,:,:ncore,:ncore], r1_c, optimize = True)
        s2[:,:ncore,:ncore] +=              lib.einsum('jaki,i->ajk', eris_ovoo[:ncore,:,:ncore,ncore:], r1_v, optimize = True)

        s2[:,:ncore,ncore:] +=              lib.einsum('jaki,i->ajk', eris_ovoo[:ncore,:,ncore:,:ncore], r1_c, optimize = True)
        s2[:,ncore:,:ncore] +=              lib.einsum('jaki,i->ajk', eris_ovoo[ncore:,:,:ncore,:ncore], r1_c, optimize = True)
        s2[:,:ncore,ncore:] += alpha_proj * lib.einsum('jaki,i->ajk', eris_ovoo[:ncore,:,ncore:,ncore:], r1_v, optimize = True)
        s2[:,ncore:,:ncore] += alpha_proj * lib.einsum('jaki,i->ajk', eris_ovoo[ncore:,:,:ncore,ncore:], r1_v, optimize = True)
        
        s2[:,ncore:,ncore:] += alpha_proj * lib.einsum('jaki,i->ajk', eris_ovoo[ncore:,:,ncore:,:ncore], r1_c, optimize = True)
        s2[:,ncore:,ncore:] +=              lib.einsum('jaki,i->ajk', eris_ovoo[ncore:,:,ncore:,ncore:], r1_v, optimize = True)

################ ADC(2) ajk - bil block ############################

        ###s[s2:f2] += D_aij * r2.reshape(-1)

        temp = D_aij * r2.reshape(-1) 
        s2[:,:,:] += temp.reshape(nvir,nocc,nocc)
        
        cput0 = log.timer_debug1("completed sigma vector ADC(2) calculation", *cput0)

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):
        
               t2_2 = adc.t2[1]

               eris_oooo = eris.oooo
               eris_oovv = eris.oovv
               eris_ovvo = eris.ovvo
               
               #r2_dd = np.einsum('aijbkl,bkl->aijbkl', alpha_aij_bkl, r2, optimize=True)  
               #temp_test = -0.5*lib.einsum('kijl,ajkali->ajk',eris_oooo, r2_dd, optimize = True).reshape(-1)
               #temp_test -= 0.5*lib.einsum('klji,akjail->ajk',eris_oooo ,r2_dd, optimize = True).reshape(-1)
               #norm_ref = np.linalg.norm(temp_test)

               #s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
               #s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)
               
               #s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               
               #s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               #s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               #s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               #s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               
               #s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               
               #s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               #s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               #s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               #s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               
               ###s[s2:f2] -= 0.5*lib.einsum('kijl,ali->ajk',eris_oooo, r2, optimize = True).reshape(-1)
               ###s[s2:f2] -= 0.5*lib.einsum('klji,ail->ajk',eris_oooo ,r2, optimize = True).reshape(-1)
               
               # Sigma_ecc
               
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,:ncore,:ncore,:ncore], r2_ecc, optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,:ncore,:ncore,:ncore] ,r2_ecc, optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,ncore:,:ncore,:ncore], r2_ecv, optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,:ncore,:ncore,ncore:] ,r2_evc, optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,:ncore,:ncore,ncore:], r2_evc, optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,ncore:,:ncore,:ncore] ,r2_ecv, optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,ncore:,:ncore,ncore:], r2_evv, optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,ncore:,:ncore,ncore:] ,r2_evv, optimize = True)
               
               # Sigma_ecv
               
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,:ncore,:ncore,:ncore], r2_ecc, optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,:ncore,:ncore,:ncore] ,r2_ecc, optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,ncore:,:ncore,:ncore], r2_ecv, optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,:ncore,:ncore,ncore:] ,r2_evc, optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,:ncore,:ncore,ncore:], r2_evc, optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,ncore:,:ncore,:ncore] ,r2_ecv, optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,ncore:,:ncore,ncore:], r2_evv, optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,ncore:,:ncore,ncore:] ,r2_evv, optimize = True)
               
               # Sigma_evc
               
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,:ncore,ncore:,:ncore], r2_ecc, optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,:ncore,ncore:,:ncore] ,r2_ecc, optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,ncore:,ncore:,:ncore], r2_ecv, optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,:ncore,ncore:,ncore:] ,r2_evc, optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,:ncore,ncore:,ncore:], r2_evc, optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,ncore:,ncore:,:ncore] ,r2_ecv, optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[:ncore,ncore:,ncore:,ncore:], r2_evv, optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[:ncore,ncore:,ncore:,ncore:] ,r2_evv, optimize = True)

               # Sigma_evv

               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,:ncore,ncore:,:ncore], r2_ecc, optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,:ncore,ncore:,:ncore] ,r2_ecc, optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,ncore:,ncore:,:ncore], r2_ecv, optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,:ncore,ncore:,ncore:] ,r2_evc, optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,:ncore,ncore:,ncore:], r2_evc, optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,ncore:,ncore:,:ncore] ,r2_ecv, optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('kijl,ali->ajk',eris_oooo[ncore:,ncore:,ncore:,ncore:], r2_evv, optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('klji,ail->ajk',eris_oooo[ncore:,ncore:,ncore:,ncore:] ,r2_evv, optimize = True)
           
               ###s[s2:f2] += 0.5*lib.einsum('klba,bjl->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               
               # Sigma_ecc
               
               s2[:,:ncore,:ncore] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,ncore:,:,:],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,:ncore,:,:],r2_evc,optimize = True)
               ##s2[:,:ncore,:ncore] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
               
               
               # Sigma_ecv
               
               s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,:ncore,ncore:] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,ncore:,:,:],r2_ecv,optimize = True)
               ##s2[:,:ncore,ncore:] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,:ncore,:,:],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore,ncore:,:,:],r2_evv,optimize = True)

               
               # Sigma_evc
               
               ##s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,ncore:,:ncore] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,ncore:,:,:],r2_ecv,optimize = True)
               s2[:,ncore:,:ncore] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,:ncore,:,:],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
               
               
               # Sigma_evv
               
               ##s2[:,ncore:,ncore:] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,ncore:,:,:],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,:ncore,:,:],r2_evc,optimize = True)
               s2[:,ncore:,ncore:] +=              0.5*lib.einsum('klba,bjl->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)
               

               ###s[s2:f2] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               ###s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               ###s[s2:f2] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               ###s[s2:f2] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               
               # Sigma_ecc

               s2[:,:ncore,:ncore] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)               
               ##s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_evc,optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,ncore:],r2_evc,optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,:ncore,:ncore] +=  0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo,r2_evv,optimize = True)
               ##s2[:,:ncore,:ncore] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2_evv,optimize = True)
               ##s2[:,:ncore,:ncore] +=  0.5*lib.einsum('jlba,blk->ajk',eris_oovv,r2_evv,optimize = True)
               ##s2[:,:ncore,:ncore] -=  0.5*lib.einsum('jabl,blk->ajk',eris_ovvo,r2_evv,optimize = True)
               
               
               # Sigma_ecv

               ##s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:nocre,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_ecv,optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)
               s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecv,optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)               
               s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,ncore:],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)
               s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)


               # Sigma_evc

               s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_ecv,optimize = True)
               ##s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)
               ##s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecv,optimize = True)
               ##s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)               
               ##s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,ncore:],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               ##s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore,:,:,ncore:],r2_evv,optimize = True)
               ##s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)
               ##s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)

               # Sigma_evv

               ##s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[:nocre,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)               
               s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_evc,optimize = True)
               ##s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,ncore:],r2_evc,optimize = True)
               ##s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jabl,bkl->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jlba,blk->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabl,blk->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               
               
               #s[s2:f2] += 0.5*lib.einsum('kiba,bji->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               
               # Sigma_ecc
               
               s2[:,:ncore,:ncore] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,ncore:,:,:],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,:ncore,:,:],r2_evc,optimize = True)
               ##s2[:,:ncore,:ncore] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
               
                             
               # Sigma_ecv
               
               s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,:ncore,ncore:] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,ncore:,:,:],r2_ecv,optimize = True)
               ##s2[:,:ncore,ncore:] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,:ncore,:,:],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)


               # Sigma_evc
               
               ##s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,ncore:,:ncore] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,ncore:,:,:],r2_ecv,optimize = True)
               s2[:,ncore:,:ncore] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,:ncore,:,:],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
 
               # Sigma_evv
               
               ##s2[:,ncore:,ncore:] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,ncore:,:,:],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evc,optimize = True)
               s2[:,ncore:,ncore:] +=              0.5*lib.einsum('kiba,bji->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)

               
               ###s[s2:f2] += 0.5*lib.einsum('jiba,bik->ajk',eris_oovv,r2,optimize = True).reshape(-1)
               ###s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               ###s[s2:f2] -= 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               ###s[s2:f2] += 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo,r2,optimize = True).reshape(-1)
               

               # Sigma_ecc


               s2[:,:ncore,:ncore] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,:ncore,:ncore] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)               
               ##s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)
               ##s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)
               s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_ecv,optimize = True)
               s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evc,optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               s2[:,:ncore,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,:ncore,:ncore] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_evc,optimize = True)               
               ##s2[:,:ncore,:ncore] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
               ##s2[:,:ncore,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)
               ##s2[:,:ncore,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)
               ##s2[:,:ncore,:ncore] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)

               
               
               # Sigma_ecv


               ##s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecc,optimize = True)               
               s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,:ncore,:,:],r2_ecv,optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)
               s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_ecv,optimize = True)
               ##s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_ecv,optimize = True)
               ##s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,:ncore,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evc,optimize = True)
               s2[:,:ncore,ncore:] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,:ncore],r2_evc,optimize = True)               
               s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[:ncore,ncore:,:,:],r2_evv,optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)
               s2[:,:ncore,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)
               s2[:,:ncore,ncore:] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[:ncore,:,:,ncore:],r2_evv,optimize = True)

               
               
               # Sigma_evc


               s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecc,optimize = True)
               s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecc,optimize = True)               
               ##s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore;,:ncore,:,:],r2_ecv,optimize = True)
               ##s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)
               ##s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)
               s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_ecv,optimize = True)
               s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               s2[:,ncore:,:ncore] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,ncore:,:ncore] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_evc,optimize = True)               
               ##s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)
               ##s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               ##s2[:,ncore:,:ncore] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               ##s2[:,ncore:,:ncore] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)

               
               # Sigma_evv


               ##s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore,:ncore,:,:],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore,:,:,:ncore],r2_ecc,optimize = True)
               ##s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore,:,:,:ncore],r2_ecc,optimize = True)               
               s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore:,:ncore,:,:],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)
               s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_ecv,optimize = True)
               ##s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_ecv,optimize = True)
               ##s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evc,optimize = True)
               ##s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               ##s2[:,ncore:,ncore:] -= alpha_proj * 0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evc,optimize = True)
               s2[:,ncore:,ncore:] += alpha_proj * 0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,:ncore],r2_evc,optimize = True)               
               s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jiba,bik->ajk',eris_oovv[ncore:,ncore:,:,:],r2_evv,optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               s2[:,ncore:,ncore:] -=              0.5*lib.einsum('jabi,bik->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)
               s2[:,ncore:,ncore:] +=              0.5*lib.einsum('jabi,bki->ajk',eris_ovvo[ncore:,:,:,ncore:],r2_evv,optimize = True)

                              
        cput0 = log.timer_debug1("completed sigma vector ADC(2)-x calculation", *cput0)
        if (method == "adc(3)"):

               eris_ovoo = eris.ovoo

################ ADC(3) i - kja block and ajk - i ############################

               ##temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
               ##temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
               ##temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
               ##temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)

               ##temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)

               #temp_ecc
  
               temp =  0.25 * lib.einsum('ijbc,aij->abc',t2_1, r2, optimize=True)
               temp -= 0.25 * lib.einsum('ijbc,aji->abc',t2_1, r2, optimize=True)
               temp -= 0.25 * lib.einsum('jibc,aij->abc',t2_1, r2, optimize=True)
               temp += 0.25 * lib.einsum('jibc,aji->abc',t2_1, r2, optimize=True)
               temp_1 = lib.einsum('kjcb,ajk->abc',t2_1,r2, optimize=True)               

               if isinstance(eris.ovvv, type(None)):
                   chnk_size = radc_ao2mo.calculate_chunk_size(adc)
               else :
                   chnk_size = nocc
               a = 0
               temp_singles = np.zeros((nocc))
               temp_doubles = np.zeros((nvir,nvir,nvir))
               for p in range(0,nocc,chnk_size):
                   if getattr(adc, 'with_df', None):
                       eris_ovvv = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                   else :
                       eris_ovvv = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
                   k = eris_ovvv.shape[0]

                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] -= lib.einsum('abc,ibac->i',temp, eris_ovvv, optimize=True)
                   temp_singles[a:a+k] += lib.einsum('abc,icab->i',temp_1, eris_ovvv, optimize=True)
                   temp_doubles = lib.einsum('i,icab->cba',r1[a:a+k],eris_ovvv,optimize=True)
                   s[s2:f2] += lib.einsum('cba,kjcb->ajk',temp_doubles, t2_1, optimize=True).reshape(-1)
                   del eris_ovvv
                   del temp_doubles
                   a += k

               s[s1:f1] += temp_singles
               temp = np.zeros_like(r2)
               temp =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
               temp -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
               temp -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)
               temp += lib.einsum('ljab,akj->blk',t2_1,r2,optimize=True)
               temp += lib.einsum('ljba,ajk->blk',t2_1,r2,optimize=True)

               temp_1 = np.zeros_like(r2)
               temp_1 =  lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
               temp_1 -= lib.einsum('jlab,akj->blk',t2_1,r2,optimize=True)
               temp_1 += lib.einsum('jlab,ajk->blk',t2_1,r2,optimize=True)
               temp_1 -= lib.einsum('ljab,ajk->blk',t2_1,r2,optimize=True)

               temp_2 = lib.einsum('jlba,akj->blk',t2_1,r2, optimize=True)

               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo,optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo,optimize=True)
               del temp
               del temp_1
               del temp_2

               temp = np.zeros_like(r2)
               temp = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
               temp += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
               temp += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)
               temp -= lib.einsum('lkab,ajk->blj',t2_1,r2,optimize=True)
               temp -= lib.einsum('lkba,akj->blj',t2_1,r2,optimize=True)

               temp_1 = np.zeros_like(r2)
               temp_1  = -lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
               temp_1 += lib.einsum('klab,ajk->blj',t2_1,r2,optimize=True)
               temp_1 -= lib.einsum('klab,akj->blj',t2_1,r2,optimize=True)
               temp_1 += lib.einsum('lkab,akj->blj',t2_1,r2,optimize=True)

               temp_2 = -lib.einsum('klba,ajk->blj',t2_1,r2,optimize=True)

               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp,eris_ovoo,optimize=True)
               s[s1:f1] -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo,optimize=True)
               s[s1:f1] += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo,optimize=True)
               
               del temp
               del temp_1
               del temp_2

               temp_1  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)
               temp_1  -= lib.einsum('i,iblk->kbl',r1,eris_ovoo)
               temp_2  = lib.einsum('i,lbik->kbl',r1,eris_ovoo)

               temp  = lib.einsum('kbl,ljba->ajk',temp_1,t2_1,optimize=True)
               temp += lib.einsum('kbl,jlab->ajk',temp_2,t2_1,optimize=True)
               temp -= lib.einsum('kbl,ljab->ajk',temp_2,t2_1,optimize=True)
               s[s2:f2] += temp.reshape(-1)

               temp  = -lib.einsum('i,iblj->jbl',r1,eris_ovoo,optimize=True)
               temp_1 = -lib.einsum('jbl,klba->ajk',temp,t2_1,optimize=True)
               s[s2:f2] -= temp_1.reshape(-1)
               cput0 = log.timer_debug1("completed sigma vector ADC(3) calculation", *cput0)
               del temp
               del temp_1
               del temp_2
        
        s[:n_singles] = s1
        s[n_singles:] = s2.reshape(-1)
        s *= -1.0

        return s

    return sigma_

def ea_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T = np.zeros((dim))

######## ADC(2) part  ############################################

    if orb < nocc:

        T[s1:f1] = -t1_2[orb,:]

        t2_1_t = -t2_1.transpose(1,0,2,3).copy()

        T[s2:f2] += t2_1_t[:,orb,:,:].reshape(-1)

    else :

        T[s1:f1] += idn_vir[(orb-nocc), :]
        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)

        T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_1, optimize = True)

######## ADC(3) 2p-1h  part  ############################################

    if(method=="adc(2)-x"or adc.method=="adc(3)"):

        t2_2 = adc.t2[1]

        if orb < nocc:

            t2_2_t = -t2_2.transpose(1,0,2,3).copy()

            T[s2:f2] += t2_2_t[:,orb,:,:].reshape(-1)

########## ADC(3) 1p part  ############################################

    if(adc.method=="adc(3)"):

        t1_3 = adc.t1[1]

        if orb < nocc:

            T[s1:f1] += 0.5*lib.einsum('kac,ck->a',t2_1[:,orb,:,:], t1_2.T,optimize = True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize = True)
            T[s1:f1] -= 0.5*lib.einsum('kac,ck->a',t2_1[orb,:,:,:], t1_2.T,optimize = True)

            T[s1:f1] -= t1_3[orb,:]

        else:

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] += 0.25*lib.einsum('klc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] += 0.25*lib.einsum('lkc,klac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkc,lkac->a',t2_1[:,:,(orb-nocc),:], t2_2, optimize = True)

            T[s1:f1] -= 0.25*lib.einsum('klac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] += 0.25*lib.einsum('klac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] += 0.25*lib.einsum('lkac,klc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('lkac,lkc->a',t2_1, t2_2[:,:,(orb-nocc),:],optimize = True)

    T_aaa = T[n_singles:].reshape(nocc,nvir,nvir).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def ip_compute_trans_moments(adc, orb):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t2_1 = adc.t2[0]
    t1_2 = adc.t1[0]
    t2_1_a = t2_1 - t2_1.transpose(1,0,2,3).copy()

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    dim = n_singles + n_doubles

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    idn_occ = np.identity(nocc)
    idn_vir = np.identity(nvir)

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    T = np.zeros((dim))

######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T[s1:f1]  = idn_occ[orb, :]
        T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_1, optimize = True)
        T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_1, optimize = True)
    else :
        T[s1:f1] += t1_2[:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        t2_1_t = t2_1.transpose(2,3,1,0).copy()

        T[s2:f2] = t2_1_t[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 2h-1p  part  ############################################

    if(method=='adc(2)-x'or method=='adc(3)'):

        t2_2 = adc.t2[1]
        t2_2_a = t2_2 - t2_2.transpose(1,0,2,3).copy()

        if orb >= nocc:
            t2_2_t = t2_2.transpose(2,3,1,0).copy()

            T[s2:f2] += t2_2_t[(orb-nocc),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

    if(method=='adc(3)'):

        t1_3 = adc.t1[1]

        if orb < nocc:
            T[s1:f1] += 0.25*lib.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_2_a, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[orb,:,:,:], t2_2, optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[orb,:,:,:], t2_2, optimize = True)

            T[s1:f1] += 0.25*lib.einsum('ikdc,kdc->i',t2_1_a, t2_2_a[:,orb,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikcd,kcd->i',t2_1, t2_2[orb,:,:,:],optimize = True)
            T[s1:f1] -= 0.25*lib.einsum('ikdc,kdc->i',t2_1, t2_2[orb,:,:,:],optimize = True)
        else:
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1_a[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] += 0.5*lib.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc),:], t1_2,optimize = True)
            T[s1:f1] += t1_3[:,(orb-nocc)]

    T_aaa = T[n_singles:].reshape(nvir,nocc,nocc).copy()
    T_aaa = T_aaa - T_aaa.transpose(0,2,1)
    T[n_singles:] += T_aaa.reshape(-1)

    return T


def get_trans_moments(adc):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)
    nmo  = adc.nmo

    T = []

    for orb in range(nmo):

            T_a = adc.compute_trans_moments(orb)
            T.append(T_a)

    T = np.array(T)
    cput0 = log.timer_debug1("Completed trans moments calculation", *cput0)
    return T


def get_spec_factors_ea(adc, T, U, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nvir
    n_doubles = nocc * nvir * nvir

    U = U.reshape(nroots,-1)

    for I in range(U.shape[0]):
        U1 = U[I, :n_singles]
        U2 = U[I, n_singles:].reshape(nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[I,:] /= np.sqrt(UdotU)

    X = np.dot(T, U.T).reshape(-1, nroots)

    P = 2.0*lib.einsum("pi,pi->i", X, X)

    return P

def get_spec_factors_ip(adc, T, U, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc
    n_doubles = nvir * nocc * nocc

    U = U.reshape(nroots,-1)

    for I in range(U.shape[0]):
        U1 = U[I, :n_singles]
        U2 = U[I, n_singles:].reshape(nvir,nocc,nocc)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
        U[I,:] /= np.sqrt(UdotU)

    X = np.dot(T, U.T).reshape(-1, nroots)

    P = 2.0*lib.einsum("pi,pi->i", X, X)

    return P

def analyze_eigenvector_ip(adc, U):
    
    nocc = adc._nocc
    nvir = adc._nvir
    
    n_singles = nocc
    n_doubles = nvir * nocc * nocc
    evec_print_tol = 0.05
    
    logger.info(adc, "Number of occupied orbitals = %d", nocc)
    logger.info(adc, "Number of virtual orbitals =  %d", nvir)
    logger.info(adc, "Print eigenvector elements > %f\n", evec_print_tol)
    #U = np.array(adc.U)  
  
    for I in range(U.shape[0]):
        U1 = U[I, :n_singles]
        U2 = U[I, n_singles:].reshape(nvir,nocc,nocc)
        U1dotU1 = np.dot(U1, U1) 
        U2dotU2 =  2.*np.dot(U2.ravel(), U2.ravel()) - np.dot(U2.ravel(), U2.transpose(0,2,1).ravel())
       
        U_sq = U[I,:].copy()**2
        ind_idx = np.argsort(-U_sq)
        U_sq = U_sq[ind_idx] 
        U_sorted = U[I,ind_idx].copy()
        
        U_sorted = U_sorted[U_sq > evec_print_tol**2]
        ind_idx = ind_idx[U_sq > evec_print_tol**2]
             
        temp_doubles_idx = [0,0,0]  
        singles_idx = []
        doubles_idx = []
        singles_val = []
        doubles_val = []
        iter_num = 0
                
        for orb_idx in ind_idx:
            
            if orb_idx < n_singles:
                orb_s_idx = orb_idx + 1
                singles_idx.append(orb_s_idx)
                singles_val.append(U_sorted[iter_num])
            if orb_idx >= n_singles:
                orb_d_idx = orb_idx - n_singles
                      
                a_rem = orb_d_idx % (nocc*nocc)
                a_idx = (orb_d_idx - a_rem )//(nocc*nocc)
                temp_doubles_idx[0] = int(a_idx + 1 + n_singles) 
                j_rem = a_rem % nocc
                i_idx = (a_rem - j_rem)//nocc
                temp_doubles_idx[1] = int(i_idx + 1)
                temp_doubles_idx[2] = int(j_rem + 1)
                doubles_idx.append(temp_doubles_idx)
                doubles_val.append(U_sorted[iter_num])
                temp_doubles_idx = [0,0,0]
                
            iter_num += 1 

        logger.info(adc,'%s | root %d | norm(1h)  = %6.4f | norm(2h1p) = %6.4f ',adc.method ,I, U1dotU1, U2dotU2)

        if singles_val:
            logger.info(adc, "\n1h block: ") 
            logger.info(adc, "     i     U(i)")
            logger.info(adc, "------------------")
            for idx, print_singles in enumerate(singles_idx):
                logger.info(adc, '  %4d   %7.4f', print_singles, singles_val[idx])

        if doubles_val:
            logger.info(adc, "\n2h1p block: ") 
            logger.info(adc, "     i     j     a     U(i,j,a)")
            logger.info(adc, "-------------------------------")
            for idx, print_doubles in enumerate(doubles_idx):
                logger.info(adc, '  %4d  %4d  %4d     %7.4f', print_doubles[1], print_doubles[2], print_doubles[0], doubles_val[idx])

        logger.info(adc, "\n*************************************************************\n")


def analyze_spec_factor(adc):

    X = adc.X
    X_2 = (X.copy()**2)*2
    #thresh = 0.000000001
    thresh = adc.spec_thresh

    for i in range(X_2.shape[1]):

        print("----------------------------------------------------------------------------------------------------------------------------------------------")   
        logger.info(adc, 'Root %d', i)
        print("----------------------------------------------------------------------------------------------------------------------------------------------")   

        sort = np.argsort(-X_2[:,i])
        X_2_row = X_2[:,i]

        X_2_row = X_2_row[sort]
        
        if adc.mol.symmetry == False:
            sym = np.repeat(['A'], X_2_row.shape[0])
        else:
            sym = [symm.irrep_id2name(adc.mol.groupname, x) for x in adc._scf.mo_coeff.orbsym]
            sym = np.array(sym)

            sym = sym[sort]

        spec_Contribution = X_2_row[X_2_row > thresh]
        index_mo = sort[X_2_row > thresh]+1

        for c in range(index_mo.shape[0]):
            logger.info(adc, 'HF MO   %3.d | Spec. Contribution   %10.8f | Orbital symmetry   %s', index_mo[c], spec_Contribution[c], sym[c])

        logger.info(adc, 'Partial spec. Factor sum = %10.8f', np.sum(spec_Contribution))
        print("----------------------------------------------------------------------------------------------------------------------------------------------")   

class RADCEA(RADC):
    '''restricted ADC for EA energies and spectroscopic amplitudes

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

            >>> myadc = adc.RADC(mf).run()
            >>> myadcea = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''
    def __init__(self, adc):
        self.mol = adc.mol
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
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_mpn_energy = True
        self.nfc_orb = adc.nfc_orb
        self.nkop_chk = adc.nkop_chk
        self.kop_pick = adc.kop_pick
        self.cvs_pick = adc.cvs_pick
        self.fc_bool = adc.fc_bool
        self.ncore_proj = adc.ncore_proj
        self.ncore_proj_valence = adc.ncore_proj_valence
        self.alpha_proj = adc.alpha_proj
        self.mom_skd_iter = adc.mom_skd_iter
        keys = set(('conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy', 'max_memory', 't1', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)
    
    kernel = kernel
    get_imds = get_imds_ea
    matvec = ea_adc_matvec
    get_diag = ea_adc_diag
    compute_trans_moments = ea_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors_ea
    
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
    

    def gen_matvec(self, imds=None, eris=None, cvs=True, fc_bool=True):
        if imds is None: imds = self.get_imds(eris)
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag


class RADCIP(RADC):
    '''restricted ADC for IP energies and spectroscopic amplitudes

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

            >>> myadc = adc.RADC(mf).run()
            >>> myadcip = adc.RADC(myadc).run()

    Saved results

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
    '''
    def __init__(self, adc):
        self.mol = adc.mol
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
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_mpn_energy = True
        self.nfc_orb = adc.nfc_orb
        self.nkop_chk = adc.nkop_chk
        self.kop_npick = adc.kop_npick
        self.cvs_npick = adc.cvs_npick
        self.fc_bool = adc.fc_bool
        self.ncore_proj = adc.ncore_proj
        self.ncore_proj_valence = adc.ncore_proj_valence
        self.alpha_proj = adc.alpha_proj
        self.mom_skd_iter = adc.mom_skd_iter 
        keys = set(('conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b', 'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kernel
    get_imds = get_imds_ip
    get_diag = ip_adc_diag
    matvec = ip_adc_matvec
    compute_trans_moments = ip_compute_trans_moments
    get_trans_moments = get_trans_moments
    get_spec_factors = get_spec_factors_ip
    analyze_eigenvector_ip = analyze_eigenvector_ip

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

    def gen_matvec(self, imds=None, eris=None, cvs=False, fc_bool=True, mom_skd=False, alpha_proj=0):
        if imds is None: imds = self.get_imds(eris, fc_bool)
        diag = self.get_diag(imds, eris, cvs, fc_bool, mom_skd, alpha_proj)
        matvec = self.matvec(imds, eris, cvs, fc_bool, mom_skd, alpha_proj)
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
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    myadc = adc.ADC(mf)
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr -  -0.3220169236051954)

    myadcip = RADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(2) IP energies")
    print (e[0] - 0.5434389910483670)
    print (e[1] - 0.6240296243595950)
    print (e[2] - 0.6240296243595956)

    print("ADC(2) IP spectroscopic factors")
    print (p[0] - 1.7688097076459075)
    print (p[1] - 1.8192921131700284)
    print (p[2] - 1.8192921131700293)

    myadcea = RADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)
    print("ADC(2) EA energies")
    print (e[0] - 0.0961781923822576)
    print (e[1] - 0.1258326916409743)
    print (e[2] - 0.1380779405750178)

    print("ADC(2) EA spectroscopic factors")
    print (p[0] - 1.9832854445007961)
    print (p[1] - 1.9634368668786559)
    print (p[2] - 1.9783719593912672)

    myadc = adc.ADC(mf)
    myadc.method = "adc(3)"
    ecorr, t_amp1, t_amp2 = myadc.kernel_gs()
    print(ecorr - -0.31694173142858517)

    myadcip = RADCIP(myadc)
    e,v,p = kernel(myadcip,nroots=3)
    print("ADC(3) IP energies")
    print (e[0] - 0.5667526829981027)
    print (e[1] - 0.6099995170092525)
    print (e[2] - 0.6099995170092529)

    print("ADC(3) IP spectroscopic factors")
    print (p[0] - 1.8173191958988848)
    print (p[1] - 1.8429224413853840)
    print (p[2] - 1.8429224413853851)

    myadcea = RADCEA(myadc)
    e,v,p = kernel(myadcea,nroots=3)

    print("ADC(3) EA energies")
    print (e[0] - 0.0936790850738445)
    print (e[1] - 0.0983654552141278)
    print (e[2] - 0.1295709313652367)

    print("ADC(3) EA spectroscopic factors")
    print (p[0] - 1.8324175318668088)
    print (p[1] - 1.9840991060607487)
    print (p[2] - 1.9638550014980212)

    myadc.method = "adc(2)-x"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x IP energies")
    print (e[0] - 0.5405255360673724)
    print (e[1] - 0.6208026698756577)
    print (e[2] - 0.6208026698756582)
    print (e[3] - 0.6465332771967947)

    myadc.method_type = "ea"
    e,v,p = myadc.kernel(nroots=4)
    print("ADC(2)-x EA energies")
    print (e[0] - 0.0953065329985665)
    print (e[1] - 0.1238833070823509)
    print (e[2] - 0.1365693811939308)
    print (e[3] - 0.1365693811939316)
