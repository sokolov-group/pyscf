# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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
# Author: Terrenc Stahl <terrencestahl4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

'''
Restricted algebraic diagrammatic construction
'''
import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc
from pyscf.adc import radc_ao2mo, radc_amplitudes
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import symm
import sys

def get_imds(adc, eris=None):

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    t1 = adc.t1
    t2 = adc.t2

    t1_ccee = t2[0][:]

    t1_2 = t1[0]

    einsum = lib.einsum
    einsum_type = True

    eris_ovvo = eris.ovvo
    ncore = adc._nocc
    nextern = adc._nvir

    n_singles = ncore * nextern

    e_core = adc.mo_energy[:ncore].copy()
    e_extern = adc.mo_energy[ncore:].copy()


    if eris is None:
        eris = adc.transform_integrals()

    v_ccee = eris.oovv
    v_cece = eris.ovvo
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    occ_list = np.array(range(ncore))
    vir_list = np.array(range(nextern))
    M_ab = np.zeros((ncore*nextern, ncore*nextern))

    ####000#####################
    d_ai_a = adc.mo_energy[ncore:][:,None] - adc.mo_energy[:ncore]
    np.fill_diagonal(M_ab, d_ai_a.transpose().reshape(-1))
    M_ab = M_ab.reshape(ncore,nextern,ncore,nextern).copy()
    
    
    ####010#####################

    M_ab -= einsum('ILAD->IDLA', v_ccee, optimize = einsum_type).copy()
    M_ab += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()
    
    M_ab += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()

    ####020#####################

    M_ab += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += einsum('iIDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += einsum('iLAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab -= einsum('LAai,IiDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab += 1/2 * einsum('LAai,iIDa->IDLA',v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab += 1/2 * einsum('iAaL,IiDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab -= 1/2 * einsum('iAaL,iIDa->IDLA', v_ceec, t1_ccee, optimize = einsum_type) #
    M_ab -= einsum('LiAa,IDai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab += 1/2 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('iLAa,IDai->IDLA',t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab -= 1/2 * einsum('iLAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type) #
    M_ab -= einsum('IiDa,LAai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab += 1/2 * einsum('IiDa,iAaL->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab += 1/2 * einsum('iIDa,LAai->IDLA', t1_ccee, v_ceec, optimize = einsum_type) ##


    M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Iiab,Labi->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Lbai->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] -= 2 * einsum('Liab,Iabi->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[:,vir_list,:,vir_list] += einsum('Liab,Ibai->IL', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijAa,iDaj->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] += einsum('ijAa,jDai->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] -= 2 * einsum('ijDa,iAaj->DA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab[occ_list,:,occ_list,:] += einsum('ijDa,jAai->DA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab[occ_list,:,occ_list,:] += einsum('iAaj,ijDa->DA', v_ceec, t1_ccee, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] -= 1/2 * einsum('iAaj,jiDa->DA', v_ceec, t1_ccee, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] += einsum('ijAa,iDaj->DA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab[occ_list,:,occ_list,:] -= 1/2 * einsum('ijAa,jDai->DA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab[:,vir_list,:,vir_list] += einsum('Iabi,Liab->IL', v_ceec, t1_ccee, optimize = einsum_type)##
    M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('Iabi,Liba->IL', v_ceec, t1_ccee, optimize = einsum_type)##
    M_ab[:,vir_list,:,vir_list] += einsum('Iiab,Labi->IL', t1_ccee, v_ceec, optimize = einsum_type) ##
    M_ab[:,vir_list,:,vir_list] -= 1/2 * einsum('Iiab,Lbai->IL', t1_ccee, v_ceec, optimize = einsum_type) ##
    
    
    M_ab += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_ab -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)

    M_ab -= einsum('LiAa,IDai->IDLA',t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#
    M_ab += 1/2 * einsum('iLAa,IDai->IDLA', t1_ccee, v_ceec, optimize = einsum_type)#


    print("M_ab", np.linalg.norm(M_ab))

    
    M_ab = M_ab.reshape(n_singles, n_singles)



    return M_ab


def get_diag(adc,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ = adc.get_imds()

    M_ = M_ab

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles
    diag = np.zeros(dim)

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None]+e_occ
    d_ab = e_vir[:,None]+e_vir

    D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))
    diag[s2:f2] = D_ijab.reshape(-1)

    diag[s1:f1] = np.diagonal(M_)

    return diag


def matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    
    if M_ab is None:
        M_  = adc.get_imds()

    M_ = M_ab
    
    if eris is None:
        eris = adc.transform_integrals()

    einsum = lib.einsum
    einsum_type = True

    v_ccee = eris.oovv
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo

    nocc = adc._nocc
    nvir = adc._nvir

    ij_ind_a = np.tril_indices(nocc, k=-1)
    ab_ind_a = np.tril_indices(nvir, k=-1)

    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles
    diag = np.zeros(dim)

    e_occ = adc.mo_energy[:nocc]
    e_vir = adc.mo_energy[nocc:]

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    d_ij = e_occ[:,None]+e_occ
    d_ab = e_vir[:,None]+e_vir

    d_ia = e_occ[:,None]+e_vir
    e_core = adc.mo_energy[:nocc].copy()
    e_extern = adc.mo_energy[nocc:].copy()


    #Calculate sigma vector
    #@profile
    #@profile
    def sigma_(r):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)
       
        r1 = r[s1:f1]

        Y = r1.reshape(nocc, nvir).copy()
        
        r2 = r[s2:f2].reshape(nocc,nocc,nvir,nvir).copy()

        s = np.zeros(dim)

        s[s1:f1] = lib.einsum('ab,b->a',M_,r1, optimize = True)

        D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))
        s[s2:f2] = (D_ijab.reshape(-1))*r[s2:f2]
        del D_ijab

        if isinstance(eris.ovvv, type(None)):
            M_11Y0 = np.zeros((nocc,nocc,nvir,nvir))
            chnk_size = radc_ao2mo.calculate_chunk_size(adc)
            a = 0
            for p in range(0,nocc,chnk_size):
                v_ceee = dfadc.get_ovvv_df(adc, eris.Lov, eris.Lvv, p, chnk_size).reshape(-1,nvir,nvir,nvir)
                k = v_ceee.shape[0]
                M_11Y0[:,a:a+k,:,:] += einsum('Ia,JDaC->IJCD', Y, v_ceee, optimize = einsum_type)
                M_11Y0[a:a+k,:,:,:] += einsum('Ja,ICaD->IJCD', Y, v_ceee, optimize = einsum_type)

                s[s1:f1] += -einsum('Iiab,iabD->ID', r2[:,a:a+k,:,:], v_ceee, optimize = einsum_type).reshape(-1)
                s[s1:f1] += 2*einsum('Iiab,ibDa->ID', r2[:,a:a+k,:,:], v_ceee, optimize = einsum_type).reshape(-1)
                del v_ceee
                a += k
            s[s2:f2] += M_11Y0.reshape(-1)
            del M_11Y0
        else:
            v_ceee = radc_ao2mo.unpack_eri_1(eris.ovvv, nvir)
            M_11Y0 = einsum('Ia,JDaC->IJCD', Y, v_ceee, optimize = einsum_type)
            M_11Y0 += einsum('Ja,ICaD->IJCD', Y, v_ceee, optimize = einsum_type)
            s[s2:f2] += M_11Y0.reshape(-1)


            M_01Y1 = -einsum('Iiab,iabD->ID', r2, v_ceee, optimize = einsum_type)
            M_01Y1 += 2*einsum('Iiab,ibDa->ID', r2, v_ceee, optimize = einsum_type)
            s[s1:f1] += M_01Y1.reshape(-1)
            del M_11Y0
            del M_01Y1

        M_11Y0 = -einsum('iC,JDIi->IJCD', Y, v_cecc, optimize = einsum_type)
        M_11Y0 -= einsum('iD,ICJi->IJCD', Y, v_cecc, optimize = einsum_type)
        s[s2:f2] += M_11Y0.reshape(-1)

        M_01Y1 = -2*einsum('ijDa,jaiI->ID', r2, v_cecc, optimize = einsum_type)
        M_01Y1 += einsum('ijDa,iajI->ID', r2, v_cecc, optimize = einsum_type)
        s[s1:f1] += M_01Y1.reshape(-1)

        if (adc.method == "adc(2)-x"):
            del Y
            Y = r2.copy()

            if isinstance(eris.ovvv, type(None)):
                s[s2:f2] += radc_amplitudes.contract_ladder(adc,Y,eris.Lvv).reshape(-1)
            else:
                v_eeee = eris.vvvv.reshape(nvir, nvir, nvir,nvir)
                M_1Y1_aa  = einsum('IJab,CDab->IJCD', Y, v_eeee, optimize = einsum_type)
                s[s2:f2] += M_1Y1_aa.reshape(-1)
                del M_1Y1_aa
            
                
            M_1Y1_aa = 2 * einsum('IiCa,JDai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa -= einsum('IiCa,iJDa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa -= einsum('IiaC,JDai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa -= einsum('IiaD,iJCa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa += 2 * einsum('JiDa,ICai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa -= einsum('JiDa,iICa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa -= einsum('JiaC,iIDa->IJCD', Y, v_ccee, optimize = einsum_type)
            M_1Y1_aa -= einsum('JiaD,ICai->IJCD', Y, v_ceec, optimize = einsum_type)
            M_1Y1_aa += einsum('ijCD,IiJj->IJCD', Y, v_cccc, optimize = einsum_type)






            s[s2:f2] += M_1Y1_aa.reshape(-1)




        return s



    return sigma_





def renormalize_eigenvectors(adc, nroots=1):

    nocc = adc._nocc
    nvir = adc._nvir

    n_singles = nocc * nvir

    U = adc.U
    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nocc,nocc,nvir,nvir)
        UdotU = np.dot(U1, U1) + 2.*np.dot(U2.ravel(), U2.ravel()) - 4.*np.dot(U2.ravel(), U2.transpose(0,1,3,2).ravel())
        U[:,I] /= np.sqrt(UdotU)

    return U




def get_X(adc):
    U = renormalize_eigenvectors(adc)

    #U = adc.U
    U = U.T.copy()

    nroots = U.shape[0]

    dip_ints = -adc.mol.intor('int1e_r',comp=3)
    dm = np.zeros_like((dip_ints))
    for i in range(dip_ints.shape[0]):
        dip = dip_ints[i,:,:]
        dm[i,:,:] = np.dot(adc.mo_coeff.T,np.dot(dip,adc.mo_coeff))

   

    x = np.array([])
    
    nocc = adc._nocc
    nvir = adc._nvir

    nmo = nocc + nvir


    n_singles = nocc * nvir
    n_doubles = nocc * nocc * nvir * nvir


    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    t1 = adc.t1
    t2 = adc.t2

    t1_ccee = t2[0][:]
    t2_ccee = t2[1][:]

    t2_ce = t1[0]


    einsum = lib.einsum
    einsum_type = True

    TY = np.zeros((nmo, nmo))

    TY_ = []
    x = np.array([])

    for r in range(U.shape[0]):


        r1 = U[r][s1:f1]

        Y = r1.reshape(nocc, nvir).copy()
        #print(np.linalg.norm(Y))
        #exit()
        
       # print(np.linalg.norm(Y))
       # exit()
        r2 = U[r][s2:f2].reshape(nocc,nocc,nvir,nvir).copy()

        TY[:nocc,nocc:] = Y.copy()

        TY[nocc:,:nocc]  = einsum('ia,LiAa->AL', Y, t1_ccee, optimize = einsum_type)
        TY[nocc:,:nocc] -= einsum('ia,iLAa->AL', Y, t1_ccee, optimize = einsum_type)

        TY[:nocc,:nocc] =- einsum('Iiab,Liab->IL', r2, t1_ccee, optimize = einsum_type)
        TY[:nocc,:nocc] += einsum('Iiab,Liba->IL', r2, t1_ccee, optimize = einsum_type)


        TY[nocc:,nocc:]  = 2 * einsum('ijCa,ijAa->AC', r2, t1_ccee, optimize = einsum_type)
        TY[nocc:,nocc:] -= 2 * einsum('ijCa,jiAa->AC', r2, t1_ccee, optimize = einsum_type)


        TY[:nocc,:nocc] -= einsum('Iiab,Liab->IL', r2, t1_ccee, optimize = einsum_type)
        TY[nocc:,nocc:] += einsum('ijCa,ijAa->AC', r2, t1_ccee, optimize = einsum_type)


        TY[:nocc,:nocc] -= einsum('Ia,La->IL', Y, t2_ce, optimize = einsum_type)
        TY[nocc:,nocc:] += einsum('iC,iA->AC', Y, t2_ce, optimize = einsum_type)

        TY[:nocc,nocc:] -= einsum('Ia,ijab,ijCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('Ia,ijab,jiCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= einsum('iC,ijab,Ijab->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('iC,ijab,Ijba->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += einsum('ia,ijab,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijab,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] -= 1/2 * einsum('ia,ijba,IjCb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)
        TY[:nocc,nocc:] += 1/2 * einsum('ia,ijba,jICb->IC', Y, t1_ccee, t1_ccee, optimize = einsum_type)

        TY[nocc:,:nocc] += einsum('ia,LiAa->AL', Y, t2_ccee, optimize = einsum_type)
        TY[nocc:,:nocc] -= einsum('ia,iLAa->AL', Y, t2_ccee, optimize = einsum_type)


        dx = lib.einsum("rqp,qp->r", dm, TY, optimize = True)

        TY_ = np.append(TY_,TY)

        x = np.append(x,dx)
    x = x.reshape(nroots, 3)

    return TY_, x

#def analyze_spec_factor(adc):




def get_properties(adc,nroots):

    TY, dx  = adc.get_X()


    X = TY.reshape(nroots,-1)

    P = np.square(dx.T)*adc.E*(2/3)
    P = P[0] + P[1] + P[2]

    return P, X


def analyze(myadc):

    if myadc.compute_properties:

        header = ("\n*************************************************************"
                  "\n            Spectroscopic factors analysis summary"
                  "\n*************************************************************")
        logger.info(myadc, header)

        myadc.analyze_spec_factor()

def compute_dyson_mo(myadc):

    X = myadc.X

    if X is None:
        nroots = myadc.U.shape[1]
        P,X = myadc.get_properties(nroots)

    nroots = X.shape[1]
    dyson_mo = np.dot(myadc.mo_coeff,X)

    return dyson_mo


class RADCEE(radc.RADC):
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
            Space size to hold trial vectors for Davidson iterative
            diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcea = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float
            number. If nroots > 1, it is a list of floats for the lowest
            nroots eigenvalues.
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
        self._nmo = adc._nmo
        self.mo_coeff = adc.mo_coeff
        self.mo_energy = adc.mo_energy
        self.nmo = adc._nmo
        self.transform_integrals = adc.transform_integrals
        self.with_df = adc.with_df
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments
        self.E = None
        self.U = None
        self.P = None
        self.X = None
        self.evec_print_tol = adc.evec_print_tol
        self.spec_factor_print_tol = adc.spec_factor_print_tol

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
                    'mo_energy', 'max_memory', 't1', 'max_space', 't2',
                    'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = radc.kernel
    get_imds = get_imds
    matvec = matvec
    get_diag = get_diag
    #get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
    get_X = get_X
    get_properties = get_properties
#    analyze_spec_factor = analyze_spec_factor
 #   analyze_eigenvector = analyze_eigenvector
 #   analyze = analyze
 #   compute_dyson_mo = compute_dyson_mo

    def get_init_guess(self, nroots=1, diag=None, ascending = True):
        if diag is None :
            diag = self.get_diag()
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
        diag = self.get_diag(imds, eris)
        matvec = self.matvec(imds, eris)
        return matvec, diag


def contract_r_vvvv(myadc,r2,vvvv):

    nocc = myadc._nocc
    nvir = myadc._nvir

    r2_vvvv = np.zeros((nocc,nocc,nvir,nvir))
    r2 = np.ascontiguousarray(r2.reshape(nocc*nocc,-1))
    chnk_size = radc_ao2mo.calculate_chunk_size(myadc)

    a = 0
    if isinstance(vvvv, list):
        for dataset in vvvv:
            k = dataset.shape[0]
            dataset = dataset[:].reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,dataset.T).reshape(nocc,-1,nvir)
            del dataset
            a += k
    elif getattr(myadc, 'with_df', None):
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(myadc, vvvv, p, chnk_size)
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc*nocc,-1)
            del vvvv_p
            a += k
    else:
        raise Exception("Unknown vvvv type")

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv
