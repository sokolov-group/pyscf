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
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
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
from pyscf.adc import radc_ao2mo
from pyscf.adc import dfadc
from pyscf import __config__
from pyscf import df
from pyscf import symm


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

    eris_ovvo = eris.ovvo
    ncore = adc._nocc
    nextern = adc._nvir

    n_singles = ncore * nextern

    e_core = adc.mo_energy[:ncore].copy()
    e_extern = adc.mo_energy[ncore:].copy()


    if eris is None:
        eris = adc.transform_integrals()
    einsum = np.einsum
    einsum_type = True

    v_ccee = eris.oovv
    v_cece = eris.ovvo.transpose(0,1,3,2)
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    ####000#####################
    
    M_aa  = einsum('A,AD,IL->IDLA', e_extern, np.identity(nextern), np.identity(ncore), optimize = einsum_type)
    M_aa -= einsum('L,AD,IL->IDLA', e_core, np.identity(nextern), np.identity(ncore), optimize = einsum_type)
    
    ####010#####################

    M_aa -= einsum('ILAD->IDLA', v_ccee, optimize = einsum_type).copy()
    M_aa += einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()
    
    M_aabb = einsum('LADI->IDLA', v_ceec, optimize = einsum_type).copy()

    ####020#####################

    M_aa += 2 * einsum('IiDa,LAia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= einsum('IiDa,iALa->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa += 2 * einsum('LiAa,IDia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= einsum('LiAa,iDIa->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= einsum('iIDa,LAia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('iIDa,iALa->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= einsum('iLAa,IDia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('iLAa,iDIa->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('A,LiAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('A,iLAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('A,iLAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('D,LiAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('D,iLAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('I,LiAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('I,iLAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('L,LiAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('L,iLAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('L,iLAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('a,iLAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('a,iLAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('i,IiDa,LiAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('i,IiDa,iLAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('i,LiAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('i,iIDa,LiAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,iIDa,iLAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('i,iLAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,iLAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 2 * einsum('AD,Iiab,Laib->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('AD,Iiab,Lbia->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= 2 * einsum('AD,Liab,Iaib->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('AD,Liab,Ibia->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= 2 * einsum('IL,ijAa,iDja->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('IL,ijAa,jDia->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= 2 * einsum('IL,ijDa,iAja->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
    M_aa += einsum('IL,ijDa,jAia->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
    M_aa -= einsum('A,IL,ijAa,ijDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('A,IL,ijAa,jiDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= einsum('D,IL,ijAa,ijDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += 1/2 * einsum('D,IL,ijAa,jiDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('I,AD,Iiab,Liab->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('I,AD,Iiab,Liba->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('L,AD,Iiab,Liab->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('L,AD,Iiab,Liba->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 2 * einsum('a,AD,Liab,Iiab->IDLA', e_extern, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('a,AD,Liab,Iiba->IDLA', e_extern, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('a,AD,Liba,Iiab->IDLA', e_extern, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 2 * einsum('a,AD,Liba,Iiba->IDLA', e_extern, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 2 * einsum('a,IL,ijAa,ijDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('a,IL,ijAa,jiDa->IDLA', e_extern, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('i,AD,Iiab,Liab->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,AD,Iiab,Liba->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('i,AD,Liab,Iiab->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,AD,Liab,Iiba->IDLA', e_core, np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('i,IL,ijAa,ijDa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,IL,ijAa,jiDa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('i,IL,ijDa,ijAa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,IL,ijDa,jiAa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,IL,jiAa,ijDa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('i,IL,jiAa,jiDa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa -= 1/2 * einsum('i,IL,jiDa,ijAa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    M_aa += einsum('i,IL,jiDa,jiAa->IDLA', e_core, np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
    
    
    M_aabb += 2 * einsum('IiDa,LAia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aabb -= einsum('IiDa,iALa->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aabb += 2 * einsum('LiAa,IDia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aabb -= einsum('LiAa,iDIa->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aabb -= einsum('iIDa,LAia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aabb -= einsum('iLAa,IDia->IDLA', t1_ccee, v_cece, optimize = einsum_type)
    M_aabb += einsum('A,LiAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= 1/2 * einsum('A,iLAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += einsum('D,LiAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= einsum('I,LiAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= einsum('L,LiAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('L,iLAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= einsum('a,LiAa,iIDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= einsum('a,iLAa,IiDa->IDLA', e_extern, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= einsum('i,IiDa,LiAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('i,IiDa,iLAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb -= einsum('i,LiAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('i,LiAa,iIDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('i,iIDa,LiAa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)
    M_aabb += 1/2 * einsum('i,iLAa,IiDa->IDLA', e_core, t1_ccee, t1_ccee, optimize = einsum_type)



    print("M_a", np.linalg.norm(M_aa))
    print("M_aabb",np.linalg.norm(M_aabb))
    
    M_aa = M_aa.reshape(n_singles, n_singles)
    M_aabb = M_aabb.reshape(n_singles, n_singles)

    M_ab = (M_aa, M_aabb)


    return M_ab


def get_diag(adc,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ab = adc.get_imds()

    M_aa, M_aabb = M_ab

    nocc = adc._nocc
    nvir = adc._nvir

    ij_ind = np.tril_indices(nocc, k=-1)
    ab_ind = np.tril_indices(nvir, k=-1)

    n_singles = nocc * nvir
    n_doubles = nocc * (nocc - 1) * nvir * (nvir -1) // 4

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

    D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1)).reshape((nocc,nocc,nvir,nvir))[:,:,ab_ind[0],ab_ind[1]]
    diag[s2:f2] = D_ijab[ij_ind[0],ij_ind[1]].reshape(-1)


    diag[s1:f1] = np.diagonal(M_aa)

    print("diag", np.linalg.norm(diag))
    exit()


    return diag


def matvec(adc, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    return sigma_


def get_trans_moments(adc):

    return T









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
#    get_trans_moments = get_trans_moments
    #renormalize_eigenvectors = renormalize_eigenvectors
 #   get_properties = get_properties
 #   analyze_spec_factor = analyze_spec_factor
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

    r2_vvvv = np.zeros((nocc,nvir,nvir))
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))
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
            r2_vvvv[:,a:a+k] = np.dot(r2,vvvv_p.T).reshape(nocc,-1,nvir)
            del vvvv_p
            a += k
    else:
        raise Exception("Unknown vvvv type")

    r2_vvvv = r2_vvvv.reshape(-1)

    return r2_vvvv
