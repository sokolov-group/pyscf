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

import time
import numpy as np
import pyscf.ao2mo as ao2mo
import pyscf.adc
import pyscf.adc.radc
from pyscf.adc import radc_ao2mo
import itertools
import concurrent
#from pathos.multiprocessing import ProcessingPool
from multiprocessing.pool import ThreadPool as Pool

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import mp
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_rhf_amplitudes
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx,_padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa

import h5py
import tempfile
import sys
import tracemalloc 
 
np.set_printoptions(threshold=sys.maxsize)
# Note : All interals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba

def kernel(adc, nroots=1, guess=None, eris=None, kptlist=None, verbose=None):

    tracemalloc.start()
    adc.method = adc.method.lower()
    if adc.method not in ("adc(2)", "adc(2)-x","adc(3)"):
        raise NotImplementedError(adc.method)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    
    if eris is None:
        eris = adc.transform_integrals()
    print(f'[memalloc current+max Kernel-ERI-DF-post-transoform_integrals [GB] = {np.array(tracemalloc.get_traced_memory())/1024**3}')
    def diag_full_M(matvec, diag, kpt_i, nroots):
        r = np.identity(diag.size)
        M = np.zeros((diag.size, diag.size))
        for i in range(diag.size):
             M[:,i] = matvec(r[:,i])

        print("Hermiticity: ")
        print("Full M: ", np.linalg.norm(M - M.T))
        
        if nroots == 'fullproj':
            singles = adc.nocc
        if nroots == 'fulleff':
            singles = adc.ncvs

        M_i_j = M[:singles,:singles]
        M_i_ajk = M[:singles,singles:]   
        M_ajk_i = M[singles:,:singles]
        M_ajk_bli = M[singles:,singles:]
        print("ADC Method Type: ", adc.method_type)
        print('M_i_j - M_i_j.T: ', np.linalg.norm(M_i_j - M_i_j.T))
        print('M_ajk_bli - M_ajk_bli.T: ', np.linalg.norm(M_ajk_bli - M_ajk_bli))
        print('M_ajk_i - M_i_ajk.T: ', np.linalg.norm(M_ajk_i - M_i_ajk.T))
        print('M norm: ', np.linalg.norm(M))
        print('M_i_j norm: ', np.linalg.norm(M_i_j))
        print('M_i_ajk norm: ', np.linalg.norm(M_i_ajk))
        print('M_ajk_i norm: ', np.linalg.norm(M_ajk_i))
        print('M_i_ajk sum: ', np.sum(M_i_ajk))
        print('M_ajk_i sum: ', np.sum(M_ajk_i))
        print('M_ajk_bli norm: ', np.linalg.norm(M_ajk_bli))

        E,U = np.linalg.eig(M)
        M_size = E.size    
        U = U[:, E != 0]
        E = E[E != 0]
        idx_E_sort = np.argsort(E)
        E = E[idx_E_sort]
        U[:, idx_E_sort]
        print(f'Energies for kpt_i = {kpt_i}: ')
        with np.printoptions(threshold=np.inf):
            print(np.column_stack(np.unique(np.around(E, decimals=8), return_counts=True))) 
        #P, X = get_properties(adc, E.size ,U)
        #idx_P_sort = np.argsort(-P)
        #P = P[idx_P_sort]
        #P_100 = P[:100]
        P = None

        #Mat_i_ajk = np.dot(M_i_ajk.conj().T, M_i_ajk)
        Mat_i_ajk = np.dot(M_i_ajk.conj(), M_i_ajk.T)
        #Mat_ajk_i = np.dot(M_ajk_i, M_ajk_i.conj().T)
        Mat_ajk_i = np.dot(M_ajk_i.T, M_ajk_i.conj())
        E_i_ajk,_ = np.linalg.eig(Mat_i_ajk)
        E_ajk_i,_ = np.linalg.eigh(Mat_ajk_i)
        print(f'eigvals for Mat_i_ajk {Mat_i_ajk.shape}: ')
        with np.printoptions(threshold=np.inf):
            print(np.column_stack(np.unique(np.around(E_i_ajk, decimals=8), return_counts=True))) 
        print(f'eigvals for Mat_ajk_i {Mat_ajk_i.shape}: ')
        with np.printoptions(threshold=np.inf):
            print(np.column_stack(np.unique(np.around(E_ajk_i, decimals=8), return_counts=True))) 

        E_P_size = (E, P, M_size)
        return E_P_size 
 
    #def kernel_micro(k):

    #    matvec, diag = adc.gen_matvec(k, imds, eris)
    #    diag_full_M(matvec, diag, k)
    #    exit()

    imds = adc.get_imds(eris)
    if nroots == ('fullproj' or 'fulleff'):
        for k in range(adc.nkpts):
            matvec, diag = adc.gen_matvec(k, imds, eris)
            diag_full_M(matvec, diag, k, nroots)
            exit()

    size = adc.vector_size()
    nroots = min(nroots,size)
    nkpts = adc.nkpts
    nmo = adc.nmo

    if kptlist is None:
        kptlist = range(nkpts)

    #dtype = np.result_type(adc.t2[0])
    dtype = np.result_type(eris.Loo.dtype)

    evals = np.zeros((len(kptlist),nroots), np.float64)
    evecs = np.zeros((len(kptlist),nroots,size), dtype)
    conv = np.zeros((len(kptlist),nroots), np.bool_)
    P = np.zeros((len(kptlist),nroots), np.float64)
    X = np.zeros((len(kptlist),nmo,nroots), dtype)

    #imds = adc.get_imds(eris)


    #for k, kshift in enumerate(kptlist):
    #    matvec, diag = adc.gen_matvec(kshift, imds, eris)

    #    guess = adc.get_init_guess(nroots, diag, ascending=True)

    #    conv_k,evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(
    #            lambda xs : [matvec(x) for x in xs], guess, diag,
    #            nroots=nroots, verbose=log, tol=adc.conv_tol,
    #            max_cycle=adc.max_cycle, max_space=adc.max_space,
    #            tol_residual=adc.tol_residual)

    #    evals_k = evals_k.real
    #    evals[k] = evals_k
    #    evecs[k] = evecs_k
    #    conv[k] = conv_k.real

    #    U = np.array(evecs[k]).T.copy()

    #    if adc.compute_properties:
    #        spec_fac,spec_amp = adc.get_properties(kshift,U,nroots)
    #        P[k] = spec_fac
    #        X[k] = spec_amp

    #if nroots == 'full':

    def kernel_micro(k):

        matvec, diag = adc.gen_matvec(k, imds, eris)
        #print(f'diag.size = {diag.size}')
        #ones_temp = np.ones(diag.size)
        #out_norm = np.linalg.norm(matvec(diag))
        #print(f'out norm for kpt = {k} is = {out_norm} ')

        guess = adc.get_init_guess(nroots, diag, ascending=True)

        conv_k,evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(
                lambda xs : [matvec(x) for x in xs], guess, diag,
                nroots=nroots, verbose=log, tol=adc.conv_tol,
                max_cycle=adc.max_cycle, max_space=adc.max_space,
                tol_residual=adc.tol_residual)

        print(f'dtype = {dtype}')
        print(f'diag.dtype = {diag.dtype}')
        print(f'matvec(diag).dtype = {matvec(diag).dtype}')
        print(f'imds.dtype = {imds.dtype}')
        print(f'evals_k.dtype = {evals_k.dtype}')

        #U = np.array(evecs[k]).T.copy()
        U = np.array(evecs_k).T.copy()

        #return conv_k, evals_k, evecs_k
        if adc.compute_properties:
            P_k, X_k = adc.get_properties(k,U,nroots)

        if adc.compute_properties:
            return conv_k, evals_k, evecs_k, P_k, X_k
        else:
            return conv_k, evals_k, evecs_k

    #result_list = []
    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    for k in kptlist:
    #        result_list.append( executor.submit(kernel_micro, k) )
    #result_list = ProcessingPool().map(kernel_micro, [kptlist])
    #result_list = Pool().map(kernel_micro, [[k for k in kptlist]])
    #for k in kptlist:
    #    if adc.compute_properties:
    #        conv_k, evals_k, evecs_k, P_k, X_k = kernel_micro(k)
    #        P[k] = P_k
    #        X[k] = X_k
    #    else:
    #        conv_k, evals_k, evecs_k = kernel_micro(k)
    #    conv[k] = conv_k.real
    #    evals[k] = evals_k.real
    #    evecs[k] = evecs_k#.real

    k = 0
    #for result_k in result_list:
    cput1 = (logger.process_clock(), logger.perf_counter())
    #for k in kptlist:
    #    kernel_micro(k)
    #print('matvec operations finished')
    #exit()
    for k in kptlist:
    #for result_k in Pool().map(kernel_micro, kptlist):
        if adc.compute_properties:
            conv_k, evals_k, evecs_k, P_k, X_k = kernel_micro(k)#result_k#.result()
            P[k] = P_k
            X[k] = X_k
        else:
            conv_k, evals_k, evecs_k = kernel_micro(k)#result_k#.result()
        conv[k] = conv_k.real
        evals[k] = evals_k.real
        evecs[k] = evecs_k#.real
        k += 1
    nfalse = np.shape(conv)[0] - np.sum(conv)
    print(f'[memalloc current+max Kernel-ERI-DF-post-Davidson [GB] = {np.array(tracemalloc.get_traced_memory())/1024**3}')
    tracemalloc.stop()

    print(f'Lov shape = {eris.Lov.shape}')
    str = ("\n*************************************************************"
           "\n            ADC calculation summary"
           "\n*************************************************************")
    logger.info(adc, str)
    if nfalse >= 1:
        #logger.warn(adc, "Davidson iterations for " + str(nfalse) + " root(s) not converged\n")
        logger.warn(adc, f'Davidson iterations for {nfalse} root(s) not converged\n')

    for k, kshift in enumerate(kptlist):
        for n in range(nroots):
            print_string = ('%s k-point %d | root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                            (adc.method, kshift, n, evals[k][n], evals[k][n]*27.2114))
            if adc.compute_properties:
                print_string += ("|  Spec factors = %10.8f  " % P[k][n])
            print_string += ("|  conv = %s" % conv[k][n].real)
            logger.info(adc, print_string)

    log.timer('ADC', *cput0)
    log.timer('ADC kernel micro', *cput1)

    return evals, evecs, P, X


class RADC(pyscf.adc.radc.RADC):

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):

        from pyscf.pbc.cc.ccsd import _adjust_occ
        assert (isinstance(mf, scf.khf.KSCF))

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        self._scf = mf
        self.kpts = self._scf.kpts
        self.exxdiv = self._scf.exxdiv
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.method = "adc(2)"
        self.method_type = "ip"

        self.max_space = getattr(__config__, 'adc_kadc_RADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_kadc_RADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_kadc_RADC_conv_tol', 1e-7)
        self.tol_residual = getattr(__config__, 'adc_kadc_RADC_tol_res', 1e-4)
        self.scf_energy = mf.e_tot

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.cell = self._scf.cell
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.frozen = frozen
        self.compute_properties = True
        self.approx_trans_moments = True

        self.Lpq_contract = False
        self.ncvs = None
        self.ncvs_proj = None
        self.eris_direct = False
        self.cvs_compact = True
        self._nocc = None
        self._nmo = None
        self._nvir = None
        self.nkop_chk = False
        self.kop_npick = False

        self.t1 = None
        self.t2 = None
        self.e_corr = None
        self.chnk_size = None
        self.imds = lambda:None

        self.keep_exxdiv = False
        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff',
                    'mol', 'mo_energy', 'max_memory', 'incore_complete',
                    'scf_energy', 'e_tot', 't1', 'frozen', 'chkfile',
                    'max_space', 't2', 'mo_occ', 'max_cycle','kpts', 'khelper'))

        self._keys = set(self.__dict__.keys()).union(keys)

        self.mo_energy = mf.mo_energy

    transform_integrals = kadc_ao2mo.transform_integrals_incore
    #compute_amplitudes = kadc_rhf_amplitudes.compute_amplitudes
    #compute_energy = kadc_rhf_amplitudes.compute_energy
    #compute_amplitudes_energy = kadc_rhf_amplitudes.compute_amplitudes_energy
    get_chnk_size = kadc_ao2mo.calculate_chunk_size

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()


    get_nocc = get_nocc
    get_nmo = get_nmo


    def kernel_gs(self):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        #ncvs = self.ncvs
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        nkpts = self.nkpts
        mem_incore = nkpts ** 3 * (nocc + nvir) ** 4
        mem_incore *= 4
        mem_incore *= 16 /1e6
        mem_now = lib.current_memory()[0]

        if isinstance(self._scf.with_df, df.GDF):
            self.chnk_size = self.get_chnk_size()
            self.with_df = self._scf.with_df
            def df_transform():
                return kadc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (mem_incore+mem_now >= self.max_memory and not self.incore_complete):
            def outcore_transform():
                return kadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()
        #self.e_corr,self.t1,self.t2 = kadc_rhf_amplitudes.compute_amplitudes_energy(
        #    self, eris=eris, verbose=self.verbose)
        #print ("MPn:",self.e_corr)
        self.e_corr, self.t1,self.t2 = None
        
        #self._finalize()
        return self.e_corr, self.t1,self.t2

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        assert(self.mo_coeff is not None)
        assert(self.mo_occ is not None)

        self.method = self.method.lower()
        if self.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
            raise NotImplementedError(self.method)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags_gs()

        #ncvs = self.ncvs
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        nkpts = self.nkpts
        mem_incore = nkpts ** 3 * (nocc + nvir) ** 4
        mem_incore *= 4
        mem_incore *= 16 /1e6
        mem_now = lib.current_memory()[0]

        if isinstance(self._scf.with_df, df.GDF):
            self.chnk_size = self.get_chnk_size()
            self.with_df = self._scf.with_df
            def df_transform():
                return kadc_ao2mo.transform_integrals_df(self)
            self.transform_integrals = df_transform
        elif (mem_incore+mem_now >= self.max_memory and not self.incore_complete):
            def outcore_transform():
                return kadc_ao2mo.transform_integrals_outcore(self)
            self.transform_integrals = outcore_transform

        eris = self.transform_integrals()

        if not myadc.eris_direct:
            self.e_corr, self.t1, self.t2 = kadc_rhf_amplitudes.compute_amplitudes_energy(
                self, eris=eris, verbose=self.verbose)
        else:
            self.e_corr = kadc_rhf_amplitudes.compute_amplitudes_energy(
                self, eris=eris, verbose=self.verbose)

        print ("MPn:",self.e_corr)
        self._finalize()

        self.method_type = self.method_type.lower()
        self.ncvs_proj = self.ncvs_proj
        self.ncvs = self.ncvs
        self.eris_direct = self.eris_direct
        self.cvs_compact = self.cvs_compact

        print(f'value of Lpq_contract == {self.Lpq_contract}')
        if(self.method_type == "ea"):
            e_exc, v_exc, spec_fac, x, adc_es = self.ea_adc(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        elif(self.method_type == "ip" and self.ncvs == None and self.Lpq_contract == False):
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        elif(self.method_type == "ip" and type(self.ncvs) == int and self.ncvs > 0):
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc_cvs(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)

        elif(self.method_type == "ip" and self.Lpq_contract == True):
            print('executing Lpq_contract elif statement')
            e_exc, v_exc, spec_fac, x, adc_es = self.ip_adc_df(
                nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)
        else:
            raise NotImplementedError(self.method_type)
        self._adc_es = adc_es
        return e_exc, v_exc, spec_fac, x

    def ip_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ip
        adc_es = kadc_rhf_ip.RADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ea_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ea
        adc_es = kadc_rhf_ea.RADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_adc_cvs(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ip_cvs
        adc_es = kadc_rhf_ip_cvs.RADCIPCVS(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, adc_es

    def ip_adc_df(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ip_df
        adc_es = kadc_rhf_ip_df.RADCIPDF(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, adc_es

    def density_fit(self, auxbasis=None, with_df=None):
        if with_df is None:
            self.with_df = df.DF(self._scf.mol)
            self.with_df.max_memory = self.max_memory
            self.with_df.stdout = self.stdout
            self.with_df.verbose = self.verbose
            if auxbasis is None:
                self.with_df.auxbasis = self._scf.with_df.auxbasis
            else:
                self.with_df.auxbasis = auxbasis
        else:
            self.with_df = with_df
        return self
