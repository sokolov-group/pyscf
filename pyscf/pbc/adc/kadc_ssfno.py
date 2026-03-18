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
# Author: Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import time
import numpy as np
import pyscf.ao2mo as ao2mo
import pyscf.adc
import pyscf.adc.radc
from pyscf.adc import radc_ao2mo
import itertools

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
from pyscf.data.nist import HARTREE2EV

import h5py
import tempfile

class RFNOADC(RADC):
    #J. Chem. Phys. 159, 084113 (2023)
    _keys = RADC._keys | {'delta_e','e_can','v_can','e_corr_can',
                          'rdm1_ss','trans_guess','mode','ref_state',
                          'if_adc2_guess','div_pct_orb','if_cc','delta_e_corr'
                          }

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.delta_e = None
        self.delta_e_corr = None
        self.method = "adc(3)"
        self.e_can = None
        self.v_can = None
        self.e_corr_can = None
        self.rdm1_ss = None
        self.trans_guess = False
        self.mode = "min"
        self.ref_state = None
        self.div_pct_orb = 0.70
        self.if_adc2_guess = False
        self.if_cc = False

    def kernel_gs(self, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.method != "adc(3)":
            raise NotImplementedError(self.method)
        logger.info(self, "Do fno kadc calculation")
        self.ref_state = None

        self.make_ss_rdm1(log, cput0, if_gs=True)
        log.timer('make gs rdm1', *cput0)
        self.make_fno(self.rdm1_ss, self._scf, log, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)
        eris=self.compute_correction(if_gs=True)
        logger.info(self, 'current use %d MB',lib.current_memory()[0])
        _,t1,t2 = RADC.kernel_gs(self, eris=eris)
        logger.info(self, 'current use %d MB',lib.current_memory()[0])
        log.timer(f'MP{self.method[4]}', *cput0)
        logger.info(self, 'current use %d MB',lib.current_memory()[0])

        self.e_corr = self.e_corr + self.delta_e_corr

        msg = ("\n*************************************************************"
            "\n            FNOMP calculation summary"
            "\n*************************************************************")
        logger.info(self, msg)
        logger.info(self, 'FNO MP%s correlation energy of reference state (a.u.) = %.8f',
                    self.method[4], self.e_corr)
        log.timer('RFNOMP', *cput0)
        return self.e_corr, t1, t2

    def kernel(self, nroots=1, guess=None, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None, kptlist=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.ref_state is None:
            logger.info(self, "Do fno kadc calculation")
        elif (isinstance(self.ref_state, int) and 0<self.ref_state<=nroots) or \
                (hasattr(self.ref_state, '__len__') and len(self.ref_state) == 2) :
            logger.info(self, "Do ss-fno kadc calculation")
        else:
            raise ValueError("self.ref_state should be an int type or or a array-like object with two elements")

        self.make_ss_rdm1(log, cput0, nroots, guess, kptlist)
        log.timer('make ss rdm1', *cput0)
        self.make_fno(self.rdm1_ss, self._scf, log, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)
        logger.info(self, 'current use %d MB',lib.current_memory()[0])
        if self.method == "adc(2)-x":
            self.if_corr = False
        self.if_div = False
        self.ext_vir = 0
        eris,v2_ssfno = self.compute_correction(nroots, guess, kptlist=kptlist)
        logger.info(self, 'current use %d MB',lib.current_memory()[0])

        if self.if_cc:
            from pyscf.pbc import cc
            from pyscf.pbc.cc import eom_kccsd_rhf as eom_krccsd
            mycc = cc.KRCCSD(self._scf, frozen=self.frozen, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            mycc.conv_tol = self.conv_tol
            mycc.conv_tol_normt = self.tol_residual
            mycc.max_cycle = self.max_cycle
            mycc.verbose = self.verbose
            self.e_corr,_,_ = mycc.kernel()
            if self.method_type == "ea":
                myeom = eom_krccsd.EOMEA(mycc)
            elif self.method_type == "ip":
                myeom = eom_krccsd.EOMIP(mycc)
            myeom.max_space = self.max_space
            e_exc,v_exc = myeom.kernel(nroots=nroots, kptlist=kptlist)
            spec_fac, x = None, None
        elif self.if_adc2_guess:
            e_exc, v_exc, spec_fac, x = RADC.kernel(self, nroots, guess=v2_ssfno, eris=eris, kptlist=kptlist)
        else:
            e_exc, v_exc, spec_fac, x = RADC.kernel(self, nroots, guess, eris=eris, kptlist=kptlist)
        log.timer(f'ADC{self.method[4]}', *cput0)

        e_exc = e_exc + self.delta_e
        self.e_corr = self.e_corr + self.delta_e_corr

        if self.if_cc:
            msg = ("\n*************************************************************"
                "\n            FNOCC calculation summary"
                "\n*************************************************************")
        else:
            msg = ("\n*************************************************************"
                "\n            FNOADC calculation summary"
                "\n*************************************************************")
        logger.info(self, msg)
        logger.info(self, 'FNO MP%s correlation energy of reference state (a.u.) = %.8f',
                    self.method[4], self.e_corr)

        for k, kshift in enumerate(kptlist):
            sort_indices = np.argsort(e_exc[k])
            e_exc[k] = e_exc[k][sort_indices]
            v_exc[k] = v_exc[k][sort_indices]
            if self.compute_properties and not self.if_cc:
                spec_fac[k] = spec_fac[k][sort_indices]
                x[k] = x[k][:, sort_indices]
            for n in range(nroots):
                print_string = ('%s k-point %d | root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                                (self.method, kshift, n, e_exc[k][n], e_exc[k][n]*HARTREE2EV))
                if self.compute_properties and not self.if_cc:
                    print_string += ("|  Spec factors = %10.8f  " % spec_fac[k][n])
                logger.info(self, print_string)
        log.timer('RFNOADC', *cput0)
        return e_exc, v_exc, spec_fac, x

    def compute_correction(self, nroots=None, guess=None, kptlist=None, if_gs=False):
        method_tmp = self.method
        self.method = "adc(2)"
        self.if_heri_eris = True
        if if_gs:
            _,_,_,eris = RADC.kernel_gs(self)
        else:
            e2_ssfno,v2_ssfno,_,_,eris = RADC.kernel(self, nroots, guess=guess, kptlist=kptlist)
            self.delta_e = self.e_can - e2_ssfno
        self.delta_e_corr = self.e_corr_can - self.e_corr
        self.method = method_tmp
        self.if_heri_eris = False
        self.t1 = None
        self.t2 = None
        self._adc_es = None
        def incore_transform():
            return kadc_ao2mo.transform_integrals_incore(self)
        self.transform_integrals = incore_transform
        if if_gs:
            return eris
        else:
            return eris,v2_ssfno

    def make_ss_rdm1(self, log, cput0, nroots=None, guess=None, kptlist=None, if_gs=False):
        method_tmp = self.method
        naf_tmp = self.if_naf
        self.method = "adc(2)"
        self.if_naf = False
        if if_gs:
            _,_,_ = RADC.kernel_gs(self)
        else:
            self.e_can,self.v_can,_,_ = RADC.kernel(self,nroots,guess=guess,kptlist=kptlist,pct_orb=self.div_pct_orb)
        log.info('current use %d MB',lib.current_memory()[0])
        self.e_corr_can = self.e_corr
        if self.ref_state is not None:
            if isinstance(self.ref_state,(int, np.integer)):
                idx = np.argsort(self.e_can.ravel())
                sidx = idx[self.ref_state - 1]% self.nkpts
                kidx = idx[self.ref_state - 1]// self.nkpts
            elif hasattr(self.ref_state, '__len__'):
                if len(self.ref_state) != 2:
                    raise ValueError
                elif hasattr(self.ref_state[0], '__len__'):
                    (sidx,kidx) = (self.ref_state[0],self.ref_state[1])
                else:
                    (sidx,kidx) = ([self.ref_state[0]],self.ref_state[1])
            log.info(f"the specific state is {sidx} with kidx {kidx}")
            es_DM = self.make_rdm1(root=sidx,kptlist=[kidx],if_ss=True)
            self.rdm1_ss = np.zeros_like(es_DM[0][0])
            for i in range(len(sidx)):
                self.rdm1_ss += es_DM[i][0]/len(sidx)
            log.info('current use %d MB',lib.current_memory()[0])
            log.timer('make ss rdm1', *cput0)
        else:
            self.rdm1_ss = self.make_ref_rdm1()
            log.info('current use %d MB',lib.current_memory()[0])
            log.timer('make ref rdm1', *cput0)
        self.method = method_tmp
        self.if_naf = naf_tmp
        def incore_transform():
            return kadc_ao2mo.transform_integrals_incore(self)
        self.transform_integrals = incore_transform
        self.t1 = None
        self.t2 = None
        self._adc_es = None
        self.imds.t2_1_vvvv = None

    def make_fno(self, rdm1_ss, mf, log, thresh=None, pct_occ=None, nvir_act=None):
        nocc = mf.mol.nelectron//2
        masks = mo_splitter(self)
        no_coeff=[]
        no_frozen=[]
        no_energy=[]
        V = []
        padding_convention = padding_k_idx(self, kind="joint")

        if self.mode.lower() == "min":
            if nvir_act is None:
                T = []
                if pct_occ is None:
                    for kpt in range(self.nkpts):
                        rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
                        n,V_k = np.linalg.eigh(rdm1_ss_comp[nocc:,nocc:])
                        idx = np.argsort(n)[::-1]
                        n,V_k = n[idx], V_k[:,idx]
                        T_k = n > thresh
                        V.append(V_k)
                        T.append(T_k)
                else:
                    for kpt in range(self.nkpts):
                        rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
                        n,V_k = np.linalg.eigh(rdm1_ss_comp[nocc:,nocc:])
                        idx = np.argsort(n)[::-1]
                        n,V_k = n[idx], V_k[:,idx]
                        cumsum = np.cumsum(n/np.sum(n))
                        T_k = np.array(
                            [c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum])
                        V.append(V_k)
                        T.append(T_k)
                T_min = np.stack(T)
            else:
                for kpt in range(self.nkpts):
                    rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
                    n,V_k = np.linalg.eigh(rdm1_ss_comp[nocc:,nocc:])
                    idx = np.argsort(n)[::-1]
                    n,V_k = n[idx], V_k[:,idx]
                    T_k = np.array(
                        [i < nvir_act for i in range(len(n))])
                    V.append(V_k)
                T_min = np.zeros((self.nkpts,self.mo_energy[0][nocc:].shape[0]), dtype=bool)
                T_min[:,:nvir_act] = True

            T_min = np.logical_or.reduce(T_min,axis=0)
            n_fro_vir = np.sum(T_min == 0)
            if n_fro_vir == self.nmo - self.nocc:
                log.warn("All virtual orbitals were requested to be frozen.\n"
                "At least one virtual orbital must be retained for ADC calculations.\n"
                "Keeping one virtual orbital automatically.")
                n_fro_vir -= 1
                T_min[0] = True
            T_k = np.diag(T_min)

        else:
            if nvir_act is None:
                T = []
                if pct_occ is None:
                    for kpt in range(self.nkpts):
                        rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
                        n,V_k = np.linalg.eigh(rdm1_ss_comp[nocc:,nocc:])
                        idx = np.argsort(n)[::-1]
                        n,V_k = n[idx], V_k[:,idx]
                        T_k = n > thresh
                        V.append(V_k)
                        T.append(T_k)
                else:
                    for kpt in range(self.nkpts):
                        rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
                        n,V_k = np.linalg.eigh(rdm1_ss_comp[nocc:,nocc:])
                        idx = np.argsort(n)[::-1]
                        n,V_k = n[idx], V_k[:,idx]
                        cumsum = np.cumsum(n/np.sum(n))
                        T_k = np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum])
                        V.append(V_k)
                        T.append(T_k)
            else:
                for kpt in range(self.nkpts):
                    rdm1_ss_comp = rdm1_ss[kpt][np.ix_(padding_convention[kpt], padding_convention[kpt])]
                    n,V_k = np.linalg.eigh(rdm1_ss_comp[nocc:,nocc:])
                    idx = np.argsort(n)[::-1]
                    n,V_k = n[idx], V_k[:,idx]
                    T_k = np.array([i < nvir_act for i in range(len(n))])
                    V.append(V_k)
                    T.append(T_k)

        for kpt in range(self.nkpts):
            if self.mode.lower() != "min":
                n_fro_vir = np.sum(T[kpt] == 0)
                T_k = np.diag(T[kpt])
            V_trunc = V[kpt].dot(T_k)
            n_keep = V_trunc.shape[0]-n_fro_vir

            moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[kpt][m] for m in masks[kpt]]
            orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[kpt][:,m] for m in masks[kpt]]
            F_can =  np.diag(moevir)
            F_trunc = V_trunc.T.conj().dot(F_can).dot(V_trunc)
            e_trunc,Z_trunc = np.linalg.eigh(F_trunc[:n_keep,:n_keep])
            U_vir_act = orbvir.dot(V_trunc[:,:n_keep]).dot(Z_trunc)
            U_vir_fro = orbvir.dot(V_trunc[:,n_keep:])
            no_comp = (orboccfrz0,orbocc,U_vir_act,U_vir_fro,orbvirfrz0)
            no_e_comp = (moeoccfrz0,moeocc,e_trunc,moevir[n_keep:],moevirfrz0)
            no_coeff_k = np.hstack(no_comp)
            no_energy_k = np.hstack(no_e_comp)
            nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
            no_frozen_k = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                    np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)
            no_coeff.append(no_coeff_k)
            no_energy.append(no_energy_k)
            no_frozen.append(no_frozen_k)

        self.mo_coeff,self.mo_energy,self.frozen = no_coeff,no_energy,no_frozen
