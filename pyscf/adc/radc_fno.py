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

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
from pyscf.adc import radc_amplitudes
from pyscf.adc import radc
from pyscf import __config__
from pyscf import df
from pyscf.mp import mp2
from pyscf.data.nist import HARTREE2EV

class RADC2FNO(radc.RADC):
    #J. Chem. Phys. 159, 084113 (2023)
    _keys = radc.RADC._keys | {'delta_e','delta_e_corr','e_can','v_can','e_corr_can',
                          'mo_energy','rdm1_ss','ref_state','trans_guess'
                          }

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        self.delta_e = None
        self.delta_e_corr = None
        self.mo_energy = None
        self.method = "adc(3)"
        self.e_can = None
        self.v_can = None
        self.e_corr_can = None
        self.rdm1_ss = None
        self.ref_state = None
        self.if_naf = True
        self.trans_guess = False

    def compute_correction(self, mf, frozen, nroots, eris=None, guess=None):
        if getattr(self, 'with_df', None) or getattr(self._scf, 'with_df', None):
            if self.with_df is None:
                self.with_df = self._scf.with_df
        adc2_ssfno = radc.RADC(mf, frozen, self.mo_coeff, mo_energy = self.mo_energy).set(verbose = self.verbose,
                                                        method_type = self.method_type,
                                                        with_df = self.with_df,if_naf = self.if_naf,
                                                        thresh_naf = self.thresh_naf,naux = self.naux,
                                                        ncvs = self.ncvs,
                                                        approx_trans_moments = self.approx_trans_moments,
                                                        conv_tol = self.conv_tol,tol_residual = self.tol_residual,
                                                        max_space = self.max_space, max_cycle = self.max_cycle)
        e2_ssfno,_,_,_ = adc2_ssfno.kernel(nroots, eris = eris, guess=guess)
        self.delta_e = self.e_can - e2_ssfno
        self.delta_e_corr = self.e_corr_can - adc2_ssfno.e_corr

    def kernel(self, nroots=1, guess=None, eris=None, thresh = 1e-4, ref_state = None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        self.ref_state = ref_state
        self.naux = None
        self.if_heri_eris = True
        if ref_state is None:
            logger.info(self,"Do fno adc calculation")
        elif isinstance(ref_state, int) and 0<ref_state<=nroots:
            logger.info(self,f"Do ss-fno adc calculation, the specic state is {ref_state}")
        else:
            raise ValueError("ref_state should be an int type and in (0,nroots]")

        if not getattr(self, 'with_df', None) and not getattr(self._scf, 'with_df', None):
            self.if_naf = False

        self.make_ss_rdm1(nroots, self.ref_state, guess)
        log.timer('make ss rdm1', *cput0)
        self.mo_coeff,self.mo_energy,frozen = self.make_fno(self.rdm1_ss, self._scf, thresh)
        log.timer('get frozen info', *cput0)
        adc3_ssfno = radc.RADC(self._scf, frozen, self.mo_coeff, mo_energy = self.mo_energy).set(verbose = self.verbose,
                                                                    method_type = self.method_type,method = self.method,
                                                                    with_df = self.with_df,
                                                                    if_naf = self.if_naf,thresh_naf = self.thresh_naf,
                                                                    if_heri_eris = self.if_heri_eris,ncvs = self.ncvs,
                                                                    approx_trans_moments = self.approx_trans_moments,
                                                                    conv_tol = self.conv_tol,
                                                                    tol_residual = self.tol_residual,
                                                                    max_space = self.max_space, max_cycle = self.max_cycle)

        if self.if_naf:
            if self.trans_guess:
                e_exc, v_exc, spec_fac, x, eris, self.naux = adc3_ssfno.kernel(nroots, guess=self.v_can, eris=eris)
            else:
                e_exc, v_exc, spec_fac, x, eris, self.naux = adc3_ssfno.kernel(nroots, guess, eris)
        else:
            if self.trans_guess:
                e_exc, v_exc, spec_fac, x, eris = adc3_ssfno.kernel(nroots, guess=self.v_can, eris=eris)
            else:
                e_exc, v_exc, spec_fac, x, eris = adc3_ssfno.kernel(nroots, guess, eris)

        self.e_corr = adc3_ssfno.e_corr
        log.timer(f'ADC{self.method[4]}', *cput0)

        self.compute_correction(self._scf, frozen, nroots, eris, guess=v_exc)
        e_exc = e_exc + self.delta_e
        self.e_corr = self.e_corr + self.delta_e_corr

        msg = ("\n*************************************************************"
            "\n            FNOADC calculation summary"
            "\n*************************************************************")
        logger.info(self, msg)
        logger.info(self, 'FNO MP%s correlation energy of reference state (a.u.) = %.8f',
                    self.method[4], self.e_corr)

        for n in range(nroots):
            print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                            (self.method, n, e_exc[n], e_exc[n]*HARTREE2EV))
            if self.compute_properties:
                print_string += ("|  Spec factors = %10.8f  " % spec_fac[n])
            logger.info(self, print_string)
        log.timer('RFNOADC', *cput0)
        return e_exc, v_exc, spec_fac, x

    def make_ss_rdm1(self,nroots,ref_state,guess):
        adc2_can = radc.RADC(self._scf,self.frozen).set(verbose = self.verbose,method_type = self.method_type,
                                        with_df = self.with_df,if_naf = self.if_naf,thresh_naf = self.thresh_naf,
                                        ncvs = self.ncvs, approx_trans_moments = self.approx_trans_moments,
                                        conv_tol = self.conv_tol,tol_residual = self.tol_residual,
                                        max_space = self.max_space, max_cycle = self.max_cycle)
        self.e_can,self.v_can,_,_ = adc2_can.kernel(nroots,guess=guess)
        rdm1_gs = adc2_can.make_ref_rdm1()
        self.e_corr_can = adc2_can.e_corr
        if ref_state is not None:
            rdm1_ref = adc2_can.make_rdm1()[ref_state - 1]
            self.rdm1_ss = rdm1_ref + rdm1_gs
        else:
            self.rdm1_ss = rdm1_gs

    def make_fno(self, rdm1_ss, mf, thresh):
        from pyscf.mp import mp2
        nocc = mf.mol.nelectron//2
        nmo = self._nmo
        self._nmo = None
        masks = mp2._mo_splitter(self)
        self._nmo = nmo

        n,V = np.linalg.eigh(rdm1_ss[nocc:,nocc:])
        idx = np.argsort(n)[::-1]
        n,V = n[idx], V[:,idx]
        T = n > thresh
        n_fro_vir = np.sum(T == 0)
        T = np.diag(T)
        V_trunc = V.dot(T)
        n_keep = V_trunc.shape[0]-n_fro_vir

        moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[m] for m in masks]
        orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[:,m] for m in masks]
        F_can =  np.diag(moevir)
        F_trunc = V_trunc.T.dot(F_can).dot(V_trunc)
        e_trunc,Z_trunc = np.linalg.eigh(F_trunc[:n_keep,:n_keep])
        U_vir_act = orbvir.dot(V_trunc[:,:n_keep]).dot(Z_trunc)
        U_vir_fro = orbvir.dot(V_trunc[:,n_keep:])
        no_comp = (orboccfrz0,orbocc,U_vir_act,U_vir_fro,orbvirfrz0)
        no_e_comp = (moeoccfrz0,moeocc,e_trunc,moevir[n_keep:],moevirfrz0)
        no_coeff = np.hstack(no_comp)
        no_energy = np.hstack(no_e_comp)
        nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
        no_frozen = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)
        return no_coeff,no_energy,no_frozen
