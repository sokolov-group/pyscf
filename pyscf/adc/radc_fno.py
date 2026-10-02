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
from pyscf.lib import logger
from pyscf.adc import radc
from pyscf import __config__

class RADC2FNO(radc.RADC):
    '''ADC-based frozen natural orbital (FNO) generator for spin-restricted
    references, following J. Chem. Phys. 159, 084113 (2023).

    Attributes:
        delta_e : list of floats
            Additive correction to the excitation energies.
        delta_e_corr : float
            Additive correction to the ground state correlation energy.
        delta_e_qp : float
            Additive correction to the quasiparticle energies.
        if_ref_qp : bool
            When True, ref_state counts quasiparticle states in IP/EA
            calculations, identified by spec. factor larger than is_qp, and
            delta_e_qp is computed. Default is False.
        is_qp : float
            Threshold for the spec. factor to determine whether an IP/EA state
            is a quasiparticle state. Default value is 0.5.
        e_can : list of floats
            Canonical ADC excitation energies.
        v_can : array
            Canonical ADC eigenvectors.
        p_can : array
            Canonical ADC spec. factor/oscillator strength.
        p_ssfno : array
            State-specific FNO ADC spec. factor/oscillator strength.
        e_corr_can : float
            Canonical ADC correlation energy.
        rdm1_ss : array
            State-specific one-particle reduced density matrix.
        ref_state : int or list of ints
            Target state(s) for the state-specific/averaged RDM1.
            ref_state = 0 (default) is the ground state; ref_state = n is
            the nth root (SS-FNO), or the nth quasiparticle state in IP/EA
            when if_ref_qp is True; ref_state = [n1, n2, ...] averages the
            excited-state RDM1s of the listed roots (SA-FNO).
        trans_guess : bool
            Whether to use the canonical ADC eigenvectors, projected onto the
            truncated FNO basis (see project_guess), as the initial guess for
            the truncated ADC calculation (IP incl. CVS, EA, EE). Combined with
            pick = True for character-based root following inside the Davidson solver.
            Default value is False.

    After kernel() or kernel_gs(), frozen, mo_coeff, mo_occ, and mo_energy
    describe the truncated FNO space to be passed to a target correlated method.
    '''

    _keys = radc.RADC._keys | {'delta_e', 'delta_e_corr', 'e_can', 'v_can', 'e_corr_can',
                               'rdm1_ss', 'ref_state', 'trans_guess',
                               'p_can', 'p_ssfno', 'delta_e_qp', 'is_qp', 'if_ref_qp',
                               'S_vir', 'w_guess_lost', 'ovl_guess'}

    def __init__(self, mf, frozen=0, mo_coeff=None, mo_occ=None, mo_energy=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ, mo_energy)
        self.delta_e = None
        self.delta_e_corr = None
        self.delta_e_qp = None
        self.is_qp = 0.5
        self.e_can = None
        self.v_can = None
        self.p_can = None
        self.p_ssfno = None
        self.e_corr_can = None
        self.rdm1_ss = None
        self.ref_state = None
        self.if_ref_qp = False
        self.trans_guess = False
        self.S_vir = None
        self.w_guess_lost = None
        self.ovl_guess = None

    def project_guess(self, v):
        """Project canonical-basis RADC eigenvector(s) onto the truncated FNO
        basis: contract the particle (virtual) index of every excitation
        block with the canonical-MO -> active-FNO transformation stored in
        self.S_vir; occupied indices are untouched.

        Args:
            v : (ndim_canonical, nroots) array or 1D vector of canonical
                eigenvectors, e.g. self.v_can.

        Returns:
            (v_proj, w_lost): renormalized projected vectors and the
            particle weight lost to the frozen virtuals per root,
            1 - |P v|^2.
        """
        if getattr(self, 'S_vir', None) is None:
            raise RuntimeError('project_guess requires make_fno to have been '
                               'called first (S_vir not set)')
        v = np.asarray(v)
        single = (v.ndim == 1)
        vcol = v.reshape(v.shape[0], -1)

        nocc = self._nocc
        nvir = self._nvir
        S = self.S_vir

        if self.method_type == 'ip' and self.ncvs:
            nval = nocc - self.ncvs
            s_ecc = self.ncvs
            f_ecc = s_ecc + nvir*self.ncvs*self.ncvs
            s_ecv, f_ecv = f_ecc, f_ecc + nvir*self.ncvs*nval
            s_evc, f_evc = f_ecv, f_ecv + nvir*nval*self.ncvs

            def proj(vec):
                return np.concatenate([
                    vec[:s_ecc],
                    S.T.dot(vec[s_ecc:f_ecc].reshape(nvir, -1)).reshape(-1),
                    S.T.dot(vec[s_ecv:f_ecv].reshape(nvir, -1)).reshape(-1),
                    S.T.dot(vec[s_evc:f_evc].reshape(nvir, -1)).reshape(-1)])

        elif self.method_type == 'ip':
            def proj(vec):
                v2 = S.T.dot(vec[nocc:].reshape(nvir, -1)).reshape(-1)
                return np.concatenate([vec[:nocc], v2])
        elif self.method_type == 'ea':
            def proj(vec):
                v2 = vec[nvir:].reshape(nocc, nvir, nvir)
                out = [S.T.dot(vec[:nvir]),
                       np.einsum('iab,ap,bq->ipq', v2, S, S).reshape(-1)]
                return np.concatenate(out)
        elif self.method_type == 'ee':
            def proj(vec):
                v2 = vec[nocc*nvir:].reshape(nocc, nocc, nvir, nvir)
                out = [vec[:nocc*nvir].reshape(nocc, -1).dot(S).reshape(-1),
                       np.einsum('ijab,ap,bq->ijpq', v2, S, S).reshape(-1)]
                return np.concatenate(out)
        else:
            raise NotImplementedError('project_guess for method_type = %s'
                                      % self.method_type)

        vps = [proj(vcol[:, r]) for r in range(vcol.shape[1])]
        raw = np.array([np.dot(x, x) for x in vps])
        w_lost = 1.0 - raw
        vps = [x/np.sqrt(r) if r > 0 else x for x, r in zip(vps, raw)]
        vp = vps[0] if single else np.column_stack(vps)
        return vp, w_lost

    def kernel_gs(self, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, "generate fno with correction for the ground state")
        self.ref_state = None

        if not getattr(self, 'with_df', None) and not getattr(self._scf, 'with_df', None):
            self.if_naf = False

        self.make_ss_rdm1(log, cput0, if_gs=True)
        log.timer('make gs rdm1', *cput0)
        self.make_fno(self.rdm1_ss, self._scf, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)
        self.compute_correction(self._scf, self.frozen, eris=eris, if_gs=True)

        log.timer('gs FNO', *cput0)

    def kernel(self, nroots=1, guess=None, eris=None, thresh = 1e-4, pct_occ=None, nvir_act=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.ref_state is None or self.ref_state == 0:
            logger.info(self,"Do fno adc calculation")
        elif isinstance(self.ref_state, int) and 0<self.ref_state<=nroots:
            logger.info(self,f"Do ss-fno adc calculation, the specic state is {self.ref_state}")
        elif isinstance(self.ref_state, (list, tuple)) and len(self.ref_state) > 0 and \
                all(isinstance(s, (int, np.integer)) and 0 < s <= nroots for s in self.ref_state):
            logger.info(self, f"Do sa-fno adc calculation, the specic states are {list(self.ref_state)}")
        else:
            raise ValueError("ref_state should be an int or a non-empty list of ints in [1,nroots]")

        if not getattr(self, 'with_df', None) and not getattr(self._scf, 'with_df', None):
            self.if_naf = False

        self.make_ss_rdm1(nroots, guess)
        log.timer('make ss rdm1', *cput0)
        self.make_fno(self.rdm1_ss, self._scf, thresh, pct_occ, nvir_act)
        log.timer('get frozen info', *cput0)

        if self.trans_guess and self.method_type in ('ip', 'ea', 'ee'):
            guess_proj, w_lost = self.project_guess(self.v_can)
            self.w_guess_lost = w_lost
            logger.info(self, "trans_guess: canonical guesses projected onto "
                        "the FNO basis; weight lost to frozen virtuals per root: %s",
                        np.array2string(w_lost, precision=4))
            self.compute_correction(self._scf, nroots, eris, guess=guess_proj)
        else:
            self.compute_correction(self._scf, nroots, eris, guess)

        log.timer('es FNO', *cput0)

    def compute_correction(self, mf, nroots=None, eris=None, guess=None, if_gs=False):
        adc_ssfno = radc.RADC(mf, self.frozen, self.mo_coeff, mo_energy = self.mo_energy).set(verbose = self.verbose,
                                                        method = self.method,method_type = self.method_type,
                                                        with_df = self.with_df,if_naf = self.if_naf,
                                                        thresh_naf = self.thresh_naf,naux = self.naux,
                                                        if_heri_eris = self.if_heri_eris,ncvs = self.ncvs,
                                                        approx_trans_moments = self.approx_trans_moments,
                                                        conv_tol = self.conv_tol,tol_residual = self.tol_residual,
                                                        max_space = self.max_space, max_cycle = self.max_cycle)
        adc_ssfno.pick = self.pick
        if if_gs:
            _,_,_ = adc_ssfno.kernel_gs(eris)
        else:
            self.e_ssfno,self.v_ssfno,self.p_ssfno,_ = adc_ssfno.kernel(nroots,guess,eris)
            self.ovl_guess = adc_ssfno.ovl_guess
            self.delta_e = self.e_can - self.e_ssfno
            if self.if_ref_qp and self.method_type in ('ip', 'ea'):
                mask_fno = self.p_ssfno > self.is_qp
                mask_can = self.p_can > self.is_qp
                e_can_qp = self.e_can[mask_can]
                e_ssfno_qp = self.e_ssfno[mask_fno]
                n_qp = min(len(e_can_qp), len(e_ssfno_qp))
                self.delta_e_qp = e_can_qp[:n_qp] - e_ssfno_qp[:n_qp]
        self.naux = adc_ssfno.naux
        self.eris = adc_ssfno.eris
        self.delta_e_corr = self.e_corr_can - adc_ssfno.e_corr

    def correct(self, e):
        """Additively-corrected excitation energies e + delta_e."""
        return e + self.delta_e

    def correct_corr(self, e):
        """Additively-corrected correlation energy e + delta_e_corr."""
        return e + self.delta_e_corr

    def make_ss_rdm1(self,nroots,guess,if_gs=False):
        heri_tmp = self.if_heri_eris
        self.if_heri_eris = False
        pick_tmp = self.pick
        self.pick = None
        if if_gs:
            _,_,_ = radc.RADC.kernel_gs(self)
        else:
            self.e_can,self.v_can,self.p_can,_ = radc.RADC.kernel(self,nroots,guess)
        self.pick = pick_tmp
        self.if_heri_eris = heri_tmp
        rdm1_gs = self.make_ref_rdm1()
        self.e_corr_can = self.e_corr
        if self.ref_state is not None and self.ref_state != 0:
            rdm1_es = self.make_rdm1()
            if isinstance(self.ref_state, (list, tuple, np.ndarray)):
                if self.if_ref_qp and self.method_type in ('ip', 'ea'):
                    qp_idx = np.where(self.p_can > self.is_qp)[0]
                    states = [qp_idx[s - 1] for s in self.ref_state]
                else:
                    states = [s - 1 for s in self.ref_state]
                rdm1_es = np.mean([rdm1_es[st] for st in states], axis=0)
            else:
                if self.if_ref_qp and self.method_type in ('ip', 'ea'):
                    qp_idx = np.where(self.p_can > self.is_qp)[0]
                    state = qp_idx[self.ref_state - 1]
                else:
                    state = self.ref_state - 1
                rdm1_es = rdm1_es[state]
            self.rdm1_ss = rdm1_es + rdm1_gs
        else:
            self.rdm1_ss = rdm1_gs

    def make_fno(self, rdm1_ss, mf, thresh, pct_occ, nvir_act):
        from pyscf.mp import mp2
        nocc = mf.mol.nelectron//2
        nmo = self._nmo
        self._nmo = None
        masks = mp2._mo_splitter(self)
        self._nmo = nmo

        n,V = np.linalg.eigh(rdm1_ss[nocc:,nocc:])
        idx = np.argsort(n)[::-1]
        n,V_trunc = n[idx], V[:,idx]
        if nvir_act is None:
            if pct_occ is None:
                T = n > thresh
            else:
                cumsum = np.cumsum(n/np.sum(n))
                T = np.array([c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum])
        else:
            T = np.array([i < nvir_act for i in range(len(n))])

        if not T.any():
            logger.warn(self, "All virtual natural orbitals were requested to be "
                        "frozen.\nAt least one virtual must be retained for ADC "
                        "calculations.\nKeeping one automatically.")
            T[0] = True

        n_keep = int(np.sum(T))

        moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[m] for m in masks]
        orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[:,m] for m in masks]
        F_can =  np.diag(moevir)
        F_trunc = V_trunc.T.dot(F_can).dot(V_trunc)
        e_trunc,Z_trunc = np.linalg.eigh(F_trunc[:n_keep,:n_keep])
        e_fro = np.diagonal(F_trunc[n_keep:, n_keep:]).copy()

        self.S_vir = V_trunc[:, :n_keep].dot(Z_trunc)
        U_vir_act = orbvir.dot(self.S_vir)
        U_vir_fro = orbvir.dot(V_trunc[:,n_keep:])

        no_comp = (orboccfrz0,orbocc,U_vir_act,U_vir_fro,orbvirfrz0)
        no_e_comp = (moeoccfrz0,moeocc,e_trunc,e_fro,moevirfrz0)
        no_coeff = np.hstack(no_comp)
        no_energy = np.hstack(no_e_comp)
        nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
        no_frozen = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)

        self.mo_coeff,self.mo_energy,self.frozen = no_coeff,no_energy,no_frozen
