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

from pyscf.lib import logger
from pyscf.lib.parameters import LARGE_DENOM
from pyscf.pbc import tools
from pyscf.pbc.adc import kadc_rhf
from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs
from pyscf.pbc.adc.lib import check_implemented, einsum
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import padding_k_idx


class RADCEA(kadc_rhf.KRADC):
    """Restricted periodic EA-ADC solver.

    Args:
        adc (KRADC): Reference driver with amplitudes, normally prepared
            by ``ground_state_energy()``. Reference scratch ownership is
            retained while this solver uses its disk-backed amplitudes.

    Vectors contain one-particle and two-particle/one-hole blocks. The doubles
    shape is ``(nk, nk, nocc, nvir, nvir)``; the fourth momentum is conserved.
    ``kernel(nroots, kptlist=...)`` returns E, U, P and X with sectors in
    kptlist order. E and P have shape (nsectors, nroots), U has shape
    (nsectors, nroots, vector_size), and X has shape
    (nsectors, nmo, nroots). P contains spectroscopic factors.

    Example:
        >>> reference = adc.KRADC(kmf)
        >>> reference.ground_state_energy()
        >>> solver = RADCEA(reference)
        >>> E, U, P, X = solver.kernel(nroots=3, kptlist=[0])
    """

    _keys = ({'tol_residual', 'conv_tol', 'e_corr', 'method', 'mo_coeff',
        'mo_energy_b', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'kpts', 'exxdiv', 'khelper', 'cell', 'nkop_chk', 'kop_npick',
        'chunk_size'})

    def __init__(self, adc):
        self._init_from_ground_state(adc)
        self.type_ = 'ea'

    kernel = kadc_rhf.kernel

    def get_init_guess(self, nroots=1, diag=None, ascending=True,
                       type=None, ini=None, kshift=None):
        return self._charged_init_guess(nroots, diag, ascending,
                                        type, ini, kshift)

    def gen_matvec(self, kshift, imds=None, eris=None):
        if eris is None:
            eris = self.ao2mo()
        if imds is None:
            imds = self.get_imds(eris)
        diag = self.get_diag(kshift, imds, eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag

    def vector_size(self):
        """Return the charged-excitation vector length."""
        return self.n_singles + int(np.prod(self.doubles_shape))

    def get_imds(self, eris=None):
        cput0 = (time.process_time(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        self.method = check_implemented(self.method)

        method = self.method
        nkpts = self.nkpts

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)
        nocc = self.nocc
        nvir = self.nvir
        kconserv = self.khelper.kconserv

        idn_vir = np.identity(nvir)
        if eris is None:
            eris = self.ao2mo()

        eris_ovov = eris.ovov

        # a-b block
        # Zeroth-order terms
        M_ab = np.empty((nkpts, nvir, nvir), dtype=mo_coeff[0].dtype)

        for ka in range(nkpts):
            kb = ka
            M_ab[ka] = einsum('ab,a->ab', idn_vir, e_vir[ka])
            for kl in range(nkpts):
                for km in range(nkpts):

                    kd = kconserv[kl, ka, km]
                    # Second-order terms
                    t2_1_mla = self.t2[0][km, kl, ka]
                    M_ab[ka] += 0.5 * 0.5 * \
                        einsum('mlad,lbmd->ab', t2_1_mla, eris_ovov[kl, kb,
                            km])
                    M_ab[ka] -= 0.5 * 0.5 * \
                        einsum('mlad,ldmb->ab', t2_1_mla, eris_ovov[kl, kd,
                            km])

                    t2_1_lma = self.t2[0][kl, km, ka]
                    M_ab[ka] -= 0.5 * 0.5 * \
                        einsum('lmad,lbmd->ab', t2_1_lma, eris_ovov[kl, kb,
                            km])
                    M_ab[ka] -= 0.5 * \
                        einsum('lmad,lbmd->ab', t2_1_lma, eris_ovov[kl, kb,
                            km])
                    M_ab[ka] += 0.5 * 0.5 * \
                        einsum('lmad,ldmb->ab', t2_1_lma, eris_ovov[kl, kd,
                            km])
                    del t2_1_lma

                    t2_1_lmb = self.t2[0][kl, km, kb]
                    M_ab[ka] -= 0.5 * 0.5 * \
                        einsum('lmbd,lamd->ab', t2_1_lmb.conj(),
                                   eris_ovov[kl, ka, km].conj())
                    M_ab[ka] -= 0.5 * \
                        einsum('lmbd,lamd->ab', t2_1_lmb.conj(),
                                   eris_ovov[kl, ka, km].conj())
                    M_ab[ka] += 0.5 * 0.5 * \
                        einsum('lmbd,ldma->ab', t2_1_lmb.conj(),
                                   eris_ovov[kl, kd, km].conj())
                    del t2_1_lmb

                    t2_1_mlb = self.t2[0][km, kl, kb]
                    M_ab[ka] += 0.5 * 0.5 * \
                        einsum('mlbd,lamd->ab', t2_1_mlb.conj(),
                                   eris_ovov[kl, ka, km].conj())
                    M_ab[ka] -= 0.5 * 0.5 * \
                        einsum('mlbd,ldma->ab', t2_1_mlb.conj(),
                                   eris_ovov[kl, kd, km].conj())
                    del t2_1_mlb

        cput0 = (log.timer_debug1(
            'Completed M_ab second-order terms ADC(2) calculation', *cput0))

        if (method == 'adc(3)'):

            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            t1_2 = self.t1[0]

            for ka in range(nkpts):
                kb = ka
                for kl in range(nkpts):

                    if eris.ovvv is None:
                        chunk_size = self.chunk_size
                        chunk_size = min(chunk_size, nocc)
                        a = 0
                        for p in range(0, nocc, chunk_size):
                            kd = kconserv[ka, kb, kl]
                            eris_ovvv = (_ChemistsERIs.ovvv_df(eris.Lov[kl,
                                kd], eris.Lvv[ka, kb], p, chunk_size).reshape(-
                                1, nvir, nvir, nvir) / nkpts)
                            k = eris_ovvv.shape[0]
                            M_ab[ka] += 2.0 * einsum('ld,ldab->ab', t1_2[kl][
                                a:a + k], eris_ovvv)
                            del eris_ovvv
                            kd = kconserv[kb, ka, kl]
                            eris_ovvv = (_ChemistsERIs.ovvv_df(eris.Lov[kl,
                                kd], eris.Lvv[kb, ka], p, chunk_size).reshape(-
                                1, nvir, nvir, nvir) / nkpts)
                            k = eris_ovvv.shape[0]
                            M_ab[ka] += 2.0 * einsum('ld,ldba->ab', t1_2[kl][
                                a:a + k].conj(), eris_ovvv.conj())
                            del eris_ovvv
                            kd = kconserv[ka, kb, kl]
                            eris_ovvv = (_ChemistsERIs.ovvv_df(eris.Lov[kl,
                                kb], eris.Lvv[ka, kd], p, chunk_size).reshape(-
                                1, nvir, nvir, nvir) / nkpts)
                            k = eris_ovvv.shape[0]
                            M_ab[ka] -= einsum('ld,lbad->ab', t1_2[kl][a:a +
                                k], eris_ovvv)
                            del eris_ovvv
                            kd = kconserv[kb, ka, kl]
                            eris_ovvv = (_ChemistsERIs.ovvv_df(eris.Lov[kl,
                                ka], eris.Lvv[kb, kd], p, chunk_size).reshape(-
                                1, nvir, nvir, nvir) / nkpts)
                            k = eris_ovvv.shape[0]
                            M_ab[ka] -= einsum('ld,labd->ab', t1_2[kl, a:a +
                                k].conj(), eris_ovvv.conj())
                            del eris_ovvv
                            a += k

                    else:
                        eris_ovvv = eris.ovvv
                        kd = kconserv[ka, kb, kl]
                        M_ab[ka] += 2. * einsum('ld,ldab->ab', t1_2[kl],
                                                    eris_ovvv[kl, kd,
                                                        ka])
                        M_ab[ka] -= einsum('ld,lbad->ab', t1_2[kl],
                                               eris_ovvv[kl, kb,
                                                   ka])
                        kd = kconserv[kb, ka, kl]
                        M_ab[ka] += 2.0 * einsum('ld,ldba->ab', t1_2[kl].conj(
                            ), eris_ovvv[kl, kd, kb].conj())
                        M_ab[ka] -= einsum('ld,labd->ab', t1_2[kl].conj(),
                                               eris_ovvv[kl, ka,
                                                   kb].conj())

            cput0 = log.timer_debug1("Completed M_ab ovvv ADC(3) calculation",
                *cput0)

            for ka in range(nkpts):
                kb = ka
                log.timer_debug1("start:t2_ijbc*v_acij||O(No^2*Nv^3*Nk^3)")
                for kl in range(nkpts):
                    for km in range(nkpts):
                        kd = kconserv[km, ka, kl]

                        t2_1 = self.t2[0]
                        t2_2_lma = self.t2[1][kl, km, ka]
                        M_ab[ka] -= 0.5 * 0.5 * \
                            einsum('lmad,lbmd->ab', t2_2_lma, eris_ovov[kl, kb,
                                km])
                        M_ab[ka] += 0.5 * 0.5 * \
                            einsum('lmad,ldmb->ab', t2_2_lma, eris_ovov[kl, kd,
                                km])
                        M_ab[ka] -= 0.5 * einsum('lmad,lbmd->ab',
                                                     t2_2_lma, eris_ovov[kl,
                                                         kb,
                                                         km])

                        t2_2_mla = self.t2[1][km, kl, ka]
                        M_ab[ka] += 0.5 * 0.5 * \
                            einsum('mlad,lbmd->ab', t2_2_mla, eris_ovov[kl, kb,
                                km])
                        M_ab[ka] -= 0.5 * 0.5 * \
                            einsum('mlad,ldmb->ab', t2_2_mla, eris_ovov[kl, kd,
                                km])

                        t2_2_lmb = self.t2[1][kl, km, kb]
                        M_ab[ka] -= 0.5 * 0.5 * \
                            einsum('lmbd,lamd->ab', t2_2_lmb.conj(),
                                       eris_ovov[kl, ka,
                                           km].conj())
                        M_ab[ka] += 0.5 * 0.5 * \
                            einsum('lmbd,ldma->ab', t2_2_lmb.conj(),
                                       eris_ovov[kl, kd,
                                           km].conj())
                        M_ab[ka] -= 0.5 * einsum('lmbd,lamd->ab',
                            t2_2_lmb.conj(), eris_ovov[kl, ka, km].conj())

                        t2_2_mlb = self.t2[1][km, kl, kb]
                        M_ab[ka] += 0.5 * 0.5 * \
                            einsum('mlbd,mdla->ab', t2_2_mlb.conj(),
                                       eris_ovov[km, kd,
                                           kl].conj())
                        M_ab[ka] -= 0.5 * 0.5 * \
                            einsum('mlbd,ldma->ab', t2_2_mlb.conj(),
                                       eris_ovov[kl, kd,
                                           km].conj())
                        del t2_2_mlb

                log.timer_debug1("start:temp1_ovov*v_ovov||O(No^3*Nv^3*Nk^3)")
                for kn, ke, kd in kpts_helper.loop_kkk(nkpts):

                    kl = kconserv[ke, kn, kd]
                    km = kconserv[kb, kl, kd]
                    temp_t2_v_1 = einsum(
                        'lned,mlbd->nemb', t2_1[kl, kn, ke], t2_1[km, kl,
                            kb].conj())
                    M_ab[ka] -= 0.5 * einsum('nemb,nmae->ab',
                                                 temp_t2_v_1, eris_oovv[kn, km,
                                                     ka])
                    M_ab[ka] += 0.5 * 2.0 * einsum('nemb,neam->ab',
                        temp_t2_v_1, eris_ovvo[kn, ke, ka])
                    M_ab[ka] += 0.5 * 2.0 * einsum('nema,nebm->ab',
                        temp_t2_v_1, eris_ovvo[kn, ke, kb])
                    M_ab[ka] -= 0.5 * einsum('nema,nmbe->ab',
                                                 temp_t2_v_1, eris_oovv[kn, km,
                                                     kb])
                    del temp_t2_v_1

                    temp_t2_v_1 = einsum(
                        'nled,lmbd->mbne', t2_1[kn, kl, ke], t2_1[kl, km,
                            kb].conj())
                    M_ab[ka] -= 0.5 * einsum('mbne,nmae->ab',
                                                 temp_t2_v_1, eris_oovv[kn, km,
                                                     ka])
                    M_ab[ka] += 0.5 * 2.0 * einsum('mbne,neam->ab',
                        temp_t2_v_1, eris_ovvo[kn, ke, ka])
                    M_ab[ka] -= 0.5 * einsum('mane,nmbe->ab',
                                                 temp_t2_v_1, eris_oovv[kn, km,
                                                     kb])
                    M_ab[ka] += 0.5 * 2.0 * einsum('mane,nebm->ab',
                        temp_t2_v_1, eris_ovvo[kn, ke, kb])
                    del temp_t2_v_1

                    temp_t2_v_2 = einsum(
                        'nled,mlbd->nemb', t2_1[kn, kl, ke], t2_1[km, kl,
                            kb].conj())
                    M_ab[ka] += 0.5 * 2.0 * einsum('nemb,nmae->ab',
                        temp_t2_v_2, eris_oovv[kn, km, ka])
                    M_ab[ka] -= 0.5 * 4.0 * einsum('nemb,neam->ab',
                        temp_t2_v_2, eris_ovvo[kn, ke, ka])
                    M_ab[ka] += 0.5 * 2.0 * einsum('nema,nmbe->ab',
                        temp_t2_v_2, eris_oovv[kn, km, kb])
                    M_ab[ka] -= 0.5 * 4.0 * einsum('nema,nebm->ab',
                        temp_t2_v_2, eris_ovvo[kn, ke, kb])
                    del temp_t2_v_2

                    temp_t2_v_3 = einsum(
                        'lned,lmbd->nemb', t2_1[kl, kn, ke], t2_1[kl, km,
                            kb].conj())
                    M_ab[ka] -= 0.5 * einsum('nemb,neam->ab',
                                                 temp_t2_v_3, eris_ovvo[kn, ke,
                                                     ka])
                    M_ab[ka] += 0.5 * 2.0 * einsum('nemb,nmae->ab',
                        temp_t2_v_3, eris_oovv[kn, km, ka])
                    M_ab[ka] += 0.5 * 2.0 * einsum('nema,nmbe->ab',
                        temp_t2_v_3, eris_oovv[kn, km, kb])
                    M_ab[ka] -= 0.5 * einsum('nema,nebm->ab',
                                                 temp_t2_v_3, eris_ovvo[kn, ke,
                                                     kb])
                    del temp_t2_v_3

                log.timer_debug1("start:temp2_oo*v_ovov||O(No^3*Nv^2*Nk^3)")
                for kn, ke, kd in kpts_helper.loop_kkk(nkpts):
                    kl = kconserv[ke, kn, kd]
                    km = kconserv[ke, kl, kd]
                    temp_t2_v_8 = einsum(
                        'lned,mled->mn', t2_1[kl, kn, ke], t2_1[km, kl,
                            ke].conj())
                    M_ab[ka] += 2. * einsum('mn,nmab->ab', temp_t2_v_8,
                                                eris_oovv[kn, km,
                                                    ka])
                    M_ab[ka] -= einsum('mn,nbam->ab', temp_t2_v_8,
                                           eris_ovvo[kn, kb,
                                               ka])
                    del temp_t2_v_8

                    temp_t2_v_9 = einsum(
                        'nled,mled->mn', t2_1[kn, kl, ke], t2_1[km, kl,
                            ke].conj())
                    M_ab[ka] -= 4. * einsum('mn,nmab->ab', temp_t2_v_9,
                                                eris_oovv[kn, km,
                                                    ka])
                    M_ab[ka] += 2. * einsum('mn,nbam->ab', temp_t2_v_9,
                                                eris_ovvo[kn, kb,
                                                    ka])
                    del temp_t2_v_9

                log.timer_debug1("start:Mab vvvv||O(No^2*Nv^3*Nk^3)")
                for km in range(nkpts):
                    for kl in range(nkpts):
                        kf = kconserv[km, kl, ka]
                        temp_t2 = self.imds.t2_1_vvvv[:]
                        t2_1_mla = self.t2[0][km, kl, ka]
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('mlaf,mlbf->ab', t2_1_mla,
                                       temp_t2[km, kl,
                                           kb].conj())
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('mlaf,lmbf->ab', t2_1_mla,
                                       temp_t2[kl, km,
                                           kb].conj())
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('mlaf,mlfb->ab', t2_1_mla,
                                       temp_t2[km, kl,
                                           kf].conj())
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('mlaf,lmfb->ab', t2_1_mla,
                                       temp_t2[kl, km,
                                           kf].conj())
                        M_ab[ka] -= 0.5 * einsum('mlaf,mlbf->ab', t2_1_mla,
                            temp_t2[km, kl, kb].conj())
                        del t2_1_mla

                        t2_1_lma = self.t2[0][kl, km, ka]
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('lmaf,mlbf->ab', t2_1_lma,
                                       temp_t2[km, kl,
                                           kb].conj())
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('lmaf,lmbf->ab', t2_1_lma,
                                       temp_t2[kl, km,
                                           kb].conj())
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('lmaf,mlfb->ab', t2_1_lma,
                                       temp_t2[km, kl,
                                           kf].conj())
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('lmaf,lmfb->ab', t2_1_lma,
                                       temp_t2[kl, km,
                                           kf].conj())
                        del t2_1_lma

                        kd = kconserv[km, ka, kl]
                        t2_1_mlb = self.t2[0][km, kl, kb]
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('mlad,mlbd->ab', temp_t2[km,
                                       kl, ka].conj(), t2_1_mlb)
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('lmad,mlbd->ab', temp_t2[kl,
                                       km, ka].conj(), t2_1_mlb)
                        M_ab[ka] -= 0.5 * einsum('mlad,mlbd->ab', temp_t2[km,
                            kl, ka].conj(), t2_1_mlb)

                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('lmad,mlbd->ab', temp_t2[kl,
                                       km, ka].conj(), t2_1_mlb)
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('mlad,mlbd->ab', temp_t2[km,
                                       kl, ka].conj(), t2_1_mlb)
                        del t2_1_mlb

                        t2_1_lmb = self.t2[0][kl, km, kb]
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('mlad,lmbd->ab', temp_t2[km,
                                       kl, ka].conj(), t2_1_lmb)
                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('lmad,lmbd->ab', temp_t2[kl,
                                       km, ka].conj(), t2_1_lmb)

                        M_ab[ka] -= 0.5 * 0.25 * \
                            einsum('lmad,lmbd->ab', temp_t2[kl,
                                       km, ka].conj(), t2_1_lmb)
                        M_ab[ka] += 0.5 * 0.25 * \
                            einsum('mlad,lmbd->ab', temp_t2[km,
                                       kl, ka].conj(), t2_1_lmb)
                        del temp_t2
                        del t2_1_lmb

                        for kd in range(nkpts):
                            kf = kconserv[km, kd, kl]
                            ke = kconserv[kb, ka, kf]
                            temp_t2_vv_1 = einsum(
                                'mldf,mled->ef', t2_1[km, kl, kd], t2_1[km, kl,
                                    ke].conj())
                            temp_t2_vv_2 = einsum(
                                'mldf,lmed->ef', t2_1[km, kl, kd], t2_1[kl, km,
                                    ke].conj())
                            temp_t2_vv_3 = einsum(
                                'lmdf,mled->ef', t2_1[kl, km, kd], t2_1[km, kl,
                                    ke].conj())
                            temp_t2_vv_4 = einsum(
                                'lmdf,lmed->ef', t2_1[kl, km, kd], t2_1[kl, km,
                                    ke].conj())
                            temp_t2_vv_5 = einsum(
                                'mlfd,mled->ef', t2_1[km, kl, kf], t2_1[km, kl,
                                    ke].conj())
                            if eris.vvvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nvir)
                                a = 0
                                for p in range(0, nvir, chunk_size):
                                    eris_vvvv = _ChemistsERIs.vvvv_df(
                                        eris.Lvv[kb, ka],
                                        eris.Lvv[ke, kf],
                                        p,
                                        chunk_size,
                                    ) / nkpts
                                    k = eris_vvvv.shape[0]
                                    M_ab[ka, a:a + k] -= einsum('ef,aebf->ab',
                                        temp_t2_vv_1, eris_vvvv)
                                    M_ab[ka, a:a + k] += einsum('ef,aebf->ab',
                                        temp_t2_vv_2, eris_vvvv)
                                    M_ab[ka, a:a + k] += einsum('ef,aebf->ab',
                                        temp_t2_vv_3, eris_vvvv)
                                    M_ab[ka, a:a + k] -= einsum('ef,aebf->ab',
                                        temp_t2_vv_4, eris_vvvv)
                                    M_ab[ka, a:a + k] += 2. * \
                                        einsum('ef,aebf->ab', temp_t2_vv_5,
                                            eris_vvvv)
                                    del eris_vvvv

                                    eris_vvvv = _ChemistsERIs.vvvv_df(
                                        eris.Lvv[kb, kf],
                                        eris.Lvv[ke, ka],
                                        p,
                                        chunk_size,
                                    ) / nkpts
                                    M_ab[ka, a:a + k] += 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_1,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] -= 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_2,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] -= 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_3,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] += 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_4,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] -= einsum('ef,aefb->ab',
                                        temp_t2_vv_5, eris_vvvv)
                                    del eris_vvvv
                                    a += k

                            elif isinstance(eris.vvvv, np.ndarray):
                                eris_vvvv = eris.vvvv
                                M_ab[ka] -= einsum('ef,aebf->ab', temp_t2_vv_1,
                                    eris_vvvv[ka, ke, kb])
                                M_ab[ka] += einsum('ef,aebf->ab', temp_t2_vv_2,
                                    eris_vvvv[ka, ke, kb])
                                M_ab[ka] += einsum('ef,aebf->ab', temp_t2_vv_3,
                                    eris_vvvv[ka, ke, kb])
                                M_ab[ka] -= einsum('ef,aebf->ab', temp_t2_vv_4,
                                    eris_vvvv[ka, ke, kb])
                                M_ab[ka] += 2.0 * einsum('ef,aebf->ab',
                                    temp_t2_vv_5, eris_vvvv[ka, ke, kb])
                                M_ab[ka] += 0.5 * einsum('ef,aefb->ab',
                                    temp_t2_vv_1, eris_vvvv[ka, ke, kf])
                                M_ab[ka] -= 0.5 * einsum('ef,aefb->ab',
                                    temp_t2_vv_2, eris_vvvv[ka, ke, kf])
                                M_ab[ka] -= 0.5 * einsum('ef,aefb->ab',
                                    temp_t2_vv_3, eris_vvvv[ka, ke, kf])
                                M_ab[ka] += 0.5 * einsum('ef,aefb->ab',
                                    temp_t2_vv_4, eris_vvvv[ka, ke, kf])
                                M_ab[ka] -= einsum('ef,aefb->ab', temp_t2_vv_5,
                                    eris_vvvv[ka, ke, kf])
                            else:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nvir)
                                a = 0
                                for p in range(0, nvir, chunk_size):
                                    eris_vvvv = eris.vvvv[kb, ke, ka,
                                        p:p + chunk_size]
                                    k = eris_vvvv.shape[0]
                                    M_ab[ka, a:a + k] -= einsum('ef,aebf->ab',
                                        temp_t2_vv_1, eris_vvvv)
                                    M_ab[ka, a:a + k] += einsum('ef,aebf->ab',
                                        temp_t2_vv_2, eris_vvvv)
                                    M_ab[ka, a:a + k] += einsum('ef,aebf->ab',
                                        temp_t2_vv_3, eris_vvvv)
                                    M_ab[ka, a:a + k] -= einsum('ef,aebf->ab',
                                        temp_t2_vv_4, eris_vvvv)
                                    M_ab[ka, a:a + k] += 2. * \
                                        einsum('ef,aebf->ab', temp_t2_vv_5,
                                            eris_vvvv)
                                    del eris_vvvv
                                    eris_vvvv = eris.vvvv[ka, ke, kf,
                                        p:p + chunk_size]
                                    M_ab[ka, a:a + k] += 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_1,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] -= 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_2,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] -= 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_3,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] += 0.5 * \
                                        einsum('ef,aefb->ab', temp_t2_vv_4,
                                            eris_vvvv)
                                    M_ab[ka, a:a + k] -= einsum('ef,aefb->ab',
                                        temp_t2_vv_5, eris_vvvv)
                                    del eris_vvvv
                                    a += k
                            del temp_t2_vv_1
                            del temp_t2_vv_2
                            del temp_t2_vv_3
                            del temp_t2_vv_4
                            del temp_t2_vv_5

        cput0 = log.timer_debug1("Completed M_ab ADC(3) calculation", *cput0)

        return M_ab

    def get_diag(self, kshift, M_ab=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)

        self.method = check_implemented(self.method)

        if M_ab is None:
            M_ab = self.get_imds()

        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        nocc = self.nocc
        nvir = self.nvir
        n_singles = nvir
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        dim = n_singles + n_doubles

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)
        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc

        diag = np.zeros((dim), dtype=np.complex128)
        doubles = np.zeros((nkpts, nkpts, nocc * nvir * nvir),
            dtype=np.complex128)

        # Compute precond in h1-h1 block
        M_ab_diag = np.diagonal(M_ab[kshift])

        # Compute precond in 2p1h-2p1h block
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                d_ab = e_vir[ka][:nvir, None] + e_vir[kb][:nvir]
                d_i = e_occ[kj][:, None]
                D_n = -d_i + d_ab.reshape(-1)
                doubles[kj, ka] += D_n.reshape(-1)

        if (self.frozen is not None or not np.all([x.shape[
            1] == self.nmo for x in self.mo_coeff])):
            doubles = doubles.reshape(nkpts, nkpts, nocc, nvir, nvir)
            (M_ab_diag, doubles) = self.mask_frozen_ea(M_ab_diag, doubles,
                kshift)

        diag[s1:f1] = M_ab_diag.copy()
        diag[s2:f2] = doubles.reshape(-1)
        log.timer_debug1("Completed ea_diag calculation")

        return diag

    def matvec(self, kshift, M_ab=None, eris=None):
        self.method = check_implemented(self.method)

        method = self.method

        nkpts = self.nkpts
        nocc = self.nocc
        kconserv = self.khelper.kconserv
        nvir = self.nvir
        n_singles = self.nmo - self.nocc
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        s_singles = 0
        f_singles = n_singles
        s_doubles = f_singles
        f_doubles = s_doubles + n_doubles

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)

        if M_ab is None:
            M_ab = self.get_imds()

        # Calculate sigma vector
        def sigma_(r):
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles]
            r2 = r[s_doubles:f_doubles]

            r2 = r2.reshape(nkpts, nkpts, nocc, nvir, nvir)
            s2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir),
                dtype=np.complex128)
            cell = self.cell
            kpts = self.kpts
            madelung = tools.madelung(cell, kpts)

    ############ ADC(2) ij block ############################

            s1 = einsum('ab,b->a', M_ab[kshift], r1)

    ########### ADC(2) coupling blocks #########################

            for kb in range(nkpts):
                for kc in range(nkpts):
                    ki = kconserv[kb, kshift, kc]
                    if eris.ovvv is None:
                        chunk_size = self.chunk_size
                        chunk_size = min(chunk_size, nocc)
                        a = 0
                        for p in range(0, nocc, chunk_size):
                            eris_ovvv = (_ChemistsERIs.ovvv_df(eris.Lov[ki,
                                kc], eris.Lvv[kshift, kb], p,
                                chunk_size).reshape(-1, nvir, nvir, nvir) /
                                nkpts)
                            k = eris_ovvv.shape[0]
                            s1 += 2. * einsum('icab,ibc->a', eris_ovvv.conj(),
                                                  r2[ki, kb,
                                                      a:a + k])
                            s2[ki, kb, a:a + k] += einsum('icab,a->ibc',
                                eris_ovvv, r1)
                            del eris_ovvv

                            eris_ovvv = (_ChemistsERIs.ovvv_df(eris.Lov[ki,
                                kb], eris.Lvv[kshift, kc], p,
                                chunk_size).reshape(-1, nvir, nvir, nvir) /
                                nkpts)
                            s1 -= einsum('ibac,ibc->a', eris_ovvv.conj(),
                                             r2[ki, kb,
                                                 a:a + k])
                            del eris_ovvv
                            a += k
                    else:
                        direct = eris.ovvv[ki, kc, kshift]
                        exchange = eris.ovvv[ki, kb, kshift]
                        s1 += 2 * einsum('icab,ibc->a', direct.conj(),
                                         r2[ki, kb])
                        s1 -= einsum('ibac,ibc->a', exchange.conj(),
                                     r2[ki, kb])
                        s2[ki, kb] += einsum('icab,a->ibc', direct, r1)

    #########    ########## ADC(2) ajk - bil block ############################

                    s2[ki, kb] -= einsum('i,ibc->ibc', e_occ[ki], r2[ki, kb])
                    s2[ki, kb] += einsum('b,ibc->ibc', e_vir[kb], r2[ki, kb])
                    s2[ki, kb] += einsum('c,ibc->ibc', e_vir[kc], r2[ki, kb])

    ################ ADC(3) ajk - bil block ############################

            if (method == "adc(2)-x" or method == "adc(3)"):
                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo

                for kx in range(nkpts):
                    for ky in range(nkpts):
                        ki = kconserv[ky, kshift, kx]
                        for kz in range(nkpts):
                            kw = kconserv[kx, kz, ky]

                            if isinstance(eris.vvvv, np.ndarray):
                                eris_vvvv = eris.vvvv.reshape(nkpts, nkpts,
                                    nkpts, nvir * nvir, nvir * nvir)
                                r2_1 = r2.reshape(nkpts, nkpts, nocc,
                                    nvir * nvir)
                                s2[ki, kx] += np.dot(r2_1[ki, kw], eris_vvvv[
                                    kx, ky, kw].T.conj()).reshape(nocc, nvir,
                                    nvir)
                            elif eris.vvvv is None:
                                s2[ki, kx] += self.ea_contract_r_vvvv(r2[ki,
                                    kw], eris.Lvv, kx, ky, kw)
                            else:
                                s2[ki, kx] += self.ea_contract_r_vvvv(r2[ki,
                                    kw], eris.vvvv, kx, ky, kw)
                        for kj in range(nkpts):
                            kz = kconserv[ky, ki, kj]
                            s2[ki, kx] -= 0.5 * einsum('iyzj,jzx->ixy',
                                eris_ovvo[ki, ky, kz], r2[kj, kz])
                            s2[ki, kx] += einsum('iyzj,jxz->ixy', eris_ovvo[ki,
                                ky, kz], r2[kj, kx])
                            s2[ki, kx] -= 0.5 * einsum('ijzy,jxz->ixy',
                                eris_oovv[ki, kj, kz], r2[kj, kx])

                            kz = kconserv[kj, ki, kx]
                            s2[ki, kx] -= 0.5 * einsum('ijzx,jzy->ixy',
                                eris_oovv[ki, kj, kz], r2[kj, kz])

                            kw = kconserv[kj, ki, kx]
                            s2[ki, kx] -= 0.5 * einsum('ijwx,jwy->ixy',
                                eris_oovv[ki, kj, kw], r2[kj, kw])

                            kw = kconserv[kj, ki, ky]
                            s2[ki, kx] -= 0.5 * einsum('ijwy,jxw->ixy',
                                eris_oovv[ki, kj, kw], r2[kj, kx])
                            s2[ki, kx] += einsum('iywj,jxw->ixy', eris_ovvo[ki,
                                ky, kw], r2[kj, kx])
                            s2[ki, kx] -= 0.5 * einsum('iywj,jwx->ixy',
                                eris_ovvo[ki, ky, kw], r2[kj, kw])

                if self.exxdiv is not None:
                    s2 += -madelung * r2
            if (method == "adc(3)"):

                eris_ovoo = eris.ovoo

    # ############### ADC(3) a - ibc block and ibc-a coupling blocks
    # ########################

                t2_1 = self.t2[0]

                for kz in range(nkpts):
                    for kw in range(nkpts):
                        kj = kconserv[kw, kshift, kz]

                        for kl in range(nkpts):
                            km = kconserv[kw, kl, kz]
                            t2_1_lmz = self.t2[0][kl, km, kz]
                            ka = kconserv[km, kj, kl]
                            temp_1 = einsum('lmzw,jzw->jlm', t2_1_lmz, r2[kj,
                                kz])
                            temp = 0.25 * einsum('lmzw,jzw->jlm', t2_1_lmz,
                                r2[kj, kz])
                            temp -= 0.25 * einsum('lmzw,jwz->jlm', t2_1_lmz,
                                r2[kj, kw])
                            del t2_1_lmz

                            t2_1_mlz = self.t2[0][km, kl, kz]
                            temp -= 0.25 * einsum('mlzw,jzw->jlm', t2_1_mlz,
                                r2[kj, kz])
                            temp += 0.25 * einsum('mlzw,jwz->jlm', t2_1_mlz,
                                r2[kj, kw])
                            del t2_1_mlz

                            s1 += einsum('jlm,lamj->a', temp, eris_ovoo[kl, ka,
                                km])
                            s1 -= einsum('jlm,malj->a', temp, eris_ovoo[km, ka,
                                kl])
                            s1 += einsum('jlm,lamj->a', temp_1, eris_ovoo[kl,
                                ka, km])

                            del temp
                            del temp_1

                for kx in range(nkpts):
                    for ky in range(nkpts):
                        ki = kconserv[ky, kshift, kx]
                        for kl in range(nkpts):
                            km = kconserv[kx, kl, ky]
                            kb = kconserv[km, ki, kl]
                            temp = einsum(
                                'b,lbmi->lmi', r1, eris_ovoo[kl, kb,
                                    km].conj())
                            s2[ki, kx] += einsum('lmi,lmxy->ixy', temp, t2_1[
                                kl, km, kx].conj())
                            del temp

                for kz in range(nkpts):
                    for kw in range(nkpts):
                        kj = kconserv[kz, kshift, kw]
                        for kl in range(nkpts):
                            kd = kconserv[kj, kw, kl]
                            t2_1_jlw = self.t2[0][kj, kl, kw]

                            temp_s_a = einsum('jlwd,jzw->lzd', t2_1_jlw, r2[kj,
                                kz])
                            temp_s_a -= einsum('jlwd,jwz->lzd', t2_1_jlw,
                                r2[kj, kw])

                            temp_t2_r2_1 = einsum('jlwd,jzw->lzd', t2_1_jlw,
                                r2[kj, kz])
                            temp_t2_r2_1 -= einsum('jlwd,jwz->lzd', t2_1_jlw,
                                r2[kj, kw])
                            temp_t2_r2_1 += einsum('jlwd,jzw->lzd', t2_1_jlw,
                                r2[kj, kz])
                            del t2_1_jlw

                            t2_1_ljw = self.t2[0][kl, kj, kw]
                            temp_s_a -= einsum('ljwd,jzw->lzd', t2_1_ljw,
                                r2[kj, kz])
                            temp_s_a += einsum('ljwd,jwz->lzd', t2_1_ljw,
                                r2[kj, kw])
                            temp_s_a += einsum('ljdw,jzw->lzd',
                                                   t2_1[kl, kj, kd], r2[kj,
                                                       kz])

                            temp_t2_r2_1 -= einsum('ljwd,jzw->lzd', t2_1_ljw,
                                r2[kj, kz])
                            del t2_1_ljw
                            temp_t2_r2_4 = einsum(
                                'jldw,jwz->lzd', t2_1[kj, kl, kd], r2[kj,
                                    kw])

                            if eris.ovvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):
                                    ka = kconserv[kz, kd, kl]
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kd],
                                        eris.Lvv[kz, ka],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    k = eris_ovvv.shape[0]

                                    s1 += 0.5 * einsum('lzd,ldza->a', temp_s_a[
                                        a:a + k], eris_ovvv)
                                    s1 += 0.5 * einsum('lzd,ldza->a',
                                        temp_t2_r2_1[a:a + k], eris_ovvv)
                                    del eris_ovvv

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, ka],
                                        eris.Lvv[kz, kd],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    s1 -= 0.5 * einsum('lzd,lazd->a', temp_s_a[
                                        a:a + k], eris_ovvv)
                                    s1 -= 0.5 * einsum('lzd,lazd->a',
                                        temp_t2_r2_4[a:a + k], eris_ovvv)
                                    del eris_ovvv
                                    a += k
                            else:
                                ka = kconserv[kz, kd, kl]
                                eris_ovvv = eris.ovvv[:]
                                s1 += 0.5 * einsum('lzd,ldza->a', temp_s_a,
                                                       eris_ovvv[kl, kd,
                                                           kz])
                                s1 += 0.5 * einsum('lzd,ldza->a', temp_t2_r2_1,
                                                       eris_ovvv[kl, kd,
                                                           kz])

                                s1 -= 0.5 * einsum('lzd,lazd->a', temp_s_a,
                                                       eris_ovvv[kl, ka,
                                                           kz])
                                s1 -= 0.5 * einsum('lzd,lazd->a', temp_t2_r2_4,
                                                       eris_ovvv[kl, ka,
                                                           kz])
                                del eris_ovvv
                            del temp_s_a
                            del temp_t2_r2_1
                            del temp_t2_r2_4

                for kz in range(nkpts):
                    for kw in range(nkpts):
                        kj = kconserv[kz, kshift, kw]
                        for kl in range(nkpts):
                            kd = kconserv[kj, kz, kl]
                            t2_1_jlz = self.t2[0][kj, kl, kz]

                            temp_s_a_1 = -einsum('jlzd,jwz->lwd',
                                                     t2_1_jlz, r2[kj,
                                                         kw])
                            temp_s_a_1 += einsum('jlzd,jzw->lwd',
                                                     t2_1_jlz, r2[kj,
                                                         kz])
                            temp_t2_r2_2 = -einsum('jlzd,jwz->lwd',
                                                       t2_1_jlz, r2[kj,
                                                           kw])
                            temp_t2_r2_2 += einsum('jlzd,jzw->lwd',
                                                       t2_1_jlz, r2[kj,
                                                           kz])
                            temp_t2_r2_2 -= einsum('jlzd,jwz->lwd',
                                                       t2_1_jlz, r2[kj,
                                                           kw])
                            del t2_1_jlz

                            t2_1_ljz = self.t2[0][kl, kj, kz]
                            temp_s_a_1 += einsum('ljzd,jwz->lwd', t2_1_ljz,
                                r2[kj, kw])
                            temp_s_a_1 -= einsum('ljzd,jzw->lwd', t2_1_ljz,
                                r2[kj, kz])
                            temp_s_a_1 -= einsum('ljdz,jwz->lwd',
                                                     t2_1[kl, kj, kd], r2[kj,
                                                         kw])

                            temp_t2_r2_2 += einsum('ljzd,jwz->lwd', t2_1_ljz,
                                r2[kj, kw])

                            temp_t2_r2_3 = -einsum('ljzd,jzw->lwd', t2_1_ljz,
                                r2[kj, kz])
                            del t2_1_ljz

                            if eris.ovvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):

                                    ka = kconserv[kw, kd, kl]
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kd],
                                        eris.Lvv[kw, ka],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    k = eris_ovvv.shape[0]
                                    s1 -= 0.5 * einsum('lwd,ldwa->a',
                                        temp_s_a_1[a:a + k], eris_ovvv)
                                    s1 -= 0.5 * einsum('lwd,ldwa->a',
                                        temp_t2_r2_2[a:a + k], eris_ovvv)
                                    del eris_ovvv

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, ka],
                                        eris.Lvv[kw, kd],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    s1 += 0.5 * einsum('lwd,lawd->a',
                                        temp_s_a_1[a:a + k], eris_ovvv)
                                    s1 += 0.5 * einsum('lwd,lawd->a',
                                        temp_t2_r2_3[a:a + k], eris_ovvv)
                                    del eris_ovvv
                                    a += k

                            else:
                                ka = kconserv[kw, kd, kl]
                                eris_ovvv = eris.ovvv[:]
                                s1 -= 0.5 * einsum('lwd,ldwa->a', temp_s_a_1,
                                                       eris_ovvv[kl, kd,
                                                           kw])
                                s1 += 0.5 * einsum('lwd,lawd->a', temp_s_a_1,
                                                       eris_ovvv[kl, ka,
                                                           kw])

                                s1 -= 0.5 * einsum('lwd,ldwa->a', temp_t2_r2_2,
                                                       eris_ovvv[kl, kd,
                                                           kw])
                                s1 += 0.5 * einsum('lwd,lawd->a', temp_t2_r2_3,
                                                       eris_ovvv[kl, ka,
                                                           kw])
                                del eris_ovvv

                            del temp_s_a_1
                            del temp_t2_r2_2
                            del temp_t2_r2_3

                for kx in range(nkpts):
                    for ky in range(nkpts):
                        ki = kconserv[ky, kshift, kx]
                        for kl in range(nkpts):
                            temp = np.zeros((nocc, nvir, nvir),
                                dtype=np.complex128)
                            temp_1_1 = np.zeros((nocc, nvir, nvir),
                                dtype=np.complex128)
                            temp_2_1 = np.zeros((nocc, nvir, nvir),
                                dtype=np.complex128)
                            if eris.ovvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):
                                    kd = kconserv[kl, kshift, kx]
                                    kb = kconserv[kl, kd, kx]

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kd],
                                        eris.Lvv[kx, kb],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    k = eris_ovvv.shape[0]
                                    temp_1_1[a:a + k] += einsum('ldxb,b->lxd',
                                        eris_ovvv, r1)
                                    temp_2_1[a:a + k] += einsum('ldxb,b->lxd',
                                        eris_ovvv, r1)
                                    del eris_ovvv

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kb],
                                        eris.Lvv[kx, kd],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    temp_1_1[a:a + k] -= einsum('lbxd,b->lxd',
                                        eris_ovvv, r1)
                                    del eris_ovvv

                                    kd = kconserv[kl, kshift, ky]
                                    kb = kconserv[ky, kd, kl]
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kb],
                                        eris.Lvv[ky, kd],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    temp[a:a + k] -= einsum('lbyd,b->lyd',
                                        eris_ovvv, r1)
                                    del eris_ovvv
                                    a += k
                            else:
                                kd = kconserv[ki, ky, kl]
                                kb = kconserv[kl, kd, kx]
                                eris_ovvv = eris.ovvv[:]
                                temp_1_1 += einsum('ldxb,b->lxd', eris_ovvv[kl,
                                    kd, kx], r1)
                                temp_1_1 -= einsum('lbxd,b->lxd', eris_ovvv[kl,
                                    kb, kx], r1)
                                temp_2_1 += einsum('ldxb,b->lxd', eris_ovvv[kl,
                                    kd, kx], r1)

                                kd = kconserv[kl, kshift, ky]
                                kb = kconserv[ky, kd, kl]
                                temp -= einsum('lbyd,b->lyd', eris_ovvv[kl, kb,
                                    ky], r1)
                                del eris_ovvv

                            kd = kconserv[kl, ky, ki]
                            s2[ki, kx] += einsum('lxd,lidy->ixy',
                                temp_1_1.conj(), t2_1[kl, ki, kd].conj())
                            s2[ki, kx] += einsum('lxd,ilyd->ixy',
                                temp_2_1.conj(), t2_1[ki, kl, ky].conj())
                            s2[ki, kx] -= einsum('lxd,ildy->ixy',
                                temp_2_1.conj(), t2_1[ki, kl, kd].conj())

                            kd = kconserv[kl, kx, ki]
                            s2[ki, kx] += einsum('lyd,lixd->ixy', temp.conj(),
                                t2_1[kl, ki, kx].conj())
                            del temp
                            del temp_1_1
                            del temp_2_1

            s2 = s2.reshape(-1)
            s = np.hstack((s1, s2))
            del s1
            del s2

            cput0 = log.timer_debug1("completed sigma vector calculation",
                *cput0)

            return s
        return sigma_

    def get_trans_moments(self, kshift):
        nmo = self.nmo
        T = []
        for orb in range(nmo):
            T_a = self.get_trans_moments_orbital(orb, kshift)
            T.append(T_a)

        T = np.array(T)
        return T

    def get_trans_moments_orbital(self, orb, kshift):
        self.method = check_implemented(self.method)

        method = self.method
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nvir
        t2_1 = self.t2[0]

        idn_vir = np.identity(nvir)

        T1 = np.zeros((nvir), dtype=np.complex128)
        T2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)

    ######## ADC(2) 1h part  ############################################

        if orb < nocc:

            if (self.approx_trans_moments is False or self.method == "adc(3)"):
                t1_2 = self.t1[0]
                T1 -= t1_2[kshift][orb, :]

            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = self.khelper.kconserv[kj, ka, kshift]
                    ki = self.khelper.kconserv[ka, kj, kb]

                    t2_1_t = t2_1[ki, kj, ka].transpose(1, 0, 2, 3)
                    T2[kj, ka] -= t2_1_t[:, orb, :, :].conj()

        else:

            T1 += idn_vir[(orb - nocc), :]
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kl = self.khelper.kconserv[kc, kk, kshift]
                    ka = self.khelper.kconserv[kc, kl, kk]
                    T1 -= 0.25 * einsum('klc,klac->a', t2_1[kk, kl, kshift][:,
                        :, orb - nocc, :], t2_1[kk, kl, ka].conj())
                    T1 -= 0.25 * einsum('lkc,lkac->a', t2_1[kl, kk, kshift][:,
                        :, orb - nocc, :], t2_1[kl, kk, ka].conj())

                    T1 -= 0.25 * einsum('klc,klac->a', t2_1[kk, kl, kshift][:,
                        :, orb - nocc, :], t2_1[kk, kl, ka].conj())
                    T1 += 0.25 * einsum('lkc,klac->a', t2_1[kl, kk, kshift][:,
                        :, orb - nocc, :], t2_1[kk, kl, ka].conj())
                    T1 += 0.25 * einsum('klc,lkac->a', t2_1[kk, kl, kshift][:,
                        :, orb - nocc, :], t2_1[kl, kk, ka].conj())
                    T1 -= 0.25 * einsum('lkc,lkac->a', t2_1[kl, kk, kshift][:,
                        :, orb - nocc, :], t2_1[kl, kk, ka].conj())

    ######### ADC(3) 2p-1h  part  ############################################

        if (self.method == 'adc(2)-x' and self.approx_trans_moments is False
            or self.method == 'adc(3)'):
            t2_2 = self.t2[1]

            if orb < nocc:

                for kj in range(nkpts):
                    for ka in range(nkpts):
                        kb = self.khelper.kconserv[kj, ka, kshift]
                        ki = self.khelper.kconserv[ka, kj, kb]

                        t2_2_t = t2_2[ki, kj, ka].conj().transpose(1, 0, 2, 3)

                        T2[kj, ka] -= t2_2_t[:, orb, :, :].conj()

    ########### ADC(3) 1p part  ############################################

        if (method == 'adc(3)'):
            if orb < nocc:
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        ka = self.khelper.kconserv[kk, kc, kshift]
                        T1 += 0.5 * einsum('kac,ck->a', t2_1[kk, kshift, kc][:,
                            orb, :, :], t1_2[kc].T)
                        T1 -= 0.5 * einsum('kac,ck->a', t2_1[kshift, kk, ka][
                            orb, :, :, :], t1_2[kc].T)
                        T1 -= 0.5 * einsum('kac,ck->a', t2_1[kshift, kk, ka][
                            orb, :, :, :], t1_2[kc].T)

            else:
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        kl = self.khelper.kconserv[kk, kc, kshift]
                        ka = self.khelper.kconserv[kl, kc, kk]

                        T1 -= 0.25 * einsum('klc,klac->a', t2_1[kk, kl,
                            kshift][:, :, orb - nocc, :], t2_2[kk, kl,
                            ka].conj())
                        T1 -= 0.25 * einsum('lkc,lkac->a', t2_1[kl, kk,
                            kshift][:, :, orb - nocc, :], t2_2[kl, kk,
                            ka].conj())
                        T1 -= 0.25 * einsum('klc,klac->a', t2_1[kk, kl,
                            kshift][:, :, orb - nocc, :], t2_2[kk, kl,
                            ka].conj())
                        T1 += 0.25 * einsum('klc,lkac->a', t2_1[kk, kl,
                            kshift][:, :, orb - nocc, :], t2_2[kl, kk,
                            ka].conj())
                        T1 += 0.25 * einsum('lkc,klac->a', t2_1[kl, kk,
                            kshift][:, :, orb - nocc, :], t2_2[kk, kl,
                            ka].conj())
                        T1 -= 0.25 * einsum('lkc,lkac->a', t2_1[kl, kk,
                            kshift][:, :, orb - nocc, :], t2_2[kl, kk,
                            ka].conj())

                        T1 -= 0.25 * \
                            einsum('klac,klc->a', t2_1[kk, kl, ka].conj(),
                                       t2_2[kk, kl, kshift][:, :, (orb - nocc),
                                           :])
                        T1 -= 0.25 * \
                            einsum('lkac,lkc->a', t2_1[kl, kk, ka].conj(),
                                       t2_2[kl, kk, kshift][:, :, (orb - nocc),
                                           :])
                        T1 -= 0.25 * \
                            einsum('klac,klc->a', t2_1[kk, kl, ka].conj(),
                                       t2_2[kk, kl, kshift][:, :, (orb - nocc),
                                           :])
                        T1 += 0.25 * \
                            einsum('klac,lkc->a', t2_1[kk, kl, ka].conj(),
                                       t2_2[kl, kk, kshift][:, :, (orb - nocc),
                                           :])
                        T1 += 0.25 * \
                            einsum('lkac,klc->a', t2_1[kl, kk, ka].conj(),
                                       t2_2[kk, kl, kshift][:, :, (orb - nocc),
                                           :])
                        T1 -= 0.25 * \
                            einsum('lkac,lkc->a', t2_1[kl, kk, ka].conj(),
                                       t2_2[kl, kk, kshift][:, :, (orb - nocc),
                                           :])

            del t2_2
        del t2_1

        for ka in range(nkpts):
            for kb in range(nkpts):
                ki = self.khelper.kconserv[kb, kshift, ka]
                T2[ki, ka] += T2[ki, ka] - T2[ki, kb].transpose(0, 2, 1)

        T2 = T2.reshape(-1)
        T = np.hstack((T1, T2))

        return T

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        nkpts = self.nkpts
        nocc = self.t2[0].shape[3]
        nvir = self.nvir
        n_singles = nvir

        for I in range(U.shape[1]):
            U1 = U[:n_singles, I]
            U2 = U[n_singles:, I].reshape(nkpts, nkpts, nocc, nvir, nvir)
            UdotU = np.dot(U1.conj().ravel(), U1.ravel())
            for ka in range(nkpts):
                for kb in range(nkpts):
                    ki = self.khelper.kconserv[kb, kshift, ka]
                    UdotU += 2.0 * np.dot(U2[ki, ka].conj().ravel(), U2[ki,
                        ka].ravel()) - np.dot(U2[ki, ka].conj().ravel(), U2[ki,
                        kb].transpose(0, 2, 1).ravel())
            U[:, I] /= np.sqrt(UdotU)

        U = U.reshape(-1, nroots)

        return U

    def make_rdm1(self, root=None, kptlist=None, K_idx=None, if_ss=False):
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        rdm1 = []

        for kpt, kshift in enumerate(K_idx):
            k = kptlist.index(kshift)
            U = np.array(self.U[k]).T.copy()
            U = self.renormalize_eigenvectors(kshift, U, self.U.shape[1])
            rdm1_k = []
            for i in root[kpt]:
                rdm1_k.append(self.make_rdm1_eigenvectors(U[:, i], U[:, i],
                    kshift, if_ss))
            rdm1_k = np.stack(rdm1_k, axis=0)
            rdm1.append(rdm1_k)
        cput0 = log.timer_debug1("completed OPDM calculation", *cput0)
        return rdm1

    def make_rdm1_eigenvectors_slow(self, L, R, kshift):
        L = np.array(L).ravel().conj()
        R = np.array(R).ravel()

        t2_ce = self.t1[0]
        t1_oovv = self.t2[0]

        kconserv = self.khelper.kconserv

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts

        n_singles = nvir
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        kd_oc = np.identity(nocc)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        R2 = R2.reshape(nkpts, nkpts, nocc, nvir, nvir)

    # block- ij
        # 000
        for ki in range(nkpts):
            rdm1[ki][:nocc, :nocc] += 2 * einsum('ij,m,m->ij', kd_oc, L1, R1)
        # 101
        for ka in range(nkpts):
            for ki in range(nkpts):
                kb = kconserv[ki, ka, kshift]
                rdm1[ki][:nocc, :nocc] -= 2 * einsum('Jab,Iab->IJ', L2[ki][ka],
                    R2[ki][ka])
                rdm1[ki][:nocc, :nocc] += einsum('Jab,Iba->IJ', L2[ki][ka],
                    R2[ki][kb])
        for kj in range(nkpts):
            for ki in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[ki, ka, kshift]
                    rdm1[kj][:nocc, :nocc] += 4 * einsum('iab,iab,IJ->IJ',
                        L2[ki][ka], R2[ki][ka], kd_oc)
                    rdm1[kj][:nocc, :nocc] -= 2 * einsum('iab,iba,IJ->IJ',
                        L2[ki][ka], R2[ki][kb], kd_oc)
        # 020
        for kj in range(nkpts):
            for ki in range(nkpts):
                for kb in range(nkpts):
                    kc = kconserv[kj, kb, ki]
                    rdm1[kj][:nocc, :nocc] -= 4 * einsum('a,a,Iibc,Jibc->IJ',
                        L1, R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kb].conj(
                        ))
                    rdm1[kj][:nocc, :nocc] += 2 * einsum('a,a,Iibc,Jicb->IJ',
                        L1, R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kc].conj(
                        ))
                kb = kshift
                kc = kconserv[kj, kb, ki]
                rdm1[kj][:nocc, :nocc] += 2 * einsum('a,b,Iiac,Jibc->IJ', L1,
                    R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kb].conj())
                rdm1[kj][:nocc, :nocc] -= einsum('a,b,Iiac,Jicb->IJ', L1, R1,
                    t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kc].conj())
                rdm1[kj][:nocc, :nocc] -= einsum('a,b,Iica,Jibc->IJ', L1, R1,
                    t1_oovv[kj][ki][kc], t1_oovv[kj][ki][kb].conj())
                rdm1[kj][:nocc, :nocc] += 2 * einsum('a,b,Iica,Jicb->IJ', L1,
                    R1, t1_oovv[kj][ki][kc], t1_oovv[kj][ki][kc].conj())

    # block- ab
        # 000
        rdm1[kshift][nocc:, nocc:] += einsum('A,B->AB', L1, R1)
        # 101
        for ka in range(nkpts):
            for ki in range(nkpts):
                kb = kconserv[ki, ka, kshift]
                rdm1[kb][nocc:, nocc:] += 2 * einsum('iAa,iBa->AB', L2[ki][kb],
                    R2[ki][kb])
                rdm1[kb][nocc:, nocc:] -= einsum('iAa,iaB->AB', L2[ki][kb],
                    R2[ki][ka])
                rdm1[kb][nocc:, nocc:] -= einsum('iaA,iBa->AB', L2[ki][ka],
                    R2[ki][kb])
                rdm1[kb][nocc:, nocc:] += 2 * einsum('iaA,iaB->AB', L2[ki][ka],
                    R2[ki][ka])
        # 020
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kshift
                kb = kconserv[ki, ka, kj]
                rdm1[ka][nocc:, nocc:] -= einsum('A,a,ijBb,ijab->AB', L1, R1,
                    t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ka][nocc:, nocc:] += 1 / 2 * einsum('A,a,ijBb,ijba->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())
                rdm1[ka][nocc:, nocc:] -= einsum('a,B,ijab,ijAb->AB', L1, R1,
                    t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ka][nocc:, nocc:] += 1 / 2 * einsum('a,B,ijab,ijbA->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())
                for ka in range(nkpts):
                    kb = kconserv[ki, ka, kj]
                    rdm1[ka][nocc:, nocc:] += 4 * einsum('a,a,ijBb,ijAb->AB',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj(
                        ))
                    rdm1[ka][nocc:, nocc:] -= 2 * einsum('a,a,ijBb,ijbA->AB',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj(
                        ))
                kb = kshift
                ka = kconserv[ki, kb, kj]
                rdm1[ka][nocc:, nocc:] -= 2 * einsum('a,b,ijBa,ijAb->AB', L1,
                    R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ka][nocc:, nocc:] += einsum('a,b,ijBa,ijbA->AB', L1, R1,
                    t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())

    # block- ia
        # 100
        for ki in range(nkpts):
            rdm1[ki][:nocc, nocc:] -= einsum('a,IAa->IA', L1, R2[ki][ki])
            ka = kshift
            rdm1[ki][:nocc, nocc:] += 2 * einsum('a,IaA->IA', L1, R2[ki][ka])
        # 110
        for ki in range(nkpts):
            for ka in range(nkpts):
                kj = kshift
                kb = kconserv[ki, ka, kj]
                rdm1[kj][:nocc, nocc:] -= 2 * einsum('iab,A,Iiab->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][ka])
                rdm1[kj][:nocc, nocc:] += einsum('iab,A,Iiba->IA', L2[ki][ka],
                    R1, t1_oovv[kj][ki][kb])
            for kj in range(nkpts):
                ka = kshift
                kb = kconserv[kj, kj, ki]
                rdm1[kj][:nocc, nocc:] += 4 * einsum('iab,a,IiAb->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kj])
                rdm1[kj][:nocc, nocc:] -= 2 * einsum('iab,a,IibA->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kb])
                rdm1[kj][:nocc, nocc:] -= 2 * einsum('iab,b,IiAa->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kj])
                rdm1[kj][:nocc, nocc:] += einsum('iab,b,IiaA->IA', L2[ki][ka],
                    R1, t1_oovv[kj][ki][ka])

    # block- ai
        for ki in range(nkpts):
            rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

        if self.approx_trans_moments is False or self.method == "adc(3)":
            ki = kshift
            ### block- ia ###
            rdm1[ki][:nocc, nocc:] -= einsum('a,A,Ia->IA', L1, R1, t2_ce[ki])
            for kj in range(nkpts):
                rdm1[kj][:nocc, nocc:] += 2 * einsum('a,a,IA->IA', L1, R1,
                    t2_ce[kj])
            ### block- ai ###
                rdm1[kj][nocc:, :nocc] = rdm1[kj][:nocc, nocc:].conj().T

        return rdm1

    def make_rdm1_eigenvectors(self, L, R, kshift, if_ss):
        L = np.array(L).ravel().conj()
        R = np.array(R).ravel()

        t2_ce = self.t1[0]
        t1_oovv = self.t2[0]

        kconserv = self.khelper.kconserv
        kconserv_s = kconserv[kshift, :, :]

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        occ_list = range(nocc)

        t1_oovv_sjb = np.zeros_like(t1_oovv[0])
        t1_oovv_sja = t1_oovv[kshift]
        t1_oovv_ijs = t1_oovv[:, :, kshift, :, :, :]

        kj, ka = np.indices((nkpts, nkpts))
        kb = self.khelper.kconserv[kshift, ka, kj]
        t1_oovv_sjb[kj, ka] = t1_oovv_sja[kj, kb]

        n_singles = nvir
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        R2 = R2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        R2_iba = np.zeros_like(R2)
        L2_iba = np.zeros_like(L2)

        ka, ki = np.indices((nkpts, nkpts))
        kb = kconserv_s[ka, ki]
        R2_iba[ki, ka] = R2[ki, kb]
        L2_iba[ki, ka] = L2[ki, kb]
        R2_isb = R2[:, kshift, :, :, :]
        L2_isb = L2[:, kshift, :, :, :]
    # block- ij
        # 000
        rdm1[:, occ_list, occ_list] += 2 * einsum('m,m->', L1, R1)
        # 101
        rdm1[:, :nocc, :nocc] -= 2 * einsum('KkJab,KkIab->KIJ', L2, R2)
        rdm1[:, :nocc, :nocc] += einsum('KkJab,KkIba->KIJ', L2, R2_iba)
        rdm1[:, occ_list, occ_list] += 4 * einsum('Kkiab,Kkiab->', L2, R2)
        rdm1[:, occ_list, occ_list] -= 2 * einsum('Kkiab,Kkiba->', L2, R2_iba)
        # 020
        rdm1[:, :nocc, :nocc] += 2 * einsum('a,b,KkIiac,KkJibc->KIJ', L1, R1,
            t1_oovv_ijs, t1_oovv_ijs.conj())

    # block- ab
        # 000
        rdm1[kshift, nocc:, nocc:] += einsum('A,B->AB', L1, R1)
        # 101
        rdm1[:, nocc:, nocc:] += 2 * einsum('kKiAa,kKiBa->KAB', L2, R2)
        rdm1[:, nocc:, nocc:] -= einsum('kKiAa,kKiaB->KAB', L2, R2_iba)
        rdm1[:, nocc:, nocc:] -= einsum('kKiaA,kKiBa->KAB', L2_iba, R2)
        rdm1[:, nocc:, nocc:] += 2 * einsum('kKiaA,kKiaB->KAB', L2_iba, R2_iba)
        # 020
        rdm1[kshift, nocc:, nocc:] -= einsum('A,a,kKijBb,kKijab->AB', L1, R1,
            t1_oovv_ijs, t1_oovv_ijs.conj())
        rdm1[kshift, nocc:, nocc:] -= einsum('a,B,kKijab,kKijAb->AB', L1, R1,
            t1_oovv_ijs, t1_oovv_ijs.conj())

    # block- ia
        # 100
        rdm1[:, :nocc, nocc:] -= einsum('a,KKIAa->KIA', L1, R2)
        rdm1[:, :nocc, nocc:] += 2 * einsum('a,KIaA->KIA', L1, R2_isb)
        # 110
        rdm1[kshift, :nocc, nocc:] -= 2 * einsum('kKiab,A,kKIiab->IA', L2, R1,
            t1_oovv_sja)
        rdm1[kshift, :nocc, nocc:] += einsum('kKiab,A,kKIiba->IA', L2, R1,
            t1_oovv_sjb)
        rdm1[:, :nocc, nocc:] += einsum('kiab,b,KkIiaA->KIA', L2_isb, R1,
            t1_oovv_ijs)

        del (R2_iba)
        del (L2_iba)
        del (R2_isb)
        del (t1_oovv_sjb)
        del (t1_oovv_sja)

        kj, ka = np.indices((nkpts, nkpts))
        idx0 = np.arange(nkpts)[:, None]
        LR = np.dot(L1, R1)
        if if_ss:
            rdm1[:, :nocc, :nocc] += np.identity(nocc)
            LR += 1.
        t1_oovv_ijs_b = np.zeros_like(t1_oovv[0])
        oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
        vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
        ov = np.zeros((nkpts, nocc, nvir), dtype=np.complex128)
        for ki in range(nkpts):
            kb = self.khelper.kconserv[ki, ka, kj]
            t1_oovv_np = np.array(t1_oovv[ki])
            t1_oovv_ijb = t1_oovv_np[idx0, kb]
            t1_oovv_ijs_b[ki] = t1_oovv_ijb[:, kshift].copy()
    # block- ij
            oo[ki] -= 4 * einsum('KkIibc,KkJibc->IJ', t1_oovv_np,
                t1_oovv_np.conj())
            oo[ki] += 2 * einsum('KkIibc,KkJicb->IJ', t1_oovv_np,
                t1_oovv_ijb.conj())
    # block- ab
            vv += 4 * einsum('kKijBb,kKijAb->KAB', t1_oovv_np,
                t1_oovv_np.conj())
            vv -= 2 * einsum('kKijBb,kKijbA->KAB', t1_oovv_np,
                t1_oovv_ijb.conj())
    # block- ia
            ov[ki] += 4 * einsum('kiab,a,kIiAb->IA', L2_isb, R1, t1_oovv_np[:,
                ki])
            ov[ki] -= 2 * einsum('kiab,a,kkIibA->IA', L2_isb, R1, t1_oovv_np)
            ov[ki] -= 2 * einsum('kiab,b,kIiAa->IA', L2_isb, R1, t1_oovv_np[:,
                ki])
            del (t1_oovv_ijb)
            del (t1_oovv_np)

        oo *= LR
        vv *= LR
        rdm1[:, :nocc, :nocc] += oo
        rdm1[:, nocc:, nocc:] += vv
        rdm1[:, :nocc, nocc:] += ov
        del (oo)
        del (vv)
        del (ov)
        del (L2_isb)

        ki, ka = np.indices((nkpts, nkpts))
        kj = kconserv_s[ki, ka]
        t1_oovv_ijas = t1_oovv_ijs_b[idx0, kj]
        t1_oovv_ijsa = t1_oovv_ijs[idx0, kj]
    # block- ij
        rdm1[:, :nocc, :nocc] -= einsum('a,b,KkIiac,KkJicb->KIJ', L1, R1,
            t1_oovv_ijs, t1_oovv_ijs_b.conj())
        rdm1[:, :nocc, :nocc] -= einsum('a,b,KkIica,KkJibc->KIJ', L1, R1,
            t1_oovv_ijs_b, t1_oovv_ijs.conj())
        rdm1[:, :nocc, :nocc] += 2 * einsum('a,b,KkIica,KkJicb->KIJ', L1, R1,
            t1_oovv_ijs_b, t1_oovv_ijs_b.conj())
    # block- ab
        rdm1[kshift, nocc:, nocc:] += 1 / 2 * einsum('A,a,kKijBb,kKijba->AB',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs_b.conj())
        rdm1[kshift, nocc:, nocc:] += 1 / 2 * einsum('a,B,kKijab,kKijbA->AB',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs_b.conj())
        rdm1[:, nocc:, nocc:] -= 2 * einsum('a,b,kKijBa,kKijAb->KAB', L1, R1,
            t1_oovv_ijas, t1_oovv_ijas.conj())
        rdm1[:, nocc:, nocc:] += einsum('a,b,kKijBa,kKijbA->KAB', L1, R1,
            t1_oovv_ijas, t1_oovv_ijsa.conj())
        del (t1_oovv_ijas)
        del (t1_oovv_ijsa)
        del (t1_oovv_ijs)
        del (t1_oovv_ijs_b)

    # block- ai
        rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

        if self.approx_trans_moments is False or self.method == "adc(3)":
            rdm1[kshift, :nocc, nocc:] -= einsum('a,A,Ia->IA', L1, R1,
                t2_ce[kshift])
            rdm1[:, :nocc, nocc:] += 2 * LR * t2_ce
            ### block- ai ###
            rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0,
                2, 1)

        return rdm1

    def get_properties(self, kshift, U, nroots=1):
        # Transition moments
        T = self.get_trans_moments(kshift)

        # Spectroscopic amplitudes
        U = self.renormalize_eigenvectors(kshift, U, nroots)
        X = np.dot(T, U).reshape(-1, nroots)

        # Spectroscopic factors
        P = 2.0 * einsum("pi,pi->i", X.conj(), X)
        P = P.real

        return P, X

    def mask_frozen_ea(self, v1, v2, kshift, const=LARGE_DENOM):
        '''Replaces all frozen orbital indices of `vector` with the value
        `const`.'''
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self)

        new_v1 = const * np.ones_like(v1)
        new_v2 = const * np.ones_like(v2)

        new_v1[nonzero_vpadding[kshift]] = v1[nonzero_vpadding[kshift]]

        for ki in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, ki]
                idx = np.ix_([ki], [ka], nonzero_opadding[ki],
                    nonzero_vpadding[ka], nonzero_vpadding[kb])
                new_v2[idx] = v2[idx]

        return (new_v1, new_v2)

    def ea_contract_r_vvvv(self, r2, vvvv, ka, kb, kc):
        nocc = r2.shape[0]
        nvir = r2.shape[1]
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        kd = kconserv[ka, kc, kb]
        r2 = np.ascontiguousarray(r2.reshape(nocc, -1))
        r2_vvvv = np.zeros((nvir, nvir, nocc),
                           dtype=np.result_type(r2, self.mo_coeff[0]))
        chunk_size = self.chunk_size
        chunk_size = min(chunk_size, nvir)

        a = 0
        if isinstance(vvvv, np.ndarray) and vvvv.ndim == 5:
            vv1 = vvvv[ka, kc]
            vv2 = vvvv[kb, kd]
            for p in range(0, nvir, chunk_size):
                vvvv_p = _ChemistsERIs.vvvv_df(vv1, vv2, p,
                    chunk_size) / nkpts
                k = vvvv_p.shape[0]
                vvvv_p = vvvv_p.reshape(-1, nvir * nvir)
                r2_vvvv[a:a + k] += np.dot(vvvv_p.conj(), r2.T).reshape(-1,
                    nvir, nocc)
                del vvvv_p
                a += k
        else:
            for p in range(0, nvir, chunk_size):
                vvvv_p = vvvv[ka, kb, kc][p:p + chunk_size].reshape(-1,
                    nvir * nvir)
                k = vvvv_p.shape[0]
                r2_vvvv[a:a + k] += np.dot(vvvv_p.conj(), r2.T).reshape(-1,
                    nvir, nocc)
                del vvvv_p
                a += k

        r2_vvvv = np.ascontiguousarray(r2_vvvv.transpose(2, 0, 1))

        return r2_vvvv

    _mask_frozen = mask_frozen_ea


def _as_solver(adc):
    """Accept both excitation solvers and ground-state ADC drivers."""
    return adc if isinstance(adc, RADCEA) else RADCEA(adc)


def vector_size(adc):
    """Compatibility wrapper for :meth:`RADCEA.vector_size`."""
    return _as_solver(adc).vector_size()


def get_imds(adc, eris=None):
    """Compatibility wrapper for :meth:`RADCEA.get_imds`."""
    return _as_solver(adc).get_imds(eris)


def get_diag(adc, kshift, M_ab=None, eris=None):
    """Compatibility wrapper for :meth:`RADCEA.get_diag`."""
    return _as_solver(adc).get_diag(kshift, M_ab, eris)


def matvec(adc, kshift, M_ab=None, eris=None):
    """Compatibility wrapper for :meth:`RADCEA.matvec`."""
    return _as_solver(adc).matvec(kshift, M_ab, eris)


def get_trans_moments(adc, kshift):
    """Compatibility wrapper for :meth:`RADCEA.get_trans_moments`."""
    return _as_solver(adc).get_trans_moments(kshift)


def get_trans_moments_orbital(adc, orb, kshift):
    """Compatibility wrapper for :meth:`RADCEA.get_trans_moments_orbital`."""
    return _as_solver(adc).get_trans_moments_orbital(orb, kshift)


def renormalize_eigenvectors(adc, kshift, U, nroots=1):
    """Compatibility wrapper for :meth:`RADCEA.renormalize_eigenvectors`."""
    return _as_solver(adc).renormalize_eigenvectors(kshift, U, nroots)


def make_rdm1(adc, root=None, kptlist=None, K_idx=None, if_ss=False):
    """Compatibility wrapper for :meth:`RADCEA.make_rdm1`."""
    return _as_solver(adc).make_rdm1(root, kptlist, K_idx, if_ss)


def make_rdm1_eigenvectors_slow(adc, L, R, kshift):
    """Compatibility wrapper for :meth:`RADCEA.make_rdm1_eigenvectors_slow`."""
    return _as_solver(adc).make_rdm1_eigenvectors_slow(L, R, kshift)


def make_rdm1_eigenvectors(adc, L, R, kshift, if_ss):
    """Compatibility wrapper for :meth:`RADCEA.make_rdm1_eigenvectors`."""
    return _as_solver(adc).make_rdm1_eigenvectors(L, R, kshift, if_ss)


def get_properties(adc, kshift, U, nroots=1):
    """Compatibility wrapper for :meth:`RADCEA.get_properties`."""
    return _as_solver(adc).get_properties(kshift, U, nroots)


def mask_frozen_ea(adc, v1, v2, kshift, const=LARGE_DENOM):
    """Compatibility wrapper for :meth:`RADCEA.mask_frozen_ea`."""
    return _as_solver(adc).mask_frozen_ea(v1, v2, kshift, const)


def ea_contract_r_vvvv(adc, r2, vvvv, ka, kb, kc):
    """Compatibility wrapper for :meth:`RADCEA.ea_contract_r_vvvv`."""
    return _as_solver(adc).ea_contract_r_vvvv(r2, vvvv, ka, kb, kc)
