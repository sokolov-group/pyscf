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


class RADCIP(kadc_rhf.KRADC):
    """Restricted periodic IP-ADC solver.

    Args:
        adc (KRADC): Reference driver with amplitudes, normally prepared
            by ``ground_state_energy()``. Reference scratch ownership is
            retained while this solver uses its disk-backed amplitudes.

    Vectors contain one-hole and two-hole/one-particle blocks. The doubles
    shape is ``(nk, nk, nvir, nocc, nocc)``; the fourth momentum is conserved.
    ``kernel(nroots, kptlist=...)`` returns E, U, P and X with sectors in
    kptlist order. E and P have shape (nsectors, nroots), U has shape
    (nsectors, nroots, vector_size), and X has shape
    (nsectors, nmo, nroots). P contains spectroscopic factors.

    Example:
        >>> reference = adc.KRADC(kmf)
        >>> reference.ground_state_energy()
        >>> solver = RADCIP(reference)
        >>> E, U, P, X = solver.kernel(nroots=3, kptlist=[0])
    """

    _keys = ({'tol_residual', 'conv_tol', 'e_corr', 'method', 'mo_coeff',
        'mo_energy_b', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle',
        'kpts', 'exxdiv', 'khelper', 'cell', 'nkop_chk', 'kop_npick',
        'chunk_size'})

    def __init__(self, adc):
        self._init_from_ground_state(adc)
        self.type_ = 'ip'

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
        kconserv = self.khelper.kconserv

        idn_occ = np.identity(nocc)
        M_ij = np.empty((nkpts, nocc, nocc), dtype=mo_coeff[0].dtype)

        if eris is None:
            eris = self.ao2mo()

        # i-j block
        # Zeroth-order terms

        t2_1 = self.t2[0]
        eris_ovov = eris.ovov
        for ki in range(nkpts):
            kj = ki
            M_ij[ki] = einsum('ij,j->ij', idn_occ, e_occ[kj])
            for kl in range(nkpts):
                for kd in range(nkpts):
                    ke = kconserv[kj, kd, kl]
                    t2_1 = self.t2[0]
                    t2_1_ild = self.t2[0][ki, kl, kd]

                    M_ij[ki] += 0.5 * 0.5 * \
                        einsum('ilde,jdle->ij', t2_1_ild, eris_ovov[kj, kd,
                            kl])
                    M_ij[ki] -= 0.5 * 0.5 * \
                        einsum('ilde,jeld->ij', t2_1_ild, eris_ovov[kj, ke,
                            kl])
                    M_ij[ki] += 0.5 * einsum('ilde,jdle->ij', t2_1_ild,
                                                 eris_ovov[kj, kd,
                                                     kl])
                    del t2_1_ild

                    t2_1_lid = self.t2[0][kl, ki, kd]
                    M_ij[ki] -= 0.5 * 0.5 * \
                        einsum('lide,jdle->ij', t2_1_lid, eris_ovov[kj, kd,
                            kl])
                    M_ij[ki] += 0.5 * 0.5 * \
                        einsum('lide,jeld->ij', t2_1_lid, eris_ovov[kj, ke,
                            kl])
                    del t2_1_lid

                    t2_1_jld = self.t2[0][kj, kl, kd]
                    M_ij[ki] += 0.5 * 0.5 * \
                        einsum('jlde,idle->ij', t2_1_jld.conj(),
                                   eris_ovov[ki, kd, kl].conj())
                    M_ij[ki] -= 0.5 * 0.5 * \
                        einsum('jlde,ield->ij', t2_1_jld.conj(),
                                   eris_ovov[ki, ke, kl].conj())
                    M_ij[ki] += 0.5 * einsum('jlde,idle->ij', t2_1_jld.conj(),
                                                 eris_ovov[ki, kd,
                                                     kl].conj())
                    del t2_1_jld

                    t2_1_ljd = self.t2[0][kl, kj, kd]
                    M_ij[ki] -= 0.5 * 0.5 * \
                        einsum('ljde,idle->ij', t2_1_ljd.conj(),
                                   eris_ovov[ki, kd, kl].conj())
                    M_ij[ki] += 0.5 * 0.5 * \
                        einsum('ljde,ield->ij', t2_1_ljd.conj(),
                                   eris_ovov[ki, ke, kl].conj())
                    del t2_1_ljd
                    del t2_1

        cput0 = (log.timer_debug1(
            'Completed M_ij second-order terms ADC(2) calculation', *cput0))
        if (method == "adc(3)"):
            t1_2 = self.t1[0]
            eris_ovoo = eris.ovoo
            eris_ovvo = eris.ovvo
            eris_oovv = eris.oovv
            eris_oooo = eris.oooo

            for ki in range(nkpts):
                kj = ki
                for kl in range(nkpts):
                    kd = kconserv[kj, ki, kl]
                    M_ij[ki] += 2. * einsum('ld,ldji->ij', t1_2[kl],
                                                eris_ovoo[kl, kd,
                                                    kj])
                    M_ij[ki] -= einsum('ld,jdli->ij', t1_2[kl],
                                           eris_ovoo[kj, kd,
                                               kl])

                    kd = kconserv[ki, kj, kl]
                    M_ij[ki] += 2. * einsum('ld,ldij->ij', t1_2[kl].conj(),
                                                eris_ovoo[kl, kd,
                                                    ki].conj())
                    M_ij[ki] -= einsum('ld,idlj->ij', t1_2[kl].conj(),
                                           eris_ovoo[ki, kd,
                                               kl].conj())

                    for kd in range(nkpts):
                        t2_1 = self.t2[0]

                        ke = kconserv[kj, kd, kl]
                        t2_2_ild = self.t2[1][ki, kl, kd]
                        M_ij[ki] += 0.5 * 0.5 * \
                            einsum('ilde,jdle->ij', t2_2_ild, eris_ovov[kj, kd,
                                kl])
                        M_ij[ki] -= 0.5 * 0.5 * \
                            einsum('ilde,jeld->ij', t2_2_ild, eris_ovov[kj, ke,
                                kl])
                        M_ij[ki] += 0.5 * \
                            einsum('ilde,jdle->ij', t2_2_ild, eris_ovov[kj, kd,
                                kl])
                        del t2_2_ild

                        t2_2_lid = self.t2[1][kl, ki, kd]
                        M_ij[ki] -= 0.5 * 0.5 * \
                            einsum('lide,jdle->ij', t2_2_lid, eris_ovov[kj, kd,
                                kl])
                        M_ij[ki] += 0.5 * 0.5 * \
                            einsum('lide,jeld->ij', t2_2_lid, eris_ovov[kj, ke,
                                kl])

                        t2_2_jld = self.t2[1][kj, kl, kd]
                        M_ij[ki] += 0.5 * 0.5 * \
                            einsum('jlde,leid->ij', t2_2_jld.conj(),
                                       eris_ovov[kl, ke,
                                           ki].conj())
                        M_ij[ki] -= 0.5 * 0.5 * \
                            einsum('jlde,ield->ij', t2_2_jld.conj(),
                                       eris_ovov[ki, ke,
                                           kl].conj())
                        M_ij[ki] += 0.5 * einsum('jlde,leid->ij',
                            t2_2_jld.conj(), eris_ovov[kl, ke, ki].conj())

                        t2_2_ljd = self.t2[1][kl, kj, kd]
                        M_ij[ki] -= 0.5 * 0.5 * \
                            einsum('ljde,leid->ij', t2_2_ljd.conj(),
                                       eris_ovov[kl, ke,
                                           ki].conj())
                        M_ij[ki] += 0.5 * 0.5 * \
                            einsum('ljde,ield->ij', t2_2_ljd.conj(),
                                       eris_ovov[ki, ke,
                                           kl].conj())

                for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                    t2_1 = self.t2[0]

                    kl = kconserv[kd, km, ke]
                    kf = kconserv[kj, kd, kl]
                    temp_t2_v_1 = einsum(
                        'lmde,jldf->mejf', t2_1[kl, km, kd].conj(), t2_1[kj,
                            kl, kd])
                    M_ij[ki] -= 0.5 * 2 * einsum('mejf,ifem->ij',
                                                     temp_t2_v_1, eris_ovvo[ki,
                                                         kf,
                                                         ke])
                    M_ij[ki] -= 0.5 * 2 * einsum('meif,jfem->ij',
                        temp_t2_v_1.conj(), eris_ovvo[kj, kf, ke].conj())
                    M_ij[ki] += 0.5 * einsum('mejf,imef->ij',
                                                 temp_t2_v_1, eris_oovv[ki, km,
                                                     ke])
                    M_ij[ki] += 0.5 * einsum('meif,jmef->ij', temp_t2_v_1.conj(
                        ), eris_oovv[kj, km, ke].conj())

                    temp_t2_v_new = einsum(
                        'mlde,ljdf->mejf', t2_1[km, kl, kd].conj(), t2_1[kl,
                            kj, kd])
                    M_ij[ki] -= 0.5 * 2 * einsum('mejf,ifem->ij',
                        temp_t2_v_new, eris_ovvo[ki, kf, ke])
                    M_ij[ki] += 0.5 * einsum('mejf,imef->ij',
                                                 temp_t2_v_new, eris_oovv[ki,
                                                     km, ke])
                    M_ij[ki] -= 0.5 * 2 * einsum('meif,jfem->ij',
                        temp_t2_v_new.conj(), eris_ovvo[kj, kf, ke].conj())
                    M_ij[ki] += 0.5 * einsum('meif,jmef->ij',
                        temp_t2_v_new.conj(), eris_oovv[kj, km, ke].conj())
                    del temp_t2_v_new

                    temp_t2_v_2 = einsum(
                        'lmde,ljdf->mejf', t2_1[kl, km, kd].conj(), t2_1[kl,
                            kj, kd])
                    M_ij[ki] += 0.5 * 4 * einsum('mejf,ifem->ij',
                                                     temp_t2_v_2, eris_ovvo[ki,
                                                         kf,
                                                         ke])
                    M_ij[ki] += 0.5 * 4 * einsum('meif,jfem->ij',
                        temp_t2_v_2.conj(), eris_ovvo[kj, kf, ke].conj())
                    M_ij[ki] -= 0.5 * 2 * einsum('meif,jmef->ij',
                        temp_t2_v_2.conj(), eris_oovv[kj, km, ke].conj())
                    M_ij[ki] -= 0.5 * 2 * einsum('mejf,imef->ij',
                                                     temp_t2_v_2, eris_oovv[ki,
                                                         km,
                                                         ke])
                    del temp_t2_v_2

                    temp_t2_v_3 = einsum(
                        'mlde,jldf->mejf', t2_1[km, kl, kd].conj(), t2_1[kj,
                            kl, kd])
                    M_ij[ki] += 0.5 * einsum('mejf,ifem->ij',
                                                 temp_t2_v_3, eris_ovvo[ki, kf,
                                                     ke])
                    M_ij[ki] += 0.5 * einsum('meif,jfem->ij', temp_t2_v_3.conj(
                        ), eris_ovvo[kj, kf, ke].conj())
                    M_ij[ki] -= 0.5 * 2 * einsum('meif,jmef->ij',
                        temp_t2_v_3.conj(), eris_oovv[kj, km, ke].conj())
                    M_ij[ki] -= 0.5 * 2 * einsum('mejf,imef->ij',
                                                     temp_t2_v_3, eris_oovv[ki,
                                                         km,
                                                         ke])
                    del temp_t2_v_3

                for km, ke, kd in kpts_helper.loop_kkk(nkpts):

                    kl = kconserv[kd, km, ke]
                    kf = kconserv[kl, kd, km]
                    temp_t2_v_8 = einsum(
                        'lmdf,lmde->fe', t2_1[kl, km, kd], t2_1[kl, km,
                            kd].conj())
                    M_ij[ki] += 3.0 * einsum('fe,jief->ij', temp_t2_v_8,
                                                 eris_oovv[kj, ki,
                                                     ke])
                    M_ij[ki] -= 1.5 * einsum('fe,jfei->ij', temp_t2_v_8,
                                                 eris_ovvo[kj, kf,
                                                     ke])
                    M_ij[ki] += einsum('ef,jief->ij', temp_t2_v_8.T,
                                           eris_oovv[kj, ki,
                                               ke])
                    M_ij[ki] -= 0.5 * einsum('ef,jfei->ij', temp_t2_v_8.T,
                                                 eris_ovvo[kj, kf,
                                                     ke])
                    del temp_t2_v_8

                    temp_t2_v_9 = einsum(
                        'lmdf,mlde->fe', t2_1[kl, km, kd], t2_1[km, kl,
                            kd].conj())
                    M_ij[ki] -= 1.0 * einsum('fe,jief->ij', temp_t2_v_9,
                                                 eris_oovv[kj, ki,
                                                     ke])
                    M_ij[ki] -= 1.0 * einsum('ef,jief->ij', temp_t2_v_9.T,
                                                 eris_oovv[kj, ki,
                                                     ke])
                    M_ij[ki] += 0.5 * einsum('fe,jfei->ij', temp_t2_v_9,
                                                 eris_ovvo[kj, kf,
                                                     ke])
                    M_ij[ki] += 0.5 * einsum('ef,jfei->ij', temp_t2_v_9.T,
                                                 eris_ovvo[kj, kf,
                                                     ke])
                    del temp_t2_v_9

                    kl = kconserv[kd, km, ke]
                    kn = kconserv[kd, kl, ke]
                    temp_t2_v_10 = einsum(
                        'lnde,lmde->nm', t2_1[kl, kn, kd], t2_1[kl, km,
                            kd].conj())
                    M_ij[ki] -= 3.0 * einsum('nm,jinm->ij', temp_t2_v_10,
                                                 eris_oooo[kj, ki,
                                                     kn])
                    M_ij[ki] -= 1.0 * einsum('mn,jinm->ij', temp_t2_v_10.T,
                                                 eris_oooo[kj, ki,
                                                     kn])
                    M_ij[ki] += 1.5 * einsum('nm,jmni->ij', temp_t2_v_10,
                                                 eris_oooo[kj, km,
                                                     kn])
                    M_ij[ki] += 0.5 * einsum('mn,jmni->ij', temp_t2_v_10.T,
                                                 eris_oooo[kj, km,
                                                     kn])
                    del temp_t2_v_10

                    temp_t2_v_11 = einsum(
                        'lnde,mlde->nm', t2_1[kl, kn, kd], t2_1[km, kl,
                            kd].conj())
                    M_ij[ki] += 1.0 * einsum('nm,jinm->ij', temp_t2_v_11,
                                                 eris_oooo[kj, ki,
                                                     kn])
                    M_ij[ki] -= 0.5 * einsum('nm,jmni->ij', temp_t2_v_11,
                                                 eris_oooo[kj, km,
                                                     kn])
                    M_ij[ki] -= 0.5 * einsum('mn,jmni->ij', temp_t2_v_11.T,
                                                 eris_oooo[kj, km,
                                                     kn])
                    M_ij[ki] += 1.0 * einsum('mn,jinm->ij', temp_t2_v_11.T,
                                                 eris_oooo[kj, ki,
                                                     kn])
                    del temp_t2_v_11

                for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                    t2_1 = self.t2[0]
                    kl = kconserv[kd, km, ke]
                    kn = kconserv[kd, ki, ke]
                    temp_t2_v_12 = einsum(
                        'inde,lmde->inlm', t2_1[ki, kn, kd].conj(), t2_1[kl,
                            km, kd])
                    M_ij[ki] += 0.5 * 1.25 * \
                        einsum('inlm,jlnm->ij', temp_t2_v_12,
                                   eris_oooo[kj, kl, kn].conj())
                    M_ij[ki] -= 0.5 * 0.25 * \
                        einsum('inlm,jmnl->ij', temp_t2_v_12,
                                   eris_oooo[kj, km, kn].conj())

                    M_ij[ki] += 0.5 * 1.25 * \
                        einsum('jnlm,ilnm->ij', temp_t2_v_12.conj(),
                                   eris_oooo[ki, kl, kn])
                    M_ij[ki] -= 0.5 * 0.25 * \
                        einsum('jnlm,imnl->ij', temp_t2_v_12.conj(),
                                   eris_oooo[ki, km, kn])
                    del temp_t2_v_12

                    temp_t2_v_12_1 = einsum(
                        'nide,mlde->inlm', t2_1[kn, ki, kd].conj(), t2_1[km,
                            kl, kd])
                    M_ij[ki] += 0.5 * 0.25 * \
                        einsum('inlm,jlnm->ij', temp_t2_v_12_1,
                                   eris_oooo[kj, kl, kn].conj())
                    M_ij[ki] -= 0.5 * 0.25 * \
                        einsum('inlm,jmnl->ij', temp_t2_v_12_1,
                                   eris_oooo[kj, km, kn].conj())
                    M_ij[ki] += 0.5 * 0.25 * \
                        einsum('jnlm,ilnm->ij', temp_t2_v_12_1.conj(),
                                   eris_oooo[ki, kl, kn])
                    M_ij[ki] -= 0.5 * 0.25 * \
                        einsum('jnlm,imnl->ij', temp_t2_v_12_1.conj(),
                                   eris_oooo[ki, km, kn])

                    temp_t2_v_13 = einsum(
                        'inde,mlde->inml', t2_1[ki, kn, kd].conj(), t2_1[km,
                            kl, kd])
                    M_ij[ki] -= 0.5 * 0.5 * \
                        einsum('inml,jlnm->ij', temp_t2_v_13,
                                   eris_oooo[kj, kl, kn].conj())
                    M_ij[ki] += 0.5 * 0.25 * \
                        einsum('inml,jmnl->ij', temp_t2_v_13,
                                   eris_oooo[kj, km, kn].conj())

                    M_ij[ki] -= 0.5 * 0.5 * \
                        einsum('jnml,ilnm->ij', temp_t2_v_13.conj(),
                                   eris_oooo[ki, kl, kn])
                    M_ij[ki] += 0.5 * 0.25 * \
                        einsum('jnml,imnl->ij', temp_t2_v_13.conj(),
                                   eris_oooo[ki, km, kn])
                    del temp_t2_v_13

                    temp_t2_v_13_1 = einsum(
                        'nide,lmde->inml', t2_1[kn, ki, kd].conj(), t2_1[kl,
                            km, kd])
                    M_ij[ki] += 0.5 * 0.25 * \
                        einsum('inml,jmnl->ij', temp_t2_v_13_1,
                                   eris_oooo[kj, km, kn].conj())
                    M_ij[ki] += 0.5 * 0.25 * \
                        einsum('jnml,imnl->ij', temp_t2_v_13_1.conj(),
                                   eris_oooo[ki, km, kn])
                    del temp_t2_v_13_1

        return M_ij

    def get_diag(self, kshift, M_ij=None, eris=None):
        log = logger.Logger(self.stdout, self.verbose)

        self.method = check_implemented(self.method)

        if M_ij is None:
            M_ij = self.get_imds()

        nkpts = self.nkpts
        kconserv = self.khelper.kconserv
        nocc = self.nocc
        n_singles = nocc
        nvir = self.nvir
        n_doubles = nkpts * nkpts * nvir * nocc * nocc

        dim = n_singles + n_doubles

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)
        nocc = self.nocc

        diag = np.zeros((dim), dtype=np.complex128)
        doubles = np.zeros((nkpts, nkpts, nvir * nocc * nocc),
            dtype=np.complex128)

        # Compute precond in h1-h1 block
        M_ij_diag = np.diagonal(M_ij[kshift])

        # Compute precond in 2p1h-2p1h block
        for ka in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kshift, ki, ka]
                d_ij = e_occ[ki][:, None] + e_occ[kj]
                d_a = e_vir[ka][:nvir, None]
                D_n = -d_a + d_ij.reshape(-1)
                doubles[ka, ki] += D_n.reshape(-1)

        if (self.frozen is not None or not np.all([x.shape[
            1] == self.nmo for x in self.mo_coeff])):
            doubles = doubles.reshape(nkpts, nkpts, nvir, nocc, nocc)
            (M_ij_diag, doubles) = self.mask_frozen_ip(M_ij_diag, doubles,
                kshift)

        diag[s1:f1] = M_ij_diag.copy()
        diag[s2:f2] = doubles.reshape(-1)
        diag = -diag

        log.timer_debug1("Completed ip_diag calculation")

        return diag

    def matvec(self, kshift, M_ij=None, eris=None):
        self.method = check_implemented(self.method)

        method = self.method

        nkpts = self.nkpts
        nocc = self.nocc
        kconserv = self.khelper.kconserv
        n_singles = nocc
        nvir = self.nvir
        n_doubles = nkpts * nkpts * nvir * nocc * nocc

        s_singles = 0
        f_singles = n_singles
        s_doubles = f_singles
        f_doubles = s_doubles + n_doubles

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)

        if M_ij is None:
            M_ij = self.get_imds()

        # Calculate sigma vector
        def sigma_(r):
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles]
            r2 = r[s_doubles:f_doubles]

            r2 = r2.reshape(nkpts, nkpts, nvir, nocc, nocc)
            s2 = np.zeros((nkpts, nkpts, nvir, nocc, nocc),
                dtype=np.complex128)
            cell = self.cell
            kpts = self.kpts
            madelung = tools.madelung(cell, kpts)

            eris_ovoo = eris.ovoo

    ############ ADC(2) ij block ############################

            s1 = einsum('ij,j->i', M_ij[kshift], r1)

    ########### ADC(2) i - kja block #########################
            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    ki = kconserv[kj, kk, ka]

                    s1 += 2. * einsum('jaki,ajk->i',
                                          eris_ovoo[kj, ka, kk].conj(), r2[ka,
                                              kj])
                    s1 -= einsum('kaji,ajk->i',
                                     eris_ovoo[kk, ka, kj].conj(), r2[ka,
                                         kj])
    #################### ADC(2) ajk - i block ############################

                    s2[ka, kj] += einsum('jaki,i->ajk', eris_ovoo[kj, ka, kk],
                        r1)

    ################# ADC(2) ajk - bil block ############################

                    s2[ka, kj] -= einsum('a,ajk->ajk', e_vir[ka], r2[ka, kj])
                    s2[ka, kj] += einsum('j,ajk->ajk', e_occ[kj], r2[ka, kj])
                    s2[ka, kj] += einsum('k,ajk->ajk', e_occ[kk], r2[ka, kj])

    ############### ADC(3) ajk - bil block ############################

            if (method == "adc(2)-x" or method == "adc(3)"):

                eris_oooo = eris.oooo
                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kk, kshift, kj]
                        for kl in range(nkpts):
                            ki = kconserv[kj, kl, kk]

                            s2[ka, kj] -= 0.5 * einsum('kijl,ali->ajk',
                                eris_oooo[kk, ki, kj], r2[ka, kl])
                            s2[ka, kj] -= 0.5 * einsum('klji,ail->ajk',
                                eris_oooo[kk, kl, kj], r2[ka, ki])

                        for kl in range(nkpts):
                            kb = kconserv[ka, kk, kl]
                            s2[ka, kj] += 0.5 * einsum('klba,bjl->ajk',
                                eris_oovv[kk, kl, kb], r2[kb, kj])

                            kb = kconserv[kl, kj, ka]
                            s2[ka, kj] += 0.5 * einsum('jabl,bkl->ajk',
                                eris_ovvo[kj, ka, kb], r2[kb, kk])
                            s2[ka, kj] -= einsum('jabl,blk->ajk',
                                                     eris_ovvo[kj, ka,
                                                         kb], r2[kb,
                                                         kl])
                            kb = kconserv[ka, kj, kl]
                            s2[ka, kj] += 0.5 * einsum('jlba,blk->ajk',
                                eris_oovv[kj, kl, kb], r2[kb, kl])

                        for ki in range(nkpts):
                            kb = kconserv[ka, kk, ki]
                            s2[ka, kj] += 0.5 * einsum('kiba,bji->ajk',
                                eris_oovv[kk, ki, kb], r2[kb, kj])

                            kb = kconserv[ka, kj, ki]
                            s2[ka, kj] += 0.5 * einsum('jiba,bik->ajk',
                                eris_oovv[kj, ki, kb], r2[kb, ki])
                            s2[ka, kj] -= einsum('jabi,bik->ajk', eris_ovvo[kj,
                                ka, kb], r2[kb, ki])
                            kb = kconserv[ki, kj, ka]
                            s2[ka, kj] += 0.5 * einsum('jabi,bki->ajk',
                                eris_ovvo[kj, ka, kb], r2[kb, kk])

                if self.exxdiv is not None:
                    s2 += madelung * r2

            if (method == "adc(3)"):

                eris_ovoo = eris.ovoo

    # ################ ADC(3) i - kja block and ajk - i
    # ############################

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kj, kshift, kk]

                        for kb in range(nkpts):
                            kc = kconserv[kj, kb, kk]
                            t2_1 = self.t2[0]
                            temp_1 = einsum(
                                'jkbc,ajk->abc', t2_1[kj, kk, kb], r2[ka,
                                    kj])
                            temp = 0.25 * einsum('jkbc,ajk->abc',
                                                     t2_1[kj, kk, kb], r2[ka,
                                                         kj])
                            temp -= 0.25 * einsum('jkbc,akj->abc',
                                                      t2_1[kj, kk, kb], r2[ka,
                                                          kk])
                            temp -= 0.25 * einsum('kjbc,ajk->abc',
                                                      t2_1[kk, kj, kb], r2[ka,
                                                          kj])
                            temp += 0.25 * einsum('kjbc,akj->abc',
                                                      t2_1[kk, kj, kb], r2[ka,
                                                          kk])
                            ki = kconserv[kc, ka, kb]
                            if eris.ovvv is None:
                                chunk_size = self.chunk_size
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[ki, kc],
                                        eris.Lvv[ka, kb],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    k = eris_ovvv.shape[0]
                                    s1[a:a + k] += einsum('abc,icab->i',
                                        temp_1, eris_ovvv)
                                    s1[a:a + k] += einsum('abc,icab->i', temp,
                                        eris_ovvv)
                                    del eris_ovvv
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[ki, kb],
                                        eris.Lvv[ka, kc],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    s1[a:a + k] -= einsum('abc,ibac->i', temp,
                                        eris_ovvv)
                                    del eris_ovvv
                                    a += k
                            else:
                                eris_ovvv = eris.ovvv[:]
                                s1 += einsum('abc,icab->i', temp_1,
                                                 eris_ovvv[ki, kc,
                                                     ka])
                                s1 += einsum('abc,icab->i', temp,
                                                 eris_ovvv[ki, kc,
                                                     ka])
                                s1 -= einsum('abc,ibac->i', temp,
                                                 eris_ovvv[ki, kb,
                                                     ka])
                                del eris_ovvv
                del temp
                del temp_1

                t2_1 = self.t2[0]

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kj, kshift, kk]
                        for kc in range(nkpts):
                            kb = kconserv[kj, kc, kk]
                            ki = kconserv[kb, ka, kc]
                            if eris.ovvv is None:
                                chunk_size = self.chunk_size
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[ki, kc],
                                        eris.Lvv[ka, kb],
                                        p,
                                        chunk_size,
                                    ).reshape(
                                        -1,
                                        nvir,
                                        nvir,
                                        nvir,
                                    ) / nkpts
                                    k = eris_ovvv.shape[0]
                                    temp = (einsum('i,icab->cba', r1[a:a + k],
                                        eris_ovvv.conj()))
                                    del eris_ovvv
                                    a += k
                            else:
                                eris_ovvv = eris.ovvv[:]
                                temp = einsum(
                                    'i,icab->cba', r1, eris_ovvv[ki, kc,
                                        ka].conj())
                                del eris_ovvv
                            s2[ka, kj] += einsum('cba,jkbc->ajk', temp, t2_1[
                                kj, kk, kb].conj())
                del temp

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kj, kshift, kk]
                        for kb in range(nkpts):
                            kl = kconserv[ka, kj, kb]

                            t2_1 = self.t2[0]
                            temp = einsum('ljba,ajk->blk', t2_1[kl, kj, kb],
                                r2[ka, kj])
                            temp_2 = einsum('jlba,akj->blk', t2_1[kj, kl, kb],
                                r2[ka, kk])
                            del t2_1

                            t2_1_jla = self.t2[0][kj, kl, ka]
                            temp += einsum('jlab,ajk->blk', t2_1_jla, r2[ka,
                                kj])
                            temp -= einsum('jlab,akj->blk', t2_1_jla, r2[ka,
                                kk])

                            temp_1 = einsum('jlab,ajk->blk', t2_1_jla, r2[ka,
                                kj])
                            temp_1 -= einsum('jlab,akj->blk', t2_1_jla, r2[ka,
                                kk])
                            temp_1 += einsum('jlab,ajk->blk', t2_1_jla, r2[ka,
                                kj])
                            del t2_1_jla

                            t2_1_lja = self.t2[0][kl, kj, ka]
                            temp -= einsum('ljab,ajk->blk', t2_1_lja, r2[ka,
                                kj])
                            temp += einsum('ljab,akj->blk', t2_1_lja, r2[ka,
                                kk])

                            temp_1 -= einsum('ljab,ajk->blk', t2_1_lja, r2[ka,
                                kj])
                            del t2_1_lja

                            ki = kconserv[kk, kl, kb]
                            s1 += 0.5 * einsum('blk,lbik->i', temp,
                                eris_ovoo[kl, kb, ki])
                            s1 -= 0.5 * einsum('blk,iblk->i', temp,
                                eris_ovoo[ki, kb, kl])
                            s1 += 0.5 * einsum('blk,lbik->i', temp_1,
                                eris_ovoo[kl, kb, ki])
                            s1 -= 0.5 * einsum('blk,iblk->i', temp_2,
                                eris_ovoo[ki, kb, kl])
                            del temp
                            del temp_1
                            del temp_2

                        for kb in range(nkpts):
                            kl = kconserv[ka, kk, kb]

                            t2_1 = self.t2[0]

                            temp = -einsum('lkba,akj->blj', t2_1[kl, kk, kb],
                                r2[ka, kk])
                            temp_2 = -einsum('klba,ajk->blj', t2_1[kk, kl, kb],
                                r2[ka, kj])
                            del t2_1

                            t2_1_kla = self.t2[0][kk, kl, ka]
                            temp -= einsum('klab,akj->blj', t2_1_kla, r2[ka,
                                kk])
                            temp += einsum('klab,ajk->blj', t2_1_kla, r2[ka,
                                kj])
                            temp_1 = -2.0 * einsum('klab,akj->blj',
                                                       t2_1_kla, r2[ka,
                                                           kk])
                            temp_1 += einsum('klab,ajk->blj', t2_1_kla, r2[ka,
                                kj])
                            del t2_1_kla

                            t2_1_lka = self.t2[0][kl, kk, ka]
                            temp += einsum('lkab,akj->blj', t2_1_lka, r2[ka,
                                kk])
                            temp -= einsum('lkab,ajk->blj', t2_1_lka, r2[ka,
                                kj])
                            temp_1 += einsum('lkab,akj->blj', t2_1_lka, r2[ka,
                                kk])
                            del t2_1_lka

                            ki = kconserv[kj, kl, kb]
                            s1 -= 0.5 * einsum('blj,lbij->i', temp,
                                eris_ovoo[kl, kb, ki])
                            s1 += 0.5 * einsum('blj,iblj->i', temp,
                                eris_ovoo[ki, kb, kl])
                            s1 -= 0.5 * einsum('blj,lbij->i', temp_1,
                                eris_ovoo[kl, kb, ki])
                            s1 += 0.5 * einsum('blj,iblj->i', temp_2,
                                eris_ovoo[ki, kb, kl])

                            del temp
                            del temp_1
                            del temp_2

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kk, kshift, kj]
                        for kl in range(nkpts):
                            kb = kconserv[kj, ka, kl]
                            ki = kconserv[kk, kl, kb]
                            temp_1 = einsum(
                                'i,lbik->kbl', r1, eris_ovoo[kl, kb,
                                    ki].conj())
                            temp = einsum(
                                'i,lbik->kbl', r1, eris_ovoo[kl, kb,
                                    ki].conj())
                            temp -= einsum('i,iblk->kbl', r1,
                                               eris_ovoo[ki, kb,
                                                   kl].conj())

                            t2_1 = self.t2[0]
                            s2[ka, kj] += einsum('kbl,ljba->ajk', temp, t2_1[
                                kl, kj, kb].conj())
                            s2[ka, kj] += einsum('kbl,jlab->ajk', temp_1, t2_1[
                                kj, kl, ka].conj())
                            s2[ka, kj] -= einsum('kbl,ljab->ajk', temp_1, t2_1[
                                kl, kj, ka].conj())

                            kb = kconserv[kk, ka, kl]
                            ki = kconserv[kj, kl, kb]
                            temp_2 = -einsum('i,iblj->jbl', r1,
                                                 eris_ovoo[ki, kb,
                                                     kl].conj())
                            s2[ka, kj] += einsum('jbl,klba->ajk', temp_2, t2_1[
                                kk, kl, kb].conj())
                            del t2_1
            s2 = s2.reshape(-1)
            s = np.hstack((s1, s2))
            del s1
            del s2
            cput0 = log.timer_debug1("completed sigma vector calculation",
                *cput0)
            s *= -1.0

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

        idn_occ = np.identity(nocc)

        T1 = np.zeros((nocc), dtype=np.complex128)
        T2 = np.zeros((nkpts, nkpts, nvir, nocc, nocc), dtype=np.complex128)

    ######## ADC(2) 1h part  ############################################

        if orb < nocc:
            T1 += idn_occ[orb, :]
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = self.khelper.kconserv[kk, kc, kshift]
                    ki = self.khelper.kconserv[kc, kk, kd]
                    T1 += 0.25 * einsum('kdc,ikdc->i', t2_1[kk, kshift, kd]
                                            [:, orb, :, :].conj(), t2_1[ki, kk,
                                                kd])
                    T1 -= 0.25 * einsum('kcd,ikdc->i', t2_1[kk, kshift, kc]
                                            [:, orb, :, :].conj(), t2_1[ki, kk,
                                                kd])
                    T1 -= 0.25 * einsum('kdc,ikcd->i', t2_1[kk, kshift, kd]
                                            [:, orb, :, :].conj(), t2_1[ki, kk,
                                                kc])
                    T1 += 0.25 * einsum('kcd,ikcd->i', t2_1[kk, kshift, kc]
                                            [:, orb, :, :].conj(), t2_1[ki, kk,
                                                kc])
                    T1 -= 0.25 * einsum('kdc,ikdc->i', t2_1[kshift, kk, kd]
                                            [orb, :, :, :].conj(), t2_1[ki, kk,
                                                kd])
                    T1 -= 0.25 * einsum('kcd,ikcd->i', t2_1[kshift, kk, kc]
                                            [orb, :, :, :].conj(), t2_1[ki, kk,
                                                kc])
        else:
            if (self.approx_trans_moments is False or self.method == "adc(3)"):
                t1_2 = self.t1[0]
                T1 += t1_2[kshift][:, (orb - nocc)]

    ######## ADC(2) 2h-1p  part  ############################################

            for ki in range(nkpts):
                for kj in range(nkpts):
                    ka = self.khelper.kconserv[kj, kshift, ki]

                    t2_1_t = -t2_1[ki, kj, ka].transpose(2, 3, 1, 0)
                    T2[ka, kj] += t2_1_t[:, (orb - nocc), :, :].conj()

            del t2_1_t
    ####### ADC(3) 2h-1p  part  ############################################

        if (self.method == 'adc(2)-x' and self.approx_trans_moments is False
            or self.method == 'adc(3)'):
            t2_2 = self.t2[1]

            if orb >= nocc:
                for ki in range(nkpts):
                    for kj in range(nkpts):
                        ka = self.khelper.kconserv[kj, kshift, ki]

                        t2_2_t = -t2_2[ki, kj, ka].transpose(2, 3, 1, 0)
                        T2[ka, kj] += t2_2_t[:, (orb - nocc), :, :].conj()

    ######### ADC(3) 1h part  ############################################

        if (method == 'adc(3)'):
            if orb < nocc:
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        kd = self.khelper.kconserv[kk, kc, kshift]
                        ki = self.khelper.kconserv[kd, kk, kc]
                        T1 += 0.25 * einsum('kdc,ikdc->i', t2_1[kk, ki, kd][:,
                            orb, :, :].conj(), t2_2[ki, kk, kd])
                        T1 -= 0.25 * einsum('kcd,ikdc->i', t2_1[kk, ki, kc][:,
                            orb, :, :].conj(), t2_2[ki, kk, kd])
                        T1 -= 0.25 * einsum('kdc,ikcd->i', t2_1[kk, ki, kd][:,
                            orb, :, :].conj(), t2_2[ki, kk, kc])
                        T1 += 0.25 * einsum('kcd,ikcd->i', t2_1[kk, ki, kc][:,
                            orb, :, :].conj(), t2_2[ki, kk, kc])
                        T1 -= 0.25 * einsum('kdc,ikdc->i', t2_1[ki, kk, kd][
                            orb, :, :, :].conj(), t2_2[ki, kk, kd])
                        T1 -= 0.25 * einsum('kcd,ikcd->i', t2_1[ki, kk, kc][
                            orb, :, :, :].conj(), t2_2[ki, kk, kc])

                        T1 += 0.25 * \
                            einsum('ikdc,kdc->i', t2_1[ki, kk, kd],
                                       t2_2[kk, ki, kd][:, orb, :,
                                           :].conj())
                        T1 -= 0.25 * \
                            einsum('ikcd,kdc->i', t2_1[ki, kk, kc],
                                       t2_2[kk, ki, kd][:, orb, :,
                                           :].conj())
                        T1 -= 0.25 * \
                            einsum('ikdc,kcd->i', t2_1[ki, kk, kd],
                                       t2_2[kk, ki, kc][:, orb, :,
                                           :].conj())
                        T1 += 0.25 * \
                            einsum('ikcd,kcd->i', t2_1[ki, kk, kc],
                                       t2_2[kk, ki, kc][:, orb, :,
                                           :].conj())
                        T1 -= 0.25 * \
                            einsum('ikcd,kcd->i', t2_1[ki, kk, kc],
                                       t2_2[ki, kk, kc][orb, :, :,
                                           :].conj())
                        T1 -= 0.25 * \
                            einsum('ikdc,kdc->i', t2_1[ki, kk, kd],
                                       t2_2[ki, kk, kd][orb, :, :,
                                           :].conj())
            else:

                for kk in range(nkpts):
                    for kc in range(nkpts):
                        ki = self.khelper.kconserv[kshift, kk, kc]

                        T1 += 0.5 * einsum('kic,kc->i', t2_1[kk, ki, kc][:, :,
                            :, orb - nocc], t1_2[kk])
                        T1 -= 0.5 * einsum('ikc,kc->i', t2_1[ki, kk, kc][:, :,
                            :, orb - nocc], t1_2[kk])
                        T1 += 0.5 * einsum('kic,kc->i', t2_1[kk, ki, kc][:, :,
                            :, orb - nocc], t1_2[kk])

            del t2_2
        del t2_1

        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = self.khelper.kconserv[kj, kshift, ki]
                T2[ka, kj] += T2[ka, kj] - T2[ka, ki].transpose(0, 2, 1)

        T2 = T2.reshape(-1)
        T = np.hstack((T1, T2))

        return T

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        nkpts = self.nkpts
        nocc = self.t2[0].shape[3]
        n_singles = nocc
        nvir = self.nvir

        for I in range(U.shape[1]):
            U1 = U[:n_singles, I]
            U2 = U[n_singles:, I].reshape(nkpts, nkpts, nvir, nocc, nocc)
            UdotU = np.dot(U1.conj().ravel(), U1.ravel())
            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = self.khelper.kconserv[kj, kshift, kk]
                    UdotU += 2.0 * np.dot(U2[ka, kj].conj().ravel(), U2[ka,
                        kj].ravel()) - np.dot(U2[ka, kj].conj().ravel(), U2[ka,
                        kk].transpose(0, 2, 1).ravel())
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

        n_singles = nocc
        n_doubles = nkpts * nkpts * nvir * nocc * nocc

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

        L2 = L2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        R2 = R2.reshape(nkpts, nkpts, nvir, nocc, nocc)

    # block- ij
        # 000
        for ki in range(nkpts):
            rdm1[ki][:nocc, :nocc] += 2 * einsum('ij,m,m->ij', kd_oc, L1, R1)
        rdm1[kshift][:nocc, :nocc] -= einsum('i,j->ij', L1, R1)
        # 101
        for ki in range(nkpts):
            for ka in range(nkpts):
                kj = kconserv[ka, ki, kshift]
                for kl in range(nkpts):
                    rdm1[kl][:nocc, :nocc] += 4 * einsum('aij,aij,IJ->IJ',
                        L2[ka][ki], R2[ka][ki], kd_oc)
                    rdm1[kl][:nocc, :nocc] -= 2 * einsum('aij,aji,IJ->IJ',
                        L2[ka][ki], R2[ka][kj], kd_oc)

                rdm1[kj][:nocc, :nocc] -= 2 * einsum('aJi,aIi->IJ', L2[ka][kj],
                    R2[ka][kj])
                rdm1[kj][:nocc, :nocc] += einsum('aJi,aiI->IJ', L2[ka][kj],
                    R2[ka][ki])
                rdm1[kj][:nocc, :nocc] += einsum('aiJ,aIi->IJ', L2[ka][ki],
                    R2[ka][kj])
                rdm1[kj][:nocc, :nocc] -= 2 * einsum('aiJ,aiI->IJ', L2[ka][ki],
                    R2[ka][ki])
        # 020
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                ki = kshift
                rdm1[ki][:nocc, :nocc] += einsum('J,i,Ijab,ijab->IJ', L1, R1,
                    t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ki][:nocc, :nocc] -= 1 / 2 * einsum('J,i,Ijab,ijba->IJ',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())
                rdm1[ki][:nocc, :nocc] += einsum('i,I,ijab,Jjab->IJ', L1, R1,
                    t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ki][:nocc, :nocc] -= 1 / 2 * einsum('i,I,ijab,Jjba->IJ',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())

                rdm1[kj][:nocc, :nocc] += 2 * einsum('i,j,Iiab,Jjab->IJ', L1,
                    R1, t1_oovv[kj][ki][ka], t1_oovv[kj][ki][ka].conj())
                rdm1[kj][:nocc, :nocc] -= einsum('i,j,Iiab,Jjba->IJ', L1, R1,
                    t1_oovv[kj][ki][ka], t1_oovv[kj][ki][kb].conj())
                for ki in range(nkpts):
                    kb = kconserv[ki, ka, kj]
                    rdm1[ki][:nocc, :nocc] -= 4 * einsum('i,i,Ijab,Jjab->IJ',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj(
                        ))
                    rdm1[ki][:nocc, :nocc] += 2 * einsum('i,i,Ijab,Jjba->IJ',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj(
                        ))
    # block- ab
        # 101
        for ki in range(nkpts):
            for ka in range(nkpts):
                kj = kconserv[ka, ki, kshift]
                rdm1[ka][nocc:, nocc:] += 2 * einsum('Aij,Bij->AB', L2[ka][ki],
                    R2[ka][ki])
                rdm1[ka][nocc:, nocc:] -= einsum('Aij,Bji->AB', L2[ka][ki],
                    R2[ka][kj])

        # 020
        for kk in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kk]
                ki = kshift
                rdm1[ka][nocc:, nocc:] -= 2 * einsum('i,j,ikBa,jkAa->AB', L1,
                    R1, t1_oovv[ki][kk][ka], t1_oovv[ki][kk][ka].conj())
                rdm1[ka][nocc:, nocc:] += einsum('i,j,ikBa,jkaA->AB', L1, R1,
                    t1_oovv[ki][kk][ka], t1_oovv[ki][kk][kb].conj())
                rdm1[ka][nocc:, nocc:] += einsum('i,j,ikaB,jkAa->AB', L1, R1,
                    t1_oovv[ki][kk][kb], t1_oovv[ki][kk][ka].conj())
                rdm1[ka][nocc:, nocc:] -= 2 * einsum('i,j,ikaB,jkaA->AB', L1,
                    R1, t1_oovv[ki][kk][kb], t1_oovv[ki][kk][kb].conj())

                for kj in range(nkpts):
                    kb = kconserv[kj, ka, kk]
                    rdm1[ka][nocc:, nocc:] += 4 * einsum('i,i,jkBa,jkAa->AB',
                        L1, R1, t1_oovv[kj][kk][ka], t1_oovv[kj][kk][ka].conj(
                        ))
                    rdm1[ka][nocc:, nocc:] -= 2 * einsum('i,i,jkBa,jkaA->AB',
                        L1, R1, t1_oovv[kj][kk][ka], t1_oovv[kj][kk][kb].conj(
                        ))

    # block- ia
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kshift
                ka = kconserv[ki, kb, kj]
                rdm1[kb][:nocc, nocc:] += einsum('aij,I,ijAa->IA', L2[ka][ki],
                    R1, t1_oovv[ki][kj][kb])
                rdm1[kb][:nocc, nocc:] -= 2 * einsum('aij,I,ijaA->IA',
                    L2[ka][ki], R1, t1_oovv[ki][kj][ka])
        for ka in range(nkpts):
            ki = kshift
            kj = kconserv[ka, ki, kshift]
            for kl in range(nkpts):
                km = kl
                rdm1[kl][:nocc, nocc:] -= 2 * einsum('aij,i,IjAa->IA',
                    L2[ka][ki], R1, t1_oovv[kl][kj][km])
                rdm1[kl][:nocc, nocc:] += 1 * einsum('aij,i,IjaA->IA',
                    L2[ka][ki], R1, t1_oovv[kl][kj][ka])
                rdm1[kl][:nocc, nocc:] += 4 * einsum('aij,j,IiAa->IA',
                    L2[ka][kj], R1, t1_oovv[kl][kj][km])
                rdm1[kl][:nocc, nocc:] -= 2 * einsum('aij,j,IiaA->IA',
                    L2[ka][kj], R1, t1_oovv[kl][kj][ka])
        for ka in range(nkpts):
            ki = kshift
            rdm1[ka][:nocc, nocc:] += 2 * einsum('i,AIi->IA', L1, R2[ka][ka])
            rdm1[ka][:nocc, nocc:] -= einsum('i,AiI->IA', L1, R2[ka][ki])

    # block- ai
        for ki in range(nkpts):
            rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

        if self.approx_trans_moments is False or self.method == "adc(3)":
            ki = kshift
            ### block- ia ###
            rdm1[ki][:nocc, nocc:] -= einsum('i,I,iA->IA', L1, R1, t2_ce[ki])
            for ki in range(nkpts):
                rdm1[ki][:nocc, nocc:] += 2 * einsum('i,i,IA->IA', L1, R1,
                    t2_ce[ki])
            ### block- ai ###
                rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

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

        t1_oovv_sjb = np.zeros_like(t1_oovv[0])
        t1_oovv_isb = np.zeros_like(t1_oovv[0])
        t1_oovv_sja = t1_oovv[kshift]
        t1_oovv_isa = t1_oovv[:, kshift, :, :, :, :]
        t1_oovv_ijs = t1_oovv[:, :, kshift, :, :, :]

        kj, ka = np.indices((nkpts, nkpts))
        kb = self.khelper.kconserv[kshift, ka, kj]
        t1_oovv_sjb[kj, ka] = t1_oovv_sja[kj, kb]
        t1_oovv_isb[kj, ka] = t1_oovv_isa[kj, kb]

        n_singles = nocc
        n_doubles = nkpts * nkpts * nvir * nocc * nocc
        occ_list = range(nocc)
        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        R2 = R2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        R2_aji = np.zeros_like(R2)
        L2_aji = np.zeros_like(L2)
        L2_jia = np.zeros_like(L2)

        ka, ki = np.indices((nkpts, nkpts))
        kj = kconserv_s[ki, ka]
        R2_aji[ka, ki] = R2[ka, kj]
        L2_aji[ka, ki] = L2[ka, kj]
        L2_asj = L2[:, kshift, :, :, :]

        kconserv_s = kconserv[:, kshift, :]
        kj, ki = np.indices((nkpts, nkpts))
        ka = kconserv_s[ki, kj]
        L2_jia[kj, ki] = L2[ka, ki]

    # block- ij
        # 000
        rdm1[:, occ_list, occ_list] += 2 * einsum('m,m->', L1, R1)
        rdm1[kshift, :nocc, :nocc] -= einsum('i,j->ij', L1, R1)
        # 101
        rdm1[:, occ_list, occ_list] += 4 * einsum('kKaij,kKaij->', L2, R2)
        rdm1[:, occ_list, occ_list] -= 2 * einsum('kKaij,kKaji->', L2, R2_aji)
        rdm1[:, :nocc, :nocc] -= 2 * einsum('kKaJi,kKaIi->KIJ', L2, R2)
        rdm1[:, :nocc, :nocc] += einsum('kKaJi,kKaiI->KIJ', L2, R2_aji)
        rdm1[:, :nocc, :nocc] += einsum('kKaiJ,kKaIi->KIJ', L2_aji, R2)
        rdm1[:, :nocc, :nocc] -= 2 * einsum('kKaiJ,kKaiI->KIJ', L2_aji, R2_aji)
        # 020
        rdm1[kshift, :nocc, :nocc] += einsum('J,i,kKIjab,kKijab->IJ', L1, R1,
            t1_oovv_sja, t1_oovv_sja.conj())
        rdm1[kshift, :nocc, :nocc] -= 1 / 2 * einsum('J,i,kKIjab,kKijba->IJ',
            L1, R1, t1_oovv_sja, t1_oovv_sjb.conj())
        rdm1[kshift, :nocc, :nocc] += einsum('i,I,kKijab,kKJjab->IJ', L1, R1,
            t1_oovv_sja, t1_oovv_sja.conj())
        rdm1[kshift, :nocc, :nocc] -= 1 / 2 * einsum('i,I,kKijab,kKJjba->IJ',
            L1, R1, t1_oovv_sja, t1_oovv_sjb.conj())
        rdm1[:, :nocc, :nocc] += 2 * einsum('i,j,KkIiab,KkJjab->KIJ', L1, R1,
            t1_oovv_isa, t1_oovv_isa.conj())
        rdm1[:, :nocc, :nocc] -= einsum('i,j,KkIiab,KkJjba->KIJ', L1, R1,
            t1_oovv_isa, t1_oovv_isb.conj())

    # block- ab
        # 101
        rdm1[:, nocc:, nocc:] += 2 * einsum('KkAij,KkBij->KAB', L2, R2)
        rdm1[:, nocc:, nocc:] -= einsum('KkAij,KkBji->KAB', L2, R2_aji)
        # 020
        rdm1[:, nocc:, nocc:] -= 2 * einsum('i,m,kKijBa,kKmjAa->KAB', L1, R1,
            t1_oovv_sja, t1_oovv_sja.conj())
        rdm1[:, nocc:, nocc:] += einsum('i,m,kKijBa,kKmjaA->KAB', L1, R1,
            t1_oovv_sja, t1_oovv_sjb.conj())
        rdm1[:, nocc:, nocc:] += einsum('i,m,kKijaB,kKmjAa->KAB', L1, R1,
            t1_oovv_sjb, t1_oovv_sja.conj())
        rdm1[:, nocc:, nocc:] -= 2 * einsum('i,m,kKijaB,kKmjaA->KAB', L1, R1,
            t1_oovv_sjb, t1_oovv_sjb.conj())

    # block- ia
        rdm1[kshift, :nocc, nocc:] += einsum('kKaij,I,KkijAa->IA', L2_jia, R1,
            t1_oovv_ijs)
        rdm1[:, :nocc, nocc:] += 2 * einsum('i,KKAIi->KIA', L1, R2)
        rdm1[:, :nocc, nocc:] -= einsum('i,KAiI->KIA', L1, R2[:, kshift, :, :,
            :])

        del (t1_oovv_sjb)
        del (t1_oovv_sja)
        del (t1_oovv_isa)
        del (t1_oovv_ijs)
        del (t1_oovv_isb)
        del (R2_aji)
        del (L2_aji)

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
            oo[ki] -= 4 * einsum('KkIjab,KkJjab->IJ', t1_oovv_np,
                t1_oovv_np.conj())
            oo[ki] += 2 * einsum('KkIjab,KkJjba->IJ', t1_oovv_np,
                t1_oovv_ijb.conj())
    # block- ab
            vv += 4 * einsum('kKijBa,kKijAa->KAB', t1_oovv_np,
                t1_oovv_np.conj())
            vv -= 2 * einsum('kKijBa,kKijaA->KAB', t1_oovv_np,
                t1_oovv_ijb.conj())
    # block- ia
            ov[ki] -= 2 * einsum('kaij,i,kIjAa->IA', L2_asj, R1, t1_oovv_np[:,
                ki])
            ov[ki] += 1 * einsum('kaij,i,kkIjaA->IA', L2_asj, R1, t1_oovv_np)
            ov[ki] += 4 * einsum('kkaij,j,kIiAa->IA', L2, R1, t1_oovv_np[:,
                ki])
            ov[ki] -= 2 * einsum('kkaij,j,kkIiaA->IA', L2, R1, t1_oovv_np)
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
        del (L2_asj)
    # block- ia
        rdm1[kshift, :nocc, nocc:] -= 2 * einsum('kKaij,I,KkijaA->IA', L2_jia,
            R1, t1_oovv_ijs_b)
        del (L2_jia)
        del (t1_oovv_ijs_b)

    # block- ai
        rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

        if self.approx_trans_moments is False or self.method == "adc(3)":
            rdm1[kshift, :nocc, nocc:] -= einsum('i,I,iA->IA', L1, R1,
                t2_ce[kshift])
            rdm1[:, :nocc, nocc:] += 2 * LR * t2_ce
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

    def mask_frozen_ip(self, v1, v2, kshift, const=-LARGE_DENOM):
        '''Replaces all frozen orbital indices of `vector` with the value
        `const`.'''
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self)

        new_v1 = const * np.ones_like(v1)
        new_v2 = const * np.ones_like(v2)

        new_v1[nonzero_opadding[kshift]] = v1[nonzero_opadding[kshift]]
        for ka in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kshift, ki, ka]
                idx = np.ix_([ka], [ki], nonzero_vpadding[ka],
                    nonzero_opadding[ki], nonzero_opadding[kj])
                new_v2[idx] = v2[idx]

        return (new_v1, new_v2)

    _mask_frozen = mask_frozen_ip


def _as_solver(adc):
    """Accept both excitation solvers and ground-state ADC drivers."""
    return adc if isinstance(adc, RADCIP) else RADCIP(adc)


def vector_size(adc):
    """Compatibility wrapper for :meth:`RADCIP.vector_size`."""
    return _as_solver(adc).vector_size()


def get_imds(adc, eris=None):
    """Compatibility wrapper for :meth:`RADCIP.get_imds`."""
    return _as_solver(adc).get_imds(eris)


def get_diag(adc, kshift, M_ij=None, eris=None):
    """Compatibility wrapper for :meth:`RADCIP.get_diag`."""
    return _as_solver(adc).get_diag(kshift, M_ij, eris)


def matvec(adc, kshift, M_ij=None, eris=None):
    """Compatibility wrapper for :meth:`RADCIP.matvec`."""
    return _as_solver(adc).matvec(kshift, M_ij, eris)


def get_trans_moments(adc, kshift):
    """Compatibility wrapper for :meth:`RADCIP.get_trans_moments`."""
    return _as_solver(adc).get_trans_moments(kshift)


def get_trans_moments_orbital(adc, orb, kshift):
    """Compatibility wrapper for :meth:`RADCIP.get_trans_moments_orbital`."""
    return _as_solver(adc).get_trans_moments_orbital(orb, kshift)


def renormalize_eigenvectors(adc, kshift, U, nroots=1):
    """Compatibility wrapper for :meth:`RADCIP.renormalize_eigenvectors`."""
    return _as_solver(adc).renormalize_eigenvectors(kshift, U, nroots)


def make_rdm1(adc, root=None, kptlist=None, K_idx=None, if_ss=False):
    """Compatibility wrapper for :meth:`RADCIP.make_rdm1`."""
    return _as_solver(adc).make_rdm1(root, kptlist, K_idx, if_ss)


def make_rdm1_eigenvectors_slow(adc, L, R, kshift):
    """Compatibility wrapper for :meth:`RADCIP.make_rdm1_eigenvectors_slow`."""
    return _as_solver(adc).make_rdm1_eigenvectors_slow(L, R, kshift)


def make_rdm1_eigenvectors(adc, L, R, kshift, if_ss):
    """Compatibility wrapper for :meth:`RADCIP.make_rdm1_eigenvectors`."""
    return _as_solver(adc).make_rdm1_eigenvectors(L, R, kshift, if_ss)


def get_properties(adc, kshift, U, nroots=1):
    """Compatibility wrapper for :meth:`RADCIP.get_properties`."""
    return _as_solver(adc).get_properties(kshift, U, nroots)


def mask_frozen_ip(adc, v1, v2, kshift, const=-LARGE_DENOM):
    """Compatibility wrapper for :meth:`RADCIP.mask_frozen_ip`."""
    return _as_solver(adc).mask_frozen_ip(v1, v2, kshift, const)
