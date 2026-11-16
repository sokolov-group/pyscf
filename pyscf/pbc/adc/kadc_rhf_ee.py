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
from itertools import product
import time

import numpy as np

from pyscf.lib import logger
from pyscf.lib.parameters import LOOSE_ZERO_TOL
from pyscf.pbc.adc import kadc_ee_terms, kadc_rhf
from pyscf.pbc.adc.lib import MomentumContractions, check_implemented, einsum
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import _add_padding


class RADCEE(kadc_rhf.KRADC):
    """Restricted periodic singlet EE-ADC(2), ADC(2)-x and ADC(3).

    Excitation vectors contain full singles and doubles blocks.
    Transition densities and density-response strengths use the same
    momentum sectors as the excitation vectors.
    """

    def __init__(self, adc):
        self._init_from_ground_state(adc)
        self.type_ = 'ee'
        self._ee_momentum = {}
        self._transition_cache = {}
        self.density_gv = np.asarray(adc.density_gv).copy()
        self.E = self.T = self.U = self.X = self.P = None
        self._keys = set(self.__dict__)

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        # Rebuild cached data if orbitals or the k mesh changed between runs.
        self._ee_momentum.clear()
        self._transition_cache.clear()
        return kadc_rhf.kernel(self, nroots, guess, eris, kptlist)

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def singles_shape(self):
        return (self.nkpts, self.nocc, self.nvir)

    @property
    def doubles_shape(self):
        return (self.nkpts,) * 3 + (self.nocc,) * 2 + (self.nvir,) * 2

    @property
    def n_singles(self):
        return int(np.prod(self.singles_shape))

    @property
    def n_doubles(self):
        return int(np.prod(self.doubles_shape))

    def vector_size(self):
        """Return the EE vector length (shared ADC solver interface)."""
        return self.n_singles + self.n_doubles

    def _momentum_maps(self, kshift):
        """Return singles, inverse singles and doubles momentum maps."""
        if kshift not in self._ee_momentum:
            ia = self.khelper.kconserv[:, kshift, 0].copy()
            ai = np.empty_like(ia)
            ai[ia] = np.arange(self.nkpts)
            lattice = self.cell.lattice_vectors() / (2 * np.pi)
            kpts = self.kpts
            momenta = (kpts[:, None, None, :] - kpts[:, None, :]
                       + kpts - kpts[kshift])
            doubles = np.full((self.nkpts,) * 3, -1, dtype=int)
            for index, kpt in enumerate(kpts):
                scaled = einsum('wx,klmx->wklm', lattice, momenta - kpt)
                mask = np.sum(abs(scaled - np.rint(scaled)), axis=0) < 1e-9
                doubles[mask] = index
            if np.any(doubles < 0):
                raise ValueError('K-point mesh is not closed under momentum '
                                 'conservation for this excitation shift.')
            self._ee_momentum[kshift] = ia, ai, doubles
        return self._ee_momentum[kshift]

    def _doubles_diagonal(self, kshift):
        """Build orbital energy differences for one momentum sector."""
        _, e_occ, e_vir = (self.padded_coeff,
            self.e_occ, self.e_vir)
        _, _, kconserv = self._momentum_maps(kshift)
        diagonal = np.empty(self.doubles_shape,
                            dtype=np.result_type(e_occ, e_vir))
        for ki, kj, ka in kpts_helper.loop_kkk(self.nkpts):
            kb = kconserv[ki, ka, kj]
            occupied = e_occ[ki][:, None] + e_occ[kj]
            virtual = e_vir[ka][:, None] + e_vir[kb]
            diagonal[ki, kj, ka] = (virtual[None, None, :, :]
                                     - occupied[:, :, None, None])
        return diagonal

    def _get_adc1(self, eris, kshift):
        """Build the zeroth- and first-order singles block."""
        coeff, e_occ, e_vir = (self.padded_coeff,
            self.e_occ, self.e_vir)
        ia, _, _ = self._momentum_maps(kshift)
        block = np.zeros((self.nkpts, self.nkpts, self.nocc, self.nvir,
                          self.nocc, self.nvir), dtype=coeff.dtype)
        occ_identity = np.eye(self.nocc)
        vir_identity = np.eye(self.nvir)
        for ki, kl in product(range(self.nkpts), repeat=2):
            kd, ka = ia[ki], ia[kl]
            if ki == kl and ka == kd:
                block[ki, kl] += einsum('A,AD,IL->IDLA', e_vir[ka],
                                        vir_identity, occ_identity)
                block[ki, kl] -= einsum('L,AD,IL->IDLA', e_occ[kl],
                                        vir_identity, occ_identity)
            block[ki, kl] += 2 * einsum('LADI->IDLA', eris.ovvo[kl, ka, kd])
            block[ki, kl] -= einsum('ILAD->IDLA', eris.oovv[ki, kl, ka].conj())
        return block

    def get_imds(self, eris=None, kshift=0):
        """Build the singles block through the requested ADC order."""
        self.method = check_implemented(self.method)
        if eris is None:
            eris = self.ao2mo()
        nkpts, nocc, nvir = self.nkpts, self.nocc, self.nvir
        kconserv = self.khelper.kconserv
        kshift_ia, _, _ = self._momentum_maps(kshift)
        v_ovov = eris.ovov
        t1_oovv = t1d_oovv = self.t2[0]
        occ_identity, vir_identity = np.eye(nocc), np.eye(nvir)
        M_ = self._get_adc1(eris, kshift)
        # ADC(2)
        for kI in range(nkpts):
            kD = kshift_ia[kI]
            for kL in range(nkpts):
                kA = kshift_ia[kL]
                for ki in range(nkpts):
                    M_[kI, kL] += 2 * einsum(
                        'IiDa,LAia->IDLA',
                        t1_oovv[kI, ki, kD],
                        v_ovov[kL, kA, ki])
                    M_[kI, kL] -= einsum(
                        'IiDa,iALa->IDLA',
                        t1_oovv[kI, ki, kD],
                        v_ovov[ki, kA, kL])
                    M_[kI, kL] += 2 * einsum(
                        'LiAa,IDia->IDLA',
                        t1d_oovv[kL, ki, kA].conj(),
                        v_ovov[kI, kD, ki].conj())
                    M_[kI, kL] -= einsum(
                        'LiAa,iDIa->IDLA',
                        t1d_oovv[kL, ki, kA].conj(),
                        v_ovov[ki, kD, kI].conj())
                    M_[kI, kL] -= einsum(
                        'iIDa,LAia->IDLA',
                        t1_oovv[ki, kI, kD],
                        v_ovov[kL, kA, ki])
                    M_[kI, kL] += 0.5 * einsum(
                        'iIDa,iALa->IDLA',
                        t1_oovv[ki, kI, kD],
                        v_ovov[ki, kA, kL])
                    M_[kI, kL] -= einsum(
                        'iLAa,IDia->IDLA',
                        t1d_oovv[ki, kL, kA].conj(),
                        v_ovov[kI, kD, ki].conj())
                    M_[kI, kL] += 0.5 * einsum(
                        'iLAa,iDIa->IDLA',
                        t1d_oovv[ki, kL, kA].conj(),
                        v_ovov[ki, kD, kI].conj())
                if kI == kL:
                    M_IL = np.zeros((nocc, nocc), dtype=M_.dtype)
                    M_DA = np.zeros((nvir, nvir), dtype=M_.dtype)
                    for ki in range(nkpts):
                        for ka in range(nkpts):
                            kb = kconserv[kI, ka, ki]
                            M_IL -= einsum(
                                'Iiab,Laib->IL',
                                t1_oovv[kI, ki, ka],
                                v_ovov[kL, ka, ki])
                            M_IL += 0.5 * einsum(
                                'Iiab,Lbia->IL',
                                t1_oovv[kI, ki, ka],
                                v_ovov[kL, kb, ki])
                            M_IL -= einsum(
                                'Liab,Iaib->IL',
                                t1d_oovv[kL, ki, ka].conj(),
                                v_ovov[kI, ka, ki].conj())
                            M_IL += 0.5 * einsum(
                                'Liab,Ibia->IL',
                                t1d_oovv[kL, ki, ka].conj(),
                                v_ovov[kI, kb, ki].conj())
                        for kj in range(nkpts):
                            M_DA -= einsum(
                                'ijAa,iDja->DA',
                                t1d_oovv[ki, kj, kA].conj(),
                                v_ovov[ki, kD, kj].conj())
                            M_DA += 0.5 * einsum(
                                'ijAa,jDia->DA',
                                t1d_oovv[ki, kj, kA].conj(),
                                v_ovov[kj, kD, ki].conj())
                            M_DA -= einsum(
                                'ijDa,iAja->DA',
                                t1_oovv[ki, kj, kD],
                                v_ovov[ki, kA, kj])
                            M_DA += 0.5 * einsum(
                                'ijDa,jAia->DA',
                                t1_oovv[ki, kj, kD],
                                v_ovov[kj, kA, ki])
                    M_[kI, kL] += einsum('IL,DA->IDLA', M_IL, vir_identity)
                    M_[kI, kL] += einsum('IL,DA->IDLA', occ_identity, M_DA)

        if self.method == 'adc(3)':
            self._add_singles_terms(M_, eris, kshift, kadc_ee_terms.SINGLES_3)
        return M_.transpose(0, 2, 3, 1, 4, 5).reshape(
            self.n_singles, self.n_singles)

    def _add_singles_terms(self, block, eris, kshift, table):
        context = MomentumContractions(self, eris, kshift)
        ia, _, _ = self._momentum_maps(kshift)
        for ki, kl in product(range(self.nkpts), repeat=2):
            momenta = dict(I=ki, D=ia[ki], L=kl, A=ia[kl])
            for term in table.get('ss', ()):
                output = term[1].split('->')[1]
                if len(output) == 2 and ki != kl:
                    continue
                signs = {'IDLA': (1, -1, -1, 1),
                         'IL': (1, -1), 'DA': (-1, 1)}[output]
                fixed = {x: momenta[x] for x in output}
                value = context.contract(term, fixed, signs)
                if value is None:
                    continue
                if output == 'IL':
                    value = einsum('IL,DA->IDLA', value, np.eye(self.nvir))
                elif output == 'DA':
                    value = einsum('IL,DA->IDLA', np.eye(self.nocc), value)
                block[ki, kl] += value

    def _add_sigma_terms(self, s1, s2, r1, r2, context, table):
        context.data.update(r1=r1, r2=r2)
        ia, _, doubles = self._momentum_maps(context.kshift)
        for ki in range(self.nkpts):
            for term in table.get('s1', ()):
                value = context.contract(term, {'I': ki, 'D': ia[ki]},
                                         (1, -1))
                if value is not None:
                    s1[ki] += value
        for ki, kj, kc in kpts_helper.loop_kkk(self.nkpts):
            fixed = dict(I=ki, J=kj, C=kc, D=doubles[ki, kc, kj])
            # Contraction labels may differ in case; use the term's output.
            for term in table.get('s2', ()):
                output = term[1].split('->')[1]
                momenta = dict(zip(output, fixed.values()))
                value = context.contract(term, momenta, (1, 1, -1, -1))
                if value is not None:
                    s2[ki, kj, kc] += value

    def get_diag(self, kshift=0, M_ab=None, eris=None):
        """Return the singles and doubles Davidson preconditioner."""
        self.method = check_implemented(self.method)
        if M_ab is None:
            M_ab = self.get_imds(eris, kshift)
        dtype = np.result_type(self.t2[0], M_ab)
        diagonal = np.empty(self.vector_size(), dtype=dtype)
        diagonal[:self.n_singles] = np.diagonal(M_ab).real
        diagonal[self.n_singles:] = self._doubles_diagonal(kshift).ravel()
        return diagonal

    def matvec(self, kshift, M_ab=None, eris=None):
        """Return the ADC matrix action for one momentum sector."""

        self.method = check_implemented(self.method)

        nkpts, nocc, nvir = self.nkpts, self.nocc, self.nvir
        kconserv = self.khelper.kconserv
        f_singles = s_doubles = self.n_singles
        s_singles = 0
        f_doubles = self.vector_size()
        if eris is None:
            eris = self.ao2mo()
        doubles_diagonal = self._doubles_diagonal(kshift)

        v_ovoo = eris.ovoo
        if eris.ovvv is None:
            shape = (nkpts,) * 3 + (nocc,) + (nvir,) * 3
            v_ovvv = np.empty(shape, dtype=eris.Lov.dtype)
            for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
                ks = kconserv[kp, kq, kr]
                v_ovvv[kp, kq, kr] = einsum(
                    'Lpq,Lrs->pqrs',
                    eris.Lov[kp, kq],
                    eris.Lvv[kr, ks]) / nkpts
        else:
            v_ovvv = eris.ovvv

        kshift_ia, kshift_ai, kconserv_r2 = self._momentum_maps(kshift)

        if M_ab is None:
            M_ab = self.get_imds(eris, kshift)

        context = MomentumContractions(self, eris, kshift)

        # Calculate sigma vector
        def sigma_(r):
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles].reshape(self.singles_shape)
            r2 = r[s_doubles:f_doubles].reshape(self.doubles_shape)

            s1 = (M_ab @ r[s_singles:f_singles]).reshape(self.singles_shape)
            s2 = np.empty(self.doubles_shape,
                          dtype=np.result_type(self.t2[0], M_ab, r))

            # doubles rows: holes I, J at kI, kJ; particles C, D at kC, kD
            for kI, kJ, kC in kpts_helper.loop_kkk(nkpts):
                kD = kconserv_r2[kI, kC, kJ]

                s2[kI, kJ, kC] = doubles_diagonal[kI, kJ, kC] * r2[kI, kJ, kC]

                s2[kI, kJ, kC] += einsum(
                    'Ia,JDaC->IJCD',
                    r1[kI],
                    v_ovvv[kJ, kD, kshift_ia[kI]].conj())
                s2[kI, kJ, kC] += einsum(
                    'Ja,ICaD->IJCD',
                    r1[kJ],
                    v_ovvv[kI, kC, kshift_ia[kJ]].conj())
                s2[kI, kJ, kC] -= einsum(
                    'iC,JDIi->IJCD',
                    r1[kshift_ai[kC]],
                    v_ovoo[kJ, kD, kI].conj())
                s2[kI, kJ, kC] -= einsum(
                    'iD,ICJi->IJCD',
                    r1[kshift_ai[kD]],
                    v_ovoo[kI, kC, kJ].conj())

            # singles rows: hole I at kI, particle D at kD
            for kI in range(nkpts):
                kD = kshift_ia[kI]
                for ki in range(nkpts):
                    for ka in range(nkpts):
                        kb = kconserv_r2[kI, ka, ki]
                        s1[kI] -= einsum(
                            'Iiab,iaDb->ID',
                            r2[kI, ki, ka],
                            v_ovvv[ki, ka, kD])
                        s1[kI] += 2 * einsum(
                            'Iiab,ibDa->ID',
                            r2[kI, ki, ka],
                            v_ovvv[ki, kb, kD])
                    for kj in range(nkpts):
                        ka = kconserv_r2[ki, kD, kj]
                        s1[kI] -= 2 * einsum(
                            'ijDa,jaiI->ID',
                            r2[ki, kj, kD],
                            v_ovoo[kj, ka, ki])
                        s1[kI] += einsum(
                            'ijDa,iajI->ID',
                            r2[ki, kj, kD],
                            v_ovoo[ki, ka, kj])

            if self.method in ('adc(2)-x', 'adc(3)'):
                self._add_sigma_terms(s1, s2, r1, r2, context,
                                      kadc_ee_terms.SIGMA_2X)
            if self.method == 'adc(3)':
                self._add_sigma_terms(s1, s2, r1, r2, context,
                                      kadc_ee_terms.SIGMA_3)
            s = np.hstack((s1.reshape(-1), s2.reshape(-1)))

            log.timer_debug1("completed sigma vector calculation", *cput0)

            return s
        return sigma_

    def get_trans_moments(self, kshift, U=None):
        """Spin-free transition densities, flattened by root.

        ``U`` has shape (vector_size, nroots). The returned T has shape
        (nroots, nkpts*nmo*nmo). Its MO pair at k connects k to the
        singles particle momentum. Densities include ADC-order corrections
        rather than constructing a large transition-operator matrix.
        """
        if U is None:
            if self.U is None or kshift not in self._transition_cache:
                raise ValueError('Supply eigenvectors or run kernel first.')
            return self._transition_cache[kshift].reshape(
                self._transition_cache[kshift].shape[0], -1)
        U = self.renormalize_eigenvectors(kshift, U, U.shape[1])
        context = MomentumContractions(self, kshift=kshift)
        tables = [kadc_ee_terms.TRANSITION_2]
        if self.method in ('adc(2)-x', 'adc(3)'):
            tables.append(kadc_ee_terms.TRANSITION_2X)
        if self.method == 'adc(3)':
            tables.append(kadc_ee_terms.TRANSITION_3)
        densities = np.zeros((U.shape[1], self.nkpts, self.nmo, self.nmo),
                             dtype=np.result_type(U, self.t2[0]))
        ia, _, _ = self._momentum_maps(kshift)
        occupied, virtual = slice(None, self.nocc), slice(self.nocc, None)
        for root in range(U.shape[1]):
            context.data.update(
                r1=U[:self.n_singles, root].reshape(self.singles_shape),
                r2=U[self.n_singles:, root].reshape(self.doubles_shape))
            for table in tables:
                for target, terms in table.items():
                    # Targets retain the molecular reference's block names.
                    rows = (virtual if target.startswith(
                        'TY[nocc:') else occupied)
                    cols = virtual if target.endswith('nocc:]') else occupied
                    for kpt in range(self.nkpts):
                        for term in terms:
                            output = term[1].split('->')[1]
                            fixed = dict(zip(output, (kpt, ia[kpt])))
                            value = context.contract(term, fixed, (1, -1))
                            if value is not None:
                                densities[root, kpt, rows, cols] += value
        self._transition_cache[kshift] = densities
        return densities.reshape(U.shape[1], -1)

    def get_density_operator(self, kshift):
        r"""MO matrix of exp[-i(q+G).r], q = -kpts[kshift].

        G is ``density_gv`` in inverse Bohr. At q=G=0 the operator is
        total particle number, so inelastic strengths should vanish.
        """
        from pyscf.pbc.df import ft_ao
        coeff, _, _ = (self.padded_coeff,
            self.e_occ, self.e_vir)
        ia, _, _ = self._momentum_maps(kshift)
        q = -self.kpts[kshift]
        operator = np.empty((self.nkpts, self.nmo, self.nmo),
                            dtype=np.complex128)
        for ki, ka in enumerate(ia):
            ao = ft_ao.ft_aopair(
                self.cell, self.density_gv.reshape(1, 3),
                kpti_kptj=np.array([self.kpts[ki], self.kpts[ka]]), q=q)[0]
            operator[ki] = coeff[ki].conj().T @ ao @ coeff[ka]
        return operator

    def get_properties(self, kshift, U, nroots=1):
        r"""Return density-response weights P and transition densities X.

        P_n = 2/Nk |sum_kpq rho_kpq X_nkpq|^2 is the singlet response
        per primitive cell. It is not an optical oscillator strength.
        X has shape (nroots, nkpts, nmo, nmo); T is its flattened form.
        """
        if U.shape[1] != nroots:
            raise ValueError('nroots does not match the eigenvector array.')
        T = self.get_trans_moments(kshift, U)
        X = T.reshape(nroots, self.nkpts, self.nmo, self.nmo)
        amplitudes = einsum('kpq,nkpq->n', self.get_density_operator(kshift),
            X)
        P = (2.0 / self.nkpts) * abs(amplitudes)**2
        return P.real, X

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        """Normalize EE vectors with the spin-adapted singlet metric.

        The occupied exchange swaps both orbital and k-point indices. This
        avoids requiring an unstored fourth (particle) momentum index.
        """
        if U.shape != (self.vector_size(), nroots):
            raise ValueError(
                'Eigenvectors must have shape (vector_size, nroots).')
        vectors = np.array(U, dtype=np.result_type(U, float), copy=True)
        for root in range(nroots):
            singles = vectors[:self.n_singles, root]
            doubles = vectors[self.n_singles:, root].reshape(
                self.doubles_shape)
            exchanged = doubles.transpose(1, 0, 2, 4, 3, 5, 6)
            norm = (np.vdot(singles, singles) + np.vdot(doubles, doubles)
                    - 0.5 * np.vdot(doubles, exchanged)).real
            if norm <= LOOSE_ZERO_TOL:
                raise ValueError(
                    'Cannot normalize an EE vector with zero norm.')
            vectors[:, root] /= np.sqrt(norm)
        return vectors

    def get_init_guess(self, nroots=1, diag=None, ascending=True,
                       type=None, ini=None, kshift=None):
        """Build unit guesses or validate user-supplied root vectors."""
        dim = self.vector_size()
        if not 1 <= nroots <= dim:
            raise ValueError('nroots must be between 1 and vector_size().')
        if type == 'read':
            logger.info(self, 'Obtain initial guess from input variable')
            g = np.asarray(ini).T.copy()
            if g.shape != (dim, nroots):
                raise ValueError(
                    f'Shape of guess each k point should be ({dim},{nroots})')
        else:
            if diag is None:
                diag = self.get_diag(0 if kshift is None else kshift)
            idx = np.argsort(diag)
            if not ascending:
                idx = idx[::-1]
            g = np.zeros((dim, nroots), dtype=diag.dtype)
            g[idx[:nroots], np.arange(nroots)] = 1

        for root in range(nroots):
            if np.linalg.norm(g[:, root]) < LOOSE_ZERO_TOL:
                raise ValueError(
                    f'Guess vector for root {root} has zero norm; '
                    'check the input and frozen orbitals.')
        return [g[:, root] for root in range(nroots)]

    def gen_matvec(self, kshift, imds=None, eris=None):
        if eris is None:
            eris = self.ao2mo()
        if imds is None:
            imds = self.get_imds(eris, kshift)
        diag = self.get_diag(kshift, imds, eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag
