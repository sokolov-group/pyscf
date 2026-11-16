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
r"""K-point-adapted restricted singlet excitation-energy ADC.

The vector contains all momentum-conserving singles and doubles for one
hole-minus-particle sector kshift. ADC(2), ADC(2)-x and ADC(3) use shared
reference amplitudes from kadc_rhf.KRADC. EE properties are transition
densities and density-response strengths at physical q = -kpts[kshift],
optionally shifted by density_gv; they are not optical oscillator strengths.
Numerical examples include their imports and build a two-k-point helium
reference ``kmf`` locally, without relying on another example.
"""
from itertools import product
import time

import numpy as np

from pyscf.lib import logger
from pyscf.lib.parameters import LOOSE_ZERO_TOL
from pyscf.pbc.adc import kadc_ee_terms, kadc_rhf
from pyscf.pbc.adc.lib import MomentumContractions, check_implemented, einsum
from pyscf.pbc.lib import kpts_helper


class KRADCEE(kadc_rhf.KRADC):
    r"""Solve singlet periodic EE-ADC and compute momentum-resolved density
    response.

    Args:
        adc (KRADC): Reference with orbitals and perturbative amplitudes,
            normally prepared by ground_state_energy(). Method and solver
            settings are shared without transferring amplitude-file ownership.

    Attributes:
        type_ (str): Fixed to 'ee'.
        method (str): 'adc(2)', 'adc(2)-x' or 'adc(3)'.
        E (ndarray): Excitation energies in Hartree, shape (nsectors, nroots).
        U (ndarray): Singlet-metric eigenvectors, shape
            (nsectors, nroots, vector_size()).
        X (ndarray): Spin-free transition MO densities, shape
            (nsectors, nroots, nkpts, nmo, nmo).
        T (ndarray): Flattened view of X, shape
            (nsectors, nroots, nkpts*nmo*nmo), or None without properties.
        P (ndarray): Dimensionless density-response strengths per primitive
            cell, shape (nsectors, nroots).
        density_gv (ndarray): Reciprocal vector G, in Cartesian inverse Bohr.
            Properties use the operator exp[-i(q+G).r], q=-kpts[kshift].
        converged (ndarray): Boolean root-convergence flags with E's shape.

    Singles have axes (k_i, i, a); doubles have axes
    (k_i, k_j, k_a, i, j, a, b), with the remaining particle momentum fixed by
    k_i+k_j-k_a-k_b=kshift modulo a reciprocal vector. Vectors concatenate
    these blocks in C order. All saved sector axes follow kptlist order.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
        >>> _ = kmf.kernel()
        >>> mykradc = kadc_rhf.KRADC(kmf)
        >>> mykradc.type_ = 'ee'
        >>> _ = mykradc.ground_state_energy()
        >>> mykradcee = KRADCEE(mykradc)
        >>> _ = mykradcee.kernel(2, kptlist=[0])
        >>> energies = mykradcee.E
        >>> vectors = mykradcee.U
        >>> strengths = mykradcee.P
        >>> densities = mykradcee.X
        >>> energies.round(6).tolist()
        [[1.756312, 2.102047]]
        >>> vectors.shape, strengths.shape, densities.shape
        ((1, 2, 10), (1, 2), (1, 2, 2, 2, 2))
    """

    def __init__(self, adc):
        r"""Share a ground-state reference and initialize neutral response
        caches.

        Args:
            adc (KRADC): Reference containing the orbitals, amplitudes and
                settings
                used to build the EE matrix and transition densities.

        Returns:
            None: Set type_='ee', clear saved results and initialize momentum
            and
            transition-density caches. Copy density_gv so editing the solver's
            response vector does not change the reference's vector. The
            reference
            retains ownership of amplitude scratch files.
        """
        self._init_from_ground_state(adc)
        self.type_ = 'ee'
        self._ee_momentum = {}
        self._transition_cache = {}
        self.density_gv = np.asarray(adc.density_gv).copy()
        self.E = self.T = self.U = self.X = self.P = None
        self._keys = set(self.__dict__)

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        # Rebuild cached data if orbitals or the k mesh changed between runs.
        r"""Solve selected EE momentum sectors after clearing response/momentum
        caches.

        Args:
            nroots (int): Roots per sector, capped at vector_size(). With zero
                or a negative count, compute only the reference and clear
                roots.
            guess (array_like, optional): Shape
                (len(kptlist), nroots, vector_size()), in requested sector
                order.
            eris (_ChemistsERIs, optional): Active integrals; None calls
                ao2mo().
            kptlist (sequence[int], optional): Sector indices, default all k
                points.
                Positive-root runs require previously computed ground
                amplitudes.

        Returns:
            tuple: E, U, P, X in the shapes documented by KRADCEE;
            with nroots <
            1,
            return the scalar correlation energy in Hartree per primitive cell.
            Results are saved on the solver. With compute_properties=False, P/X
            remain zero arrays and T is None. See the class worked example.
        """
        self._ee_momentum.clear()
        self._transition_cache.clear()
        return kadc_rhf.kernel(self, nroots, guess, eris, kptlist)

    @property
    def nvir(self):
        r"""Return the padded active virtual count for neutral excitation
        tensors.

        Returns:
            int: nmo - nocc. Frozen virtuals are excluded, while k-point
            padding
            is retained to keep common singles/doubles orbital dimensions.
        """
        return self.nmo - self.nocc

    @property
    def singles_shape(self):
        r"""Describe the neutral singles tensor before C-order flattening.

        Returns:
            tuple[int]: (nkpts, nocc, nvir), with axes (k_i, i, a). The
            selected
            kshift fixes k_a from k_i-k_a=kshift; it is not another independent
            tensor axis. See _momentum_maps() for the explicit lookup.
        """
        return (self.nkpts, self.nocc, self.nvir)

    @property
    def doubles_shape(self):
        r"""Describe the neutral doubles tensor before C-order flattening.

        Returns:
            tuple[int]: (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), with
            axes
            (k_i, k_j, k_a, i, j, a, b). The k_b momentum follows conservation
            in the selected sector. No orbital-pair permutation symmetry is
            used
            to compress this tensor.
        """
        return (self.nkpts,) * 3 + (self.nocc,) * 2 + (self.nvir,) * 2

    @property
    def n_singles(self):
        r"""Return the number of neutral particle-hole configurations.

        Returns:
            int: nkpts*nocc*nvir, the product of singles_shape. Each
            configuration
            lies in the same selected momentum sector; its particle k index is
            inferred from its hole k index rather than stored independently.
        """
        return int(np.prod(self.singles_shape))

    @property
    def n_doubles(self):
        r"""Return the full number of neutral two-hole/two-particle
        configurations.

        Returns:
            int: nkpts**3*nocc**2*nvir**2, the product of doubles_shape. This
            counts
            the full spin-adapted tensor layout used by matrix actions and
            normalization, including padding and exchanged orbital
            configurations.
        """
        return int(np.prod(self.doubles_shape))

    def vector_size(self):
        r"""Return the total length of a flattened EE response vector.

        Returns:
            int: n_singles+n_doubles. A vector concatenates the C-flattened
            singles_shape and doubles_shape blocks in that order.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> mykradcee = KRADCEE(mykradc)
            >>> (mykradcee.n_singles, mykradcee.n_doubles,
            ...  mykradcee.vector_size())
            (2, 8, 10)
        """
        return self.n_singles + self.n_doubles

    def _momentum_maps(self, kshift):
        r"""Construct or reuse the neutral-sector particle-momentum lookup
        tables.

        Args:
            kshift (int): Hole-minus-particle sector index, in [0, nkpts).

        Returns:
            tuple[ndarray, ndarray, ndarray]: ia, its inverse ai, and the
            doubles
            lookup. ia[ki] gives ka with ki-ka=kshift; ai[ka] gives the hole
            index. doubles[ki, ka, kj] gives kb with ki-ka+kj-kb=kshift, modulo
            a reciprocal vector. Shapes are (nkpts,), (nkpts,) and (nkpts,)*3.
            Maps are cached per sector until kernel() clears them.

        Raises:
            ValueError: The supplied mesh is not closed under the required
                momentum combinations within the lattice-coordinate tolerance.
        """
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
        r"""Build the zeroth-order EE doubles energy differences for one
        sector.

        Args:
            kshift (int): Sector used to determine the unstored fourth
                momentum.

        Returns:
            ndarray: Shape doubles_shape, ordered as (k_i,k_j,k_a,i,j,a,b),
            with
            entries epsilon_a(k_a)+epsilon_b(k_b)-epsilon_i(k_i)-epsilon_j(k_j)
            in Hartree. The result includes padded slots and uses the real
            energy
            dtype. No integral or interaction correction is added here.
        """
        _, e_occ, e_vir = (self.padded_coeff,
            self.e_occ, self.e_vir)
        _, _, kconserv = self._momentum_maps(kshift)
        diagonal = np.empty(
            self.doubles_shape, dtype=np.result_type(e_occ, e_vir))
        for ki, kj, ka in kpts_helper.loop_kkk(self.nkpts):
            kb = kconserv[ki, ka, kj]
            occupied = e_occ[ki][:, None] + e_occ[kj]
            virtual = e_vir[ka][:, None] + e_vir[kb]
            diagonal[ki, kj, ka] = (virtual[None, None, :, :]
                                     - occupied[:, :, None, None])
        return diagonal

    @property
    def _singles_ham_shape(self):
        r"""Describe the orbital tensor used while assembling the singles M
        matrix.

        Returns:
            tuple[int]: (nkpts, nkpts, nocc, nvir, nocc, nvir), whose axes are
            (k_i, k_l, i, a, l, b). get_imds() groups (k_i,i,a) and (k_l,l,b)
            into its row and column indices only after all order corrections.
        """
        return (self.nkpts, self.nkpts, self.nocc, self.nvir,
                self.nocc, self.nvir)

    def _get_adc1_ham(self, eris, kshift=0):
        r"""Build the zeroth- and first-order effective Hamiltonian M matrix.

        In the spin-adapted singles basis, the orbital block is

        .. math::
            M^{(0+1)}_{ID,LA} = (\epsilon_D-\epsilon_I)
            \delta_{IL}\delta_{DA} + 2(LA|DI) - (IL|AD)^*.

        Args:
            eris (_ChemistsERIs): Chemists' MO integral blocks, normalized
                by the number of k points.
            kshift (int): Hole-minus-particle momentum sector.

        Returns:
            ndarray: M with axes (k_I, k_L, I, D, L, A), shape
            (nkpts, nkpts, nocc, nvir, nocc, nvir). The particle momenta
            follow the selected sector. No ground-state amplitudes are used.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee._get_adc1_ham(eris, kshift=1)
            >>> M.shape
            (2, 2, 1, 1, 1, 1)
        """
        e_occ, e_vir = self.e_occ, self.e_vir
        ia, _, _ = self._momentum_maps(kshift)
        M = np.zeros(self._singles_ham_shape, dtype=self.padded_coeff.dtype)
        for ki in range(self.nkpts):
            kd = ia[ki]
            # Prototype:
            # M[ki, ki] += einsum('A,AD,IL->IDLA', e_vir[kd],
            #                     np.eye(self.nvir), np.eye(self.nocc))
            # M[ki, ki] -= einsum('L,AD,IL->IDLA', e_occ[ki],
            #                     np.eye(self.nvir), np.eye(self.nocc))
            # A repeated-index view updates only the i-a diagonal, without
            # constructing identities or full four-index energy tensors.
            diagonal = np.einsum('idid->id', M[ki, ki])
            diagonal += e_vir[kd][None, :] - e_occ[ki][:, None]
        for ki, kl in product(range(self.nkpts), repeat=2):
            kd, ka = ia[ki], ia[kl]
            M[ki, kl] += 2 * eris.ovvo[kl, ka, kd].transpose(3, 2, 0, 1)
            M[ki, kl] -= eris.oovv[ki, kl, ka].conj().transpose(0, 3, 1, 2)
        return M

    def _get_adc2_ham(self, eris, kshift=0, M=None):
        r"""Accumulate the second-order correction to the singles M matrix.

        The correction contracts first-order doubles amplitudes with MO
        integrals. ADC(2)-x uses the same singles block as ADC(2).

        Args:
            eris (_ChemistsERIs): Active-space MO integral container.
            kshift (int): Hole-minus-particle momentum sector.
            M (ndarray, optional): Existing Hamiltonian tensor to update
                in place. With None, allocate a zero tensor instead.

        Returns:
            ndarray: The updated M, with axes (k_I, k_L, I, D, L, A).
            If M was omitted, this contains only :math:`\Delta M^{(2)}`.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee._get_adc1_ham(eris, kshift=1)
            ...     updated = mykradcee._get_adc2_ham(eris, 1, M=M)
            >>> updated is M, updated.shape
            (True, (2, 2, 1, 1, 1, 1))
        """
        nkpts, nocc, nvir = self.nkpts, self.nocc, self.nvir
        kconserv = self.khelper.kconserv
        kshift_ia, _, _ = self._momentum_maps(kshift)
        v_ovov = eris.ovov
        t1_oovv = t1d_oovv = self.t2[0]
        if M is None:
            M = np.zeros(self._singles_ham_shape, dtype=t1_oovv.dtype)
        occ_identity, vir_identity = np.eye(nocc), np.eye(nvir)
        # ADC(2)
        for kI in range(nkpts):
            kD = kshift_ia[kI]
            for kL in range(nkpts):
                kA = kshift_ia[kL]
                for ki in range(nkpts):
                    M[kI, kL] += 2 * einsum(
                        'IiDa,LAia->IDLA',
                        t1_oovv[kI, ki, kD], v_ovov[kL, kA, ki])
                    M[kI, kL] -= einsum(
                        'IiDa,iALa->IDLA',
                        t1_oovv[kI, ki, kD], v_ovov[ki, kA, kL])
                    M[kI, kL] += 2 * einsum(
                        'LiAa,IDia->IDLA',
                        t1d_oovv[kL, ki, kA].conj(), v_ovov[kI, kD, ki].conj())
                    M[kI, kL] -= einsum(
                        'LiAa,iDIa->IDLA',
                        t1d_oovv[kL, ki, kA].conj(), v_ovov[ki, kD, kI].conj())
                    M[kI, kL] -= einsum(
                        'iIDa,LAia->IDLA',
                        t1_oovv[ki, kI, kD], v_ovov[kL, kA, ki])
                    M[kI, kL] += 0.5 * einsum(
                        'iIDa,iALa->IDLA',
                        t1_oovv[ki, kI, kD], v_ovov[ki, kA, kL])
                    M[kI, kL] -= einsum(
                        'iLAa,IDia->IDLA',
                        t1d_oovv[ki, kL, kA].conj(), v_ovov[kI, kD, ki].conj())
                    M[kI, kL] += 0.5 * einsum(
                        'iLAa,iDIa->IDLA',
                        t1d_oovv[ki, kL, kA].conj(), v_ovov[ki, kD, kI].conj())
                if kI == kL:
                    M_IL = np.zeros((nocc, nocc), dtype=M.dtype)
                    M_DA = np.zeros((nvir, nvir), dtype=M.dtype)
                    for ki in range(nkpts):
                        for ka in range(nkpts):
                            kb = kconserv[kI, ka, ki]
                            M_IL -= einsum(
                                'Iiab,Laib->IL',
                                t1_oovv[kI, ki, ka], v_ovov[kL, ka, ki])
                            M_IL += 0.5 * einsum(
                                'Iiab,Lbia->IL',
                                t1_oovv[kI, ki, ka], v_ovov[kL, kb, ki])
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
                                t1_oovv[ki, kj, kD], v_ovov[ki, kA, kj])
                            M_DA += 0.5 * einsum(
                                'ijDa,jAia->DA',
                                t1_oovv[ki, kj, kD], v_ovov[kj, kA, ki])
                    M[kI, kL] += einsum('IL,DA->IDLA', M_IL, vir_identity)
                    M[kI, kL] += einsum('IL,DA->IDLA', occ_identity, M_DA)

        return M

    def _get_adc3_ham(self, eris, kshift=0, M=None):
        r"""Accumulate the third-order correction to the singles M matrix.

        Ground-state singles and first-/second-order doubles enter the
        contractions in SINGLES_3. MomentumContractions retains independent
        k axes and selects the conjugations for complex Bloch orbitals.

        Args:
            eris (_ChemistsERIs): Active-space MO integrals or DF factors.
            kshift (int): Hole-minus-particle momentum sector.
            M (ndarray, optional): Existing Hamiltonian tensor to update
                in place. With None, allocate a zero tensor instead.

        Returns:
            ndarray: The updated M, shape _singles_ham_shape. If M was
            omitted, the result contains only :math:`\Delta M^{(3)}`.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> mykradc.method = 'adc(3)'
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee._get_adc1_ham(eris, kshift=1)
            ...     updated = mykradcee._get_adc3_ham(eris, 1, M=M)
            >>> updated is M, updated.shape
            (True, (2, 2, 1, 1, 1, 1))
        """
        if M is None:
            M = np.zeros(self._singles_ham_shape, dtype=self.t2[0].dtype)
        context = MomentumContractions(self, eris, kshift)
        ia, _, _ = self._momentum_maps(kshift)
        for ki, kl in product(range(self.nkpts), repeat=2):
            momenta = dict(I=ki, D=ia[ki], L=kl, A=ia[kl])
            for term in kadc_ee_terms.SINGLES_3.get('ss', ()):
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
                M[ki, kl] += value

        return M

    def get_imds(self, eris=None, kshift=0):
        r"""Assemble the singles effective Hamiltonian through the ADC order.

        .. math::
            M_{11} = M^{(0+1)} + \Delta M^{(2)}
                     + \delta_{n,3}\Delta M^{(3)}.

        Args:
            eris (_ChemistsERIs, optional): Pretransformed MO integrals.
                With None, call ao2mo().
            kshift (int): Hole-minus-particle momentum sector.

        Returns:
            ndarray: A matrix of shape (n_singles, n_singles), using the
            flattened (k_I, I, D) and (k_L, L, A) ordering. All order
            builders accumulate into one tensor to avoid an extra dense
            Hamiltonian allocation per perturbative order.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee.get_imds(eris, kshift=1)
            >>> M.shape
            (2, 2)
        """
        self.method = check_implemented(self.method)
        if eris is None:
            eris = self.ao2mo()
        M = self._get_adc1_ham(eris, kshift)
        self._get_adc2_ham(eris, kshift, M=M)
        if self.method == 'adc(3)':
            self._get_adc3_ham(eris, kshift, M=M)
        return M.transpose(0, 2, 3, 1, 4, 5).reshape(
            self.n_singles, self.n_singles)

    def _add_sigma_terms(self, s1, s2, r1, r2, context, table):
        r"""Accumulate table-driven higher-order singles and doubles matrix
        actions.

        Args:
            s1, s2 (ndarray): Writable sigma tensors shaped singles_shape and
                doubles_shape. Contributions are added in place.
            r1, r2 (ndarray): Response blocks in those same shapes. Stored in
                context.data so each contraction uses the current trial vector.
            context (MomentumContractions): Reference/integral context at the
                selected sector, with reusable ground-only contraction caches.
            table (dict): Lists of term triples under the s1/s2 output keys.
                See kadc_ee_terms for coefficient and orbital-label
                conventions.

        Returns:
            None: Update s1/s2 and the response entries of context.data. Each
            term
            sums only the momentum assignments allowed by the conserved sector.
        """
        context.data.update(r1=r1, r2=r2)
        ia, _, doubles = self._momentum_maps(context.kshift)
        for ki in range(self.nkpts):
            for term in table.get('s1', ()):
                value = context.contract(term, {'I': ki, 'D': ia[ki]}, (1, -1))
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
        r"""Build the diagonal used to precondition EE Davidson residuals.

        Singles use the real diagonal of the effective Hamiltonian M_11.
        Doubles use zeroth-order energy differences,
        :math:`\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j`, with the
        fourth k point fixed by the selected momentum sector. ADC(2)-x and
        ADC(3) doubles interaction corrections remain in matvec; this
        preconditioner omits them and is therefore an approximate diagonal.

        Args:
            kshift (int): Hole-minus-particle momentum sector, default zero.
            M_ab (ndarray, optional): Precomputed singles matrix, shape
                (n_singles, n_singles). With None, call get_imds().
            eris (_ChemistsERIs, optional): Integrals used if M_ab is absent.
                With both absent, get_imds() transforms the integrals.

        Returns:
            ndarray: Diagonal of length vector_size(), ordered as flattened
            singles followed by doubles. The dtype accommodates M_ab and
            t2[0], although the energy entries are real.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee.get_imds(eris, kshift=1)
            ...     diagonal = mykradcee.get_diag(1, M, eris)
            >>> M.shape, diagonal.shape
            ((2, 2), (10,))
            >>> bool(np.allclose(diagonal[:mykradcee.n_singles], M.diagonal()))
            True
        """
        self.method = check_implemented(self.method)
        if M_ab is None:
            M_ab = self.get_imds(eris, kshift)
        dtype = np.result_type(self.t2[0], M_ab)
        diagonal = np.empty(self.vector_size(), dtype=dtype)
        diagonal[:self.n_singles] = np.diagonal(M_ab).real
        diagonal[self.n_singles:] = self._doubles_diagonal(kshift).ravel()
        return diagonal

    def matvec(self, kshift, M_ab=None, eris=None):
        r"""Prepare a callable applying the EE effective Hamiltonian at one
        momentum.

        The action evaluates :math:`\sigma=M(k)r` through the selected ADC
        order,
        using full singles/doubles vectors without constructing a dense ADC or
        supercell matrix. A missing DF ovvv tensor is reconstructed once when
        preparing the closure; higher-order table contractions reuse a bounded
        MomentumContractions cache across trial vectors.

        Args:
            kshift (int): Hole-minus-particle momentum sector.
            M_ab (ndarray, optional): Precomputed singles matrix, shape
                (n_singles, n_singles). None builds it with get_imds(eris,
                kshift).
            eris (_ChemistsERIs, optional): Active integrals or DF factors.
                With
                None, transform them using ao2mo().

        Returns:
            callable: Function accepting a flat vector of length vector_size()
            and returning a new sigma vector of the same length. Integral
            scratch
            files supplied by the caller must remain open while it is used.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> with mykradcee.ao2mo() as eris:
            ...     action = mykradcee.matvec(0, eris=eris)
            ...     sigma = action(np.ones(mykradcee.vector_size()))
            >>> sigma.shape
            (10,)
        """

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
                    eris.Lov[kp, kq], eris.Lvv[kr, ks]) / nkpts
        else:
            v_ovvv = eris.ovvv

        kshift_ia, kshift_ai, kconserv_r2 = self._momentum_maps(kshift)

        if M_ab is None:
            M_ab = self.get_imds(eris, kshift)

        context = MomentumContractions(self, eris, kshift)

        # Calculate sigma vector
        def sigma_(r):
            r"""Apply the captured EE Hamiltonian to one trial vector.

            Args:
                r (ndarray): Length-vector_size() singles-then-doubles response
                    vector, with both blocks in C-flattened order. Input is not
                    modified.

            Returns:
                ndarray: Sigma vector in the same layout, including the singles
                matrix, zeroth-order doubles energies, coupling terms and the
                ADC(2)-x/ADC(3) corrections selected by the enclosing solver.
            """
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles].reshape(self.singles_shape)
            r2 = r[s_doubles:f_doubles].reshape(self.doubles_shape)

            s1 = (M_ab @ r[s_singles:f_singles]).reshape(self.singles_shape)
            s2 = np.empty(
                self.doubles_shape, dtype=np.result_type(self.t2[0], M_ab, r))

            # doubles rows: holes I, J at kI, kJ; particles C, D at kC, kD
            for kI, kJ, kC in kpts_helper.loop_kkk(nkpts):
                kD = kconserv_r2[kI, kC, kJ]

                s2[kI, kJ, kC] = doubles_diagonal[kI, kJ, kC] * r2[kI, kJ, kC]

                s2[kI, kJ, kC] += einsum(
                    'Ia,JDaC->IJCD',
                    r1[kI], v_ovvv[kJ, kD, kshift_ia[kI]].conj())
                s2[kI, kJ, kC] += einsum(
                    'Ja,ICaD->IJCD',
                    r1[kJ], v_ovvv[kI, kC, kshift_ia[kJ]].conj())
                s2[kI, kJ, kC] -= einsum(
                    'iC,JDIi->IJCD',
                    r1[kshift_ai[kC]], v_ovoo[kJ, kD, kI].conj())
                s2[kI, kJ, kC] -= einsum(
                    'iD,ICJi->IJCD',
                    r1[kshift_ai[kD]], v_ovoo[kI, kC, kJ].conj())

            # singles rows: hole I at kI, particle D at kD
            for kI in range(nkpts):
                kD = kshift_ia[kI]
                for ki in range(nkpts):
                    for ka in range(nkpts):
                        kb = kconserv_r2[kI, ka, ki]
                        s1[kI] -= einsum(
                            'Iiab,iaDb->ID',
                            r2[kI, ki, ka], v_ovvv[ki, ka, kD])
                        s1[kI] += 2 * einsum(
                            'Iiab,ibDa->ID',
                            r2[kI, ki, ka], v_ovvv[ki, kb, kD])
                    for kj in range(nkpts):
                        ka = kconserv_r2[ki, kD, kj]
                        s1[kI] -= 2 * einsum(
                            'ijDa,jaiI->ID',
                            r2[ki, kj, kD], v_ovoo[kj, ka, ki])
                        s1[kI] += einsum(
                            'ijDa,iajI->ID',
                            r2[ki, kj, kD], v_ovoo[ki, ka, kj])

            if self.method in ('adc(2)-x', 'adc(3)'):
                self._add_sigma_terms(
                    s1, s2, r1, r2, context, kadc_ee_terms.SIGMA_2X)
            if self.method == 'adc(3)':
                self._add_sigma_terms(
                    s1, s2, r1, r2, context, kadc_ee_terms.SIGMA_3)
            s = np.hstack((s1.reshape(-1), s2.reshape(-1)))

            log.timer_debug1('completed sigma vector calculation', *cput0)

            return s
        return sigma_

    def get_trans_moments(self, kshift, U=None):
        r"""Evaluate spin-free EE transition densities, flattened by root.

        Instead of constructing the full transition-operator matrix, contract
        the
        normalized response amplitudes with the ADC-order transition terms.
        Each
        MO pair connects hole momentum k to particle momentum ia[k] in this
        sector.
        The resulting T is the flattened density X for these particular roots.

        Args:
            kshift (int): Hole-minus-particle momentum sector.
            U (ndarray, optional): Eigenvector columns of shape
                (vector_size(), nroots). A normalized copy is used; U is
                unchanged.
                With None, read densities cached by an earlier property-enabled
                kernel run for this sector.

        Returns:
            ndarray: Complex/real transition densities of shape
            (nroots, nkpts*nmo*nmo), in k,p,q C order. The returned array is a
            view
            of the cached density tensor; treat it as read-only when reusing
            caches.
            Higher-order terms follow method and approx_trans_moments.

        Raises:
            ValueError: U is absent and a saved solver run/cached sector is
            missing.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> _ = mykradcee.kernel(2, kptlist=[0])
            >>> mykradcee.get_trans_moments(0).shape
            (2, 8)
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
        densities = np.zeros(
            (U.shape[1], self.nkpts, self.nmo, self.nmo),
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
        r"""Transform the Fourier density operator exp[-i(q+G).r] to active
        MOs.

        For q=-kpts[kshift] and G=density_gv, the AO-pair Fourier transform is
        sandwiched between the coefficients at k and ia[k]. G is Cartesian and
        uses inverse Bohr. At q+G=0 the operator is total particle number, so
        its
        inelastic response vanishes up to the approximation/numerical accuracy.

        Args:
            kshift (int): Sector fixing physical q and the particle momentum
                map.

        Returns:
            ndarray: Complex operator matrices of shape (nkpts, nmo, nmo), with
            bra MOs at k and ket MOs at ia[k]. No 1/nkpts strength
            normalization
            is included here; get_properties() applies it after contraction.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> mykradcee = KRADCEE(mykradc)
            >>> mykradcee.get_density_operator(0).shape
            (2, 2, 2)
        """
        from pyscf.pbc.df import ft_ao
        coeff, _, _ = (self.padded_coeff,
            self.e_occ, self.e_vir)
        ia, _, _ = self._momentum_maps(kshift)
        q = -self.kpts[kshift]
        operator = np.empty(
            (self.nkpts, self.nmo, self.nmo), dtype=np.complex128)
        for ki, ka in enumerate(ia):
            ao = ft_ao.ft_aopair(
                self.cell,
                self.density_gv.reshape(1, 3),
                kpti_kptj=np.array([self.kpts[ki], self.kpts[ka]]),
                q=q)[0]
            operator[ki] = coeff[ki].conj().T @ ao @ coeff[ka]
        return operator

    def get_properties(self, kshift, U, nroots=1):
        r"""Contract transition densities into density-response strengths per
        cell.

        For the Fourier density matrix rho and the spin-free transition density
        X,

        .. math::
            P_n(q+G)=\frac{2}{N_k}\left|\sum_{kpq}
                \rho_{kpq}(q+G)X_{nkpq}\right|^2.

        The factor of two follows the singlet convention. These are
        dimensionless
        density-response strengths, distinct from optical oscillator strengths.

        Args:
            kshift (int): Momentum sector; physical q=-kpts[kshift].
            U (ndarray): Eigenvector columns of shape (vector_size(), nroots).
                Use a normalized copy, preserving the supplied U.
            nroots (int): Number of roots, which must equal U's column count.

        Returns:
            tuple[ndarray, ndarray]: Real P of shape (nroots,) and transition
            MO
            densities X of shape (nroots, nkpts, nmo, nmo). Their flattened
            form
            T is cached by get_trans_moments() for this sector.

        Raises:
            ValueError: nroots does not match the supplied eigenvector columns.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> _ = mykradc.ground_state_energy()
            >>> mykradcee = KRADCEE(mykradc)
            >>> _ = mykradcee.kernel(1, kptlist=[1])
            >>> strengths, densities = mykradcee.get_properties(
            ...     1, mykradcee.U[0].T, 1)
            >>> strengths.shape, densities.shape
            ((1,), (1, 2, 2, 2))
        """
        if U.shape[1] != nroots:
            raise ValueError('nroots does not match the eigenvector array.')
        T = self.get_trans_moments(kshift, U)
        X = T.reshape(nroots, self.nkpts, self.nmo, self.nmo)
        amplitudes = einsum(
            'kpq,nkpq->n',
            self.get_density_operator(kshift), X)
        P = (2.0 / self.nkpts) * abs(amplitudes)**2
        return P.real, X

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        r"""Return copies of EE eigenvectors normalized in the singlet metric.

        The norm is :math:`U_1^\dagger U_1+U_2^\dagger U_2
        -\frac12 U_2^\dagger P_h U_2`. The hole exchange P_h swaps both
        occupied
        orbital and hole k indices. This convention matches the molecular limit
        while retaining the full, uncompressed doubles tensor.

        Args:
            kshift (int): Sector label accepted by the shared normalization
                API.
                The hole-exchange implementation itself does not need this
                index.
            U (ndarray): Columns of shape (vector_size(), nroots). Integer
                input
                is promoted to floating point. The original array is not
                modified.
            nroots (int): Number of columns to normalize; must match U's shape.

        Returns:
            ndarray: Independent normalized eigenvectors with U's shape.

        Raises:
            ValueError: U has the wrong shape or a nonpositive/zero metric
            norm.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> mykradcee = KRADCEE(mykradc)
            >>> vectors = np.zeros((mykradcee.vector_size(), 1))
            >>> vectors[0, 0] = 2
            >>> normalized = mykradcee.renormalize_eigenvectors(0, vectors)
            >>> float(normalized[0, 0]), float(vectors[0, 0])
            (1.0, 2.0)
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

    def _neutral_init_guess(self, nroots=1, diag=None, ascending=True,
                            type_=None, ini=None, kshift=None):
        r"""Build EE unit guesses or validate user-supplied root vectors.

        Args:
            nroots (int): Number of guesses, from 1 to vector_size().
            diag (ndarray, optional): Preconditioner of length vector_size().
                With None, compute it for the selected momentum sector.
            ascending (bool): Select the lowest diagonal entries first;
                False selects the highest. Ignored for supplied vectors.
            type_ (str, optional): None selects unit guesses; 'read' selects
                ini. Other values are unsupported.
            ini (array_like, optional): Trial vectors with shape
                (nroots, vector_size()). Input is copied without modification.
            kshift (int, optional): Momentum sector, default zero.

        Returns:
            list[ndarray]: nroots vectors of length vector_size(). Invalid
            shapes and zero-norm guesses raise ValueError.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = kadc_rhf.KRADC(kmf)
            >>> mykradcee = KRADCEE(mykradc)
            >>> guesses = mykradcee._neutral_init_guess(
            ...     2, diag=np.arange(10.0))
            >>> [int(np.argmax(vector)) for vector in guesses]
            [0, 1]
        """
        if type_ not in (None, 'read'):
            raise NotImplementedError(f'Unknown initial-guess type: {type_!r}')
        dim = self.vector_size()
        if not 1 <= nroots <= dim:
            raise ValueError('nroots must be between 1 and vector_size().')
        if type_ == 'read':
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
        r"""Prepare a consistent neutral matrix action and Davidson diagonal.

        Args:
            kshift (int): Hole-minus-particle momentum sector.
            imds (ndarray, optional): Precomputed sector-dependent singles
                matrix,
                shape (n_singles, n_singles). With None, call get_imds().
            eris (_ChemistsERIs, optional): Shared active integrals or DF
                factors.
                With None, transform once and reuse for both outputs.

        Returns:
            tuple[callable, ndarray]: Matrix action sigma(r) and
            length-vector_size()
            preconditioner. The latter uses interacting singles and
            zeroth-order
            doubles energies; see get_diag(). Keep any supplied integral owner
            open for the lifetime of the returned action.
        """
        if eris is None:
            eris = self.ao2mo()
        if imds is None:
            imds = self.get_imds(eris, kshift)
        diag = self.get_diag(kshift, imds, eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag


# Earlier class name retained for importing applications.
RADCEE = KRADCEE
