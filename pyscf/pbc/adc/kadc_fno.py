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

r"""Frozen-natural-orbital spaces and additive corrections for periodic ADC.

Diagonalize a correlated virtual one-particle density at each k point,
select the leading natural orbitals, then diagonalize the projected Fock
matrix in the retained space. The driver supports ground-state and charged
state-specific/state-averaged densities, together with multiple independent
truncation settings. Numerical examples include their imports and build
their own two-k-point helium reference ``kmf``.
"""
import warnings

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_rhf
from pyscf.pbc.mp.kmp2 import padding_k_idx


def _count_thresholds(thresh, pct_occ, nvir_act):
    r"""Determine how many independent FNO truncation runs were requested.

    Args:
        thresh: Occupation cutoff, scalar, sequence or None.
        pct_occ: Cumulative-occupation fraction, scalar, sequence or None.
        nvir_act: Retained virtual count, scalar, sequence or None.

    Returns:
        int: Maximum sequence length, with a minimum of one. Scalars are
        reused by _pick() for every run. Sequence lengths are not validated
        here; callers should supply equally long sequences for all varying
        parameters so later indexed selections are defined.

    Examples:
        >>> from pyscf.pbc.adc.kadc_fno import _count_thresholds
        >>> _count_thresholds([0.0001, 1e-05], None, None)
        2
        >>> _count_thresholds(0.0001, None, 3)
        1
    """
    n = 1
    for val in (thresh, pct_occ, nvir_act):
        if val is not None and isinstance(val, (list, tuple, np.ndarray)):
            n = max(n, len(val))
    return n


def _pick(param, i):
    r"""Select a per-run FNO parameter while broadcasting scalar settings.

    Args:
        param: Scalar, list, tuple, NumPy array or None.
        i (int): Zero-based run index; used only for sequence parameters.

    Returns:
        object: param[i] for a sequence, otherwise param itself. None remains
        None. Ordinary indexing errors propagate if a sequence is too short.

    Examples:
        >>> from pyscf.pbc.adc.kadc_fno import _pick
        >>> _pick([0.2, 0.4], 1), _pick(0.2, 1), _pick(None, 0) is None
        (0.4, 0.2, True)
    """
    if param is not None and isinstance(param, (list, tuple, np.ndarray)):
        return param[i]
    return param


class _KRADCFNO(kadc_rhf.KRADCBase):
    r"""Generate periodic frozen natural orbitals and additive ADC corrections.

    The natural orbitals diagonalize the virtual block of a correlated density,
    :math:`D_{vv}(k)V(k)=V(k)n(k)`. The retained subspace is semicanonicalized
    by diagonalizing :math:`V_{keep}^\dagger F_{vv}V_{keep}`. Occupied orbitals
    and previously frozen spaces are retained in the resulting MO ordering.

    Args:
        mf (KRHF, Cell or KRADCBase): Restricted periodic reference or ADC
            object.
        frozen (int or sequence, optional): Initial KMP2 frozen-space setting.
        mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
            orbital overrides as in KRADCBase.
        kpts (array_like, optional): Mesh for Cell inputs, default Gamma.

    Attributes:
        mode (str): 'min' keeps a common virtual count equal to the largest
            per-k requested count; 'per_kpt' keeps each count independently. At
            least one active virtual is retained at every k point.
        ref_state: None selects a ground-state density. A positive integer
            selects a ranked charged root using the legacy indexing rule;
            (root_lists, sector_list) explicitly selects zero-based roots,
            either shared across sectors or as one list per sector.
        if_ref_qp (bool): Filter explicit selected roots and correction lists
            by the quasiparticle threshold is_qp (default True).
        is_qp (float): Minimum spectroscopic factor for quasiparticle
            selection, default 0.5.
        rdm1_ss (ndarray): Ground/state-averaged density defining the FNOs.
        e_can, v_can, p_can: Canonical charged energies, vectors and strengths.
        e_corr_can (float): Canonical reference correlation energy per cell.
        e_ssfno, v_ssfno, p_ssfno: Corresponding charged results in FNO spaces.
        e_corr_fno: Active-space correlation energy, scalar or per-run list.
        delta_e, delta_e_corr: Canonical-minus-FNO excitation/correlation
            corrections, in Hartree. Use correct()/correct_corr() to add them.
        delta_e_qp: Canonical-minus-FNO differences for matched selected poles.

    nvir_act takes precedence over pct_occ, which takes precedence over thresh.
    Sequence-valued settings request independent truncations from one canonical
    density; result/orbital attributes become per-run lists. Amplitude scratch
    arrays correspond to the final run, rather than a list of all runs.

    Examples:
        Create the two-k-point helium reference below; lattice vectors
        are in Angstrom and energies are in Hartree:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradc_fno = KRADCIPFNO(kmf)
        >>> energy = mykradc_fno.ground_state_energy(nvir_act=1)
        >>> round(energy, 8), round(mykradc_fno.correct_corr(energy), 8)
        (-0.00416766, -0.00416766)
        >>> int(mykradc_fno.nvir), len(mykradc_fno.mo_coeff)
        (1, 2)
    """
    _keys = kadc_rhf.KRADCBase._keys | {'delta_e', 'e_can', 'v_can', 'e_corr_can',
                                   'rdm1_ss', 'mode', 'ref_state',
                                   'delta_e_corr', 'p_can',
                                   'if_ref_qp', 'delta_e_qp', 'is_qp',
                                   'e_corr_fno'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 mo_energy=None, kpts=None):
        r"""Initialize the reference driver and unset FNO/correction results.

        Args:
            mf (KRHF, Cell or KRADCBase): Reference for the concrete FNO
                sector.
            frozen (int or sequence, optional): Initial frozen-space
                specification, None inherits an ADC source setting or freezes
                no orbitals.
            mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Orbital
                replacements passed to the KRADCBase reference initializer.
            kpts (array_like, optional): Mesh for Cell inputs, default Gamma.

        Returns:
            None: Choose common-count mode, ground-density FNOs and
            quasiparticle filtering with threshold 0.5. No canonical ADC
            calculation or natural orbital transformation is performed during
            construction.
        """
        super().__init__(mf, frozen, mo_coeff, mo_occ, mo_energy, kpts)
        self.delta_e = None
        self.delta_e_corr = None
        self.delta_e_qp = None
        self.is_qp = 0.5
        self.e_can = None
        self.v_can = None
        self.p_can = None
        self.e_corr_can = None
        self.rdm1_ss = None
        self.mode = "min"
        self.ref_state = None
        self.if_ref_qp = True
        self.e_corr_fno = None

    def _reset_adc_state(self):
        r"""Clear reference/solver caches before recalculating in a changed FNO
        space.

        Returns:
            None: Drop t1, t2, _adc_es and the virtual-ladder intermediate;
            clear
            cached orbital padding and any automatic chunk estimate. Canonical
            energies, FNO densities and correction results remain available.
            The next amplitude calculation replaces the previous scratch owner.
        """
        self.t1 = None
        self.t2 = None
        self._adc_es = None
        self._invalidate_orbitals()
        if hasattr(self.imds, 't2_1_vvvv'):
            self.imds.t2_1_vvvv = None

    def ground_state_energy(self, eris=None, thresh=1e-4,
                            pct_occ=None, nvir_act=None):
        r"""Generate FNO spaces and compute their reference correlation
        energies.

        Compute one canonical reference density, then construct each requested
        truncation from the same canonical MOs. Record the per-cell additive
        correction :math:`\Delta E_c=E_c^{can}-E_c^{FNO}` separately from the
        returned active-space energy. This method selects ground-density FNOs,
        setting ref_state to None.

        Args:
            eris: Accepted for signature compatibility; this FNO workflow
            rebuilds
                integrals for the canonical and changed orbital spaces.
            thresh (float or sequence): Natural occupation cutoff, default
                1e-4;
                retain occupations strictly greater than this cutoff when no
                higher-priority selection is given.
            pct_occ (float or sequence, optional): Fraction in [0,1] used by
                the
                cumulative-occupation rule documented in make_fno().
            nvir_act (int or sequence, optional): Explicit virtual count,
                taking
                precedence over both pct_occ and thresh.

        Returns:
            float or list[float]: Active-space correlation energy in Hartree
            per
            primitive cell for each run. Store it as e_corr_fno; delta_e_corr
            is
            separate. With multiple runs, MO coefficients, energies,
            occupations
            and frozen lists become per-run collections. Amplitudes remain
            those
            of the last run. correct_corr() adds the corresponding correction.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradc_fno = KRADCIPFNO(kmf)
            >>> energies = mykradc_fno.ground_state_energy(nvir_act=[1, 1])
            >>> np.round(energies, 8).tolist()
            [-0.00416766, -0.00416766]
            >>> len(mykradc_fno.mo_coeff), len(mykradc_fno.delta_e_corr)
            (2, 2)
        """
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, 'generate fno with correction for the ground state')
        self.ref_state = None

        self.make_ss_rdm1(log, cput0, if_gs=True)
        log.timer('make gs rdm1', *cput0)

        # Snapshot canonical orbital layout needed by make_fno
        _can_frozen = self.frozen
        _can_mo_coeff = self.mo_coeff
        _can_mo_energy = self.mo_energy
        _can_mo_occ = [np.copy(oc) for oc in self.mo_occ]

        n_thresh = _count_thresholds(thresh, pct_occ, nvir_act)

        all_mo_coeff, all_mo_energy, all_frozen = [], [], []
        all_delta_e_corr, all_e_corr_fno = [], []

        for i in range(n_thresh):
            # Restore canonical orbitals so make_fno always starts from the
            # full space
            self.frozen = _can_frozen
            self.mo_coeff = _can_mo_coeff
            self.mo_energy = _can_mo_energy

            t = _pick(thresh, i)
            p = _pick(pct_occ, i)
            v = _pick(nvir_act, i)

            self.make_fno(
                self.rdm1_ss, self._scf, log, thresh=t, pct_occ=p, nvir_act=v)

            # Snapshot FNO orbitals before solver state is overwritten
            fno_mo_coeff = list(self.mo_coeff)
            fno_mo_energy = list(self.mo_energy)
            fno_frozen = list(self.frozen)

            self._reset_adc_state()
            self.compute_correction(if_gs=True)

            all_mo_coeff.append(fno_mo_coeff)
            all_mo_energy.append(fno_mo_energy)
            all_frozen.append(fno_frozen)
            all_delta_e_corr.append(self.delta_e_corr)
            all_e_corr_fno.append(self.e_corr)

            log.timer(f'get frozen info ({i + 1}/{n_thresh})', *cput0)

        log.timer('gs FNO', *cput0)

        # Expose results — single threshold keeps scalar attrs (backward
        # compatible)
        if n_thresh == 1:
            self.mo_coeff = all_mo_coeff[0]
            self.mo_energy = all_mo_energy[0]
            self.frozen = all_frozen[0]
            self.delta_e_corr = all_delta_e_corr[0]
            self.e_corr_fno = all_e_corr_fno[0]
        else:
            self.mo_coeff = all_mo_coeff
            self.mo_energy = all_mo_energy
            self.frozen = all_frozen
            self.mo_occ = [[np.copy(oc) for oc in _can_mo_occ]
                           for threshold_index in range(n_thresh)]
            self.delta_e_corr = all_delta_e_corr
            self.e_corr_fno = all_e_corr_fno

        return self.e_corr_fno

    def kernel_gs(self, eris=None, thresh=1e-4, pct_occ=None, nvir_act=None):
        r"""Generate ground-density FNOs through the deprecated method name.

        Args:
            eris, thresh, pct_occ, nvir_act: Same settings and precedence as
                ground_state_energy(). eris is accepted but not reused across
                the canonical/FNO orbital transformations.

        Returns:
            None: Emit a DeprecationWarning, invoke ground_state_energy(), and
            retain results in e_corr_fno and delta_e_corr. The earlier wrapper
            does not forward the scalar/list return value; use the primary
            method
            when the computed energy is required directly.
        """
        warnings.warn(
            'Use ground_state_energy() for FNO generation.',
            DeprecationWarning,
            stacklevel=2)
        self.ground_state_energy(eris, thresh, pct_occ, nvir_act)

    def kernel(self, nroots=1, guess=None, eris=None, thresh=1e-4,
        pct_occ=None, nvir_act=None, kptlist=None):
        r"""Generate ground/state-specific FNO spaces and charged excitation
        corrections.

        Compute canonical results, form the reference density selected by
        ref_state,
        and solve each FNO space using the current ADC approximation. Store
        :math:`\Delta E_n=E_n^{can}-E_n^{FNO}` for subsequent additive
        correction.
        State-specific/state-averaged densities use the charged IP/EA density
        API.

        Args:
            nroots (int): Canonical/FNO roots per selected sector. Nonpositive
                counts request ground-state FNO energies only and clear old
                roots.
            guess (array_like, optional): Trial vectors supplied to canonical
                and
                FNO kernels. They must match the vector layout of the
                corresponding
                calculation; None allows new guesses after virtual truncation.
            eris: Retained for the driver signature; changing FNO spaces causes
                fresh integral transformations rather than reusing this object.
            thresh (float or sequence): Natural-occupation cutoff, default
                1e-4.
            pct_occ (float or sequence, optional): Cumulative occupation
                criterion.
            nvir_act (int or sequence, optional): Explicit active virtual
                count;
                selection precedence is nvir_act, pct_occ, then thresh.
            kptlist (sequence[int], optional): Charged sectors, default all k
                points.

        Returns:
            None or float or list[float]: Positive-root runs store results in
            e_ssfno/v_ssfno/p_ssfno, e_corr_fno and correction attributes and
            return
            None. Multiple settings produce per-run lists. Nonpositive-root
            runs
            return the scalar/list from ground_state_energy() instead.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradc_fno = KRADCIPFNO(kmf)
            >>> result = mykradc_fno.kernel(1, nvir_act=1, kptlist=[0])
            >>> result is None, mykradc_fno.e_ssfno.shape
            (True, (1, 1))
            >>> mykradc_fno.correct(mykradc_fno.e_ssfno).round(6).tolist()
            [[0.371589]]
        """
        if not isinstance(nroots, (int, np.integer)):
            raise TypeError('nroots must be an integer.')
        if nroots < 1:
            energy = self.ground_state_energy(
                eris, thresh=thresh, pct_occ=pct_occ, nvir_act=nvir_act)
            self._clear_excitations()
            logger.info(self, f'nroots = {nroots}; only the FNO ground state '
                        'was calculated.')
            return energy
        if isinstance(self, kadc_rhf.KRADCEE) and self.ref_state is not None:
            raise NotImplementedError(
                "EE FNO calculations require ref_state=None.")
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)
        if self.ref_state is None:
            logger.info(
                self, 'generate fno with correction for the excited state')
        elif (isinstance(self.ref_state, int) and 0 < self.ref_state <= nroots
            or (hasattr(self.ref_state, '__len__') and len(
                self.ref_state) == 2)):
            logger.info(
                self, 'generate ss-fno with correction for the excited state')
        else:
            raise ValueError((
                'ref_state should be an int type or or a array-like'
                ' object with two elements'))

        if kptlist is None:
            kptlist = range(self.nkpts)

        self.make_ss_rdm1(log, cput0, kptlist, nroots, guess)
        log.timer('make ss rdm1', *cput0)

        # Snapshot canonical orbital layout
        _can_frozen = self.frozen
        _can_mo_coeff = self.mo_coeff
        _can_mo_energy = self.mo_energy
        _can_mo_occ = [np.copy(oc) for oc in self.mo_occ]

        n_thresh = _count_thresholds(thresh, pct_occ, nvir_act)

        all_mo_coeff, all_mo_energy, all_frozen = [], [], []
        all_delta_e_corr, all_e_corr_fno = [], []
        all_delta_e, all_delta_e_qp = [], []
        all_e_ssfno, all_v_ssfno, all_p_ssfno = [], [], []

        for i in range(n_thresh):
            self.frozen = _can_frozen
            self.mo_coeff = _can_mo_coeff
            self.mo_energy = _can_mo_energy

            t = _pick(thresh, i)
            p = _pick(pct_occ, i)
            v = _pick(nvir_act, i)

            self.make_fno(
                self.rdm1_ss, self._scf, log, thresh=t, pct_occ=p, nvir_act=v)

            fno_mo_coeff = list(self.mo_coeff)
            fno_mo_energy = list(self.mo_energy)
            fno_frozen = list(self.frozen)

            self._reset_adc_state()
            self.compute_correction(kptlist, nroots, guess)

            all_mo_coeff.append(fno_mo_coeff)
            all_mo_energy.append(fno_mo_energy)
            all_frozen.append(fno_frozen)
            all_delta_e_corr.append(self.delta_e_corr)
            all_e_corr_fno.append(self.e_corr)
            all_delta_e.append(self.delta_e)
            all_delta_e_qp.append(self.delta_e_qp)
            all_e_ssfno.append(self.e_ssfno)
            all_v_ssfno.append(self.v_ssfno)
            all_p_ssfno.append(self.p_ssfno)

            log.timer(f'es FNO ({i + 1}/{n_thresh})', *cput0)

        log.timer('es FNO', *cput0)

        if n_thresh == 1:
            self.mo_coeff = all_mo_coeff[0]
            self.mo_energy = all_mo_energy[0]
            self.frozen = all_frozen[0]
            self.delta_e_corr = all_delta_e_corr[0]
            self.e_corr_fno = all_e_corr_fno[0]
            self.delta_e = all_delta_e[0]
            self.delta_e_qp = all_delta_e_qp[0]
        else:
            self.mo_coeff = all_mo_coeff
            self.mo_energy = all_mo_energy
            self.frozen = all_frozen
            self.mo_occ = [[np.copy(oc) for oc in _can_mo_occ]
                           for threshold_index in range(n_thresh)]
            self.delta_e_corr = all_delta_e_corr
            self.e_corr_fno = all_e_corr_fno
            self.delta_e = all_delta_e
            self.delta_e_qp = all_delta_e_qp
            self.e_ssfno = all_e_ssfno
            self.v_ssfno = all_v_ssfno
            self.p_ssfno = all_p_ssfno

    def compute_correction(self, kptlist=None, nroots=None, guess=None,
        if_gs=False):
        r"""Calculate canonical-minus-FNO corrections with the current ADC
        method.

        The canonical energies must already be stored by make_ss_rdm1(). Solve
        the current FNO orbital space using the base KRADCBase implementation,
        so this method does not recursively enter the FNO driver. Default
        ADC(2) gives MP2/ADC(2) correction data; other method settings use
        their own equations.

        Args:
            kptlist (sequence[int], optional): Sectors to solve for charged
                roots; required when if_gs=False. Quasiparticle corrections
                currently index saved energy arrays directly by these sector
                numbers; use their full mesh order, or set if_ref_qp=False for
                arbitrary subsets.
            nroots (int, optional): Roots per sector, required for charged
                runs.
            guess (array_like, optional): Trial vectors in the current FNO
                layout.
            if_gs (bool): Compute only the FNO reference correlation when True.

        Returns:
            None: Update e_corr and delta_e_corr=e_corr_can-e_corr. Charged
            runs also update e_ssfno/v_ssfno/p_ssfno and delta_e=e_can-e_ssfno,
            plus matched quasiparticle differences when if_ref_qp=True.
        """
        if if_gs:
            kadc_rhf.KRADCBase.ground_state_energy(self)
        else:
            (self.e_ssfno, self.v_ssfno, self.p_ssfno, transition_amplitudes) = (
                kadc_rhf.KRADCBase.kernel(
                    self, nroots, guess=guess, kptlist=kptlist)
            )
            self.delta_e = self.e_can - self.e_ssfno
            if self.if_ref_qp:
                self.delta_e_qp = []
                mask_fno = self.p_ssfno > self.is_qp
                mask_can = self.p_can > self.is_qp
                for kpt in kptlist:
                    e_can_qp_k = self.e_can[kpt][mask_can[kpt]]
                    e_ssfno_qp_k = self.e_ssfno[kpt][mask_fno[kpt]]
                    n_qp = min(len(e_can_qp_k), len(e_ssfno_qp_k))
                    self.delta_e_qp.append(
                        e_can_qp_k[:n_qp] - e_ssfno_qp_k[:n_qp])
        self.delta_e_corr = self.e_corr_can - self.e_corr

    def correct(self, e, i=None):
        r"""Add a stored canonical-minus-FNO correction to excitation energies.

        Args:
            e (float or ndarray): Excitation energies in Hartree, with shape
                compatible with the stored sector/root correction.
            i (int, optional): Run index for sequence-valued truncation
                settings.
                With None, use delta_e directly, as for a single FNO space.

        Returns:
            float or ndarray: e + delta_e[i], or e + delta_e for a single run.
            The input and stored correction are not modified. Correction data
            must already exist from kernel()/compute_correction().

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradc_fno = KRADCIPFNO(kmf)
            >>> mykradc_fno.delta_e = np.array([[0.01, 0.02]])
            >>> mykradc_fno.correct(np.array([[1.0, 2.0]])).tolist()
            [[1.01, 2.02]]
        """
        return e + (self.delta_e[i] if i is not None else self.delta_e)

    def correct_corr(self, e, i=None):
        r"""Add a stored canonical-minus-FNO reference correlation correction.

        Args:
            e (float or ndarray): Reference correlation contribution in Hartree
                per primitive cell, rather than the SCF total energy.
            i (int, optional): Truncation-run index when delta_e_corr is a
                list.
                With None, use the scalar single-run correction directly.

        Returns:
            float or ndarray: e plus the selected delta_e_corr. Correction data
            must already exist; neither input nor stored data are modified.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradc_fno = KRADCIPFNO(kmf)
            >>> energy = mykradc_fno.ground_state_energy(nvir_act=1)
            >>> round(mykradc_fno.correct_corr(energy), 8)
            -0.00416766
        """
        return (e + (self.delta_e_corr[
            i] if i is not None else self.delta_e_corr))

    def make_ss_rdm1(self, log, cput0, kptlist=None, nroots=None, guess=None,
        if_gs=False):
        r"""Compute the canonical reference and the density used to define
        FNOs.

        Args:
            log (logger.Logger): Logger for memory, selection and timing
                messages.
            cput0 (tuple[float, float]): Initial CPU/wall timer pair.
            kptlist (sequence[int], optional): Canonical charged sectors when
                if_gs=False; their order determines stored excitation arrays.
            nroots (int, optional): Charged roots per sector for the canonical
                run.
            guess (array_like, optional): Canonical-layout initial root
                vectors.
            if_gs (bool): Compute only the canonical ground state when True;
                ref_state must then be None.

        Returns:
            None: Store e_corr_can and rdm1_ss, shaped (nkpts, nmo_full,
            nmo_full),
            with frozen MOs restored by the density API. Charged runs
            additionally
            store e_can/v_can/p_can. Clear amplitudes and the old solver after
            the
            density is built, preserving the canonical results for correction.

        For ref_state=None, use the ground density. Explicit (root_lists,
        sectors)
        selection averages the requested charged densities with their
        additional
        reference contribution. If if_ref_qp=True, root-list entries select
        among
        poles with P>is_qp. The integer ref_state form is one-based and uses
        nkpts
        as its legacy flattening stride; explicit lists avoid that indexing
        rule.

        Examples:
            >>> import numpy as np
            >>> from pyscf.lib import logger
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradc_fno = KRADCIPFNO(kmf)
            >>> log = logger.Logger(mykradc_fno.stdout, 0)
            >>> started = logger.process_clock(), logger.perf_counter()
            >>> mykradc_fno.make_ss_rdm1(log, started, if_gs=True)
            >>> mykradc_fno.rdm1_ss.shape, mykradc_fno.t2 is None
            ((2, 2, 2), True)
        """
        if if_gs:
            kadc_rhf.KRADCBase.ground_state_energy(self)
        else:
            self.e_can, self.v_can, self.p_can, transition_amplitudes = kadc_rhf.KRADCBase.kernel(
                self, nroots, guess=guess, kptlist=kptlist)
        log.info(f'Current memory use: {lib.current_memory()[0]:.0f} MB')
        self.e_corr_can = self.e_corr
        if self.ref_state is not None:
            if isinstance(self.ref_state, (int, np.integer)):
                idx = np.argsort(self.e_can.ravel()).tolist()
                sidx = [[idx[self.ref_state - 1] % self.nkpts]]
                kidx = [idx[self.ref_state - 1] // self.nkpts]
            elif hasattr(self.ref_state, '__len__'):
                if len(self.ref_state) != 2:
                    raise ValueError
                if not isinstance(
                    self.ref_state[0], list
                ) or not isinstance(self.ref_state[1], list):
                    raise ValueError((
                        'when ref_state is a array-like object, both '
                        'elements should be list type'))
                if not isinstance(self.ref_state[1][0], (int, np.integer)):
                    raise ValueError((
                        'elements in the second list of ref_state should be'
                        ' int type'))
                if isinstance(self.ref_state[0][0], (int, np.integer)):
                    sidx = ([self.ref_state[0] for sector_index in range(
                        len(self.ref_state[1]))])
                else:
                    if len(self.ref_state[0]) != len(self.ref_state[1]):
                        raise ValueError((
                            'when the first element of ref_state is a '
                            'array-like object, its length should be the same '
                            'as the second element'))
                    sidx = self.ref_state[0]
                kidx = self.ref_state[1]
                if self.if_ref_qp:
                    state_list = []
                    mask_can = self.p_can > self.is_qp
                    for kpt, kshift in enumerate(kidx):
                        k = kptlist.index(kshift)
                        state_list_k = np.arange(nroots)
                        state_list_k = state_list_k[mask_can[k]].tolist()
                        state_list.append([state_list_k[s] for s in sidx[kpt]])
                    sidx = state_list

            log.info(f'the specific state is {sidx} with kidx {kidx}')
            es_DM = self.make_rdm1(kptlist, root=sidx, K_idx=kidx, if_ss=True)
            self.rdm1_ss = np.zeros_like(es_DM[0][0])
            n_state = sum([len(s_k) for s_k in sidx])
            for k in range(len(kidx)):
                for i in range(len(sidx[k])):
                    self.rdm1_ss += es_DM[k][i] / n_state
            log.info(f'Current memory use: {lib.current_memory()[0]:.0f} MB')
            log.timer('make ss rdm1', *cput0)
        else:
            self.rdm1_ss = self.make_ref_rdm1()
            log.info(f'Current memory use: {lib.current_memory()[0]:.0f} MB')
            log.timer('make ref rdm1', *cput0)

        self.t1 = None
        self.t2 = None
        self._adc_es = None
        self.imds.t2_1_vvvv = None

    def make_fno(self, rdm1_ss, mf, log, thresh=None, pct_occ=None,
        nvir_act=None):
        r"""Rotate virtual MOs to retained semicanonical natural-orbital
        subspaces.

        Diagonalize each correlated virtual density, sort natural occupations
        in
        descending order, select leading vectors and diagonalize the canonical
        Fock matrix projected into the retained space. Frozen occupied and
        virtual
        MOs are retained; discarded natural orbitals become frozen virtuals.

        Args:
            rdm1_ss (ndarray): Density with frozen MOs restored, shape
                (nkpts, nmo_full, nmo_full), matching the current canonical
                orbitals.
            mf (KRHF): Retained for interface compatibility; the transformation
                reads orbital coefficients/energies from this FNO driver.
            log (logger.Logger): Logger for automatic one-orbital retention
                messages.
            thresh (float, optional): Keep natural occupations strictly above
                this
                value, if neither higher-priority criterion is supplied.
            pct_occ (float, optional): Keep entries whose normalized cumulative
                occupation is <= pct_occ, including np.isclose boundary values.
                This is the implemented prefix rule; it does not add the next
                orbital when that would exceed the requested fraction.
            nvir_act (int, optional): Keep this many leading natural orbitals,
                taking precedence over pct_occ and thresh. At least one
                selection
                criterion must be supplied to this low-level method.

        Returns:
            None: Replace mo_coeff, mo_energy and frozen with one entry per k
            point.
            MO order becomes frozen occupied, active occupied, retained
            virtual,
            discarded virtual, earlier frozen virtual. mode='min' keeps the
            largest
            per-k requested count everywhere; other modes use individual
            counts.
            At least one virtual is retained. The driver clears
            amplitude/orbital
            caches before using the changed space for a new ADC calculation.

        Examples:
            >>> import numpy as np
            >>> from pyscf.lib import logger
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_fno import KRADCIPFNO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradc_fno = KRADCIPFNO(kmf)
            >>> log = logger.Logger(mykradc_fno.stdout, 0)
            >>> started = logger.process_clock(), logger.perf_counter()
            >>> mykradc_fno.make_ss_rdm1(log, started, if_gs=True)
            >>> mykradc_fno.make_fno(mykradc_fno.rdm1_ss, kmf, log, nvir_act=1)
            >>> len(mykradc_fno.mo_coeff), mykradc_fno.mo_coeff[0].shape
            (2, (2, 2))
        """
        nocc = self.nocc
        no_coeff = []
        no_frozen = []
        no_energy = []
        V_trunc = []
        padding_convention = padding_k_idx(self, kind='joint')

        T = []
        for kpt in range(self.nkpts):
            masks = kadc_rhf.mo_splitter(self, kpt)
            if self.frozen is None:
                idx_k = padding_convention[kpt]
            else:
                idx_k = np.where(masks[1] | masks[2])[0]
            rdm1_ss_comp = rdm1_ss[kpt][np.ix_(idx_k, idx_k)]
            n, V_k = np.linalg.eigh(rdm1_ss_comp[nocc:, nocc:])
            idx = np.argsort(n)[::-1]
            n, V_k = n[idx], V_k[:, idx]
            V_trunc.append(V_k)
            if nvir_act is not None:
                T.append(np.arange(len(n)) < nvir_act)
            elif pct_occ is not None:
                cumsum = np.cumsum(n / np.sum(n))
                T.append(np.array(
                    [c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum]))
            else:
                T.append(n > thresh)

        # "min": union the per-kpt masks so every k-point keeps the same count
        n_keep_union = None
        if self.mode.lower() == "min":
            n_keep_union = max(int(np.sum(t)) for t in T)
            if n_keep_union == 0:
                log.warn(('All virtual orbitals were requested to be frozen.\n'
                    'At least one virtual orbital must be retained for '
                    'ADC calculations.\nKeeping one virtual orbital '
                    'automatically.'))
                n_keep_union = 1

        for kpt in range(self.nkpts):
            masks = kadc_rhf.mo_splitter(self, kpt)
            V_trunc_k = V_trunc[kpt]
            if n_keep_union is not None:
                n_keep = min(n_keep_union, V_trunc_k.shape[1])
            else:
                n_keep = int(np.sum(T[kpt]))
                if n_keep == 0:
                    log.warn(f'All virtual orbitals frozen at kpt {kpt}; '
                             'keeping one.')
                    n_keep += 1
                    T[kpt][0] = True

            (moeoccfrz0, moeocc, moevir, moevirfrz0) = ([self.mo_energy[kpt][
                m] for m in masks])
            orboccfrz0, orbocc, orbvir, orbvirfrz0 = [self.mo_coeff[kpt][:,
                m] for m in masks]
            F_can = np.diag(moevir)
            F_trunc = V_trunc_k.T.conj().dot(F_can).dot(V_trunc_k)
            e_trunc, Z_trunc = np.linalg.eigh(F_trunc[:n_keep, :n_keep])
            e_fro = np.diagonal(F_trunc[n_keep:, n_keep:]).copy()
            U_vir_act = orbvir.dot(V_trunc_k[:, :n_keep]).dot(Z_trunc)
            U_vir_fro = orbvir.dot(V_trunc_k[:, n_keep:])
            no_comp = (orboccfrz0, orbocc, U_vir_act, U_vir_fro, orbvirfrz0)
            no_e_comp = (moeoccfrz0, moeocc, e_trunc, e_fro, moevirfrz0)
            no_coeff_k = np.hstack(no_comp)
            no_energy_k = np.hstack(no_e_comp)
            nocc_loc = (np.cumsum([0] + [x.shape[1] for x in no_comp]).astype(
                int))
            no_frozen_k = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                    np.arange(
                                        nocc_loc[3], nocc_loc[5]))).astype(int)
            no_coeff.append(no_coeff_k)
            no_energy.append(no_energy_k)
            no_frozen.append(no_frozen_k)

        (self.mo_coeff, self.mo_energy, self.frozen) = ((no_coeff, no_energy,
            no_frozen))


class KRADCIPFNO(_KRADCFNO, kadc_rhf.KRADCIP):
    r"""IP-ADC with frozen natural orbitals and optional state-specific
    densities.

    Accepts the same reference inputs as KRADCIP. Common FNO settings and
    truncation/correction methods are documented by _KRADCFNO.
    """


class KRADCEAFNO(_KRADCFNO, kadc_rhf.KRADCEA):
    r"""EA-ADC with frozen natural orbitals and optional state-specific
    densities.

    Accepts the same reference inputs as KRADCEA. Common FNO settings and
    truncation/correction methods are documented by _KRADCFNO.
    """


class KRADCEEFNO(_KRADCFNO, kadc_rhf.KRADCEE):
    r"""EE-ADC with FNOs generated from the correlated reference density.

    Accepts the same reference inputs as KRADCEE. Set ref_state=None: EE
    transition densities do not supply state-specific FNO densities.
    """


# Earlier FNO name selects IP; explicit sector classes are preferred.
RADC2FNO = KRADCIPFNO
