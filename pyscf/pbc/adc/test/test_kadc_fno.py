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
from pyscf.pbc import gto, scf, adc
import numpy as np
import unittest
import warnings
warnings.simplefilter('ignore', ResourceWarning)


nroots = 3


def setUpModule():
    r"""Build the four-k carbon KRHF reference used by all FNO checks.

    Returns:
        None: Store cell and kmf for a two-carbon gth-szv/gth-pade cell,
        on a 2x2x1 mesh with exxdiv=None. Canonical and truncated calculations
        then share the same converged reference orbitals.
    """
    global cell, kmf
    cell = gto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.atom = '''
    C 0.000000000000   0.000000000000   0.000000000000
    C 1.685068664391   1.685068664391   1.685068664391'''
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.a = '''
    0.000000000, 3.370137329, 3.370137329
    3.370137329, 0.000000000, 3.370137329
    3.370137329, 3.370137329, 0.000000000'''
    cell.build()
    kpts = cell.make_kpts([2, 2, 1])
    kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
    kmf.verbose = 0
    kmf.kernel()


def tearDownModule():
    r"""Release the shared carbon cell and converged KRHF fixture.

    Returns:
        None: Delete module cell/kmf references after the FNO test group.
    """
    global cell, kmf
    del cell, kmf


def _fno(pct_occ=None, thresh=None, nvir_act=None, mode=None):
    r"""Generate an ADC(2) ground-density FNO space for a truncation setting.

    Args:
        pct_occ (float or sequence, optional): Cumulative occupation criterion.
        thresh (float or sequence, optional): Natural occupation cutoff.
        nvir_act (int or sequence, optional): Explicit virtual count.
        mode (str, optional): Common-count or per-k truncation mode override.

    Returns:
        KRADCIPFNO: Driver containing FNO orbitals, canonical MP2 reference
        information and additive correlation corrections. ADC(3) is evaluated
        separately by _gs_corr(), rather than by this helper.
    """
    mykradc_fno = adc.KRADC2FNO(kmf)
    mykradc_fno.approx_trans_moments = True
    mykradc_fno.verbose = 0
    if mode is not None:
        mykradc_fno.mode = mode
    mykradc_fno.ground_state_energy(
        pct_occ=pct_occ, thresh=thresh, nvir_act=nvir_act)
    return mykradc_fno


def _fno_es(solver_type, ref_state, pct_occ):
    r"""Generate charged state-specific/state-averaged ADC(2) FNO data.

    Args:
        solver_type (type): KRADCIP or KRADCEA.
        ref_state: Explicit selected root/sector lists for the FNO density.
        pct_occ (float): Cumulative virtual occupation criterion.

    Returns:
        KRADCIPFNO: Driver with sector-zero canonical/FNO roots, their vectors
        and additive ADC(2) corrections, using module-level nroots.
    """
    mykradc_fno = adc.KRADC2FNO(solver_type(kmf))
    mykradc_fno.approx_trans_moments = True
    mykradc_fno.verbose = 0
    mykradc_fno.ref_state = ref_state
    mykradc_fno.kernel(nroots, pct_occ=pct_occ, kptlist=[0])
    return mykradc_fno


def _gs_corr(fg, i=None):
    r"""Compute an ADC(3) reference energy with a selected FNO correction.

    Args:
        fg (KRADCIPFNO): Generator holding FNO orbitals and ADC(2) correction
            data.
        i (int, optional): Run index for a multi-threshold generator.

    Returns:
        float: Per-cell ADC(3) correlation contribution in Hartree plus the
        selected canonical-minus-FNO MP2 correction. A separate KRADCBase
        driver uses the selected coefficients, energies, occupations and frozen
        list.
    """
    frozen = fg.frozen[i] if i is not None else fg.frozen
    mo_coeff = fg.mo_coeff[i] if i is not None else fg.mo_coeff
    mo_energy = fg.mo_energy[i] if i is not None else fg.mo_energy
    mo_occ = fg.mo_occ[i] if (i is not None and isinstance(
        fg.mo_coeff, list)) else fg.mo_occ
    mykradc = adc.KRADCIP(kmf, frozen, mo_coeff, mo_occ, mo_energy)
    mykradc.verbose = 0
    mykradc.method = 'adc(3)'
    return fg.correct_corr(mykradc.ground_state_energy(), i)


def _es_corr(fg, solver_type, i=None):
    r"""Compute the first corrected ADC(3) charged excitation in an FNO space.

    Args:
        solver_type (type): KRADCIP or KRADCEA.
        fg (KRADCIPFNO): Generator with FNO orbitals, initial root vectors and
            additive ADC(2) excitation corrections.

        i (int, optional): Run index for multi-threshold data.

    Returns:
        float: Corrected root-zero sector-zero energy in Hartree, after a
        separate ADC(3) calculation initialized with the selected FNO vectors.
    """
    frozen = fg.frozen[i] if i is not None else fg.frozen
    mo_coeff = fg.mo_coeff[i] if i is not None else fg.mo_coeff
    mo_occ = fg.mo_occ[i] if (i is not None and isinstance(
        fg.mo_coeff, list)) else fg.mo_occ
    mo_energy = fg.mo_energy[i] if i is not None else fg.mo_energy
    guess = fg.v_ssfno[i] if i is not None else fg.v_ssfno
    mykradc = solver_type(kmf, frozen, mo_coeff, mo_occ, mo_energy)
    mykradc.verbose = 0
    mykradc.method = 'adc(3)'
    e, v, p, x = mykradc.kernel(nroots, guess=guess, kptlist=[0])
    return fg.correct(e, i)[0][0]


class KnownValues(unittest.TestCase):

    r"""Check canonical and additively corrected FNO reference/charged
    energies.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_kadc_fno.KnownValues``. The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.
    """
    def test_canonical(self):
        # Canonical references (no FNO truncation).
        r"""Check untruncated carbon MP2 and MP3 reference energies.

        Compare the per-cell ADC reference correlation energies with stored
        four-decimal values before assessing any FNO approximation.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADCIP(kmf)
        mykradc.verbose = 0
        e_mp2 = mykradc.ground_state_energy()
        mykradc.method = 'adc(3)'
        e_mp3 = mykradc.ground_state_energy()
        self.assertAlmostEqual(e_mp2, -0.1490009409, 4)
        self.assertAlmostEqual(e_mp3, -0.1191959948, 4)

    def test_gs_fno_pct_occ(self):
        r"""Check corrected ADC(3) correlation energy with occupation-fraction
        FNOs.

        Generate a 0.5-fraction ground-density virtual space and compare its
        MP2-corrected ADC(3) contribution with the stored reference.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(_gs_corr(_fno(pct_occ=0.5)), -0.1502090671, 4)

    def test_gs_fno_thresh(self):
        r"""Check corrected ADC(3) correlation energy with a natural-occupation
        cutoff.

        Generate ground-density FNOs with cutoff 0.05 and compare the corrected
        per-cell contribution with the stored reference.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(_gs_corr(_fno(thresh=0.05)), -0.1259622004, 4)

    def test_gs_fno_nvir_act(self):
        r"""Check corrected ADC(3) energy with an explicit retained virtual
        count.

        Keep two virtual natural orbitals and compare the additively corrected
        reference contribution with the stored four-decimal value.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(_gs_corr(_fno(nvir_act=2)), -0.1392170203, 4)

    def test_gs_fno_per_kpt(self):
        r"""Check independent per-k virtual truncation and its energy
        correction.

        Use occupation cutoff 0.05 with mode=per_kpt and compare the corrected
        ADC(3) reference contribution with the stored result.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(
            _gs_corr(_fno(thresh=0.05, mode='per_kpt')), -0.1360134673, 4)

    def test_gs_fno_multithreshold(self):
        r"""Check independent corrections for two FNO occupation-fraction runs.

        Generate 0.5 and 0.9 truncations from the common canonical density,
        select each orbital/correction collection separately, and compare both
        corrected ADC(3) energies with stored references.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        fg = _fno(pct_occ=[0.5, 0.9])
        self.assertAlmostEqual(_gs_corr(fg, 0), -0.1502090671, 4)
        self.assertAlmostEqual(_gs_corr(fg, 1), -0.1259622004, 4)

    def test_ss_fno_ea(self):
        r"""Check the first corrected EA-ADC(3) root in a state-specific FNO
        space.

        Use the first selected sector-zero EA pole to define a 0.9-fraction
        FNO space and compare the corrected ADC(3) excitation with its
        reference.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(
            _es_corr(_fno_es(adc.KRADCEA, [[0], [0]], 0.9), adc.KRADCEA), 1.0069869404, 4)

    def test_sa_fno_ea(self):
        r"""Check corrected EA-ADC(3) with a two-state averaged FNO density.

        Average the first two selected sector-zero EA poles before truncation
        at fraction 0.9, then compare the corrected lowest ADC(3) root.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(
            _es_corr(_fno_es(adc.KRADCEA, [[0, 1], [0]], 0.9), adc.KRADCEA), 0.9951643793, 4)

    def test_ss_fno_ip(self):
        r"""Check the first corrected IP-ADC(3) root in a state-specific FNO
        space.

        Use the first selected sector-zero IP pole for a 0.9-fraction FNO
        space and compare its corrected ADC(3) energy with the stored result.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self.assertAlmostEqual(
            _es_corr(_fno_es(adc.KRADCIP, [[0], [0]], 0.9), adc.KRADCIP), -0.7491959409, 4)


if __name__ == '__main__':
    print('k-point FNO (conventional / SS / SA) tests')
    unittest.main(warnings='ignore')
