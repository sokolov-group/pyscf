# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
import unittest
import numpy
from pyscf.pbc import gto
from pyscf.pbc import scf, adc, mp
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell


def setUpModule():
    r"""Prepare matching EA molecular-supercell and k-point references.

    Returns:
        None: Store cell, myadc and kadc as module globals for the two-copy
        helium gth-dzv/gth-pade
        system. The molecular calculation uses a Gamma supercell; the periodic
        driver uses its commensurate Gamma-origin mesh with exxdiv=None.
    """
    global cell, myadc, mykradc
    cell = gto.M(
        unit='B',
        a=[[0., 3.37013733, 3.37013733],
           [3.37013733, 0., 3.37013733],
           [3.37013733, 3.37013733, 0.]],
        mesh=[13] * 3,
        atom='''He 0 0 0
                  He 1.68506866 1.68506866 1.68506866''',
        basis='gth-dzv',
        pseudo='gth-pade',
        verbose=0,
    )

    nmp = [1, 1, 2]

    # A single Gamma point describes the commensurate supercell.
    supcell = super_cell(cell, nmp)
    mf = scf.RHF(supcell, exxdiv=None).density_fit()
    ehf = mf.kernel()
    myadc = mol_adc.RADC(mf)
    myadc.method_type = 'ea'
    myadc.approx_trans_moments = True

    # Use the primitive cell with a Gamma-centered mesh.
    kpts = cell.make_kpts(nmp)
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
    mykradc = adc.KRADCEA(kmf)


def tearDownModule():
    r"""Release the paired molecular and periodic reference fixtures.

    Returns:
        None: Delete cell, myadc and kadc so reference/scratch resources can
        be reclaimed after all comparisons.
    """
    global cell, myadc, mykradc
    del cell, myadc, mykradc


class KnownValues(unittest.TestCase):
    r"""Compare EA spectra from commensurate supercell and k-point ADC.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_k_vs_supercell_ea.KnownValues``.
    The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.

    Methods:
        test_ea_adc2_supercell: Check the molecular-supercell EA-ADC(2)
            reference spectrum.
        test_ea_adc2_supercell_vs_k: Compare EA-ADC2 between supercell and
            sector-zero k ADC.
        test_ea_adc2x_supercell_vs_k: Compare EA-ADC2X between supercell and
            sector-zero k ADC.
        test_ea_adc3_supercell_vs_k: Compare EA-ADC3 between supercell and
            sector-zero k ADC.
    """
    def test_ea_adc2_supercell(self):
        r"""Check the molecular-supercell EA-ADC(2) reference spectrum.

        Compare selected supercell energies and spectroscopic factors with
        stored six-decimal values before using that reference for k
        comparisons.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        e1, v1, p1, x1 = myadc.kernel(nroots=4)
        self.assertAlmostEqual(e1[1], 0.83425538, 6)
        self.assertAlmostEqual(e1[3], 1.29595017, 6)

        self.assertAlmostEqual(p1[0], 1.98185661, 6)
        self.assertAlmostEqual(p1[1], 1.96030165, 6)
        self.assertAlmostEqual(p1[3], 0.00398214, 6)

    def test_ea_adc2_supercell_vs_k(self):
        r"""Compare EA-ADC2 between supercell and sector-zero k ADC.

        Use matched states in the two-copy helium gth-dzv/gth-pade
        system and assert agreement of their energy and spectroscopic factor.
        The test uses the precision stated by its explicit numerical
        assertions.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        e1, v1, p1, x1 = myadc.kernel(nroots=5)
        e2, v2, p2, x2 = mykradc.kernel(nroots=5, kptlist=[0])

        ediff = e1[1] - e2[0][0]
        pdiff = p1[1] - p2[0][0]

        self.assertAlmostEqual(ediff, 0.0, 4)
        self.assertAlmostEqual(pdiff, 0.0, 4)

    def test_ea_adc2x_supercell_vs_k(self):
        r"""Compare EA-ADC2X between supercell and sector-zero k ADC.

        Use matched states in the two-copy helium gth-dzv/gth-pade
        system and assert agreement of their energy and spectroscopic factor.
        The test uses the precision stated by its explicit numerical
        assertions.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        myadc.method = 'adc(2)-x'
        e1, v1, p1, x1 = myadc.kernel(nroots=3)

        mykradc.method = 'adc(2)-x'
        e2, v2, p2, x2 = mykradc.kernel(nroots=3, kptlist=[0])

        ediff = e1[1] - e2[0][0]
        pdiff = p1[1] - p2[0][0]

        self.assertAlmostEqual(ediff, 0.0, 4)
        self.assertAlmostEqual(pdiff, 0.0, 4)

    def test_ea_adc3_supercell_vs_k(self):
        r"""Compare EA-ADC3 between supercell and sector-zero k ADC.

        Use matched states in the two-copy helium gth-dzv/gth-pade
        system and assert agreement of their energy and spectroscopic factor.
        The test uses the precision stated by its explicit numerical
        assertions.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        myadc.method = 'adc(3)'
        e1, v1, p1, x1 = myadc.kernel(nroots=3)

        mykradc.method = 'adc(3)'
        e2, v2, p2, x2 = mykradc.kernel(nroots=3, kptlist=[0])

        ediff = e1[1] - e2[0][0]
        pdiff = p1[1] - p2[0][0]

        self.assertAlmostEqual(ediff, 0.0, 3)
        self.assertAlmostEqual(pdiff, 0.0, 3)


if __name__ == "__main__":
    print('Supercell vs k-point calculations for EA-ADC methods')
    unittest.main()
