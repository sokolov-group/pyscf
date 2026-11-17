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
from pyscf.pbc import scf,adc,mp
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell

def setUpModule():
    r"""Prepare matching EA molecular-supercell and k-point references.

    Returns:
        None: Store cell, myadc and kadc as module globals for the
        three-copy carbon gth-dzv/gth-pade system. The molecular calculation
        uses a Gamma supercell; the periodic
        driver uses its commensurate Gamma-origin mesh with exxdiv=None.
    """
    global cell, myadc, mykradc
    cell = gto.Cell()
    cell.build(
        a='''
            0.000000     1.783500     1.783500
            1.783500     0.000000     1.783500
            1.783500     1.783500     0.000000
        ''',
        atom='C 1.337625 1.337625 1.337625; C 2.229375 2.229375 2.229375',
        verbose=5,
        output='/dev/null',
        basis='gth-dzv',
        pseudo='gth-pade',
    )

    nmp = [1,1,3]

    # treating supercell at gamma point
    supcell = super_cell(cell, nmp)
    mf  = scf.RHF(supcell, exxdiv=None).density_fit()
    ehf  = mf.kernel()
    myadc = mol_adc.RADC(mf)
    myadc.method_type = 'ea'

    # periodic calculation at gamma point
    kpts = cell.make_kpts(nmp)
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
    mykradc  = adc.KRADCEA(kmf)

def tearDownModule():
    r"""Release the paired molecular and periodic reference fixtures.

    Returns:
        None: Close cell.stdout and delete cell, myadc and kadc so
        reference/scratch resources can
        be reclaimed after all comparisons.
    """
    global cell, myadc, mykradc
    cell.stdout.close()
    del cell, myadc, mykradc

class KnownValues(unittest.TestCase):

    r"""Compare EA spectra from commensurate supercell and k-point ADC.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest
    test.test_k_vs_supercell_ea_highcost.KnownValues``. The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.
    """
    def test_ea_adc2_supercell_vs_k_high_cost(self):
        r"""Compare EA-ADC2 between supercell and sector-zero k ADC.

        Use matched states in the three-copy carbon gth-dzv/gth-pade
        system and assert agreement of their energy.
        The test uses the precision stated by its explicit numerical
        assertions.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        myadc.approx_trans_moments = True
        e1,v1,p1,x1 = myadc.kernel(nroots=3)
        e2, v2, p2, x2 = mykradc.kernel(nroots=3, kptlist=[0])

        ediff = e1[0] - e2[0][0]
        self.assertAlmostEqual(ediff, 0.0, 2)

    def test_ea_adc2x_supercell_vs_k_high_cost(self):

        r"""Compare EA-ADC2X between supercell and sector-zero k ADC.

        Use matched states in the three-copy carbon gth-dzv/gth-pade
        system and assert agreement of their energy.
        The test uses the precision stated by its explicit numerical
        assertions.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        myadc.approx_trans_moments = True
        myadc.method = 'adc(2)-x'
        e1,v1,p1,x1 = myadc.kernel(nroots=3)

        mykradc.method = 'adc(2)-x'
        e2, v2, p2, x2 = mykradc.kernel(nroots=3, kptlist=[0])

        ediff = e1[0] - e2[0][0]
        self.assertAlmostEqual(ediff, 0.0, 2)

    def test_ea_adc3_supercell_vs_k_high_cost(self):

        r"""Compare EA-ADC3 between supercell and sector-zero k ADC.

        Use matched states in the three-copy carbon gth-dzv/gth-pade
        system and assert agreement of their energy.
        The test uses the precision stated by its explicit numerical
        assertions.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        myadc.approx_trans_moments = True
        myadc.method = 'adc(3)'
        e1,v1,p1,x1 = myadc.kernel(nroots=1)

        mykradc.method = 'adc(3)'
        e2, v2, p2, x2 = mykradc.kernel(nroots=1, kptlist=[0])

        ediff = e1[0] - e2[0][0]
        self.assertAlmostEqual(ediff, 0.0, 2)

if __name__ == "__main__":
    print('supercell vs k-point calculations for EA-ADC methods')
    unittest.main()
