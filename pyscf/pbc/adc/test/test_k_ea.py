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
    r"""Prepare the two-k helium KRHF reference for EA-ADC regression values.

    Returns:
        None: Store cell and the restricted periodic ADC driver as module
        globals. Use gth-dzv/gth-pade, no exchange-divergence correction and
        a 1x1x2 Gamma-origin k mesh.
    """
    global cell, mykradc
    cell = gto.M(
        unit='B',
        a=[[0.,          3.37013733,  3.37013733],
             [3.37013733,  0.,          3.37013733],
             [3.37013733,  3.37013733,  0.        ]],
        mesh=[13]*3,
        atom='''He 0 0 0
                  He 1.68506866 1.68506866 1.68506866''',
        basis='gth-dzv',
        pseudo='gth-pade',
        verbose=0,
    )

    nmp = [1,1,2]

    # periodic calculation at gamma point
    kpts = cell.make_kpts(nmp)
    kpts -= kpts[0]
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
    mykradc  = adc.KRADCEA(kmf)

def tearDownModule():
    r"""Release shared helium cell and charged ADC fixture references.

    Returns:
        None: Delete the module globals after all charged-method checks,
        allowing owned amplitude scratch storage to be reclaimed.
    """
    global cell, mykradc
    del cell, mykradc

class KnownValues(unittest.TestCase):

    r"""Check stored helium EA energies and spectroscopic factors.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_k_ea.KnownValues``. The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.
    """
    def test_ea_adc2_k(self):

        r"""Check three EA-ADC(2) roots and their pole strengths.

        Solve sector zero on the two-k helium reference and compare energies
        and spin-summed spectroscopic factors with stored four-decimal values.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        e, v, p, x = mykradc.kernel(nroots=3, kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.83425538, 4)
        self.assertAlmostEqual(e[0][1], 1.29595017, 4)
        self.assertAlmostEqual(e[0][2], 1.68125009, 4)

        self.assertAlmostEqual(p[0][0], 1.96030215, 4)
        self.assertAlmostEqual(p[0][1], 0.00398257, 4)
        self.assertAlmostEqual(p[0][2], 2.99e-06, 4)

    def test_ea_adc2x_k_high_cost(self):

        r"""Check EA-ADC(2)-x roots on a denser eight-k mesh.

        Build a separate 2x2x2 reference, solve three roots at sector zero and
        compare their energies and spectroscopic factors with stored values.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        nmp = [2,2,2]
        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
        mykradc  = adc.KRADCEA(kmf)
        mykradc.method = 'adc(2)-x'
        e, v, p, x = mykradc.kernel(nroots=3, kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.82483536, 4)
        self.assertAlmostEqual(e[0][1], 1.38987893, 4)
        self.assertAlmostEqual(e[0][2], 1.38987895, 4)

        self.assertAlmostEqual(p[0][0], 1.95209795, 4)
        self.assertAlmostEqual(p[0][1], 1e-08, 4)
        self.assertAlmostEqual(p[0][2], 2e-08, 4)

    def test_ea_adc3_k_skip(self):

        r"""Check three EA-ADC(3) roots and pole strengths.

        Solve the two-k reference at sector zero and compare with stored
        four-decimal results. The historical method name does not change the
        unittest execution or the selected ADC approximation.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc.method = 'adc(3)'
        e, v, p, x = mykradc.kernel(nroots=3, kptlist=[0])

        self.assertAlmostEqual(e[0][0], 0.83396073, 4)
        self.assertAlmostEqual(e[0][1], 1.26993734, 4)
        self.assertAlmostEqual(e[0][2], 1.56058118, 4)

        self.assertAlmostEqual(p[0][0], 1.95976365, 4)
        self.assertAlmostEqual(p[0][1], 0.0011169, 4)
        self.assertAlmostEqual(p[0][2], 0.00385444, 4)

if __name__ == "__main__":
    print('k-point calculations for EA-ADC methods')
    unittest.main()
