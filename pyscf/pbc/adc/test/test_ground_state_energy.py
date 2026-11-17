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

class KnownValues(unittest.TestCase):
    r"""Check per-cell MP2/MP3 reference energies for periodic helium.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_ground_state_energy.KnownValues``.
    The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.
    """
    def test_check_mol_mp2(self):
        # treating supercell at gamma point
        r"""Check molecular supercell MP2 energy after primitive-cell
        normalization.

        Construct a two-copy Gamma supercell, obtain molecular ADC reference
        amplitudes, divide its correlation contribution by the copy count, and
        compare with the stored helium reference to four decimal places.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        supcell = super_cell(cell, nmp)
        mf  = scf.RHF(supcell, exxdiv=None).density_fit()
        ehf  = mf.kernel()
        myadc  = mol_adc.ADC(mf)
        e_mp, t1, t2 = myadc.kernel_gs()
        e_mp = e_mp/(numpy.prod(nmp))

        self.assertAlmostEqual(e_mp, -0.02095766698, 4)

    def test_check_periodic_mp2(self):
        r"""Check the k-point ADC(2) reference correlation energy.

        Use the two-k helium mesh and compare the scalar per-cell Hartree
        result
        with the same stored MP2 reference used by the supercell calculation.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
        mykradc  = adc.KRADCBase(kmf)
        e_mp = mykradc.ground_state_energy()
        t1 = mykradc.t1
        t2 = mykradc.t2

        self.assertAlmostEqual(e_mp, -0.02095766698, 4)

    def test_check_periodic_mp2_2_high_cost(self):
        r"""Compare ADC reference MP2 with KMP2 on an eight-k-point mesh.

        Construct a 2x2x2 KRHF reference and assert agreement of the
        independently
        computed per-cell correlation energies to four decimal places.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        nmp = [2,2,2]
        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
        mykradc  = adc.KRADCBase(kmf)
        e_mp2_1 = mykradc.ground_state_energy()
        t1 = mykradc.t1
        t2 = mykradc.t2

        mp2 = mp.KMP2(kmf)
        e_mp2_2 = mp2.kernel()[0]

        diff_mp2 = e_mp2_2 - e_mp2_1

        self.assertAlmostEqual(diff_mp2, 0.0, 4)

    def test_check_periodic_mp3(self):
        r"""Check the ADC(3) reference MP3 correlation contribution.

        Use the two-k helium reference and compare the per-cell Hartree result
        with the stored MP3 value to four decimal places.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit().run()
        mykradc  = adc.KRADCBase(kmf)
        mykradc.method = 'adc(3)'
        e_mp = mykradc.ground_state_energy()
        t1 = mykradc.t1
        t2 = mykradc.t2

        self.assertAlmostEqual(e_mp, -0.0208366684841, 4)

if __name__ == "__main__":
    print('Ground state calculations for helium')
    unittest.main()
