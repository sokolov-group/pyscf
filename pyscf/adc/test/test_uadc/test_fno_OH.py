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
# Author: Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
from pyscf import gto
from pyscf import scf
from pyscf import adc
from pyscf.adc.uadc_ea import get_spin_square

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0., 0.)],
        ['H', (0., 0., 0.9697)],]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.spin = 1
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_fno_gs(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0)
        ADCFG.kernel_gs(pct_occ=0.95)

        self.assertAlmostEqual(ADCFG.e_corr_can, -0.1509990493, 6)
        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0233741322, 6)
        self.assertEqual(len(ADCFG.frozen[0]), 6)
        self.assertEqual(len(ADCFG.frozen[1]), 7)

    def test_osfno_gs(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0)
        ADCFG.if_osfno = True
        ADCFG.kernel_gs(pct_occ=0.95)

        self.assertAlmostEqual(ADCFG.delta_e_corr, -0.0240946057, 6)
        self.assertEqual(len(ADCFG.frozen[0]), 6)
        self.assertEqual(len(ADCFG.frozen[1]), 6)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        self.assertEqual(myadc.nvir_b - myadc.nvir_a, 1)

    def test_osfno_ea(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ea')
        ADCFG.if_osfno = True
        ADCFG.kernel(nroots=3, pct_occ=0.90)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy)
        myadc.verbose = 0
        myadc.method = 'adc(3)'
        myadc.method_type = 'ea'
        e,v,p,x = myadc.kernel(nroots=3)
        spin = get_spin_square(myadc._adc_es)[0]

        e = ADCFG.correct(e)
        self.assertAlmostEqual(e[0], 0.03549859, 6)
        self.assertAlmostEqual(e[1], 0.16628554, 6)
        self.assertAlmostEqual(e[2], 0.18655859, 6)

        self.assertAlmostEqual(spin[0], 0.04230651 , 4)
        self.assertAlmostEqual(spin[1], 1.01062603 , 4)
        self.assertAlmostEqual(spin[2], 2.00184363 , 4)

        self.assertEqual(len(ADCFG.frozen[0]), 7)
        self.assertEqual(len(ADCFG.frozen[1]), 7)
        self.assertEqual(myadc.nvir_b - myadc.nvir_a, 1)

if __name__ == "__main__":
    print("FNO/OSFNO calculations for UADC for open-shell OH molecule")
    unittest.main()
