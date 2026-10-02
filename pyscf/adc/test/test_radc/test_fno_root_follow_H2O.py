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
from pyscf.adc import radc_fno

def setUpModule():
    global mol, mf
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0., 0.)],
        ['H', (0., 0., 0.96)],
        ['H', (0., 0., -0.96)],]
    mol.basis = 'cc-pvdz'
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

class KnownValues(unittest.TestCase):

    def test_ssfno_ee_root_following(self):
        ADCFG = radc_fno.RADC2FNO(mf).set(verbose=0, method_type='ee', ref_state=2,
                                         conv_tol=1e-10, tol_residual=1e-7)
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, pct_occ=0.99)

        self.assertAlmostEqual(ADCFG.e_can[0], 0.2637299106, 6)
        self.assertAlmostEqual(ADCFG.e_can[1], 0.2637299106, 6)
        self.assertAlmostEqual(ADCFG.e_can[2], 0.3569181782, 6)

        self.assertAlmostEqual(ADCFG.e_ssfno[0], 0.2657581507, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[1], 0.2640681164, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[2], 0.7426922421, 6)

        # the off-diagonal within the degenerate pair is arbitrary
        for i in range(3):
            self.assertGreater(ADCFG.ovl_guess[i, i], 0.95)
            for j in range(3):
                if i != j:
                    self.assertLess(ADCFG.ovl_guess[i, j], 0.1)

        self.assertEqual(len(ADCFG.frozen), 10)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff,
                         mo_energy=ADCFG.mo_energy).set(verbose=0,
                                                        method='adc(3)',
                                                        method_type='ee',
                                                        conv_tol=1e-10,
                                                        tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        self.assertAlmostEqual(ADCFG.correct(e)[0], 0.2698798613, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[1], 0.2700696086, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[2], 0.3337693460, 6)

    def test_fno_ip_root_following(self):
        ADCFG = radc_fno.RADC2FNO(mf).set(verbose=0, method_type='ip',
                                          conv_tol=1e-10, tol_residual=1e-7)
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, thresh=1e-4)

        self.assertAlmostEqual(ADCFG.e_can[0], 0.3817434909, 6)
        self.assertAlmostEqual(ADCFG.e_can[1], 0.3817434909, 6)
        self.assertAlmostEqual(ADCFG.e_can[2], 0.7158407913, 6)

        self.assertAlmostEqual(ADCFG.e_ssfno[0], 0.3832657316, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[1], 0.3832657316, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[2], 0.7168995234, 6)

        for i in range(3):
            self.assertGreater(ADCFG.ovl_guess[i, i], 0.95)
            for j in range(3):
                if i != j:
                    self.assertLess(ADCFG.ovl_guess[i, j], 0.05)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff,
                         mo_energy=ADCFG.mo_energy).set(verbose=0,
                                                        method='adc(3)',
                                                        method_type='ip',
                                                        conv_tol=1e-10,
                                                        tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        self.assertAlmostEqual(ADCFG.correct(e)[0], 0.4237115288, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[1], 0.4237115288, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[2], 0.7437905527, 6)

        ADCFG = radc_fno.RADC2FNO(mf).set(verbose=0, method_type='ip',
                                          ref_state=1, conv_tol=1e-10,
                                          tol_residual=1e-7)
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, thresh=1e-4)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff,
                         mo_energy=ADCFG.mo_energy).set(verbose=0,
                                                        method='adc(3)',
                                                        method_type='ip',
                                                        conv_tol=1e-10,
                                                        tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        self.assertAlmostEqual(ADCFG.correct(e)[0], 0.4250042457, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[1], 0.4250046184, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[2], 0.7448131806, 6)

    def test_safno_ee_root_following(self):
        ADCFG = radc_fno.RADC2FNO(mf).set(verbose=0, method_type='ee',
                                         ref_state=[1, 2], conv_tol=1e-10,
                                         tol_residual=1e-7)
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, pct_occ=0.99)

        self.assertAlmostEqual(ADCFG.e_can[0], 0.2637299106, 6)
        self.assertAlmostEqual(ADCFG.e_can[1], 0.2637299106, 6)
        self.assertAlmostEqual(ADCFG.e_can[2], 0.3569181782, 6)

        e_pair = 0.5 * (ADCFG.e_ssfno[0] + ADCFG.e_ssfno[1])
        self.assertAlmostEqual(e_pair, 0.264480, 5)
        self.assertLess(abs(ADCFG.e_ssfno[0] - ADCFG.e_ssfno[1]), 1e-3)
        self.assertAlmostEqual(ADCFG.e_ssfno[2], 0.742671, 5)

        for i in range(3):
            self.assertGreater(ADCFG.ovl_guess[i, i], 0.6)
        for i in range(2):
            self.assertGreater(ADCFG.ovl_guess[i, 0]**2 + ADCFG.ovl_guess[i, 1]**2,
                               0.9)
        self.assertEqual(len(ADCFG.frozen), 10)

        myadc = adc.RADC(mf, ADCFG.frozen, ADCFG.mo_coeff,
                         mo_energy=ADCFG.mo_energy).set(verbose=0,
                                                        method='adc(3)',
                                                        method_type='ee',
                                                        conv_tol=1e-10,
                                                        tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        ec = ADCFG.correct(e)
        self.assertAlmostEqual(0.5 * (ec[0] + ec[1]), 0.2699681, 6)
        self.assertLess(abs(ec[0] - ec[1]), 1e-3)
        self.assertAlmostEqual(ec[2], 0.333775, 5)

if __name__ == "__main__":
    print("FNO/SS-FNO calculations with character-based root following")
    unittest.main()
