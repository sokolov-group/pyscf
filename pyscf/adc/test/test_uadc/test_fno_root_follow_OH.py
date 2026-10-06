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

    def test_ssfno_ip_root_following(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ip', ref_state=1,
                                    conv_tol=1e-10, tol_residual=1e-7)
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, thresh=1e-4)

        self.assertAlmostEqual(ADCFG.e_can[0], 0.4276327175, 6)
        self.assertAlmostEqual(ADCFG.e_can[1], 0.4686705615, 6)
        self.assertAlmostEqual(ADCFG.e_can[2], 0.5765914247, 6)

        self.assertAlmostEqual(ADCFG.e_ssfno[0], 0.4276528437, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[1], 0.4687378409, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[2], 0.5764764614, 6)

        for de in ADCFG.delta_e:
            self.assertLess(abs(de), 1e-3)

        for wl in ADCFG.w_guess_lost:
            self.assertLess(wl, 1e-4)

        for i in range(3):
            self.assertGreater(ADCFG.ovl_guess[i, i], 0.95)
            for j in range(3):
                if i != j:
                    self.assertLess(ADCFG.ovl_guess[i, j], 0.05)

        self.assertEqual(len(ADCFG.frozen[0]), 1)
        self.assertEqual(len(ADCFG.frozen[1]), 1)

        # truncated ADC(2) eigenvectors, root following kept enabled
        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy).set(verbose=0, method='adc(3)',
                                              method_type='ip',
                                              conv_tol=1e-10, tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        self.assertAlmostEqual(ADCFG.correct(e)[0], 0.4569445210, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[1], 0.4680543340, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[2], 0.6013935227, 6)

    def test_osfno_ee_root_following(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ee', ref_state=1,
                                    conv_tol=1e-10, tol_residual=1e-7)
        ADCFG.if_osfno = True
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, pct_occ=0.90)

        self.assertAlmostEqual(ADCFG.e_can[0], 0.0023521304, 6)
        self.assertAlmostEqual(ADCFG.e_can[1], 0.1648082469, 6)
        self.assertAlmostEqual(ADCFG.e_can[2], 0.2987782521, 6)

        self.assertAlmostEqual(ADCFG.e_ssfno[0], 0.008787034566775532, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[1], 0.16854423075480343, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[2], 0.6779518685742937, 6)

        for i in range(3):
            self.assertGreater(ADCFG.ovl_guess[i, i], 0.9)
            for j in range(3):
                if i != j:
                    self.assertLess(ADCFG.ovl_guess[i, j], 0.05)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy).set(verbose=0, method='adc(3)',
                                              method_type='ee',
                                              conv_tol=1e-10, tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        self.assertAlmostEqual(ADCFG.correct(e)[0], -0.0002714981365512228, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[1], 0.15855671456041287, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[2], 0.25143055181704, 6)

    def test_safno_ee_root_following(self):
        ADCFG = adc.ADC2FNO(mf).set(verbose=0, method_type='ee', ref_state=[1, 2],
                                    conv_tol=1e-10, tol_residual=1e-7)
        ADCFG.trans_guess = True
        ADCFG.pick = True
        ADCFG.kernel(nroots=3, thresh=1e-4)

        self.assertAlmostEqual(ADCFG.e_can[0], 0.0023521304, 6)
        self.assertAlmostEqual(ADCFG.e_can[1], 0.1648082469, 6)
        self.assertAlmostEqual(ADCFG.e_can[2], 0.2987782521, 6)

        self.assertAlmostEqual(ADCFG.e_ssfno[0], 0.0023530214, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[1], 0.1647682275, 6)
        self.assertAlmostEqual(ADCFG.e_ssfno[2], 0.4836588090, 6)

        for i in range(3):
            self.assertGreater(ADCFG.ovl_guess[i, i], 0.9)

        myadc = adc.UADC(mf, ADCFG.frozen, ADCFG.mo_coeff, ADCFG.mo_occ,
                         ADCFG.mo_energy).set(verbose=0, method='adc(3)',
                                              method_type='ee',
                                              conv_tol=1e-10, tol_residual=1e-7)
        myadc.pick = True
        e, v, p, x = myadc.kernel(nroots=3, guess=ADCFG.v_ssfno)

        self.assertAlmostEqual(ADCFG.correct(e)[0], -0.0018486943, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[1], 0.1573727720, 6)
        self.assertAlmostEqual(ADCFG.correct(e)[2], 0.2710617763, 6)

if __name__ == "__main__":
    print("FNO/OSFNO/SS-FNO calculations with character-based root following")
    unittest.main()
