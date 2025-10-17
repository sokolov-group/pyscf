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
# Author: Terrence Stahl <terrencestahl1@@gmail.com>
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

import unittest
import numpy as np
import math
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc, myadc_fr
    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', (0., 0.    , -r/2   )],
        ['N', (0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)
    myadc_fr = adc.ADC(mf,frozen=1)

def tearDownModule():
    global mol, mf, myadc, myadc_fr
    del mol, mf, myadc, myadc_fr

class KnownValues(unittest.TestCase):

    def test_ee_adc2(self):
        myadc.method = "adc(2)"

        myadc.method_type = "ee"
        e,v,p,x = myadc.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3526821493, 6)
        self.assertAlmostEqual(e[1],0.3526821493, 6)
        self.assertAlmostEqual(e[2],0.3834249651, 6)
        self.assertAlmostEqual(e[3],0.4023911887, 6)


    def test_ee_adc2x(self):
        myadc.method = "adc(2)-x"


        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3200995643, 6)
        self.assertAlmostEqual(e[1],0.3200995643, 6)
        self.assertAlmostEqual(e[2],0.3671739857, 6)
        self.assertAlmostEqual(e[3],0.3825795703, 6)


    def test_ee_adc3(self):
        myadc.method = "adc(3)"
        e, t_amp1, t_amp2 = myadc.kernel_gs()

        myadcee = adc.radc_ee.RADCEE(myadc)
        e,v,p,x = myadcee.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3424250025, 6)
        self.assertAlmostEqual(e[1],0.3424250025, 6)
        self.assertAlmostEqual(e[2],0.3534967080, 6)
        self.assertAlmostEqual(e[3],0.3673275757, 6)


    def test_ee_adc3_frozen(self):
        myadc_fr.method = "adc(3)"
        myadc_fr.method_type = "ee"
        e, t_amp1, t_amp2 = myadc_fr.kernel_gs()

        myadcee_fr = adc.radc_ee.RADCEE(myadc_fr)
        e,v,p,x = myadcee_fr.kernel(nroots=4)

        self.assertAlmostEqual(e[0],0.3424857152380730, 6)
        self.assertAlmostEqual(e[1],0.3424857152380766, 6)
        self.assertAlmostEqual(e[2],0.3535001951670751, 6)
        self.assertAlmostEqual(e[3],0.3673334752099558, 6)

if __name__ == "__main__":
    print("EE calculations for different ADC methods for nitrogen molecule")
    unittest.main()
