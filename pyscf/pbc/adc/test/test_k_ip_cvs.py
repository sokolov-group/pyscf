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
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc import scf,adc,mp
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell

def setUpModule():
    global cell, kpts, kadc
    cell = gto.Cell()
    cell.a = np.ones((3,3)) - np.eye(3)
    cell.a *= 1.7835
    cell.atom = [['C', [0., 0., 0.]], ['C', [0.89175, 0.89175, 0.89175]]]  
    cell.basis = 'cc-pvdz'
    cell.build() 
    #nmp = [1,1,2]

    ## periodic calculation at gamma point
    #kpts = cell.make_kpts((nmp))
    #kpts -= kpts[0]
    #kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
    #kadc  = adc.KRADC(kmf)

def tearDownModule():
    global cell, kadc
    del cell, kadc

class KnownValues(unittest.TestCase):

    def test_ip_adc2_k(self):

        nmp = [2,2,2]
        kpts = cell.make_kpts((nmp))
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
        kadc  = adc.KRADC(kmf)
        kadc.ncvs_proj = 2
        kadc.approx_trans_moments = True
        e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

        self.assertAlmostEqual(e[0][0], 264.03604823, 4)
        self.assertAlmostEqual(e[0][1], 264.11439854, 4)
        self.assertAlmostEqual(e[0][2], 284.41577439, 4)

        self.assertAlmostEqual(p[0][0], 1.51206280, 4)
        self.assertAlmostEqual(p[0][1], 1.51144873, 4)
        self.assertAlmostEqual(p[0][2], 0.00000041, 4)

    #def test_ip_adc2x_k_high_cost(self):

    #    nmp = [2,2,2]
    #    kpts = cell.make_kpts((nmp))
    #    kpts -= kpts[0]
    #    kmf = scf.KRHF(cell, kpts,exxdiv=None).density_fit().run()
    #    kadc  = adc.KRADC(kmf)
    #    kadc.ncvs = 2
    #    kadc.approx_trans_moments = True
    #    kadc.method = 'adc(2)-x'
    #    e, v, p, x = kadc.kernel(nroots=3,kptlist=[0])

    #    self.assertAlmostEqual(e[0][0], 264.57744394, 4)
    #    self.assertAlmostEqual(e[0][1], 264.65733337, 4)
    #    self.assertAlmostEqual(e[0][2], 279.59392024, 4)

    #    self.assertAlmostEqual(p[0][0], 1.53670841, 4)
    #    self.assertAlmostEqual(p[0][1], 1.53594853, 4)
    #    self.assertAlmostEqual(p[0][2], 0.00000164, 4)

if __name__ == "__main__":
    print("k-point calculations for IP-ADC methods")
    unittest.main()
