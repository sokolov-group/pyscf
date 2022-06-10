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
from pyscf import gto
from pyscf import scf
from pyscf import adc

def setUpModule():
    global mol, mf, myadc
    r = 1.098
    mol = gto.Mole()
    mol.atom = [
        ['N', ( 0., 0.    , -r/2   )],
        ['N', ( 0., 0.    ,  r/2)],]
    mol.basis = {'N':'aug-cc-pvdz'}
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    myadc = adc.ADC(mf)

def tearDownModule():
    global mol, mf, myadc
    del mol, mf, myadc

class KnownValues(unittest.TestCase):

    def test_ea_adc2(self):
  
        myadc.higher_excitations = True
        e, t_amp1, t_amp2 = myadc.kernel_gs()
        self.assertAlmostEqual(e, -0.32201692499346535, 6)

        e,v,p,x,es = myadc.ea_adc(nroots=3)
        es.analyze()        

        self.assertAlmostEqual(e[0], 0.0961781923822576, 6)
        self.assertAlmostEqual(e[1], 0.1258326916409743, 6)
        self.assertAlmostEqual(e[2], 0.1380779405750178, 6)

        self.assertAlmostEqual(p[0], 1.9832854445007961, 6)
        self.assertAlmostEqual(p[1], 1.9634368668786559, 6)
        self.assertAlmostEqual(p[2], 1.9783719593912672, 6)

    def test_ea_adc2_oneroot(self):
  
        myadc.higher_excitations = True
        myadc.method_type = "ea"
        e,v,p,x = myadc.kernel()

        self.assertAlmostEqual(e[0], 0.0961781923822576, 6)

        self.assertAlmostEqual(p[0], 1.9832854445007961, 6)

    def test_ea_adc2x(self):
  
        myadc.higher_excitations = True
        myadc.method = "adc(2)-x"
        myadc.method_type = "ea"
        
        e,v,p,x = myadc.kernel(nroots=4)
        e_corr = myadc.e_corr

        self.assertAlmostEqual(e_corr, -0.32201692499346535, 6)

        self.assertAlmostEqual(e[0], 0.0953065329895602, 6)
        self.assertAlmostEqual(e[1], 0.1238833071439568, 6)
        self.assertAlmostEqual(e[2], 0.1365693813556231, 6)
        self.assertAlmostEqual(e[3], 0.1365693813556253, 6)

        self.assertAlmostEqual(p[0],1.9781770712894666, 6)
        self.assertAlmostEqual(p[1],1.9515196916710356, 6)
        self.assertAlmostEqual(p[2],1.9689940350592570, 6)
        self.assertAlmostEqual(p[3],1.9689940350592559, 6)

    def test_ea_adc3(self):
  
        myadc.higher_excitations = True
        myadc.method = "adc(3)"
        myadc.method_type = "ea"
        
        e,v,p,x = myadc.kernel(nroots=3)
        e_corr = myadc.e_corr        

        self.assertAlmostEqual(e_corr, -0.31694173142858517 , 6)

        self.assertAlmostEqual(e[0], 0.0936790850738445, 6)
        self.assertAlmostEqual(e[1], 0.09836545539216629, 6)
        self.assertAlmostEqual(e[2], 0.1295709313652367, 6)

        self.assertAlmostEqual(p[0], 1.8324175318668088, 6)
        self.assertAlmostEqual(p[1], 1.9840991060607487, 6)
        self.assertAlmostEqual(p[2], 1.9638550014980212, 6)
      
if __name__ == "__main__":
    print("EA calculations for different RADC methods for nitrogen molecule")
    unittest.main()
