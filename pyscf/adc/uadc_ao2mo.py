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

import numpy as np
import pyscf.ao2mo as ao2mo
from pyscf import lib
from pyscf.lib import logger
from pyscf.adc import radc_ao2mo
import time
import tempfile

### Integral transformation for integrals in Chemists' notation###
#@profile
def transform_integrals_incore(myadc):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]

    nocc_a = occ_a.shape[1]
    nocc_b = occ_b.shape[1]
    nvir_a = vir_a.shape[1]
    nvir_b = vir_b.shape[1]
    ind_vv_g = np.tril_indices(nvir_a, k=-1)
    ind_VV_g = np.tril_indices(nvir_b, k=-1)

    # Number of CVS orbital (it is assumed that the number of ionized core alpha electrons equals the number of ionized core beta electrons)
    ncvs = ncvs = ncvs = myadc.ncvs
    # Slices of occupied MO coeffcients needed for forming CVS integrals
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs
    core_a = occ_a[:,:ncvs]
    core_b = occ_b[:,:ncvs]
    val_a = occ_a[:,ncvs:]
    val_b = occ_b[:,ncvs:]    
   
    eris = lambda:None

    # TODO: check if myadc._scf._eri is not None
    eris.oooo = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_a, occ_a), compact=False).reshape(nocc_a, nocc_a, nocc_a, nocc_a).copy()
    eris.ovoo = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_a, occ_a), compact=False).reshape(nocc_a, nvir_a, nocc_a, nocc_a).copy()
    eris.ovvo = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_a, occ_a), compact=False).reshape(nocc_a, nvir_a, nvir_a, nocc_a).copy()
    eris.oovv = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, vir_a, vir_a), compact=False).reshape(nocc_a, nocc_a, nvir_a, nvir_a).copy()
    eris.ovvv = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_a, vir_a), compact=True).reshape(nocc_a, nvir_a, -1).copy()

    eris.OOOO = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, occ_b, occ_b), compact=False).reshape(nocc_b, nocc_b, nocc_b, nocc_b).copy()
    eris.OVOO = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_b, occ_b), compact=False).reshape(nocc_b, nvir_b, nocc_b, nocc_b).copy()
    eris.OOVV = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, vir_b, vir_b), compact=False).reshape(nocc_b, nocc_b, nvir_b, nvir_b).copy()
    eris.OVVO = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_b, occ_b), compact=False).reshape(nocc_b, nvir_b, nvir_b, nocc_b).copy()
    eris.OVVV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_b, vir_b), compact=True).reshape(nocc_b, nvir_b, -1).copy()

    eris.ooOO = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_b, occ_b), compact=False).reshape(nocc_a, nocc_a, nocc_b, nocc_b).copy()
    eris.ovOO = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, occ_b, occ_b), compact=False).reshape(nocc_a, nvir_a, nocc_b, nocc_b).copy()
    eris.ooVV = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, vir_b, vir_b), compact=False).reshape(nocc_a, nocc_a, nvir_b, nvir_b).copy()
    eris.ovVO = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, occ_b), compact=False).reshape(nocc_a, nvir_a, nvir_b, nocc_b).copy()
    eris.ovVV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, vir_b), compact=True).reshape(nocc_a, nvir_a, -1).copy()

    eris.OVoo = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, occ_a, occ_a), compact=False).reshape(nocc_b, nvir_b, nocc_a, nocc_a).copy()
    eris.OOvv = ao2mo.general(myadc._scf._eri, (occ_b, occ_b, vir_a, vir_a), compact=False).reshape(nocc_b, nocc_b, nvir_a, nvir_a).copy()
    eris.OVvo = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_a, occ_a), compact=False).reshape(nocc_b, nvir_b, nvir_a, nocc_a).copy()
    eris.OVvv = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, vir_a, vir_a), compact=True).reshape(nocc_b, nvir_b, -1).copy()

    # CVS integrals for matvec function (c: core, e: external, v: valence, o: all occupied orbitals)
    #if myadc.method_type == 'midnight_testing':#'ip-cvs':
    if myadc.method_type == 'ip-cvs':
    
        #----- ADC(2) integrals --------
        eris.cecc = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_a, core_a), compact=False).reshape(ncvs, nvir_a, ncvs, ncvs).copy()
        eris.vecc = ao2mo.general(myadc._scf._eri, (val_a, vir_a, core_a, core_a), compact=False).reshape(nval_a, nvir_a, ncvs, ncvs).copy()
        eris.CECC = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_b, core_b), compact=False).reshape(ncvs, nvir_b, ncvs, ncvs).copy()
        eris.VECC = ao2mo.general(myadc._scf._eri, (val_b, vir_b, core_b, core_b), compact=False).reshape(nval_b, nvir_b, ncvs, ncvs).copy()
        eris.ceCC = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_b, core_b), compact=False).reshape(ncvs, nvir_a, ncvs, ncvs).copy()
        eris.veCC = ao2mo.general(myadc._scf._eri, (val_a, vir_a, core_b, core_b), compact=False).reshape(nval_a, nvir_a, ncvs, ncvs).copy()
        eris.CEcc = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_a, core_a), compact=False).reshape(ncvs, nvir_b, ncvs, ncvs).copy()
        eris.VEcc = ao2mo.general(myadc._scf._eri, (val_b, vir_b, core_a, core_a), compact=False).reshape(nval_b, nvir_b, ncvs, ncvs).copy()

        eris.cecv = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_a, val_a), compact=False).reshape(ncvs, nvir_a, ncvs, nval_a).copy()
        eris.CECV = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_b, val_b), compact=False).reshape(ncvs, nvir_b, ncvs, nval_b).copy()
        eris.ceCV = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_b, val_b), compact=False).reshape(ncvs, nvir_a, ncvs, nval_b).copy()
        eris.CEcv = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_a, val_a), compact=False).reshape(ncvs, nvir_b, ncvs, nval_a).copy()
        #----- ADC(2)-x integrals --------
        eris.cccc = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_a, core_a), compact=False).reshape(ncvs, ncvs, ncvs, ncvs).copy()
        eris.cccv = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_a, val_a), compact=False).reshape(ncvs, ncvs, ncvs, nval_a).copy()
        eris.ccvv = ao2mo.general(myadc._scf._eri, (core_a, core_a, val_a, val_a), compact=False).reshape(ncvs, ncvs, nval_a, nval_a).copy()
        eris.CCCC = ao2mo.general(myadc._scf._eri, (core_b, core_b, core_b, core_b), compact=False).reshape(ncvs, ncvs, ncvs, ncvs).copy()
        eris.CCCV = ao2mo.general(myadc._scf._eri, (core_b, core_b, core_b, val_b), compact=False).reshape(ncvs, ncvs, ncvs, nval_b).copy()
        eris.CCVV = ao2mo.general(myadc._scf._eri, (core_b, core_b, val_b, val_b), compact=False).reshape(ncvs, ncvs, nval_b, nval_b).copy()
        eris.ccCC = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_b, core_b), compact=False).reshape(ncvs, ncvs, ncvs, ncvs).copy()
        eris.ccCV = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_b, val_b), compact=False).reshape(ncvs, ncvs, ncvs, nval_b).copy()
        eris.vvCC = ao2mo.general(myadc._scf._eri, (val_a, val_a, core_b, core_b), compact=False).reshape(nval_a, nval_a, ncvs, ncvs).copy()
        eris.ccVV = ao2mo.general(myadc._scf._eri, (core_a, core_a, val_b, val_b), compact=False).reshape(ncvs, ncvs, nval_b, nval_b).copy()
        eris.ccee = ao2mo.general(myadc._scf._eri, (core_a, core_a, vir_a, vir_a), compact=False).reshape(ncvs, ncvs, nvir_a, nvir_a).copy()
        eris.vvee = ao2mo.general(myadc._scf._eri, (val_a, val_a, vir_a, vir_a), compact=False).reshape(nval_a, nval_a, nvir_a, nvir_a).copy()
        eris.CCEE = ao2mo.general(myadc._scf._eri, (core_b, core_b, vir_b, vir_b), compact=False).reshape(ncvs, ncvs, nvir_b, nvir_b).copy()
        eris.VVEE = ao2mo.general(myadc._scf._eri, (val_b, val_b, vir_b, vir_b), compact=False).reshape(nval_b, nval_b, nvir_b, nvir_b).copy()
        eris.ccEE = ao2mo.general(myadc._scf._eri, (core_a, core_a, vir_b, vir_b), compact=False).reshape(ncvs, ncvs, nvir_b, nvir_b).copy()
        eris.vvEE = ao2mo.general(myadc._scf._eri, (val_a, val_a, vir_b, vir_b), compact=False).reshape(nval_a, nval_a, nvir_b, nvir_b).copy()
        eris.CCee = ao2mo.general(myadc._scf._eri, (core_b, core_b, vir_a, vir_a), compact=False).reshape(ncvs, ncvs, nvir_a, nvir_a).copy()
        eris.VVee = ao2mo.general(myadc._scf._eri, (val_b, val_b, vir_a, vir_a), compact=False).reshape(nval_b, nval_b, nvir_a, nvir_a).copy()

        eris.cvcv = ao2mo.general(myadc._scf._eri, (core_a, val_a , core_a, val_a),compact=False).reshape((ncvs, nval_a , ncvs, nval_a)).copy()
        eris.CVCV = ao2mo.general(myadc._scf._eri, (core_b, val_b , core_b, val_b),compact=False).reshape((ncvs, nval_b , ncvs, nval_b)).copy()
        eris.cvCC = ao2mo.general(myadc._scf._eri, (core_a, val_a , core_b, core_b),compact=False).reshape((ncvs, nval_a , ncvs,  ncvs)).copy()
        eris.cvCV = ao2mo.general(myadc._scf._eri, (core_a, val_a,  core_b, val_b),compact=False).reshape((ncvs, nval_a,  ncvs, nval_b)).copy()
        eris.cece = ao2mo.general(myadc._scf._eri, (core_a, vir_a , core_a, vir_a),compact=False).reshape((ncvs, nvir_a , ncvs, nvir_a)).copy()
        eris.vece = ao2mo.general(myadc._scf._eri, (val_a,  vir_a , core_a, vir_a),compact=False).reshape((nval_a, nvir_a ,ncvs, nvir_a)).copy()
        eris.veve = ao2mo.general(myadc._scf._eri, (val_a,  vir_a , val_a , vir_a),compact=False).reshape((nval_a, nvir_a, nval_a, nvir_a)).copy()
        eris.CECE = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_b, vir_b),compact=False).reshape((ncvs, nvir_b , ncvs, nvir_b)).copy()
        eris.VECE = ao2mo.general(myadc._scf._eri, (val_b,  vir_b, core_b, vir_b),compact=False).reshape((nval_b, nvir_b, ncvs, nvir_b) ).copy()
        eris.VEVE = ao2mo.general(myadc._scf._eri, (val_b,  vir_b, val_b , vir_b),compact=False).reshape((nval_b, nvir_b, nval_b, nvir_b)).copy()
        eris.CEce = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_a,vir_a),compact=False).reshape((ncvs, nvir_b , ncvs,  nvir_a)).copy()
        eris.VEce = ao2mo.general(myadc._scf._eri, (val_b,  vir_b, core_a,vir_a),compact=False).reshape((nval_b,  nvir_b , ncvs,  nvir_a) ).copy()
        eris.CEve = ao2mo.general(myadc._scf._eri, (core_b, vir_b , val_a , vir_a),compact=False).reshape((ncvs, nvir_b, nval_a,  nvir_a)).copy()
        eris.VEve = ao2mo.general(myadc._scf._eri, (val_b,  vir_b , val_a , vir_a),compact=False).reshape((nval_b, nvir_b, nval_a, nvir_a)).copy()
        eris.cvee = ao2mo.general(myadc._scf._eri, (core_a,  val_a, vir_a , vir_a) ,compact=False).reshape((ncvs, nval_a, nvir_a, nvir_a)).copy()
        eris.CVEE = ao2mo.general(myadc._scf._eri, (core_b, val_b , vir_b , vir_b) ,compact=False).reshape((ncvs, nval_b , nvir_b, nvir_b)).copy()
        eris.cvEE = ao2mo.general(myadc._scf._eri, (core_a, val_a , vir_b ,  vir_b),compact=False).reshape((ncvs, nval_a , nvir_b, nvir_b)).copy()
        eris.CVee = ao2mo.general(myadc._scf._eri, (core_b, val_b , vir_a ,  vir_a),compact=False).reshape((ncvs, nval_b , nvir_a, nvir_a)).copy()
        #----- ADC(3) integrals --------
        eris.oecc = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_a, core_a), compact=False).reshape(nocc_a, nvir_a, ncvs, ncvs).copy()
        eris.oecv = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_a, val_a), compact=False).reshape(nocc_a, nvir_a, ncvs, nval_a).copy()
        eris.OECC = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_b, core_b), compact=False).reshape(nocc_b, nvir_b, ncvs, ncvs).copy()
        eris.OECV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_b, val_b), compact=False).reshape(nocc_b, nvir_b, ncvs, nval_b).copy()
        eris.CEOV = ao2mo.general(myadc._scf._eri, (core_b, vir_b, occ_b, val_b), compact=False).reshape(ncvs, nvir_b, nocc_b, nval_b).copy()
        eris.OEcc = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_a, core_a), compact=False).reshape(nocc_b, nvir_b, ncvs, ncvs).copy()
        eris.OEcv = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_a, val_a), compact=False).reshape(nocc_b, nvir_b, ncvs, nval_a).copy()
        eris.oeCC = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_b, core_b), compact=False).reshape(nocc_a, nvir_a, ncvs, ncvs).copy()
        eris.oeCV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_b, val_b), compact=False).reshape(nocc_a, nvir_a, ncvs, nval_b).copy()
        eris.ceee = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_a, vir_a), compact=True).reshape(ncvs, nvir_a, -1).copy()
        eris.CEEE = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_b, vir_b), compact=True).reshape(ncvs, nvir_b, -1).copy()
        eris.ceEE = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_b, vir_b), compact=True).reshape(ncvs, nvir_a, -1).copy()
        eris.CEee = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_a, vir_a), compact=True).reshape(ncvs, nvir_b, -1).copy()

        eris.ceco = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_a, occ_a), compact=False).reshape(ncvs, nvir_a, ncvs  , nocc_a).copy()
        eris.cevo = ao2mo.general(myadc._scf._eri, (core_a, vir_a, val_a , occ_a), compact=False).reshape(ncvs, nvir_a, nval_a, nocc_a).copy()
        eris.CECO = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_b, occ_b), compact=False).reshape(ncvs, nvir_b, ncvs  , nocc_b).copy()
        eris.CEVO = ao2mo.general(myadc._scf._eri, (core_b, vir_b, val_b , occ_b), compact=False).reshape(ncvs, nvir_b, nval_b, nocc_b).copy()
        eris.CEco = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_a, occ_a), compact=False).reshape(ncvs, nvir_b, ncvs  , nocc_a).copy()
        eris.CEvo = ao2mo.general(myadc._scf._eri, (core_b, vir_b, val_a , occ_a), compact=False).reshape(ncvs, nvir_b, nval_a, nocc_a).copy()
        eris.ceCO = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_b, occ_b), compact=False).reshape(ncvs, nvir_a, ncvs  , nocc_b).copy()
        eris.ceVO = ao2mo.general(myadc._scf._eri, (core_a, vir_a, val_b , occ_b), compact=False).reshape(ncvs, nvir_a, nval_b, nocc_b).copy()

        # Addtional CVS integrals for get_imds function (c: core, e: external, o: all occupied orbitals)
        eris.ceEO = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_b, occ_b), compact=False).reshape(ncvs, nvir_a, nvir_b, nocc_b).copy()
        eris.cooo = ao2mo.general(myadc._scf._eri, (core_a, occ_a, occ_a, occ_a), compact=False).reshape(ncvs, nocc_a, nocc_a, nocc_a).copy()
        eris.ccoo = ao2mo.general(myadc._scf._eri, (core_a, core_a, occ_a, occ_a), compact=False).reshape(ncvs, ncvs, nocc_a, nocc_a).copy()
        eris.COOO = ao2mo.general(myadc._scf._eri, (core_b, occ_b, occ_b, occ_b), compact=False).reshape(ncvs, nocc_b, nocc_b, nocc_b).copy()
        eris.CCOO = ao2mo.general(myadc._scf._eri, (core_b, core_b, occ_b, occ_b), compact=False).reshape(ncvs, ncvs, nocc_b, nocc_b).copy()
        eris.ccOO = ao2mo.general(myadc._scf._eri, (core_a, core_a, occ_b, occ_b), compact=False).reshape(ncvs, ncvs, nocc_b, nocc_b).copy()
        eris.ooCC = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, core_b, core_b), compact=False).reshape(nocc_a, nocc_a, ncvs, ncvs).copy()
        eris.coOO = ao2mo.general(myadc._scf._eri, (core_a, occ_a, occ_b, occ_b), compact=False).reshape(ncvs, nocc_a, nocc_b, nocc_b).copy()

        eris.ceoe = ao2mo.general(myadc._scf._eri, (core_a, vir_a, occ_a, vir_a), compact=False).reshape(ncvs, nvir_a   , nocc_a,nvir_a).copy()
        eris.CEOE = ao2mo.general(myadc._scf._eri, (core_b, vir_b, occ_b, vir_b), compact=False).reshape(ncvs, nvir_b   , nocc_b,nvir_b).copy()
        eris.ceOE = ao2mo.general(myadc._scf._eri, (core_a, vir_a, occ_b, vir_b), compact=False).reshape(ncvs, nvir_a   , nocc_b,nvir_b).copy()
        eris.oeCE = ao2mo.general(myadc._scf._eri, (occ_a, vir_a , core_b, vir_b),compact=False).reshape(nocc_a, nvir_a , ncvs  ,nvir_b).copy()
        eris.coee = ao2mo.general(myadc._scf._eri, (core_a, occ_a, vir_a, vir_a), compact=False).reshape(ncvs, nocc_a, nvir_a, nvir_a).copy()
        eris.COEE = ao2mo.general(myadc._scf._eri, (core_b, occ_b, vir_b, vir_b), compact=False).reshape(ncvs, nocc_b, nvir_b, nvir_b).copy()
        eris.coEE = ao2mo.general(myadc._scf._eri, (core_a, occ_a, vir_b, vir_b), compact=False).reshape(ncvs, nocc_a, nvir_b, nvir_b).copy()
        eris.COee = ao2mo.general(myadc._scf._eri, (core_b, occ_b, vir_a, vir_a), compact=False).reshape(ncvs, nocc_b    ,nvir_a ,nvir_a).copy()
        eris.coco = ao2mo.general(myadc._scf._eri, (core_a, occ_a, core_a, occ_a), compact=False).reshape(ncvs, nocc_a   ,ncvs   ,nocc_a).copy()
        eris.COCO = ao2mo.general(myadc._scf._eri, (core_b, occ_b, core_b, occ_b), compact=False).reshape(ncvs, nocc_b   ,ncvs   ,nocc_b).copy()
        eris.ooCO = ao2mo.general(myadc._scf._eri, (occ_a, occ_a , core_b, occ_b), compact=False).reshape(nocc_a, nocc_a ,ncvs   ,nocc_b).copy()

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):

        eris.vvvv_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_a, vir_a), compact=False).reshape(nvir_a, nvir_a, nvir_a, nvir_a)
        eris.vvvv_p = eris.vvvv_p.transpose(0,2,1,3)
        eris.vvvv_p -= eris.vvvv_p.transpose(0,1,3,2)
        eris.vvvv_p = eris.vvvv_p[:, :, ind_vv_g[0], ind_vv_g[1]]
        eris.vvvv_p = eris.vvvv_p[ind_vv_g[0], ind_vv_g[1]].copy()

        eris.VVVV_p = ao2mo.general(myadc._scf._eri, (vir_b, vir_b, vir_b, vir_b), compact=False).reshape(nvir_b, nvir_b, nvir_b, nvir_b)
        eris.VVVV_p = eris.VVVV_p.transpose(0,2,1,3)
        eris.VVVV_p -= eris.VVVV_p.transpose(0,1,3,2)
        eris.VVVV_p = eris.VVVV_p[:, :, ind_VV_g[0], ind_VV_g[1]]
        eris.VVVV_p = eris.VVVV_p[ind_VV_g[0], ind_VV_g[1]].copy()

        eris.vVvV_p = ao2mo.general(myadc._scf._eri, (vir_a, vir_a, vir_b, vir_b), compact=False).reshape(nvir_a, nvir_a, nvir_b, nvir_b)
        eris.vVvV_p = np.ascontiguousarray(eris.vVvV_p.transpose(0,2,1,3)) 
        eris.vVvV_p = eris.vVvV_p.reshape(nvir_a*nvir_b, nvir_a*nvir_b) 

    log.timer('ADC incore integral transformation', *cput0)

    return eris


def transform_integrals_outcore(myadc):

    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)
    
    mo_a = myadc.mo_coeff[0]
    mo_b = myadc.mo_coeff[1]
    nmo_a = mo_a.shape[1]
    nmo_b = mo_b.shape[1]

    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]

    nocc_a = occ_a.shape[1]
    nocc_b = occ_b.shape[1]
    nvir_a = vir_a.shape[1]
    nvir_b = vir_b.shape[1]

    nvpair_a = nvir_a * (nvir_a+1) // 2
    nvpair_b = nvir_b * (nvir_b+1) // 2

    # Number of CVS orbital (it is assumed that the number of ionized core alpha electrons equals the number of ionized core beta electrons)
    ncvs = myadc.ncvs
    # Slices of occupied MO coeffcients needed for forming CVS integrals
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs
    core_a = occ_a[:,:ncvs]
    core_b = occ_b[:,:ncvs]
    val_a = occ_a[:,ncvs:]
    val_b = occ_b[:,ncvs:]    

    eris = lambda:None

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc_a,nocc_a,nocc_a,nocc_a), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc_a,nocc_a,nvir_a,nvir_a), 'f8', chunks=(nocc_a,nocc_a,1,nvir_a))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc_a,nvir_a,nocc_a,nocc_a), 'f8', chunks=(nocc_a,1,nocc_a,nocc_a))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc_a,nvir_a,nvir_a,nocc_a), 'f8', chunks=(nocc_a,1,nvir_a,nocc_a))
    eris.ovvv = eris.feri1.create_dataset('ovvv', (nocc_a,nvir_a,nvpair_a), 'f8')

    eris.OOOO = eris.feri1.create_dataset('OOOO', (nocc_b,nocc_b,nocc_b,nocc_b), 'f8')
    eris.OOVV = eris.feri1.create_dataset('OOVV', (nocc_b,nocc_b,nvir_b,nvir_b), 'f8', chunks=(nocc_b,nocc_b,1,nvir_b))
    eris.OVOO = eris.feri1.create_dataset('OVOO', (nocc_b,nvir_b,nocc_b,nocc_b), 'f8', chunks=(nocc_b,1,nocc_b,nocc_b))
    eris.OVVO = eris.feri1.create_dataset('OVVO', (nocc_b,nvir_b,nvir_b,nocc_b), 'f8', chunks=(nocc_b,1,nvir_b,nocc_b))
    eris.OVVV = eris.feri1.create_dataset('OVVV', (nocc_b,nvir_b,nvpair_b), 'f8')

    eris.ooOO = eris.feri1.create_dataset('ooOO', (nocc_a,nocc_a,nocc_b,nocc_b), 'f8')
    eris.ooVV = eris.feri1.create_dataset('ooVV', (nocc_a,nocc_a,nvir_b,nvir_b), 'f8', chunks=(nocc_a,nocc_a,1,nvir_b))
    eris.ovOO = eris.feri1.create_dataset('ovOO', (nocc_a,nvir_a,nocc_b,nocc_b), 'f8', chunks=(nocc_a,1,nocc_b,nocc_b))
    eris.ovVO = eris.feri1.create_dataset('ovVO', (nocc_a,nvir_a,nvir_b,nocc_b), 'f8', chunks=(nocc_a,1,nvir_b,nocc_b))
    eris.ovVV = eris.feri1.create_dataset('ovVV', (nocc_a,nvir_a,nvpair_b), 'f8')

    eris.OOvv = eris.feri1.create_dataset('OOvv', (nocc_b,nocc_b,nvir_a,nvir_a), 'f8', chunks=(nocc_b,nocc_b,1,nvir_a))
    eris.OVoo = eris.feri1.create_dataset('OVoo', (nocc_b,nvir_b,nocc_a,nocc_a), 'f8', chunks=(nocc_b,1,nocc_a,nocc_a))
    eris.OVvo = eris.feri1.create_dataset('OVvo', (nocc_b,nvir_b,nvir_a,nocc_a), 'f8', chunks=(nocc_b,1,nvir_a,nocc_a))
    eris.OVvv = eris.feri1.create_dataset('OVvv', (nocc_b,nvir_b,nvpair_a), 'f8')

    # CVS integrals for matvec function (c: core, e: external, v: valence, o: all occupied orbitals)
    if myadc.method_type == 'ip-cvs':

        #----- ADC(2) integrals --------
        eris.cecc = eris.feri1.create_dataset( 'cecc', (ncvs, nvir_a, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.vecc = eris.feri1.create_dataset( 'vecc', (nval_a,  nvir_a, ncvs, ncvs), 'f8', chunks=(nval_a,  1, ncvs, ncvs))
        eris.CECC = eris.feri1.create_dataset( 'CECC', (ncvs, nvir_b, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.VECC = eris.feri1.create_dataset( 'VECC', (nval_b,  nvir_b, ncvs, ncvs), 'f8', chunks=(nval_b,  1, ncvs, ncvs))
        eris.ceCC = eris.feri1.create_dataset( 'ceCC', (ncvs, nvir_a, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.veCC = eris.feri1.create_dataset( 'veCC', (nval_a,  nvir_a, ncvs, ncvs), 'f8', chunks=(nval_a,  1, ncvs, ncvs))
        eris.CEcc = eris.feri1.create_dataset( 'CEcc', (ncvs, nvir_b, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.VEcc = eris.feri1.create_dataset( 'VEcc', (nval_b,  nvir_b, ncvs, ncvs), 'f8', chunks=(nval_b,  1, ncvs, ncvs))

        eris.cecv = eris.feri1.create_dataset( 'cecv', (ncvs, nvir_a,  ncvs, nval_a), 'f8', chunks=(ncvs, 1,  ncvs, nval_a))
        eris.CECV = eris.feri1.create_dataset( 'CECV', (ncvs, nvir_b,  ncvs, nval_b), 'f8', chunks=(ncvs, 1,  ncvs, nval_b))
        eris.ceCV = eris.feri1.create_dataset( 'ceCV', (ncvs, nvir_a,  ncvs, nval_b), 'f8', chunks=(ncvs, 1,  ncvs, nval_b))
        eris.CEcv = eris.feri1.create_dataset( 'CEcv', (ncvs, nvir_b,  ncvs, nval_a), 'f8', chunks=(ncvs, 1,  ncvs, nval_a))

        #----- ADC(2)-x integrals --------
        eris.cccc =  eris.feri1.create_dataset( 'cccc', (ncvs, ncvs, ncvs, ncvs), 'f8') 
        eris.cccv =  eris.feri1.create_dataset( 'cccv', (ncvs, ncvs, ncvs, nval_a ), 'f8') 
        eris.ccvv =  eris.feri1.create_dataset( 'ccvv', (ncvs, ncvs, nval_a,  nval_a ), 'f8') 
        eris.CCCC =  eris.feri1.create_dataset( 'CCCC', (ncvs, ncvs, ncvs, ncvs), 'f8') 
        eris.CCCV =  eris.feri1.create_dataset( 'CCCV', (ncvs, ncvs, ncvs, nval_b ), 'f8') 
        eris.CCVV =  eris.feri1.create_dataset( 'CCVV', (ncvs, ncvs, nval_b,  nval_b ), 'f8') 
        eris.ccCC =  eris.feri1.create_dataset( 'ccCC', (ncvs, ncvs, ncvs, ncvs), 'f8') 
        eris.ccCV =  eris.feri1.create_dataset( 'ccCV', (ncvs, ncvs, ncvs, nval_b ), 'f8') 
        eris.vvCC =  eris.feri1.create_dataset( 'vvCC', (nval_a,  nval_a,  ncvs, ncvs), 'f8') 
        eris.ccVV =  eris.feri1.create_dataset( 'ccVV', (ncvs, ncvs, nval_b,  nval_b ), 'f8') 
        eris.ccee =  eris.feri1.create_dataset( 'ccee', (ncvs, ncvs, nvir_a,  nvir_a ), 'f8', chunks=(ncvs, ncvs, 1,  nvir_a )) 
        eris.vvee =  eris.feri1.create_dataset( 'vvee', (nval_a,  nval_a,  nvir_a,  nvir_a ), 'f8', chunks=(nval_a,  nval_a,  1,  nvir_a )) 
        eris.CCEE =  eris.feri1.create_dataset( 'CCEE', (ncvs, ncvs, nvir_b,  nvir_b ), 'f8', chunks=(ncvs, ncvs, 1,  nvir_b )) 
        eris.VVEE =  eris.feri1.create_dataset( 'VVEE', (nval_b,  nval_b,  nvir_b,  nvir_b ), 'f8', chunks=(nval_b,  nval_b,  1,  nvir_b )) 
        eris.ccEE =  eris.feri1.create_dataset( 'ccEE', (ncvs, ncvs, nvir_b,  nvir_b ), 'f8', chunks=(ncvs, ncvs, 1,  nvir_b )) 
        eris.vvEE =  eris.feri1.create_dataset( 'vvEE', (nval_a,  nval_a,  nvir_b,  nvir_b),  'f8', chunks=(nval_a,  nval_a,  1,  nvir_b),) 
        eris.CCee =  eris.feri1.create_dataset( 'CCee', (ncvs, ncvs, nvir_a,  nvir_a),  'f8', chunks=(ncvs, ncvs, 1,  nvir_a)) 
        eris.VVee =  eris.feri1.create_dataset( 'VVee', (nval_b,  nval_b,  nvir_a,  nvir_a),  'f8', chunks=(nval_b,  nval_b,  1,  nvir_a))

        eris.cvcv =  eris.feri1.create_dataset( 'cvcv', (ncvs, nval_a, ncvs,  nval_a), 'f8') 
        eris.CVCV =  eris.feri1.create_dataset( 'CVCV', (ncvs, nval_b, ncvs,  nval_b), 'f8') 
        eris.cece =  eris.feri1.create_dataset( 'cece', (ncvs, nvir_a, ncvs,  nvir_a), 'f8', chunks=(ncvs, 1,  ncvs, nvir_a)) 
        eris.vece =  eris.feri1.create_dataset( 'vece', (nval_a,  nvir_a, ncvs,  nvir_a), 'f8', chunks=(nval_a,  1,  ncvs, nvir_a)) 
        eris.veve =  eris.feri1.create_dataset( 'veve', (nval_a,  nvir_a, nval_a ,  nvir_a), 'f8', chunks=(nval_a,  1,  nval_a , nvir_a)) 
        eris.CECE =  eris.feri1.create_dataset( 'CECE', (ncvs, nvir_b, ncvs,  nvir_b), 'f8', chunks=(ncvs, 1,  ncvs, nvir_b)) 
        eris.VECE =  eris.feri1.create_dataset( 'VECE', (nval_b,  nvir_b, ncvs,  nvir_b), 'f8', chunks=(nval_b,  1,  ncvs, nvir_b)) 
        eris.VEVE =  eris.feri1.create_dataset( 'VEVE', (nval_b,  nvir_b, nval_b ,  nvir_b), 'f8', chunks=(nval_b,  1,  nval_b , nvir_b)) 
        eris.CEce =  eris.feri1.create_dataset( 'CEce', (ncvs, nvir_b, ncvs,  nvir_a), 'f8', chunks=(ncvs, 1,  ncvs, nvir_a)) 
        eris.VEce =  eris.feri1.create_dataset( 'VEce', (nval_b,  nvir_b, ncvs,  nvir_a), 'f8', chunks=(nval_b,  1,  ncvs, nvir_a)) 
        eris.CEve =  eris.feri1.create_dataset( 'CEve', (ncvs, nvir_b,  nval_a, nvir_a ), 'f8', chunks=(ncvs, 1,  nval_a , nvir_a)) 
        eris.VEve =  eris.feri1.create_dataset( 'VEve', (nval_b,  nvir_b,  nval_a, nvir_a ), 'f8', chunks=(nval_b,  1,  nval_a , nvir_a)) 
        eris.cvCC =  eris.feri1.create_dataset( 'cvCC', (ncvs, nval_a,  ncvs, ncvs), 'f8') 
        eris.cvCV =  eris.feri1.create_dataset( 'cvCV', (ncvs, nval_a,  ncvs, nval_b ), 'f8') 
        eris.cvee =  eris.feri1.create_dataset( 'cvee', (ncvs, nval_a,  nvir_a,  nvir_a ), 'f8', chunks=(ncvs, nval_a,  1,  nvir_a )) 
        eris.CVEE =  eris.feri1.create_dataset( 'CVEE', (ncvs, nval_b,  nvir_b,  nvir_b ), 'f8', chunks=(ncvs, nval_b,  1,  nvir_b )) 
        eris.cvEE =  eris.feri1.create_dataset( 'cvEE', (ncvs, nval_a,  nvir_b,  nvir_b ), 'f8', chunks=(ncvs, nval_a,  1,  nvir_b )) 
        eris.CVee =  eris.feri1.create_dataset( 'CVee', (ncvs, nval_b,  nvir_a,  nvir_a),  'f8', chunks=(ncvs, nval_b,  1,  nvir_a)) 
 
        #----- ADC(3) integrals --------
        eris.oecc = eris.feri1.create_dataset( 'oecc', (nocc_a,  nvir_a, ncvs, ncvs), 'f8', chunks=(nocc_a,  1, ncvs, ncvs)) 
        eris.oecv = eris.feri1.create_dataset( 'oecv', (nocc_a,  nvir_a, ncvs, nval_a ), 'f8', chunks=(nocc_a,  1, ncvs, nval_a )) 
        eris.OECC = eris.feri1.create_dataset( 'OECC', (nocc_b,  nvir_b, ncvs, ncvs), 'f8', chunks=(nocc_b,  1, ncvs, ncvs)) 
        eris.OEcc = eris.feri1.create_dataset( 'OEcc', (nocc_b,  nvir_b, ncvs, ncvs), 'f8', chunks=(nocc_b,  1, ncvs, ncvs)) 
        eris.OEcv = eris.feri1.create_dataset( 'OEcv', (nocc_b,  nvir_b, ncvs, nval_a ), 'f8', chunks=(nocc_b,  1, ncvs, nval_a )) 
        eris.oeCC = eris.feri1.create_dataset( 'oeCC', (nocc_a,  nvir_a, ncvs, ncvs), 'f8', chunks=(nocc_a,  1, ncvs, ncvs)) 
        eris.oeCV = eris.feri1.create_dataset( 'oeCV', (nocc_a,  nvir_a, ncvs, nval_b ), 'f8', chunks=(nocc_a,  1, ncvs, nval_b )) 
        eris.ceee = eris.feri1.create_dataset( 'ceee', (ncvs, nvir_a, nvpair_a ), 'f8') 
        eris.CEEE = eris.feri1.create_dataset( 'CEEE', (ncvs, nvir_b, nvpair_b ), 'f8') 
        eris.ceEE = eris.feri1.create_dataset( 'ceEE', (ncvs, nvir_a, nvpair_b ), 'f8') 
        eris.CEee = eris.feri1.create_dataset( 'CEee', (ncvs, nvir_b, nvpair_a ), 'f8')
        eris.OECV = eris.feri1.create_dataset( 'OECV', (nocc_b,  nvir_b, ncvs, nval_b ), 'f8', chunks=(nocc_b,  1, ncvs, nval_b )) 
        eris.ceco = eris.feri1.create_dataset( 'ceco', (ncvs, nvir_a,  ncvs, nocc_a), 'f8', chunks=(ncvs, 1,  ncvs, nocc_a)) 
        eris.cevo = eris.feri1.create_dataset( 'cevo', (ncvs, nvir_a,  nval_a , nocc_a), 'f8', chunks=(ncvs, 1,  nval_a , nocc_a)) 
        eris.CECO = eris.feri1.create_dataset( 'CECO', (ncvs, nvir_b,  ncvs, nocc_b), 'f8', chunks=(ncvs, 1,  ncvs, nocc_b)) 
        eris.CEVO = eris.feri1.create_dataset( 'CEVO', (ncvs, nvir_b,  nval_b , nocc_b), 'f8', chunks=(ncvs, 1,  nval_b , nocc_b)) 
        eris.CEco = eris.feri1.create_dataset( 'CEco', (ncvs, nvir_b,  ncvs, nocc_a), 'f8', chunks=(ncvs, 1,  ncvs, nocc_a)) 
        eris.CEvo = eris.feri1.create_dataset( 'CEvo', (ncvs, nvir_b,  nval_a , nocc_a), 'f8', chunks=(ncvs, 1,  nval_a , nocc_a)) 
        eris.ceCO = eris.feri1.create_dataset( 'ceCO', (ncvs, nvir_a,  ncvs, nocc_b), 'f8', chunks=(ncvs, 1,  ncvs, nocc_b)) 
        eris.ceVO = eris.feri1.create_dataset( 'ceVO', (ncvs, nvir_a,  nval_b , nocc_b), 'f8', chunks=(ncvs, 1,  nval_b , nocc_b)) 
 
        # Addtional CVS integrals for get_imds function (c: core, e: external, o: all occupied orbitals)
        eris.cooo = eris.feri1.create_dataset( 'cooo', (ncvs, nocc_a,  nocc_a,  nocc_a ), 'f8') 
        eris.ccoo = eris.feri1.create_dataset( 'ccoo', (ncvs, ncvs, nocc_a,  nocc_a ), 'f8') 
        eris.COOO = eris.feri1.create_dataset( 'COOO', (ncvs, nocc_b,  nocc_b,  nocc_b ), 'f8') 
        eris.CCOO = eris.feri1.create_dataset( 'CCOO', (ncvs, ncvs, nocc_b,  nocc_b ), 'f8') 
        eris.ccOO = eris.feri1.create_dataset( 'ccOO', (ncvs, ncvs, nocc_b,  nocc_b ), 'f8') 
        eris.ooCC = eris.feri1.create_dataset( 'ooCC', (nocc_a,  nocc_a,  ncvs, ncvs), 'f8') 
        eris.coOO = eris.feri1.create_dataset( 'coOO', (ncvs, nocc_a,  nocc_b,  nocc_b ), 'f8') 
        eris.coee = eris.feri1.create_dataset( 'coee', (ncvs, nocc_a,  nvir_a,  nvir_a ), 'f8', chunks=(ncvs, nocc_a, 1,  nvir_a )) 
        eris.COEE = eris.feri1.create_dataset( 'COEE', (ncvs, nocc_b,  nvir_b,  nvir_b ), 'f8', chunks=(ncvs, nocc_b, 1,  nvir_b )) 
        eris.coEE = eris.feri1.create_dataset( 'coEE', (ncvs, nocc_a,  nvir_b,  nvir_b ), 'f8', chunks=(ncvs, nocc_a, 1,  nvir_b )) 
        eris.COee = eris.feri1.create_dataset( 'COee', (ncvs, nocc_b,  nvir_a,  nvir_a ), 'f8', chunks=(ncvs, nocc_b, 1,  nvir_a )) 
        eris.ceoe = eris.feri1.create_dataset( 'ceoe', (ncvs, nvir_a, nocc_a , nvir_a), 'f8', chunks=(ncvs, 1,  nocc_a , nvir_a)) 
        eris.CEOE = eris.feri1.create_dataset( 'CEOE', (ncvs, nvir_b, nocc_b , nvir_b), 'f8', chunks=(ncvs, 1,  nocc_b , nvir_b)) 
        eris.ceOE = eris.feri1.create_dataset( 'ceOE', (ncvs, nvir_a, nocc_b , nvir_b), 'f8', chunks=(ncvs, 1,  nocc_b , nvir_b)) 
        eris.oeCE = eris.feri1.create_dataset( 'oeCE', (nocc_a,  nvir_a, ncvs, nvir_b), 'f8', chunks=(nocc_a,  1,  ncvs, nvir_b)) 
        eris.coco = eris.feri1.create_dataset( 'coco', (ncvs, nocc_a, ncvs, nocc_a), 'f8') 
        eris.COCO = eris.feri1.create_dataset( 'COCO', (ncvs, nocc_b, ncvs, nocc_b), 'f8') 
        eris.ooCO = eris.feri1.create_dataset( 'ooCO', (nocc_a,  nocc_a, ncvs, nocc_b), 'f8') 
    
        nval_a_s = slice(ncvs,nocc_a)
        nval_b_s = slice(ncvs,nocc_b)

    cput1 = time.clock(), time.time()
    mol = myadc.mol
    tmpf = lib.H5TmpFile()

    ao2mo.general(mol, (occ_a,mo_a,mo_a,mo_a), tmpf, 'aa')
    buf = np.empty((nmo_a,nmo_a,nmo_a))
    for i in range(nocc_a):
        lib.unpack_tril(tmpf['aa'][i*nmo_a:(i+1)*nmo_a], out=buf)
        eris.oooo[i] = buf[:nocc_a,:nocc_a,:nocc_a]
        eris.oovv[i] = buf[:nocc_a,nocc_a:,nocc_a:]
        eris.ovvv[i] = lib.pack_tril(buf[nocc_a:,nocc_a:,nocc_a:])
        eris.ovoo[i] = buf[nocc_a:,:nocc_a,:nocc_a]
        eris.ovvo[i] = buf[nocc_a:,nocc_a:,:nocc_a]

        if myadc.method_type == 'ip-cvs':
            eris.oecc[i] = buf[nocc_a:,:ncvs,:ncvs]    
            eris.oecv[i] = buf[nocc_a:,:ncvs,nval_a_s]    
            if i <  ncvs:
                eris.cecc[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.ceoe[i] = buf[nocc_a:,:nocc_a,nocc_a:]
                eris.cccc[i] = buf[:ncvs,:ncvs,:ncvs]   
                eris.cccv[i] = buf[:ncvs,:ncvs,nval_a_s]
                eris.ccvv[i] = buf[:ncvs,nval_a_s,nval_a_s]
                eris.ccee[i] = buf[:ncvs,nocc_a:,nocc_a:]
                eris.cooo[i] = buf[:nocc_a,:nocc_a,:nocc_a]
                eris.ccoo[i] = buf[:ncvs,:nocc_a,:nocc_a]
                eris.ceee[i] = lib.pack_tril(buf[nocc_a:,nocc_a:,nocc_a:])
                eris.coee[i] = buf[:nocc_a,nocc_a:,nocc_a:]
                eris.cecv[i] = buf[nocc_a:,:ncvs ,nval_a_s]
                eris.cvcv[i] = buf[nval_a_s,:ncvs ,nval_a_s]
                eris.cece[i] = buf[nocc_a:,:ncvs,nocc_a:]
                eris.ceco[i] = buf[nocc_a:,:ncvs,:nocc_a]  
                eris.cevo[i] = buf[nocc_a:,nval_a_s,:nocc_a]    
                eris.ceoe[i] = buf[nocc_a:,:nocc_a,nocc_a:]
                eris.coco[i] = buf[:nocc_a,:ncvs,:nocc_a ]
                eris.cvee[i] = buf[nval_a_s,nocc_a:,nocc_a:]
            if i >= ncvs:
                i -= ncvs
                eris.vecc[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.vvee[i] = buf[nval_a_s,nocc_a:,nocc_a:]
                eris.vece[i] = buf[nocc_a:,:ncvs,nocc_a:] 
                eris.veve[i] = buf[nocc_a:,nval_a_s,nocc_a:]
    del(tmpf['aa'])

    buf = np.empty((nmo_b,nmo_b,nmo_b))
    ao2mo.general(mol, (occ_b,mo_b,mo_b,mo_b), tmpf, 'bb')
    for i in range(nocc_b):
        lib.unpack_tril(tmpf['bb'][i*nmo_b:(i+1)*nmo_b], out=buf)
        eris.OOOO[i] = buf[:nocc_b,:nocc_b,:nocc_b]
        eris.OVOO[i] = buf[nocc_b:,:nocc_b,:nocc_b]
        eris.OOVV[i] = buf[:nocc_b,nocc_b:,nocc_b:]
        eris.OVVO[i] = buf[nocc_b:,nocc_b:,:nocc_b]
        eris.OVVV[i] = lib.pack_tril(buf[nocc_b:,nocc_b:,nocc_b:])

        if myadc.method_type == 'ip-cvs':
            eris.OECC[i] = buf[nocc_b:,:ncvs,:ncvs]     
            eris.OECV[i] = buf[nocc_b:,:ncvs,nval_b_s]    
            if i < ncvs:
                eris.CECC[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.CCCC[i] = buf[:ncvs,:ncvs,:ncvs] 
                eris.CCCV[i] = buf[:ncvs,:ncvs,nval_b_s]
                eris.CCVV[i] = buf[:ncvs,nval_b_s,nval_b_s]
                eris.CCEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
                eris.COOO[i] = buf[:nocc_b,:nocc_b,:nocc_b]
                eris.CCOO[i] = buf[:ncvs,:nocc_b,:nocc_b]
                eris.CEEE[i] = lib.pack_tril(buf[nocc_b:,nocc_b:,nocc_b:])
                eris.COEE[i] = buf[:nocc_b,nocc_b:,nocc_b:]
                eris.CECV[i] = buf[nocc_b:,:ncvs,nval_b_s]
                eris.CEOE[i] = buf[nocc_b:,:nocc_b,nocc_b: ]
                eris.CVCV[i] = buf[nval_b_s,:ncvs,nval_b_s]
                eris.CECE[i] = buf[nocc_b:,:ncvs,nocc_b: ]
                eris.CECO[i] = buf[nocc_b:,:ncvs,:nocc_b:]     
                eris.CEVO[i] = buf[nocc_b:,nval_b_s,:nocc_b ]    
                eris.CEOE[i] = buf[nocc_b:,:nocc_b,nocc_b: ]
                eris.COCO[i] = buf[:nocc_b,:ncvs,:nocc_b ]
                eris.CVEE[i] = buf[nval_b_s,nocc_b:,nocc_b:]
            if i >= ncvs:
                i -= ncvs
                eris.VECC[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.VVEE[i] = buf[nval_b_s,nocc_b:,nocc_b:]
                eris.VECE[i] = buf[nocc_b:,:ncvs,nocc_b:] 
                eris.VEVE[i] = buf[nocc_b:,nval_b_s,nocc_b:]
    del(tmpf['bb'])

    buf = np.empty((nmo_a,nmo_b,nmo_b))
    ao2mo.general(mol, (occ_a,mo_a,mo_b,mo_b), tmpf, 'ab')
    for i in range(nocc_a):
        lib.unpack_tril(tmpf['ab'][i*nmo_a:(i+1)*nmo_a], out=buf)
        eris.ooOO[i] = buf[:nocc_a,:nocc_b,:nocc_b]
        eris.ovOO[i] = buf[nocc_a:,:nocc_b,:nocc_b]
        eris.ooVV[i] = buf[:nocc_a,nocc_b:,nocc_b:]
        eris.ovVO[i] = buf[nocc_a:,nocc_b:,:nocc_b]
        eris.ovVV[i] = lib.pack_tril(buf[nocc_a:,nocc_b:,nocc_b:])

        if myadc.method_type == 'ip-cvs':
            eris.oeCC[i] = buf[nocc_a:,:ncvs,:ncvs] 
            eris.oeCV[i] = buf[nocc_a:,:ncvs,nval_b_s]
            eris.ooCC[i] = buf[:nocc_a,:ncvs,:ncvs]
            eris.oeCE[i] = buf[nocc_a:,:ncvs,nocc_b:]
            eris.ooCO[i] = buf[:nocc_a,:ncvs,:nocc_b]
            eris.oeCE[i] = buf[nocc_a:,:ncvs,nocc_b:]
            if i < ncvs:
                eris.ceCC[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.ccCC[i] = buf[:ncvs,:ncvs,:ncvs] 
                eris.ccCV[i] = buf[:ncvs,:ncvs,nval_b_s]
                eris.ccVV[i] = buf[:ncvs,nval_b_s,nval_b_s]
                eris.ccEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
                eris.ccOO[i] = buf[:ncvs,:nocc_b,:nocc_b]
                eris.coOO[i] = buf[:nocc_a,:nocc_b,:nocc_b]
                eris.ceEE[i] = lib.pack_tril(buf[nocc_a:,nocc_b:,nocc_b:])
                eris.ceCV[i] = buf[nocc_a:,:ncvs,nval_b_s]
                eris.ceOE[i] = buf[nocc_a:,:nocc_b,nocc_b:]
                eris.ceCO[i] = buf[nocc_a:,:ncvs,:nocc_b] 
                eris.ceVO[i] = buf[nocc_a:,nval_b_s,:nocc_b]
                eris.ceOE[i] = buf[nocc_a:,:nocc_b,nocc_b:]
                eris.cvCC[i] = buf[nval_a_s,:ncvs,:ncvs] 
                eris.cvCV[i] = buf[nval_a_s,:ncvs,nval_b_s]
                eris.cvEE[i] = buf[nval_a_s,nocc_b:,nocc_b:]
                eris.coEE[i] = buf[:nocc_a,nocc_b:,nocc_b:]
            if i >= ncvs:
                i -= ncvs
                eris.veCC[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.vvCC[i] = buf[nval_a_s,:ncvs,:ncvs]
                eris.vvEE[i] = buf[nval_a_s,nocc_b:,nocc_b:]
    del(tmpf['ab'])

    buf = np.empty((nmo_b,nmo_a,nmo_a))
    ao2mo.general(mol, (occ_b,mo_b,mo_a,mo_a), tmpf, 'ba')
    for i in range(nocc_b):
        lib.unpack_tril(tmpf['ba'][i*nmo_b:(i+1)*nmo_b], out=buf)
        eris.OVoo[i] = buf[nocc_b:,:nocc_a,:nocc_a]
        eris.OOvv[i] = buf[:nocc_b,nocc_a:,nocc_a:]
        eris.OVvo[i] = buf[nocc_b:,nocc_a:,:nocc_a]
        eris.OVvv[i] = lib.pack_tril(buf[nocc_b:,nocc_a:,nocc_a:])

        if myadc.method_type == 'ip-cvs':
            eris.OEcc[i] = buf[nocc_b:,:ncvs,:ncvs]    
            eris.OEcv[i] = buf[nocc_b:,:ncvs,nval_a_s]
            if i < ncvs:
                eris.CEcc[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.CCee[i] = buf[:ncvs,nocc_a:,nocc_a:]
                eris.CEee[i] = lib.pack_tril(buf[nocc_b:,nocc_a:,nocc_a:])
                eris.CEcv[i] = buf[nocc_b:,:ncvs,nval_a_s]
                eris.CEce[i] = buf[nocc_b:,:ncvs,nocc_a:] 
                eris.CEve[i] = buf[nocc_b:,nval_a_s,nocc_a:]
                eris.CEco[i] = buf[nocc_b:,:ncvs,:nocc_a]     
                eris.CEvo[i] = buf[nocc_b:,nval_a_s,:nocc_a]
                eris.CVee[i] = buf[nval_b_s,nocc_a:,nocc_a:]
                eris.COee[i] = buf[:nocc_b,nocc_a:,nocc_a:]
            if i >= ncvs:
                i -= ncvs
                eris.VEcc[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.VVee[i] = buf[nval_b_s,nocc_a:,nocc_a:]
                eris.VEce[i] = buf[nocc_b:,:ncvs,nocc_a:]  
                eris.VEve[i] = buf[nocc_b:,nval_a_s,nocc_a:]
    del(tmpf['ba'])

    buf = None
    cput1 = logger.timer_debug1(myadc, 'transforming oopq, ovpq', *cput1)

    ############### forming eris_vvvv ########################################

    if (myadc.method == "adc(2)-x" or myadc.method == "adc(3)"):
    
        cput2 = time.clock(), time.time()

        ind_vv_g = np.tril_indices(nvir_a, k=-1)
        ind_VV_g = np.tril_indices(nvir_b, k=-1)

        eris.vvvv_p = []
        eris.VVVV_p = []
        eris.vVvV_p = []
        eris.VvVv_p = []

        avail_mem = (myadc.max_memory - lib.current_memory()[0]) * 0.25 
        vvv_mem = (nvir_a**3) * 8/1e6

        chnk_size =  int(avail_mem/vvv_mem)

        if chnk_size <= 0 :
            chnk_size = 1

        for p in range(0,vir_a.shape[1],chnk_size):

            if chnk_size < vir_a.shape[1] :
                orb_slice = vir_a[:, p:p+chnk_size]
            else :
                orb_slice = vir_a[:, p:]

            _, tmp = tempfile.mkstemp()
            ao2mo.outcore.general(mol, (orb_slice, vir_a, vir_a, vir_a), tmp, max_memory = avail_mem, ioblk_size=100, compact=False)
            vvvv = radc_ao2mo.read_dataset(tmp,'eri_mo')
            del (tmp)
            vvvv = vvvv.reshape(orb_slice.shape[1], vir_a.shape[1], vir_a.shape[1], vir_a.shape[1])
            vvvv = np.ascontiguousarray(vvvv.transpose(0,2,1,3))
            vvvv -= np.ascontiguousarray(vvvv.transpose(0,1,3,2))
            vvvv = vvvv[:, :, ind_vv_g[0], ind_vv_g[1]]

            vvvv_p = radc_ao2mo.write_dataset(vvvv)
            del vvvv
            eris.vvvv_p.append(vvvv_p)       


        for p in range(0,vir_b.shape[1],chnk_size):

            if chnk_size < vir_b.shape[1] :
                orb_slice = vir_b[:, p:p+chnk_size]
            else :
                orb_slice = vir_b[:, p:]

            _, tmp = tempfile.mkstemp()
            ao2mo.outcore.general(mol, (orb_slice, vir_b, vir_b, vir_b), tmp, max_memory = avail_mem, ioblk_size=100, compact=False)
            VVVV = radc_ao2mo.read_dataset(tmp,'eri_mo')
            del (tmp)
            VVVV = VVVV.reshape(orb_slice.shape[1], vir_b.shape[1], vir_b.shape[1], vir_b.shape[1])
            VVVV = np.ascontiguousarray(VVVV.transpose(0,2,1,3))
            VVVV -= np.ascontiguousarray(VVVV.transpose(0,1,3,2))
            VVVV = VVVV[:, :, ind_VV_g[0], ind_VV_g[1]]

            VVVV_p = radc_ao2mo.write_dataset(VVVV)
            del VVVV
            eris.VVVV_p.append(VVVV_p)       


        for p in range(0,vir_a.shape[1],chnk_size):

            if chnk_size < vir_a.shape[1] :
                orb_slice = vir_a[:, p:p+chnk_size]
            else :
                orb_slice = vir_a[:, p:]

            _, tmp = tempfile.mkstemp()
            ao2mo.outcore.general(mol, (orb_slice, vir_a, vir_b, vir_b), tmp, max_memory = avail_mem, ioblk_size=100, compact=False)
            vVvV = radc_ao2mo.read_dataset(tmp,'eri_mo')
            del (tmp)
            vVvV = vVvV.reshape(orb_slice.shape[1], vir_a.shape[1], vir_b.shape[1], vir_b.shape[1])
            vVvV = np.ascontiguousarray(vVvV.transpose(0,2,1,3))
            vVvV = vVvV.reshape(-1, vir_b.shape[1], vir_a.shape[1] * vir_b.shape[1])

            vVvV_p = radc_ao2mo.write_dataset(vVvV)
            del vVvV
            eris.vVvV_p.append(vVvV_p)       


        for p in range(0,vir_b.shape[1],chnk_size):

            if chnk_size < vir_b.shape[1] :
                orb_slice = vir_b[:, p:p+chnk_size]
            else :
                orb_slice = vir_b[:, p:]

            _, tmp = tempfile.mkstemp()
            ao2mo.outcore.general(mol, (orb_slice, vir_b, vir_a, vir_a), tmp, max_memory = avail_mem, ioblk_size=100, compact=False)
            VvVv = radc_ao2mo.read_dataset(tmp,'eri_mo')
            del tmp
            VvVv = VvVv.reshape(orb_slice.shape[1], vir_b.shape[1], vir_a.shape[1], vir_a.shape[1])
            VvVv = np.ascontiguousarray(VvVv.transpose(0,2,1,3))
            VvVv = VvVv.reshape(-1, vir_a.shape[1], vir_b.shape[1] * vir_a.shape[1])

            VvVv_p = radc_ao2mo.write_dataset(VvVv)
            del VvVv
            eris.VvVv_p.append(VvVv_p)       
    
        cput2 = logger.timer_debug1(myadc, 'transforming vvvv', *cput2)

    log.timer('ADC outcore integral transformation', *cput0)
    return eris


def transform_integrals_df(myadc):
    cput0 = (time.clock(), time.time())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    mo_coeff_a = np.asarray(myadc.mo_coeff[0], order='F')
    mo_coeff_b = np.asarray(myadc.mo_coeff[1], order='F')
    mo_a = myadc.mo_coeff[0]
    mo_b = myadc.mo_coeff[1]
    nmo_a = mo_a.shape[1]
    nmo_b = mo_b.shape[1]

    occ_a = myadc.mo_coeff[0][:,:myadc._nocc[0]]
    occ_b = myadc.mo_coeff[1][:,:myadc._nocc[1]]
    vir_a = myadc.mo_coeff[0][:,myadc._nocc[0]:]
    vir_b = myadc.mo_coeff[1][:,myadc._nocc[1]:]

    nocc_a = occ_a.shape[1]
    nocc_b = occ_b.shape[1]
    nvir_a = vir_a.shape[1]
    nvir_b = vir_b.shape[1]
    nvir_pair_a = nvir_a*(nvir_a+1)//2
    nvir_pair_b = nvir_b*(nvir_b+1)//2

    # Number of CVS orbital (it is assumed that the number of ionized core alpha electrons equals the number of ionized core beta electrons)
    ncvs = ncore_a = ncore_b = myadc.ncvs
    nval_a = nocc_a - ncvs
    nval_b = nocc_b - ncvs
    nval_a_s = slice(ncvs,nocc_a)
    nval_b_s = slice(ncvs,nocc_b)

    eris = lambda:None
    eris.vvvv = None
    with_df = myadc.with_df 
    naux = with_df.get_naoaux()
    Loo = np.empty((naux,nocc_a,nocc_a))
    Lvo = np.empty((naux,nvir_a,nocc_a))
    LOO = np.empty((naux,nocc_b,nocc_b))
    LVO = np.empty((naux,nvir_b,nocc_b))
    eris.Lov = np.empty((naux,nocc_a,nvir_a))
    eris.LOV = np.empty((naux,nocc_b,nvir_b))
    eris.Lvv = np.empty((naux,nvir_a,nvir_a))
    eris.LVV = np.empty((naux,nvir_b,nvir_b))
    ijslice = (0, nmo_a, 0, nmo_a)
    Lpq = None
    p1 = 0

    # CVS auxiliary integrals (c: core, e: external, v: valence, o: all occupied orbitals)
    eris.L_cc = np.empty((naux,ncvs,ncvs))
    eris.L_cv = np.empty((naux,ncvs,nval_a))
    eris.L_ce = np.empty((naux,ncvs,nvir_a))
    eris.L_vv = np.empty((naux,nval_a,nval_a))
    eris.L_ve = np.empty((naux,nval_a,nvir_a))
    eris.L_ee = np.empty((naux,nvir_a,nvir_a))
    eris.L_CC = np.empty((naux,ncvs,ncvs))
    eris.L_CV = np.empty((naux,ncvs,nval_b))
    eris.L_CE = np.empty((naux,ncvs,nvir_b))
    eris.L_VV = np.empty((naux,nval_b,nval_b))
    eris.L_VE = np.empty((naux,nval_b,nvir_b))
    eris.L_EE = np.empty((naux,nvir_b,nvir_b))

    eris.L_oo = np.empty((naux,nocc_a,nocc_a))  
    eris.L_oe = np.empty((naux,nocc_a,nvir_a))  
    eris.L_co = np.empty((naux,ncvs,nocc_a))  
    eris.L_vo = np.empty((naux,nval_a,nocc_a))  
    eris.L_OO = np.empty((naux,nocc_b,nocc_b))  
    eris.L_OE = np.empty((naux,nocc_b,nvir_b))  
    eris.L_CO = np.empty((naux,ncvs,nocc_b))  
    eris.L_VO = np.empty((naux,nval_b,nocc_b))  

    #for eri1 in myadc._scf.with_df.loop():
    for eri1 in myadc.with_df.loop():
        Lpq = ao2mo._ao2mo.nr_e2(eri1, mo_coeff_a, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo_a,nmo_a)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Loo[p0:p1] = Lpq[:,:nocc_a,:nocc_a]
        eris.Lov[p0:p1] = Lpq[:,:nocc_a,nocc_a:]
        Lvo[p0:p1] = Lpq[:,nocc_a:,:nocc_a]
        eris.Lvv[p0:p1] = Lpq[:,nocc_a:,nocc_a:]

        if myadc.method_type == 'ip-cvs':
            eris.L_cc[p0:p1] = Lpq[:,:ncvs,:ncvs ]    
            eris.L_cv[p0:p1] = Lpq[:,:ncvs,nval_a_s ]
            eris.L_ce[p0:p1] = Lpq[:,:ncvs,nocc_a: ]
            eris.L_vv[p0:p1] = Lpq[:,nval_a_s,nval_a_s ]
            eris.L_ve[p0:p1] = Lpq[:, nval_a_s, nocc_a:]
            eris.L_ee[p0:p1] = Lpq[:, nocc_a:,nocc_a:]
            eris.L_oo[p0:p1] = Lpq[:, :nocc_a,:nocc_a]
            eris.L_oe[p0:p1] = Lpq[:, :nocc_a,nocc_a:]
            eris.L_co[p0:p1] = Lpq[:,:ncvs,:nocc_a]
            eris.L_vo[p0:p1] = Lpq[:,nval_a_s,:nocc_a]

    ijslice = (0, nmo_b, 0, nmo_b)
    Lpq = None
    p1 = 0
    #for eri1 in myadc._scf.with_df.loop():
    for eri1 in myadc.with_df.loop():
        Lpq = ao2mo._ao2mo.nr_e2(eri1, mo_coeff_b, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo_b,nmo_b)
        p0, p1 = p1, p1 + Lpq.shape[0]
        LOO[p0:p1] = Lpq[:,:nocc_b,:nocc_b]
        eris.LOV[p0:p1] = Lpq[:,:nocc_b,nocc_b:]
        LVO[p0:p1] = Lpq[:,nocc_b:,:nocc_b]
        eris.LVV[p0:p1] = Lpq[:,nocc_b:,nocc_b:]

        if myadc.method_type == 'ip-cvs':
            eris.L_CC[p0:p1] = Lpq[:,:ncvs,:ncvs ]      
            eris.L_CV[p0:p1] = Lpq[:,:ncvs,nval_b_s ]
            eris.L_CE[p0:p1] = Lpq[:,:ncvs,nocc_b: ]
            eris.L_VV[p0:p1] = Lpq[:,nval_b_s,nval_b_s] 
            eris.L_VE[p0:p1] = Lpq[:, nval_b_s, nocc_b:]
            eris.L_EE[p0:p1] = Lpq[:, nocc_b:,nocc_b:]
            eris.L_OO[p0:p1] = Lpq[:, :nocc_b,:nocc_b]
            eris.L_OE[p0:p1] = Lpq[:, :nocc_b,nocc_b:]
            eris.L_CO[p0:p1] = Lpq[:,:ncvs,:nocc_b]
            eris.L_VO[p0:p1] = Lpq[:,nval_b_s,:nocc_b]


    Loo = Loo.reshape(naux,nocc_a*nocc_a)
    eris.Lov = eris.Lov.reshape(naux,nocc_a*nvir_a)
    Lvo = Lvo.reshape(naux,nocc_a*nvir_a)
    LOO = LOO.reshape(naux,nocc_b*nocc_b)
    eris.LOV = eris.LOV.reshape(naux,nocc_b*nvir_b)
    LVO = LVO.reshape(naux,nocc_b*nvir_b)

    eris.L_ce_t = eris.L_ce.copy()
    eris.L_ee_t = eris.L_ee.copy()
    eris.L_CE_t = eris.L_CE.copy()
    eris.L_EE_t = eris.L_EE.copy()

    eris.L_cc = eris.L_cc.reshape(naux,ncvs*ncvs)
    eris.L_cv = eris.L_cv.reshape(naux,ncvs*nval_a)
    eris.L_ce = eris.L_ce.reshape(naux,ncvs*nvir_a)
    eris.L_vv = eris.L_vv.reshape(naux,nval_a*nval_a)
    eris.L_ve = eris.L_ve.reshape(naux,nval_a*nvir_a)
    eris.L_ee = eris.L_ee.reshape(naux,nvir_a*nvir_a)
    eris.L_CC = eris.L_CC.reshape(naux,ncvs*ncvs)
    eris.L_CV = eris.L_CV.reshape(naux,ncvs*nval_b)
    eris.L_CE = eris.L_CE.reshape(naux,ncvs*nvir_b)
    eris.L_VV = eris.L_VV.reshape(naux,nval_b*nval_b)
    eris.L_VE = eris.L_VE.reshape(naux,nval_b*nvir_b)
    eris.L_EE = eris.L_EE.reshape(naux,nvir_b*nvir_b)
    eris.L_oo = eris.L_oo.reshape(naux,nocc_a*nocc_a) 
    eris.L_oe = eris.L_oe.reshape(naux,nocc_a*nvir_a) 
    eris.L_co = eris.L_co.reshape(naux,ncvs*nocc_a)  
    eris.L_vo = eris.L_vo.reshape(naux,nval_a*nocc_a) 
    eris.L_OO = eris.L_OO.reshape(naux,nocc_b*nocc_b) 
    eris.L_OE = eris.L_OE.reshape(naux,nocc_b*nvir_b) 
    eris.L_CO = eris.L_CO.reshape(naux,ncvs*nocc_b)  
    eris.L_VO = eris.L_VO.reshape(naux,nval_b*nocc_b) 

    eris.L_ee_p = Lvv_p = lib.pack_tril(eris.Lvv)
    eris.L_EE_p = LVV_p = lib.pack_tril(eris.LVV)

    eris.vvvv_p = None
    eris.VVVV_p = None
    eris.vVvV_p = None
    eris.VvVv_p = None

    eris.ovvv = None
    eris.OVVV = None
    eris.OVVV = None
    eris.ovVV = None
    eris.OVvv = None

    eris.ceee = None
    eris.CEEE = None
    eris.ceEE = None
    eris.CEee = None

    eris.feri1 = lib.H5TmpFile()
    eris.oooo = eris.feri1.create_dataset('oooo', (nocc_a,nocc_a,nocc_a,nocc_a), 'f8')
    eris.oovv = eris.feri1.create_dataset('oovv', (nocc_a,nocc_a,nvir_a,nvir_a), 'f8', chunks=(nocc_a,nocc_a,1,nvir_a))
    eris.ovoo = eris.feri1.create_dataset('ovoo', (nocc_a,nvir_a,nocc_a,nocc_a), 'f8', chunks=(nocc_a,1,nocc_a,nocc_a))
    eris.ovvo = eris.feri1.create_dataset('ovvo', (nocc_a,nvir_a,nvir_a,nocc_a), 'f8', chunks=(nocc_a,1,nvir_a,nocc_a))

    eris.oooo[:] = lib.ddot(Loo.T, Loo).reshape(nocc_a,nocc_a,nocc_a,nocc_a)
    eris.ovoo[:] = lib.ddot(eris.Lov.T, Loo).reshape(nocc_a,nvir_a,nocc_a,nocc_a)
    eris.oovv[:] = lib.unpack_tril(lib.ddot(Loo.T, Lvv_p)).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
    eris.ovvo[:] = lib.ddot(eris.Lov.T, Lvo).reshape(nocc_a,nvir_a,nvir_a,nocc_a)

    eris.OOOO = eris.feri1.create_dataset('OOOO', (nocc_b,nocc_b,nocc_b,nocc_b), 'f8')
    eris.OOVV = eris.feri1.create_dataset('OOVV', (nocc_b,nocc_b,nvir_b,nvir_b), 'f8', chunks=(nocc_b,nocc_b,1,nvir_b))
    eris.OVOO = eris.feri1.create_dataset('OVOO', (nocc_b,nvir_b,nocc_b,nocc_b), 'f8', chunks=(nocc_b,1,nocc_b,nocc_b))
    eris.OVVO = eris.feri1.create_dataset('OVVO', (nocc_b,nvir_b,nvir_b,nocc_b), 'f8', chunks=(nocc_b,1,nvir_b,nocc_b))

    eris.OOOO[:] = lib.ddot(LOO.T, LOO).reshape(nocc_b,nocc_b,nocc_b,nocc_b)
    eris.OVOO[:] = lib.ddot(eris.LOV.T, LOO).reshape(nocc_b,nvir_b,nocc_b,nocc_b)
    eris.OOVV[:] = lib.unpack_tril(lib.ddot(LOO.T, LVV_p)).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
    eris.OVVO[:] = lib.ddot(eris.LOV.T, LVO).reshape(nocc_b,nvir_b,nvir_b,nocc_b)

    eris.ooOO = eris.feri1.create_dataset('ooOO', (nocc_a,nocc_a,nocc_b,nocc_b), 'f8')
    eris.ooVV = eris.feri1.create_dataset('ooVV', (nocc_a,nocc_a,nvir_b,nvir_b), 'f8', chunks=(nocc_a,nocc_a,1,nvir_b))
    eris.ovOO = eris.feri1.create_dataset('ovOO', (nocc_a,nvir_a,nocc_b,nocc_b), 'f8', chunks=(nocc_a,1,nocc_b,nocc_b))
    eris.ovVO = eris.feri1.create_dataset('ovVO', (nocc_a,nvir_a,nvir_b,nocc_b), 'f8', chunks=(nocc_a,1,nvir_b,nocc_b))

    eris.ooOO[:] = lib.ddot(Loo.T, LOO).reshape(nocc_a,nocc_a,nocc_b,nocc_b)
    eris.ooVV[:] = lib.unpack_tril(lib.ddot(Loo.T, LVV_p)).reshape(nocc_a,nocc_a,nvir_b,nvir_b)
    eris.ovOO[:] = lib.ddot(eris.Lov.T, LOO).reshape(nocc_a,nvir_a,nocc_b,nocc_b)
    eris.ovVO[:] = lib.ddot(eris.Lov.T, LVO).reshape(nocc_a,nvir_a,nvir_b,nocc_b)

    eris.OOvv = eris.feri1.create_dataset('OOvv', (nocc_b,nocc_b,nvir_a,nvir_a), 'f8', chunks=(nocc_b,nocc_b,1,nvir_a))
    eris.OVoo = eris.feri1.create_dataset('OVoo', (nocc_b,nvir_b,nocc_a,nocc_a), 'f8', chunks=(nocc_b,1,nocc_a,nocc_a))
    eris.OVvo = eris.feri1.create_dataset('OVvo', (nocc_b,nvir_b,nvir_a,nocc_a), 'f8', chunks=(nocc_b,1,nvir_a,nocc_a))

    eris.OOvv[:] = lib.unpack_tril(lib.ddot(LOO.T, Lvv_p)).reshape(nocc_b,nocc_b,nvir_a,nvir_a)  
    eris.OVoo[:] = lib.ddot(eris.LOV.T, Loo).reshape(nocc_b,nvir_b,nocc_a,nocc_a)
    eris.OVvo[:] = lib.ddot(eris.LOV.T, Lvo).reshape(nocc_b,nvir_b,nvir_a,nocc_a)

    eris.Lov = eris.Lov.reshape(naux,nocc_a,nvir_a)
    eris.LOV = eris.LOV.reshape(naux,nocc_b,nvir_b)
    eris.Lvv = eris.Lvv.reshape(naux,nvir_a,nvir_a)
    eris.LVV = eris.LVV.reshape(naux,nvir_b,nvir_b)

    if myadc.method_type == 'ip-cvs':
        #----- ADC(2) integrals --------
        eris.cecc = eris.feri1.create_dataset( 'cecc', (ncvs, nvir_a, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.vecc = eris.feri1.create_dataset( 'vecc', (nval_a,  nvir_a, ncvs, ncvs), 'f8', chunks=(nval_a,  1, ncvs, ncvs))
        eris.CECC = eris.feri1.create_dataset( 'CECC', (ncvs, nvir_b, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.VECC = eris.feri1.create_dataset( 'VECC', (nval_b,  nvir_b, ncvs, ncvs), 'f8', chunks=(nval_b,  1, ncvs, ncvs))
        eris.ceCC = eris.feri1.create_dataset( 'ceCC', (ncvs, nvir_a, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.veCC = eris.feri1.create_dataset( 'veCC', (nval_a,  nvir_a, ncvs, ncvs), 'f8', chunks=(nval_a,  1, ncvs, ncvs))
        eris.CEcc = eris.feri1.create_dataset( 'CEcc', (ncvs, nvir_b, ncvs, ncvs), 'f8', chunks=(ncvs, 1, ncvs, ncvs))
        eris.VEcc = eris.feri1.create_dataset( 'VEcc', (nval_b,  nvir_b, ncvs, ncvs), 'f8', chunks=(nval_b,  1, ncvs, ncvs))
        eris.cecv = eris.feri1.create_dataset( 'cecv', (ncvs, nvir_a,  ncvs, nval_a), 'f8', chunks=(ncvs, 1,  ncvs, nval_a))
        eris.CECV = eris.feri1.create_dataset( 'CECV', (ncvs, nvir_b,  ncvs, nval_b), 'f8', chunks=(ncvs, 1,  ncvs, nval_b))
        eris.ceCV = eris.feri1.create_dataset( 'ceCV', (ncvs, nvir_a,  ncvs, nval_b), 'f8', chunks=(ncvs, 1,  ncvs, nval_b))
        eris.CEcv = eris.feri1.create_dataset( 'CEcv', (ncvs, nvir_b,  ncvs, nval_a), 'f8', chunks=(ncvs, 1,  ncvs, nval_a))

        eris.cecc[:] = lib.ddot(eris.L_ce.T,eris.L_cc).reshape((ncore_a, nvir_a, ncore_a, ncore_a)) 
        eris.cecv[:] = lib.ddot(eris.L_ce.T,eris.L_cv).reshape((ncore_a, nvir_a,  ncore_a, nval_a))
        eris.vecc[:] = lib.ddot(eris.L_ve.T,eris.L_cc).reshape((nval_a,  nvir_a, ncore_a, ncore_a))
        eris.CECC[:] = lib.ddot(eris.L_CE.T,eris.L_CC).reshape((ncore_b, nvir_b, ncore_b, ncore_b))
        eris.CECV[:] = lib.ddot(eris.L_CE.T,eris.L_CV).reshape((ncore_b, nvir_b,  ncore_b, nval_b))
        eris.VECC[:] = lib.ddot(eris.L_VE.T,eris.L_CC).reshape((nval_b,  nvir_b, ncore_b, ncore_b))
        eris.ceCC[:] = lib.ddot(eris.L_ce.T,eris.L_CC).reshape((ncore_a, nvir_a, ncore_b, ncore_b))
        eris.ceCV[:] = lib.ddot(eris.L_ce.T,eris.L_CV).reshape((ncore_a, nvir_a,  ncore_b, nval_b))
        eris.veCC[:] = lib.ddot(eris.L_ve.T,eris.L_CC).reshape((nval_a,  nvir_a, ncore_b, ncore_b))
        eris.CEcc[:] = lib.ddot(eris.L_CE.T,eris.L_cc).reshape((ncore_b, nvir_b, ncore_a, ncore_a))
        eris.CEcv[:] = lib.ddot(eris.L_CE.T,eris.L_cv).reshape((ncore_b, nvir_b,  ncore_a, nval_a))
        eris.VEcc[:] = lib.ddot(eris.L_VE.T,eris.L_cc).reshape((nval_b,  nvir_b, ncore_a, ncore_a))

        #----- ADC(2)-x integrals --------
        eris.cccc =  eris.feri1.create_dataset( 'cccc', (ncvs, ncvs, ncvs,   ncvs         ), 'f8') 
        eris.cccv =  eris.feri1.create_dataset( 'cccv', (ncvs, ncvs, ncvs,   nval_a       ), 'f8') 
        eris.ccvv =  eris.feri1.create_dataset( 'ccvv', (ncvs, ncvs, nval_a, nval_a       ), 'f8') 
        eris.CCCC =  eris.feri1.create_dataset( 'CCCC', (ncvs, ncvs, ncvs,   ncvs         ), 'f8') 
        eris.CCCV =  eris.feri1.create_dataset( 'CCCV', (ncvs, ncvs, ncvs,   nval_b       ), 'f8') 
        eris.CCVV =  eris.feri1.create_dataset( 'CCVV', (ncvs, ncvs, nval_b, nval_b       ), 'f8') 
        eris.ccCC =  eris.feri1.create_dataset( 'ccCC', (ncvs, ncvs, ncvs,   ncvs         ), 'f8') 
        eris.ccCV =  eris.feri1.create_dataset( 'ccCV', (ncvs, ncvs, ncvs,   nval_b       ), 'f8') 
        eris.vvCC =  eris.feri1.create_dataset( 'vvCC', (nval_a, nval_a,  ncvs, ncvs      ), 'f8') 
        eris.ccVV =  eris.feri1.create_dataset( 'ccVV', (ncvs, ncvs, nval_b,  nval_b      ), 'f8') 
        eris.ccee =  eris.feri1.create_dataset( 'ccee', (ncvs, ncvs, nvir_a,  nvir_a      ), 'f8', chunks=(ncvs, ncvs, 1,  nvir_a )) 
        eris.vvee =  eris.feri1.create_dataset( 'vvee', (nval_a, nval_a, nvir_a, nvir_a   ), 'f8', chunks=(nval_a,  nval_a,  1,  nvir_a )) 
        eris.CCEE =  eris.feri1.create_dataset( 'CCEE', (ncvs, ncvs, nvir_b,  nvir_b      ), 'f8', chunks=(ncvs, ncvs, 1,  nvir_b )) 
        eris.VVEE =  eris.feri1.create_dataset( 'VVEE', (nval_b, nval_b, nvir_b, nvir_b   ), 'f8', chunks=(nval_b,  nval_b,  1,  nvir_b )) 
        eris.ccEE =  eris.feri1.create_dataset( 'ccEE', (ncvs, ncvs, nvir_b,  nvir_b      ), 'f8', chunks=(ncvs, ncvs, 1,  nvir_b )) 
        eris.vvEE =  eris.feri1.create_dataset( 'vvEE', (nval_a,  nval_a,  nvir_b,  nvir_b),  'f8', chunks=(nval_a,  nval_a,  1,  nvir_b),) 
        eris.CCee =  eris.feri1.create_dataset( 'CCee', (ncvs, ncvs, nvir_a,  nvir_a      ),  'f8', chunks=(ncvs, ncvs, 1,  nvir_a)) 
        eris.VVee =  eris.feri1.create_dataset( 'VVee', (nval_b,  nval_b,  nvir_a,  nvir_a),  'f8', chunks=(nval_b,  nval_b,  1,  nvir_a))
        eris.cvcv =  eris.feri1.create_dataset( 'cvcv', (ncvs, nval_a, ncvs,  nval_a      ), 'f8') 
        eris.CVCV =  eris.feri1.create_dataset( 'CVCV', (ncvs, nval_b, ncvs,  nval_b      ), 'f8') 
        eris.cece =  eris.feri1.create_dataset( 'cece', (ncvs, nvir_a, ncvs,  nvir_a      ), 'f8', chunks=(ncvs, 1,  ncvs, nvir_a)) 
        eris.vece =  eris.feri1.create_dataset( 'vece', (nval_a,  nvir_a, ncvs,  nvir_a   ), 'f8', chunks=(nval_a,  1,  ncvs, nvir_a)) 
        eris.veve =  eris.feri1.create_dataset( 'veve', (nval_a,  nvir_a, nval_a ,  nvir_a), 'f8', chunks=(nval_a,  1,  nval_a , nvir_a)) 
        eris.CECE =  eris.feri1.create_dataset( 'CECE', (ncvs, nvir_b, ncvs,  nvir_b      ), 'f8', chunks=(ncvs, 1,  ncvs, nvir_b)) 
        eris.VECE =  eris.feri1.create_dataset( 'VECE', (nval_b,  nvir_b, ncvs,  nvir_b   ), 'f8', chunks=(nval_b,  1,  ncvs, nvir_b)) 
        eris.VEVE =  eris.feri1.create_dataset( 'VEVE', (nval_b,  nvir_b, nval_b ,  nvir_b), 'f8', chunks=(nval_b,  1,  nval_b , nvir_b)) 
        eris.CEce =  eris.feri1.create_dataset( 'CEce', (ncvs, nvir_b, ncvs,  nvir_a      ), 'f8', chunks=(ncvs, 1,  ncvs, nvir_a)) 
        eris.VEce =  eris.feri1.create_dataset( 'VEce', (nval_b,  nvir_b, ncvs,  nvir_a   ), 'f8', chunks=(nval_b,  1,  ncvs, nvir_a)) 
        eris.CEve =  eris.feri1.create_dataset( 'CEve', (ncvs, nvir_b,  nval_a, nvir_a    ), 'f8', chunks=(ncvs, 1,  nval_a , nvir_a)) 
        eris.VEve =  eris.feri1.create_dataset( 'VEve', (nval_b,  nvir_b,  nval_a, nvir_a ), 'f8', chunks=(nval_b,  1,  nval_a , nvir_a)) 
        eris.cvCC =  eris.feri1.create_dataset( 'cvCC', (ncvs, nval_a,  ncvs, ncvs        ), 'f8') 
        eris.cvCV =  eris.feri1.create_dataset( 'cvCV', (ncvs, nval_a,  ncvs, nval_b      ), 'f8') 
        eris.cvee =  eris.feri1.create_dataset( 'cvee', (ncvs, nval_a,  nvir_a,  nvir_a   ), 'f8', chunks=(ncvs, nval_a,  1,  nvir_a )) 
        eris.CVEE =  eris.feri1.create_dataset( 'CVEE', (ncvs, nval_b,  nvir_b,  nvir_b   ), 'f8', chunks=(ncvs, nval_b,  1,  nvir_b )) 
        eris.cvEE =  eris.feri1.create_dataset( 'cvEE', (ncvs, nval_a,  nvir_b,  nvir_b   ), 'f8', chunks=(ncvs, nval_a,  1,  nvir_b )) 
        eris.CVee =  eris.feri1.create_dataset( 'CVee', (ncvs, nval_b,  nvir_a,  nvir_a   ),  'f8', chunks=(ncvs, nval_b,  1,  nvir_a)) 

        eris.cccc[:] = lib.ddot(eris.L_cc.T,eris.L_cc).reshape((ncvs, ncvs, ncvs,   ncvs         ))
        eris.cccv[:] = lib.ddot(eris.L_cc.T,eris.L_cv).reshape((ncvs, ncvs, ncvs,   nval_a       ))  
        eris.ccvv[:] = lib.ddot(eris.L_cc.T,eris.L_vv).reshape((ncvs, ncvs, nval_a, nval_a       ))
        eris.CCCC[:] = lib.ddot(eris.L_CC.T,eris.L_CC).reshape((ncvs, ncvs, ncvs,   ncvs         ))
        eris.CCCV[:] = lib.ddot(eris.L_CC.T,eris.L_CV).reshape((ncvs, ncvs, ncvs,   nval_b       ))
        eris.CCVV[:] = lib.ddot(eris.L_CC.T,eris.L_VV).reshape((ncvs, ncvs, nval_b, nval_b       ))
        eris.ccCC[:] = lib.ddot(eris.L_cc.T,eris.L_CC).reshape((ncvs, ncvs, ncvs,   ncvs         ))
        eris.ccCV[:] = lib.ddot(eris.L_cc.T,eris.L_CV).reshape((ncvs, ncvs, ncvs,   nval_b       ))
        eris.vvCC[:] = lib.ddot(eris.L_vv.T,eris.L_CC).reshape((nval_a, nval_a,  ncvs, ncvs      ))
        eris.ccVV[:] = lib.ddot(eris.L_cc.T,eris.L_VV).reshape((ncvs, ncvs, nval_b,  nval_b      ))
        eris.ccee[:] = lib.unpack_tril(lib.ddot(eris.L_cc.T,eris.L_ee_p)).reshape((ncvs, ncvs, nvir_a,  nvir_a      ))
        eris.vvee[:] = lib.unpack_tril(lib.ddot(eris.L_vv.T,eris.L_ee_p)).reshape((nval_a, nval_a, nvir_a, nvir_a   ))
        eris.CCEE[:] = lib.unpack_tril(lib.ddot(eris.L_CC.T,eris.L_EE_p)).reshape((ncvs, ncvs, nvir_b,  nvir_b      ))
        eris.VVEE[:] = lib.unpack_tril(lib.ddot(eris.L_VV.T,eris.L_EE_p)).reshape((nval_b, nval_b, nvir_b, nvir_b   ))
        eris.ccEE[:] = lib.unpack_tril(lib.ddot(eris.L_cc.T,eris.L_EE_p)).reshape((ncvs, ncvs, nvir_b,  nvir_b      ))
        eris.vvEE[:] = lib.unpack_tril(lib.ddot(eris.L_vv.T,eris.L_EE_p)).reshape((nval_a,  nval_a,  nvir_b,  nvir_b))
        eris.CCee[:] = lib.unpack_tril(lib.ddot(eris.L_CC.T,eris.L_ee_p)).reshape((ncvs, ncvs, nvir_a,  nvir_a      ))
        eris.VVee[:] = lib.ddot(eris.L_VV.T,eris.L_ee).reshape((nval_b,  nval_b,  nvir_a,  nvir_a))
        eris.cvcv[:] = lib.ddot(eris.L_cv.T,eris.L_cv).reshape((ncvs, nval_a, ncvs,  nval_a      ))
        eris.CVCV[:] = lib.ddot(eris.L_CV.T,eris.L_CV).reshape((ncvs, nval_b, ncvs,  nval_b      ))
        eris.cece[:] = lib.ddot(eris.L_ce.T,eris.L_ce).reshape((ncvs, nvir_a, ncvs,  nvir_a      ))
        eris.vece[:] = lib.ddot(eris.L_ve.T,eris.L_ce).reshape((nval_a,  nvir_a, ncvs,  nvir_a   ))
        eris.veve[:] = lib.ddot(eris.L_ve.T,eris.L_ve).reshape((nval_a,  nvir_a, nval_a ,  nvir_a))
        eris.CECE[:] = lib.ddot(eris.L_CE.T,eris.L_CE).reshape((ncvs, nvir_b, ncvs,  nvir_b      ))
        eris.VECE[:] = lib.ddot(eris.L_VE.T,eris.L_CE).reshape((nval_b,  nvir_b, ncvs,  nvir_b   ))
        eris.VEVE[:] = lib.ddot(eris.L_VE.T,eris.L_VE).reshape((nval_b,  nvir_b, nval_b ,  nvir_b))
        eris.CEce[:] = lib.ddot(eris.L_CE.T,eris.L_ce).reshape((ncvs, nvir_b, ncvs,  nvir_a      ))
        eris.VEce[:] = lib.ddot(eris.L_VE.T,eris.L_ce).reshape((nval_b,  nvir_b, ncvs,  nvir_a   ))
        eris.CEve[:] = lib.ddot(eris.L_CE.T,eris.L_ve).reshape((ncvs, nvir_b,  nval_a, nvir_a    ))
        eris.VEve[:] = lib.ddot(eris.L_VE.T,eris.L_ve).reshape((nval_b,  nvir_b,  nval_a, nvir_a ))
        eris.cvCC[:] = lib.ddot(eris.L_cv.T,eris.L_CC).reshape((ncvs, nval_a,  ncvs, ncvs        ))
        eris.cvCV[:] = lib.ddot(eris.L_cv.T,eris.L_CV).reshape((ncvs, nval_a,  ncvs, nval_b      ))
        eris.cvee[:] = lib.unpack_tril(lib.ddot(eris.L_cv.T,eris.L_ee_p)).reshape((ncvs, nval_a,  nvir_a,  nvir_a   ))
        eris.CVEE[:] = lib.unpack_tril(lib.ddot(eris.L_CV.T,eris.L_EE_p)).reshape((ncvs, nval_b,  nvir_b,  nvir_b   ))
        eris.cvEE[:] = lib.unpack_tril(lib.ddot(eris.L_cv.T,eris.L_EE_p)).reshape((ncvs, nval_a,  nvir_b,  nvir_b   ))
        eris.CVee[:] = lib.unpack_tril(lib.ddot(eris.L_CV.T,eris.L_ee_p)).reshape((ncvs, nval_b,  nvir_a,  nvir_a   ))

        #----- ADC(3) integrals --------
        eris.oecc = eris.feri1.create_dataset( 'oecc', (nocc_a,  nvir_a, ncvs, ncvs   ), 'f8', chunks=(nocc_a,  1, ncvs, ncvs)) 
        eris.oecv = eris.feri1.create_dataset( 'oecv', (nocc_a,  nvir_a, ncvs, nval_a ), 'f8', chunks=(nocc_a,  1, ncvs, nval_a )) 
        eris.OECC = eris.feri1.create_dataset( 'OECC', (nocc_b,  nvir_b, ncvs, ncvs   ), 'f8', chunks=(nocc_b,  1, ncvs, ncvs)) 
        eris.OEcc = eris.feri1.create_dataset( 'OEcc', (nocc_b,  nvir_b, ncvs, ncvs   ), 'f8', chunks=(nocc_b,  1, ncvs, ncvs)) 
        eris.OEcv = eris.feri1.create_dataset( 'OEcv', (nocc_b,  nvir_b, ncvs, nval_a ), 'f8', chunks=(nocc_b,  1, ncvs, nval_a )) 
        eris.oeCC = eris.feri1.create_dataset( 'oeCC', (nocc_a,  nvir_a, ncvs, ncvs   ), 'f8', chunks=(nocc_a,  1, ncvs, ncvs)) 
        eris.oeCV = eris.feri1.create_dataset( 'oeCV', (nocc_a,  nvir_a, ncvs, nval_b ), 'f8', chunks=(nocc_a,  1, ncvs, nval_b )) 
        eris.OECV = eris.feri1.create_dataset( 'OECV', (nocc_b,  nvir_b, ncvs, nval_b ), 'f8', chunks=(nocc_b,  1, ncvs, nval_b )) 
        eris.ceco = eris.feri1.create_dataset( 'ceco', (ncvs, nvir_a,  ncvs, nocc_a   ), 'f8', chunks=(ncvs, 1,  ncvs, nocc_a)) 
        eris.cevo = eris.feri1.create_dataset( 'cevo', (ncvs, nvir_a,  nval_a , nocc_a), 'f8', chunks=(ncvs, 1,  nval_a , nocc_a)) 
        eris.CECO = eris.feri1.create_dataset( 'CECO', (ncvs, nvir_b,  ncvs, nocc_b   ), 'f8', chunks=(ncvs, 1,  ncvs, nocc_b)) 
        eris.CEVO = eris.feri1.create_dataset( 'CEVO', (ncvs, nvir_b,  nval_b , nocc_b), 'f8', chunks=(ncvs, 1,  nval_b , nocc_b)) 
        eris.CEco = eris.feri1.create_dataset( 'CEco', (ncvs, nvir_b,  ncvs, nocc_a   ), 'f8', chunks=(ncvs, 1,  ncvs, nocc_a)) 
        eris.CEvo = eris.feri1.create_dataset( 'CEvo', (ncvs, nvir_b,  nval_a , nocc_a), 'f8', chunks=(ncvs, 1,  nval_a , nocc_a)) 
        eris.ceCO = eris.feri1.create_dataset( 'ceCO', (ncvs, nvir_a,  ncvs, nocc_b   ), 'f8', chunks=(ncvs, 1,  ncvs, nocc_b)) 
        eris.ceVO = eris.feri1.create_dataset( 'ceVO', (ncvs, nvir_a,  nval_b , nocc_b), 'f8', chunks=(ncvs, 1,  nval_b , nocc_b)) 

        eris.oecc[:] = lib.ddot(eris.L_oe.T,eris.L_cc).reshape((nocc_a,  nvir_a, ncvs, ncvs   ))  
        eris.oecv[:] = lib.ddot(eris.L_oe.T,eris.L_cv).reshape((nocc_a,  nvir_a, ncvs, nval_a ))
        eris.OECC[:] = lib.ddot(eris.L_OE.T,eris.L_CC).reshape((nocc_b,  nvir_b, ncvs, ncvs   ))
        eris.OEcc[:] = lib.ddot(eris.L_OE.T,eris.L_cc).reshape((nocc_b,  nvir_b, ncvs, ncvs   ))
        eris.OEcv[:] = lib.ddot(eris.L_OE.T,eris.L_cv).reshape((nocc_b,  nvir_b, ncvs, nval_a ))
        eris.oeCC[:] = lib.ddot(eris.L_oe.T,eris.L_CC).reshape((nocc_a,  nvir_a, ncvs, ncvs   ))
        eris.oeCV[:] = lib.ddot(eris.L_oe.T,eris.L_CV).reshape((nocc_a,  nvir_a, ncvs, nval_b ))
        eris.OECV[:] = lib.ddot(eris.L_OE.T,eris.L_CV).reshape((nocc_b,  nvir_b, ncvs, nval_b ))
        eris.ceco[:] = lib.ddot(eris.L_ce.T,eris.L_co).reshape((ncvs, nvir_a,  ncvs, nocc_a   ))
        eris.cevo[:] = lib.ddot(eris.L_ce.T,eris.L_vo).reshape((ncvs, nvir_a,  nval_a , nocc_a))
        eris.CECO[:] = lib.ddot(eris.L_CE.T,eris.L_CO).reshape((ncvs, nvir_b,  ncvs, nocc_b   ))
        eris.CEVO[:] = lib.ddot(eris.L_CE.T,eris.L_VO).reshape((ncvs, nvir_b,  nval_b , nocc_b))
        eris.CEco[:] = lib.ddot(eris.L_CE.T,eris.L_co).reshape((ncvs, nvir_b,  ncvs, nocc_a   ))
        eris.CEvo[:] = lib.ddot(eris.L_CE.T,eris.L_vo).reshape((ncvs, nvir_b,  nval_a , nocc_a))
        eris.ceCO[:] = lib.ddot(eris.L_ce.T,eris.L_CO).reshape((ncvs, nvir_a,  ncvs, nocc_b   ))
        eris.ceVO[:] = lib.ddot(eris.L_ce.T,eris.L_VO).reshape((ncvs, nvir_a,  nval_b , nocc_b))
 
        # Addtional CVS integrals for get_imds function (c: core, e: external, o: all occupied orbitals)
        eris.cooo = eris.feri1.create_dataset( 'cooo', (ncvs, nocc_a,  nocc_a,  nocc_a), 'f8') 
        eris.ccoo = eris.feri1.create_dataset( 'ccoo', (ncvs, ncvs, nocc_a,  nocc_a   ), 'f8') 
        eris.COOO = eris.feri1.create_dataset( 'COOO', (ncvs, nocc_b,  nocc_b,  nocc_b), 'f8') 
        eris.CCOO = eris.feri1.create_dataset( 'CCOO', (ncvs, ncvs, nocc_b,  nocc_b   ), 'f8') 
        eris.ccOO = eris.feri1.create_dataset( 'ccOO', (ncvs, ncvs, nocc_b,  nocc_b   ), 'f8') 
        eris.ooCC = eris.feri1.create_dataset( 'ooCC', (nocc_a,  nocc_a,  ncvs, ncvs  ), 'f8') 
        eris.coOO = eris.feri1.create_dataset( 'coOO', (ncvs, nocc_a,  nocc_b,  nocc_b), 'f8') 
        eris.coee = eris.feri1.create_dataset( 'coee', (ncvs, nocc_a,  nvir_a,  nvir_a), 'f8', chunks=(ncvs, nocc_a, 1,  nvir_a )) 
        eris.COEE = eris.feri1.create_dataset( 'COEE', (ncvs, nocc_b,  nvir_b,  nvir_b), 'f8', chunks=(ncvs, nocc_b, 1,  nvir_b )) 
        eris.coEE = eris.feri1.create_dataset( 'coEE', (ncvs, nocc_a,  nvir_b,  nvir_b), 'f8', chunks=(ncvs, nocc_a, 1,  nvir_b )) 
        eris.COee = eris.feri1.create_dataset( 'COee', (ncvs, nocc_b,  nvir_a,  nvir_a), 'f8', chunks=(ncvs, nocc_b, 1,  nvir_a )) 
        eris.ceoe = eris.feri1.create_dataset( 'ceoe', (ncvs, nvir_a, nocc_a , nvir_a ), 'f8', chunks=(ncvs, 1,  nocc_a , nvir_a)) 
        eris.CEOE = eris.feri1.create_dataset( 'CEOE', (ncvs, nvir_b, nocc_b , nvir_b ), 'f8', chunks=(ncvs, 1,  nocc_b , nvir_b)) 
        eris.ceOE = eris.feri1.create_dataset( 'ceOE', (ncvs, nvir_a, nocc_b , nvir_b ), 'f8', chunks=(ncvs, 1,  nocc_b , nvir_b)) 
        eris.oeCE = eris.feri1.create_dataset( 'oeCE', (nocc_a,  nvir_a, ncvs, nvir_b ), 'f8', chunks=(nocc_a,  1,  ncvs, nvir_b)) 
        eris.coco = eris.feri1.create_dataset( 'coco', (ncvs, nocc_a, ncvs, nocc_a    ), 'f8') 
        eris.COCO = eris.feri1.create_dataset( 'COCO', (ncvs, nocc_b, ncvs, nocc_b    ), 'f8') 
        eris.ooCO = eris.feri1.create_dataset( 'ooCO', (nocc_a,  nocc_a, ncvs, nocc_b ), 'f8') 

        eris.cooo[:] = lib.ddot(eris.L_co.T,eris.L_oo).reshape((ncvs, nocc_a,  nocc_a,  nocc_a)) 
        eris.ccoo[:] = lib.ddot(eris.L_cc.T,eris.L_oo).reshape((ncvs, ncvs, nocc_a,  nocc_a   ))
        eris.COOO[:] = lib.ddot(eris.L_CO.T,eris.L_OO).reshape((ncvs, nocc_b,  nocc_b,  nocc_b))
        eris.CCOO[:] = lib.ddot(eris.L_CC.T,eris.L_OO).reshape((ncvs, ncvs, nocc_b,  nocc_b   ))
        eris.ccOO[:] = lib.ddot(eris.L_cc.T,eris.L_OO).reshape((ncvs, ncvs, nocc_b,  nocc_b   ))
        eris.ooCC[:] = lib.ddot(eris.L_oo.T,eris.L_CC).reshape((nocc_a,  nocc_a,  ncvs, ncvs  ))
        eris.coOO[:] = lib.ddot(eris.L_co.T,eris.L_OO).reshape((ncvs, nocc_a,  nocc_b,  nocc_b))
        eris.coee[:] = lib.ddot(eris.L_co.T,eris.L_ee).reshape((ncvs, nocc_a,  nvir_a,  nvir_a))
        eris.COEE[:] = lib.ddot(eris.L_CO.T,eris.L_EE).reshape((ncvs, nocc_b,  nvir_b,  nvir_b))
        eris.coEE[:] = lib.ddot(eris.L_co.T,eris.L_EE).reshape((ncvs, nocc_a,  nvir_b,  nvir_b))
        eris.COee[:] = lib.ddot(eris.L_CO.T,eris.L_ee).reshape((ncvs, nocc_b,  nvir_a,  nvir_a))
        eris.ceoe[:] = lib.ddot(eris.L_ce.T,eris.L_oe).reshape((ncvs, nvir_a, nocc_a , nvir_a ))
        eris.CEOE[:] = lib.ddot(eris.L_CE.T,eris.L_OE).reshape((ncvs, nvir_b, nocc_b , nvir_b ))
        eris.ceOE[:] = lib.ddot(eris.L_ce.T,eris.L_OE).reshape((ncvs, nvir_a, nocc_b , nvir_b ))
        eris.oeCE[:] = lib.ddot(eris.L_oe.T,eris.L_CE).reshape((nocc_a,  nvir_a, ncvs, nvir_b ))
        eris.coco[:] = lib.ddot(eris.L_co.T,eris.L_co).reshape((ncvs, nocc_a, ncvs, nocc_a    ))
        eris.COCO[:] = lib.ddot(eris.L_CO.T,eris.L_CO).reshape((ncvs, nocc_b, ncvs, nocc_b    ))
        eris.ooCO[:] = lib.ddot(eris.L_oo.T,eris.L_CO).reshape((nocc_a,  nocc_a, ncvs, nocc_b ))

    log.timer('DF-ADC integral transformation', *cput0)

    return eris

def calculate_chunk_size(myadc):

    avail_mem = (myadc.max_memory - lib.current_memory()[0]) * 0.25 
    vvv_mem = (myadc._nvir[0]**3) * 8/1e6

    chnk_size =  int(avail_mem/vvv_mem)

    if chnk_size <= 0 :
        chnk_size = 1

    return chnk_size
