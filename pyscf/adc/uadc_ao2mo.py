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
    ncvs = myadc.ncvs
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
    
    #----- ADC(2) integrals --------
    eris.cecc = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_a, core_a), compact=False).reshape(ncvs, nvir_a, ncvs, ncvs).copy()
    eris.cevc = ao2mo.general(myadc._scf._eri, (core_a, vir_a, val_a, core_a), compact=False).reshape(ncvs, nvir_a, nval_a, ncvs).copy()
    eris.vecc = ao2mo.general(myadc._scf._eri, (val_a, vir_a, core_a, core_a), compact=False).reshape(nval_a, nvir_a, ncvs, ncvs).copy()
    eris.CECC = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_b, core_b), compact=False).reshape(ncvs, nvir_b, ncvs, ncvs).copy()
    eris.CEVC = ao2mo.general(myadc._scf._eri, (core_b, vir_b, val_b, core_b), compact=False).reshape(ncvs, nvir_b, nval_b, ncvs).copy()
    eris.VECC = ao2mo.general(myadc._scf._eri, (val_b, vir_b, core_b, core_b), compact=False).reshape(nval_b, nvir_b, ncvs, ncvs).copy()
    eris.ceCC = ao2mo.general(myadc._scf._eri, (core_a, vir_a, core_b, core_b), compact=False).reshape(ncvs, nvir_a, ncvs, ncvs).copy()
    eris.ceVC = ao2mo.general(myadc._scf._eri, (core_a, vir_a, val_b, core_b), compact=False).reshape(ncvs, nvir_a, nval_b, ncvs).copy()
    eris.veCC = ao2mo.general(myadc._scf._eri, (val_a, vir_a, core_b, core_b), compact=False).reshape(nval_a, nvir_a, ncvs, ncvs).copy()
    eris.CEcc = ao2mo.general(myadc._scf._eri, (core_b, vir_b, core_a, core_a), compact=False).reshape(ncvs, nvir_b, ncvs, ncvs).copy()
    eris.CEvc = ao2mo.general(myadc._scf._eri, (core_b, vir_b, val_a, core_a), compact=False).reshape(ncvs, nvir_b, nval_a, ncvs).copy()
    eris.VEcc = ao2mo.general(myadc._scf._eri, (val_b, vir_b, core_a, core_a), compact=False).reshape(nval_b, nvir_b, ncvs, ncvs).copy()

    #----- ADC(2)-x integrals --------
    eris.cccc = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_a, core_a), compact=False).reshape(ncvs, ncvs, ncvs, ncvs).copy()
    eris.cccv = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_a, val_a), compact=False).reshape(ncvs, ncvs, ncvs, nval_a).copy()
    eris.cvvc = ao2mo.general(myadc._scf._eri, (core_a, val_a, val_a, core_a), compact=False).reshape(ncvs, nval_a, nval_a, ncvs).copy()
    eris.ccvv = ao2mo.general(myadc._scf._eri, (core_a, core_a, val_a, val_a), compact=False).reshape(ncvs, ncvs, nval_a, nval_a).copy()
    eris.CCCC = ao2mo.general(myadc._scf._eri, (core_b, core_b, core_b, core_b), compact=False).reshape(ncvs, ncvs, ncvs, ncvs).copy()
    eris.CCCV = ao2mo.general(myadc._scf._eri, (core_b, core_b, core_b, val_b), compact=False).reshape(ncvs, ncvs, ncvs, nval_b).copy()
    eris.CVVC = ao2mo.general(myadc._scf._eri, (core_b, val_b, val_b, core_b), compact=False).reshape(ncvs, nval_b, nval_b, ncvs).copy()
    eris.CCVV = ao2mo.general(myadc._scf._eri, (core_b, core_b, val_b, val_b), compact=False).reshape(ncvs, ncvs, nval_b, nval_b).copy()
    eris.ccCC = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_b, core_b), compact=False).reshape(ncvs, ncvs, ncvs, ncvs).copy()
    eris.vcCC = ao2mo.general(myadc._scf._eri, (val_a, core_a, core_b, core_b), compact=False).reshape(nval_a, ncvs, ncvs, ncvs).copy()
    eris.ccCV = ao2mo.general(myadc._scf._eri, (core_a, core_a, core_b, val_b), compact=False).reshape(ncvs, ncvs, ncvs, nval_b).copy()
    eris.vvCC = ao2mo.general(myadc._scf._eri, (val_a, val_a, core_b, core_b), compact=False).reshape(nval_a, nval_a, ncvs, ncvs).copy()
    eris.vcCV = ao2mo.general(myadc._scf._eri, (val_a, core_a, core_b, val_b), compact=False).reshape(nval_a, ncvs, ncvs, nval_b).copy()
    eris.ccVV = ao2mo.general(myadc._scf._eri, (core_a, core_a, val_b, val_b), compact=False).reshape(ncvs, ncvs, nval_b, nval_b).copy()
    eris.ceec = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_a, core_a), compact=False).reshape(ncvs, nvir_a, nvir_a, ncvs).copy()
    eris.veec = ao2mo.general(myadc._scf._eri, (val_a, vir_a, vir_a, core_a), compact=False).reshape(nval_a, nvir_a, nvir_a, ncvs).copy()
    eris.veev = ao2mo.general(myadc._scf._eri, (val_a, vir_a, vir_a, val_a), compact=False).reshape(nval_a, nvir_a, nvir_a, nval_a).copy()
    eris.CEEC = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_b, core_b), compact=False).reshape(ncvs, nvir_b, nvir_b, ncvs).copy()
    eris.VEEC = ao2mo.general(myadc._scf._eri, (val_b, vir_b, vir_b, core_b), compact=False).reshape(nval_b, nvir_b, nvir_b, ncvs).copy()
    eris.VEEV = ao2mo.general(myadc._scf._eri, (val_b, vir_b, vir_b, val_b), compact=False).reshape(nval_b, nvir_b, nvir_b, nval_b).copy()
    eris.CEec = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_a, core_a), compact=False).reshape(ncvs, nvir_b, nvir_a, ncvs).copy()
    eris.VEec = ao2mo.general(myadc._scf._eri, (val_b, vir_b, vir_a, core_a), compact=False).reshape(nval_b, nvir_b, nvir_a, ncvs).copy()
    eris.CEev = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_a, val_a), compact=False).reshape(ncvs, nvir_b, nvir_a, nval_a).copy()
    eris.VEev = ao2mo.general(myadc._scf._eri, (val_b, vir_b, vir_a, val_a), compact=False).reshape(nval_b, nvir_b, nvir_a, nval_a).copy()
    eris.ccee = ao2mo.general(myadc._scf._eri, (core_a, core_a, vir_a, vir_a), compact=False).reshape(ncvs, ncvs, nvir_a, nvir_a).copy()
    eris.vcee = ao2mo.general(myadc._scf._eri, (val_a, core_a, vir_a, vir_a), compact=False).reshape(nval_a, ncvs, nvir_a, nvir_a).copy()
    eris.vvee = ao2mo.general(myadc._scf._eri, (val_a, val_a, vir_a, vir_a), compact=False).reshape(nval_a, nval_a, nvir_a, nvir_a).copy()
    eris.CCEE = ao2mo.general(myadc._scf._eri, (core_b, core_b, vir_b, vir_b), compact=False).reshape(ncvs, ncvs, nvir_b, nvir_b).copy()
    eris.VCEE = ao2mo.general(myadc._scf._eri, (val_b, core_b, vir_b, vir_b), compact=False).reshape(nval_b, ncvs, nvir_b, nvir_b).copy()
    eris.VVEE = ao2mo.general(myadc._scf._eri, (val_b, val_b, vir_b, vir_b), compact=False).reshape(nval_b, nval_b, nvir_b, nvir_b).copy()
    eris.ccEE = ao2mo.general(myadc._scf._eri, (core_a, core_a, vir_b, vir_b), compact=False).reshape(ncvs, ncvs, nvir_b, nvir_b).copy()
    eris.vcEE = ao2mo.general(myadc._scf._eri, (val_a, core_a, vir_b, vir_b), compact=False).reshape(nval_a, ncvs, nvir_b, nvir_b).copy()
    eris.vvEE = ao2mo.general(myadc._scf._eri, (val_a, val_a, vir_b, vir_b), compact=False).reshape(nval_a, nval_a, nvir_b, nvir_b).copy()
    eris.CCee = ao2mo.general(myadc._scf._eri, (core_b, core_b, vir_a, vir_a), compact=False).reshape(ncvs, ncvs, nvir_a, nvir_a).copy()
    eris.VCee = ao2mo.general(myadc._scf._eri, (val_b, core_b, vir_a, vir_a), compact=False).reshape(nval_b, ncvs, nvir_a, nvir_a).copy()
    eris.VVee = ao2mo.general(myadc._scf._eri, (val_b, val_b, vir_a, vir_a), compact=False).reshape(nval_b, nval_b, nvir_a, nvir_a).copy()

    #----- ADC(3) integrals --------
    eris.oecc = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_a, core_a), compact=False).reshape(nocc_a, nvir_a, ncvs, ncvs).copy()
    eris.ceoc = ao2mo.general(myadc._scf._eri, (core_a, vir_a, occ_a, core_a), compact=False).reshape(ncvs, nvir_a, nocc_a, ncvs).copy()
    eris.oecv = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_a, val_a), compact=False).reshape(nocc_a, nvir_a, ncvs, nval_a).copy()
    eris.ceov = ao2mo.general(myadc._scf._eri, (core_a, vir_a, occ_a, val_a), compact=False).reshape(ncvs, nvir_a, nocc_a, nval_a).copy()
    eris.OECC = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_b, core_b), compact=False).reshape(nocc_b, nvir_b, ncvs, ncvs).copy()
    eris.CEOC = ao2mo.general(myadc._scf._eri, (core_b, vir_b, occ_b, core_b), compact=False).reshape(ncvs, nvir_b, nocc_b, ncvs).copy()
    eris.OECV = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_b, val_b), compact=False).reshape(nocc_b, nvir_b, ncvs, nval_b).copy()
    eris.CEOV = ao2mo.general(myadc._scf._eri, (core_b, vir_b, occ_b, val_b), compact=False).reshape(ncvs, nvir_b, nocc_b, nval_b).copy()
    eris.OEcc = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_a, core_a), compact=False).reshape(nocc_b, nvir_b, ncvs, ncvs).copy()
    eris.CEoc = ao2mo.general(myadc._scf._eri, (core_b, vir_b, occ_a, core_a), compact=False).reshape(ncvs, nvir_b, nocc_a, ncvs).copy()
    eris.OEcv = ao2mo.general(myadc._scf._eri, (occ_b, vir_b, core_a, val_a), compact=False).reshape(nocc_b, nvir_b, ncvs, nval_a).copy()
    eris.CEov = ao2mo.general(myadc._scf._eri, (core_b, vir_b, occ_a, val_a), compact=False).reshape(ncvs, nvir_b, nocc_a, nval_a).copy()
    eris.oeCC = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_b, core_b), compact=False).reshape(nocc_a, nvir_a, ncvs, ncvs).copy()
    eris.ceOC = ao2mo.general(myadc._scf._eri, (core_a, vir_a, occ_b, core_b), compact=False).reshape(ncvs, nvir_a, nocc_b, ncvs).copy()
    eris.oeCV = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, core_b, val_b), compact=False).reshape(nocc_a, nvir_a, ncvs, nval_b).copy()
    eris.ceOV = ao2mo.general(myadc._scf._eri, (core_a, vir_a, occ_b, val_b), compact=False).reshape(ncvs, nvir_a, nocc_b, nval_b).copy()
    eris.ceee = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_a, vir_a), compact=True).reshape(ncvs, nvir_a, -1).copy()
    eris.CEEE = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_b, vir_b), compact=True).reshape(ncvs, nvir_b, -1).copy()
    eris.ceEE = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_b, vir_b), compact=True).reshape(ncvs, nvir_a, -1).copy()
    eris.CEee = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_a, vir_a), compact=True).reshape(ncvs, nvir_b, -1).copy()

    # Addtional CVS integrals for get_imds function (c: core, e: external, o: all occupied orbitals)

    eris.ceeo = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_a, occ_a), compact=False).reshape(ncvs, nvir_a, nvir_a, nocc_a).copy()
    eris.CEEO = ao2mo.general(myadc._scf._eri, (core_b, vir_b, vir_b, occ_b), compact=False).reshape(ncvs, nvir_b, nvir_b, nocc_b).copy()
    eris.ocee = ao2mo.general(myadc._scf._eri, (occ_a, core_a, vir_a, vir_a), compact=False).reshape(nocc_a, ncvs, nvir_a, nvir_a).copy()
    eris.OCEE = ao2mo.general(myadc._scf._eri, (occ_b, core_b, vir_b, vir_b), compact=False).reshape(nocc_b, ncvs, nvir_b, nvir_b).copy()
    eris.ocEE = ao2mo.general(myadc._scf._eri, (occ_a, core_a, vir_b, vir_b), compact=False).reshape(nocc_a, ncvs, nvir_b, nvir_b).copy()
    eris.OCee = ao2mo.general(myadc._scf._eri, (occ_b, core_b, vir_a, vir_a), compact=False).reshape(nocc_b, ncvs, nvir_a, nvir_a).copy()
    eris.ceEO = ao2mo.general(myadc._scf._eri, (core_a, vir_a, vir_b, occ_b), compact=False).reshape(ncvs, nvir_a, nvir_b, nocc_b).copy()
    eris.oeEC = ao2mo.general(myadc._scf._eri, (occ_a, vir_a, vir_b, core_b), compact=False).reshape(nocc_a, nvir_a, nvir_b, ncvs).copy()
    eris.cooo = ao2mo.general(myadc._scf._eri, (core_a, occ_a, occ_a, occ_a), compact=False).reshape(ncvs, nocc_a, nocc_a, nocc_a).copy()
    eris.ccoo = ao2mo.general(myadc._scf._eri, (core_a, core_a, occ_a, occ_a), compact=False).reshape(ncvs, ncvs, nocc_a, nocc_a).copy()
    eris.cooc = ao2mo.general(myadc._scf._eri, (core_a, occ_a, occ_a, core_a), compact=False).reshape(ncvs, nocc_a, nocc_a, ncvs).copy()
    eris.COOO = ao2mo.general(myadc._scf._eri, (core_b, occ_b, occ_b, occ_b), compact=False).reshape(ncvs, nocc_b, nocc_b, nocc_b).copy()
    eris.CCOO = ao2mo.general(myadc._scf._eri, (core_b, core_b, occ_b, occ_b), compact=False).reshape(ncvs, ncvs, nocc_b, nocc_b).copy()
    eris.COOC = ao2mo.general(myadc._scf._eri, (core_b, occ_b, occ_b, core_b), compact=False).reshape(ncvs, nocc_b, nocc_b, ncvs).copy()
    eris.ccOO = ao2mo.general(myadc._scf._eri, (core_a, core_a, occ_b, occ_b), compact=False).reshape(ncvs, ncvs, nocc_b, nocc_b).copy()
    eris.ooCC = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, core_b, core_b), compact=False).reshape(nocc_a, nocc_a, ncvs, ncvs).copy()
    eris.coOO = ao2mo.general(myadc._scf._eri, (core_a, occ_a, occ_b, occ_b), compact=False).reshape(ncvs, nocc_a, nocc_b, nocc_b).copy()
    eris.ooOC = ao2mo.general(myadc._scf._eri, (occ_a, occ_a, occ_b, core_b), compact=False).reshape(nocc_a, nocc_a, nocc_b, ncvs).copy()

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


#@profile
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

    ncore_a = ncore_b = ncvs # This is unecessary. ncore_a and ncore_b can be simply replaced with ncvs

    #----- ADC(2) integrals --------
    eris.cecc = eris.feri1.create_dataset( 'cecc', (ncore_a, nvir_a, ncore_a, ncore_a), 'f8', chunks=(ncore_a, 1, ncore_a, ncore_a))
    eris.cevc = eris.feri1.create_dataset( 'cevc', (ncore_a, nvir_a, nval_a,  ncore_a), 'f8', chunks=(ncore_a, 1, nval_a,  ncore_a))
    eris.vecc = eris.feri1.create_dataset( 'vecc', (nval_a,  nvir_a, ncore_a, ncore_a), 'f8', chunks=(nval_a,  1, ncore_a, ncore_a))
    eris.CECC = eris.feri1.create_dataset( 'CECC', (ncore_b, nvir_b, ncore_b, ncore_b), 'f8', chunks=(ncore_b, 1, ncore_b, ncore_b))
    eris.CEVC = eris.feri1.create_dataset( 'CEVC', (ncore_b, nvir_b, nval_b,  ncore_b), 'f8', chunks=(ncore_b, 1, nval_b,  ncore_b))
    eris.VECC = eris.feri1.create_dataset( 'VECC', (nval_b,  nvir_b, ncore_b, ncore_b), 'f8', chunks=(nval_b,  1, ncore_b, ncore_b))
    eris.ceCC = eris.feri1.create_dataset( 'ceCC', (ncore_a, nvir_a, ncore_b, ncore_b), 'f8', chunks=(ncore_a, 1, ncore_b, ncore_b))
    eris.ceVC = eris.feri1.create_dataset( 'ceVC', (ncore_a, nvir_a, nval_b,  ncore_b), 'f8', chunks=(ncore_a, 1, nval_b,  ncore_b))
    eris.veCC = eris.feri1.create_dataset( 'veCC', (nval_a,  nvir_a, ncore_b, ncore_b), 'f8', chunks=(nval_a,  1, ncore_b, ncore_b))
    eris.CEcc = eris.feri1.create_dataset( 'CEcc', (ncore_b, nvir_b, ncore_a, ncore_a), 'f8', chunks=(ncore_b, 1, ncore_a, ncore_a))
    eris.CEvc = eris.feri1.create_dataset( 'CEvc', (ncore_b, nvir_b, nval_a,  ncore_a), 'f8', chunks=(ncore_b, 1, nval_a,  ncore_a))
    eris.VEcc = eris.feri1.create_dataset( 'VEcc', (nval_b,  nvir_b, ncore_a, ncore_a), 'f8', chunks=(nval_b,  1, ncore_a, ncore_a))
    #----- ADC(2)-x integrals --------
    eris.cccc =  eris.feri1.create_dataset( 'cccc', (ncore_a, ncore_a, ncore_a, ncore_a), 'f8') 
    eris.cccv =  eris.feri1.create_dataset( 'cccv', (ncore_a, ncore_a, ncore_a, nval_a ), 'f8') 
    eris.cvvc =  eris.feri1.create_dataset( 'cvvc', (ncore_a, nval_a,  nval_a,  ncore_a), 'f8') 
    eris.ccvv =  eris.feri1.create_dataset( 'ccvv', (ncore_a, ncore_a, nval_a,  nval_a ), 'f8') 
    eris.CCCC =  eris.feri1.create_dataset( 'CCCC', (ncore_b, ncore_b, ncore_b, ncore_b), 'f8') 
    eris.CCCV =  eris.feri1.create_dataset( 'CCCV', (ncore_b, ncore_b, ncore_b, nval_b ), 'f8') 
    eris.CVVC =  eris.feri1.create_dataset( 'CVVC', (ncore_b, nval_b,  nval_b,  ncore_b), 'f8') 
    eris.CCVV =  eris.feri1.create_dataset( 'CCVV', (ncore_b, ncore_b, nval_b,  nval_b ), 'f8') 
    eris.ccCC =  eris.feri1.create_dataset( 'ccCC', (ncore_a, ncore_a, ncore_b, ncore_b), 'f8') 
    eris.vcCC =  eris.feri1.create_dataset( 'vcCC', (nval_a,  ncore_a, ncore_b, ncore_b), 'f8') 
    eris.ccCV =  eris.feri1.create_dataset( 'ccCV', (ncore_a, ncore_a, ncore_b, nval_b ), 'f8') 
    eris.vvCC =  eris.feri1.create_dataset( 'vvCC', (nval_a,  nval_a,  ncore_b, ncore_b), 'f8') 
    eris.vcCV =  eris.feri1.create_dataset( 'vcCV', (nval_a,  ncore_a, ncore_b, nval_b ), 'f8') 
    eris.ccVV =  eris.feri1.create_dataset( 'ccVV', (ncore_a, ncore_a, nval_b,  nval_b ), 'f8') 
    eris.ceec =  eris.feri1.create_dataset( 'ceec', (ncore_a, nvir_a,  nvir_a,  ncore_a), 'f8', chunks=(ncore_a, 1,  nvir_a,  ncore_a)) 
    eris.veec =  eris.feri1.create_dataset( 'veec', (nval_a,  nvir_a,  nvir_a,  ncore_a), 'f8', chunks=(nval_a,  1,  nvir_a,  ncore_a)) 
    eris.veev =  eris.feri1.create_dataset( 'veev', (nval_a,  nvir_a,  nvir_a,  nval_a ), 'f8', chunks=(nval_a,  1,  nvir_a,  nval_a )) 
    eris.CEEC =  eris.feri1.create_dataset( 'CEEC', (ncore_b, nvir_b,  nvir_b,  ncore_b), 'f8', chunks=(ncore_b, 1,  nvir_b,  ncore_b)) 
    eris.VEEC =  eris.feri1.create_dataset( 'VEEC', (nval_b,  nvir_b,  nvir_b,  ncore_b), 'f8', chunks=(nval_b,  1,  nvir_b,  ncore_b)) 
    eris.VEEV =  eris.feri1.create_dataset( 'VEEV', (nval_b,  nvir_b,  nvir_b,  nval_b ), 'f8', chunks=(nval_b,  1,  nvir_b,  nval_b )) 
    eris.CEec =  eris.feri1.create_dataset( 'CEec', (ncore_b, nvir_b,  nvir_a,  ncore_a), 'f8', chunks=(ncore_b, 1,  nvir_a,  ncore_a)) 
    eris.VEec =  eris.feri1.create_dataset( 'VEec', (nval_b,  nvir_b,  nvir_a,  ncore_a), 'f8', chunks=(nval_b,  1,  nvir_a,  ncore_a)) 
    eris.CEev =  eris.feri1.create_dataset( 'CEev', (ncore_b, nvir_b,  nvir_a,  nval_a ), 'f8', chunks=(ncore_b, 1,  nvir_a,  nval_a )) 
    eris.VEev =  eris.feri1.create_dataset( 'VEev', (nval_b,  nvir_b,  nvir_a,  nval_a ), 'f8', chunks=(nval_b,  1,  nvir_a,  nval_a )) 
    eris.ccee =  eris.feri1.create_dataset( 'ccee', (ncore_a, ncore_a, nvir_a,  nvir_a ), 'f8', chunks=(ncore_a, ncore_a, 1,  nvir_a )) 
    eris.vcee =  eris.feri1.create_dataset( 'vcee', (nval_a,  ncore_a, nvir_a,  nvir_a ), 'f8', chunks=(nval_a,  ncore_a, 1,  nvir_a )) 
    eris.vvee =  eris.feri1.create_dataset( 'vvee', (nval_a,  nval_a,  nvir_a,  nvir_a ), 'f8', chunks=(nval_a,  nval_a,  1,  nvir_a )) 
    eris.CCEE =  eris.feri1.create_dataset( 'CCEE', (ncore_b, ncore_b, nvir_b,  nvir_b ), 'f8', chunks=(ncore_b, ncore_b, 1,  nvir_b )) 
    eris.VCEE =  eris.feri1.create_dataset( 'VCEE', (nval_b,  ncore_b, nvir_b,  nvir_b ), 'f8', chunks=(nval_b,  ncore_b, 1,  nvir_b )) 
    eris.VVEE =  eris.feri1.create_dataset( 'VVEE', (nval_b,  nval_b,  nvir_b,  nvir_b ), 'f8', chunks=(nval_b,  nval_b,  1,  nvir_b )) 
    eris.ccEE =  eris.feri1.create_dataset( 'ccEE', (ncore_a, ncore_a, nvir_b,  nvir_b ), 'f8', chunks=(ncore_a, ncore_a, 1,  nvir_b )) 
    eris.vcEE =  eris.feri1.create_dataset( 'vcEE', (nval_a,  ncore_a, nvir_b,  nvir_b ), 'f8', chunks=(nval_a,  ncore_a, 1,  nvir_b )) 
    eris.vvEE =  eris.feri1.create_dataset( 'vvEE', (nval_a,  nval_a,  nvir_b,  nvir_b),  'f8', chunks=(nval_a,  nval_a,  1,  nvir_b),) 
    eris.CCee =  eris.feri1.create_dataset( 'CCee', (ncore_b, ncore_b, nvir_a,  nvir_a),  'f8', chunks=(ncore_b, ncore_b, 1,  nvir_a)) 
    eris.VCee =  eris.feri1.create_dataset( 'VCee', (nval_b,  ncore_b, nvir_a,  nvir_a),  'f8', chunks=(nval_b,  ncore_b, 1,  nvir_a)) 
    eris.VVee =  eris.feri1.create_dataset( 'VVee', (nval_b,  nval_b,  nvir_a,  nvir_a),  'f8', chunks=(nval_b,  nval_b,  1,  nvir_a)) 
    #----- ADC(3) integrals --------
    eris.oecc = eris.feri1.create_dataset( 'oecc', (nocc_a,  nvir_a, ncore_a, ncore_a), 'f8', chunks=(nocc_a,  1, ncore_a, ncore_a)) 
    eris.ceoc = eris.feri1.create_dataset( 'ceoc', (ncore_a, nvir_a, nocc_a,  ncore_a), 'f8', chunks=(ncore_a, 1, nocc_a,  ncore_a)) 
    eris.oecv = eris.feri1.create_dataset( 'oecv', (nocc_a,  nvir_a, ncore_a, nval_a ), 'f8', chunks=(nocc_a,  1, ncore_a, nval_a )) 
    eris.ceov = eris.feri1.create_dataset( 'ceov', (ncore_a, nvir_a, nocc_a,  nval_a ), 'f8', chunks=(ncore_a, 1, nocc_a,  nval_a )) 
    eris.OECC = eris.feri1.create_dataset( 'OECC', (nocc_b,  nvir_b, ncore_b, ncore_b), 'f8', chunks=(nocc_b,  1, ncore_b, ncore_b)) 
    eris.CEOC = eris.feri1.create_dataset( 'CEOC', (ncore_b, nvir_b, nocc_b,  ncore_b), 'f8', chunks=(ncore_b, 1, nocc_b,  ncore_b)) 
    eris.OECV = eris.feri1.create_dataset( 'OECV', (nocc_b,  nvir_b, ncore_b, nval_b ), 'f8', chunks=(nocc_b,  1, ncore_b, nval_b )) 
    eris.CEOV = eris.feri1.create_dataset( 'CEOV', (ncore_b, nvir_b, nocc_b,  nval_b ), 'f8', chunks=(ncore_b, 1, nocc_b,  nval_b )) 
    eris.OEcc = eris.feri1.create_dataset( 'OEcc', (nocc_b,  nvir_b, ncore_a, ncore_a), 'f8', chunks=(nocc_b,  1, ncore_a, ncore_a)) 
    eris.CEoc = eris.feri1.create_dataset( 'CEoc', (ncore_b, nvir_b, nocc_a,  ncore_a), 'f8', chunks=(ncore_b, 1, nocc_a,  ncore_a)) 
    eris.OEcv = eris.feri1.create_dataset( 'OEcv', (nocc_b,  nvir_b, ncore_a, nval_a ), 'f8', chunks=(nocc_b,  1, ncore_a, nval_a )) 
    eris.CEov = eris.feri1.create_dataset( 'CEov', (ncore_b, nvir_b, nocc_a,  nval_a ), 'f8', chunks=(ncore_b, 1, nocc_a,  nval_a )) 
    eris.oeCC = eris.feri1.create_dataset( 'oeCC', (nocc_a,  nvir_a, ncore_b, ncore_b), 'f8', chunks=(nocc_a,  1, ncore_b, ncore_b)) 
    eris.ceOC = eris.feri1.create_dataset( 'ceOC', (ncore_a, nvir_a, nocc_b,  ncore_b), 'f8', chunks=(ncore_a, 1, nocc_b,  ncore_b)) 
    eris.oeCV = eris.feri1.create_dataset( 'oeCV', (nocc_a,  nvir_a, ncore_b, nval_b ), 'f8', chunks=(nocc_a,  1, ncore_b, nval_b )) 
    eris.ceOV = eris.feri1.create_dataset( 'ceOV', (ncore_a, nvir_a, nocc_b,  nval_b ), 'f8', chunks=(ncore_a, 1, nocc_b,  nval_b )) 
    eris.ceee = eris.feri1.create_dataset( 'ceee', (ncore_a, nvir_a, nvpair_a ), 'f8') 
    eris.CEEE = eris.feri1.create_dataset( 'CEEE', (ncore_b, nvir_b, nvpair_b ), 'f8') 
    eris.ceEE = eris.feri1.create_dataset( 'ceEE', (ncore_a, nvir_a, nvpair_b ), 'f8') 
    eris.CEee = eris.feri1.create_dataset( 'CEee', (ncore_b, nvir_b, nvpair_a ), 'f8') 
    # Addtional CVS integrals for get_imds function (c: core, e: external, o: all occupied orbitals)
    eris.ceeo = eris.feri1.create_dataset( 'ceeo', (ncore_a, nvir_a,  nvir_a,  nocc_a ), 'f8', chunks=(ncore_a, 1,  nvir_a,  nocc_a )) 
    eris.CEEO = eris.feri1.create_dataset( 'CEEO', (ncore_b, nvir_b,  nvir_b,  nocc_b ), 'f8', chunks=(ncore_b, 1,  nvir_b,  nocc_b )) 
    eris.ocee = eris.feri1.create_dataset( 'ocee', (nocc_a,  ncore_a, nvir_a,  nvir_a ), 'f8', chunks=(nocc_a,  ncore_a, 1,  nvir_a )) 
    eris.OCEE = eris.feri1.create_dataset( 'OCEE', (nocc_b,  ncore_b, nvir_b,  nvir_b ), 'f8', chunks=(nocc_b,  ncore_b, 1,  nvir_b )) 
    eris.ocEE = eris.feri1.create_dataset( 'ocEE', (nocc_a,  ncore_a, nvir_b,  nvir_b ), 'f8', chunks=(nocc_a,  ncore_a, 1,  nvir_b )) 
    eris.OCee = eris.feri1.create_dataset( 'OCee', (nocc_b,  ncore_b, nvir_a,  nvir_a ), 'f8', chunks=(nocc_b,  ncore_b, 1,  nvir_a )) 
    eris.ceEO = eris.feri1.create_dataset( 'ceEO', (ncore_a, nvir_a,  nvir_b,  nocc_b ), 'f8', chunks=(ncore_a, 1,  nvir_b,  nocc_b )) 
    eris.oeEC = eris.feri1.create_dataset( 'oeEC', (nocc_a,  nvir_a,  nvir_b,  ncore_b), 'f8', chunks=(nocc_a,  1,  nvir_b,  ncore_b)) 
    eris.cooo = eris.feri1.create_dataset( 'cooo', (ncore_a, nocc_a,  nocc_a,  nocc_a ), 'f8') 
    eris.ccoo = eris.feri1.create_dataset( 'ccoo', (ncore_a, ncore_a, nocc_a,  nocc_a ), 'f8') 
    eris.cooc = eris.feri1.create_dataset( 'cooc', (ncore_a, nocc_a,  nocc_a,  ncore_a), 'f8') 
    eris.COOO = eris.feri1.create_dataset( 'COOO', (ncore_b, nocc_b,  nocc_b,  nocc_b ), 'f8') 
    eris.CCOO = eris.feri1.create_dataset( 'CCOO', (ncore_b, ncore_b, nocc_b,  nocc_b ), 'f8') 
    eris.COOC = eris.feri1.create_dataset( 'COOC', (ncore_b, nocc_b,  nocc_b,  ncore_b), 'f8') 
    eris.ccOO = eris.feri1.create_dataset( 'ccOO', (ncore_a, ncore_a, nocc_b,  nocc_b ), 'f8') 
    eris.ooCC = eris.feri1.create_dataset( 'ooCC', (nocc_a,  nocc_a,  ncore_b, ncore_b), 'f8') 
    eris.coOO = eris.feri1.create_dataset( 'coOO', (ncore_a, nocc_a,  nocc_b,  nocc_b ), 'f8') 
    eris.ooOC = eris.feri1.create_dataset( 'ooOC', (nocc_a,  nocc_a,  nocc_b,  ncore_b), 'f8') 
    

    cput1 = time.clock(), time.time()
    mol = myadc.mol
    tmpf = lib.H5TmpFile()
    nval_a_s = slice(ncvs,nocc_a)
    nval_b_s = slice(ncvs,nocc_b)

    ao2mo.general(mol, (occ_a,mo_a,mo_a,mo_a), tmpf, 'aa')
    buf = np.empty((nmo_a,nmo_a,nmo_a))
    for i in range(nocc_a):
        lib.unpack_tril(tmpf['aa'][i*nmo_a:(i+1)*nmo_a], out=buf)
        eris.oooo[i] = buf[:nocc_a,:nocc_a,:nocc_a]
        eris.ovoo[i] = buf[nocc_a:,:nocc_a,:nocc_a]
        eris.oovv[i] = buf[:nocc_a,nocc_a:,nocc_a:]
        eris.ovvo[i] = buf[nocc_a:,nocc_a:,:nocc_a]
        eris.ovvv[i] = lib.pack_tril(buf[nocc_a:,nocc_a:,nocc_a:])

        if myadc.method_type == 'ip-cvs':
            eris.oecc[i] = buf[nocc_a:,:ncvs,:ncvs]    
            eris.oecv[i] = buf[nocc_a:,:ncvs,nval_a_s]    
            eris.ocee[i] = buf[:ncvs,nocc_a:,nocc_a:]
            if i <  ncvs:
                eris.cecc[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.cevc[i] = buf[nocc_a:,nval_a_s,:ncvs]
                eris.ceeo[i] = buf[nocc_a:,nocc_a:,:nocc_a]
                eris.cccc[i] = buf[:ncvs,:ncvs,:ncvs]   
                eris.cccv[i] = buf[:ncvs,:ncvs,nval_a_s]
                eris.cvvc[i] = buf[nval_a_s,nval_a_s,:ncvs]
                eris.ccvv[i] = buf[:ncvs,nval_a_s,nval_a_s]
                eris.ceec[i] = buf[nocc_a:,nocc_a:,:ncvs]
                eris.ccee[i] = buf[:ncvs,nocc_a:,nocc_a:]
                eris.ceoc[i] = buf[nocc_a:,:nocc_a,:ncvs]  
                eris.ceov[i] = buf[nocc_a:,:nocc_a,nval_a_s]    
                eris.ceeo[i] = buf[nocc_a:,nocc_a:,:nocc_a]
                eris.cooo[i] = buf[:nocc_a,:nocc_a,:nocc_a]
                eris.ccoo[i] = buf[:ncvs,:nocc_a,:nocc_a]
                eris.cooc[i] = buf[:nocc_a,:nocc_a,:ncvs]
                eris.ceee[i] = lib.pack_tril(buf[nocc_a:,nocc_a:,nocc_a:])
            if i >= ncvs:
                i -= ncvs
                eris.vecc[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.veec[i] = buf[nocc_a:,nocc_a:,:ncvs] 
                eris.veev[i] = buf[nocc_a:,nocc_a:,nval_a_s]
                eris.vcee[i] = buf[:ncvs,nocc_a:,nocc_a:]
                eris.vvee[i] = buf[nval_a_s,nocc_a:,nocc_a:]
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
            eris.OCEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
            if i < ncvs:
                eris.CECC[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.CEVC[i] = buf[nocc_b:,nval_b_s,:ncvs]
                eris.CEEO[i] = buf[nocc_b:,nocc_b:,:nocc_b]
                eris.CCCC[i] = buf[:ncvs,:ncvs,:ncvs] 
                eris.CCCV[i] = buf[:ncvs,:ncvs,nval_b_s]
                eris.CVVC[i] = buf[nval_b_s,nval_b_s,:ncvs]
                eris.CCVV[i] = buf[:ncvs,nval_b_s,nval_b_s]
                eris.CEEC[i] = buf[nocc_b:,nocc_b:,:ncvs]
                eris.CCEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
                eris.CEOC[i] = buf[nocc_b:,:nocc_b:,:ncvs]     
                eris.CEOV[i] = buf[nocc_b:,:nocc_b,nval_b_s]    
                eris.CEEO[i] = buf[nocc_b:,nocc_b:,:nocc_b]
                eris.COOO[i] = buf[:nocc_b,:nocc_b,:nocc_b]
                eris.CCOO[i] = buf[:ncvs,:nocc_b,:nocc_b]
                eris.COOC[i] = buf[:nocc_b,:nocc_b,:ncvs]
                eris.CEEE[i] = lib.pack_tril(buf[nocc_b:,nocc_b:,nocc_b:])
            if i >= ncvs:
                i -= ncvs
                eris.VECC[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.VEEC[i] = buf[nocc_b:,nocc_b:,:ncvs] 
                eris.VEEV[i] = buf[nocc_b:,nocc_b:,nval_b_s]
                eris.VCEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
                eris.VVEE[i] = buf[nval_b_s,nocc_b:,nocc_b:]
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
            eris.oeEC[i] = buf[nocc_a:,nocc_b:,:ncvs]
            eris.oeCC[i] = buf[nocc_a:,:ncvs,:ncvs] 
            eris.oeCV[i] = buf[nocc_a:,:ncvs,nval_b_s]
            eris.ooCC[i] = buf[:nocc_a,:ncvs,:ncvs]
            eris.ooOC[i] = buf[:nocc_a,:nocc_b,:ncvs]
            eris.ocEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
            eris.oeEC[i] = buf[nocc_a:,nocc_b:,:ncvs]
            if i < ncvs:
                eris.ceCC[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.ceVC[i] = buf[nocc_a:,nval_b_s,:ncvs]
                eris.ceEO[i] = buf[nocc_a:,nocc_b:,:nocc_b]
                eris.ccCC[i] = buf[:ncvs,:ncvs,:ncvs] 
                eris.ccCV[i] = buf[:ncvs,:ncvs,nval_b_s]
                eris.ccVV[i] = buf[:ncvs,nval_b_s,nval_b_s]
                eris.ccEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
                eris.ceOC[i] = buf[nocc_a:,:nocc_b,:ncvs] 
                eris.ceOV[i] = buf[nocc_a:,:nocc_b,nval_b_s]
                eris.ccOO[i] = buf[:ncvs,:nocc_b,:nocc_b]
                eris.coOO[i] = buf[:nocc_a,:nocc_b,:nocc_b]
                eris.ceEO[i] = buf[nocc_a:,nocc_b:,:nocc_b]
                eris.ceEE[i] = lib.pack_tril(buf[nocc_a:,nocc_b:,nocc_b:])
            if i >= ncvs:
                i -= ncvs
                eris.veCC[i] = buf[nocc_a:,:ncvs,:ncvs]
                eris.vcCC[i] = buf[:ncvs,:ncvs,:ncvs] 
                eris.vvCC[i] = buf[nval_a_s,:ncvs,:ncvs]
                eris.vcCV[i] = buf[:ncvs,:ncvs,nval_b_s]
                eris.vcEE[i] = buf[:ncvs,nocc_b:,nocc_b:]
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
            eris.OCee[i] = buf[:ncvs,nocc_a:,nocc_a:]
            if i < ncvs:
                eris.CEcc[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.CEvc[i] = buf[nocc_b:,nval_a_s,:ncvs]
                eris.CEec[i] = buf[nocc_b:,nocc_a:,:ncvs] 
                eris.CEev[i] = buf[nocc_b:,nocc_a:,nval_a_s]
                eris.CCee[i] = buf[:ncvs,nocc_a:,nocc_a:]
                eris.CEoc[i] = buf[nocc_b:,:nocc_a,:ncvs]     
                eris.CEov[i] = buf[nocc_b:,:nocc_a,nval_a_s]
                eris.CEee[i] = lib.pack_tril(buf[nocc_b:,nocc_a:,nocc_a:])
            if i >= ncvs:
                i -= ncvs
                eris.VEcc[i] = buf[nocc_b:,:ncvs,:ncvs]
                eris.VEec[i] = buf[nocc_b:,nocc_a:,:ncvs]  
                eris.VEev[i] = buf[nocc_b:,nocc_a:,nval_a_s]
                eris.VCee[i] = buf[:ncvs,nocc_a:,nocc_a:]
                eris.VVee[i] = buf[nval_b_s,nocc_a:,nocc_a:]
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

    #for eri1 in myadc._scf.with_df.loop():
    for eri1 in myadc.with_df.loop():
        Lpq = ao2mo._ao2mo.nr_e2(eri1, mo_coeff_a, ijslice, aosym='s2', out=Lpq).reshape(-1,nmo_a,nmo_a)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Loo[p0:p1] = Lpq[:,:nocc_a,:nocc_a]
        eris.Lov[p0:p1] = Lpq[:,:nocc_a,nocc_a:]
        Lvo[p0:p1] = Lpq[:,nocc_a:,:nocc_a]
        eris.Lvv[p0:p1] = Lpq[:,nocc_a:,nocc_a:]


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

    Loo = Loo.reshape(naux,nocc_a*nocc_a)
    eris.Lov = eris.Lov.reshape(naux,nocc_a*nvir_a)
    Lvo = Lvo.reshape(naux,nocc_a*nvir_a)
    LOO = LOO.reshape(naux,nocc_b*nocc_b)
    eris.LOV = eris.LOV.reshape(naux,nocc_b*nvir_b)
    LVO = LVO.reshape(naux,nocc_b*nvir_b)

    Lvv_p = lib.pack_tril(eris.Lvv)
    LVV_p = lib.pack_tril(eris.LVV)

    eris.vvvv_p = None
    eris.VVVV_p = None
    eris.vVvV_p = None
    eris.VvVv_p = None

    eris.ovvv = None
    eris.OVVV = None
    eris.OVVV = None
    eris.ovVV = None
    eris.OVvv = None

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

    log.timer('DF-ADC integral transformation', *cput0)

    return eris

def calculate_chunk_size(myadc):

    avail_mem = (myadc.max_memory - lib.current_memory()[0]) * 0.25 
    vvv_mem = (myadc._nvir[0]**3) * 8/1e6

    chnk_size =  int(avail_mem/vvv_mem)

    if chnk_size <= 0 :
        chnk_size = 1

    return chnk_size
