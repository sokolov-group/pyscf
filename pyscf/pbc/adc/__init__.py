#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
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

from pyscf.pbc import scf
from pyscf.pbc.adc import kadc_fno, kadc_rhf, kadc_rhf_ea, kadc_rhf_ip


def KRADC(mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
    """Create a restricted periodic ADC driver from a k-point SCF reference.

    See :class:`kadc_rhf.KRADC` for orbital overrides and frozen spaces.
    ``ground_state_energy()`` returns e_corr; amplitudes stay on the driver.
    """
    if not isinstance(mf, scf.khf.KRHF):
        mf = mf.to_rhf()
    return kadc_rhf.KRADC(mf, frozen, mo_coeff, mo_occ, mo_energy)


def KRADC2FNO(mf, frozen=None, mo_coeff=None, mo_occ=None):
    if not isinstance(mf, scf.khf.KRHF):
        mf = mf.to_rhf()
    return kadc_fno.RADC2FNO(mf, frozen, mo_coeff, mo_occ)
