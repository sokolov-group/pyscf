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

r"""Public constructors for restricted periodic ADC and FNO-ADC drivers.

KRADC wraps the reference driver used for charged and neutral excitation
calculations. KRADC2FNO selects the frozen-natural-orbital workflow.
Both factories accept k-point restricted SCF references or references
convertible to them. Worked examples include their imports and construct
a small two-k-point helium reference from scratch.
"""
from pyscf.pbc import scf
from pyscf.pbc.adc import kadc_fno, kadc_rhf, kadc_rhf_ea, kadc_rhf_ip


def KRADC(mf, frozen=None, mo_coeff=None, mo_occ=None, mo_energy=None):
    r"""Create a restricted periodic ADC driver from a k-point SCF reference.

    Args:
        mf (KRHF): SCF reference with orbitals, cell and k points. A reference
            of another compatible SCF type is converted with to_rhf().
        frozen (int or sequence, optional): KMP2 frozen-space specification:
            an occupied count, shared MO index list or one list per k point.
        mo_coeff (sequence[ndarray], optional): Override SCF coefficients,
            one (nao, nmo_k) matrix per k point. None uses mf.mo_coeff.
        mo_occ (sequence[ndarray], optional): Corresponding MO occupations;
            None uses mf.mo_occ.
        mo_energy (sequence[ndarray], optional): Corresponding energies in
            Hartree; None uses mf.mo_energy.

    Returns:
        kadc_rhf.KRADC: Unrun ADC(2)/IP driver. ground_state_energy() returns
        the correlation contribution per cell and stores amplitudes in t1/t2.
        Set type_='ea' or 'ee' to change excitation sector, and method to
        select ADC(2), ADC(2)-x or ADC(3).

    Examples:
        Create the two-k-point helium reference below; lattice vectors
        are in Angstrom and energies are in Hartree:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import KRADC
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
        >>> _ = kmf.kernel()
        >>> mykradc = KRADC(kmf)
        >>> round(mykradc.ground_state_energy(), 8)
        -0.00416766
        >>> mykradc.type_ = 'ee'
        >>> _ = mykradc.kernel(2, kptlist=[0])
        >>> mykradc.E.round(6).tolist()
        [[1.756312, 2.102047]]
    """
    if not isinstance(mf, scf.khf.KRHF):
        mf = mf.to_rhf()
    return kadc_rhf.KRADC(mf, frozen, mo_coeff, mo_occ, mo_energy)


def KRADC2FNO(mf, frozen=None, mo_coeff=None, mo_occ=None):
    r"""Create a periodic ADC driver with frozen-natural-orbital truncation.

    Args:
        mf (KRHF): Restricted k-point SCF reference, or a compatible reference
            converted to restricted form by to_rhf().
        frozen (int or sequence, optional): Initial KMP2 frozen-space setting.
        mo_coeff (sequence[ndarray], optional): Coefficient override matching
            the reference MO ordering and occupations.
        mo_occ (sequence[ndarray], optional): Occupation override matching
            coefficient columns. The initial energies come from mf.mo_energy.

    Returns:
        kadc_fno.RADC2FNO: Unrun FNO driver. See its class documentation for
        occupation cutoffs, cumulative fractions, explicit virtual counts,
        multiple thresholds and state-specific densities. Energies and additive
        corrections are separate result attributes.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import KRADC2FNO
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
        >>> _ = kmf.kernel()
        >>> mykradc_fno = KRADC2FNO(kmf)
        >>> energy = mykradc_fno.ground_state_energy(nvir_act=1)
        >>> round(energy, 8), int(mykradc_fno.nvir)
        (-0.00416766, 1)
    """
    if not isinstance(mf, scf.khf.KRHF):
        mf = mf.to_rhf()
    return kadc_fno.RADC2FNO(mf, frozen, mo_coeff, mo_occ)
