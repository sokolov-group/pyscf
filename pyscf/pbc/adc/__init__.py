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

r"""Restricted periodic ADC reference, sector solvers and FNO constructors.

Use KRADCIP, KRADCEA or KRADCEE to select the excitation sector. All accept a
KRHF reference, a periodic Cell (Gamma by default), or another periodic ADC
object. KRADCBase computes reference amplitudes and correlation energies.
"""
from pyscf.pbc.adc.kadc_rhf import KRADCBase, KRADCIP, KRADCEA, KRADCEE
from pyscf.pbc.adc.kadc_fno import KRADCIPFNO, KRADCEAFNO, KRADCEEFNO


def KRADC2FNO(mf, frozen=None, mo_coeff=None, mo_occ=None,
              mo_energy=None, kpts=None):
    r"""Create an FNO solver with the sector of a supplied periodic ADC object.

    Args:
        mf (KRADCBase, KRHF or Cell): Reference. ADC sector objects select
            their corresponding FNO class; other inputs default to KRADCIPFNO.
        frozen (int or sequence, optional): KMP2 frozen-space override.
        mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
            orbital overrides. None inherits the reference data.
        kpts (array_like, optional): Cartesian inverse-Bohr mesh for Cell
            inputs, shape (nkpts, 3). None selects Gamma.

    Returns:
        KRADCIPFNO, KRADCEAFNO or KRADCEEFNO: Unrun FNO solver. Conversion
        copies common ADC settings while clearing roots and FNO-specific
        results. Truncation controls are documented by the solver's
        ground_state_energy and kernel methods. Concrete FNO classes can also
        be called directly.

    Raises:
        TypeError: The input does not provide a supported periodic reference.
        ValueError: kpts was supplied with an SCF or ADC object.

    Examples:
        Build and run a small two-k helium reference, then preserve the EA
        sector when enabling FNO truncation. Lattice vectors are in Angstrom:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import KRADCEA, KRADC2FNO, KRADCEAFNO
        >>> cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]], verbose=0)
        >>> kmf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcea_fno = KRADC2FNO(KRADCEA(kmf))
        >>> type(mykradcea_fno) is KRADCEAFNO
        True
        >>> energy = mykradcea_fno.ground_state_energy(nvir_act=1)
        >>> round(energy, 8)
        -0.00416766
    """
    if isinstance(mf, KRADCEE):
        solver_type = KRADCEEFNO
    elif isinstance(mf, KRADCEA):
        solver_type = KRADCEAFNO
    else:
        solver_type = KRADCIPFNO
    return solver_type(mf, frozen, mo_coeff, mo_occ, mo_energy, kpts)
