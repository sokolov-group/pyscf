#!/usr/bin/env python3
#
# Authors: 
#       Ziqiu Wang <sgwzq0810@gmail.com>
#       Samragni Banerjee <samragnibanerjee4@gmail.com>
#

# --- Options ---
sector = 'ee'
method = 'adc(2)'
# --- End of Options ---

'''
This example shows the use of periodic EA/IP-ADC with k-point sampling.
Also shown are molecular EA/IP-ADC Gamma-point calculations using a supercell.
'''

import numpy as np
from pyscf.pbc import gto, scf, adc
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc.adc.lib import log_table
from pyscf.lib import logger


cell = gto.Cell()
cell.verbose = 3
cell.unit = 'B'

#
# Helium crystal
#
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000
'''
cell.build()

ncopies = [1, 1, 2]
kptlist = np.arange(np.prod(ncopies)) # [0, 1, 2, 3] if ncopies = [1, 2, 2]
nroots_test = 4

# KRHF
kpoints = cell.make_kpts(ncopies)
kmf = scf.KRHF(cell, kpts=kpoints, exxdiv=None).density_fit().run()

# KRADC
solver_type = {'ip': adc.KRADCIP, 'ea': adc.KRADCEA, 'ee': adc.KRADCEE}[sector]
mykradc = solver_type(kmf)

# EE-KRADC
mykradc.method = method
mykradc.verbose = 4
mykradc.approx_trans_moments = False
mykradc.kernel(nroots_test, kptlist=kptlist)
k_e_ee, v_e_ee = mykradc.E, mykradc.U
k_p_ee, k_x_ee = mykradc.P, mykradc.X

# Supercell
scell = super_cell(cell, ncopies)

# PBC Gamma-point RHF based on supercell
mf = scf.RHF(scell, exxdiv=None).density_fit().run()

# Molecular ADC
myadc = mol_adc.RADC(mf)

# Molecular EE-ADC
myadc.method_type = sector
myadc.method = method
myadc.verbose = 4
mol_e_ee,mol_v_ee,mol_p_ee,mol_x_ee = myadc.kernel(
    nroots_test * np.prod(ncopies))

