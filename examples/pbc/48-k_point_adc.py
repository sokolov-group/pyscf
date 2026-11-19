#!/usr/bin/env python
#
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#

'''
This example shows the use of periodic EA/IP-ADC with k-point sampling.
Also shown are molecular EA/IP-ADC Gamma-point calculations using a supercell.
'''

import numpy as np
from pyscf.pbc import gto, scf
from pyscf.pbc.adc import KRADCEA, KRADCIP, KRADCEE
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell


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

nmp = [1,2,2]
nroots_test = 2
adc_method = 'adc(2)' # Options: 'adc(2)', 'adc(2)-x', 'adc(3)'
verbose = 4

# KRHF
kpts = cell.make_kpts(nmp)
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit().run()


# IP-KRADC
myipadc = KRADCIP(kmf)
myipadc.method = adc_method
myipadc.nroots = nroots_test
myipadc.verbose = verbose
myipadc.kernel()
# To get properties, we can try using these codes below:
# e_ipadc = myipadc.E
# t_ipadc = myipadc.T
# p_ipadc = myipadc.P
# x_ipadc = myipadc.X


# EA-KRADC
myeaadc = KRADCEA(myipadc)
myeaadc.kernel()

# EE-KRADC
myeeadc = KRADCEE(myipadc)
# Include the higher-order amplitudes in the EE transition moments on both
# sides of the comparison. Periodic ADC normally approximates these.
myeeadc.approx_trans_moments = False
myeeadc.kernel()

# To compare, we can run a supercell ADC calculation.
# ncopies = np.prod(nmp)
# supcell = super_cell(cell, nmp)
# supmf = scf.RHF(supcell, exxdiv=None).density_fit().run()
# mysupadc = mol_adc.RADC(supmf)
# mysupadc.method, mysupadc.method_type = (adc_method, 'ee')
# mysupadc.verbose = verbose
# mysupadc.approx_trans_moments = myeeadc.approx_trans_moments
# sup_e, _, sup_f, _ = mysupadc.kernel(nroots=nroots_test * ncopies)

# A supercell root can mix degenerate states from different k sectors.
# Compare strengths summed over energy-matched groups with equal root counts.
# print('Energy (Eh)   Roots    k-point sum    supercell sum')
# remaining = np.ones(sup_e.size, dtype=bool)
# for root in np.argsort(sup_e):
    # if remaining[root]:
        # energy = sup_e[root]
        # group = remaining & (abs(sup_e - energy) < 1e-6)
        # remaining[group] = False
        # matched = abs(myeeadc.E - energy) < 1e-6
        # if matched.sum() == group.sum():
            # print(f'{energy:11.7f} {group.sum():7d} '
                  # f'{myeeadc.P[matched].sum():14.8f} '
                  # f'{sup_f[group].sum():16.8f}')
