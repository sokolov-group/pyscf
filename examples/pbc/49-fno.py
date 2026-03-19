#!/usr/bin/env python
#
# Author: Ning-Yuan Chen <cny003@outlook.com>
#

'''
This example file shows how to use MPn/ADC based FNO/SS-FNO framework to do k-point post-HF calculations.
The first case uses ADC2 to generate SS-FNOs and then do SS-FNO-ADC(3) calculation.
The second case uses these SS-FNOs to do FNO-EOM-CCSD calculation.
The third case uses MP2 to generate FNOs and then do FNO-ADC(3) calculation.
'''

# case1 SS-FNO kADC3
import numpy as np
from pyscf.pbc import gto, scf, adc
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell
from scipy.linalg import eigh
import time

def qp_correct(FG, E, P, kptlist):
    if E.shape!=FG.delta_e.shape:
        raise ValueError("The shape of uncorrected E and delta_e should be the same")
    print("start to correct the quasiparticle energy")

    E_p_corrected = []
    mask_fno = FG.p2_ssfno>0.5
    mask = P>0.5

    if kptlist is None:
        kptlist = range(FG.nkpts)

    msg = ("\n*************************************************************"
        "\n            FNO quasiparticle energy summary"
        "\n*************************************************************")
    print(msg)

    for k, kshift in enumerate(kptlist):
        delta_e_k = FG.delta_e[k,mask_fno[k]]
        E_p_k = E[k,mask[k]]
        E_p_corrected_k = E_p_k + delta_e_k[:E_p_k.shape[0]]
        sort_indices = np.argsort(E_p_corrected_k)
        E_p_corrected.append(E_p_corrected_k[sort_indices])
        for n in range(E_p_corrected[k].shape[0]):
            print_string = ('%s-FNO k-point %d | qp %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  |  %s-FNO Spec factors = %10.8f  |  target-FNO Spec factors = %10.8f' %
                            (FG.method, kshift, n, E_p_corrected[k][n], E_p_corrected[k][n]*27.2114, FG.method, FG.p2_ssfno[k,mask_fno[k]][sort_indices[n]], P[k,mask[k]][sort_indices[n]]))
            print(print_string)

    return E_p_corrected

cell = gto.Cell()
cell.verbose = 0
cell.unit = 'B'

#
# Helium crystal
#
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.build()

nmp = [2,2,1]
nroots_test = 5

# KRHF
kpts = cell.make_kpts(nmp)
kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None).density_fit()
ekrhf = kmf.kernel()

# FNOGenerator
ADCFG  = adc.KRADC2FNO(kmf)
ADCFG.method_type = 'ea'
ADCFG.approx_trans_moments = True
ADCFG.verbose = 5
ADCFG.ref_state = [0,0]
ADCFG.kernel(nroots_test,pct_occ=0.90,kptlist=[0])

# kadc3
kadc = adc.KRADC(kmf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ,ADCFG.mo_energy)
kadc.method_type = "ea"
kadc.approx_trans_moments = True
kadc.verbose = 5
kadc.method="adc(3)"
k_e_ea, k_v_ea, k_p_ea, k_x_ea = kadc.kernel(nroots_test,guess=ADCFG.v2_ssfno,kptlist=[0])

e_qp=qp_correct(ADCFG,k_e_ea,k_p_ea,kptlist=[0])

# case2 SS-FNO kEOM-CCSD
from pyscf.pbc import cc
from pyscf.pbc.cc.eom_kccsd_rhf import EOMEA

# KRCCSD
mycc = cc.KRCCSD(kmf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ)
mycc.verbose = 5
ekrcc, t1, t2 = mycc.kernel()
# K-EA-EOM-CCSD
myeom = EOMEA(mycc)
eomcc_e,eomcc_v = myeom.kernel(nroots_test,kptlist=[0])

eomcc_e_corrected = eomcc_e+ADCFG.delta_e

# case3 FNO-MP3
MPFG = adc.KRADC2FNO(kmf)
MPFG.approx_trans_moments = True
MPFG.verbose = 5
MPFG.kernel_gs(pct_occ=0.1)

kadc_gs  = adc.KRADC(kmf,MPFG.frozen,MPFG.mo_coeff,MPFG.mo_occ,MPFG.mo_energy)
kadc_gs.approx_trans_moments = True
kadc_gs.verbose = 5
kadc_gs.method = "adc(3)"
e_corr,t1,t2 = kadc_gs.kernel_gs()

e_corr_correct = e_corr+MPFG.delta_e_corr