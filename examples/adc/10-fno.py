#!/usr/bin/env python

"FNO approximation based on ADC/MP"

from pyscf import gto, scf, adc, cc

#1. close shell

mol = gto.M(atom='C 0 0 0; O 0 0 1.283', basis='augccpvtz')
mol.verbose=5
mf = scf.RHF(mol).set(verbose=1).run()

#1.1 SS-FNO-IP-ADC(3) calculation

# Instantiate the FNO object for IP-ADC(2) calculation, which will be used to generate the FNO space
# and the correction for the IP-ADC(3) calculation
# The specific state for the SS-FNO-ADC calculation can be set by ref_state,
# which should be an int type and in [0,nroots]
# eg. ref_state = 1 means the first excited state
ADCFG = adc.ADC2FNO(mf).set(verbose=5,method_type="ip",ref_state=1)
# when trans_guess is True SS-FNO-IP-ADC would use the Can-ADC(2) eigenvector as the guess
# only available for IP
ADCFG.trans_guess=True
# There are three kind of threshold which can be used to divide the natural orbitals, thresh, pct_occ and vir_act.
# The default one is thresh=1e-4, user can change the threshold by passing the parameters in kernel function.
ADCFG.kernel(nroots=3,pct_occ=0.99)

# Perform the IP-ADC(3) calculation in the generated FNO space
myadc = adc.RADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy)
myadc.verbose = 5
myadc.method_type = "ip"
myadc.method = "adc(3)"

e,v,p,x=myadc.kernel(nroots=3)

# Correct the SS-FNO-ADC(3) excitation energies and MP3 correlation energy
# with the correction obtained from the SS-FNO-ADC(2) and Can-ADC(2) calculation
print("SS-FNO-IP-ADC(3) excitation energies (eV) are")
print((e+ADCFG.delta_e)*27.2114)
print("SS-FNO-MP3 correlation energy correction is")
print(myadc.e_corr+ADCFG.delta_e_corr)

#1.2 FNO-EA-ADC(3) calculation

# FNO calculation would be performed when ref_state is not set or ref_state is set to 0
# For most cases FNO would result in larger error than SS-FNO
ADCFG = adc.ADC2FNO(mf).set(verbose=5,method_type="ea",approx_trans_moments=True).density_fit('augccpvtz-ri')
ADCFG.kernel(nroots=2,thresh=10**(-4.5))

myadc = adc.RADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy).density_fit('augccpvtz-ri')
myadc.approx_trans_moments = True
myadc.verbose = 5
myadc.method_type = "ea"
myadc.method = "adc(3)"

# SS-FNO-ADC(2) eigenvector can be used as the guess for SS-FNO-ADC(3) calculation
e,v,p,x=myadc.kernel(nroots=2,guess=ADCFG.v_ssfno)

print("FNO-EA-ADC(3) excitation energies (eV) are")
print((e+ADCFG.delta_e)*27.2114)

#1.3 SS-FNO-EE-ADC(3) calculation with NAF

# When density fitting is enabled, FNO calculation can be accelerated by using the NAF approximation
ADCFG = adc.ADC2FNO(mf).set(verbose=5,method_type="ee",ref_state=2,if_naf=True).density_fit('augccpvtz-ri')
ADCFG.kernel(nroots=2,nvir_act=56,guess="cis")

myadc = adc.RADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy).density_fit('augccpvtz-ri')
myadc.verbose = 5
myadc.method_type = "ee"
myadc.method = "adc(3)"
myadc.if_naf = True

e,v,p,x=myadc.kernel(nroots=2)

print("SS-FNO-EE-ADC(3) excitation energies (eV) are")
print((e+ADCFG.delta_e)*27.2114)

#1.4 FNO-MP3 calculation

# eris used in FNO object can also pass to following calculation by setting if_heri_eris to True
ADCFG = adc.ADC2FNO(mf).set(verbose=5,if_heri_eris=True,if_naf=True).density_fit('augccpvdz-ri')
ADCFG.kernel_gs()

myadc = adc.RADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy).density_fit('augccpvdz-ri')
myadc.verbose = 5
# when NAF is enabled and eris is passed, naux in ADC should be set to the naux in FNO object
myadc.naux=ADCFG.naux
myadc.method_type = "ee"
myadc.method = "adc(3)"
myadc.if_naf = True
_,_,_=myadc.kernel_gs(eris=ADCFG.eris)
print(myadc.e_corr+ADCFG.delta_e_corr)

#1.5 SS-FNO-IP-EOM-CCSD

# SS-FNO can also be used for EOM-CCSD calculation,
# which can be implemented by passing the frozen list, mo_coeff from FNO object to CCSD object.
mol = gto.M(atom='C 0 0 0; O 0 0 1.283', basis='augccpvtz')
ADCFG = adc.ADC2FNO(mf).set(verbose=5,if_naf=True,ref_state=1,approx_trans_moments=True).density_fit('augccpvdz-ri')
ADCFG.kernel(nroots=4)

mycc = cc.RCCSD(mf,ADCFG.frozen,ADCFG.mo_coeff).density_fit('augccpvdz-ri')
mycc.ccsd()
eip,cip = mycc.ipccsd(nroots=4)
print("SS-FNO-IP-EOM-CCSD excitation energies (eV) are")
print((eip+ADCFG.delta_e)*27.2114)
print("SS-FNO-CCSD correlation energy correction is")
print(mycc.e_corr+ADCFG.delta_e_corr)

#2. open shell

#2.1 UHF reference

# ADC2FNO can also be used for open-shell system, which may result in different frozen orbitals for alpha/beta spin.
# The settings for open-shell FNO calculation is the same as close-shell case
mol = gto.M(atom='H 0 0 0; O 0 0 0.8', basis='ccpvtz',spin=1)
mol.verbose=5
mf = scf.UHF(mol).set(verbose=1).run()
ADCFG = adc.ADC2FNO(mf).set(ref_state=1,if_naf=True).density_fit('ccpvdz-ri')
# Besides ADC(2), ADC(2)-X can also be used as the method for generating FNO space and the correction
ADCFG.method = "adc(2)-X"
ADCFG.kernel(nroots=4,pct_occ=0.95)

myadc = adc.UADC(mf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ,ADCFG.mo_energy).density_fit('ccpvdz-ri')
myadc.method = "adc(3)"
myadc.if_naf = True
e,v,p,x=myadc.kernel(nroots=4)
print("SS-FNO-IP-UADC excitation energies (eV) are")
print((e+ADCFG.delta_e)*27.2114)
print("SS-FNO-UMP3 correlation energy correction is")
print(myadc.e_corr+ADCFG.delta_e_corr)

#2.2 ROHF reference

# when ROHF reference is used, user should pass the f_ov matrix from FNO object to ADC object
mf = scf.ROHF(mol).set(verbose=1).run()
ADCFG = adc.ADC2FNO(mf).set(ncvs=1,ref_state=1,if_naf=True,approx_trans_moments=True).density_fit('ccpvdz-ri')
ADCFG.kernel(nroots=4)

myadc = adc.UADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy,f_ov=ADCFG.f_ov).density_fit('ccpvdz-ri')
myadc.method = "adc(3)"
myadc.ncvs = 1
myadc.approx_trans_moments = True
myadc.if_naf = True
myadc.conv_tol = 1e-8
myadc.tol_residual = 1e-6
e,v,p,x=myadc.kernel(nroots=4,guess=ADCFG.v_ssfno)
print("SS-FNO-IP-CVS-UADC excitation energies (eV) are")
print((e+ADCFG.delta_e)*27.2114)