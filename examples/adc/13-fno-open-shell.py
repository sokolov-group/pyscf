#!/usr/bin/env python

"FNO approximation based on ADC/MP (open-shell references)"

from pyscf import gto, scf, adc
from pyscf.adc.uadc_ee import get_spin_square as uadc_ee_get_spin_square

mol = gto.M(atom='H 0 0 0; O 0 0 0.8', basis='ccpvtz',spin=1)
mol.verbose=5
mf = scf.UHF(mol).set(verbose=1).run()

#1. UHF reference

# ADC2FNO can also be used for open-shell system, which may result in different frozen orbitals for alpha/beta spin.
# The settings for open-shell FNO calculation is the same as close-shell case
ADCFG = adc.ADC2FNO(mf).set(ref_state=1,if_naf=True).density_fit('ccpvdz-ri')
# Besides ADC(2), ADC(2)-X can also be used as the method for generating FNO space and the correction
ADCFG.method = "adc(2)-X"
ADCFG.kernel(nroots=4,pct_occ=0.95)

myadc = adc.UADC(mf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ,ADCFG.mo_energy).density_fit('ccpvdz-ri')
myadc.method = "adc(3)"
myadc.if_naf = True
e,v,p,x=myadc.kernel(nroots=4)
print("SS-FNO-IP-UADC excitation energies (eV) are")
print(ADCFG.correct(e)*27.2114)
print("SS-FNO-UMP3 correlation energy correction is")
print(ADCFG.correct_corr(myadc.e_corr))

#2. ROHF reference

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
print(ADCFG.correct(e)*27.2114)

#3. OSFNO: open-shell FNO for UHF references

# For open-shell references the plain FNO scheme truncates the alpha and beta
# virtual spaces independently, which unbalances the two spin spaces and may
# contaminate the spin of the target states. The OSFNO scheme
# (J. Chem. Phys. 152, 034105 (2020)) identifies, via SVD of the overlap
# between majority-spin occupied and minority-spin virtual orbitals, the
# virtual partners of the singly occupied orbitals, which are always kept
# active, and truncates the remaining virtuals as alpha-beta natural-orbital
# pairs obtained from the SVD of the singlet part of the state density.
# It is enabled by setting if_osfno = True (UADC2FNO only).
mol = gto.M(atom='H 0 0 0; O 0 0 0.8', basis='ccpvtz',spin=1)
mol.verbose=5
mf = scf.UHF(mol).set(verbose=1).run()

ADCFG = adc.ADC2FNO(mf, frozen=[0,0]).set(verbose=5, method_type='ee')
ADCFG.if_osfno = True
# canonical orbitals frozen in advance are combined with the OSFNO truncation
ADCFG.kernel(nroots=4, pct_occ=0.90)

myadc = adc.UADC(mf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ,ADCFG.mo_energy)
myadc.method = "adc(3)"
myadc.method_type = "ee"
e,v,p,x=myadc.kernel(nroots=4)
print("OSFNO-UADC excitation energies (eV) are")
print(ADCFG.correct(e)*27.2114)

#4. OSFNO with the ROHF reference: spin purity of the truncated states

mf = scf.ROHF(mol).set(verbose=1).run()
ADCFG = adc.ADC2FNO(mf).set(verbose=5, method_type='ee')
ADCFG.if_osfno = True
ADCFG.kernel(nroots=4, pct_occ=0.90)

# f_ov must be passed when the ROHF reference is used with explicit orbitals
myadc = adc.UADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy,f_ov=ADCFG.f_ov)
myadc.method = "adc(2)-x"
myadc.method_type = 'ee'
e,v,p,x=myadc.kernel(nroots=4)
spin = uadc_ee_get_spin_square(myadc._adc_es)[0]
print("OSFNO-EE-UADC excitation energies (eV) are")
print(ADCFG.correct(e)*27.2114)
print("OSFNO-EE-UADC <S^2> values are")
print(spin)

#5. SS-FNO-EE with character-based root following (open shell)

mf = scf.UHF(mol).set(verbose=1).run()
ADCFG = adc.ADC2FNO(mf).set(verbose=5, method_type='ee', ref_state=1)
ADCFG.trans_guess = True
ADCFG.pick = True
ADCFG.kernel(nroots=4, thresh=1e-4)
print("particle weight lost to frozen virtuals per root:")
print(ADCFG.w_guess_lost)
print("root x guess overlap matrix of the truncated ADC(2) run:")
print(ADCFG.ovl_guess)
print("root-wise truncation correction (same physical states on both sides):")
print(ADCFG.delta_e)

myadc = adc.UADC(mf,ADCFG.frozen,ADCFG.mo_coeff,ADCFG.mo_occ,ADCFG.mo_energy)
myadc.method = "adc(3)"
myadc.method_type = "ee"
myadc.pick = True
e,v,p,x=myadc.kernel(nroots=4,guess=ADCFG.v_ssfno)
print("root x guess overlap matrix of the truncated ADC(3) run:")
print(myadc.ovl_guess)
spin = uadc_ee_get_spin_square(myadc._adc_es)[0]
print("SS-FNO-EE-UADC(3) excitation energies with root following (eV) are")
print(ADCFG.correct(e)*27.2114)
print("with <S^2> values")
print(spin)
