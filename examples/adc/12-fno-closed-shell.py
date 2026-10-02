#!/usr/bin/env python

"FNO approximation based on ADC/MP (closed-shell references)"

from pyscf import gto, scf, adc, cc

mol = gto.M(atom='C 0 0 0; O 0 0 1.283', basis='augccpvtz')
mol.verbose=5
mf = scf.RHF(mol).set(verbose=1).run()

#1. SS-FNO-IP-ADC(3) calculation

# Instantiate the FNO object for IP-ADC(2) calculation, which will be used to generate the FNO space
# and the correction for the IP-ADC(3) calculation
# The specific state for the SS-FNO-ADC calculation can be set by ref_state,
# which should be an int type and in [0,nroots]
# eg. ref_state = 1 means the first excited state
ADCFG = adc.ADC2FNO(mf).set(verbose=5,method_type="ip",ref_state=1)
# when trans_guess is True SS-FNO-ADC would use the Can-ADC(2) eigenvectors,
# projected onto the truncated FNO basis, as the guesses of the truncated
# ADC(2) calculation (available for IP, EA and EE)
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
print(ADCFG.correct(e)*27.2114)
print("SS-FNO-MP3 correlation energy correction is")
print(ADCFG.correct_corr(myadc.e_corr))

#2. FNO-EA-ADC(3) calculation

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
print(ADCFG.correct(e)*27.2114)

#3. SS-FNO-EE-ADC(3) calculation with NAF

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
print(ADCFG.correct(e)*27.2114)

#4. FNO-MP3 calculation

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
print(ADCFG.correct_corr(myadc.e_corr))

#5. SS-FNO-IP-EOM-CCSD

# SS-FNO can also be used for EOM-CCSD calculation,
# which can be implemented by passing the frozen list, mo_coeff from FNO object to CCSD object.
mol = gto.M(atom='C 0 0 0; O 0 0 1.283', basis='augccpvtz')
ADCFG = adc.ADC2FNO(mf).set(verbose=5,if_naf=True,ref_state=1,approx_trans_moments=True).density_fit('augccpvdz-ri')
ADCFG.kernel(nroots=4)

mycc = cc.RCCSD(mf,ADCFG.frozen,ADCFG.mo_coeff).density_fit('augccpvdz-ri')
mycc.ccsd()
eip,cip = mycc.ipccsd(nroots=4)
print("SS-FNO-IP-EOM-CCSD excitation energies (eV) are")
print(ADCFG.correct(eip)*27.2114)
print("SS-FNO-CCSD correlation energy correction is")
print(ADCFG.correct_corr(mycc.e_corr))

#6. SS-FNO-EE-ADC(3) with character-based root following

# In dense or heavily truncated spectra the energy-ordered roots of the truncated
# calculation may not correspond root-by-root to the canonical ones. Setting
# trans_guess = True seeds the truncated Davidson with the canonical ADC(2)
# eigenvectors projected onto the FNO basis (project_guess), and pick = True
# enables the overlap-ranked root selection inside the solver, so that root k
# of the truncated calculation carries the character of canonical root k.
# The weight of the projected guess lost to the frozen virtuals (w_guess_lost)
# and the converged root x guess overlap matrix (ovl_guess) are provided as
# diagnostics of the state following.
ADCFG = adc.ADC2FNO(mf).set(verbose=5,method_type="ee",ref_state=2).density_fit('augccpvtz-ri')
ADCFG.trans_guess=True
ADCFG.pick=True
ADCFG.kernel(nroots=3,thresh=10**(-4.5))
print("particle weight lost to frozen virtuals per root:")
print(ADCFG.w_guess_lost)
print("root x guess overlap matrix of the truncated ADC(2) run:")
print(ADCFG.ovl_guess)

# Seed the truncated ADC(3) with the homed truncated ADC(2) eigenvectors
# (same truncated basis, no further transformation needed) and keep the
# overlap-ranked selection enabled, so the correction delta_e pairs the
# same physical states on both sides.
myadc = adc.RADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy).density_fit('augccpvtz-ri')
myadc.verbose = 5
myadc.method_type = "ee"
myadc.method = "adc(3)"
myadc.pick = True
e,v,p,x=myadc.kernel(nroots=3,guess=ADCFG.v_ssfno)
print("root x guess overlap matrix of the truncated ADC(3) run:")
print(myadc.ovl_guess)

print("SS-FNO-EE-ADC(3) excitation energies with root following (eV) are")
print(ADCFG.correct(e)*27.2114)

#1.7 SA-FNO-EE-ADC(3): state-averaged FNO over a list of states

# Giving ref_state as a LIST of roots switches to the state-averaged scheme:
# the excited-state 1-RDMs of the listed roots are averaged and added once
# to the ground-state density, generating a single FNO space that serves
# all selected states (Dutta et al.: SA-FNO reaches SS-FNO accuracy while
# avoiding per-state integral transformations). Averaging a degenerate
# pair (here the first two excited states of CO) is also the component-
# symmetric choice for degenerate targets.
ADCFG = adc.ADC2FNO(mf).set(verbose=5,method_type="ee",ref_state=[1,2]).density_fit('augccpvtz-ri')
ADCFG.trans_guess=True
ADCFG.pick=True
ADCFG.kernel(nroots=3,thresh=10**(-4.5))
print("SA-FNO root x guess overlap matrix of the truncated ADC(2) run:")
print(ADCFG.ovl_guess)

myadc = adc.RADC(mf,ADCFG.frozen,ADCFG.mo_coeff,mo_energy=ADCFG.mo_energy).density_fit('augccpvtz-ri')
myadc.verbose = 5
myadc.method_type = "ee"
myadc.method = "adc(3)"
myadc.pick = True
e,v,p,x=myadc.kernel(nroots=3,guess=ADCFG.v_ssfno)
print("SA-FNO-EE-ADC(3) excitation energies (eV) are")
print(ADCFG.correct(e)*27.2114)
