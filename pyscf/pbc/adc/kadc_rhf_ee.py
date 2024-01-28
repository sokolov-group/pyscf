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
#
# Author: Samragni Banerjee <samragnibanerjee4@gmail.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#
import time
import numpy as np
import pyscf.ao2mo as ao2mo
import pyscf.adc
import pyscf.adc.radc
from pyscf.adc import radc_ao2mo
import itertools

from itertools import product
from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import df
from pyscf.pbc import mp
from pyscf.lib import logger
from pyscf.pbc.adc import kadc_rhf
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc import dfadc
from pyscf import __config__
from pyscf.pbc.mp.kmp2 import (get_nocc, get_nmo, padding_k_idx,_padding_k_idx,
                               padded_mo_coeff, get_frozen_mask, _add_padding)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa

from pyscf.pbc import tools
import h5py
import tempfile


def vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc

    n_singles = nocc*nvir
    n_doubles = nkpts * nkpts * nocc * nocc * nvir * nvir
    size = n_singles + n_doubles

    return size


def get_imds(adc, eris=None):

    cput0 = (time.process_time(), time.time())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    ncore = adc.nocc
    nextern = adc.nmo - adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_core = [mo_energy[k][:ncore] for k in range(nkpts)]
    e_extern = [mo_energy[k][ncore:] for k in range(nkpts)]

    e_core = np.array(e_core)
    e_extern = np.array(e_extern)

    idn_vir = np.identity(nextern)
    if eris is None:
        eris = adc.transform_integrals()

    v_ccee = eris.oovv
    v_cece = eris.ovvo
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    # Zeroth-order terms

    M_ = np.empty((nkpts,ncore,nextern,ncore,nextern),dtype=mo_coeff.dtype)

    einsum = lib.einsum
    einsum_type = True

    for ka in range(nkpts):
        kb = ka
        M_[ka] = einsum('A,AD,IL->IDLA', e_extern[ka], np.identity(nextern), np.identity(ncore), optimize = einsum_type)
        M_[ka] -= einsum('L,AD,IL->IDLA', e_core[ka], np.identity(nextern), np.identity(ncore), optimize = einsum_type)
        #print(np.linalg.norm(M_))
        #exit()
        for kl in range(nkpts):
            for km in range(nkpts):
                t1_ccee = adc.t2[0][km,kl,ka]

                M_[ka] -= einsum('ILAD->IDLA', v_ccee[kl,kb,km], optimize = einsum_type).copy()
                M_[ka] += einsum('LADI->IDLA', v_ceec[kl,kb,km], optimize = einsum_type).copy()
                M_[ka] += einsum('LADI->IDLA', v_ceec[kl,kb,km], optimize = einsum_type).copy()

                v_cece = v_ceec[kl,kb,km].copy()

                M_[ka] += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('iIDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('iLAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('A,LiAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('A,iLAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('A,iLAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('D,LiAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('D,iLAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('I,LiAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('I,iLAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('L,LiAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('L,iLAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('L,iLAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('a,LiAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('a,iLAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('a,iLAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('i,IiDa,LiAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,IiDa,iLAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('i,LiAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,LiAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,iIDa,LiAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,iIDa,iLAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,iLAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,iLAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 2 * einsum('AD,Iiab,Labi->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('AD,Iiab,Lbai->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= 2 * einsum('AD,Liab,Iabi->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('AD,Liab,Ibai->IDLA', np.identity(nextern), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= 2 * einsum('IL,ijAa,iDaj->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('IL,ijAa,jDai->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= 2 * einsum('IL,ijDa,iAaj->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += einsum('IL,ijDa,jAai->IDLA', np.identity(ncore), t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('A,IL,ijAa,ijDa->IDLA', e_extern[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('A,IL,ijAa,jiDa->IDLA', e_extern[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('D,IL,ijAa,ijDa->IDLA', e_extern[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('D,IL,ijAa,jiDa->IDLA', e_extern[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('I,AD,Iiab,Liab->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('I,AD,Iiab,Liba->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('L,AD,Iiab,Liab->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('L,AD,Iiab,Liba->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 2 * einsum('a,AD,Liab,Iiab->IDLA', e_extern[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('a,AD,Liab,Iiba->IDLA', e_extern[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('a,AD,Liba,Iiab->IDLA', e_extern[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 2 * einsum('a,AD,Liba,Iiba->IDLA', e_extern[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 2 * einsum('a,IL,ijAa,ijDa->IDLA', e_extern[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('a,IL,ijAa,jiDa->IDLA', e_extern[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('i,AD,Iiab,Liab->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,AD,Iiab,Liba->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('i,AD,Liab,Iiab->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,AD,Liab,Iiba->IDLA', e_core[ka], np.identity(nextern), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('i,IL,ijAa,ijDa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,IL,ijAa,jiDa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('i,IL,ijDa,ijAa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,IL,ijDa,jiAa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,IL,jiAa,ijDa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('i,IL,jiAa,jiDa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('i,IL,jiDa,ijAa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('i,IL,jiDa,jiAa->IDLA', e_core[ka], np.identity(ncore), t1_ccee, t1_ccee, optimize = einsum_type)


                M_[ka] += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece, optimize = einsum_type)
                M_[ka] -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece, optimize = einsum_type)

                M_[ka] += einsum('A,LiAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('A,LiAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('A,iLAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += einsum('D,LiAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('D,LiAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= 1/2 * einsum('D,iLAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('I,LiAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('I,LiAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('I,iLAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('L,LiAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('L,LiAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('L,iLAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 2 * einsum('a,LiAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('a,LiAa,iIDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('a,iLAa,IiDa->IDLA', e_extern[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('i,IiDa,LiAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,IiDa,iLAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] -= einsum('i,LiAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,LiAa,iIDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,iIDa,LiAa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)
                M_[ka] += 1/2 * einsum('i,iLAa,IiDa->IDLA', e_core[ka], t1_ccee, t1_ccee, optimize = einsum_type)





    M_ = M_.reshape(nkpts,ncore*nextern,ncore*nextern)

    return M_


def get_diag(adc,kshift,M_ab=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ab is None:
        M_ab = adc.get_imds()
    M_ = M_ab

    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    n_singles = nvir*nocc
    n_doubles = nkpts * nkpts * nocc * nocc * nvir * nvir

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    diag = np.zeros((dim), dtype=np.complex128)
    doubles = np.zeros((nkpts,nkpts,nocc*nocc*nvir*nvir),dtype=np.complex128)

    diag[s1:f1] = np.diagonal(M_[kshift])

    # Compute precond in 2p1h-2p1h block
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift,ka,kj]

                d_ij = e_occ[ki][:,None]+e_occ[kj]
                d_ab = e_vir[ka][:,None]+e_vir[kb]

                D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1))
                doubles[kj,ka] += D_ijab.reshape(-1)

    diag[s2:f2] = doubles.reshape(-1)
    log.timer_debug1("Completed ee_diag calculation")

    return diag


def matvec(adc, kshift, M_ab=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    nvir = adc.nmo - adc.nocc
    n_singles = nvir*nocc
    n_doubles = nkpts * nkpts * nocc* nocc * nvir * nvir

    s_singles = 0
    f_singles = n_singles
    s_doubles = f_singles
    f_doubles = s_doubles + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    v_ccee = eris.oovv
    v_ceec = eris.ovvo
    v_cccc = eris.oooo
    v_cecc = eris.ovoo
    v_ceee = eris.ovvv

    if M_ab is None:
        M_ab = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        cput0 = (time.process_time(), time.time())
        log = logger.Logger(adc.stdout, adc.verbose)

        r1 = r[s_singles:f_singles]
        r2 = r[s_doubles:f_doubles]
        Y = r1.reshape(nocc, nvir).copy()

        r2 = r2.reshape(nkpts,nkpts,nocc,nocc,nvir,nvir)
        s2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=np.complex128)
        cell = adc.cell
        kpts = adc.kpts
        madelung = tools.madelung(cell, kpts)

############ ADC(2) block ############################

        s1 = lib.einsum('ab,b->a',M_ab[kshift],r1)
        einsum = lib.einsum
        einsum_type = True

        for ki in range(nkpts):
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[ka,kshift,kj]

                    d_ij = e_occ[ki][:,None]+e_occ[kj]
                    d_ab = e_vir[ka][:,None]+e_vir[kb]

                    D_ijab = (-d_ij.reshape(-1,1) + d_ab.reshape(-1))
                    interm = D_ijab.reshape(-1)*r2[kb,ki].reshape(-1)
                    s2[kb,ki] += interm.reshape(nocc,nocc,nvir,nvir)

                    s2[kb,ki] += einsum('Ia,JDaC->IJCD', Y, v_ceee[ki,kj,kshift], optimize = einsum_type)
                    s2[kb,ki] += einsum('Ja,ICaD->IJCD', Y, v_ceee[ki,kj,kshift], optimize = einsum_type)
                    s2[kb,ki] -= einsum('iC,JDIi->IJCD', Y, v_cecc[ki,kj,kshift], optimize = einsum_type)
                    s2[kb,ki] -= einsum('iD,ICJi->IJCD', Y, v_cecc[ki,kj,kshift], optimize = einsum_type)

                    s1 -= einsum('Iiab,iabD->ID', r2[kb,ki], v_ceee[ki,kj,kshift], optimize = einsum_type).reshape(-1)
                    s1 += 2*einsum('Iiab,ibDa->ID', r2[kb,ki], v_ceee[ki,kj,kshift], optimize = einsum_type).reshape(-1)
                    s1 -= 2*einsum('ijDa,jaiI->ID', r2[kb,ki], v_cecc[ki,kj,kshift], optimize = einsum_type).reshape(-1)
                    s1 += einsum('ijDa,iajI->ID', r2[kb,ki], v_cecc[ki,kj,kshift], optimize = einsum_type).reshape(-1)
        s2 = s2.reshape(-1)
        s = np.hstack((s1,s2))
        del s1
        del s2

        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)

        return s
    return sigma_


def get_trans_moments(adc,kshift):

    nmo  = adc.nmo
    T = []
    for orb in range(nmo):
        T_a = get_trans_moments_orbital(adc,orb,kshift)
        T.append(T_a)

    T = np.array(T)
    return T


def get_trans_moments_orbital(adc, orb, kshift):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    t2_1 = adc.t2[0]

    idn_vir = np.identity(nvir)

    T1 = np.zeros((nvir),dtype=np.complex128)
    T2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir), dtype=np.complex128)

######## ADC(2) 1h part  ############################################

    if orb < nocc:

        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            t1_2 = adc.t1[0]
            T1 -= t1_2[kshift][orb,:]

        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = adc.khelper.kconserv[kj, ka, kshift]
                ki = adc.khelper.kconserv[ka, kj, kb]

                t2_1_t= t2_1[ki,kj,ka].transpose(1,0,2,3)
                T2[kj,ka] -= t2_1_t[:,orb,:,:].conj()

    else:

        T1 += idn_vir[(orb-nocc), :]
        for kk in range(nkpts):
            for kc in range(nkpts):
                kl = adc.khelper.kconserv[kc, kk, kshift]
                ka = adc.khelper.kconserv[kc, kl, kk]
                T1 -= 0.25* \
                    lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,
                               (orb-nocc),:], t2_1[kk,kl,ka].conj(), optimize=True)
                T1 -= 0.25* \
                    lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,
                               (orb-nocc),:], t2_1[kl,kk,ka].conj(), optimize=True)

                T1 -= 0.25* \
                    lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,
                               (orb-nocc),:], t2_1[kk,kl,ka].conj(), optimize=True)
                T1 += 0.25* \
                    lib.einsum('lkc,klac->a',t2_1[kl,kk,kshift][:,:,
                               (orb-nocc),:], t2_1[kk,kl,ka].conj(), optimize=True)
                T1 += 0.25* \
                    lib.einsum('klc,lkac->a',t2_1[kk,kl,kshift][:,:,
                               (orb-nocc),:], t2_1[kl,kk,ka].conj(), optimize=True)
                T1 -= 0.25* \
                    lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,
                               (orb-nocc),:], t2_1[kl,kk,ka].conj(), optimize=True)

######### ADC(3) 2p-1h  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1]

        if orb < nocc:

            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = adc.khelper.kconserv[kj, ka, kshift]
                    ki = adc.khelper.kconserv[ka, kj, kb]

                    t2_2_t = t2_2[ki,kj,ka].conj().transpose(1,0,2,3)

                    T2[kj,ka] -= t2_2_t[:,orb,:,:].conj()


########### ADC(3) 1p part  ############################################

    if(method=='adc(3)'):
        if orb < nocc:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    ka = adc.khelper.kconserv[kk, kc, kshift]
                    T1 += 0.5*lib.einsum('kac,ck->a',
                                         t2_1[kk,kshift,kc][:,orb,:,:], t1_2[kc].T,optimize=True)
                    T1 -= 0.5*lib.einsum('kac,ck->a',
                                         t2_1[kshift,kk,ka][orb,:,:,:], t1_2[kc].T,optimize=True)
                    T1 -= 0.5*lib.einsum('kac,ck->a',
                                         t2_1[kshift,kk,ka][orb,:,:,:], t1_2[kc].T,optimize=True)

        else:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kl = adc.khelper.kconserv[kk, kc, kshift]
                    ka = adc.khelper.kconserv[kl, kc, kk]

                    T1 -= 0.25* \
                        lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,
                                   (orb-nocc),:], t2_2[kk,kl,ka].conj(), optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,
                                   (orb-nocc),:], t2_2[kl,kk,ka].conj(), optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('klc,klac->a',t2_1[kk,kl,kshift][:,:,
                                   (orb-nocc),:], t2_2[kk,kl,ka].conj(), optimize=True)
                    T1 += 0.25* \
                        lib.einsum('klc,lkac->a',t2_1[kk,kl,kshift][:,:,
                                   (orb-nocc),:], t2_2[kl,kk,ka].conj(), optimize=True)
                    T1 += 0.25* \
                        lib.einsum('lkc,klac->a',t2_1[kl,kk,kshift][:,:,
                                   (orb-nocc),:], t2_2[kk,kl,ka].conj(), optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('lkc,lkac->a',t2_1[kl,kk,kshift][:,:,
                                   (orb-nocc),:], t2_2[kl,kk,ka].conj(), optimize=True)

                    T1 -= 0.25* \
                        lib.einsum('klac,klc->a',t2_1[kk,kl,ka].conj(),
                                   t2_2[kk,kl,kshift][:,:,(orb-nocc),:],optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('lkac,lkc->a',t2_1[kl,kk,ka].conj(),
                                   t2_2[kl,kk,kshift][:,:,(orb-nocc),:],optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('klac,klc->a',t2_1[kk,kl,ka].conj(),
                                   t2_2[kk,kl,kshift][:,:,(orb-nocc),:],optimize=True)
                    T1 += 0.25* \
                        lib.einsum('klac,lkc->a',t2_1[kk,kl,ka].conj(),
                                   t2_2[kl,kk,kshift][:,:,(orb-nocc),:],optimize=True)
                    T1 += 0.25* \
                        lib.einsum('lkac,klc->a',t2_1[kl,kk,ka].conj(),
                                   t2_2[kk,kl,kshift][:,:,(orb-nocc),:],optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('lkac,lkc->a',t2_1[kl,kk,ka].conj(),
                                   t2_2[kl,kk,kshift][:,:,(orb-nocc),:],optimize=True)

        del t2_2
    del t2_1

    for ka in range(nkpts):
        for kb in range(nkpts):
            ki = adc.khelper.kconserv[kb,kshift, ka]
            T2[ki,ka] += T2[ki,ka] - T2[ki,kb].transpose(0,2,1)

    T2 = T2.reshape(-1)
    T = np.hstack((T1,T2))

    return T


def renormalize_eigenvectors(adc, kshift, U, nroots=1):

    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    nvir = adc.nmo - adc.nocc
    n_singles = nvir

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nkpts,nkpts,nocc,nvir,nvir)
        UdotU = np.dot(U1.conj().ravel(),U1.ravel())
        for ka in range(nkpts):
            for kb in range(nkpts):
                ki = adc.khelper.kconserv[kb,kshift, ka]
                UdotU +=  2.*np.dot(U2[ki,ka].conj().ravel(), U2[ki,ka].ravel()) - \
                                    np.dot(U2[ki,ka].conj().ravel(),
                                           U2[ki,kb].transpose(0,2,1).ravel())
        U[:,I] /= np.sqrt(UdotU)

    U = U.reshape(-1,nroots)

    return U


def get_properties(adc, kshift, U, nroots=1):

    #Transition moments
    T = adc.get_trans_moments(kshift)

    #Spectroscopic amplitudes
    U = adc.renormalize_eigenvectors(kshift,U,nroots)
    X = np.dot(T, U).reshape(-1, nroots)

    #Spectroscopic factors
    P = 2.0*lib.einsum("pi,pi->i", X.conj(), X)
    P = P.real

    return P,X


class RADCEE(kadc_rhf.RADC):
    '''restricted ADC for EA energies and spectroscopic amplitudes

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default value equals to :class:`Mole.max_memory`
        incore_complete : bool
            Avoid all I/O. Default is False.
        method : string
            nth-order ADC method. Options are : ADC(2), ADC(2)-X, ADC(3). Default is ADC(2).
        conv_tol : float
            Convergence threshold for Davidson iterations.  Default is 1e-12.
        max_cycle : int
            Number of Davidson iterations.  Default is 50.
        max_space : int
            Space size to hold trial vectors for Davidson iterative diagonalization.  Default is 12.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested. Default value is 1.

            >>> myadc = adc.RADC(mf).run()
            >>> myadcip = adc.RADC(myadc).run()

    Saved results

        e_ea : float or list of floats
            EA energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1,
            it is a list of floats for the lowest nroots eigenvalues.
        v_ea : array
            Eigenvectors for each EA transition.
        p_ea : float
            Spectroscopic amplitudes for each EA transition.
    '''

    def __init__(self, adc):
        self.verbose = adc.verbose
        self.stdout = adc.stdout
        self.max_memory = adc.max_memory
        self.max_space = adc.max_space
        self.max_cycle = adc.max_cycle
        self.conv_tol  = adc.conv_tol
        self.tol_residual  = adc.tol_residual
        self.t1 = adc.t1
        self.t2 = adc.t2
        self.method = adc.method
        self.method_type = adc.method_type
        self._scf = adc._scf
        self.compute_properties = adc.compute_properties
        self.approx_trans_moments = adc.approx_trans_moments

        self.kpts = adc._scf.kpts
        self.exxdiv = adc.exxdiv
        self.verbose = adc.verbose
        self.max_memory = adc.max_memory
        self.method = adc.method

        self.khelper = adc.khelper
        self.cell = adc.cell
        self.mo_coeff = adc.mo_coeff
        self.mo_occ = adc.mo_occ
        self.frozen = adc.frozen

        self._nocc = adc._nocc
        self._nmo = adc._nmo
        self._nvir = adc._nvir
        self.nkop_chk = adc.nkop_chk
        self.kop_npick = adc.kop_npick

        self.t2 = adc.t2
        self.e_corr = adc.e_corr
        self.mo_energy = adc.mo_energy
        self.imds = adc.imds
        self.chnk_size = adc.chnk_size

        keys = set(('tol_residual','conv_tol', 'e_corr', 'method', 'mo_coeff', 'mo_energy_b',
                   'max_memory', 't1', 'mo_energy_a', 'max_space', 't2', 'max_cycle'))

        self._keys = set(self.__dict__.keys()).union(keys)

    kernel = kadc_rhf.kernel
    get_imds = get_imds
    get_diag = get_diag
    matvec = matvec
    vector_size = vector_size
    get_trans_moments = get_trans_moments
    renormalize_eigenvectors = renormalize_eigenvectors
    get_properties = get_properties

    def get_init_guess(self, nroots=1, diag=None, ascending=True):
        if diag is None :
            diag = self.get_diag()
        idx = None
        dtype = getattr(diag, 'dtype', np.complex128)
        if ascending:
            idx = np.argsort(diag)
        else:
            idx = np.argsort(diag)[::-1]
        guess = np.zeros((diag.shape[0], nroots), dtype=dtype)
        min_shape = min(diag.shape[0], nroots)
        guess[:min_shape,:min_shape] = np.identity(min_shape)
        g = np.zeros((diag.shape[0], nroots), dtype=dtype)
        g[idx] = guess.copy()
        guess = []
        for p in range(g.shape[1]):
            guess.append(g[:,p])
        return guess

    def gen_matvec(self,kshift,imds=None, eris=None):
        if imds is None:
            imds = self.get_imds(eris)
        diag = self.get_diag(kshift,imds,eris)
        matvec = self.matvec(kshift, imds, eris)
        return matvec, diag


def ea_contract_r_vvvv(adc,r2,vvvv,ka,kb,kc):

    nocc = r2.shape[0]
    nvir = r2.shape[1]
    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv

    kd = kconserv[ka, kc, kb]
    r2 = np.ascontiguousarray(r2.reshape(nocc,-1))
    r2_vvvv = np.zeros((nvir,nvir,nocc),dtype=r2.dtype)
    chnk_size = adc.chnk_size
    if chnk_size > nvir:
        chnk_size = nvir

    a = 0
    if isinstance(vvvv, np.ndarray):
        vv1 = vvvv[ka,kc]
        vv2 = vvvv[kb,kd]
        for p in range(0,nvir,chnk_size):
            vvvv_p = dfadc.get_vvvv_df(adc, vv1, vv2, p, chnk_size)/nkpts
            k = vvvv_p.shape[0]
            vvvv_p = vvvv_p.reshape(-1,nvir*nvir)
            r2_vvvv[a:a+k] += np.dot(vvvv_p.conj(),r2.T).reshape(-1,nvir,nocc)
            del vvvv_p
            a += k
    else :
        for p in range(0,nvir,chnk_size):
            vvvv_p = vvvv[ka,kb,kc][p:p+chnk_size].reshape(-1,nvir*nvir)
            k = vvvv_p.shape[0]
            r2_vvvv[a:a+k] += np.dot(vvvv_p.conj(),r2.T).reshape(-1,nvir,nocc)
            del vvvv_p
            a += k

    r2_vvvv = np.ascontiguousarray(r2_vvvv.transpose(2,0,1))

    return r2_vvvv
