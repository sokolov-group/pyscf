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
                               padded_mo_coeff, get_frozen_mask, _add_padding, padded_mo_energy)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.cc.kccsd_t_rhf import _get_epqr
from pyscf.pbc.lib import kpts_helper
from pyscf.lib.parameters import LOOSE_ZERO_TOL, LARGE_DENOM  # noqa

from pyscf.pbc import tools
import h5py
import tempfile
from pyscf.pbc.adc.kadc_rhf_amplitudes import gen_t2_1

def vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc
    size = n_singles + n_doubles

    return size
def get_imds(adc, eris=None):

    #cput0 = (time.process_time(), time.time())
    cput0 = (time.process_time(), time.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc =  adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_occ = np.identity(nocc)
    M_ij = np.empty((nkpts,nocc,nocc),dtype=mo_coeff.dtype)
    #M_ij = np.empty((nkpts,nocc,nocc),dtype=eris.Loo.dtype)

    if eris is None:
        eris = adc.transform_integrals()

    # i-j block
    # Zeroth-order terms

    #t2_1 = adc.t2[0]
    #eris_ovov = eris.ovov
    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ , e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                #t2_1 = adc.t2[0]

                #t2_1_ild = adc.t2[0][ki,kl,kd]
                eris_ovov_jdl = eris_ovov_idl = 1./nkpts * lib.einsum('Ljd,Lle->jdle'
                                , eris.Lov[ki,kd], eris.Lov[kl,ke], optimize=True)
                eris_ovov_jel = eris_ovov_iel = 1./nkpts * lib.einsum('Lje,Lld->jeld'
                                , eris.Lov[ki,ke], eris.Lov[kl,kd], optimize=True)

                t2_1_jld = t2_1_ild = gen_t2_1(adc,eris,(ki,kl,kd,ke))
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov_jdl,optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ilde,jeld->ij',t2_1_ild, eris_ovov_jel,optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ild,
                                             eris_ovov_jdl,optimize=True)
                #del t2_1_ild

                #t2_1_lid = adc.t2[0][kl,ki,kd]
                t2_1_ljd = t2_1_lid = gen_t2_1(adc,eris,(kl,ki,kd,ke))
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('lide,jdle->ij',t2_1_lid, eris_ovov_jdl,optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('lide,jeld->ij',t2_1_lid, eris_ovov_jel,optimize=True)
                #del t2_1_lid
                #t2_1_jld = adc.t2[0][kj,kl,kd]
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('jlde,idle->ij',t2_1_jld.conj(),
                               eris_ovov_idl.conj(),optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('jlde,ield->ij',t2_1_jld.conj(),
                               eris_ovov_iel.conj(),optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('jlde,idle->ij',t2_1_jld.conj(),
                                             eris_ovov_idl.conj(),optimize=True)
                #del t2_1_jld,t2_1_ild
                #t2_1_ljd = adc.t2[0][kl,kj,kd]
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ljde,idle->ij',t2_1_ljd.conj(),
                               eris_ovov_idl.conj(),optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ljde,ield->ij',t2_1_ljd.conj(),
                               eris_ovov_iel.conj(),optimize=True)
                del t2_1_ljd,t2_1_lid
                #del t2_1

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    return M_ij
    #return np.ndarray.astype(M_ij, dtype=np.complex128)

def get_imds_off(adc, eris=None):

    #cput0 = (time.process_time(), time.time())
    cput0 = (time.process_time(), time.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method
    nkpts = adc.nkpts

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc =  adc.nocc
    kconserv = adc.khelper.kconserv

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_vir = np.array(e_vir)

    idn_occ = np.identity(nocc)
    M_ij = np.empty((nkpts,nocc,nocc),dtype=mo_coeff.dtype)

    if eris is None:
        eris = adc.transform_integrals()

    # i-j block
    # Zeroth-order terms
    print('running projected CVS')
    t2_1 = adc.t2[0]
    t2_1_temp = gen_t2_1
    print('successfully executed gen_t2_1(adc, eris)')
    #exit()
    eris_ovov = eris.ovov
    Lov = eris.Lov
    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ , e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                #t2_1 = adc.t2[0]
                t2_1_ild = adc.t2[0][ki,kl,kd]

                #t2_1_ild = gen_t2_1(adc,eris,(ki,kl,kd,ke))
                #t2_1_ild_test = t2_1_temp(adc,eris,(ki,kl,kd,ke))[1][0]
                #print(f'(ki={ki}|kl={kl}|kd={kd}) norm(t2_1_ild_test - t2_1_ild) \
                #= {np.linalg.norm(t2_1_ild_test - t2_1_ild)}')
                ##M_ij[ki] += 0.5 * 0.5 * \
                ##    lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov[kj,kd,kl],optimize=True)
                ##M_ij[ki] -= 0.5 * 0.5 * \
                ##    lib.einsum('ilde,jeld->ij',t2_1_ild, eris_ovov[kj,ke,kl],optimize=True)
                ##M_ij[ki] += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ild,
                ##                             eris_ovov[kj,kd,kl],optimize=True)

                ###eris_ovov_jdl = 1/nkpts * lib.einsum('Ljd,Lle->jdle'
                ###                , Lov[kj,kd], Lov[kl,ke], optimize=True)
                ###eris_ovov_jel = 1/nkpts * lib.einsum('Lje,Lld->jeld'
                ###                , Lov[kj,ke], Lov[kl,kd], optimize=True)
                eris_ovov_jdl = eris_ovov[kj,kd,kl] 
                eris_ovov_jel = eris_ovov[kj,ke,kl]

                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov_jdl,optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ilde,jeld->ij',t2_1_ild, eris_ovov_jel,optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ild,
                                             eris_ovov_jdl,optimize=True)
                #del t2_1_ild

                t2_1_lid = adc.t2[0][kl,ki,kd]
                #t2_1_lid = gen_t2_1(adc,eris,(kl,ki,kd,ke))
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('lide,jdle->ij',t2_1_lid, eris_ovov_jdl,optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('lide,jeld->ij',t2_1_lid, eris_ovov_jel,optimize=True)
                #del t2_1_lid
                #del eris_ovov_jdl
                #del eris_ovov_jel

                #eris_ovov_idl = 1./nkpts * lib.einsum('Lid,Lle->idle'
                #                , Lov[ki,kd], Lov[kl,ke], optimize=True).conj()
                #eris_ovov_iel = 1./nkpts * lib.einsum('Lie,Lld->ield'
                #                , Lov[ki,ke], Lov[kl,kd], optimize=True).conj()
                #eris_ovov_idl = eris_ovov[ki,kd,kl].conj() 
                #eris_ovov_iel = eris_ovov[ki,ke,kl].conj()
                eris_ovov_idl = eris_ovov_jdl.conj() 
                eris_ovov_iel = eris_ovov_jel.conj()
                t2_1_ljd = adc.t2[0][kl,kj,kd]
                #t2_1_ljd = t2_1_lid
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ljde,idle->ij',t2_1_ljd.conj(),
                               eris_ovov_idl,optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ljde,ield->ij',t2_1_ljd.conj(),
                               eris_ovov_iel,optimize=True)
                del t2_1_ljd
                del t2_1_lid
                #del t2_1
                del eris_ovov_idl
                del eris_ovov_iel
                del eris_ovov_jdl
                del eris_ovov_jel

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)
    if (method == "adc(3)"):
        t1_2 = adc.t1[0]
        eris_ovoo = eris.ovoo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv
        eris_oooo = eris.oooo

        for ki in range(nkpts):
            kj = ki
            for kl in range(nkpts):
                kd = kconserv[kj,ki,kl]
                M_ij[ki] += 2. * lib.einsum('ld,ldji->ij',t1_2[kl],
                                            eris_ovoo[kl,kd,kj],optimize=True)
                M_ij[ki] -=      lib.einsum('ld,jdli->ij',t1_2[kl],
                                            eris_ovoo[kj,kd,kl],optimize=True)

                kd = kconserv[ki,kj,kl]
                M_ij[ki] += 2. * lib.einsum('ld,ldij->ij',t1_2[kl].conj(),
                                            eris_ovoo[kl,kd,ki].conj(),optimize=True)
                M_ij[ki] -=      lib.einsum('ld,idlj->ij',t1_2[kl].conj(),
                                            eris_ovoo[ki,kd,kl].conj(),optimize=True)

                for kd in range(nkpts):
                    t2_1 = adc.t2[0]

                    ke = kconserv[kj,kd,kl]
                    t2_2_ild = adc.t2[1][ki,kl,kd]
                    M_ij[ki] += 0.5 * 0.5* \
                        lib.einsum('ilde,jdle->ij',t2_2_ild, eris_ovov[kj,kd,kl],optimize=True)
                    M_ij[ki] -= 0.5 * 0.5* \
                        lib.einsum('ilde,jeld->ij',t2_2_ild, eris_ovov[kj,ke,kl],optimize=True)
                    M_ij[ki] += 0.5 * \
                        lib.einsum('ilde,jdle->ij',t2_2_ild, eris_ovov[kj,kd,kl],optimize=True)
                    del t2_2_ild

                    t2_2_lid = adc.t2[1][kl,ki,kd]
                    M_ij[ki] -= 0.5 * 0.5* \
                        lib.einsum('lide,jdle->ij',t2_2_lid, eris_ovov[kj,kd,kl],optimize=True)
                    M_ij[ki] += 0.5 * 0.5* \
                        lib.einsum('lide,jeld->ij',t2_2_lid, eris_ovov[kj,ke,kl],optimize=True)

                    t2_2_jld = adc.t2[1][kj,kl,kd]
                    M_ij[ki] += 0.5 * 0.5* \
                        lib.einsum('jlde,leid->ij',t2_2_jld.conj(),
                                   eris_ovov[kl,ke,ki].conj(),optimize=True)
                    M_ij[ki] -= 0.5 * 0.5* \
                        lib.einsum('jlde,ield->ij',t2_2_jld.conj(),
                                   eris_ovov[ki,ke,kl].conj(),optimize=True)
                    M_ij[ki] += 0.5 *      lib.einsum('jlde,leid->ij',t2_2_jld.conj(),
                                                      eris_ovov[kl,ke,ki].conj(),optimize=True)

                    t2_2_ljd = adc.t2[1][kl,kj,kd]
                    M_ij[ki] -= 0.5 * 0.5* \
                        lib.einsum('ljde,leid->ij',t2_2_ljd.conj(),
                                   eris_ovov[kl,ke,ki].conj(),optimize=True)
                    M_ij[ki] += 0.5 * 0.5* \
                        lib.einsum('ljde,ield->ij',t2_2_ljd.conj(),
                                   eris_ovov[ki,ke,kl].conj(),optimize=True)

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                t2_1 = adc.t2[0]

                kl = kconserv[kd,km,ke]
                kf = kconserv[kj,kd,kl]
                temp_t2_v_1 = lib.einsum(
                    'lmde,jldf->mejf',t2_1[kl,km,kd].conj(), t2_1[kj,kl,kd],optimize=True)
                M_ij[ki] -=  0.5 * 2 * lib.einsum('mejf,ifem->ij',
                                                  temp_t2_v_1, eris_ovvo[ki,kf,ke],optimize=True)
                M_ij[ki] -=  0.5 * 2 * lib.einsum('meif,jfem->ij',temp_t2_v_1.conj(),
                                                  eris_ovvo[kj,kf,ke].conj(),optimize=True)
                M_ij[ki] +=  0.5 *     lib.einsum('mejf,imef->ij',
                                                  temp_t2_v_1, eris_oovv[ki,km,ke],optimize=True)
                M_ij[ki] +=  0.5 *     lib.einsum('meif,jmef->ij',temp_t2_v_1.conj(),
                                                  eris_oovv[kj,km,ke].conj(),optimize=True)

                temp_t2_v_new = lib.einsum(
                    'mlde,ljdf->mejf',t2_1[km,kl,kd].conj(), t2_1[kl,kj,kd],optimize=True)
                M_ij[ki] -=  0.5 * 2 * lib.einsum('mejf,ifem->ij',
                                                  temp_t2_v_new, eris_ovvo[ki,kf,ke],optimize=True)
                M_ij[ki] +=  0.5 *     lib.einsum('mejf,imef->ij',
                                                  temp_t2_v_new, eris_oovv[ki,km,ke],optimize=True)
                M_ij[ki] -=  0.5 * 2 * lib.einsum('meif,jfem->ij',temp_t2_v_new.conj(),
                                                  eris_ovvo[kj,kf,ke].conj(),optimize=True)
                M_ij[ki] +=  0.5 *     lib.einsum('meif,jmef->ij',temp_t2_v_new.conj(),
                                                  eris_oovv[kj,km,ke].conj(),optimize=True)
                del temp_t2_v_new

                temp_t2_v_2 = lib.einsum(
                    'lmde,ljdf->mejf',t2_1[kl,km,kd].conj(), t2_1[kl,kj,kd],optimize=True)
                M_ij[ki] +=  0.5 * 4 * lib.einsum('mejf,ifem->ij',
                                                  temp_t2_v_2, eris_ovvo[ki,kf,ke],optimize=True)
                M_ij[ki] +=  0.5 * 4 * lib.einsum('meif,jfem->ij',temp_t2_v_2.conj(),
                                                  eris_ovvo[kj,kf,ke].conj(),optimize=True)
                M_ij[ki] -=  0.5 * 2 * lib.einsum('meif,jmef->ij',temp_t2_v_2.conj(),
                                                  eris_oovv[kj,km,ke].conj(),optimize=True)
                M_ij[ki] -=  0.5 * 2 * lib.einsum('mejf,imef->ij',
                                                  temp_t2_v_2, eris_oovv[ki,km,ke],optimize=True)
                del temp_t2_v_2

                temp_t2_v_3 = lib.einsum(
                    'mlde,jldf->mejf',t2_1[km,kl,kd].conj(), t2_1[kj,kl,kd],optimize=True)
                M_ij[ki] += 0.5 *    lib.einsum('mejf,ifem->ij',
                                                temp_t2_v_3, eris_ovvo[ki,kf,ke],optimize=True)
                M_ij[ki] += 0.5 *    lib.einsum('meif,jfem->ij',temp_t2_v_3.conj(),
                                                eris_ovvo[kj,kf,ke].conj(),optimize=True)
                M_ij[ki] -= 0.5 *2 * lib.einsum('meif,jmef->ij',temp_t2_v_3.conj(),
                                                eris_oovv[kj,km,ke].conj(),optimize=True)
                M_ij[ki] -= 0.5 *2 * lib.einsum('mejf,imef->ij',
                                                temp_t2_v_3, eris_oovv[ki,km,ke],optimize=True)
                del temp_t2_v_3

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):

                kl = kconserv[kd,km,ke]
                kf = kconserv[kl,kd,km]
                temp_t2_v_8 = lib.einsum(
                    'lmdf,lmde->fe',t2_1[kl,km,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] += 3.0 * lib.einsum('fe,jief->ij',temp_t2_v_8,
                                             eris_oovv[kj,ki,ke], optimize=True)
                M_ij[ki] -= 1.5 * lib.einsum('fe,jfei->ij',temp_t2_v_8,
                                             eris_ovvo[kj,kf,ke], optimize=True)
                M_ij[ki] +=       lib.einsum('ef,jief->ij',temp_t2_v_8.T,
                                             eris_oovv[kj,ki,ke], optimize=True)
                M_ij[ki] -= 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_8.T,
                                             eris_ovvo[kj,kf,ke], optimize=True)
                del temp_t2_v_8

                temp_t2_v_9 = lib.einsum(
                    'lmdf,mlde->fe',t2_1[kl,km,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('fe,jief->ij',temp_t2_v_9,
                                             eris_oovv[kj,ki,ke], optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('ef,jief->ij',temp_t2_v_9.T,
                                             eris_oovv[kj,ki,ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('fe,jfei->ij',temp_t2_v_9,
                                             eris_ovvo[kj,kf,ke], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ef,jfei->ij',temp_t2_v_9.T,
                                             eris_ovvo[kj,kf,ke], optimize=True)
                del temp_t2_v_9

                kl = kconserv[kd,km,ke]
                kn = kconserv[kd,kl,ke]
                temp_t2_v_10 = lib.einsum(
                    'lnde,lmde->nm',t2_1[kl,kn,kd], t2_1[kl,km,kd].conj(),optimize=True)
                M_ij[ki] -= 3.0 * lib.einsum('nm,jinm->ij',temp_t2_v_10,
                                             eris_oooo[kj,ki,kn], optimize=True)
                M_ij[ki] -= 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_10.T,
                                             eris_oooo[kj,ki,kn], optimize=True)
                M_ij[ki] += 1.5 * lib.einsum('nm,jmni->ij',temp_t2_v_10,
                                             eris_oooo[kj,km,kn], optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_10.T,
                                             eris_oooo[kj,km,kn], optimize=True)
                del temp_t2_v_10

                temp_t2_v_11 = lib.einsum(
                    'lnde,mlde->nm',t2_1[kl,kn,kd], t2_1[km,kl,kd].conj(),optimize=True)
                M_ij[ki] += 1.0 * lib.einsum('nm,jinm->ij',temp_t2_v_11,
                                             eris_oooo[kj,ki,kn], optimize=True)
                M_ij[ki] -= 0.5 * lib.einsum('nm,jmni->ij',temp_t2_v_11,
                                             eris_oooo[kj,km,kn], optimize=True)
                M_ij[ki] -= 0.5 * lib.einsum('mn,jmni->ij',temp_t2_v_11.T,
                                             eris_oooo[kj,km,kn], optimize=True)
                M_ij[ki] += 1.0 * lib.einsum('mn,jinm->ij',temp_t2_v_11.T,
                                             eris_oooo[kj,ki,kn], optimize=True)
                del temp_t2_v_11

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                t2_1 = adc.t2[0]
                kl = kconserv[kd,km,ke]
                kn = kconserv[kd,ki,ke]
                temp_t2_v_12 = lib.einsum(
                    'inde,lmde->inlm',t2_1[ki,kn,kd].conj(), t2_1[kl,km,kd],optimize=True)
                M_ij[ki] += 0.5 * 1.25 * \
                    lib.einsum('inlm,jlnm->ij',temp_t2_v_12,
                               eris_oooo[kj,kl,kn].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('inlm,jmnl->ij',temp_t2_v_12,
                               eris_oooo[kj,km,kn].conj(), optimize=True)

                M_ij[ki] += 0.5 * 1.25 * \
                    lib.einsum('jnlm,ilnm->ij',temp_t2_v_12.conj(),
                               eris_oooo[ki,kl,kn], optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('jnlm,imnl->ij',temp_t2_v_12.conj(),
                               eris_oooo[ki,km,kn], optimize=True)
                del temp_t2_v_12

                temp_t2_v_12_1 = lib.einsum(
                    'nide,mlde->inlm',t2_1[kn,ki,kd].conj(), t2_1[km,kl,kd],optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('inlm,jlnm->ij',temp_t2_v_12_1,
                               eris_oooo[kj,kl,kn].conj(), optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('inlm,jmnl->ij',temp_t2_v_12_1,
                               eris_oooo[kj,km,kn].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('jnlm,ilnm->ij',temp_t2_v_12_1.conj(),
                               eris_oooo[ki,kl,kn], optimize=True)
                M_ij[ki] -= 0.5 * 0.25 * \
                    lib.einsum('jnlm,imnl->ij',temp_t2_v_12_1.conj(),
                               eris_oooo[ki,km,kn], optimize=True)

                temp_t2_v_13 = lib.einsum(
                    'inde,mlde->inml',t2_1[ki,kn,kd].conj(), t2_1[km,kl,kd],optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('inml,jlnm->ij',temp_t2_v_13,
                               eris_oooo[kj,kl,kn].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('inml,jmnl->ij',temp_t2_v_13,
                               eris_oooo[kj,km,kn].conj(), optimize=True)

                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('jnml,ilnm->ij',temp_t2_v_13.conj(),
                               eris_oooo[ki,kl,kn], optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('jnml,imnl->ij',temp_t2_v_13.conj(),
                               eris_oooo[ki,km,kn], optimize=True)
                del temp_t2_v_13

                temp_t2_v_13_1 = lib.einsum(
                    'nide,lmde->inml',t2_1[kn,ki,kd].conj(), t2_1[kl,km,kd],optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('inml,jmnl->ij',temp_t2_v_13_1,
                               eris_oooo[kj,km,kn].conj(), optimize=True)
                M_ij[ki] += 0.5 * 0.25 * \
                    lib.einsum('jnml,imnl->ij',temp_t2_v_13_1.conj(),
                               eris_oooo[ki,km,kn], optimize=True)
                del temp_t2_v_13_1

    cput0 = log.timer_debug1("completed imds calculation", *cput0)
    return M_ij

def make_rdm1_eigenvectors(adc, L, R):

    L = np.array(L).ravel()
    R = np.array(R).ravel()

    t2_1 = adc.t2[0][:]
    t1_2 = adc.t1[0][:]

    nocc = adc._nocc
    nvir = adc._nvir
    nmo = nocc + nvir
    nkpts = adc.nkpts

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    rdm1  = np.zeros((nkpts,nmo,nmo))
    kd_oc = np.identity(nocc)

    L1 = L[s1:f1]
    L2 = L[s2:f2]
    R1 = R[s1:f1]
    R2 = R[s2:f2]

    L2 = L2.reshape(nkpts,nkpts,nvir,nocc,nocc)
    R2 = R2.reshape(nkpts,nkpts,nvir,nocc,nocc)

    for kj in range(nkpts):
        for kk in range(nkpts):
            ka = kconserv[kk, kshift, kj]
            for kl in range(nkpts):
                ki = kconserv[kj, kl, kk]
                #G^000#### block- ij
                rdm1[i][:nocc,:nocc] =  2*np.einsum('ij,m,m->ij',kd_oc,L1,R1,optimize=True)
                rdm1[i][:nocc,:nocc] -= np.einsum('i,j->ij',L1,R1,optimize=True)

                #rdm1[i][:nocc,:nocc] += 4*np.einsum('ij,etu,etu->ij',kd_oc,L2,R2,optimize=True)
                #rdm1[i][:nocc,:nocc] -= np.einsum('ij,etu,eut->ij',kd_oc,L2,R2,optimize=True)
                #rdm1[i][:nocc,:nocc] -= np.einsum('ij,eut,etu->ij',kd_oc,L2,R2,optimize=True)
                #rdm1[i][:nocc,:nocc] -= 4*np.einsum('eti,etj->ij',L2,R2,optimize=True)
                #rdm1[i][:nocc,:nocc] += np.einsum('eti,ejt->ij',L2,R2,optimize=True)
                #rdm1[i][:nocc,:nocc] += np.einsum('eit,etj->ij',L2,R2,optimize=True)

                #rdm1[i][:nocc,:nocc] -= 2*np.einsum('g,g,hjcd,hicd->ij', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[i][:nocc,:nocc] += 2*np.einsum('g,g,jhcd,hicd->ij', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[i][:nocc,:nocc] -= 2*np.einsum('g,g,jhcd,ihcd->ij', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[i][:nocc,:nocc] += 2*np.einsum('g,h,jgcd,ihcd->ij', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[i][:nocc,:nocc] -= np.einsum('g,h,gjcd,ihcd->ij', L1,R1,t2_1,t2_1,optimize=True)
                #temp_ij = np.einsum('g,j,ghcd,ihcd->ij',L1,R1,t2_1,t2_1,optimize=True)
                #temp_ij -= 0.25*np.einsum('g,j,hgcd,ihcd->ij',L1,R1,t2_1,t2_1,optimize=True)
                #temp_ij -= 0.25*np.einsum('g,j,ghcd,hicd->ij',L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[i][:nocc,:nocc] += temp_ij
                #rdm1[i][:nocc,:nocc] += temp_ij.T

                ###### block- ab
                #rdm1[nocc:,nocc:] = np.einsum('atu,btu->ab', L2,R2,optimize=True)
                #rdm1[nocc:,nocc:] -= 0.5*np.einsum('aut,btu->ab', L2,R2,optimize=True)
                #rdm1[nocc:,nocc:] -= 0.5*np.einsum('atu,but->ab', L2,R2,optimize=True)
                #rdm1[nocc:,nocc:] += np.einsum('atu,btu->ab', L2,R2,optimize=True)

                #rdm1[nocc:,nocc:] += 4*np.einsum('g,g,hmbc,hmac->ab', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[nocc:,nocc:] -= np.einsum('g,g,mhbc,hmac->ab', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[nocc:,nocc:] -= np.einsum('g,g,hmbc,mhac->ab', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[nocc:,nocc:] -= 4*np.einsum('g,h,hmbc,gmac->ab', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[nocc:,nocc:] += np.einsum('g,h,mhbc,gmac->ab', L1,R1,t2_1,t2_1,optimize=True)
                #rdm1[nocc:,nocc:] += np.einsum('g,h,hmbc,mgac->ab', L1,R1,t2_1,t2_1,optimize=True)

                ######## block- ia
                #rdm1[:nocc,nocc:] = -np.einsum('n,ani->ia', R1,L2,optimize=True)
                #rdm1[:nocc,nocc:] += 2*np.einsum('n,ain->ia', R1,L2,optimize=True)

                #rdm1[:nocc,nocc:] -= 2*np.einsum('g,cgh,ihac->ia', L1,R2,t2_1,optimize=True)
                #rdm1[:nocc,nocc:] += np.einsum('g,cgh,hiac->ia', L1,R2,t2_1,optimize=True)
                #rdm1[:nocc,nocc:] += 4*np.einsum('g,chg,ihac->ia', L1,R2,t2_1,optimize=True)
                #rdm1[:nocc,nocc:] -= np.einsum('g,chg,hiac->ia', L1,R2,t2_1,optimize=True)
                #rdm1[:nocc,nocc:] -= np.einsum('g,cgh,ihac->ia', L1,R2,t2_1,optimize=True)
                #rdm1[:nocc,nocc:] += np.einsum('i,cgh,ghac->ia', L1,R2,t2_1,optimize=True)
                #rdm1[:nocc,nocc:] -= 2*np.einsum('i,chg,ghac->ia', L1,R2,t2_1,optimize=True)

                #rdm1[:nocc,nocc:] += np.einsum('g,g,ia->ia', L1,R1,t1_2,optimize=True)
                #rdm1[:nocc,nocc:] += np.einsum('g,g,ia->ia', L1,R1,t1_2,optimize=True)
                #rdm1[:nocc,nocc:] -= np.einsum('g,i,ga->ia', R1,L1,t1_2,optimize=True)

                ######### block- ai
                #rdm1[nocc:,:nocc] = rdm1[:nocc,nocc:].T

    return rdm1



def cvs_projector(adc, r, diag=False):
    
    ncvs_proj = adc.ncvs_proj
    nkpts = adc.nkpts
    nocc = adc.nocc
    nvir = adc.nmo - adc.nocc
    diag_shift = -1e9

    n_singles = nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc
    
    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles 
    
    Pr = r.copy()
    new_h2 = np.zeros((nkpts, nkpts, nvir, nocc, nocc),dtype=Pr.dtype)
    new_h2 = Pr[s2:f2].reshape((nkpts, nkpts, nvir, nocc, nocc)).copy()

    if diag:
        Pr[ncvs_proj:f1] += diag_shift 
        new_h2[:,:,:,ncvs_proj:,ncvs_proj:] += diag_shift
        Pr[s2:f2] = new_h2.reshape(-1)
        norm_func = np.linalg.norm
        print(f'norm of proj diag^-1 = {norm_func(np.multiply(Pr, Pr.conj())**-1)}')
    else:
        Pr[ncvs_proj:f1] = 0 
        #Pr[s1:f1] = 0 
        new_h2[:,:,:,ncvs_proj:,ncvs_proj:] = 0
        #new_h2[:,:,:,:ncvs_proj,ncvs_proj:] = 0
        #new_h2[:,:,:,ncvs_proj:,:ncvs_proj] = 0
        #new_h2[:,:,:,:ncvs_proj,:ncvs_proj] = 0
        Pr[s2:f2] = new_h2.reshape(-1)
    
    return Pr

def get_diag(adc,kshift,M_ij=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv
    nocc = adc.nocc
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

    dim = n_singles + n_doubles

    s1 = 0
    f1 = n_singles
    s2 = f1
    f2 = s2 + n_doubles

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    diag = np.zeros((dim), dtype=np.complex128)
    doubles = np.zeros((nkpts,nkpts,nvir*nocc*nocc),dtype=np.complex128)
    #diag = np.zeros((dim), dtype=np.complex64)
    #doubles = np.zeros((nkpts,nkpts,nvir*nocc*nocc),dtype=np.complex64)

    M_ij_diag = np.diagonal(M_ij[kshift])
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    for ka in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[kshift,ki,ka]
            d_ij = e_occ[ki][:,None] + e_occ[kj]
            d_a = e_vir[ka][:,None]
            D_n = -d_a + d_ij.reshape(-1)
            doubles[ka,ki] += D_n.reshape(-1)

    diag[s2:f2] = doubles.reshape(-1)

    if adc.ncvs_proj is not None:
        diag = cvs_projector(adc, diag, diag=True)

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

    return diag


def matvec(adc, kshift, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    kconserv = adc.khelper.kconserv
    n_singles = nocc
    nvir = adc.nmo - adc.nocc
    n_doubles = nkpts * nkpts * nvir * nocc * nocc

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

    if M_ij is None:
        M_ij = adc.get_imds()

    ###for nkpts_i in range(nkpts):
    ###    e_ij, _ = np.linalg.eig(M_ij[nkpts_i,:adc.ncvs_proj,:adc.ncvs_proj])
    ###    print(f'e_ij_{nkpts_i} = {e_ij}')
    ###exit()
    #Calculate sigma vector
    print('running projected CVS')
    mask_jk = np.ones((nocc,nocc), dtype=bool)
    ncvs = adc.ncvs_proj
    mask_jk[ncvs:,ncvs:] = False
    mask_kj = mask_jk.T 
    def sigma_(r):
        #cput0 = (time.process_time(), time.time())
        cput0 = (time.process_time(), time.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        if adc.ncvs_proj is not None:
            r = cvs_projector(adc, r)

        if eris.Loo.dtype == np.complex64:
            r = np.ndarray.astype(r, dtype=np.complex64)
        print(f'r vector dtype = {r.dtype}')
        r1 = r[s_singles:f_singles]
        r2 = r[s_doubles:f_doubles]

        r2 = r2.reshape(nkpts,nkpts,nvir,nocc,nocc)
        ncvs = adc.ncvs_proj
        r2_ecc = r2[:,:,:,:ncvs,:ncvs].copy()
        r2_ecv = r2[:,:,:,:ncvs,ncvs:].copy()
        r2_evc = r2[:,:,:,ncvs:,:ncvs].copy()
        #s2 = np.zeros((nkpts,nkpts,nvir,nocc,nocc), dtype=np.complex128)
        s2 = np.zeros((nkpts,nkpts,nvir,nocc,nocc), dtype=r.dtype)
        cell = adc.cell
        kpts = adc.kpts
        madelung = tools.madelung(cell, kpts)

        #eris_ovoo = eris.ovoo
        Loo = eris.Loo
        Lov = eris.Lov
        Lvo = eris.Lvo

        def get_t():
            return np.array((time.process_time(), time.perf_counter()))
        
        full_adc_time_count = np.zeros((1,2))        
        cvs_adc_time_count = np.zeros((1,2))        
############ ADC(2) ij block ############################
        #r1[ncvs:] = 0  
        s1 = lib.einsum('ij,j->i',M_ij[kshift],r1)
        #s1[ncvs:] = 0
########### ADC(2) i - kja block #########################
        norm_count = 0
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj] 
                ki = kconserv[kj, kk, ka]
                ncvs = adc.ncvs_proj
                #print(f'kj = {kj} | ka = {ka} | kk = {kk} | ki = {ki} | kshift = {kshift}')

                s_temp = np.zeros(s1.size, dtype=np.complex128)

                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk].conj(), r2[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj].conj(), r2[ka,kj], optimize=True)

                #eris_ovoo_jak = 1./nkpts * lib.einsum('Lja,Lki->jaki', Lov[kj,ka], Loo[kk,ki], optimize=True)
                #eris_ovoo_kaj = 1./nkpts * lib.einsum('Lka,Lji->kaji', Lov[kk,ka], Loo[kj,ki], optimize=True)
                full_t_1a = get_t()
                eris_vooo_aji = 1./nkpts * lib.einsum('Laj,Lik->ajik', Lvo[ka,kj], Loo[ki,kk], optimize=True)
                eris_vooo_aki = 1./nkpts * lib.einsum('Lak,Lij->akij', Lvo[ka,kk], Loo[ki,kj], optimize=True)
                full_t_1b = get_t()
                #>#full_adc_time_count += full_t_1b - full_t_1a
                #eris_ovoo_jak = eris_ovoo[kj,ka,kk] 
                #eris_ovoo_kaj = eris_ovoo[kk,ka,kj]
                #print(f'kj = {kj} | ka = {ka} | kk = {kk} ===> {np.linalg.norm(eris_ovoo_jak - eris_ovoo_jak_1)}')
                ######## TESTNG MASK_JK START ##############
                cvs_t_1a = get_t()
                1./nkpts * lib.einsum('Laj,Lik,jk->ajik', Lvo[ka,kj], Loo[ki,kk], mask_jk, optimize=True)
                1./nkpts * lib.einsum('Lak,Lij,jk->akij', Lvo[ka,kk], Loo[ki,kj], mask_jk, optimize=True)
                cvs_t_1b = get_t()
                #>#cvs_adc_time_count += cvs_t_1b - cvs_t_1a
                ######## TESTNG MASK_JK END ##############

                full_t_2a = get_t()
                s1 += 2. * lib.einsum('ajik,ajk->i',
                                      eris_vooo_aji, r2[ka,kj], optimize=True)
                s1 -= lib.einsum('akij,ajk->i',
                                 eris_vooo_aki, r2[ka,kj], optimize=True)
                full_t_2b = get_t()
                full_adc_time_count += full_t_2b - full_t_2a


                r1_c = r1[:ncvs].copy()
                eris_vooo_aji_co = eris_vooo_aji[:,:ncvs,:ncvs,:].copy()
                eris_vooo_aki_co = eris_vooo_aki[:,:,:ncvs,:ncvs].copy()
                eris_vooo_aji_vc = eris_vooo_aji[:,ncvs:,:ncvs,:ncvs].copy()
                eris_vooo_aki_vc = eris_vooo_aki[:,:ncvs,:ncvs,ncvs:].copy()
                r2_eco = r2[ka,kj,:,:ncvs,:].copy()
                r2_evc = r2[ka,kj,:,ncvs:,:ncvs].copy()
                r = np.hstack((r1_c,r2_eco,r2_evc))
                ######## TESTNG MASK_JK START ##############
                cvs_t_2a = get_t()
                lib.einsum('ajik,ajk->i',
                                      eris_vooo_aji_co, r2_eco, optimize=True)
                lib.einsum('akij,ajk->i',
                                 eris_vooo_aki_co, r2_eco, optimize=True)
                lib.einsum('ajik,ajk->i',
                                      eris_vooo_aji_vc, r2_evc, optimize=True)
                lib.einsum('akij,ajk->i',
                                 eris_vooo_aki_vc, r2_evc, optimize=True)
                cvs_t_2b = get_t()
                cvs_adc_time_count += cvs_t_2b - cvs_t_2a
                ######## TESTNG MASK_JK END ##############

                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo_jak.conj(), r2[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo_kaj.conj(), r2[ka,kj], optimize=True)



                #s1 += 2./nkpts * lib.einsum('Laj,Lik,ajk->i',
                #                      eris.Lvo[ka,kj], eris.Loo[ki,kk], r2[ka,kj], optimize=True)
                #s1 -= 1./nkpts * lib.einsum('Lak,Lij,ajk->i',
                #                 eris.Lvo[ka,kk], eris.Loo[ki,kj], r2[ka,kj], optimize=True)

                #path_1a =  np.einsum_path('Laj,Lik,ajk->i',
                #                      eris.Lvo[ka,kj], eris.Loo[ki,kk], r2[ka,kj], optimize=True)
                #path_1b = np.einsum_path('Lak,Lij,ajk->i',
                #                 eris.Lvo[ka,kk], eris.Loo[ki,kj], r2[ka,kj], optimize=True)
                #path_2a = np.einsum_path('jaki,ajk->i',
                #                      eris_ovoo_jak.conj(), r2[ka,kj], optimize=True)
                #path_2b = np.einsum_path('kaji,ajk->i',
                #                 eris_ovoo_kaj.conj(), r2[ka,kj], optimize=True)
                #print('======== PATH 1A =============')
                #print(path_1a[0])
                #print(path_1a[1])
                #print('======== PATH 1B =============')
                #print(path_1b[0])
                #print(path_1b[1])
                #print('======== PATH 2A =============')
                #print(path_2a[0])
                #print(path_2a[1])
                #print('======== PATH 2B =============')
                #print(path_2b[0])
                #print(path_2b[1])
                #exit()
                #s_temp[:ncvs] += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk,:ncvs,:,:ncvs,:ncvs].conj(), r2_ecc[ka,kj], optimize=True)
                #s_temp[:ncvs] -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj,:ncvs,:,:ncvs,:ncvs].conj(), r2_ecc[ka,kj], optimize=True)
                #s_temp[:ncvs] += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk,:ncvs,:,ncvs:,:ncvs].conj(), r2_ecv[ka,kj], optimize=True)
                #s_temp[:ncvs] -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj,ncvs:,:,:ncvs,:ncvs].conj(), r2_ecv[ka,kj], optimize=True)
                #s_temp[:ncvs] += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk,ncvs:,:,:ncvs,:ncvs].conj(), r2_evc[ka,kj], optimize=True)
                #s_temp[:ncvs] -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj,:ncvs,:,ncvs:,:ncvs].conj(), r2_evc[ka,kj], optimize=True)

                #s1 += s_temp
#################### ADC(2) ajk - i block ############################

                #s2[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk], r1, optimize=True)
                #s2[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo_jak, r1, optimize=True)
                #del eris_ovoo_jak
                #del eris_ovoo_kaj
                full_t_3a = get_t()
                s2[ka,kj] += lib.einsum('ajik,i->ajk', eris_vooo_aji.conj(), r1, optimize=True)
                full_t_3b = get_t()
                #full_adc_time_count += full_t_3b - full_t_3a
                ######## TESTNG MASK_JK START ##############
                cvs_t_3a = get_t()
                lib.einsum('ajik,i->ajk', eris_vooo_aji_co.conj(), r1_c, optimize=True)
                lib.einsum('ajik,i->ajk', eris_vooo_aji_vc.conj(), r1_c, optimize=True)
                cvs_t_3b = get_t()
                cvs_adc_time_count += cvs_t_3b - cvs_t_3a
                ######## TESTNG MASK_JK END ##############
                ##r1[2:] = 0
                #temp = np.zeros((nvir,nocc,nocc), dtype=np.complex128)
                #temp1 = lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk], r1, optimize=True)
                #temp[:,:ncvs,:ncvs] = temp1[:,:ncvs,:ncvs]
                #temp[:,:ncvs,ncvs:] = temp1[:,:ncvs,ncvs:]
                #temp[:,ncvs:,:ncvs] = temp1[:,ncvs:,:ncvs]
                #temp[:,:ncvs,:ncvs] = lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk,:ncvs,:,:ncvs,:ncvs], r1[:ncvs], optimize=True)
                #temp[:,:ncvs,ncvs:] = lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk,:ncvs,:,ncvs:,:ncvs], r1[:ncvs], optimize=True)
                #temp[:,ncvs:,:ncvs] = lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk,ncvs:,:,:ncvs,:ncvs], r1[:ncvs], optimize=True)
                #s2[ka,kj] += temp
                #s2[ka,kj] += temp1
                ##s1 *= 0 
                ####ncvs = 2
                ####vv_slice = slice(ncvs,nocc)
                ####eris_vooo_aji_t = eris_vooo_aji.copy()
                ####eris_vooo_aji_t[:,:,ncvs:,:] = None
                ####print('None array shape = ',np.shape(eris_vooo_aji_t != None))
                ####eris_vooo_aji_cvs = np.delete(eris_vooo_aji, np.where(eris_vooo_aji_t == None))
                ####print('shape of eris_vooo_aji_cvs', eris_vooo_aji_cvs.shape)
                ####print('shape of eris_vooo_aji', eris_vooo_aji.shape)
                ####s2t1 = lib.einsum('ajik,i->ajk', eris_vooo_aji.conj(), r1, optimize=True)
                ####s2t2 = lib.einsum('ajik,i->ajk', eris_vooo_aji_t.conj() != None, r1[:ncvs], optimize=True)
                ####print('expnsive = ', s2t1)
                ####print('cheap = ', s2t2)
################# ADC(2) ajk - bil block ############################

                full_t_4a = get_t()
                s2[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2[ka, kj])
                s2[ka, kj] += lib.einsum('j,ajk->ajk', e_occ[kj], r2[ka, kj])
                s2[ka, kj] += lib.einsum('k,ajk->ajk', e_occ[kk], r2[ka, kj])
                full_t_4b = get_t()
                #full_adc_time_count += full_t_4b - full_t_4a

                ######## TESTNG MASK_JK START ##############
                cvs_t_4a = get_t()
                lib.einsum('a,ajk,jk->ajk', e_vir[ka], r2[ka, kj], mask_jk)
                lib.einsum('j,ajk,jk->ajk', e_occ[kj], r2[ka, kj], mask_jk)
                lib.einsum('k,ajk,jk->ajk', e_occ[kk], r2[ka, kj], mask_jk)
                cvs_t_4b = get_t()
                #cvs_adc_time_count += cvs_t_4b - cvs_t_4a
                ######## TESTNG MASK_JK END ##############
############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    for kl in range(nkpts):
                        ki = kconserv[kj, kl, kk]

                        #eris_oooo_kij = eris_oooo[kk,ki,kj]
                        #eris_oooo_klj = eris_oooo[kk,kl,kj]
                        eris_oooo_kij = 1./nkpts * lib.einsum('Lki,Ljl->kijl'
                                       , eris.Loo[kk,ki],eris.Loo[kj,kl],optimize=True)
                        s2[ka,kj] -= lib.einsum('kijl,ali->ajk',
                                                    eris_oooo_kij, r2[ka,kl], optimize=True)
                        del eris_oooo_kij
                        #eris_oooo_klj = 1./nkpts * lib.einsum('Lkl,Lji->klji'
                        #               , eris.Loo[kk,kl],eris.Loo[kj,ki],optimize=True)                    
                        #s2[ka,kj] -= 0.5*lib.einsum('klji,ail->ajk',
                        #                            eris_oooo_klj,r2[ka,ki], optimize=True)
                        #del eris_oooo_klj

                        kb = kconserv[ka, kk, kl]
                        if kb <= ka:
                            idx_p = eris.Lvv_idx_p[(kb,ka)]
                            eris_oovv_klb = 1./nkpts * lib.einsum('Lkl,Lba->klba'
                                            , eris.Loo[kk,kl], eris.Lvv_p[idx_p], optimize=True)
                        if kb > ka:
                            idx_p = eris.Lvv_idx_p[(ka,kb)]
                            eris_oovv_klb = 1./nkpts * lib.einsum('Lkl,Lab->klba'
                                            , eris.Loo[kk,kl], eris.Lvv_p[idx_p].conj(), optimize=True)
                        #eris_oovv_klb = eris.oovv[kk,kl,kb]
                            
                        s2[ka,kj] += lib.einsum('klba,bjl->ajk',
                                                    eris_oovv_klb,r2[kb,kj],optimize=True)
                        del eris_oovv_klb

                        kb = kconserv[ka, kj, kl]
                        if kb <= ka:
                            idx_p = eris.Lvv_idx_p[(kb,ka)]
                            eris_oovv_jlb = 1./nkpts * lib.einsum('Ljl,Lba->jlba'
                                            , eris.Loo[kj,kl], eris.Lvv_p[idx_p], optimize=True)
                        if kb > ka:
                            idx_p = eris.Lvv_idx_p[(ka,kb)]
                            eris_oovv_jlb = 1./nkpts * lib.einsum('Ljl,Lab->jlba'
                                            , eris.Loo[kj,kl], eris.Lvv_p[idx_p].conj(), optimize=True)
                        #eris_oovv_jlb = eris.oovv[kj,kl,kb]
                        s2[ka,kj] +=  lib.einsum('jlba,blk->ajk',
                                                     eris_oovv_jlb,r2[kb,kl],optimize=True)
                        del eris_oovv_jlb

                        #eris_ovvo_jab = 1./nkpts * lib.einsum('Lja,Llb->jabl'
                        #                        , eris.Lov[kj,ka],eris.Lov[kl,kb].conj(),optimize=True)
                        #eris_ovvo_jab = eris.ovvo[kj,ka,kb]
                        s_tmp =  lib.einsum('Lbl,bkl->Lk',
                                                     eris.Lvo[kb,kl],r2[kb,kk],optimize=True)
                        s_tmp -= 2 * lib.einsum('Lbl,blk->Lk',
                                                     eris.Lvo[kb,kl],r2[kb,kl],optimize=True)
                        s2[ka,kj] += 1./nkpts * lib.einsum('Laj,Lk->ajk',
                                                     eris.Lvo[ka,kj].conj(),s_tmp,optimize=True)
                        #s2[ka,kj] +=  lib.einsum('jabl,bkl->ajk',
                        #                             eris_ovvo_jab,r2[kb,kk],optimize=True)
                        #s2[ka,kj] -=  2 * lib.einsum('jabl,blk->ajk',
                        #                         eris_ovvo_jab,r2[kb,kl],optimize=True)
                        #del eris_ovvo_jab
                        #for ki in range(nkpts):
                        #ki = kl
                        #kb = kconserv[ka, kk, ki]
                        #s2[ka,kj] += 0.5*lib.einsum('kiba,bji->ajk',
                        #                            eris_oovv[kk,ki,kb],r2[kb,kj],optimize=True)

                        #kb = kconserv[ka, kj, ki]
                        #s2[ka,kj] += 0.5*lib.einsum('jiba,bik->ajk',
                        #                            eris_oovv[kj,ki,kb],r2[kb,ki],optimize=True)
                        #s2[ka,kj] -= lib.einsum('jabi,bik->ajk',eris_ovvo[kj,
                        #                        ka,kb],r2[kb,ki],optimize=True)
                        #kb = kconserv[ki, kj, ka]
                        #s2[ka,kj] += 0.5*lib.einsum('jabi,bki->ajk',
                        #                            eris_ovvo[kj,ka,kb],r2[kb,kk],optimize=True)

            if adc.exxdiv is not None:
                s2 += madelung * r2

        if (method == "adc(3)"):

            eris_ovoo = eris.ovoo

################# ADC(3) i - kja block and ajk - i ############################

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kj,kshift,kk]

                    for kb in range(nkpts):
                        kc = kconserv[kj,kb,kk]
                        t2_1 = adc.t2[0]
                        temp_1 =       lib.einsum(
                            'jkbc,ajk->abc',t2_1[kj,kk,kb], r2[ka,kj], optimize=True)
                        temp  = 0.25 * lib.einsum('jkbc,ajk->abc',
                                                  t2_1[kj,kk,kb], r2[ka,kj], optimize=True)
                        temp -= 0.25 * lib.einsum('jkbc,akj->abc',
                                                  t2_1[kj,kk,kb], r2[ka,kk], optimize=True)
                        temp -= 0.25 * lib.einsum('kjbc,ajk->abc',
                                                  t2_1[kk,kj,kb], r2[ka,kj], optimize=True)
                        temp += 0.25 * lib.einsum('kjbc,akj->abc',
                                                  t2_1[kk,kj,kb], r2[ka,kk], optimize=True)
                        ki = kconserv[kc,ka,kb]
                        if isinstance(eris.ovvv, type(None)):
                            chnk_size = adc.chnk_size
                            if chnk_size > nocc:
                                chnk_size = nocc
                            a = 0
                            for p in range(0,nocc,chnk_size):
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[ki,kc], eris.Lvv[ka,kb], p,
                                    chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                k = eris_ovvv.shape[0]
                                s1[a:a+k] += lib.einsum('abc,icab->i',temp_1,
                                                        eris_ovvv, optimize=True)
                                s1[a:a+k] += lib.einsum('abc,icab->i',temp,
                                                        eris_ovvv, optimize=True)
                                del eris_ovvv
                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[ki,kb], eris.Lvv[ka,kc], p,
                                    chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                s1[a:a+k] -= lib.einsum('abc,ibac->i',temp,
                                                        eris_ovvv, optimize=True)
                                del eris_ovvv
                                a += k
                        else :
                            eris_ovvv = eris.ovvv[:]
                            s1 += lib.einsum('abc,icab->i',temp_1,
                                             eris_ovvv[ki,kc,ka], optimize=True)
                            s1 += lib.einsum('abc,icab->i',temp,
                                             eris_ovvv[ki,kc,ka], optimize=True)
                            s1 -= lib.einsum('abc,ibac->i',temp,
                                             eris_ovvv[ki,kb,ka], optimize=True)
                            del eris_ovvv
            del temp
            del temp_1

            t2_1 = adc.t2[0]

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kj, kshift, kk]
                    for kc in range(nkpts):
                        kb = kconserv[kj, kc, kk]
                        ki = kconserv[kb,ka,kc]
                        if isinstance(eris.ovvv, type(None)):
                            chnk_size = adc.chnk_size
                            if chnk_size > nocc:
                                chnk_size = nocc
                            a = 0
                            for p in range(0,nocc,chnk_size):

                                eris_ovvv = dfadc.get_ovvv_df(
                                    adc, eris.Lov[ki,kc], eris.Lvv[ka,kb], p,
                                    chnk_size).reshape(-1,nvir,nvir,nvir)/nkpts
                                k = eris_ovvv.shape[0]
                                temp = lib.einsum(
                                    'i,icab->cba',r1[a:a+k],eris_ovvv.conj(), optimize=True)
                                del eris_ovvv
                                a += k
                        else :
                            eris_ovvv = eris.ovvv[:]
                            temp = lib.einsum(
                                'i,icab->cba',r1,eris_ovvv[ki,kc,ka].conj(),optimize=True)
                            del eris_ovvv
                        s2[ka,kj] += lib.einsum('cba,jkbc->ajk',temp,
                                                t2_1[kj,kk,kb].conj(), optimize=True)
            del temp

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kj, kshift, kk]
                    for kb in range(nkpts):
                        kl = kconserv[ka, kj, kb]

                        t2_1 = adc.t2[0]
                        temp = lib.einsum('ljba,ajk->blk',t2_1[kl,kj,kb],r2[ka,kj],optimize=True)
                        temp_2 = lib.einsum('jlba,akj->blk',t2_1[kj,kl,kb],r2[ka,kk], optimize=True)
                        del t2_1

                        t2_1_jla = adc.t2[0][kj,kl,ka]
                        temp += lib.einsum('jlab,ajk->blk',t2_1_jla,r2[ka,kj],optimize=True)
                        temp -= lib.einsum('jlab,akj->blk',t2_1_jla,r2[ka,kk],optimize=True)

                        temp_1  = lib.einsum('jlab,ajk->blk',t2_1_jla,r2[ka,kj],optimize=True)
                        temp_1 -= lib.einsum('jlab,akj->blk',t2_1_jla,r2[ka,kk],optimize=True)
                        temp_1 += lib.einsum('jlab,ajk->blk',t2_1_jla,r2[ka,kj],optimize=True)
                        del t2_1_jla

                        t2_1_lja = adc.t2[0][kl,kj,ka]
                        temp -= lib.einsum('ljab,ajk->blk',t2_1_lja,r2[ka,kj],optimize=True)
                        temp += lib.einsum('ljab,akj->blk',t2_1_lja,r2[ka,kk],optimize=True)

                        temp_1 -= lib.einsum('ljab,ajk->blk',t2_1_lja,r2[ka,kj],optimize=True)
                        del t2_1_lja

                        ki = kconserv[kk, kl, kb]
                        s1 += 0.5*lib.einsum('blk,lbik->i',temp,  eris_ovoo[kl,kb,ki],optimize=True)
                        s1 -= 0.5*lib.einsum('blk,iblk->i',temp,  eris_ovoo[ki,kb,kl],optimize=True)
                        s1 += 0.5*lib.einsum('blk,lbik->i',temp_1,eris_ovoo[kl,kb,ki],optimize=True)
                        s1 -= 0.5*lib.einsum('blk,iblk->i',temp_2,eris_ovoo[ki,kb,kl],optimize=True)
                        del temp
                        del temp_1
                        del temp_2

                    for kb in range(nkpts):
                        kl = kconserv[ka, kk, kb]

                        t2_1 = adc.t2[0]

                        temp = -lib.einsum('lkba,akj->blj',t2_1[kl,kk,kb],r2[ka,kk],optimize=True)
                        temp_2 = -lib.einsum('klba,ajk->blj',t2_1[kk,kl,kb],r2[ka,kj],optimize=True)
                        del t2_1

                        t2_1_kla = adc.t2[0][kk,kl,ka]
                        temp -= lib.einsum('klab,akj->blj',t2_1_kla,r2[ka,kk],optimize=True)
                        temp += lib.einsum('klab,ajk->blj',t2_1_kla,r2[ka,kj],optimize=True)
                        temp_1  = -2.0 * lib.einsum('klab,akj->blj',
                                                    t2_1_kla,r2[ka,kk],optimize=True)
                        temp_1 += lib.einsum('klab,ajk->blj',t2_1_kla,r2[ka,kj],optimize=True)
                        del t2_1_kla

                        t2_1_lka = adc.t2[0][kl,kk,ka]
                        temp += lib.einsum('lkab,akj->blj',t2_1_lka,r2[ka,kk],optimize=True)
                        temp -= lib.einsum('lkab,ajk->blj',t2_1_lka,r2[ka,kj],optimize=True)
                        temp_1 += lib.einsum('lkab,akj->blj',t2_1_lka,r2[ka,kk],optimize=True)
                        del t2_1_lka

                        ki = kconserv[kj, kl, kb]
                        s1 -= 0.5*lib.einsum('blj,lbij->i',temp,  eris_ovoo[kl,kb,ki],optimize=True)
                        s1 += 0.5*lib.einsum('blj,iblj->i',temp,  eris_ovoo[ki,kb,kl],optimize=True)
                        s1 -= 0.5*lib.einsum('blj,lbij->i',temp_1,eris_ovoo[kl,kb,ki],optimize=True)
                        s1 += 0.5*lib.einsum('blj,iblj->i',temp_2,eris_ovoo[ki,kb,kl],optimize=True)

                        del temp
                        del temp_1
                        del temp_2

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    for kl in range(nkpts):
                        kb = kconserv[kj, ka, kl]
                        ki = kconserv[kk,kl,kb]
                        temp_1 = lib.einsum(
                            'i,lbik->kbl',r1,eris_ovoo[kl,kb,ki].conj(), optimize=True)
                        temp  = lib.einsum(
                            'i,lbik->kbl',r1,eris_ovoo[kl,kb,ki].conj(), optimize=True)
                        temp -= lib.einsum('i,iblk->kbl',r1,
                                           eris_ovoo[ki,kb,kl].conj(), optimize=True)

                        t2_1 = adc.t2[0]
                        s2[ka,kj] += lib.einsum('kbl,ljba->ajk',temp,
                                                t2_1[kl,kj,kb].conj(), optimize=True)
                        s2[ka,kj] += lib.einsum('kbl,jlab->ajk',temp_1,
                                                t2_1[kj,kl,ka].conj(), optimize=True)
                        s2[ka,kj] -= lib.einsum('kbl,ljab->ajk',temp_1,
                                                t2_1[kl,kj,ka].conj(), optimize=True)

                        kb = kconserv[kk, ka, kl]
                        ki = kconserv[kj,kl,kb]
                        temp_2 = -lib.einsum('i,iblj->jbl',r1,
                                             eris_ovoo[ki,kb,kl].conj(), optimize=True)
                        s2[ka,kj] += lib.einsum('jbl,klba->ajk',temp_2,
                                                t2_1[kk,kl,kb].conj(), optimize=True)
                        del t2_1
        s2 = s2.reshape(-1)
        s = np.hstack((s1,s2))
        del s1
        del s2
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        print(f'full_adc_time_count = {full_adc_time_count}')        
        print(f'cvs_adc_time_count = {cvs_adc_time_count}')       
        if adc.ncvs_proj is not None:
            s = cvs_projector(adc, s)

        #s = np.ndarray.astype(s, dtype=np.complex64)
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

    idn_occ = np.identity(nocc)

    T1 = np.zeros((nocc), dtype=np.complex128)
    T2 = np.zeros((nkpts,nkpts,nvir,nocc,nocc), dtype=np.complex128)

######## ADC(2) 1h part  ############################################

    if orb < nocc:
        T1 += idn_occ[orb, :]
        for kk in range(nkpts):
            for kc in range(nkpts):
                kd = adc.khelper.kconserv[kk, kc, kshift]
                ki = adc.khelper.kconserv[kc, kk, kd]
                T1 += 0.25*lib.einsum('kdc,ikdc->i',t2_1[kk,kshift,kd]
                                      [:,orb,:,:].conj(), t2_1[ki,kk,kd], optimize=True)
                T1 -= 0.25*lib.einsum('kcd,ikdc->i',t2_1[kk,kshift,kc]
                                      [:,orb,:,:].conj(), t2_1[ki,kk,kd], optimize=True)
                T1 -= 0.25*lib.einsum('kdc,ikcd->i',t2_1[kk,kshift,kd]
                                      [:,orb,:,:].conj(), t2_1[ki,kk,kc], optimize=True)
                T1 += 0.25*lib.einsum('kcd,ikcd->i',t2_1[kk,kshift,kc]
                                      [:,orb,:,:].conj(), t2_1[ki,kk,kc], optimize=True)
                T1 -= 0.25*lib.einsum('kdc,ikdc->i',t2_1[kshift,kk,kd]
                                      [orb,:,:,:].conj(), t2_1[ki,kk,kd], optimize=True)
                T1 -= 0.25*lib.einsum('kcd,ikcd->i',t2_1[kshift,kk,kc]
                                      [orb,:,:,:].conj(), t2_1[ki,kk,kc], optimize=True)
    else :
        if (adc.approx_trans_moments is False or adc.method == "adc(3)"):
            t1_2 = adc.t1[0]
            T1 += t1_2[kshift][:,(orb-nocc)]

######## ADC(2) 2h-1p  part  ############################################

        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = adc.khelper.kconserv[kj, kshift, ki]

                t2_1_t = -t2_1[ki,kj,ka].transpose(2,3,1,0)
                T2[ka,kj] += t2_1_t[:,(orb-nocc),:,:].conj()

        del t2_1_t
####### ADC(3) 2h-1p  part  ############################################

    if (adc.method == "adc(2)-x" and adc.approx_trans_moments is False) or (adc.method == "adc(3)"):

        t2_2 = adc.t2[1]

        if orb >= nocc:
            for ki in range(nkpts):
                for kj in range(nkpts):
                    ka = adc.khelper.kconserv[kj, kshift, ki]

                    t2_2_t = -t2_2[ki,kj,ka].transpose(2,3,1,0)
                    T2[ka,kj] += t2_2_t[:,(orb-nocc),:,:].conj()


######### ADC(3) 1h part  ############################################

    if(method=='adc(3)'):
        if orb < nocc:
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = adc.khelper.kconserv[kk, kc, kshift]
                    ki = adc.khelper.kconserv[kd, kk, kc]
                    T1 += 0.25* \
                        lib.einsum('kdc,ikdc->i',t2_1[kk,ki,kd][:,orb,
                                   :,:].conj(), t2_2[ki,kk,kd], optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('kcd,ikdc->i',t2_1[kk,ki,kc][:,orb,
                                   :,:].conj(), t2_2[ki,kk,kd], optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('kdc,ikcd->i',t2_1[kk,ki,kd][:,orb,
                                   :,:].conj(), t2_2[ki,kk,kc], optimize=True)
                    T1 += 0.25* \
                        lib.einsum('kcd,ikcd->i',t2_1[kk,ki,kc][:,orb,
                                   :,:].conj(), t2_2[ki,kk,kc], optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('kdc,ikdc->i',t2_1[ki,kk,kd][orb,:,
                                   :,:].conj(), t2_2[ki,kk,kd], optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('kcd,ikcd->i',t2_1[ki,kk,kc][orb,:,
                                   :,:].conj(), t2_2[ki,kk,kc], optimize=True)

                    T1 += 0.25* \
                        lib.einsum('ikdc,kdc->i',t2_1[ki,kk,kd],
                                   t2_2[kk,ki,kd][:,orb,:,:].conj(),optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('ikcd,kdc->i',t2_1[ki,kk,kc],
                                   t2_2[kk,ki,kd][:,orb,:,:].conj(),optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('ikdc,kcd->i',t2_1[ki,kk,kd],
                                   t2_2[kk,ki,kc][:,orb,:,:].conj(),optimize=True)
                    T1 += 0.25* \
                        lib.einsum('ikcd,kcd->i',t2_1[ki,kk,kc],
                                   t2_2[kk,ki,kc][:,orb,:,:].conj(),optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('ikcd,kcd->i',t2_1[ki,kk,kc],
                                   t2_2[ki,kk,kc][orb,:,:,:].conj(),optimize=True)
                    T1 -= 0.25* \
                        lib.einsum('ikdc,kdc->i',t2_1[ki,kk,kd],
                                   t2_2[ki,kk,kd][orb,:,:,:].conj(),optimize=True)
        else:

            for kk in range(nkpts):
                for kc in range(nkpts):
                    ki = adc.khelper.kconserv[kshift,kk,kc]

                    T1 += 0.5 * lib.einsum('kic,kc->i',
                                           t2_1[kk,ki,kc][:,:,:,(orb-nocc)], t1_2[kk],optimize=True)
                    T1 -= 0.5*lib.einsum('ikc,kc->i',t2_1[ki,kk,kc]
                                         [:,:,:,(orb-nocc)], t1_2[kk],optimize=True)
                    T1 += 0.5*lib.einsum('kic,kc->i',t2_1[kk,ki,kc]
                                         [:,:,:,(orb-nocc)], t1_2[kk],optimize=True)

        del t2_2
    del t2_1

    for ki in range(nkpts):
        for kj in range(nkpts):
            ka = adc.khelper.kconserv[kj,kshift, ki]
            T2[ka,kj] += T2[ka,kj] - T2[ka,ki].transpose(0,2,1)

    T2 = T2.reshape(-1)
    T = np.hstack((T1,T2))

    return T


def renormalize_eigenvectors(adc, kshift, U, nroots=1):

    nkpts = adc.nkpts
    nocc = adc.t2[0].shape[3]
    n_singles = nocc
    nvir = adc.nmo - adc.nocc

    for I in range(U.shape[1]):
        U1 = U[:n_singles,I]
        U2 = U[n_singles:,I].reshape(nkpts,nkpts,nvir,nocc,nocc)
        UdotU = np.dot(U1.conj().ravel(),U1.ravel())
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = adc.khelper.kconserv[kj, kshift, kk]
                UdotU +=  2.*np.dot(U2[ka,kj].conj().ravel(), U2[ka,kj].ravel()) - \
                                    np.dot(U2[ka,kj].conj().ravel(),
                                           U2[ka,kk].transpose(0,2,1).ravel())
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


class RADCIP(kadc_rhf.RADC):
    '''restricted ADC for IP energies and spectroscopic amplitudes

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

        e_ip : float or list of floats
            IP energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list
            of floats for the lowest nroots eigenvalues.
        v_ip : array
            Eigenvectors for each IP transition.
        p_ip : float
            Spectroscopic amplitudes for each IP transition.
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

        self.ncvs_proj = adc.ncvs_proj
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
        self.ncvs_proj = adc.ncvs_proj

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
        #dtype = getattr(diag, 'dtype', np.complex64)
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
        #return diag
