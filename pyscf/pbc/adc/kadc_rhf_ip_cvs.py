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
import multiprocessing

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
from pyscf.pbc.adc.kadc_rhf_amplitudes import gen_t2_1
import tracemalloc 

def vector_size(adc):

    nkpts = adc.nkpts
    nocc = adc.nocc
    ncvs = adc.ncvs
    nval = nocc - ncvs
    nvir = adc.nmo - nocc

    n_singles = ncvs
    n_doubles_ecc = nkpts * nkpts * nvir * ncvs * ncvs
    n_doubles_ecv = nkpts * nkpts * nvir * ncvs * nval
    size = n_singles + n_doubles_ecc + 2*n_doubles_ecv

    return size


def get_imds(adc, eris=None):

    tracemalloc.start()
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
    ncvs =  adc.ncvs

    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    #e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_cvs = [mo_energy[k][:ncvs] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    #e_occ = np.array(e_occ)
    e_cvs = np.array(e_cvs)
    e_vir = np.array(e_vir)

    #idn_occ = np.identity(nocc)
    idn_cvs = np.identity(ncvs)
    #M_ij = np.empty((nkpts,nocc,nocc),dtype=mo_coeff.dtype)

    if eris is None:
        eris = adc.transform_integrals()

    M_ij = np.empty((nkpts,ncvs,ncvs),dtype=eris.Loo.dtype)
    # i-j block
    # Zeroth-order terms
    print('running an efficient implementation of CVS')
    #t2_1 = adc.t2[0]
    #eris_ovov = eris.ovov
    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_cvs , e_cvs[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                #t2_1 = adc.t2[0]
                if not adc.eris_direct:
                    t2_1_jld = t2_1_ild = adc.t2[0][ki,kl,kd,:ncvs]
                    eris_ovov_jdl = eris_ovov_idl = eris.ovov[ki,kd,kl,:ncvs]
                    eris_ovov_jel = eris_ovov_iel = eris.ovov[ki,ke,kl,:ncvs]
                else:
                    Lce_id = eris.Lov[ki,kd,:,:ncvs,:].copy()
                    Lce_ie = eris.Lov[ki,ke,:,:ncvs,:].copy()
                    eris_ovov_jdl = eris_ovov_idl = 1./nkpts * lib.einsum('Ljd,Lle->jdle'
                                    , Lce_id, eris.Lov[kl,ke], optimize=True)
                    eris_ovov_jel = eris_ovov_iel = 1./nkpts * lib.einsum('Lje,Lld->jeld'
                                    , Lce_ie, eris.Lov[kl,kd], optimize=True)
                    t2_1_jld = t2_1_ild = gen_t2_1(adc,eris,(ki,kl,kd,ke), cvs_idx_slice='i',ncvs=ncvs)#[:ncvs]

                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov_jdl,optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ilde,jeld->ij',t2_1_ild, eris_ovov_jel,optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ild,
                                             eris_ovov_jdl,optimize=True)
                del t2_1_ild

                #t2_1_lid = adc.t2[0][kl,ki,kd][:,:ncvs].copy()
                if not adc.eris_direct:
                    t2_1_ljd = t2_1_lid = adc.t2[0][kl,ki,kd][:,:ncvs].copy()
                else:
                    t2_1_ljd = t2_1_lid = gen_t2_1(adc,eris,(kl,ki,kd,ke), cvs_idx_slice='j',ncvs=ncvs)#[:,:ncvs]
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('lide,jdle->ij',t2_1_lid, eris_ovov_jdl,optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('lide,jeld->ij',t2_1_lid, eris_ovov_jel,optimize=True)
                #del t2_1_lid

                #t2_1_jld = adc.t2[0][kj,kl,kd][:ncvs].copy()
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('jlde,idle->ij',t2_1_jld.conj(),
                               eris_ovov_idl.conj(),optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('jlde,ield->ij',t2_1_jld.conj(),
                               eris_ovov_iel.conj(),optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('jlde,idle->ij',t2_1_jld.conj(),
                                             eris_ovov_idl.conj(),optimize=True)
                #del t2_1_jld

                #t2_1_ljd = adc.t2[0][kl,kj,kd][:,:ncvs].copy()
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ljde,idle->ij',t2_1_ljd.conj(),
                               eris_ovov_idl.conj(),optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ljde,ield->ij',t2_1_ljd.conj(),
                               eris_ovov_iel.conj(),optimize=True)
                del t2_1_ljd
                #del t2_1

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

    #for nkpts_i in range(nkpts):
    #    e_ij, _ = np.linalg.eig(M_ij[nkpts_i,:adc.ncvs,:adc.ncvs])
    #    print(f'e_ij_{nkpts_i} = {e_ij}')
    #exit()
    print(f'[memalloc current+max imds [GB] = {np.array(tracemalloc.get_traced_memory())/1024**3}')
    tracemalloc.stop()
    return M_ij

def get_diag(adc,kshift,M_ij=None,eris=None):

    log = logger.Logger(adc.stdout, adc.verbose)

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    if M_ij is None:
        M_ij = adc.get_imds()

    nkpts = adc.nkpts
    kconserv = adc.khelper.kconserv
    nocc = adc.nocc
    ncvs = adc.ncvs
    nval = nocc - ncvs
    n_singles = ncvs
    nvir = adc.nmo - nocc
    n_doubles_ecc = nkpts * nkpts * nvir * ncvs * ncvs
    n_doubles_ecv = nkpts * nkpts * nvir * ncvs * nval

    dim = n_singles + n_doubles_ecc + 2 * n_doubles_ecv
     
    s1 = 0
    f1 = n_singles
    s2_ecc = f1
    f2_ecc = s2_ecc + n_doubles_ecc
    s2_ecv = f2_ecc
    f2_ecv = s2_ecv + n_doubles_ecv
    s2_evc = f2_ecv
    f2_evc = s2_evc + n_doubles_ecv

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    nocc = adc.nocc
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_cvs = [e_occ[k][:ncvs] for k in range(nkpts)]
    e_val = [e_occ[k][ncvs:] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_occ = np.array(e_occ)
    e_cvs = np.array(e_cvs)
    e_val = np.array(e_val)
    e_vir = np.array(e_vir)

    diag = np.zeros((dim), dtype=np.complex128)
    doubles_ecc = np.zeros((nkpts,nkpts,nvir*ncvs*ncvs),dtype=np.complex128)
    doubles_ecv = np.zeros((nkpts,nkpts,nvir*ncvs*nval),dtype=np.complex128)
    doubles_eco = np.zeros((nkpts,nkpts,nvir*ncvs*nocc),dtype=np.complex128)
    doubles_evc = np.zeros((nkpts,nkpts,nvir*nval*ncvs),dtype=np.complex128)

    # Compute precond in h1-h1 block
    M_ij_diag = np.diagonal(M_ij[kshift])
    diag[s1:f1] = M_ij_diag.copy()

    # Compute precond in 2p1h-2p1h block

    for ka in range(nkpts):
        for ki in range(nkpts):
            kj = kconserv[kshift,ki,ka]
            #d_ij = e_occ[ki][:,None] + e_occ[kj]
            if not adc.cvs_compact:
                d_ij_cc = e_cvs[ki][:,None] + e_cvs[kj]
                d_ij_cv = e_cvs[ki][:,None] + e_val[kj]
            else:
                d_ij_co = e_cvs[ki][:,None] + e_occ[kj]
            d_ij_vc = e_val[ki][:,None] + e_cvs[kj]
            d_a = e_vir[ka][:,None]
            #D_n = -d_a + d_ij.reshape(-1)
            if not adc.cvs_compact:
                D_n_ecc = -d_a + d_ij_cc.reshape(-1)
                D_n_ecv = -d_a + d_ij_cv.reshape(-1)
            else:
                D_n_eco = -d_a + d_ij_co.reshape(-1)
            D_n_evc = -d_a + d_ij_vc.reshape(-1)
            if not adc.cvs_compact:
                doubles_ecc[ka,ki] += D_n_ecc.reshape(-1)
                doubles_ecv[ka,ki] += D_n_ecv.reshape(-1)
            else:
                doubles_eco[ka,ki] += D_n_eco.reshape(-1)
            doubles_evc[ka,ki] += D_n_evc.reshape(-1)

    if not adc.cvs_compact:
        diag[s2_ecc:f2_ecc] = doubles_ecc.reshape(-1)
        diag[s2_ecv:f2_ecv] = doubles_ecv.reshape(-1)
    else:
        diag[s2_ecc:f2_ecv] = doubles_eco.reshape(-1)
    diag[s2_evc:f2_evc] = doubles_evc.reshape(-1)

    diag = -diag
    log.timer_debug1("Completed ea_diag calculation")

    return diag

def matvec(adc, kshift, M_ij=None, eris=None):

    tracemalloc.start()
    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    ncvs = adc.ncvs
    nval = nocc - ncvs
    kconserv = adc.khelper.kconserv
    n_singles = ncvs
    nvir = adc.nmo - nocc
    n_doubles_ecc = nkpts * nkpts * nvir * ncvs * ncvs
    n_doubles_ecv = nkpts * nkpts * nvir * ncvs * nval

    s_singles = 0
    f_singles = n_singles
    s_doubles_ecc = f_singles
    f_doubles_ecc = s_doubles_ecc + n_doubles_ecc
    s_doubles_ecv = f_doubles_ecc
    f_doubles_ecv = s_doubles_ecv + n_doubles_ecv
    s_doubles_evc = f_doubles_ecv
    f_doubles_evc = s_doubles_evc + n_doubles_ecv

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_cvs = [e_occ[k][:ncvs] for k in range(nkpts)]
    e_val = [e_occ[k][ncvs:] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    if M_ij is None:
        M_ij = adc.get_imds()

    print('[MATVEC] running an efficient implementation of CVS')
    #Calculate sigma vector
    def sigma_(r):
        #cput0 = (time.process_time(), time.time())
        cput0 = (time.process_time(), time.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)

        dtype = eris.Loo.dtype 
        r = np.ndarray.astype(r, dtype=dtype)
         
        r1 = r[s_singles:f_singles]

        if not adc.cvs_compact:
            r2_ecc = r[s_doubles_ecc:f_doubles_ecc]
            r2_ecv = r[s_doubles_ecv:f_doubles_ecv]
        else:
            r2_eco = r[s_doubles_ecc:f_doubles_ecv]
        r2_evc = r[s_doubles_evc:f_doubles_evc]

        if not adc.cvs_compact:
            r2_ecc = r2_ecc.reshape(nkpts,nkpts,nvir,ncvs,ncvs)
            r2_ecv = r2_ecv.reshape(nkpts,nkpts,nvir,ncvs,nval)
        else: 
            r2_eco = r2_eco.reshape(nkpts,nkpts,nvir,ncvs,nocc)
        r2_evc = r2_evc.reshape(nkpts,nkpts,nvir,nval,ncvs)

        if not adc.cvs_compact:
            s2_ecc = np.zeros((nkpts,nkpts,nvir,ncvs,ncvs), dtype=dtype)
            s2_ecv = np.zeros((nkpts,nkpts,nvir,ncvs,nval), dtype=dtype)
        else:
            s2_eco = np.zeros((nkpts,nkpts,nvir,ncvs,nocc), dtype=dtype)
        s2_evc = np.zeros((nkpts,nkpts,nvir,nval,ncvs), dtype=dtype)

        cell = adc.cell
        kpts = adc.kpts
        madelung = tools.madelung(cell, kpts)

############ ADC(2) ij block ############################

        s1 = lib.einsum('ij,j->i',M_ij[kshift],r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]
                ki = kconserv[kj, kk, ka]

                if adc.eris_direct:
                    eris_eooo_aji = 1./nkpts * lib.einsum('Laj,Lik->ajik'
                                      , eris.Lvo[ka,kj], eris.Loo[ki,kk], optimize=True)
                    eris_eooo_aki = 1./nkpts * lib.einsum('Lak,Lij->akij'
                                      , eris.Lvo[ka,kk], eris.Loo[ki,kj], optimize=True)
                else:
                    eris_eooo_aji = eris.vooo[ka,kj,ki]
                    eris_eooo_aki = eris.vooo[ka,kk,ki]

                if not adc.cvs_compact: 
                    eris_eccc_aji = eris_eooo_aji[:,:ncvs,:ncvs,:ncvs].copy() 
                    eris_eccc_aki = eris_eooo_aki[:,:ncvs,:ncvs,:ncvs].copy()
                    eris_eccv_aji = eris_eooo_aji[:,:ncvs,:ncvs,ncvs:].copy()
                    eris_evcc_aki = eris_eooo_aki[:,ncvs:,:ncvs,:ncvs].copy()

                else:
                    eris_ecco_aji = eris_eooo_aji[:,:ncvs,:ncvs,:].copy() 
                    eris_eocc_aki = eris_eooo_aki[:,:,:ncvs,:ncvs].copy()

                eris_evcc_aji = eris_eooo_aji[:,ncvs:,:ncvs,:ncvs].copy() 
                eris_eccv_aki = eris_eooo_aki[:,:ncvs,:ncvs,ncvs:].copy()

                if not adc.cvs_compact:
                    s1 += 2. * lib.einsum('aJIK,aJK->I',
                                          eris_eccc_aji, r2_ecc[ka,kj], optimize=True)
                    s1 -= lib.einsum('aKIJ,aJK->I',
                                     eris_eccc_aki, r2_ecc[ka,kj], optimize=True)
                    s1 += 2. * lib.einsum('aJIk,aJk->I',
                                          eris_eccv_aji, r2_ecv[ka,kj], optimize=True)
                    s1 -= lib.einsum('akIJ,aJk->I',
                                     eris_evcc_aki, r2_ecv[ka,kj], optimize=True)
                else:
                    s1 += 2. * lib.einsum('aJIk,aJk->I',
                                          eris_ecco_aji, r2_eco[ka,kj], optimize=True)
                    s1 -= lib.einsum('akIJ,aJk->I',
                                     eris_eocc_aki, r2_eco[ka,kj], optimize=True)
                s1 += 2. * lib.einsum('ajIK,ajK->I',
                                      eris_evcc_aji, r2_evc[ka,kj], optimize=True)
                s1 -= lib.einsum('aKIj,ajK->I',
                                 eris_eccv_aki, r2_evc[ka,kj], optimize=True)

#################### ADC(2) ajk - i block ############################

                if not adc.cvs_compact:
                    s2_ecc[ka,kj] += lib.einsum('aJIK,I->aJK', eris_eccc_aji.conj(), r1, optimize=True)
                    s2_ecv[ka,kj] += lib.einsum('aJIk,I->aJk', eris_eccv_aji.conj(), r1, optimize=True)
                else:
                    s2_eco[ka,kj] += lib.einsum('aJIk,I->aJk', eris_ecco_aji.conj(), r1, optimize=True)
                s2_evc[ka,kj] += lib.einsum('ajIK,I->ajK', eris_evcc_aji.conj(), r1, optimize=True)

################# ADC(2) ajk - bil block ############################

                if not adc.cvs_compact:
                    s2_ecc[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_ecc[ka, kj])
                    s2_ecc[ka, kj] += lib.einsum('j,ajk->ajk', e_cvs[kj], r2_ecc[ka, kj])
                    s2_ecc[ka, kj] += lib.einsum('k,ajk->ajk', e_cvs[kk], r2_ecc[ka, kj])
                    s2_ecv[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_ecv[ka, kj])
                    s2_ecv[ka, kj] += lib.einsum('j,ajk->ajk', e_cvs[kj], r2_ecv[ka, kj])
                    s2_ecv[ka, kj] += lib.einsum('k,ajk->ajk', e_val[kk], r2_ecv[ka, kj])
                else:
                    s2_eco[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_eco[ka, kj])
                    s2_eco[ka, kj] += lib.einsum('j,ajk->ajk', e_cvs[kj], r2_eco[ka, kj])
                    s2_eco[ka, kj] += lib.einsum('k,ajk->ajk', e_occ[kk], r2_eco[ka, kj])

                s2_evc[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_evc[ka, kj])
                s2_evc[ka, kj] += lib.einsum('j,ajk->ajk', e_val[kj], r2_evc[ka, kj])
                s2_evc[ka, kj] += lib.einsum('k,ajk->ajk', e_cvs[kk], r2_evc[ka, kj])
############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            if not adc.eris_direct:
                eris_oooo = eris.oooo
                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo

            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    for kl in range(nkpts):
                        ki = kconserv[kj, kl, kk]
                        
                        if not adc.cvs_compact:

                            if adc.eris_direct:  
                                eris_oooo_kij = 1./nkpts * lib.einsum('Lki,Ljl->kijl'
                                           , eris.Loo[kk,ki],eris.Loo[kj,kl],optimize=True)
                                eris_oooo_jik = 1./nkpts * lib.einsum('Lji,Lkl->jikl'
                                           , eris.Loo[kj,ki],eris.Loo[kk,kl],optimize=True)
                            else:
                                eris_oooo_kij = eris_oooo[kk,ki,kj]
                                eris_oooo_jik = eris_oooo[kj,ki,kk]

                            eris_cccv_kij = eris_oooo_kij[:ncvs,:ncvs,:ncvs,ncvs:].copy()   
                            eris_vccv_kij = eris_oooo_kij[ncvs:,:ncvs,:ncvs,ncvs:].copy()
                            eris_ccvv_kij = eris_oooo_kij[:ncvs,:ncvs,ncvs:,ncvs:].copy()
                            eris_cccv_jik = eris_oooo_jik[:ncvs,:ncvs,:ncvs,ncvs:].copy()
                            eris_cccc_jik = eris_oooo_jik[:ncvs,:ncvs,:ncvs,:ncvs].copy()
                            eris_ccvv_jik = eris_oooo_jik[:ncvs,:ncvs,ncvs:,ncvs:].copy()
                            eris_ccvc_jik = eris_oooo_jik[:ncvs,:ncvs,ncvs:,:ncvs].copy()
                            eris_vccc_jik = eris_oooo_jik[ncvs:,:ncvs,:ncvs,:ncvs].copy()
                            eris_vccv_jik = eris_oooo_jik[ncvs:,:ncvs,:ncvs,ncvs:].copy()
 
                            s2_ecc[ka,kj] -= lib.einsum('KIJl,alI->aJK',eris_cccv_kij,
                                                  r2_evc[ka,kl],optimize=True)
                            s2_ecc[ka,kj] -= lib.einsum('JIKl,aIl->aJK',eris_cccv_jik,
                                                  r2_ecv[ka,ki],optimize=True)
                            s2_ecc[ka,kj] -= lib.einsum('JIKL,aIL->aJK',eris_cccc_jik,
                                                           r2_ecc[ka,ki],optimize=True)
                            s2_ecv[ka,kj] -= lib.einsum('JIkl,aIl->aJk',eris_ccvv_jik,
                                                           r2_ecv[ka,ki],optimize=True)
                            s2_ecv[ka,kj] -= lib.einsum('kIJl,alI->aJk',eris_vccv_kij,
                                                  r2_evc[ka,kl],optimize=True)
                            s2_ecv[ka,kj] -= lib.einsum('JIkL,aIL->aJk',eris_ccvc_jik,
                                                  r2_ecc[ka,ki],optimize=True)
                            s2_evc[ka,kj] -= lib.einsum('KIjl,alI->ajK',eris_ccvv_kij,
                                                           r2_evc[ka,kl],optimize=True)
                            s2_evc[ka,kj] -= lib.einsum('jIKl,aIl->ajK',eris_vccv_jik,
                                                           r2_ecv[ka,ki],optimize=True)
                            s2_evc[ka,kj] -= lib.einsum('jIKL,aIL->ajK',eris_vccc_jik,
                                                           r2_ecc[ka,ki],optimize=True)

                            kb = kconserv[ka, kk, kl]
                            if adc.eris_direct: 
 
                                if kb <= ka:
                                    idx_p = eris.Lvv_idx_p[(kb,ka)]
                                    eris_ooee_klb = 1./nkpts * lib.einsum('Lkl,Lba->klba'
                                                    , eris.Loo[kk,kl], eris.Lvv_p[idx_p], optimize=True)
                                if kb > ka:
                                    idx_p = eris.Lvv_idx_p[(ka,kb)]
                                    eris_ooee_klb = 1./nkpts * lib.einsum('Lkl,Lab->klba'
                                                    , eris.Loo[kk,kl], np.conj(eris.Lvv_p[idx_p]), optimize=True)
                            else:
                                eris_ooee_klb = eris_oovv[kk,kl,kb]

                            eris_ccee_klb = eris_ooee_klb[:ncvs,:ncvs].copy()
                            eris_cvee_klb = eris_ooee_klb[:ncvs,ncvs:].copy()
                            eris_vcee_klb = eris_ooee_klb[ncvs:,:ncvs].copy()
                            eris_vvee_klb = eris_ooee_klb[ncvs:,ncvs:].copy()

                            s2_ecv[ka,kj] += lib.einsum('klba,bJl->aJk',eris_vvee_klb,
                                              r2_ecv[kb,kj],optimize=True)
                            s2_ecc[ka,kj] += lib.einsum('Klba,bJl->aJK',eris_cvee_klb,
                                              r2_ecv[kb,kj],optimize=True)
                            s2_ecc[ka,kj] += lib.einsum('KLba,bJL->aJK',eris_ccee_klb,
                                              r2_ecc[kb,kj],optimize=True)
                            s2_evc[ka,kj] += lib.einsum('KLba,bjL->ajK',eris_ccee_klb,
                                                       r2_evc[kb,kj],optimize=True)
                            s2_ecv[ka,kj] += lib.einsum('kLba,bJL->aJk',eris_vcee_klb,
                                                 r2_ecc[kb,kj],optimize=True)

                            kb = kconserv[ka, kj, kl]
                            if adc.eris_direct: 
                                if kb <= ka:
                                    idx_p = eris.Lvv_idx_p[(kb,ka)]
                                    eris_ooee_jlb = 1./nkpts * lib.einsum('Ljl,Lba->jlba'
                                                    , eris.Loo[kj,kl], eris.Lvv_p[idx_p], optimize=True)
                                if kb > ka:
                                    idx_p = eris.Lvv_idx_p[(ka,kb)]
                                    eris_ooee_jlb = 1./nkpts * lib.einsum('Ljl,Lab->jlba'
                                                , eris.Loo[kj,kl], np.conj(eris.Lvv_p[idx_p]), optimize=True)
                            else:
                                eris_ooee_jlb = eris_oovv[kj,kl,kb]

                            eris_ccee_jlb = eris_ooee_jlb[:ncvs,:ncvs].copy()
                            eris_cvee_jlb = eris_ooee_jlb[:ncvs,ncvs:].copy()
                            eris_vcee_jlb = eris_ooee_jlb[ncvs:,:ncvs].copy()
                            eris_vvee_jlb = eris_ooee_jlb[ncvs:,ncvs:].copy()

                            s2_evc[ka,kj] += lib.einsum('jlba,blK->ajK',eris_vvee_jlb,
                                                           r2_evc[kb,kl],optimize=True)
                            s2_ecc[ka,kj] += lib.einsum('Jlba,blK->aJK',eris_cvee_jlb,
                                                  r2_evc[kb,kl],optimize=True)
                            s2_ecc[ka,kj] += lib.einsum('JLba,bLK->aJK',eris_ccee_jlb,
                                                  r2_ecc[kb,kl],optimize=True)
                            s2_ecv[ka,kj] += lib.einsum('JLba,bLk->aJk',eris_ccee_jlb,
                                                  r2_ecv[kb,kl],optimize=True)
                            s2_evc[ka,kj] += lib.einsum('jLba,bLK->ajK',eris_vcee_jlb,
                                                           r2_ecc[kb,kl],optimize=True)

                            if adc.eris_direct: 
                                eris_oeeo_jab = 1./nkpts * lib.einsum('Lja,Lbl->jabl'
                                                , eris.Lov[kj,ka], eris.Lvo[kb,kl], optimize=True)
                            else:
                                eris_oeeo_jab = eris_ovvo[kj,ka,kb]

                            s2_evc[ka,kj] += lib.einsum('jabl,bKl->ajK',eris_oeeo_jab[ncvs:,:,:,ncvs:],
                                                  r2_ecv[kb,kk],optimize=True)
                            s2_evc[ka,kj] -= 2. * lib.einsum('jabl,blK->ajK',eris_oeeo_jab[ncvs:,:,:,ncvs:],
                                                             r2_evc[kb,kl],optimize=True)
                            s2_ecc[ka,kj] += lib.einsum('Jabl,bKl->aJK',eris_oeeo_jab[:ncvs,:,:,ncvs:],
                                                  r2_ecv[kb,kk],optimize=True)
                            s2_ecc[ka,kj] -= 2. * lib.einsum('Jabl,blK->aJK',eris_oeeo_jab[:ncvs,:,:,ncvs:],
                                                    r2_evc[kb,kl],optimize=True)
                            s2_ecc[ka,kj] += lib.einsum('JabL,bKL->aJK',eris_oeeo_jab[:ncvs,:,:,:ncvs],
                                                  r2_ecc[kb,kk],optimize=True)
                            s2_ecc[ka,kj] -= 2. * lib.einsum('JabL,bLK->aJK',eris_oeeo_jab[:ncvs,:,:,:ncvs],
                                                    r2_ecc[kb,kl],optimize=True)
                            s2_ecv[ka,kj] -= 2. * lib.einsum('JabL,bLk->aJk',eris_oeeo_jab[:ncvs,:,:,:ncvs],
                                                    r2_ecv[kb,kl],optimize=True)
                            s2_ecv[ka,kj] += lib.einsum('JabL,bkL->aJk',eris_oeeo_jab[:ncvs,:,:,:ncvs:],
                                                           r2_evc[kb,kk],optimize=True)
                            s2_evc[ka,kj] += lib.einsum('jabL,bKL->ajK',eris_oeeo_jab[ncvs:,:,:,:ncvs],
                                                  r2_ecc[kb,kk],optimize=True)
                            s2_evc[ka,kj] -= 2. * lib.einsum('jabL,bLK->ajK',eris_oeeo_jab[ncvs:,:,:,:ncvs],
                                                            r2_ecc[kb,kl],optimize=True)

                        else:

                            if adc.eris_direct: 
                                eris_oooo_kij = 1./nkpts * lib.einsum('Lki,Ljl->kijl'
                                           , eris.Loo[kk,ki],eris.Loo[kj,kl],optimize=True)
                                eris_oooo_jik = 1./nkpts * lib.einsum('Lji,Lkl->jikl'
                                           , eris.Loo[kj,ki],eris.Loo[kk,kl],optimize=True)
                            else:
                                eris_oooo_kij = eris_oooo[kk,ki,kj]
                                eris_oooo_jik = eris_oooo[kj,ki,kk]

                            eris_occv_kij = eris_oooo_kij[:,:ncvs,:ncvs,ncvs:].copy() 
                            eris_ccvv_kij = eris_oooo_kij[:ncvs,:ncvs,ncvs:,ncvs:].copy() 
                            eris_vcco_jik = eris_oooo_jik[ncvs:,:ncvs,:ncvs,:].copy()
                            eris_ccoo_jik = eris_oooo_jik[:ncvs,:ncvs].copy() 


                            s2_eco[ka,kj] -= lib.einsum('JIKl,aIl->aJK',eris_ccoo_jik,
                                                  r2_eco[ka,ki],optimize=True)
                            s2_eco[ka,kj] -= lib.einsum('kIJl,alI->aJk',eris_occv_kij,
                                                  r2_evc[ka,kl],optimize=True)
                            s2_evc[ka,kj] -= lib.einsum('KIjl,alI->ajK',eris_ccvv_kij,
                                                           r2_evc[ka,kl],optimize=True)
                            s2_evc[ka,kj] -= lib.einsum('jIKl,aIl->ajK',eris_vcco_jik,
                                                           r2_eco[ka,ki],optimize=True)
                        
                            kb = kconserv[ka, kk, kl]
                            if adc.eris_direct: 
                                if kb <= ka:
                                    idx_p = eris.Lvv_idx_p[(kb,ka)]
                                    eris_ooee_klb = 1./nkpts * lib.einsum('Lkl,Lba->klba'
                                                    , eris.Loo[kk,kl], eris.Lvv_p[idx_p], optimize=True)
                                if kb > ka:
                                    idx_p = eris.Lvv_idx_p[(ka,kb)]
                                    eris_ooee_klb = 1./nkpts * lib.einsum('Lkl,Lab->klba'
                                                    , eris.Loo[kk,kl], np.conj(eris.Lvv_p[idx_p]), optimize=True)
                            else:
                                eris_ooee_klb = eris_oovv[kk,kl,kb]

                            eris_ccee_klb = eris_ooee_klb[:ncvs,:ncvs].copy()
                                
                            s2_eco[ka,kj] += lib.einsum('klba,bjl->ajk',
                                                        eris_ooee_klb,r2_eco[kb,kj],optimize=True)
                            s2_evc[ka,kj] += lib.einsum('klba,bjl->ajk',
                                                        eris_ccee_klb,r2_evc[kb,kj],optimize=True)

                            kb = kconserv[ka, kj, kl]
                            if adc.eris_direct: 
                                if kb <= ka:
                                    idx_p = eris.Lvv_idx_p[(kb,ka)]
                                    eris_ooee_jlb = 1./nkpts * lib.einsum('Ljl,Lba->jlba'
                                                    , eris.Loo[kj,kl], eris.Lvv_p[idx_p], optimize=True)
                                if kb > ka:
                                    idx_p = eris.Lvv_idx_p[(ka,kb)]
                                    eris_ooee_jlb = 1./nkpts * lib.einsum('Ljl,Lab->jlba'
                                                    , eris.Loo[kj,kl], np.conj(eris.Lvv_p[idx_p]), optimize=True)
                            else:
                                eris_ooee_jlb = eris_oovv[kj,kl,kb]

                            eris_ccee_jlb = eris_ooee_jlb[:ncvs,:ncvs].copy()
                            r2_ecv_bl = r2_eco[kb,kl,:,:,ncvs:].copy()

                            r2_eoc_bl = np.empty((nvir,nocc,ncvs),dtype=eris.Loo.dtype)
                            r2_eoc_bl[:,:ncvs,:ncvs] = r2_eco[kb,kl,:,:,:ncvs]                       
                            r2_eoc_bl[:,ncvs:,:ncvs] = r2_evc[kb,kl]
                            r2_eoc_bl = np.ascontiguousarray(r2_eoc_bl)                       
 
                            s2_eco[ka,kj,:,:,ncvs:] += lib.einsum('JLba,bLk->aJk',eris_ccee_jlb,
                                                  r2_ecv_bl,optimize=True)
                            s2_eco[ka,kj,:,:,:ncvs] += lib.einsum('Jlba,blK->aJK',eris_ooee_jlb[:ncvs],
                                                  r2_eoc_bl,optimize=True)
                            s2_evc[ka,kj] += lib.einsum('jLba,bLK->ajK',eris_ooee_jlb[ncvs:],
                                                           r2_eoc_bl,optimize=True)
           
                            kb = kconserv[ka, kk, ki]
                            if adc.eris_direct: 
                                eris_oeeo_jab = 1./nkpts * lib.einsum('Lja,Lbl->jabl'
                                                , eris.Lov[kj,ka], eris.Lvo[kb,kl], optimize=True)
                            else:
                                eris_oeeo_jab = eris_ovvo[kj,ka,kb]

                            s2_evc[ka,kj] += lib.einsum('jabL,bKL->ajK',eris_oeeo_jab[ncvs:],
                                                  r2_eco[kb,kk],optimize=True)
                            s2_evc[ka,kj] -= 2. * lib.einsum('jabL,bLK->ajK',eris_oeeo_jab[ncvs:],
                                                            r2_eoc_bl,optimize=True)
                            s2_eco[ka,kj,:,:,:ncvs] += lib.einsum('JabL,bKL->aJK',eris_oeeo_jab[:ncvs],
                                                  r2_eco[kb,kk],optimize=True)
                            s2_eco[ka,kj,:,:,ncvs:] += lib.einsum('JabL,bkL->aJk',eris_oeeo_jab[:ncvs,:,:,:ncvs],
                                                           r2_evc[kb,kk],optimize=True)
                            s2_eco[ka,kj,:,:,:ncvs] -= 2. * lib.einsum('JabL,bLK->aJK',eris_oeeo_jab[:ncvs],
                                                    r2_eoc_bl,optimize=True)
                            s2_eco[ka,kj,:,:,ncvs:] -= 2. * lib.einsum('JabL,bLk->aJk',eris_oeeo_jab[:ncvs,:,:,:ncvs],
                                                    r2_ecv_bl,optimize=True)

                    '''
                    for kl in range(nkpts):
                        kb = kconserv[ka, kk, kl]
                        #eris_veev, eris_vvee
                        s2_ecv[ka,kj] += lib.einsum('klba,bJl->aJk',eris_vvee[kk,kl,kb],
                                              r2_ecv[kb,kj],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('Klba,bJl->aJK',eris_cvee[kk,kl,kb],
                                              r2_ecv[kb,kj],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('KLba,bJL->aJK',eris_ccee[kk,kl,kb],
                                              r2_ecc[kb,kj],optimize=True)
                        s2_evc[ka,kj] += lib.einsum('KLba,bjL->ajK',eris_ccee[kk,kl,kb],
                                                       r2_evc[kb,kj],optimize=True)
                        s2_ecv[ka,kj] += lib.einsum('kLba,bJL->aJk',eris_oovv[kk,kl,kb,ncvs:,:ncvs,:,:],
                                              r2_ecc[kb,kj],optimize=True)

                        #for kl in range(nkpts):
                        kb = kconserv[ka, kj, kl]

                        s2_evc[ka,kj] += lib.einsum('jlba,blK->ajK',eris_vvee[kj,kl,kb],
                                                       r2_evc[kb,kl],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('Jlba,blK->aJK',eris_cvee[kj,kl,kb],
                                              r2_evc[kb,kl],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('JLba,bLK->aJK',eris_ccee[kj,kl,kb],
                                              r2_ecc[kb,kl],optimize=True)
                        s2_ecv[ka,kj] += lib.einsum('JLba,bLk->aJk',eris_ccee[kj,kl,kb],
                                              r2_ecv[kb,kl],optimize=True)
                        s2_evc[ka,kj] += lib.einsum('jLba,bLK->ajK',eris_oovv[kj,kl,kb,ncvs:,:ncvs,:,:],
                                                       r2_ecc[kb,kl],optimize=True)
                        #del eris_vvee
                        s2_evc[ka,kj] += lib.einsum('jabl,bKl->ajK',eris_veev[kj,ka,kb],
                                              r2_ecv[kb,kk],optimize=True)
                        s2_evc[ka,kj] -= 2*lib.einsum('jabl,blK->ajK',eris_veev[kj,ka,kb],
                                                         r2_evc[kb,kl],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('Jabl,bKl->aJK',eris_ceev[kj,ka,kb],
                                              r2_ecv[kb,kk],optimize=True)
                        s2_ecc[ka,kj] -= 2*lib.einsum('Jabl,blK->aJK',eris_ceev[kj,ka,kb],
                                                r2_evc[kb,kl],optimize=True)
                        #del eris_ccee
                        s2_ecc[ka,kj] += lib.einsum('JabL,bKL->aJK',eris_ceec[kj,ka,kb],
                                              r2_ecc[kb,kk],optimize=True)
                        s2_ecc[ka,kj] -= 2*lib.einsum('JabL,bLK->aJK',eris_ceec[kj,ka,kb],
                                                r2_ecc[kb,kl],optimize=True)
                        s2_ecv[ka,kj] -= 2*lib.einsum('JabL,bLk->aJk',eris_ceec[kj,ka,kb],
                                                r2_ecv[kb,kl],optimize=True)
                        s2_ecv[ka,kj] += lib.einsum('JabL,bkL->aJk',eris_ceec[kj,ka,kb],
                                                       r2_evc[kb,kk],optimize=True)

                        #del eris_cvee
                        s2_evc[ka,kj] += lib.einsum('Lbaj,bKL->ajK',eris_ceev[kl,kb,ka].conj(),
                                              r2_ecc[kb,kk],optimize=True)
                        s2_evc[ka,kj] -= 2*lib.einsum('Lbaj,bLK->ajK',eris_ceev[kl,kb,ka].conj(),
                                                        r2_ecc[kb,kl],optimize=True)
                    '''
            #del eris_ceev
            #eris_ceec, eris_ccee
            #del eris_ceec

            #for kj in range(nkpts):
            #    for kk in range(nkpts):
            #        ka = kconserv[kk, kshift, kj]
            #        for kl in range(nkpts):
            #            ki = kconserv[kj, kl, kk]

            #            s2[ka,kj] -= 0.5*lib.einsum('kijl,ali->ajk',
            #                                        eris_oooo[kk,ki,kj], r2[ka,kl], optimize=True)
            #            s2[ka,kj] -= 0.5*lib.einsum('klji,ail->ajk',
            #                                        eris_oooo[kk,kl,kj],r2[ka,ki], optimize=True)

            #        for kl in range(nkpts):
            #            kb = kconserv[ka, kk, kl]
            #            s2[ka,kj] += 0.5*lib.einsum('klba,bjl->ajk',
            #                                        eris_oovv[kk,kl,kb],r2[kb,kj],optimize=True)

            #            kb = kconserv[kl, kj, ka]
            #            s2[ka,kj] +=  0.5*lib.einsum('jabl,bkl->ajk',
            #                                         eris_ovvo[kj,ka,kb],r2[kb,kk],optimize=True)
            #            s2[ka,kj] -=  lib.einsum('jabl,blk->ajk',
            #                                     eris_ovvo[kj,ka,kb],r2[kb,kl],optimize=True)
            #            kb = kconserv[ka, kj, kl]
            #            s2[ka,kj] +=  0.5*lib.einsum('jlba,blk->ajk',
            #                                         eris_oovv[kj,kl,kb],r2[kb,kl],optimize=True)

            #        for ki in range(nkpts):
            #            kb = kconserv[ka, kk, ki]
            #            s2[ka,kj] += 0.5*lib.einsum('kiba,bji->ajk',
            #                                        eris_oovv[kk,ki,kb],r2[kb,kj],optimize=True)

            #            kb = kconserv[ka, kj, ki]
            #            s2[ka,kj] += 0.5*lib.einsum('jiba,bik->ajk',
            #                                        eris_oovv[kj,ki,kb],r2[kb,ki],optimize=True)
            #            s2[ka,kj] -= lib.einsum('jabi,bik->ajk',eris_ovvo[kj,
            #                                    ka,kb],r2[kb,ki],optimize=True)
            #            kb = kconserv[ki, kj, ka]
            #            s2[ka,kj] += 0.5*lib.einsum('jabi,bki->ajk',
            #                                        eris_ovvo[kj,ka,kb],r2[kb,kk],optimize=True)

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

        s2_evc = s2_evc.reshape(-1)
        if not adc.cvs_compact:
            s2_ecc = s2_ecc.reshape(-1)
            s2_ecv = s2_ecv.reshape(-1)
            s = np.hstack((s1,s2_ecc,s2_ecv,s2_evc))
        else: 
            s2_eco = s2_eco.reshape(-1)
            s = np.hstack((s1,s2_eco,s2_evc))
       
        del s1
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

        print(f'[memalloc current+max matvec [GB] = {np.array(tracemalloc.get_traced_memory())/1024**3}')
        tracemalloc.stop()
        return s
    return sigma_


def matvec_off(adc, kshift, M_ij=None, eris=None):

    if adc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(adc.method)

    method = adc.method

    nkpts = adc.nkpts
    nocc = adc.nocc
    ncvs = adc.ncvs
    nval = nocc - ncvs
    kconserv = adc.khelper.kconserv
    n_singles = ncvs
    nvir = adc.nmo - nocc
    n_doubles_ecc = nkpts * nkpts * nvir * ncvs * ncvs
    n_doubles_ecv = nkpts * nkpts * nvir * ncvs * nval

    #s_singles = 0
    #f_singles = n_singles
    #s_doubles = f_singles
    #f_doubles = s_doubles + n_doubles

    s_singles = 0
    f_singles = n_singles
    s_doubles_ecc = f_singles
    f_doubles_ecc = s_doubles_ecc + n_doubles_ecc
    s_doubles_ecv = f_doubles_ecc
    f_doubles_ecv = s_doubles_ecv + n_doubles_ecv
    s_doubles_evc = f_doubles_ecv
    f_doubles_evc = s_doubles_evc + n_doubles_ecv

    mo_energy =  adc.mo_energy
    mo_coeff =  adc.mo_coeff
    mo_coeff, mo_energy = _add_padding(adc, mo_coeff, mo_energy)

    e_occ = [mo_energy[k][:nocc] for k in range(nkpts)]
    e_cvs = [e_occ[k][:ncvs] for k in range(nkpts)]
    e_val = [e_occ[k][ncvs:] for k in range(nkpts)]
    e_vir = [mo_energy[k][nocc:] for k in range(nkpts)]

    e_vir = np.array(e_vir)
    e_occ = np.array(e_occ)

    if M_ij is None:
        M_ij = adc.get_imds()

    #Calculate sigma vector
    def sigma_(r):
        #cput0 = (time.process_time(), time.time())
        cput0 = (time.process_time(), time.perf_counter())
        log = logger.Logger(adc.stdout, adc.verbose)


        if eris.Loo.dtype == np.complex64:
            r = np.ndarray.astype(r, dtype=np.complex64)
        print(f'r vector dtype = {r.dtype}')

        r1 = r[s_singles:f_singles]
        #r2 = r[s_doubles:f_doubles]
        r2_ecc = r[s_doubles_ecc:f_doubles_ecc]
        r2_ecv = r[s_doubles_ecv:f_doubles_ecv]
        #r2_eco = r[s_doubles_ecc:f_doubles_ecv]
        r2_evc = r[s_doubles_evc:f_doubles_evc]

        #r2 = r2.reshape(nkpts,nkpts,nvir,nocc,nocc)
        #s2 = np.zeros((nkpts,nkpts,nvir,nocc,nocc), dtype=np.complex128)
        #r2_ecc = r2_ecc.reshape(nkpts,nkpts,nvir,ncvs,ncvs)
        #r2_ecv = r2_ecv.reshape(nkpts,nkpts,nvir,ncvs,nval)
        r2_eco = r2_eco.reshape(nkpts,nkpts,nvir,ncvs,nocc)
        r2_evc = r2_evc.reshape(nkpts,nkpts,nvir,nval,ncvs)
        #s2_ecc = np.zeros((nkpts,nkpts,nvir,ncvs,ncvs), dtype=np.complex64)
        #s2_ecv = np.zeros((nkpts,nkpts,nvir,ncvs,nval), dtype=np.complex64)
        s2_eco = np.zeros((nkpts,nkpts,nvir,ncvs,nocc), dtype=np.complex64)
        s2_evc = np.zeros((nkpts,nkpts,nvir,nval,ncvs), dtype=np.complex64)
        #s2_ecc = s2_eco[:,:,:,:,:ncvs].copy()
        #s2_ecv = s2_eco[:,:,:,:,ncvs:].copy()
        cell = adc.cell
        kpts = adc.kpts
        madelung = tools.madelung(cell, kpts)

        #eris_ovoo = eris.ovoo
        #eris_cecc = eris_ovoo[:,:,:,:ncvs,:,:ncvs,:ncvs].copy()
        ####
        #eris_cecv = eris_ovoo[:,:,:,:ncvs,:,:ncvs,ncvs:].copy()
        ####
        #eris_cevc = eris_ovoo[:,:,:,:ncvs,:,ncvs:,:ncvs].copy()
        #eris_vecc = eris_ovoo[:,:,:,ncvs:,:,:ncvs,:ncvs].copy()

        #del eris_ovoo

        #r2_ecc *= 0
        #r2_ecv *= 0
        #r2_evc *= 0

############ ADC(2) ij block ############################

        s1 = lib.einsum('ij,j->i',M_ij[kshift],r1)

########### ADC(2) i - kja block #########################
        for kj in range(nkpts):
            for kk in range(nkpts):
                ka = kconserv[kk, kshift, kj]
                ki = kconserv[kj, kk, ka]

                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk].conj(), r2[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj].conj(), r2[ka,kj], optimize=True)

                #s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_cecc, r2_ecc, optimize=True)
                #s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_cecc, r2_ecc, optimize=True)
                #s[s1:f1] += 2. * lib.einsum('jaik,ajk->i', eris_cecv, r2_ecv, optimize=True)
                #s[s1:f1] -= lib.einsum('kaji,ajk->i', eris_vecc, r2_ecv, optimize=True)
                #s[s1:f1] += 2. * lib.einsum('jaki,ajk->i', eris_vecc, r2_evc, optimize=True)
                #s[s1:f1] -= lib.einsum('kaij,ajk->i', eris_cecv, r2_evc, optimize=True)
              
                Lec = eris.Lec
                Lev = eris.Lev
                Lcc = eris.Lcc
                Lcv = eris.Lcv
                Leo = eris.Lvo
                Lco = eris.Loo[:,:,:,:ncvs,:].copy() 
                #eris_eccc_aji = 1./nkpts * lib.einsum('LaJ,LIK->aJIK'
                #                  , Lec[ka,kj], Lcc[ki,kk], optimize=True)
                #eris_eccc_aki = 1./nkpts * lib.einsum('LaK,LIJ->aKIJ'
                #                  , Lec[ka,kk], Lcc[ki,kj], optimize=True)
                #eris_eccv_aji = 1./nkpts * lib.einsum('LaJ,LIk->aJIk'
                #                  , Lec[ka,kj], Lcv[ki,kk], optimize=True)
                #eris_evcc_aki = 1./nkpts * lib.einsum('Lak,LIJ->akIJ'
                #                  , Lev[ka,kk], Lcc[ki,kj], optimize=True)
                eris_ecco_aji = 1./nkpts * lib.einsum('LaJ,LIk->aJIk'
                                  , Lec[ka,kj], Lco[ki,kk], optimize=True)
                eris_eocc_aki = 1./nkpts * lib.einsum('Lak,LIJ->akIJ'
                                  , Leo[ka,kk], Lcc[ki,kj], optimize=True)
                eris_evcc_aji = 1./nkpts * lib.einsum('Laj,LIK->ajIK'
                                  , Lev[ka,kj], Lcc[ki,kk], optimize=True)
                eris_eccv_aki = 1./nkpts * lib.einsum('LaK,LIj->aKIj'
                                  , Lec[ka,kk], Lcv[ki,kj], optimize=True)
                #s1 += 2. * lib.einsum('aJIK,aJK->I',
                #                      eris_eccc_aji, r2_ecc[ka,kj], optimize=True)
                #s1 -= lib.einsum('aKIJ,aJK->I',
                #                 eris_eccc_aki, r2_ecc[ka,kj], optimize=True)
                #s1 += 2. * lib.einsum('aJIk,aJk->I',
                #                      eris_eccv_aji, r2_ecv[ka,kj], optimize=True)
                #s1 -= lib.einsum('akIJ,aJk->I',
                #                 eris_evcc_aki, r2_ecv[ka,kj], optimize=True)
                s1 += 2. * lib.einsum('aJIk,aJk->I',
                                      eris_ecco_aji, r2_eco[ka,kj], optimize=True)
                s1 -= lib.einsum('akIJ,aJk->I',
                                 eris_eocc_aki, r2_eco[ka,kj], optimize=True)
                s1 += 2. * lib.einsum('ajIK,ajK->I',
                                      eris_evcc_aji, r2_evc[ka,kj], optimize=True)
                s1 -= lib.einsum('aKIj,ajK->I',
                                 eris_eccv_aki, r2_evc[ka,kj], optimize=True)



                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk,:ncvs,:,:ncvs,:ncvs].conj(), r2_ecc[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj,:ncvs,:,:ncvs,:ncvs].conj(), r2_ecc[ka,kj], optimize=True)
                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk,:ncvs,:,ncvs:,:ncvs].conj(), r2_ecv[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj,ncvs:,:,:ncvs,:ncvs].conj(), r2_ecv[ka,kj], optimize=True)
                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_ovoo[kj,ka,kk,ncvs:,:,:ncvs,:ncvs].conj(), r2_evc[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_ovoo[kk,ka,kj,:ncvs,:,ncvs:,:ncvs].conj(), r2_evc[ka,kj], optimize=True)

                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_cecc[kj,ka,kk].conj(), r2_ecc[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_cecc[kk,ka,kj].conj(), r2_ecc[ka,kj], optimize=True)
                ##s1 += 2. * lib.einsum('jaik,ajk->i',
                ##                      eris_cecv[kj,ka,ki].conj(), r2_ecv[ka,kj], optimize=True)
                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_cevc[kj,ka,kk].conj(), r2_ecv[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaji,ajk->i',
                #                 eris_vecc[kk,ka,kj].conj(), r2_ecv[ka,kj], optimize=True)
                #s1 += 2. * lib.einsum('jaki,ajk->i',
                #                      eris_vecc[kj,ka,kk].conj(), r2_evc[ka,kj], optimize=True)
                #s1 -= lib.einsum('kaij,ajk->i',
                #                 eris_cecv[kk,ka,ki].conj(), r2_evc[ka,kj], optimize=True)

#################### ADC(2) ajk - i block ############################

                #s2[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk], r1, optimize=True)

                #s[s2_ecc:f2_ecc] += lib.einsum('jaki,i->ajk', eris_cecc, r1, optimize=True).reshape(-1)
                #s[s2_ecv:f2_ecv] += lib.einsum('jaik,i->ajk', eris_cecv, r1, optimize=True).reshape(-1)
                #s[s2_evc:f2_evc] += lib.einsum('jaki,i->ajk', eris_vecc, r1, optimize=True).reshape(-1)

                #s2_ecc[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk][:ncvs,:,:ncvs,:ncvs], r1, optimize=True)
                #s2_ecv[ka,kj] += lib.einsum('jaik,i->ajk', eris_ovoo[kj,ka,ki][:ncvs,:,:ncvs,ncvs:], r1, optimize=True)
                #s2_evc[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk][ncvs:,:,:ncvs,:ncvs], r1, optimize=True)

                #s2_ecc[ka,kj] += lib.einsum('jaki,i->ajk', eris_cecc[kj,ka,kk], r1, optimize=True)
                ##s2_ecv[ka,kj] += lib.einsum('jaik,i->ajk', eris_cecv[kj,ka,ki], r1, optimize=True)
                #s2_ecv[ka,kj] += lib.einsum('jaki,i->ajk', eris_cevc[kj,ka,kk], r1, optimize=True)
                #s2_evc[ka,kj] += lib.einsum('jaki,i->ajk', eris_vecc[kj,ka,kk], r1, optimize=True)
                ##s2_ecv[ka,kj] += lib.einsum('jaki,i->ajk', eris_ovoo[kj,ka,kk,:,:,:,:ncvs], r1, optimize=True)[:,:ncvs,ncvs:]

                #s2_ecc[ka,kj] += lib.einsum('jaki,i->ajk', eris_cecc[kj,ka,kk], r1, optimize=True)
                #s2_ecv[ka,kj] += lib.einsum('jaki,i->ajk', eris_cevc[kj,ka,kk], r1, optimize=True)
                #s2_evc[ka,kj] += lib.einsum('jaki,i->ajk', eris_vecc[kj,ka,kk], r1, optimize=True)
                ###>>>>>>>>>>>>>>>>>>>>>
                #s2_ecc[ka,kj] += lib.einsum('aJIK,I->aJK', eris_eccc_aji.conj(), r1, optimize=True)
                #s2_ecv[ka,kj] += lib.einsum('aJIk,I->aJk', eris_eccv_aji.conj(), r1, optimize=True)
                s2_eco[ka,kj] += lib.einsum('aJIk,I->aJk', eris_ecco_aji.conj(), r1, optimize=True)
                s2_evc[ka,kj] += lib.einsum('ajIK,I->ajK', eris_evcc_aji.conj(), r1, optimize=True)
               
                ##>>>idx1 = np.ones((nocc,nocc))
                ##>>>idx1[ncvs:,ncvs:] = 0
                ##>>>idx2 = np.nonzero(idx1) 
                ##>>>eris_eoco_aji = 1./nkpts * lib.einsum('LaJ,LIK->aJIK'
                ##>>>                  , eris.Lvo[ka,kj][:,:,idx2[1]], eris.Loo[ki,kk][:,:ncvs,idx2[0]], optimize=True)
                ##>>>shape_eoco = eris_eoco_aji.shape 
                ##>>>shape_Lvo = eris.Lvo[ka,kj].shape
                ##>>>shape_Loo = eris.Loo[ki,kk].shape
                ##>>>shape_idx2_0 = idx2[0].shape
                ##>>>shape_idx2_1 = idx2[1].shape
                ##>>>shape_Lvo_idx2_1 = eris.Lvo[ka,kj][:,:,idx2[1]].shape
                ##>>>shape_Loo_idx2_0 = eris.Loo[ki,kk][:,:,idx2[0]].shape

                ##>>>print(f'shape_eoco         = {shape_eoco      }')  
                ##>>>print(f'shape_Lvo          = {shape_Lvo       }') 
                ##>>>print(f'shape_Loo          = {shape_Loo       }') 
                ##>>>print(f'shape_idx2_0       = {shape_idx2_0    }')
                ##>>>print(f'shape_idx2_1       = {shape_idx2_1    }')
                ##>>>print(f'shape_Lvo_idx2_1   = {shape_Lvo_idx2_1}')
                ##>>>print(f'shape_Loo_idx2_0   = {shape_Loo_idx2_0}') 

                ##>>>#s_tmp = lib.einsum('ajIK,I->ajK', eris_eoco_aji.conj()
                ##>>>#, r1, optimize=True).reshape(-1)
                ##>>>exit() 
################# ADC(2) ajk - bil block ############################

                #s2[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2[ka, kj])
                #s2[ka, kj] += lib.einsum('j,ajk->ajk', e_occ[kj], r2[ka, kj])
                #s2[ka, kj] += lib.einsum('k,ajk->ajk', e_occ[kk], r2[ka, kj])

                #temp_ecc = D_aij[:,:ncvs,:ncvs].reshape(-1)
                #s[s2_ecc:f2_ecc] += temp_ecc*r2_ecc.reshape(-1)
                #temp_ecv = D_aij[:,:ncvs,ncvs:].reshape(-1)
                #s[s2_ecv:f2_ecv] += temp_ecv*r2_ecv.reshape(-1)
                #temp_evc = D_aij[:,ncvs:,:ncvs].reshape(-1)
                #s[s2_evc:f2_evc] += temp_evc*r2_evc.reshape(-1)

                #s2_ecc[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_ecc[ka, kj])
                #s2_ecc[ka, kj] += lib.einsum('j,ajk->ajk', e_cvs[kj], r2_ecc[ka, kj])
                #s2_ecc[ka, kj] += lib.einsum('k,ajk->ajk', e_cvs[kk], r2_ecc[ka, kj])
                #s2_ecv[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_ecv[ka, kj])
                #s2_ecv[ka, kj] += lib.einsum('j,ajk->ajk', e_cvs[kj], r2_ecv[ka, kj])
                #s2_ecv[ka, kj] += lib.einsum('k,ajk->ajk', e_val[kk], r2_ecv[ka, kj])
                s2_eco[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_eco[ka, kj])
                s2_eco[ka, kj] += lib.einsum('j,ajk->ajk', e_cvs[kj], r2_eco[ka, kj])
                s2_eco[ka, kj] += lib.einsum('k,ajk->ajk', e_occ[kk], r2_eco[ka, kj])
                s2_evc[ka, kj] -= lib.einsum('a,ajk->ajk', e_vir[ka], r2_evc[ka, kj])
                s2_evc[ka, kj] += lib.einsum('j,ajk->ajk', e_val[kj], r2_evc[ka, kj])
                s2_evc[ka, kj] += lib.einsum('k,ajk->ajk', e_cvs[kk], r2_evc[ka, kj])

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-x" or method == "adc(3)"):

            eris_oooo = eris.oooo
            eris_oovv = eris.oovv
            eris_ovvo = eris.ovvo

            eris_cccc = eris_oooo[:,:,:,:ncvs,:ncvs,:ncvs,:ncvs].copy()
            eris_cccv = eris_oooo[:,:,:,:ncvs,:ncvs,:ncvs,ncvs:].copy()
            eris_ccvv = eris_oooo[:,:,:,:ncvs,:ncvs,ncvs:,ncvs:].copy()
            eris_ceec = eris_ovvo[:,:,:,:ncvs,:,:,:ncvs].copy()
            eris_ceev = eris_ovvo[:,:,:,:ncvs,:,:,ncvs:].copy()
            eris_veev = eris_ovvo[:,:,:,ncvs:,:,:,ncvs:].copy()
            eris_ccee = eris_oovv[:,:,:,:ncvs,:ncvs,:,:].copy()
            eris_cvee = eris_oovv[:,:,:,:ncvs,ncvs:,:,:].copy()
            eris_vvee = eris_oovv[:,:,:,ncvs:,ncvs:,:,:].copy()

            #del eris_oooo
            #del eris_oovv
            #del eris_ovvo
            eris_cvcv = eris_oooo[:,:,:,:ncvs,ncvs:,:ncvs,ncvs:].copy()
            eris_vccv = eris_oooo[:,:,:,ncvs:,:ncvs,:ncvs,ncvs:].copy()
            eris_ccvc = eris_oooo[:,:,:,:ncvs,:ncvs,ncvs:,:ncvs].copy()
            eris_cvvc = eris_oooo[:,:,:,:ncvs,ncvs:,ncvs:,:ncvs].copy()
            eris_vcee = eris_oovv[:,:,:,ncvs:,:ncvs,:,:].copy()

            s2_ecc = s2_eco[:,:,:,:,:ncvs].copy()
            s2_ecv = s2_eco[:,:,:,:,ncvs:].copy()
            r2_ecc = r2_eco[:,:,:,:,:ncvs].copy()
            r2_ecv = r2_eco[:,:,:,:,ncvs:].copy()
            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    for kl in range(nkpts):
                        ki = kconserv[kj, kl, kk]

                        s2_ecc[ka,kj] -= lib.einsum('KIJl,alI->aJK',eris_cccv[kk,ki,kj],
                                              r2_evc[ka,kl],optimize=True)
                        s2_ecc[ka,kj] -= lib.einsum('JIKl,aIl->aJK',eris_cccv[kj,ki,kk],
                                              r2_ecv[ka,ki],optimize=True)
                        s2_ecc[ka,kj] -= lib.einsum('KLJI,aIL->aJK',eris_cccc[kk,kl,kj],
                                                       r2_ecc[ka,ki],optimize=True)
                        s2_ecv[ka,kj] -= lib.einsum('JIkl,aIl->aJk',eris_ccvv[kj,ki,kk],
                                                       r2_ecv[ka,ki],optimize=True)
                        s2_ecv[ka,kj] -= lib.einsum('kIJl,alI->aJk',eris_oooo[kk,ki,kj,ncvs:,:ncvs,:ncvs,ncvs:],
                                              r2_evc[ka,kl],optimize=True)
                        s2_ecv[ka,kj] -= lib.einsum('JIkL,aIL->aJk',eris_oooo[kj,ki,kk,:ncvs,:ncvs,ncvs:,:ncvs],
                                              r2_ecc[ka,ki],optimize=True)
                        #eris_cvcv, eris_ccvv
                        s2_evc[ka,kj] -= lib.einsum('KIjl,alI->ajK',eris_ccvv[kk,ki,kj],
                                                       r2_evc[ka,kl],optimize=True)
                        s2_evc[ka,kj] -= lib.einsum('KljI,aIl->ajK',eris_oooo[kk,kl,kj,:ncvs,ncvs:,ncvs:,:ncvs],
                                                       r2_ecv[ka,ki],optimize=True)
                        #ris_cccv
                        s2_evc[ka,kj] -= lib.einsum('KLjI,aIL->ajK',eris_oooo[kk,kl,kj,:ncvs,:ncvs,ncvs:,:ncvs],
                                                       r2_ecc[ka,ki],optimize=True)
                        #eris_cccc

                        #r2_al = np.zeros((nvir,nocc,nocc), dtype=np.complex128)
                        #r2_al[:,:ncvs,:ncvs] += r2_ecc[ka,kl]
                        #r2_al[:,:ncvs,ncvs:] += r2_ecv[ka,kl]
                        #r2_al[:,ncvs:,:ncvs] += r2_evc[ka,kl]
                        #r2_ai = np.zeros((nvir,nocc,nocc), dtype=np.complex128)
                        #r2_ai[:,:ncvs,:ncvs] += r2_ecc[ka,ki]
                        #r2_ai[:,:ncvs,ncvs:] += r2_ecv[ka,ki]
                        #r2_ai[:,ncvs:,:ncvs] += r2_evc[ka,ki]

                        #temp1 = -0.5*lib.einsum('kijl,ali->ajk',
                        #                            eris_oooo[kk,ki,kj], r2_al, optimize=True)
                        #temp1 -= 0.5*lib.einsum('klji,ail->ajk',
                        #                            eris_oooo[kk,kl,kj],r2_ai, optimize=True)
                        #s2_ecc[ka,kj] -= temp1[:,:ncvs,:ncvs]
                        #s2_ecv[ka,kj] -= temp1[:,:ncvs,ncvs:]
                        #s2_evc[ka,kj] -= temp1[:,ncvs:,:ncvs]

                    #del eris_ccvv
                    #del eris_cvcv
                    #del eris_cccv
                    #del eris_cccc

                    for kl in range(nkpts):
                        kb = kconserv[ka, kk, kl]
                        #eris_veev, eris_vvee
                        s2_ecv[ka,kj] += lib.einsum('klba,bJl->aJk',eris_vvee[kk,kl,kb],
                                              r2_ecv[kb,kj],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('Klba,bJl->aJK',eris_cvee[kk,kl,kb],
                                              r2_ecv[kb,kj],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('KLba,bJL->aJK',eris_ccee[kk,kl,kb],
                                              r2_ecc[kb,kj],optimize=True)
                        s2_evc[ka,kj] += lib.einsum('KLba,bjL->ajK',eris_ccee[kk,kl,kb],
                                                       r2_evc[kb,kj],optimize=True)
                        s2_ecv[ka,kj] += lib.einsum('kLba,bJL->aJk',eris_oovv[kk,kl,kb,ncvs:,:ncvs,:,:],
                                              r2_ecc[kb,kj],optimize=True)

                        #for kl in range(nkpts):
                        kb = kconserv[ka, kj, kl]

                        s2_evc[ka,kj] += lib.einsum('jlba,blK->ajK',eris_vvee[kj,kl,kb],
                                                       r2_evc[kb,kl],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('Jlba,blK->aJK',eris_cvee[kj,kl,kb],
                                              r2_evc[kb,kl],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('JLba,bLK->aJK',eris_ccee[kj,kl,kb],
                                              r2_ecc[kb,kl],optimize=True)
                        s2_ecv[ka,kj] += lib.einsum('JLba,bLk->aJk',eris_ccee[kj,kl,kb],
                                              r2_ecv[kb,kl],optimize=True)
                        s2_evc[ka,kj] += lib.einsum('jLba,bLK->ajK',eris_oovv[kj,kl,kb,ncvs:,:ncvs,:,:],
                                                       r2_ecc[kb,kl],optimize=True)
                        #del eris_vvee
                        s2_evc[ka,kj] += lib.einsum('jabl,bKl->ajK',eris_veev[kj,ka,kb],
                                              r2_ecv[kb,kk],optimize=True)
                        s2_evc[ka,kj] -= 2*lib.einsum('jabl,blK->ajK',eris_veev[kj,ka,kb],
                                                         r2_evc[kb,kl],optimize=True)
                        s2_ecc[ka,kj] += lib.einsum('Jabl,bKl->aJK',eris_ceev[kj,ka,kb],
                                              r2_ecv[kb,kk],optimize=True)
                        s2_ecc[ka,kj] -= 2*lib.einsum('Jabl,blK->aJK',eris_ceev[kj,ka,kb],
                                                r2_evc[kb,kl],optimize=True)
                        #del eris_ccee
                        s2_ecc[ka,kj] += lib.einsum('JabL,bKL->aJK',eris_ceec[kj,ka,kb],
                                              r2_ecc[kb,kk],optimize=True)
                        s2_ecc[ka,kj] -= 2*lib.einsum('JabL,bLK->aJK',eris_ceec[kj,ka,kb],
                                                r2_ecc[kb,kl],optimize=True)
                        s2_ecv[ka,kj] -= 2*lib.einsum('JabL,bLk->aJk',eris_ceec[kj,ka,kb],
                                                r2_ecv[kb,kl],optimize=True)
                        s2_ecv[ka,kj] += lib.einsum('JabL,bkL->aJk',eris_ceec[kj,ka,kb],
                                                       r2_evc[kb,kk],optimize=True)

                        #del eris_cvee
                        s2_evc[ka,kj] += lib.einsum('Lbaj,bKL->ajK',eris_ceev[kl,kb,ka].conj(),
                                              r2_ecc[kb,kk],optimize=True)
                        s2_evc[ka,kj] -= 2*lib.einsum('Lbaj,bLK->ajK',eris_ceev[kl,kb,ka].conj(),
                                                        r2_ecc[kb,kl],optimize=True) 


            #if adc.exxdiv is not None:
            #    s2 += madelung * r2

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
        #s2 = s2.reshape(-1)
        #s2_ecc = s2_ecc.reshape(-1)
        #s2_ecv = s2_ecv.reshape(-1) 
        s2_eco = s2_eco.reshape(-1) 
        s2_evc = s2_evc.reshape(-1)
        s2_ecc = s2_ecc.reshape(-1) 
        s2_ecv = s2_ecv.reshape(-1) 
       
        #s1 *= 0 
        #s2_ecc *= 0
        #s2_ecv *= 0
        #s2_evc *= 0

        #s = np.hstack((s1,s2))
         
        #s2 = np.hstack((s2_ecc,s2_ecv,s2_evc))
        #s2 += s_tmp
        #s = np.hstack((s1,s2))
        s = np.hstack((s1,s2_ecc,s2_ecv,s2_evc))
        #s = np.hstack((s1,s2_eco,s2_evc))
        del s1
        #del s2
        #del s2_ecc, s2_ecv, s2_evc
        del s2_eco, s2_evc
        cput0 = log.timer_debug1("completed sigma vector calculation", *cput0)
        s *= -1.0

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


class RADCIPCVS(kadc_rhf.RADC):
    '''restricted ADC for IP-CVS energies and spectroscopic amplitudes

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
            >>> myadcipcvs = adc.RADC(myadc).run()

    Saved results

        e_ip_cvs : float or list of floats
            IP-CVS energy (eigenvalue). For nroots = 1, it is a single float number. If nroots > 1, it is a list
            of floats for the lowest nroots eigenvalues.
        v_ip_cvs : array
            Eigenvectors for each IP-CVS transition.
        p_ip_cvs : float
            Spectroscopic amplitudes for each IP-CVS transition.
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
        self.ncvs = adc.ncvs
        self.eris_direct = adc.eris_direct
        self.cvs_compact = adc.cvs_compact
        self.precision_single = adc.precision_single

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
        #return diag
