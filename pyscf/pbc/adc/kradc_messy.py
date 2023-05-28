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


# Note : All interals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba


def calculate_chunk_size(myadc):

    avail_mem = (myadc.max_memory - lib.current_memory()[0]) * 0.5
    nocc = [np.count_nonzero(myadc.mo_occ[ikpt]) for ikpt in range(myadc.nkpts)]
    nocc = np.amax(nocc)
    nmo = [len(myadc.mo_occ[ikpt]) for ikpt in range(myadc.nkpts)]
    nmo = np.max(nocc) + np.max(np.array(nmo) - np.array(nocc))
    nvir = nmo - nocc
    vvv_mem = (nvir**3) * 8/1e6

    chnk_size =  int(avail_mem/vvv_mem)

    if chnk_size <= 0 :
        chnk_size = 1

    return chnk_size

def density_fit(self, auxbasis=None, with_df=None):
    from pyscf.pbc import df
    if with_df is None:
        self.with_df = df.DF(self._scf.mol)
        self.with_df.max_memory = self.max_memory
        self.with_df.stdout = self.stdout
        self.with_df.verbose = self.verbose
        if auxbasis is None:
            self.with_df.auxbasis = self._scf.with_df.auxbasis
        else:
            self.with_df.auxbasis = auxbasis
    else:
        self.with_df = with_df
    return self

def compute_amplitudes(myadc, eris):

    #cput0 = (time.process_time(), time.time())
    cput0 = (time.process_time(), time.perf_counter())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
        raise NotImplementedError(myadc.method)

    nmo = myadc.nmo
    nocc = myadc.nocc
    nvir = nmo - nocc
    nkpts = myadc.nkpts
    cell = myadc.cell
    kpts = myadc.kpts
    madelung = tools.madelung(cell, kpts)

    # Compute first-order doubles t2 (tijab)
    tf = tempfile.TemporaryFile()
    f = h5py.File(tf, 'a')
    t2_1 = f.create_dataset('t2_1', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)

    mo_energy =  myadc.mo_energy
    mo_coeff =  myadc.mo_coeff
    mo_coeff, mo_energy = _add_padding(myadc, mo_coeff, mo_energy)

    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]

    # Get location of non-zero/padded elements in occupied and virtual space
    nonzero_opadding, nonzero_vpadding = padding_k_idx(myadc, kind="split")

    kconserv = myadc.khelper.kconserv
    touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
        if touched[ki, kj, ka]:
            continue

        kb = kconserv[ki, ka, kj]
        # For discussion of LARGE_DENOM, see t1new update above
        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])

        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
                       fac=[1.0,-1.0])
        eijab = eia[:, None, :, None] + ejb[:, None, :]

        t2_1[ki,kj,ka] = eris.ovov[ki,ka,kj].conj().transpose((0,2,1,3)) / eijab

        if ka != kb:
            eijba = eijab.transpose(0, 1, 3, 2)
            t2_1[ki, kj, kb] = eris.ovov[ki,kb,kj].conj().transpose((0,2,1,3)) / eijba

        touched[ki, kj, ka] = touched[ki, kj, kb] = True

    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)

    t1_2 = None
    t2_2 = None
    t1_3 = None
    t2_1_vvvv = None
    t1 = (t1_2, t1_3)
    t2 = (t2_1, t2_2)

    return t1, t2, t2_1_vvvv

def get_imds_ip(adc, eris, t2):

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

    #if eris is None:
    #    eris = adc.transform_integrals()

    # i-j block
    # Zeroth-order terms
    print('running projected CVS')
    #t2_1 = adc.t2[0]
    t2_1 = t2[0]
    eris_ovov = eris.ovov
    for ki in range(nkpts):
        kj = ki
        M_ij[ki] = lib.einsum('ij,j->ij', idn_occ , e_occ[kj])
        for kl in range(nkpts):
            for kd in range(nkpts):
                ke = kconserv[kj,kd,kl]
                #t2_1 = adc.t2[0]
                t2_1_ild = t2_1[ki,kl,kd]

                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ilde,jdle->ij',t2_1_ild, eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ilde,jeld->ij',t2_1_ild, eris_ovov[kj,ke,kl],optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('ilde,jdle->ij',t2_1_ild,
                                             eris_ovov[kj,kd,kl],optimize=True)
                del t2_1_ild

                t2_1_lid = t2_1[kl,ki,kd]
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('lide,jdle->ij',t2_1_lid, eris_ovov[kj,kd,kl],optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('lide,jeld->ij',t2_1_lid, eris_ovov[kj,ke,kl],optimize=True)
                del t2_1_lid

                t2_1_jld = t2_1[kj,kl,kd]
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('jlde,idle->ij',t2_1_jld.conj(),
                               eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('jlde,ield->ij',t2_1_jld.conj(),
                               eris_ovov[ki,ke,kl].conj(),optimize=True)
                M_ij[ki] += 0.5 * lib.einsum('jlde,idle->ij',t2_1_jld.conj(),
                                             eris_ovov[ki,kd,kl].conj(),optimize=True)
                del t2_1_jld

                t2_1_ljd = t2_1[kl,kj,kd]
                M_ij[ki] -= 0.5 * 0.5 * \
                    lib.einsum('ljde,idle->ij',t2_1_ljd.conj(),
                               eris_ovov[ki,kd,kl].conj(),optimize=True)
                M_ij[ki] += 0.5 * 0.5 * \
                    lib.einsum('ljde,ield->ij',t2_1_ljd.conj(),
                               eris_ovov[ki,ke,kl].conj(),optimize=True)
                del t2_1_ljd
                del t2_1

    cput0 = log.timer_debug1("Completed M_ij second-order terms ADC(2) calculation", *cput0)

    return M_ij

#def compute_amplitudes(myadc, eris):
#
#    #cput0 = (time.process_time(), time.time())
#    cput0 = (time.process_time(), time.perf_counter())
#    log = logger.Logger(myadc.stdout, myadc.verbose)
#
#    if myadc.method not in ("adc(2)", "adc(2)-x", "adc(3)"):
#        raise NotImplementedError(myadc.method)
#
#    nmo = myadc.nmo
#    nocc = myadc.nocc
#    nvir = nmo - nocc
#    nkpts = myadc.nkpts
#    cell = myadc.cell
#    kpts = myadc.kpts
#    madelung = tools.madelung(cell, kpts)
#
#    # Compute first-order doubles t2 (tijab)
#    tf = tempfile.TemporaryFile()
#    f = h5py.File(tf, 'a')
#    t2_1 = f.create_dataset('t2_1', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
#
#    mo_energy =  myadc.mo_energy
#    mo_coeff =  myadc.mo_coeff
#    mo_coeff, mo_energy = _add_padding(myadc, mo_coeff, mo_energy)
#
#    mo_e_o = [mo_energy[k][:nocc] for k in range(nkpts)]
#    mo_e_v = [mo_energy[k][nocc:] for k in range(nkpts)]
#
#    # Get location of non-zero/padded elements in occupied and virtual space
#    nonzero_opadding, nonzero_vpadding = padding_k_idx(myadc, kind="split")
#
#    kconserv = myadc.khelper.kconserv
#    touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)
#
#    kpts_gen = kpts_helper.loop_kkk(nkpts)
#    #for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
#        #if touched[ki, kj, ka]:
#            #continue
#    def t_kpt_calc((ki,kj,ka)):
#
#        #ki, kj, ka = kpts[idx]
#        
#
#        kb = kconserv[ki, ka, kj]
#        # For discussion of LARGE_DENOM, see t1new update above
#        eia = _get_epq([0,nocc,ki,mo_e_o,nonzero_opadding],
#                       [0,nvir,ka,mo_e_v,nonzero_vpadding],
#                       fac=[1.0,-1.0])
#
#        ejb = _get_epq([0,nocc,kj,mo_e_o,nonzero_opadding],
#                       [0,nvir,kb,mo_e_v,nonzero_vpadding],
#                       fac=[1.0,-1.0])
#        eijab = eia[:, None, :, None] + ejb[:, None, :]
#
#        #t2_1[ki,kj,ka] = eris.ovov[ki,ka,kj].conj().transpose((0,2,1,3)) / eijab
#        t2_1_ija = eris.ovov[ki,ka,kj].conj().transpose((0,2,1,3)) / eijab
#
#        if ka != kb:
#            eijba = eijab.transpose(0, 1, 3, 2)
#            #t2_1[ki, kj, kb] = eris.ovov[ki,kb,kj].conj().transpose((0,2,1,3)) / eijba
#            t2_1_ijb = eris.ovov[ki,kb,kj].conj().transpose((0,2,1,3)) / eijba
#
#        touched[ki, kj, ka] = touched[ki, kj, kb] = True
#
#    cput0 = log.timer_debug1("Completed t2_1 amplitude calculation", *cput0)
#
#    t1_2 = None
#    t2_2 = None
#    t1_3 = None
#    t2_1_vvvv = None
#
#    t1 = (t1_2, t1_3)
#    t2 = (t2_1, t2_2)
#
#    return t1, t2, t2_1_vvvv


def compute_energy(myadc, t2, eris):

    nkpts = myadc.nkpts

    emp2 = 0.0
    eris_ovov = eris.ovov
    t2_amp = t2[0][:]

    if (myadc.method == "adc(3)"):
        t2_amp += t2[1][:]

    for ki, kj, ka in kpts_helper.loop_kkk(nkpts):

        emp2 += 2 * lib.einsum('ijab,iajb', t2_amp[ki,kj,ka], eris_ovov[ki,ka,kj],optimize=True)
        emp2 -= 1 * lib.einsum('ijab,jaib', t2_amp[ki,kj,ka], eris_ovov[kj,ka,ki],optimize=True)

    del t2_amp
    emp2 = emp2.real / nkpts
    return emp2


def compute_amplitudes_energy(myadc, eris, verbose=None):

    t1,t2,myadc.imds.t2_1_vvvv = compute_amplitudes(myadc, eris)
    e_corr = compute_energy(myadc, t2, eris)

    return e_corr, t1, t2

def transform_integrals_df(myadc):
    from pyscf.ao2mo import _ao2mo
    from pyscf.pbc.df import df
    cput0 = (time.process_time(), time.perf_counter())
    log = logger.Logger(myadc.stdout, myadc.verbose)

    cell = myadc.cell
    kpts = myadc.kpts
    nkpts = myadc.nkpts
    nocc = myadc.nocc
    nmo = myadc.nmo
    nvir = nmo - nocc
    nao = cell.nao_nr()

    myadc = density_fit(myadc)

    if myadc._scf.with_df._cderi is None:
        myadc._scf.with_df.build()
    dtype = myadc.mo_coeff[0].dtype

    mo_coeff = myadc.mo_coeff = padded_mo_coeff(myadc, myadc.mo_coeff)

    kconserv = myadc.khelper.kconserv

    # The momentum conservation array
    kconserv = myadc.khelper.kconserv

    with_df = myadc.with_df
    naux = with_df.get_naoaux()
    eris = lambda:None

    eris.dtype = dtype = np.result_type(dtype)
    eris.Lpq_mo = Lpq_mo = np.empty((nkpts, nkpts), dtype=object)
    Loo = np.empty((nkpts,nkpts,naux,nocc,nocc),dtype=dtype)
    Lvo = np.empty((nkpts,nkpts,naux,nvir,nocc),dtype=dtype)
    eris.Lvv = np.empty((nkpts,nkpts,naux,nvir,nvir),dtype=dtype)
    eris.Lov = np.empty((nkpts,nkpts,naux,nocc,nvir),dtype=dtype)

    eris.vvvv = None
    eris.ovvv = None

    #with df._load3c(with_df._cderi, 'j3c') as fload:
    with df._load3c(myadc._scf.with_df._cderi, 'j3c') as fload:
        tao = []
        ao_loc = None
        for ki, kpti in enumerate(kpts):
            for kj, kptj in enumerate(kpts):
                Lpq_ao = np.asarray(fload(kpti, kptj))

                mo = np.hstack((mo_coeff[ki], mo_coeff[kj]))
                mo = np.asarray(mo, dtype=dtype, order='F')
                if dtype == np.double:
                    out = _ao2mo.nr_e2(Lpq_ao, mo, (0, nmo, nmo, nmo+nmo), aosym='s2')
                else:
                    #Note: Lpq.shape[0] != naux if linear dependency is found in auxbasis
                    if Lpq_ao[0].size != nao**2:  # aosym = 's2'
                        Lpq_ao = lib.unpack_tril(Lpq_ao).astype(np.complex128)
                    out = _ao2mo.r_e2(Lpq_ao, mo, (0, nmo, nmo, nmo+nmo), tao, ao_loc)
                Lpq_mo[ki, kj] = out.reshape(-1, nmo, nmo)

                Loo[ki,kj] = Lpq_mo[ki,kj][:,:nocc,:nocc]
                eris.Lov[ki,kj] = Lpq_mo[ki,kj][:,:nocc,nocc:]
                Lvo[ki,kj] = Lpq_mo[ki,kj][:,nocc:,:nocc]
                eris.Lvv[ki,kj] = Lpq_mo[ki,kj][:,nocc:,nocc:]

    eris.feri = feri = lib.H5TmpFile()

    eris.oooo = feri.create_dataset('oooo', (nkpts,nkpts,nkpts,nocc,nocc,nocc,nocc), dtype=dtype)
    eris.oovv = feri.create_dataset('oovv', (nkpts,nkpts,nkpts,nocc,nocc,nvir,nvir), dtype=dtype)
    eris.ovoo = feri.create_dataset('ovoo', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nocc), dtype=dtype)
    eris.ovov = feri.create_dataset('ovov', (nkpts,nkpts,nkpts,nocc,nvir,nocc,nvir), dtype=dtype)
    eris.ovvo = feri.create_dataset('ovvo', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nocc), dtype=dtype)
    #eris.ovvv = feri.create_dataset('ovvv', (nkpts,nkpts,nkpts,nocc,nvir,nvir,nvir), dtype=dtype)

    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                ks = kconserv[kp,kq,kr]
                eris.oooo[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', Loo[kp,kq], Loo[kr,ks])/nkpts
                eris.oovv[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', Loo[kp,kq], eris.Lvv[kr,ks])/nkpts
                eris.ovoo[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', eris.Lov[kp,kq], Loo[kr,ks])/nkpts
                eris.ovov[kp,kq,kr] = lib.einsum(
                    'Lpq,Lrs->pqrs', eris.Lov[kp,kq], eris.Lov[kr,ks])/nkpts
                eris.ovvo[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', eris.Lov[kp,kq], Lvo[kr,ks])/nkpts
                #eris.ovvv[kp,kq,kr] = lib.einsum('Lpq,Lrs->pqrs', eris.Lov[kp,kq], Lvv[kr,ks])/nkpts

    cput0 = log.timer_debug1("Completed ERIS calculation", *cput0)
    return eris

