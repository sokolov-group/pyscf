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
# Author: Terrence Stahl <terrencestahl@gmail.com>
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

"""Periodic ADC MO transformations, integral storage and DF reconstruction."""
from itertools import product
import time

import numpy as np

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc.adc.lib import einsum
from pyscf.pbc.df import df


class _IntegralBlocks:
    """Index three leading k axes without expanding the full tensor."""

    def __init__(self, eris, name, read):
        self._eris = eris
        self._read = read
        sizes = {'o': eris.nocc, 'v': eris.nvir}
        self.shape = (eris.nkpts,) * 3 + tuple(sizes[x] for x in name)
        self.dtype = eris.dtype
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if len(key) < 3 or not all(isinstance(k, (int, np.integer))
                                   for k in key[:3]):
            raise IndexError('Select three k-point indices before orbital '
                             'slices; full-tensor expansion is not supported.')
        nkpts = self._eris.nkpts
        if not all(-nkpts <= k < nkpts for k in key[:3]):
            raise IndexError('K-point index out of range.')
        block = self._read(*(k % nkpts for k in key[:3]))
        return block[key[3:]] if len(key) > 3 else block


class _ChemistsERIs:
    """Own in-memory arrays or a single HDF5 scratch file and DF factors."""

    def __init__(self, adc):
        self.nocc = adc.nocc
        self.nvir = adc.nmo - adc.nocc
        self.nkpts = len(adc.kpts)
        self.kconserv = adc.khelper.kconserv
        self.dtype = np.result_type(*[c.dtype for c in adc.mo_coeff])
        self.feri = None  # Integral scratch file.
        self.Lov = self.Lvv = self.Lpq_mo = None
        for name in ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo', 'ovvv', 'vvvv'):
            setattr(self, name, None)

    def allocate(self, names, outcore=False):
        """Allocate only the requested blocks, with consistent shapes."""
        if outcore and self.feri is None:
            self.feri = lib.H5TmpFile()
        sizes = {'o': self.nocc, 'v': self.nvir}
        for name in names:
            shape = (self.nkpts,) * 3 + tuple(sizes[x] for x in name)
            if outcore:
                block = self.feri.create_dataset(name, shape, dtype=self.dtype)
            else:
                block = np.empty(shape, dtype=self.dtype)
            setattr(self, name, block)
        return self

    @staticmethod
    def ovvv_df(Lov, Lvv, start=0, chunk_size=None):
        """Reconstruct an occupied slice of (ov|vv) from DF factors.

        The result is unnormalized; k-point callers divide by ``nkpts``.
        Only the requested occupied slice is formed.
        """
        stop = Lov.shape[1] if chunk_size is None else start + chunk_size
        return einsum('Lpq,Lrs->pqrs', Lov[:, start:stop], Lvv)

    @staticmethod
    def vvvv_df(vv1, vv2, start=0, chunk_size=None):
        """Reconstruct a virtual slice in particle-particle ordering.

        The result is unnormalized; k-point callers divide by ``nkpts``.
        """
        stop = vv1.shape[1] if chunk_size is None else start + chunk_size
        return einsum('Lpq,Lrs->prqs', vv1[:, start:stop], vv2)

    @property
    def ovvv_blocks(self):
        """Lazy (k_p,k_q,k_r) access to stored or DF-reconstructed ovvv."""
        return _IntegralBlocks(self, 'ovvv', self._get_ovvv)

    @property
    def vvvv_blocks(self):
        """Lazy (k_p,k_q,k_r) access in particle-particle ordering."""
        return _IntegralBlocks(self, 'vvvv', self._get_vvvv)

    def _get_ovvv(self, kp, kq, kr):
        if self.ovvv is not None:
            return self.ovvv[kp, kq, kr]
        ks = self.kconserv[kp, kq, kr]
        return self.ovvv_df(self.Lov[kp, kq], self.Lvv[kr, ks]) / self.nkpts

    def _get_vvvv(self, kp, kq, kr):
        if self.vvvv is not None:
            return self.vvvv[kp, kq, kr]
        ks = self.kconserv[kp, kr, kq]
        return self.vvvv_df(self.Lvv[kp, kr], self.Lvv[kq, ks]) / self.nkpts

    def get_ovvv(self, kp, kq, kr):
        """Compatibility method; use ``ovvv_blocks[kp, kq, kr]``."""
        return self.ovvv_blocks[kp, kq, kr]

    def get_vvvv(self, kp, kq, kr):
        """Compatibility method; use ``vvvv_blocks[kp, kq, kr]``."""
        return self.vvvv_blocks[kp, kq, kr]

    def close(self):
        """Release disk storage; repeated close calls are harmless."""
        if self.feri is not None:
            self.feri.close()
            self.feri = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _needs_vvvv(adc):
    return (adc.method == 'adc(3)'
            or adc.if_heri_eris
            or (adc.method == 'adc(2)-x'
                and (adc.type_ in ('ee', 'ea')
                     or not adc.approx_trans_moments))
            or (adc.type_ == 'ee' and not adc.approx_trans_moments))

class AO2MO:
    """Shared context for active-space periodic integral transformations.

    Args:
        adc (KRADC): Driver supplying orbitals, occupations and k points.
        backend (str, optional): ``auto``, ``incore``, ``outcore`` or
            ``density_fit``. ``df`` remains an accepted alias.

    Calling the initialized context returns an ERI container. Backend
    methods can also be selected explicitly without recalculating padding.

    Example:
        >>> context = AO2MO(kadc, backend='density_fit')
        >>> eris = context()
    """

    def __init__(self, adc, backend=None):
        self.adc = adc
        self.log = logger.Logger(adc.stdout, adc.verbose)
        self.kpts = adc.kpts
        self.nkpts = len(self.kpts)
        self.nocc, self.nvir, self.nmo = adc.nocc, adc.nvir, adc.nmo
        self.cell = adc.cell
        self.nao = self.cell.nao_nr()
        self.mo_coeff = adc.padded_coeff
        self.dtype = np.result_type(self.mo_coeff)
        self.khelper = adc.khelper
        self.kconserv = self.khelper.kconserv
        self.with_df = getattr(adc, 'with_df', None) or adc._scf.with_df
        self.fao2mo = self.with_df.ao2mo
        self.orbv = np.asarray(self.mo_coeff[:, :, self.nocc:], order='C')
        backend = backend or adc.integral_backend
        match backend:
            case 'auto':
                if isinstance(self.with_df, df.GDF):
                    backend = 'density_fit'
                else:
                    required = self.nkpts**3 * self.nmo**4 * 64 / 1e6
                    available = adc.max_memory - lib.current_memory()[0]
                    backend = ('incore' if adc.incore_complete
                               or required < available else 'outcore')
            case 'df':
                backend = 'density_fit'
            case 'incore' | 'outcore' | 'density_fit':
                pass
            case _:
                raise ValueError(f'Unknown ADC integral backend: {backend!r}')
        self.backend = backend

    def __call__(self):
        """Execute the selected transformation and return _ChemistsERIs."""
        match self.backend:
            case 'incore':
                return self.incore()
            case 'outcore':
                return self.outcore()
            case 'density_fit':
                return self.density_fit()

    def incore(self):
        """Transform symmetry-related k-point blocks into memory.

        Returns:
            _ChemistsERIs: The active-space integral container.
        """
        eris = _ChemistsERIs(self.adc)
        self.log.info('Using incore ERI storage')
        eris.allocate(('oooo', 'oovv', 'ovoo', 'ovov', 'ovvv', 'ovvo'))

        for (ikp,ikq,ikr) in self.khelper.symm_map:
            iks = self.kconserv[ikp,ikq,ikr]
            eri_kpt = self.fao2mo(
                (
                    self.mo_coeff[ikp],
                    self.mo_coeff[ikq],
                    self.mo_coeff[ikr],
                    self.mo_coeff[iks],
                ),
                (
                    self.kpts[ikp],
                    self.kpts[ikq],
                    self.kpts[ikr],
                    self.kpts[iks],
                ),
                compact=False,
            )
            if self.dtype == np.float64:
                eri_kpt = eri_kpt.real
            eri_kpt = eri_kpt.reshape(self.nmo, self.nmo, self.nmo, self.nmo)
            for (kp, kq, kr) in self.khelper.symm_map[(ikp, ikq, ikr)]:
                eri_kpt_symm = self.khelper.transform_symm(eri_kpt, kp, kq, kr)
                eris.oooo[kp, kq, kr] = (
                    eri_kpt_symm[
                        :self.nocc,
                        :self.nocc,
                        :self.nocc,
                        :self.nocc,
                    ]
                    / self.nkpts
                )
                eris.oovv[kp, kq, kr] = (
                    eri_kpt_symm[
                        :self.nocc,
                        :self.nocc,
                        self.nocc:,
                        self.nocc:,
                    ]
                    / self.nkpts
                )
                eris.ovoo[kp, kq, kr] = (
                    eri_kpt_symm[
                        :self.nocc,
                        self.nocc:,
                        :self.nocc,
                        :self.nocc,
                    ]
                    / self.nkpts
                )
                eris.ovov[kp, kq, kr] = (
                    eri_kpt_symm[
                        :self.nocc,
                        self.nocc:,
                        :self.nocc,
                        self.nocc:,
                    ]
                    / self.nkpts
                )
                eris.ovvv[kp, kq, kr] = (
                    eri_kpt_symm[
                        :self.nocc,
                        self.nocc:,
                        self.nocc:,
                        self.nocc:,
                    ]
                    / self.nkpts
                )
                eris.ovvo[kp, kq, kr] = (
                    eri_kpt_symm[
                        :self.nocc,
                        self.nocc:,
                        self.nocc:,
                        :self.nocc,
                    ]
                    / self.nkpts
                )

        if _needs_vvvv(self.adc):
            eris.vvvv = self.with_df.ao2mo_7d(self.orbv,
                factor=1. / self.nkpts).transpose(0, 2, 1, 3, 5, 4, 6)

        return eris

    def outcore(self):
        """Transform integral blocks into one HDF5 scratch file.

        Returns:
            _ChemistsERIs: The active-space integral container.
        """
        eris = _ChemistsERIs(self.adc)
        eris.allocate(('oooo', 'oovv', 'ovoo', 'ovov', 'ovvv', 'ovvo'),
                      outcore=True)
        if _needs_vvvv(self.adc):
            eris.allocate(('vvvv',), outcore=True)

        cput1 = time.process_time(), time.time()
        for kp, kq, kr in product(range(self.nkpts), repeat=3):
            ks = self.kconserv[kp, kq, kr]
            orbo_p = self.mo_coeff[kp][:, :self.nocc]
            orbo_q = self.mo_coeff[kq][:, :self.nocc]
            buf_kpt = self.fao2mo(
                (orbo_p, orbo_q, self.mo_coeff[kr], self.mo_coeff[ks]),
                (self.kpts[kp], self.kpts[kq], self.kpts[kr], self.kpts[ks]),
                compact=False,
            )
            if self.mo_coeff[0].dtype == np.float64:
                buf_kpt = buf_kpt.real
            buf_kpt = buf_kpt.reshape(self.nocc, self.nocc, self.nmo, self.nmo)
            self.dtype = buf_kpt.dtype
            eris.oooo[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, :self.nocc,
                :self.nocc] / self.nkpts
            eris.oovv[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, self.nocc:,
                self.nocc:] / self.nkpts
        cput1 = self.log.timer_debug1('transforming oopq', *cput1)

        # <ia|pq> = (ip|aq)
        cput1 = time.process_time(), time.time()
        for kp, kq, kr in product(range(self.nkpts), repeat=3):
            ks = self.kconserv[kp, kq, kr]
            orbo_p = self.mo_coeff[kp][:, :self.nocc]
            orbv_q = self.mo_coeff[kq][:, self.nocc:]
            buf_kpt = self.fao2mo(
                (orbo_p, orbv_q, self.mo_coeff[kr], self.mo_coeff[ks]),
                (self.kpts[kp], self.kpts[kq], self.kpts[kr], self.kpts[ks]),
                compact=False,
            )
            if self.mo_coeff[0].dtype == np.float64:
                buf_kpt = buf_kpt.real
            buf_kpt = buf_kpt.reshape(self.nocc, self.nvir, self.nmo, self.nmo)
            eris.ovoo[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, :self.nocc,
                :self.nocc] / self.nkpts
            eris.ovov[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, :self.nocc,
                self.nocc:] / self.nkpts
            eris.ovvo[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, self.nocc:,
                :self.nocc] / self.nkpts
            eris.ovvv[kp, kq, kr, :, :, :, :] = buf_kpt[:, :, self.nocc:,
                self.nocc:] / self.nkpts
        cput1 = self.log.timer_debug1('transforming ovpq', *cput1)

        if _needs_vvvv(self.adc):
            mem_now = lib.current_memory()[0]
            if self.nvir ** 4 * 16 / 1e6 + mem_now < self.adc.max_memory:
                for (ikp, ikq, ikr) in self.khelper.symm_map:
                    iks = self.kconserv[ikp, ikq, ikr]
                    orbv_p = self.mo_coeff[ikp][:, self.nocc:]
                    orbv_q = self.mo_coeff[ikq][:, self.nocc:]
                    orbv_r = self.mo_coeff[ikr][:, self.nocc:]
                    orbv_s = self.mo_coeff[iks][:, self.nocc:]
                    # unit cell is small enough to handle vvvv in-core
                    buf_kpt = self.fao2mo(
                        (orbv_p, orbv_q, orbv_r, orbv_s),
                        self.kpts[[ikp, ikq, ikr, iks]],
                        compact=False,
                    )
                    if self.dtype == np.float64:
                        buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(
                        (self.nvir, self.nvir, self.nvir, self.nvir),
                    )
                    for (kp, kq, kr) in self.khelper.symm_map[(ikp, ikq, ikr)]:
                        buf_kpt_symm = self.khelper.transform_symm(
                            buf_kpt,
                            kp,
                            kq,
                            kr,
                        ).transpose(
                            0,
                            2,
                            1,
                            3,
                        )
                        eris.vvvv[kp, kr, kq] = buf_kpt_symm / self.nkpts
            else:
                for (ikp, ikq, ikr) in self.khelper.symm_map:
                    iks = self.kconserv[ikp, ikq, ikr]
                    orbv_p = self.mo_coeff[ikp][:, self.nocc:]
                    orbv_q = self.mo_coeff[ikq][:, self.nocc:]
                    orbv_r = self.mo_coeff[ikr][:, self.nocc:]
                    orbv_s = self.mo_coeff[iks][:, self.nocc:]
                    for a in range(self.nvir):
                        orbva_p = orbv_p[:, a].reshape(-1, 1)
                        buf_kpt = self.fao2mo(
                            (orbva_p, orbv_q, orbv_r, orbv_s),
                            (
                                self.kpts[ikp],
                                self.kpts[ikq],
                                self.kpts[ikr],
                                self.kpts[iks],
                            ),
                            compact=False,
                        )
                        if self.mo_coeff[0].dtype == np.float64:
                            buf_kpt = buf_kpt.real
                        buf_kpt = buf_kpt.reshape((1, self.nvir, self.nvir,
                            self.nvir)).transpose(0, 2, 1, 3)

                        eris.vvvv[ikp, ikr, ikq, a, :, :, :] = buf_kpt[0, :, :,
                            :] / self.nkpts
                        # Store symmetric permutations
                        eris.vvvv[ikr, ikp, iks, :, a, :, :] = (
                            buf_kpt.transpose(1, 0, 3, 2)[:, 0, :, :]
                            / self.nkpts
                        )
                        eris.vvvv[ikq, iks, ikp, :, :, a, :] = (
                            buf_kpt.transpose(2, 3, 0, 1).conj()[:, :, 0, :]
                            / self.nkpts
                        )
                        eris.vvvv[iks, ikq, ikr, :, :, :, a] = (
                            buf_kpt.transpose(3, 2, 1, 0).conj()[:, :, :, 0]
                            / self.nkpts
                        )
                cput1 = self.log.timer_debug1('transforming vvvv', *cput1)

        return eris

    def density_fit(self):
        """Transform three-index DF factors and store occupied blocks.

        Returns:
            _ChemistsERIs: The active-space integral container.
        """
        if self.with_df._cderi is None:
            self.with_df.build()
        naux = self.with_df.get_naoaux()
        eris = _ChemistsERIs(self.adc)

        eris.Lpq_mo = Lpq_mo = np.empty((self.nkpts, self.nkpts), dtype=object)
        Loo = np.empty(
            (self.nkpts, self.nkpts, naux, self.nocc, self.nocc),
            dtype=self.dtype,
        )
        Lvo = np.empty(
            (self.nkpts, self.nkpts, naux, self.nvir, self.nocc),
            dtype=self.dtype,
        )
        eris.Lvv = np.empty(
            (self.nkpts, self.nkpts, naux, self.nvir, self.nvir),
            dtype=self.dtype,
        )
        eris.Lov = np.empty(
            (self.nkpts, self.nkpts, naux, self.nocc, self.nvir),
            dtype=self.dtype,
        )

        eris.vvvv = None
        eris.ovvv = None

        with df._load3c(self.with_df._cderi, 'j3c') as fload:
            tao = []
            ao_loc = None
            for ki, kpti in enumerate(self.kpts):
                for kj, kptj in enumerate(self.kpts):
                    Lpq_ao = np.asarray(fload(kpti, kptj))

                    mo = np.hstack((self.mo_coeff[ki], self.mo_coeff[kj]))
                    mo = np.asarray(mo, dtype=self.dtype, order='F')
                    if self.dtype == np.double:
                        out = _ao2mo.nr_e2(
                            Lpq_ao,
                            mo,
                            (0, self.nmo, self.nmo, self.nmo + self.nmo),
                            aosym='s2',
                        )
                    else:
                        # Auxiliary linear dependence can reduce Lpq.shape[0].
                        if Lpq_ao[0].size != self.nao**2:  # aosym = 's2'
                            Lpq_ao = lib.unpack_tril(Lpq_ao).astype(
                                np.complex128,
                            )
                        out = _ao2mo.r_e2(
                            Lpq_ao,
                            mo,
                            (0, self.nmo, self.nmo, self.nmo + self.nmo),
                            tao,
                            ao_loc,
                        )
                    Lpq_mo[ki, kj] = out.reshape(-1, self.nmo, self.nmo)

                    Loo[ki, kj] = Lpq_mo[ki, kj][:, :self.nocc, :self.nocc]
                    eris.Lov[ki, kj] = Lpq_mo[ki, kj][
                        :,
                        :self.nocc,
                        self.nocc:,
                    ]
                    Lvo[ki, kj] = Lpq_mo[ki, kj][:, self.nocc:, :self.nocc]
                    eris.Lvv[ki, kj] = Lpq_mo[ki, kj][
                        :,
                        self.nocc:,
                        self.nocc:,
                    ]

        if self.adc.if_naf:
            Loo_big = Loo.transpose(2, 0, 1, 3, 4).reshape(naux, -1)
            Lov_big = eris.Lov.transpose(2, 0, 1, 3, 4).reshape(naux, -1)
            Lvo_big = Lvo.transpose(2, 0, 1, 3, 4).reshape(naux, -1)
            Lvv_big = eris.Lvv.transpose(2, 0, 1, 3, 4).reshape(naux, -1)
            L_big = np.concatenate(
                [Loo_big, Lov_big, Lvo_big, Lvv_big],
                axis=1,
            )
            W = L_big.dot(L_big.T.conj())
            n, N = np.linalg.eigh(W / self.nkpts)
            N_trunc = N[:, n > self.adc.thresh_naf].T
            self.adc.naux = N_trunc.shape[0]
            self.log.info(
                f'origin naux is {naux}||naf naux is {self.adc.naux}',
            )
            Loo = einsum('rL,mnLsj->mnrsj', N_trunc, Loo)
            eris.Lov = einsum('rL,mnLsj->mnrsj', N_trunc, eris.Lov)
            Lvo = einsum('rL,mnLsj->mnrsj', N_trunc, Lvo)
            eris.Lvv = einsum('rL,mnLsj->mnrsj', N_trunc, eris.Lvv)

            del (Loo_big)
            del (Lov_big)
            del (Lvo_big)
            del (Lvv_big)
            del (L_big)
            del (W)
            del (N)
            del (n)
            del (N_trunc)

        eris.allocate(('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo'), outcore=True)

        for kp, kq, kr in product(range(self.nkpts), repeat=3):
            ks = self.kconserv[kp, kq, kr]
            eris.oooo[kp, kq, kr] = einsum('Lpq,Lrs->pqrs', Loo[kp, kq],
                Loo[kr, ks]) / self.nkpts
            eris.oovv[kp, kq, kr] = einsum('Lpq,Lrs->pqrs', Loo[kp, kq],
                eris.Lvv[kr, ks]) / self.nkpts
            eris.ovoo[kp, kq, kr] = einsum('Lpq,Lrs->pqrs', eris.Lov[kp,
                kq], Loo[kr, ks]) / self.nkpts
            eris.ovov[kp, kq, kr] = einsum(
                'Lpq,Lrs->pqrs', eris.Lov[kp, kq], eris.Lov[kr,
                    ks]) / self.nkpts
            eris.ovvo[kp, kq, kr] = einsum('Lpq,Lrs->pqrs', eris.Lov[kp,
                kq], Lvo[kr, ks]) / self.nkpts
                    # eris.ovvv[kp,kq,kr] = einsum('Lpq,Lrs->pqrs',
                    # eris.Lov[kp,kq], Lvv[kr,ks])/nkpts

        return eris

def chunk_size(adc):
    """Estimate a bounded orbital chunk from available memory.

    The largest reconstructed block contains ``chunk_size * nvir**3``
    elements. Complex orbitals use their actual element size.

    Args:
        adc (KRADC): Driver with active orbital dimensions and memory limit.

    Returns:
        int: At least one orbital, never more than the active virtual count.
    """
    nvir = adc.nvir
    if nvir < 1:
        return 1
    itemsize = np.result_type(*[c.dtype for c in adc.mo_coeff]).itemsize
    available = max(0, adc.max_memory - lib.current_memory()[0]) * 0.5
    per_orbital = nvir**3 * itemsize / 1e6
    return max(1, min(nvir, int(available / per_orbital)))


def ao2mo(adc, backend=None):
    """Transform active MOs and return an owning _ChemistsERIs container.

    See :class:`AO2MO` for backend choices and shared transformation state.
    """
    return AO2MO(adc, backend)()


# Legacy names retained for external callers; production code uses ao2mo.
transform_integrals = ao2mo
calculate_chunk_size = chunk_size


def transform_integrals_incore(adc):
    return AO2MO(adc, 'incore').incore()


def transform_integrals_outcore(adc):
    return AO2MO(adc, 'outcore').outcore()


def transform_integrals_df(adc):
    return AO2MO(adc, 'density_fit').density_fit()
