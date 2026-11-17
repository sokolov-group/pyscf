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

r"""Periodic ADC integral transformations, storage and density fitting.

Four-index blocks retain three independent k axes. In chemists' ordering,
``kconserv[kp, kq, kr]`` selects ks for kp - kq + kr - ks = G. Stored
integrals include the factor 1/nkpts. The vvvv block instead groups the
particle pairs and has ks = kp - kr + kq, modulo a reciprocal vector.

Numerical examples include their imports and construct the small
two-k-point helium reference locally. All shapes include active-space
padding. An integral container owns its scratch file and supports ``with``.
"""
from itertools import product
import time

import numpy as np

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc.adc.lib import einsum
from pyscf.pbc.df import df


class _IntegralBlocks:
    r"""Provide array-like access to individual stored or reconstructed ERI
    blocks.

    The accessor describes a seven-dimensional tensor but reads only one
    four-index orbital block at a time. It never assembles the full tensor.

    Args:
        eris (_ChemistsERIs): Container supplying dimensions, dtype and
            storage.
        name (str): Four orbital-space labels, such as 'ovvv' or 'vvvv'.
        read (callable): Function read(kp, kq, kr) returning an orbital block.

    Attributes:
        shape (tuple[int]): Three k dimensions followed by four orbital sizes.
        dtype (numpy.dtype): Integral element type, including complex orbitals.
        ndim (int): Number of tensor axes, currently seven.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
        >>> with mykradcbase.ao2mo() as eris:
        ...     print(tuple(map(int, eris.ovvv_blocks.shape)))
        ...     print(eris.ovvv_blocks[0, 0, 0].shape)
        (2, 2, 2, 1, 1, 1, 1)
        (1, 1, 1, 1)
    """

    def __init__(self, eris, name, read):
        r"""Bind the block reader and expose the corresponding tensor metadata.

        Args:
            eris (_ChemistsERIs): Owning container; it must outlive this
                accessor.
            name (str): Orbital-space labels determining the last four
                dimensions.
            read (callable): Reader accepting three integer k indices.

        Returns:
            None: Initialize shape, dtype, ndim and the retained reader.
        """
        self._eris = eris
        self._read = read
        sizes = {'o': eris.nocc, 'v': eris.nvir}
        self.shape = (eris.nkpts,) * 3 + tuple(sizes[x] for x in name)
        self.dtype = eris.dtype
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        r"""Read one k block and apply any trailing NumPy orbital indices.

        Args:
            key (tuple): Three integer k indices followed by optional orbital
                indices or slices. Negative k indices wrap as for NumPy arrays.

        Returns:
            ndarray or scalar: Selected data from the four-index block. Stored
            HDF5 data are read into memory; DF data are reconstructed on
            demand.

        Raises:
            IndexError: Fewer than three k indices, noninteger k selectors, or
                indices outside [-nkpts, nkpts). Slicing the full k tensor is
                deliberately unsupported.
        """
        key = key if isinstance(key, tuple) else (key,)
        if len(key) < 3 or not all(
            isinstance(k, (int, np.integer)) for k in key[:3]):
            raise IndexError('Select three k-point indices before orbital '
                             'slices; full-tensor expansion is not supported.')
        nkpts = self._eris.nkpts
        if not all(-nkpts <= k < nkpts for k in key[:3]):
            raise IndexError('K-point index out of range.')
        block = self._read(*(k % nkpts for k in key[:3]))
        return block[key[3:]] if len(key) > 3 else block


class _ChemistsERIs:
    r"""Own periodic active-space MO integrals and their scratch storage.

    Args:
        adc (KRADCBase): Driver defining active dimensions, orbitals and
            momenta.

    Attributes:
        nocc, nvir, nkpts (int): Padded active occupied/virtual and k counts.
        kconserv (ndarray): Momentum-conservation lookup of shape (nkpts,)*3.
        dtype (numpy.dtype): Element type inferred from the MO coefficients.
        oooo, oovv, ovoo, ovov, ovvo: Stored arrays or HDF5 datasets with three
            leading k axes and the orbital dimensions indicated by each name.
        ovvv, vvvv: Stored four-index blocks, or None for DF reconstruction.
            vvvv uses particle-particle ordering rather than chemists'
            ordering.
        Lov, Lvv: DF factors of shape (nkpts, nkpts, naux, nocc/nvir, nvir).
            They are None until a DF transformation is performed.
        Lpq_mo: Object array indexed by a k pair, containing full MO DF blocks.
        feri: Owning HDF5 scratch file, or None for entirely in-memory storage.

    Raw stored blocks and the lazy ovvv_blocks/vvvv_blocks accessors include
    1/nkpts normalization. The static DF helpers return unnormalized blocks.
    Use close() or a context manager to release scratch storage.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
        >>> with mykradcbase.ao2mo('density_fit') as eris:
        ...     print(eris.ovov.shape, eris.ovvv is None)
        ...     print(eris.ovvv_blocks[0, 0, 0].shape)
        (2, 2, 2, 1, 1, 1, 1) True
        (1, 1, 1, 1)
    """

    def __init__(self, adc):
        r"""Initialize integral metadata without allocating any integral
        tensors.

        Args:
            adc (KRADCBase): Reference supplying orbital counts, k points and
                dtype.

        Returns:
            None: All integral and DF-factor attributes initially contain None.
            allocate() or an AO2MO backend populates the required storage
            later.
        """
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
        r"""Allocate selected four-index blocks in memory or in one scratch
        file.

        Args:
            names (sequence[str]): Block names made from four o/v labels, such
                as ('oooo', 'ovov'). Each block has shape (nkpts,)*3 plus its
                four orbital dimensions; allocation does not initialize its
                values.
            outcore (bool): Create HDF5 datasets in feri when True; otherwise
                use NumPy arrays. Reuse the same scratch file for subsequent
                blocks.

        Returns:
            _ChemistsERIs: This container, allowing chained initialization.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
            >>> with _ChemistsERIs(mykradcbase) as eris:
            ...     allocated_integrals = eris.allocate(
            ...         ('ovov',), outcore=True)
            ...     print(eris.ovov.shape)
            (2, 2, 2, 1, 1, 1, 1)
        """
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
        r"""Reconstruct an occupied slice of a chemists' (ov|vv) DF block.

        The contraction is :math:`V_{pqrs}=\sum_L L^{ov}_{Lpq}L^{vv}_{Lrs}`.
        The caller supplies the appropriate k-pair factors and their
        conjugation
        convention; this helper introduces no extra complex conjugation.

        Args:
            Lov (ndarray): Occupied-virtual factors, shape (naux, nocc, nvir).
            Lvv (ndarray): Virtual-virtual factors, shape (naux, nvir, nvir).
            start (int): First occupied index to reconstruct, default zero.
            chunk_size (int, optional): Number of occupied indices. With None,
                continue through the end of Lov's occupied axis.

        Returns:
            ndarray: Shape (selected_nocc, nvir, nvir, nvir). The result is
            unnormalized; periodic callers divide by nkpts.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs
            >>> Lov, Lvv = np.ones((2, 3, 4)), np.ones((2, 4, 4))
            >>> block = _ChemistsERIs.ovvv_df(Lov, Lvv, start=1, chunk_size=1)
            >>> block.shape, float(block[0, 0, 0, 0])
            ((1, 4, 4, 4), 2.0)
        """
        stop = Lov.shape[1] if chunk_size is None else start + chunk_size
        return einsum('Lpq,Lrs->pqrs', Lov[:, start:stop], Lvv)

    @staticmethod
    def vvvv_df(vv1, vv2, start=0, chunk_size=None):
        r"""Reconstruct a virtual slice in particle-particle ordering.

        The result is :math:`V_{prqs}=\sum_L L^{(1)}_{Lpq}L^{(2)}_{Lrs}`.
        The first orbital axis is sliced before contraction. No conjugation or
        k-point normalization is introduced by this helper.

        Args:
            vv1, vv2 (ndarray): DF virtual-pair factors, each shaped
                (naux, nvir, nvir), for the required k pairs.
            start (int): First index along vv1's first virtual axis.
            chunk_size (int, optional): Slice length; None selects to the end.

        Returns:
            ndarray: Shape (selected_nvir, nvir, nvir, nvir), in p,r,q,s order.
            Periodic callers divide this block by nkpts.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs
            >>> factors = np.ones((2, 3, 3))
            >>> block = _ChemistsERIs.vvvv_df(factors, factors, 1, 1)
            >>> block.shape, float(block[0, 0, 0, 0])
            ((1, 3, 3, 3), 2.0)
        """
        stop = vv1.shape[1] if chunk_size is None else start + chunk_size
        return einsum('Lpq,Lrs->prqs', vv1[:, start:stop], vv2)

    @property
    def ovvv_blocks(self):
        r"""Access stored or DF-reconstructed ovvv through three k indices.

        Returns:
            _IntegralBlocks: Accessor with logical shape
            (nkpts, nkpts, nkpts, nocc, nvir, nvir, nvir). Index as
            ``ovvv_blocks[kp, kq, kr, ...]``; the fourth k point follows
            chemists'
            momentum conservation. Reading one block does not fill raw ovvv.
        """
        return _IntegralBlocks(self, 'ovvv', self._get_ovvv)

    @property
    def vvvv_blocks(self):
        r"""Access vvvv in particle-particle ordering without full expansion.

        Returns:
            _IntegralBlocks: Accessor with logical shape (nkpts,)*3 +
            (nvir,)*4.
            Index as ``vvvv_blocks[kp, kq, kr, ...]``. The fourth momentum is
            kconserv[kp, kr, kq]; the returned orbital order is p,r,q,s
            relative
            to the DF pair product. Reading a block does not fill raw vvvv.
        """
        return _IntegralBlocks(self, 'vvvv', self._get_vvvv)

    def _get_ovvv(self, kp, kq, kr):
        r"""Read one normalized ovvv block from storage or DF factors.

        Args:
            kp, kq, kr (int): Chemists' leading k indices. ks is
                kconserv[kp, kq, kr]. Public access validates indices first.

        Returns:
            ndarray: Orbital block of shape (nocc, nvir, nvir, nvir). Stored
            blocks are already normalized; reconstructed blocks are divided by
            nkpts. DF factors must be present if raw ovvv is None.
        """
        if self.ovvv is not None:
            return self.ovvv[kp, kq, kr]
        ks = self.kconserv[kp, kq, kr]
        return self.ovvv_df(self.Lov[kp, kq], self.Lvv[kr, ks]) / self.nkpts

    def _get_vvvv(self, kp, kq, kr):
        r"""Read one normalized particle-particle vvvv block.

        Args:
            kp, kq, kr (int): Stored particle-pair k indices. ks is
                kconserv[kp, kr, kq]. Public access validates indices first.

        Returns:
            ndarray: Block of shape (nvir,)*4. Return stored data when
            available;
            otherwise contract the Lvv[kp, kr] and Lvv[kq, ks] factors and
            divide
            by nkpts.
        """
        if self.vvvv is not None:
            return self.vvvv[kp, kq, kr]
        ks = self.kconserv[kp, kr, kq]
        return self.vvvv_df(self.Lvv[kp, kr], self.Lvv[kq, ks]) / self.nkpts

    def close(self):
        r"""Close the owned integral scratch file and clear its file reference.

        Returns:
            None: Repeated calls are harmless. NumPy arrays and DF factors
            remain
            available, but HDF5 datasets belonging to the closed file must no
            longer be accessed. Containers do not own the ADC amplitude files.
        """
        if self.feri is not None:
            self.feri.close()
            self.feri = None

    def __enter__(self):
        r"""Enter a context that releases integral scratch storage on exit.

        Returns:
            _ChemistsERIs: This container. Integrals may be read or allocated
            inside the context; __exit__ calls close() even if an exception
            occurs.
        """
        return self

    def __exit__(self, *exc):
        r"""Release integral scratch storage when leaving a context.

        Args:
            *exc: Exception type, exception instance and traceback supplied by
                Python's context-manager protocol, or three None values.

        Returns:
            None: Any exception from the context body propagates to the caller.
        """
        self.close()


def _needs_vvvv(adc):
    r"""Decide whether an explicit four-virtual block is needed by a backend.

    Args:
        adc (KRADCBase): Driver supplying method, excitation type,
            approx_trans_moments and if_heri_eris.

    Returns:
        bool: True for ADC(3), explicit retained integrals, and the extended
        or full-transition-moment cases requiring virtual ladders. DF storage
        can supply these ladders from factors instead of an explicit tensor.
    """
    from pyscf.pbc.adc.kadc_rhf import KRADCEA, KRADCEE

    return (adc.method == 'adc(3)'
            or adc.if_heri_eris
            or (adc.method == 'adc(2)-x'
                and (isinstance(adc, (KRADCEA, KRADCEE))
                     or not adc.approx_trans_moments))
            or (isinstance(adc, KRADCEE) and not adc.approx_trans_moments))

class AO2MO:
    r"""Transform active Bloch MOs using a shared backend-selection context.

    Args:
        adc (KRADCBase): Driver supplying padded orbitals, active dimensions,
            k-point helper, memory settings and an SCF DF transformation
            object.
        backend (str, optional): 'auto', 'incore', 'outcore' or 'density_fit'.
            None uses adc.integral_backend; 'df' aliases 'density_fit'. Auto
            selects GDF factor transformation when available, otherwise uses
            an integral-memory estimate to choose incore or outcore storage.

    Attributes:
        adc (KRADCBase): Original reference driver.
        backend (str): Resolved backend; initializing the context does not run
            it.
        mo_coeff (ndarray): Padded active coefficients, shape (nkpts, nao,
            nmo).
        nocc, nvir, nmo, nao, nkpts (int): Active-space and basis dimensions.
        kpts, kconserv: K mesh and its momentum-conservation lookup.
        dtype (numpy.dtype): Element type inferred from the padded
            coefficients.
        with_df: Periodic DF object used for AO transformations.
        fao2mo (callable): AO-to-MO transformation supplied by with_df.
        orbv (ndarray): Active virtual coefficients, shape (nkpts, nao, nvir).

    Calling the context, or one of its backend methods, returns an owning
    _ChemistsERIs container. Four-index stored blocks carry 1/nkpts factors.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.kadc_ao2mo import AO2MO
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
        >>> context = AO2MO(mykradcbase, backend='density_fit')
        >>> context.backend, context.mo_coeff.shape
        ('density_fit', (2, 2, 2))
        >>> with context() as eris:
        ...     print(eris.ovov.shape)
        (2, 2, 2, 1, 1, 1, 1)
    """

    def __init__(self, adc, backend=None):
        r"""Cache active-space transformation inputs and resolve backend
        selection.

        Args:
            adc (KRADCBase): Source of padded coefficients, k-point and memory
                data.
            backend (str, optional): Backend override; see AO2MO for choices.

        Returns:
            None: Populate context attributes without allocating four-index
            ERIs.

        Raises:
            ValueError: The requested backend name is not recognized.
        """
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
        r"""Execute the resolved incore, outcore or density_fit backend.

        Returns:
            _ChemistsERIs: Newly transformed active-space integrals. The caller
            owns the returned container and should close its scratch file after
            use, preferably with a context manager. See the AO2MO class
            example.
        """
        match self.backend:
            case 'incore':
                return self.incore()
            case 'outcore':
                return self.outcore()
            case 'density_fit':
                return self.density_fit()

    def incore(self):
        r"""Transform symmetry-related four-index k blocks into NumPy storage.

        The k-point helper supplies symmetry-related permutations and complex
        conjugations. All occupied blocks and ovvv are retained in memory; vvvv
        is constructed in particle-particle ordering when _needs_vvvv is True.

        Returns:
            _ChemistsERIs: Active integrals normalized by nkpts, with no
            integral
            scratch file. Memory use scales with the stored three-k tensors.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_ao2mo import AO2MO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
            >>> context = AO2MO(mykradcbase, 'incore')
            >>> with context.incore() as eris:
            ...     print(isinstance(eris.ovov, np.ndarray), eris.feri is None)
            True True
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
            eris.vvvv = self.with_df.ao2mo_7d(
                self.orbv, factor=1.0 / self.nkpts
            ).transpose(0, 2, 1, 3, 5, 4, 6)

        return eris

    def outcore(self):
        r"""Transform four-index blocks into a single owned HDF5 scratch file.

        Build occupied and mixed blocks for each independent k triple, then
        form
        vvvv in virtual chunks if required. The temporary orbital blocks are
        smaller than the full three-k tensor, although each block remains
        dense.

        Returns:
            _ChemistsERIs: Normalized active integrals backed by HDF5 datasets.
            The container owns the scratch file and must remain open while its
            datasets are in use.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_ao2mo import AO2MO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
            >>> context = AO2MO(mykradcbase, 'outcore')
            >>> with context.outcore() as eris:
            ...     print(eris.ovov.shape, eris.feri is not None)
            (2, 2, 2, 1, 1, 1, 1) True
        """
        eris = _ChemistsERIs(self.adc)
        eris.allocate(
            ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvv', 'ovvo'), outcore=True)
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
                compact=False)
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
                compact=False)
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
                        compact=False)
                    if self.dtype == np.float64:
                        buf_kpt = buf_kpt.real
                    buf_kpt = buf_kpt.reshape(
                        (self.nvir, self.nvir, self.nvir, self.nvir))
                    for (kp, kq, kr) in self.khelper.symm_map[(ikp, ikq, ikr)]:
                        buf_kpt_symm = self.khelper.transform_symm(
                            buf_kpt, kp, kq, kr).transpose(0, 2, 1, 3)
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
                        buf_kpt = buf_kpt.reshape(
                            (1, self.nvir, self.nvir, self.nvir)
                        ).transpose(0, 2, 1, 3)

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
        r"""Transform three-index GDF factors and store occupied four-index
        blocks.

        For each k pair, transform the auxiliary AO factors into padded active
        MOs. Contract oooo, oovv, ovoo, ovov and ovvo into one HDF5 file. Keep
        Lov/Lvv factors for lazy ovvv/vvvv access instead of expanding those
        tensors. When if_naf is set, project the auxiliary factors using the
        natural-auxiliary threshold thresh_naf before forming these blocks.

        Returns:
            _ChemistsERIs: Stored blocks normalized by nkpts and unnormalized
            DF
            factors. Raw ovvv/vvvv are None; their block accessors reconstruct
            normalized data on demand. The backend requires a periodic GDF
            object
            with three-center integral storage.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_ao2mo import AO2MO
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
            >>> context = AO2MO(mykradcbase, 'density_fit')
            >>> with context.density_fit() as eris:
            ...     print(eris.ovvv is None, eris.vvvv is None)
            ...     print(eris.Lov.shape[:2], eris.Lov.shape[-2:])
            True True
            (2, 2) (1, 1)
        """
        if self.with_df._cderi is None:
            self.with_df.build()
        naux = self.with_df.get_naoaux()
        eris = _ChemistsERIs(self.adc)

        eris.Lpq_mo = Lpq_mo = np.empty((self.nkpts, self.nkpts), dtype=object)
        Loo = np.empty(
            (self.nkpts, self.nkpts, naux, self.nocc, self.nocc),
            dtype=self.dtype)
        Lvo = np.empty(
            (self.nkpts, self.nkpts, naux, self.nvir, self.nocc),
            dtype=self.dtype)
        eris.Lvv = np.empty(
            (self.nkpts, self.nkpts, naux, self.nvir, self.nvir),
            dtype=self.dtype)
        eris.Lov = np.empty(
            (self.nkpts, self.nkpts, naux, self.nocc, self.nvir),
            dtype=self.dtype)

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
                            aosym='s2')
                    else:
                        # Auxiliary linear dependence can reduce Lpq.shape[0].
                        if Lpq_ao[0].size != self.nao**2:  # aosym = 's2'
                            Lpq_ao = lib.unpack_tril(Lpq_ao).astype(
                                np.complex128)
                        out = _ao2mo.r_e2(
                            Lpq_ao,
                            mo,
                            (0, self.nmo, self.nmo, self.nmo + self.nmo),
                            tao,
                            ao_loc)
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
                [Loo_big, Lov_big, Lvo_big, Lvv_big], axis=1)
            W = L_big.dot(L_big.T.conj())
            n, N = np.linalg.eigh(W / self.nkpts)
            N_trunc = N[:, n > self.adc.thresh_naf].T
            self.adc.naux = N_trunc.shape[0]
            self.log.info(
                f'origin naux is {naux}||naf naux is {self.adc.naux}')
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
            eris.oooo[kp, kq, kr] = einsum(
                'Lpq,Lrs->pqrs',
                Loo[kp, kq], Loo[kr, ks]) / self.nkpts
            eris.oovv[kp, kq, kr] = einsum(
                'Lpq,Lrs->pqrs',
                Loo[kp, kq], eris.Lvv[kr, ks]) / self.nkpts
            eris.ovoo[kp, kq, kr] = einsum(
                'Lpq,Lrs->pqrs',
                eris.Lov[kp, kq], Loo[kr, ks]) / self.nkpts
            eris.ovov[kp, kq, kr] = einsum(
                'Lpq,Lrs->pqrs',
                eris.Lov[kp, kq], eris.Lov[kr, ks]) / self.nkpts
            eris.ovvo[kp, kq, kr] = einsum(
                'Lpq,Lrs->pqrs',
                eris.Lov[kp, kq], Lvo[kr, ks]) / self.nkpts
                    # eris.ovvv[kp,kq,kr] = einsum('Lpq,Lrs->pqrs',
                    # eris.Lov[kp,kq], Lvv[kr,ks])/nkpts

        return eris

def chunk_size(adc):
    r"""Estimate an orbital chunk from the remaining memory budget.

    A reconstructed virtual block needs approximately
    :math:`n_{chunk} n_{vir}^3` elements. Use half the positive difference
    between max_memory and current PySCF memory usage, with the actual real
    or complex element size. This is an estimate rather than a memory limit.

    Args:
        adc (KRADCBase): Driver defining nvir, mo_coeff and max_memory in MB.

    Returns:
        int: At least one orbital and at most nvir when nvir is positive.
        With no virtual orbitals, return one to keep range steps valid.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc.adc.kadc_ao2mo import chunk_size
        >>> from types import SimpleNamespace
        >>> model = SimpleNamespace(
        ...     nvir=3, max_memory=0, mo_coeff=[np.eye(4, dtype=complex)])
        >>> chunk_size(model)
        1
    """
    nvir = adc.nvir
    if nvir < 1:
        return 1
    itemsize = np.result_type(*[c.dtype for c in adc.mo_coeff]).itemsize
    available = max(0, adc.max_memory - lib.current_memory()[0]) * 0.5
    per_orbital = nvir**3 * itemsize / 1e6
    return max(1, min(nvir, int(available / per_orbital)))


def ao2mo(adc, backend=None):
    r"""Transform active MOs and return an owning integral container.

    Args:
        adc (KRADCBase): Reference defining orbitals, active spaces and
            momenta.
        backend (str, optional): Override adc.integral_backend for this call.
            See AO2MO for backend selection and storage conventions.

    Returns:
        _ChemistsERIs: Normalized active integrals and any DF factors or
        scratch files. Use a context manager to release scratch storage.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.kadc_ao2mo import ao2mo
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = kadc_rhf.KRADCBase(kmf)
        >>> with ao2mo(mykradcbase, 'density_fit') as eris:
        ...     print(eris.ovvo.shape)
        (2, 2, 2, 1, 1, 1, 1)
    """
    return AO2MO(adc, backend)()
