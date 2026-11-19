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

import numpy as np

from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.pbc.adc.lib import einsum
from pyscf.pbc.df import FFTDF, df


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

    Methods:
        __getitem__: Read one momentum block and optional orbital slices.
        __repr__: Preview the first block with NumPy print options.

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
        ...     print(tuple(map(int, eris.ovvv_kpoint_normalized.shape)))
        ...     print(eris.ovvv_kpoint_normalized[0, 0, 0].shape)
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
        self._name = name
        self._read = read
        sizes = {'o': eris.nocc, 'v': eris.nvir}
        self.shape = (eris.nkpts,) * 3 + tuple(sizes[x] for x in name)
        self.dtype = eris.dtype
        self.ndim = len(self.shape)

    def __repr__(self):
        """Preview the first k block using the active NumPy print options."""
        header = (f'{type(self).__name__}({self._name}, shape={self.shape}, '
                  f'dtype={self.dtype}')
        if getattr(self._eris, self._name) is None and self._eris.Lvv is None:
            return f'{header}, unavailable)'
        return f'{header}, block[0, 0, 0]=\n{np.array2string(self[0, 0, 0])})'

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
        # Resolve the k triple before applying any orbital-axis slices.
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

    Raw stored blocks and the lazy ovvv_kpoint_normalized and
    vvvv_kpoint_normalized accessors include 1/nkpts normalization. Printing
    an accessor previews the first k block using NumPy's print options.
    The static DF helpers return unnormalized blocks.
    Use close() or a context manager to release scratch storage.

    Methods:
        __enter__, __exit__: Manage scratch-file lifetime.
        allocate: Allocate selected four-index blocks in memory or in one
            scratch file.
        ovvv_df: Reconstruct an occupied slice of a chemists' (ov|vv) DF block.
        vvvv_df: Reconstruct a virtual slice in particle-particle ordering.
        close: Close the owned integral scratch file and clear its file
            reference.

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
        ...     print(eris.ovvv_kpoint_normalized[0, 0, 0].shape)
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
        # All disk-backed blocks share one file for the container's lifetime.
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
    def ovvv_kpoint_normalized(self):
        r"""Access stored or DF-reconstructed ovvv through three k indices.

        Returns:
            _IntegralBlocks: Accessor with logical shape
            (nkpts, nkpts, nkpts, nocc, nvir, nvir, nvir). Index as
            ``ovvv_kpoint_normalized[kp, kq, kr, ...]``; the fourth k point
            follows chemists' momentum conservation. Reading one block does not
            fill raw ovvv.

        Examples:
            Read the Gamma block on a Gamma-centered mesh with
            ``eris.ovvv_kpoint_normalized[0, 0, 0]``.
        """
        return _IntegralBlocks(self, 'ovvv', self._get_ovvv)

    @property
    def vvvv_kpoint_normalized(self):
        r"""Access vvvv in particle-particle ordering without full expansion.

        Returns:
            _IntegralBlocks: Accessor with logical shape (nkpts,)*3 +
            (nvir,)*4.
            Index as ``vvvv_kpoint_normalized[kp, kq, kr, ...]``. The fourth
            momentum is kconserv[kp, kr, kq]; the returned orbital order is
            p,r,q,s relative
            to the DF pair product. Reading a block does not fill raw vvvv.

        Examples:
            Read the Gamma block on a Gamma-centered mesh with
            ``eris.vvvv_kpoint_normalized[0, 0, 0]``.
        """
        return _IntegralBlocks(self, 'vvvv', self._get_vvvv)

    # Compatibility with existing scripts using the original accessor names.
    ovvv_blocks = ovvv_kpoint_normalized
    vvvv_blocks = vvvv_kpoint_normalized

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

        Examples:
            Call ``eris.close()`` after the final disk-backed integral read.
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

    Methods:
        __call__: Run the selected transformation backend.
        incore: Transform symmetry-related four-index k blocks into NumPy
            storage.
        outcore: Transform four-index blocks into a single owned HDF5 scratch
            file.
        density_fit: Transform three-index GDF factors and store occupied four-
            index blocks.

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
        # Keep the padded orbital basis and k mesh common to all backends.
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

        # Prefer GDF factors in auto mode, then use the four-index size.
        backend = backend or adc.integral_backend
        match backend:
            case 'auto':
                if isinstance(self.with_df, df.GDF):
                    backend = 'density_fit'
                else:
                    required = self._incore_memory_mb()
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

    def _incore_memory_mb(self):
        """Estimate additional resident MB for the in-core transformation.

        Count the stored orbital sectors and allow three full MO quartets
        for overlapping transform, normalization and symmetry buffers.
        FFTDF additionally transforms blocks of orbital pairs on its grid.
        Native library workspace and allocator reuse vary by backend.
        """
        o, v = self.nocc, self.nvir
        stored = o**4 + o**3 * v + 3 * o**2 * v**2 + o * v**3
        if _needs_vvvv(self.adc):
            stored += v**4
        storage_mb = self.dtype.itemsize * self.nkpts**3 * stored / 1e6
        return storage_mb + self._quartet_memory_mb(self.nmo)

    def _quartet_memory_mb(self, norb):
        """Estimate transient MB for one full four-index transformation."""
        required = 3 * self.dtype.itemsize * norb**4
        if isinstance(self.with_df, FFTDF):
            # Pair values and their forward and inverse FFT buffers.
            pairs = norb**2
            if (self.nkpts == 1 and self.dtype == np.float64
                    and not np.any(self.kpts)):
                pairs = norb * (norb + 1) // 2
            required += 3 * pairs * np.prod(self.with_df.mesh) * 16
        return required / 1e6

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

    def _transform_block(self, triple, coeffs):
        """Transform and normalize one momentum-conserving quartet.

        Use the three indices in triple to find the fourth k point. The
        four coefficient matrices determine the orbital axes of the
        returned array, including any occupied or virtual restrictions.
        """
        quartet = (*triple, self.kconserv[triple])
        block = self.fao2mo(coeffs, self.kpts[list(quartet)], compact=False)
        if self.dtype == np.float64:
            block = block.real
        return block.reshape(tuple(c.shape[1] for c in coeffs)) / self.nkpts

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
        # Allocate only the orbital sectors needed by this ADC calculation.
        o, v = slice(None, self.nocc), slice(self.nocc, None)
        spaces = {'oooo': (o, o, o, o), 'oovv': (o, o, v, v),
                  'ovoo': (o, v, o, o), 'ovov': (o, v, o, v),
                  'ovvo': (o, v, v, o), 'ovvv': (o, v, v, v)}
        eris = _ChemistsERIs(self.adc).allocate(spaces)
        if _needs_vvvv(self.adc):
            eris.allocate(('vvvv',))
        self.log.info('Using incore ERI storage')

        # Transform one representative and distribute its symmetry images.
        for triple, equivalents in self.khelper.symm_map.items():
            quartet = (*triple, self.kconserv[triple])
            block = self._transform_block(
                triple, tuple(self.mo_coeff[k] for k in quartet))
            for kp, kq, kr in equivalents:
                related = self.khelper.transform_symm(block, kp, kq, kr)
                for name, indices in spaces.items():
                    getattr(eris, name)[kp, kq, kr] = related[indices]
                if eris.vvvv is not None:
                    eris.vvvv[kp, kr, kq] = related[v, v, v, v].transpose(
                        0, 2, 1, 3)
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
        # Group occupied-leading blocks by their first orbital pair.
        eris = _ChemistsERIs(self.adc)
        eris.allocate(
            ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo', 'ovvv'), outcore=True)
        o, v = slice(None, self.nocc), slice(self.nocc, None)
        spaces = {'o': o, 'v': v}
        for triple in product(range(self.nkpts), repeat=3):
            p, q, r, s = (self.mo_coeff[k]
                          for k in (*triple, self.kconserv[triple]))
            for second, names in ((o, ('oooo', 'oovv')),
                                  (v, ('ovoo', 'ovov', 'ovvo', 'ovvv'))):
                block = self._transform_block(
                    triple, (p[:, o], q[:, second], r, s))
                for name in names:
                    getattr(eris, name)[triple] = block[
                        :, :, spaces[name[2]], spaces[name[3]]]
                del block

        # Store all-virtual integrals in particle-particle order if required.
        if _needs_vvvv(self.adc):
            eris.allocate(('vvvv',), outcore=True)
            self._write_vvvv(eris.vvvv)
        return eris

    def _write_vvvv(self, dataset):
        """Write normalized virtual quartets to the supplied HDF5 dataset.

        Expand symmetry-related quartets in particle-particle order. Under
        memory pressure, transform one virtual orbital at a time.
        """
        # Budget for one dense complex virtual quartet at a time.
        available_mb = self.adc.max_memory - lib.current_memory()[0]
        full_block = self._quartet_memory_mb(self.nvir) < available_mb
        for triple, equivalents in self.khelper.symm_map.items():
            kp, kq, kr = triple
            ks = self.kconserv[triple]
            p, q, r, s = (self.orbv[k] for k in (*triple, ks))
            if full_block:
                block = self._transform_block(triple, (p, q, r, s))
                for i, j, k in equivalents:
                    related = self.khelper.transform_symm(block, i, j, k)
                    dataset[i, k, j] = related.transpose(0, 2, 1, 3)
            else:
                # Transform one orbital and write its four symmetry images.
                for a in range(self.nvir):
                    block = self._transform_block(
                        triple, (p[:, a:a + 1], q, r, s))
                    prqs = block[0].transpose(1, 0, 2)
                    dataset[kp, kr, kq, a, :, :, :] = prqs
                    dataset[kr, kp, ks, :, a, :, :] = prqs.transpose(0, 2, 1)
                    dataset[kq, ks, kp, :, :, a, :] = (
                        prqs.transpose(1, 2, 0).conj())
                    dataset[ks, kq, kr, :, :, :, a] = (
                        prqs.transpose(2, 1, 0).conj())

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

        # Keep the original MO factors for consumers that need the full pair.
        eris.Lpq_mo = np.empty((self.nkpts, self.nkpts), dtype=object)
        with df._load3c(self.with_df._cderi, 'j3c') as fload:
            orbital_range = (0, self.nmo, self.nmo, 2 * self.nmo)
            for ki, kpti in enumerate(self.kpts):
                for kj, kptj in enumerate(self.kpts):
                    Lpq_ao = np.asarray(fload(kpti, kptj))

                    mo = np.hstack((self.mo_coeff[ki], self.mo_coeff[kj]))
                    mo = np.asarray(mo, dtype=self.dtype, order='F')
                    if self.dtype == np.double:
                        out = _ao2mo.nr_e2(
                            Lpq_ao,
                            mo,
                            orbital_range,
                            aosym='s2')
                    else:
                        # The complex transform expects full AO-pair blocks.
                        if Lpq_ao[0].size != self.nao**2:  # aosym = 's2'
                            Lpq_ao = lib.unpack_tril(Lpq_ao).astype(
                                np.complex128)
                        out = _ao2mo.r_e2(
                            Lpq_ao,
                            mo,
                            orbital_range,
                            [], None)
                    eris.Lpq_mo[ki, kj] = out.reshape(-1, self.nmo, self.nmo)

        # Pack orbital sectors for contractions and lazy virtual blocks.
        o, v = slice(None, self.nocc), slice(self.nocc, None)
        spaces = {'oo': (o, o), 'ov': (o, v), 'vo': (v, o), 'vv': (v, v)}
        sizes = {'o': self.nocc, 'v': self.nvir}
        pairs = {}
        for key, (left, right) in spaces.items():
            shape = (self.nkpts, self.nkpts, naux,
                     sizes[key[0]], sizes[key[1]])
            pairs[key] = np.empty(shape, dtype=self.dtype)
            for ki, kj in np.ndindex(eris.Lpq_mo.shape):
                pairs[key][ki, kj] = eris.Lpq_mo[ki, kj][:, left, right]

        if self.adc.if_naf:
            self._project_auxiliary(pairs, naux)
        eris.Lov, eris.Lvv = pairs['ov'], pairs['vv']

        # Contract only the five occupied-containing blocks used from disk.
        names = ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo')
        eris.allocate(names, outcore=True)
        for triple in product(range(self.nkpts), repeat=3):
            kp, kq, kr = triple
            ks = self.kconserv[triple]
            for name in names:
                left = pairs[name[:2]][kp, kq]
                right = pairs[name[2:]][kr, ks]
                getattr(eris, name)[triple] = (
                    einsum('Lpq,Lrs->pqrs', left, right) / self.nkpts)

        return eris

    def _project_auxiliary(self, pairs, naux):
        """Project all orbital sectors onto the retained auxiliary basis.

        Stack their three-index factors to form one auxiliary metric.
        Replace each entry in pairs with factors in the same truncated
        eigenbasis and record its dimension on the ADC object.
        """
        # Each column is one orbital pair at one pair of k points.
        columns = np.concatenate([
            factors.transpose(2, 0, 1, 3, 4).reshape(naux, -1)
            for factors in pairs.values()
        ], axis=1)
        metric = columns @ columns.conj().T
        del columns

        # Apply the same retained auxiliary eigenvectors to each sector.
        values, vectors = np.linalg.eigh(metric / self.nkpts)
        projection = vectors[:, values > self.adc.thresh_naf].T
        self.adc.naux = len(projection)
        self.log.info('NAF reduced %d auxiliary functions to %d',
                      naux, self.adc.naux)
        for key, factors in pairs.items():
            pairs[key] = einsum('rL,mnLsj->mnrsj', projection, factors)


def chunk_size(adc):
    r"""Estimate an orbital chunk from the remaining memory budget.

    A single reconstructed virtual block needs approximately
    :math:`n_{chunk} n_{vir}^3` elements. Budget for two such blocks
    because normalization, conjugation or packing can require a second
    copy during contraction. Use the remaining resident-memory budget and
    the actual orbital dtype. Other solver arrays also contribute to RSS.

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

    Raises:
        ValueError: The requested integral backend is unknown.

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
