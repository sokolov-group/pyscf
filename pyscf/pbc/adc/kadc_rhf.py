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

r"""Restricted ADC reference, amplitude and shared excitation-driver routines.

KRADCBase supplies reference data; KRADCIP, KRADCEA and KRADCEE solve
their respective excitation sectors. Ground-state correlation energies are in
Hartree per primitive cell; excitation roots are in Hartree and ordered
by the selected kptlist. Active coefficients and amplitudes use KMP2
padding to accommodate unequal orbital counts across the k mesh.

Each numerical example includes imports and builds its own two-k-point
helium reference ``kmf``. Ground amplitudes are
stored on the driver rather than returned by ground_state_energy().
"""
from itertools import product
from types import SimpleNamespace
import copy
import tempfile
import time
import warnings

import h5py
import numpy as np

from pyscf import __config__, lib
from pyscf.data.nist import HARTREE2EV
from pyscf.lib import logger
from pyscf.lib.parameters import LARGE_DENOM, LOOSE_ZERO_TOL
from pyscf.pbc import df, gto, scf, tools
from pyscf.pbc.adc import kadc_ao2mo, kadc_ee_terms
from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs
from pyscf.pbc.adc.lib import (
    MomentumContractions, check_implemented, einsum, log_table, validate,
)
from pyscf.pbc.cc.kccsd_rhf import _get_epq
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.mp.kmp2 import (
    _add_padding,
    get_frozen_mask,
    get_nmo,
    get_nocc,
    padded_mo_coeff,
    padding_k_idx,
)
from pyscf.pbc.scf.hf import INVALID_ORBITAL_ENERGY


# Note : All integrals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba

def kernel(adc, nroots=1, guess=None, eris=None, kptlist=None):
    r"""Solve the IP, EA or singlet EE effective-Hamiltonian eigenproblem.

    For each selected momentum sector, Davidson iteration solves
    :math:`M(k) U_n(k) = E_n(k) U_n(k)`. Ground-state amplitudes must
    already be present for positive nroots; the KRADCBase driver prepares them.

    Args:
        adc (KRADCIP, KRADCEA or KRADCEE): Sector solver with reference data.
        nroots (int): Roots per sector, capped at vector_size(). Zero or a
            negative value computes only the reference correlation energy.
        guess (array_like, optional): Trial vectors with shape
            (len(kptlist), nroots, vector_size()), in kptlist order.
            By default use unit vectors at the lowest diagonal entries.
        eris (_ChemistsERIs, optional): Pretransformed active-space
            integrals. With None, call adc.ao2mo().
        kptlist (sequence[int], optional): Momentum-sector indices; defaults
            to every k point. Ignored for a reference-only calculation.

    Returns:
        tuple: E, U, P, X, also stored on adc. E and P have shape
        (nsectors, nroots); U has shape (nsectors, nroots, vector_size()).
        X contains IP/EA spectroscopic amplitudes with shape
        (nsectors, nmo, nroots), or EE transition densities with shape
        (nsectors, nroots, nkpts, nmo, nmo). Energies are in Hartree.
        P contains charged spectral factors or neutral density-response
        strengths. If compute_properties=False, P and X are zero arrays.
        For nroots < 1, return scalar e_corr and clear excitation results.

    For excitation calculations, read E, U, P and X from the solver/driver
    attributes after calling kernel(). Unpacking the returned tuple is
    discouraged; the tuple remains available for compatibility.

    Examples:
        Create the two-k-point helium reference below; lattice vectors
        are in Angstrom and energies are in Hartree:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase, kernel
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = KRADCBase(kmf).run(0)
        >>> mykradcee = KRADCEE(mykradcbase)
        >>> excitation_results = kernel(mykradcee, nroots=2, kptlist=[0])
        >>> energies = mykradcee.E
        >>> vectors = mykradcee.U
        >>> strengths = mykradcee.P
        >>> densities = mykradcee.X
        >>> energies.round(6).tolist()
        [[1.756312, 2.102047]]
        >>> vectors.shape, densities.shape
        ((1, 2, 10), (1, 2, 2, 2, 2))
    """
    if not isinstance(nroots, (int, np.integer)):
        raise TypeError('nroots must be an integer.')
    if nroots < 1:
        energy = KRADCBase.ground_state_energy(adc, eris)
        adc._clear_excitations()
        logger.info(
            adc, f'nroots = {nroots}; only the ground state was calculated.')
        return energy

    adc._invalidate_orbitals()
    validate(adc)
    started = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(adc.stdout, adc.verbose)
    adc.dump_flags()
    if eris is None:
        eris = adc.ao2mo()
    vec_size = adc.vector_size()
    nroots = min(nroots, vec_size)
    nkpts, nmo = adc.nkpts, adc.nmo
    kptlist = list(range(nkpts) if kptlist is None else kptlist)
    if not (kptlist and all(
        isinstance(k, (int, np.integer)) and 0 <= k < nkpts for k in kptlist)):
        raise ValueError('kptlist must contain valid k-point indices.')
    dtype = np.result_type(adc.t2[0])
    root_shape = len(kptlist), nroots
    evals = np.zeros(root_shape)
    evecs = np.zeros((*root_shape, vec_size), dtype=dtype)
    conv = np.zeros(root_shape, dtype=bool)
    P = np.zeros(root_shape)
    neutral = isinstance(adc, KRADCEE)
    if neutral:
        X = np.zeros((*root_shape, nkpts, nmo, nmo), dtype=dtype)
        imds = None  # Neutral intermediates depend on kshift.
    else:
        X = np.zeros((len(kptlist), nmo, nroots), dtype=dtype)
        imds = adc.get_imds(eris)
    initial_guess = adc._init_guess
    match guess:
        case None:
            guess_type = None
        case str():
            raise NotImplementedError('Supply trial vectors as an array.')
        case _:
            if not hasattr(guess, '__len__'):
                raise NotImplementedError('Guess type not implemented.')
            guess_type = 'read'
            guess_k = np.asarray(guess).copy()

    for k, kshift in enumerate(kptlist):
        matvec, diag = adc.gen_matvec(kshift, imds, eris)
        match guess_type:
            case 'read':
                initial = initial_guess(
                    nroots,
                    diag,
                    ascending=True,
                    type_='read',
                    ini=guess_k[k],
                    kshift=kshift)
            case None:
                initial = initial_guess(
                    nroots, diag, ascending=True, kshift=kshift)
        conv_k, evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(
            lambda xs: [matvec(x) for x in xs],
            initial,
            diag,
            nroots=nroots,
            verbose=log,
            tol=adc.conv_tol,
            max_cycle=adc.max_cycle,
            max_space=adc.max_space,
            tol_residual=adc.tol_residual)
        evals[k], evecs[k], conv[k] = evals_k.real, evecs_k, conv_k
        U = evecs[k].T.copy()
        if neutral:
            U = adc.renormalize_eigenvectors(kshift, U, nroots)
            evecs[k] = U.T
        if adc.compute_properties:
            P[k], X[k] = adc.get_properties(kshift, U, nroots)

    adc._kptlist = kptlist
    adc.E, adc.U, adc.P, adc.X, adc.converged = evals, evecs, P, X, conv
    if neutral:
        adc.T = X.reshape(*root_shape, -1) if adc.compute_properties else None
    nfalse = conv.size - np.count_nonzero(conv)
    if nfalse:
        logger.warn(
            adc, f'Davidson iterations for {nfalse} root(s) did not converge.')
    data = {'Method': [], 'k': [], 'Root': [], 'Energy (Eh)': [],
            'Energy (eV)': []}
    strength_label = 'Density' if neutral else 'Spec factor'
    if adc.compute_properties:
        data[strength_label] = []
    data['Converged'] = []
    for k, kshift in enumerate(kptlist):
        for root in range(nroots):
            data['Method'].append(adc.method)
            data['k'].append(kshift)
            data['Root'].append(root)
            data['Energy (Eh)'].append(f'{evals[k, root]:.10f}')
            data['Energy (eV)'].append(f'{evals[k, root] * HARTREE2EV:.8f}')
            if adc.compute_properties:
                data[strength_label].append(f'{P[k, root]:.8f}')
            data['Converged'].append(conv[k, root])
    log_table(adc, data, 'ADC calculation summary')
    log.timer('ADC', *started)
    return evals, evecs, P, X


def make_ref_rdm1(adc, with_frozen=True, ao_repr=False):
    r"""Build the spin-summed reference one-particle density matrix.

    The MO density :math:`D_{pq}(k) = \sum_\sigma
    \langle a^\dagger_{qk\sigma} a_{pk\sigma}\rangle` includes corrections
    through second order for ADC(2)/ADC(2)-x and third order for ADC(3).
    Occupied-virtual terms follow approx_trans_moments. The average trace
    over the k mesh gives the active electron count before frozen occupied
    orbitals are restored. In the AO basis, :math:`D^{AO}=C D^{MO} C^\dagger`.

    Args:
        adc (KRADCBase): Reference with previously computed t1 and t2.
        with_frozen (bool): Restore frozen occupied/virtual orbitals to the
            MO density. Frozen occupied diagonal entries are two.
        ao_repr (bool): Transform to the AO basis; otherwise return the MO
            density, including active-space padding when with_frozen=False.

    Returns:
        ndarray: Complex Hermitian density with shape (nkpts, norb, norb).
        norb is nao in the AO basis, the full MO count when frozen orbitals
        are restored, or the padded active MO count otherwise.

    Examples:
        Create the two-k-point helium reference below; lattice vectors
        are in Angstrom and energies are in Hartree:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = KRADCBase(kmf).run(0)
        >>> density = mykradcbase.make_ref_rdm1()
        >>> density.shape
        (2, 2, 2)
        >>> round(float(np.trace(density, axis1=1, axis2=2).real.mean()), 6)
        2.0
        >>> mykradcbase.make_ref_rdm1(ao_repr=True).shape
        (2, 2, 2)
    """

    adc.method = check_implemented(adc.method)

    t1 = adc.t1
    t2 = adc.t2
    t2_ov = t1[0]
    t1_oovv = t2[0]

    ######################

    nocc = adc.nocc
    nmo = adc.nmo
    nvir = nmo - nocc
    nkpts = adc.nkpts

    OPDM = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    OPDM[:, :nocc, :nocc] += np.identity(nocc)

    ####### ADC(2) SPIN ADAPTED REF OPDM with SQA ################
    kj, ka = np.indices((nkpts, nkpts))
    idx0 = np.arange(nkpts)[:, None]
    oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
    vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
    for ki in range(nkpts):
        kb = adc.khelper.kconserv[ki, ka, kj]
        t1_oovv_np = np.array(t1_oovv[ki])
        t1_oovv_ijb = t1_oovv_np[idx0, kb]

        oo[ki] -= 2 * einsum(
            'KkIiab,KkJiab->IJ',
            t1_oovv_np, t1_oovv_np.conj())
        oo[ki] += einsum('KkIiab,KkJiba->IJ', t1_oovv_np, t1_oovv_ijb.conj())
        vv += 2 * einsum('kKijBa,kKijAa->KAB', t1_oovv_np, t1_oovv_np.conj())
        vv -= einsum('kKijBa,kKijaA->KAB', t1_oovv_np, t1_oovv_ijb.conj())
        del (t1_oovv_np)
        del (t1_oovv_ijb)

    OPDM[:, :nocc, :nocc] += oo
    OPDM[:, nocc:, nocc:] += vv
    del (oo)
    del (vv)

    if adc.approx_trans_moments is False or adc.method == "adc(3)":
        ### OCC-VIR ###
        OPDM[:, :nocc, nocc:] += t2_ov
        ### VIR-OCC ###
        OPDM[:, nocc:, :nocc] = OPDM[:, :nocc, nocc:].conj().transpose(0, 2, 1)

    ####### ADC(3) SPIN ADAPTED REF OPDM WITH SQA ################
    if adc.method == "adc(3)":
        t2_oovv = t2[1]

        for ki in range(nkpts):
            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = adc.khelper.kconserv[ki, ka, kj]
                    #### OCC-OCC ###
                    OPDM[ki][:nocc, :nocc] -= 2 * einsum(
                        'Iiab,Jiab->IJ',
                        t1_oovv[ki][kj][ka], t2_oovv[ki][kj][ka].conj())
                    OPDM[ki][:nocc, :nocc] += einsum(
                        'Iiab,Jiba->IJ',
                        t1_oovv[ki][kj][ka], t2_oovv[ki][kj][kb].conj())
                    OPDM[ki][:nocc, :nocc] -= 2 * einsum(
                        'Jiab,Iiab->IJ',
                        t1_oovv[ki][kj][ka].conj(), t2_oovv[ki][kj][ka])
                    OPDM[ki][:nocc, :nocc] += einsum(
                        'Jiab,Iiba->IJ',
                        t1_oovv[ki][kj][ka].conj(), t2_oovv[ki][kj][kb])
                    ##### VIR-VIR ###
                    OPDM[ka][nocc:, nocc:] += 2 * einsum(
                        'ijBa,ijAa->AB',
                        t1_oovv[ki][kj][ka], t2_oovv[ki][kj][ka].conj())
                    OPDM[ka][nocc:, nocc:] -= einsum(
                        'ijBa,ijaA->AB',
                        t1_oovv[ki][kj][ka], t2_oovv[ki][kj][kb].conj())
                    OPDM[ka][nocc:, nocc:] += 2 * einsum(
                        'ijAa,ijBa->AB',
                        t1_oovv[ki][kj][ka].conj(), t2_oovv[ki][kj][ka])
                    OPDM[ka][nocc:, nocc:] -= einsum(
                        'ijAa,ijaB->AB',
                        t1_oovv[ki][kj][ka].conj(), t2_oovv[ki][kj][kb])

                ka = ki
                kb = kj
                ##### OCC-VIR ### ####
                OPDM[ki][:nocc, nocc:] += einsum(
                    'IiAa,ia->IA',
                    t1_oovv[ki][kj][ka], t2_ov[kj].conj())
                OPDM[ki][:nocc, nocc:] -= 1 / 2 * einsum(
                    'IiaA,ia->IA',
                    t1_oovv[ki][kj][kb], t2_ov[kj].conj())
            ###### VIR-OCC ###
            OPDM[ki][nocc:, :nocc] = OPDM[ki][:nocc, nocc:].conj().T

    for ki in range(nkpts):
        OPDM[ki] += OPDM[ki].conj().T

    if with_frozen and adc.frozen is not None:
        nmo = max(mo_occ_k.size for mo_occ_k in adc.mo_occ)
        nocc = np.count_nonzero(adc.mo_occ[0] > 0)
        mask = get_frozen_mask(adc)
        dm = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        occ_idx = np.arange(nocc)
        k_idx = np.arange(nkpts)
        p_k_idx = padding_k_idx(adc, kind='joint')
        dm[k_idx[:, None], occ_idx, occ_idx] = 2
        for k in k_idx:
            moidx = np.where(mask[k])[0]
            dm[k][moidx[:, None], moidx] = OPDM[k][p_k_idx[k][:, None],
                p_k_idx[k]]
        OPDM = dm

    if ao_repr:
        if with_frozen and adc.frozen is not None:
            mo = [np.pad(c, ((0, 0), (0, nmo - c.shape[1])))
                  for c in map(np.asarray, adc.mo_coeff)]
        else:
            mo = padded_mo_coeff(adc, adc.mo_coeff)
        # AO and active MO dimensions may differ after freezing or FNO.
        OPDM = np.asarray([
            einsum('pI,IJ,qJ->pq', c, OPDM[k], c.conj())
            for k, c in enumerate(mo)])

    return OPDM


def mo_splitter(myadc, kpt=0):
    r"""Partition one k point's MOs into frozen/active occupied/virtual sets.

    Args:
        myadc (KRADCBase): Object supplying nkpts, mo_occ and frozen. Frozen
            orbitals use the KMP2 convention: a count, a shared list of MO
            indices, or one index list per k point. Occupations > 1e-6
            identify occupied MOs; these masks use the original MO ordering.
        kpt (int): K-point index in [0, nkpts), default zero. Selecting one
            k point at a time also supports unequal MO counts across the mesh.

    Returns:
        list[ndarray]: Four one-dimensional boolean masks, in the order
        frozen occupied, active occupied, active virtual, frozen virtual.
        Removed FNO orbitals are classified by get_frozen_mask as frozen;
        orbital_number_lists reports their count separately.

    Raises:
        TypeError: kpt is not an integer.
        IndexError: kpt is outside the sampled mesh.

    Examples:
        Freeze the first two occupied MOs and the last virtual MO of a
        six-orbital model, and display the four masks:

        >>> import numpy as np
        >>> from pyscf.pbc.adc.kadc_rhf import mo_splitter
        >>> from types import SimpleNamespace
        >>> model = SimpleNamespace(
        ...     nkpts=1,
        ...     frozen=[0, 1, 5],
        ...     mo_occ=[np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0])])
        >>> masks = mo_splitter(model)
        >>> print('Is Frozen Occupied?', masks[0])
        Is Frozen Occupied? [ True  True False False False False]
        >>> print('Is Active Occupied?', masks[1])
        Is Active Occupied? [False False  True False False False]
        >>> print('Is Active Virtual?', masks[2])
        Is Active Virtual? [False False False  True  True False]
        >>> print('Is Frozen Virtual?', masks[3])
        Is Frozen Virtual? [False False False False False  True]
    """
    if not isinstance(kpt, (int, np.integer)):
        raise TypeError('kpt must be an integer.')
    if not 0 <= kpt < myadc.nkpts:
        raise IndexError('kpt must be between zero and nkpts - 1.')
    active = get_frozen_mask(myadc)[kpt]
    occupied = np.asarray(myadc.mo_occ[kpt]) > 1e-6
    return [occupied & ~active, occupied & active,
            ~occupied & active, ~occupied & ~active]


def orbital_number_lists(myadc):
    r"""Count frozen, active and removed orbitals at every k point.

    Args:
        myadc (KRADCBase): Object supplying the attributes used by mo_splitter
            and, optionally, mo_energy. An energy equal to PySCF's
            INVALID_ORBITAL_ENERGY marks an orbital removed by FNO.

    Returns:
        tuple[ndarray]: Integer arrays of shape (nkpts,) in the order
        frozen occupied, active occupied, active virtual, frozen virtual,
        removed orbitals. Removed orbitals are excluded from frozen virtual
        counts. Without mo_energy, there are no removed orbitals.

    Examples:
        One occupied MO, one active virtual MO and one removed virtual MO:

        >>> import numpy as np
        >>> from pyscf.pbc.adc.kadc_rhf import orbital_number_lists
        >>> from pyscf.pbc.scf.hf import INVALID_ORBITAL_ENERGY
        >>> from types import SimpleNamespace
        >>> model = SimpleNamespace(
        ...     nkpts=1,
        ...     frozen=None,
        ...     mo_occ=[np.array([2.0, 0.0, 0.0])],
        ...     mo_energy=[np.array([-0.5, 0.2, INVALID_ORBITAL_ENERGY])])
        >>> [counts.tolist() for counts in orbital_number_lists(model)]
        [[0], [1], [1], [0], [1]]
    """
    nocc_fr, nocc_ar, nvir_ar, nvir_fr, nremoved = [], [], [], [], []
    for kpt in range(myadc.nkpts):
        frz_occ, act_occ, act_vir, frz_vir = mo_splitter(myadc, kpt)
        if getattr(myadc, 'mo_energy', None) is not None:
            valid_k = (np.asarray(
                myadc.mo_energy[kpt]) != INVALID_ORBITAL_ENERGY)
        else:
            valid_k = np.ones(frz_vir.size, dtype=bool)
        nocc_fr.append(np.count_nonzero(frz_occ))
        nocc_ar.append(np.count_nonzero(act_occ))
        nvir_ar.append(np.count_nonzero(act_vir))
        nvir_fr.append(np.count_nonzero(frz_vir & valid_k))
        nremoved.append(np.count_nonzero(~valid_k))
    return (np.array(nocc_fr), np.array(nocc_ar),
            np.array(nvir_ar), np.array(nvir_fr), np.array(nremoved))


class _AmplitudeStorage:
    r"""Own the HDF5 scratch arrays shared by converted periodic ADC objects.

    ADC instances keep a reference to this owner rather than closing another
    object's arrays. Storage is released when the last owner reference is
    dropped, including after a source ADC recomputes its amplitudes.
    """

    def __init__(self):
        r"""Open an anonymous scratch file and its HDF5 array container.

        Attributes:
            temporary_file: Binary temporary file removed on close.
            file (h5py.File): Array storage; datasets remain valid while this
                owner is retained by at least one ADC instance.
        """
        self.temporary_file = tempfile.TemporaryFile()
        self.file = h5py.File(self.temporary_file, 'a')

    def __del__(self):
        r"""Close scratch handles after the last ADC releases this owner.

        Returns:
            None: Already-closed or partially initialized handles are ignored
            so cleanup is safe during exception handling and interpreter exit.
        """
        for name in ('file', 'temporary_file'):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    handle.close()
                except (AttributeError, OSError, ValueError):
                    pass


class KRADCBase(lib.StreamObject):
    r"""Common reference, amplitudes and solver settings for periodic
    restricted ADC.

    This class inherits lib.StreamObject directly. Use KRADCIP, KRADCEA or
    KRADCEE for positive-root calculations; use KRADCBase for reference-only
    work. The concrete Python class identifies the excitation sector.

    Args:
        mf (KRHF, Cell or KRADCBase): Periodic reference. A Cell creates a
            density-fitted KRHF at Gamma, with SCF deferred until needed.
            Converting an ADC shares common reference data and amplitudes while
            clearing sector results and caches. Shared scratch arrays remain
            readable if either object recomputes its reference.
        frozen (int or sequence, optional): KMP2 frozen-space specification;
            None inherits the source setting when converting an ADC.
        mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
            orbital overrides. Energies are in Hartree; overrides clear copied
            amplitudes.
        kpts (array_like, optional): Cartesian inverse-Bohr momenta, shape
            (nkpts, 3), for Cell inputs only. None selects one Gamma point.

    Attributes:
        method (str): ``adc(2)``, ``adc(2)-x`` or ``adc(3)``.
        t1, t2: Singles and doubles amplitude tuples, ordered by perturbative
            order. Large doubles tensors remain backed by scratch storage.
        e_corr (float): Reference correlation energy per primitive cell.
        E, U, P, X: Excitation energies, eigenvectors, strengths and
            spectroscopic amplitudes/transition densities; see kernel().
        T: Flattened EE transition densities, or None without EE properties.
        converged: Boolean array matching E, one flag per root and sector.
        compute_properties (bool): Compute P and X after diagonalization.
        approx_trans_moments (bool): Omit selected higher-order transition
            amplitudes, following the molecular ADC convention (default True).
        density_gv (ndarray): Reciprocal vector G in inverse Bohr for EE
            density response at physical momentum q + G.
        integral_backend (str): ``auto``, ``incore``, ``outcore`` or
            ``density_fit``; ``df`` is accepted for compatibility.
        chunk_size (int): Cached orbital chunk size; assigning None restores
            automatic selection, and a positive integer fixes it explicitly.

    The inherited run() method returns this object for chaining. Use
    run(0) to prepare only the reference, or run(nroots) for all sectors.
    Use kernel(..., kptlist=...) when selecting individual momentum sectors.

    Examples:
        Two k points for a helium crystal with two s basis functions per
        atom. The lattice length is in Angstrom (PySCF's default unit):

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase, KRADCEE, KRADCEA
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = KRADCBase(kmf)
        >>> ground_state_energy = mykradcbase.ground_state_energy()
        >>> round(ground_state_energy, 8)
        -0.00416766
        >>> amp_singles, amp_doubles = mykradcbase.t1, mykradcbase.t2
        >>> amp_doubles[0].shape
        (2, 2, 2, 1, 1, 1, 1)
        >>> mykradcee = KRADCEE(mykradcbase)
        >>> excitation_results = mykradcee.kernel(nroots=2, kptlist=[0])
        >>> excited_energies = mykradcee.E
        >>> excited_energies.round(6).tolist()
        [[1.756312, 2.102047]]
        >>> mykradcee.U.shape, mykradcee.P.shape, mykradcee.X.shape
        ((1, 2, 10), (1, 2), (1, 2, 2, 2, 2))
        >>> mykradcea = KRADCEA(mykradcee)
        >>> type(mykradcea) is KRADCEA
        True
        >>> mykradcea.e_corr == mykradcee.e_corr, mykradcea.E is None
        (True, True)
    """

    _keys = {
        'cell', 'kpts', 'khelper', 'exxdiv', 'method', 'frozen',
        'mo_coeff', 'mo_occ', 'mo_energy', 'scf_energy', 'e_corr', 't1', 't2',
        'max_space', 'max_cycle', 'conv_tol', 'tol_residual', 'incore_complete',
        'compute_properties', 'approx_trans_moments', 'integral_backend',
        'chunk_size', 'keep_exxdiv', 'with_df', 'naux', 'if_heri_eris', 'eris',
        'if_naf', 'thresh_naf', 'if_corr', 'density_gv', 'nkop_chk', 'kop_npick',
        'imds', 'E', 'T', 'U', 'X', 'P', 'converged',
    }

    incore_complete = False

    # Only common reference data survive a sector conversion.
    _reference_keys = _keys - {'E', 'T', 'U', 'X', 'P', 'converged', 'chunk_size'}

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 mo_energy=None, kpts=None):
        r"""Initialize reference data without computing amplitudes or roots.

        Args:
            mf (KRHF, Cell or KRADCBase): Restricted periodic reference. A Cell
                creates a density-fitted KRHF with exxdiv=None; SCF runs lazily
                when orbital data are needed. An ADC instance shares reference
                orbitals, amplitudes and common settings with the new object.
            frozen (int or sequence, optional): KMP2 frozen-space
                specification. None inherits the source setting for ADC-to-ADC
                conversion.
            mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
                orbital overrides, with energies in Hartree. Changing reference
                data discards copied amplitudes and correlation energy.
            kpts (array_like, optional): Shape (nkpts, 3), Cartesian inverse
                Bohr. Used only for Cell inputs; None selects Gamma.

        Returns:
            None: Set ADC(2) defaults for a new SCF/Cell reference. ADC
            conversion preserves common settings and reference data, clears
            sector results and caches, and shares the amplitude-storage owner
            to keep scratch arrays alive without retaining the source solver.
            Mutable bookkeeping and density_gv are copied independently.

        Examples:
            Construct from a Cell and convert sectors without running SCF.
            The lattice length below is in Angstrom:

            >>> import numpy as np
            >>> from pyscf.pbc import gto
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE, KRADCEA
            >>> cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]], verbose=0)
            >>> mykradcee = KRADCEE(cell)
            >>> mykradcea = KRADCEA(mykradcee)
            >>> mykradcea.mo_coeff is None, mykradcea.nkpts
            (True, 1)
            >>> type(mykradcea) is KRADCEA
            True
        """
        # Convert an ADC object by preserving only common reference data.
        if isinstance(mf, KRADCBase):
            if kpts is not None:
                raise ValueError(
                    'Supply kpts only with a Cell; an ADC object inherits '
                    'the k-point mesh of its reference.')
            self._init_from_ground_state(mf)
            overrides = {'frozen': frozen, 'mo_coeff': mo_coeff,
                         'mo_occ': mo_occ, 'mo_energy': mo_energy}
            if any(value is not None for value in overrides.values()):
                for key, value in overrides.items():
                    if value is not None:
                        setattr(self, key, value)
                self.t1 = self.t2 = self.e_corr = None
                self.imds = SimpleNamespace()
                self._nocc = self._nmo = self._nvir = None
        else:
            # Cell inputs create a KRHF reference; SCF remains deferred.
            if isinstance(mf, gto.Cell):
                mesh = np.zeros((1, 3)) if kpts is None else np.asarray(kpts)
                mf = scf.KRHF(mf, mesh, exxdiv=None).density_fit()
            elif kpts is not None:
                raise ValueError(
                    'Supply kpts only with a Cell; an SCF reference already '
                    'defines its k-point mesh.')
            if not isinstance(mf, scf.khf.KSCF):
                raise TypeError('Expected a periodic Cell, KRHF or KRADCBase.')
            if not isinstance(mf, scf.khf.KRHF):
                mf = mf.to_rhf()
            # Reuse reference orbitals, excluding invalid-energy FNO columns.
            self._scf = mf
            self.cell, self.kpts = mf.cell, mf.kpts
            self.exxdiv = mf.exxdiv
            self.stdout, self.verbose = mf.stdout, mf.verbose
            self.max_memory = mf.max_memory
            self.scf_energy = mf.e_tot
            self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
            self.mo_coeff = mf.mo_coeff if mo_coeff is None else mo_coeff
            self.mo_occ = mf.mo_occ if mo_occ is None else mo_occ
            self.mo_energy = mf.mo_energy if mo_energy is None else mo_energy
            if (self.mo_coeff is mf.mo_coeff and self.mo_energy is not None
                    and any(np.any(np.asarray(e) == INVALID_ORBITAL_ENERGY)
                            for e in self.mo_energy)):
                valid = [np.asarray(e) != INVALID_ORBITAL_ENERGY
                         for e in self.mo_energy]
                self.mo_coeff = [np.asarray(c)[:, v]
                                 for c, v in zip(self.mo_coeff, valid)]
                self.mo_occ = [np.asarray(o)[v]
                               for o, v in zip(self.mo_occ, valid)]
                self.mo_energy = [np.asarray(e)[v]
                                  for e, v in zip(self.mo_energy, valid)]
            # Set the ADC approximation, Davidson controls and property flags.
            self.frozen = frozen
            self.method = 'adc(2)'
            self.max_space = getattr(__config__, 'adc_kadc_RADC_max_space', 12)
            self.max_cycle = getattr(__config__, 'adc_kadc_RADC_max_cycle', 50)
            self.conv_tol = getattr(__config__, 'adc_kadc_RADC_conv_tol', 1e-7)
            self.tol_residual = getattr(__config__, 'adc_kadc_RADC_tol_res', 1e-4)
            self.compute_properties = self.approx_trans_moments = True
            self._nocc = self._nmo = self._nvir = None
            self.nkop_chk = self.kop_npick = False
            self.t1 = self.t2 = self.e_corr = None
            self._chunk_size = None
            self._chunk_size_manual = False
            self.imds = SimpleNamespace()
            self.keep_exxdiv = False
            self.naux = None
            self.if_heri_eris = self.if_naf = False
            self.if_corr = True
            self.thresh_naf = 1e-2
            self.eris = None
            # Backends: auto, incore, outcore, density_fit (also spelled df).
            self.integral_backend = 'auto'
            self.density_gv = np.zeros(3)
        # Both construction paths start with empty orbital/sector caches.
        self._padded_orbitals = None
        self._ee_momentum = {}
        self._transition_cache = {}
        self._clear_excitations()

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        r"""Compute the reference and solve :math:`M(k) U_n=E_n U_n`.

        Args:
            nroots (int): Roots per selected momentum sector. Zero or a
                negative value requests only the ground-state calculation.
            guess (array_like, optional): Trial vectors with shape
                (len(kptlist), nroots, vector_size()), in kptlist order.
            eris (_ChemistsERIs, optional): Pretransformed integrals.
            kptlist: Momentum-sector indices, defaulting to every k point.

        Returns:
            tuple: E, U, P, X for an excited-state run; these are also stored
                on the driver. E/P have shape (nsectors, nroots), and U has
                shape (nsectors, nroots, vector_size()). X has shape
                (nsectors, nmo, nroots) for IP/EA, or
                (nsectors, nroots, nkpts, nmo, nmo) for EE. E is in Hartree.
                P contains spectral factors (IP/EA) or density strengths
                (EE). With properties disabled, P and X are zero arrays.
                A reference-only run returns scalar e_corr in Hartree and
                clears all excitation results.

        For excitation calculations, read E, U, P and X from the solver/driver
        attributes after calling kernel(). Unpacking the returned tuple is
        discouraged; the tuple remains available for compatibility.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = KRADCEE(kmf)
            >>> round(mykradcee.kernel(nroots=0), 8)
            -0.00416766
            >>> excitation_results = mykradcee.kernel(nroots=2, kptlist=[0])
            >>> mykradcee.E.round(6).tolist()
            [[1.756312, 2.102047]]
        """
        if not isinstance(nroots, (int, np.integer)):
            raise TypeError('nroots must be an integer.')
        if nroots > 0 and not isinstance(self, (KRADCIP, KRADCEA, KRADCEE)):
            raise NotImplementedError('Choose KRADCIP, KRADCEA or KRADCEE for '
                                      'excitation calculations.')
        self._ee_momentum.clear()
        self._transition_cache.clear()
        eris = self._compute_reference(eris)
        if nroots < 1:
            self._clear_excitations()
            return self.e_corr
        return kernel(self, nroots, guess, eris, kptlist)

    def _compute_reference(self, eris=None):
        r"""Validate settings, transform integrals, and populate reference
        results.

        Args:
            eris (_ChemistsERIs, optional): Existing active-space integrals.
                With
                None, transform using ao2mo() after invalidating cached
                padding.

        Returns:
            _ChemistsERIs: Integral container used for the amplitude and energy
            calculation. Store amplitudes in t1/t2 and the per-cell Hartree
            energy
            in e_corr. If if_heri_eris is True, also retain the container as
            eris.
            This helper logs settings and orbital information and finalizes the
            run.
        """
        self._ensure_reference()
        self._invalidate_orbitals()
        if self.mo_coeff is None or self.mo_occ is None:
            raise ValueError('MO coefficients and occupations are required.')
        validate(self)
        self.dump_flags_gs()
        self._dump_orbital_info()
        if eris is None:
            eris = self.ao2mo()
        self.amplitudes(eris)
        self.correlation_energy(eris)
        self._finalize()
        if self.if_heri_eris:
            self.eris = eris
        return eris

    def _clear_excitations(self):
        r"""Clear saved excited-state data after a reference-only calculation.

        Returns:
            None: Set E, U, P, X, T, converged and _adc_es to None.
            Ground-state
            amplitudes, correlation energy and reference orbitals remain
            available.
            This avoids exposing roots from an earlier run after
            kernel(nroots=0).
        """
        self.E = self.U = self.P = self.X = self.T = None
        self.converged = None
        self._adc_es = None
        self._kptlist = None

    def _invalidate_orbitals(self):
        r"""Discard cached padding and any automatically estimated chunk size.

        Returns:
            None: The next orbital-property access rebuilds active padding, and
            the next automatic chunk_size access estimates memory again. A
            manual
            chunk choice is retained. This helper does not recompute
            amplitudes,
            invalidate an external integral container, or clear solver momentum
            maps.
        """
        self._padded_orbitals = None
        if not self._chunk_size_manual:
            self._chunk_size = None

    def dump_flags(self, verbose=None):
        r"""Log the concrete solver class, ADC order and Davidson settings.

        Args:
            verbose (int, optional): Logging level; None uses self.verbose.

        Returns:
            KRADCBase: This object, for PySCF method chaining.
        """
        log = logger.new_logger(self, verbose)
        log.info('******** %s / %s ********', type(self).__name__, self.method)
        log.info('max_space = %d; max_cycle = %d', self.max_space, self.max_cycle)
        log.info('conv_tol = %s; tol_residual = %s',
                 self.conv_tol, self.tol_residual)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def ao2mo(self, backend=None):
        r"""Return active-space MO integrals using :class:`kadc_ao2mo.AO2MO`.

        Args:
            backend (str, optional): Override integral_backend for this call.

        Returns:
            _ChemistsERIs: The owner of arrays, DF factors and scratch files.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> with mykradcbase.ao2mo('density_fit') as eris:
            ...     print(eris.ovov.shape)
            (2, 2, 2, 1, 1, 1, 1)
        """
        return kadc_ao2mo.ao2mo(self, backend)

    def vector_size(self):
        r"""Return the flattened singles-plus-doubles excitation dimension.

        Returns:
            int: n_singles + n_doubles, in the concrete sector's C-order
            layout.
        """
        return self.n_singles + self.n_doubles

    def get_imds(self, eris=None, kshift=0):
        r"""Build the charged-sector singles effective-Hamiltonian
        intermediate.

        The zeroth-order matrix is diagonal in the canonical orbital energies.
        Ground-state doubles and integral contractions add second-order terms,
        plus third-order terms for ADC(3). ADC(2)-x retains ADC(2)'s singles
        intermediate. In the physical ionization matrix, the singles block is
        -M_ij[kshift] for IP and M_ab[kshift] for EA; orbital-energy signs are
        retained in this intermediate.

        Args:
            eris (_ChemistsERIs, optional): Active MO integral container. With
                None, call ao2mo(). Reference t1/t2 must already be available.

            kshift (int): Accepted for the common solver interface. Charged
                intermediates contain all momenta, independent of this value.

        Returns:
            ndarray: Shape (nkpts, nocc, nocc) for IP or (nkpts, nvir, nvir)
            for EA, including the requested perturbative corrections. This is
            the singles intermediate, rather than the complete
            singles-plus-doubles excitation matrix.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> with mykradcip.ao2mo() as eris:
            ...     intermediate = mykradcip.get_imds(eris)
            >>> intermediate.shape
            (2, 1, 1)
        """
        self.method = check_implemented(self.method)
        if eris is None:
            eris = self.ao2mo()
        singles_ham = self._get_adc0_ham()
        self._get_adc2_ham(eris, singles_ham)
        if self.method == 'adc(3)':
            self._get_adc3_ham(eris, singles_ham)
        return singles_ham

    def _init_guess(self, nroots=1, diag=None, ascending=True,
                    type_=None, ini=None, kshift=None):
        r"""Build sector unit guesses or validate supplied root vectors.

        Args:
            nroots (int): Number of guesses, from 1 to vector_size().
            diag (ndarray, optional): Preconditioner of length vector_size().
                With None, obtain it from get_diag(kshift).
            ascending (bool): Choose the lowest diagonal entries first;
                False chooses the highest. Ignored for supplied vectors.
            type_ (str, optional): None selects unit guesses; 'read' selects
                ini. Other values are unsupported.
            ini (array_like, optional): Input of shape (nroots, vector_size()).
                A copy is made and frozen/padded components are zeroed.
            kshift (int, optional): Momentum sector, defaulting to zero.

        Returns:
            list[ndarray]: nroots vectors of length vector_size(). Invalid
            input shapes and zero-norm guesses raise ValueError.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = KRADCIP(kmf)
            >>> guesses = mykradcip._charged_init_guess(2, diag=np.arange(5.0))
            >>> [int(np.argmax(vector)) for vector in guesses]
            [0, 1]
        """
        if type_ not in (None, 'read'):
            raise NotImplementedError(f'Unknown initial-guess type: {type_!r}')
        dim = self.vector_size()
        if not 1 <= nroots <= dim:
            raise ValueError('nroots must be between 1 and vector_size().')
        if type_ == 'read':
            g = np.asarray(ini).T.copy()
            if g.shape != (dim, nroots):
                raise ValueError(
                    f'Shape of guess each k point should be ({dim},{nroots})')
            for root in range(nroots):
                singles = g[:self.n_singles, root]
                doubles = g[self.n_singles:, root].reshape(self.doubles_shape)
                singles, doubles = self._mask_frozen(
                    singles, doubles, 0 if kshift is None else kshift, const=0.0)
                g[:, root] = np.hstack((singles, doubles.ravel()))
        else:
            if diag is None:
                diag = self.get_diag(0 if kshift is None else kshift)
            order = np.argsort(diag)
            if not ascending:
                order = order[::-1]
            g = np.zeros((dim, nroots), dtype=diag.dtype)
            g[order[:nroots], np.arange(nroots)] = 1
        for root in range(nroots):
            if np.linalg.norm(g[:, root]) < LOOSE_ZERO_TOL:
                raise ValueError(f'Guess vector {root} has zero norm; '
                                 'check frozen orbitals and the input.')
        return [g[:, root] for root in range(nroots)]

    def gen_matvec(self, kshift, imds=None, eris=None):
        r"""Prepare a sector action and Davidson diagonal from shared
        integrals.

        Args:
            kshift (int): Momentum sector.
            imds (ndarray, optional): Singles intermediate; None builds it for
                kshift. Charged intermediates contain all one-particle momenta.
            eris (_ChemistsERIs, optional): Open active integral owner. None
                transforms once, then reuses it for the action and diagonal.

        Returns:
            tuple[callable, ndarray]: Matrix action sigma(r) and flat diagonal,
            each using the same singles intermediate and active orbital layout.
            Keep any supplied integral owner open while using the action.
        """
        if eris is None:
            eris = self.ao2mo()
        if imds is None:
            imds = self.get_imds(eris=eris, kshift=kshift)
        return self.matvec(kshift, imds, eris), self.get_diag(kshift, imds, eris)

    def get_properties(self, kshift, U, nroots=1):
        # Transition moments
        r"""Compute charged spectroscopic amplitudes and spin-summed pole
        strengths.

        With effective transition matrix T and metric-normalized U,
        :math:`X=T U` and :math:`P_n=2\sum_p |X_{pn}|^2`. The factor of two
        accounts for restricted spin degeneracy. These strengths are
        dimensionless and are not optical oscillator strengths.

        Args:
            kshift (int): Charged momentum sector.
            U (ndarray): Writable eigenvector columns, shape
                (vector_size(), nroots). Renormalized in place.
            nroots (int): Number of columns to return; must match U.

        Returns:
            tuple[ndarray, ndarray]: Real P of shape (nroots,) and complex X
            of shape (nmo, nroots), in padded active MO order.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> excitation_results = mykradcip.kernel(1, kptlist=[0])
            >>> strengths, amplitudes = mykradcip.get_properties(
            ...     0, mykradcip.U[0].T.copy(), 1)
            >>> strengths.shape, amplitudes.shape
            ((1,), (2, 1))
        """
        T = self.get_trans_moments(kshift)

        # Spectroscopic amplitudes
        U = self.renormalize_eigenvectors(kshift, U, nroots)
        X = np.dot(T, U).reshape(-1, nroots)

        # Spectroscopic factors
        P = 2.0 * einsum('pi,pi->i', X.conj(), X)
        P = P.real

        return P, X

    def _ensure_reference(self):
        r"""Run the underlying KRHF if necessary and fill missing orbital data.

        Returns:
            None: Populate missing MO arrays and scf_energy from the reference.
            Explicit orbital overrides are retained. Reusing a converged SCF
            reference does not trigger another Hartree-Fock calculation.
        """
        if any(value is None for value in
               (self.mo_coeff, self.mo_occ, self.mo_energy)):
            if any(value is None for value in
                   (self._scf.mo_coeff, self._scf.mo_occ, self._scf.mo_energy)):
                self._scf.kernel()
            for name in ('mo_coeff', 'mo_occ', 'mo_energy'):
                if getattr(self, name) is None:
                    setattr(self, name, getattr(self._scf, name))
            self.scf_energy = self._scf.e_tot

    def dump_flags_gs(self, verbose=None):
        r"""Log reference settings using the common periodic ADC flag printer.

        Args:
            verbose (int, optional): Logging level; None uses self.verbose.

        Returns:
            KRADCBase: This object.
        """
        return self.dump_flags(verbose)

    def _dump_orbital_info(self):
        r"""Log the active, frozen and removed orbital counts for each k point.

        Returns:
            None: Write a common INFO table using orbital_number_lists() and
            log_table(). Also log explicit per-k frozen-index lists when
            present.
            The table reports removed FNO orbitals separately from frozen
            virtuals
            and follows the driver's stdout/verbosity settings.
        """
        frozen_occ, active_occ, active_vir, frozen_vir, removed = (
            orbital_number_lists(self))
        data = {'k': range(self.nkpts), 'Frozen occ': frozen_occ,
                'Frozen vir': frozen_vir, 'Active occ': active_occ,
                'Active vir': active_vir, 'Removed': removed}
        log_table(self, data, 'ADC orbital information')
        if (isinstance(self.frozen, (list, tuple, np.ndarray))
                and len(self.frozen)
                and isinstance(self.frozen[0], (list, tuple, np.ndarray))):
            logger.info(self, f'Frozen orbital lists: {self.frozen}')

    def amplitudes(self, eris=None):
        r"""Populate self.t1, self.t2 and the ladder intermediate.

        First-order doubles use :math:`t_{ij}^{ab,(1)} =
        (ia|jb)^*/(\epsilon_i+\epsilon_j-\epsilon_a-\epsilon_b)`.
        Three independent k axes fix the fourth by momentum conservation.
        ADC(3) adds second-order doubles. Ground-state singles through the
        orders needed for the requested transition-moment approximation
        are retained in t1; omitted orders are None.

        Args:
            eris (_ChemistsERIs, optional): Active-space integral container.

        Returns:
            None: Results remain on the driver. Doubles are backed by the
            scratch file owned by this instance.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> with mykradcbase.ao2mo() as eris:
            ...     mykradcbase.amplitudes(eris)
            ...     print(mykradcbase.t2[0].shape)
            (2, 2, 2, 1, 1, 1, 1)
        """
        if eris is None:
            eris = self.ao2mo()

        cput0 = (time.process_time(), time.time())
        log = logger.Logger(self.stdout, self.verbose)

        self.method = check_implemented(self.method)

        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        nkpts = self.nkpts
        cell = self.cell
        kpts = self.kpts
        madelung = tools.madelung(cell, kpts)

        # release the scratch file from a previous kernel call, if any
        self._close_amp_scratch()

        # Compute first-order doubles t2 (tijab)
        self._amp_storage = _AmplitudeStorage()
        tf, f = self._amp_storage.temporary_file, self._amp_storage.file
        # Keep scratch storage alive while converted ADC objects share amplitudes.
        self._amp_h5file = f
        self._amp_tmpfile = tf
        t2_1 = f.create_dataset(
            't2_1',
            (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir),
            dtype=eris.ovov.dtype)

        mo_e_o, mo_e_v = self.e_occ, self.e_vir

        # Locate occupied and virtual elements that are not padding.
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind='split')

        kconserv = self.khelper.kconserv
        touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)

        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            if touched[ki, kj, ka]:
                continue

            kb = kconserv[ki, ka, kj]
            # For discussion of LARGE_DENOM, see t1new update above
            eia = _get_epq(
                [0, nocc, ki, mo_e_o, nonzero_opadding],
                [0, nvir, ka, mo_e_v, nonzero_vpadding],
                fac=[1.0, -1.0])

            ejb = _get_epq(
                [0, nocc, kj, mo_e_o, nonzero_opadding],
                [0, nvir, kb, mo_e_v, nonzero_vpadding],
                fac=[1.0, -1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            t2_1[ki, kj, ka] = eris.ovov[ki, ka, kj].conj().transpose(
                (0, 2, 1, 3)) / eijab

            if ka != kb:
                eijba = eijab.transpose(0, 1, 3, 2)
                t2_1[ki, kj, kb] = eris.ovov[ki, kb, kj].conj().transpose(
                    (0, 2, 1, 3)) / eijba

            touched[ki, kj, ka] = touched[ki, kj, kb] = True

        cput0 = log.timer_debug1(
            'Completed t2_1 amplitude calculation', *cput0)

        t1_2 = None
        if self.approx_trans_moments is False or self.method == "adc(3)":
            # Compute second-order singles t1 (tij)
            t1_2 = np.zeros((nkpts, nocc, nvir), dtype=t2_1.dtype)
            eris_ovoo = eris.ovoo
            for ki in range(nkpts):
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        kd = kconserv[ki, kc, kk]
                        ka = kconserv[kc, kk, kd]

                        if eris.ovvv is None:
                            chunk_size = self.chunk_size
                            chunk_size = min(chunk_size, nocc)
                            a = 0
                            for p in range(0, nocc, chunk_size):
                                eris_ovvv = _ChemistsERIs.ovvv_df(
                                    eris.Lov[kk, kd],
                                    eris.Lvv[ka, kc],
                                    p,
                                    chunk_size).reshape(
                                    -1, nvir, nvir, nvir) / nkpts
                                k = eris_ovvv.shape[0]
                                t1_2[ki] += 1.5 * einsum(
                                    'kdac,ikcd->ia',
                                    eris_ovvv, t2_1[ki, kk, kc, :, a:a + k])
                                t1_2[ki] -= 0.5 * einsum(
                                    'kdac,kicd->ia',
                                    eris_ovvv, t2_1[kk, ki, kc, a:a + k, :])
                                del eris_ovvv
                                eris_ovvv = _ChemistsERIs.ovvv_df(
                                    eris.Lov[kk, kc],
                                    eris.Lvv[ka, kd],
                                    p,
                                    chunk_size).reshape(
                                    -1, nvir, nvir, nvir) / nkpts
                                t1_2[ki] -= 0.5 * einsum(
                                    'kcad,ikcd->ia',
                                    eris_ovvv, t2_1[ki, kk, kc, :, a:a + k])
                                t1_2[ki] += 0.5 * einsum(
                                    'kcad,kicd->ia',
                                    eris_ovvv, t2_1[kk, ki, kc, a:a + k, :])
                                del eris_ovvv
                                a += k
                        else:
                            eris_ovvv = eris.ovvv[:]
                            t1_2[ki] += 1.5 * einsum(
                                'kdac,ikcd->ia',
                                eris_ovvv[kk, kd, ka], t2_1[ki, kk, kc])
                            t1_2[ki] -= 0.5 * einsum(
                                'kdac,kicd->ia',
                                eris_ovvv[kk, kd, ka], t2_1[kk, ki, kc])
                            t1_2[ki] -= 0.5 * einsum(
                                'kcad,ikcd->ia',
                                eris_ovvv[kk, kc, ka], t2_1[ki, kk, kc])
                            t1_2[ki] += 0.5 * einsum(
                                'kcad,kicd->ia',
                                eris_ovvv[kk, kc, ka], t2_1[kk, ki, kc])
                            del eris_ovvv

                    for kl in range(nkpts):
                        kc = kconserv[kk, ki, kl]
                        ka = kconserv[kl, kc, kk]

                        t1_2[ki] -= 1.5 * einsum(
                            'lcki,klac->ia',
                            eris_ovoo[kl, kc, kk], t2_1[kk, kl, ka])
                        t1_2[ki] += 0.5 * einsum(
                            'lcki,lkac->ia',
                            eris_ovoo[kl, kc, kk], t2_1[kl, kk, ka])
                        t1_2[ki] -= 0.5 * einsum(
                            'kcli,lkac->ia',
                            eris_ovoo[kk, kc, kl], t2_1[kl, kk, ka])
                        t1_2[ki] += 0.5 * einsum(
                            'kcli,klac->ia',
                            eris_ovoo[kk, kc, kl], t2_1[kk, kl, ka])

            for ki in range(nkpts):
                ka = ki
                eia = _get_epq(
                    [0, nocc, ki, mo_e_o, nonzero_opadding],
                    [0, nvir, ka, mo_e_v, nonzero_vpadding],
                    fac=[1.0, -1.0])
                t1_2[ki] = t1_2[ki] / eia

            cput0 = log.timer_debug1(
                'Completed t1_2 amplitude calculation', *cput0)

        t2_2 = None
        t1_3 = None
        t2_1_vvvv = None

        if (self.method == 'adc(3)' or
                (not self.approx_trans_moments and
                 (self.method == 'adc(2)-x' or isinstance(self, KRADCEE)))):
            # Compute second-order doubles t2 (tijab)
            t2_1_vvvv = f.create_dataset(
                't2_1_vvvv',
                (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir),
                dtype=eris.ovov.dtype)
            eris_oooo = eris.oooo
            eris_ovvo = eris.ovvo
            eris_oovv = eris.oovv

            for ka, kb, kc in kpts_helper.loop_kkk(nkpts):
                kd = kconserv[ka, kc, kb]
                for ki in range(nkpts):
                    kj = kconserv[ka, ki, kb]
                    if isinstance(eris.vvvv, np.ndarray):
                        amplitude = t2_1[ki, kj, kc].reshape(
                            nocc * nocc, nvir * nvir)
                        integral = eris.vvvv[kc, kd, ka].reshape(
                            nvir * nvir, nvir * nvir)
                        t2_1_vvvv[ki, kj, ka] += (
                            amplitude @ integral.conj()).reshape(
                            nocc, nocc, nvir, nvir)
                    elif eris.vvvv is None:
                        t2_1_vvvv[ki, kj, ka] += self._contract_ladder(
                            t2_1[ki, kj, kc], eris.Lvv, ka, kb, kc)
                    else:
                        t2_1_vvvv[ki, kj, ka] += self._contract_ladder(
                            t2_1[ki, kj, kc], eris.vvvv, kc, kd, ka)

            t2_2 = f.create_dataset(
                't2_2',
                (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir),
                dtype=eris.ovov.dtype)
            for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
                amplitude = t2_1_vvvv[ki, kj, ka]
                if self.exxdiv is not None:
                    amplitude -= 2.0 * madelung * t2_1[ki, kj, ka]
                for kk in range(nkpts):

                    kc = kconserv[ki, ka, kk]
                    kb = kconserv[kj, kk, kc]
                    amplitude -= einsum(
                        'kjbc,kica->ijab',
                        eris_oovv[kk, kj, kb], t2_1[kk, ki, kc])

                    kc = kconserv[kk, ka, kj]
                    kb = kconserv[ki, kk, kc]
                    amplitude -= einsum(
                        'kibc,jkca->ijab',
                        eris_oovv[kk, ki, kb], t2_1[kj, kk, kc])

                    kc = kconserv[ka, kj, kk]
                    kb = kconserv[ki, kc, kk]
                    amplitude -= einsum(
                        'kjac,ikcb->ijab',
                        eris_oovv[kk, kj, ka], t2_1[ki, kk, kc])

                    kc = kconserv[ka, ki, kk]
                    kb = kconserv[kk, kc, kj]
                    amplitude -= einsum(
                        'kiac,kjcb->ijab',
                        eris_oovv[kk, ki, ka], t2_1[kk, kj, kc])

                for kl in range(nkpts):
                    kk = kconserv[kj, kl, ki]
                    amplitude += einsum(
                        'kilj,klab->ijab',
                        eris_oooo[kk, ki, kl], t2_1[kk, kl, ka])

                for kk in range(nkpts):

                    kc = kconserv[ki, ka, kk]
                    kb = kconserv[kc, kk, kj]

                    amplitude += 2 * einsum(
                        'kcbj,kica->ijab',
                        eris_ovvo[kk, kc, kb], t2_1[kk, ki, kc])
                    kc = kconserv[kk, ka, ki]
                    amplitude -= einsum(
                        'kcbj,ikca->ijab',
                        eris_ovvo[kk, kc, kb], t2_1[ki, kk, kc])

                    kc = kconserv[kk, ki, ka]
                    kb = kconserv[kk, kj, kc]
                    amplitude += 2 * einsum(
                        'kcai,kjcb->ijab',
                        eris_ovvo[kk, kc, ka], t2_1[kk, kj, kc])

                    kc = kconserv[kk, ki, ka]
                    kb = kconserv[kj, kk, kc]
                    amplitude -= einsum(
                        'kcai,jkcb->ijab',
                        eris_ovvo[kk, kc, ka], t2_1[kj, kk, kc])

                kb = kconserv[ki, ka, kj]
                eia = _get_epq(
                    [0, nocc, ki, mo_e_o, nonzero_opadding],
                    [0, nvir, ka, mo_e_v, nonzero_vpadding],
                    fac=[1.0, -1.0])

                ejb = _get_epq(
                    [0, nocc, kj, mo_e_o, nonzero_opadding],
                    [0, nvir, kb, mo_e_v, nonzero_vpadding],
                    fac=[1.0, -1.0])
                eijab = eia[:, None, :, None] + ejb[:, None, :]

                t2_2[ki, kj, ka] = amplitude / eijab

            cput0 = log.timer_debug1(
                'Completed t2_2 amplitude calculation', *cput0)

            if (self.method == "adc(3)"
                    and self.approx_trans_moments is False):
                from pyscf.pbc.adc.lib import MomentumContractions
                from pyscf.pbc.adc.kadc_ee_terms import SINGLES_AMPLITUDES_3
                context = MomentumContractions(
                    self, eris, t1=(t1_2, None), t2=(t2_1, t2_2))
                t1_3 = np.zeros_like(t1_2)
                for ki in range(nkpts):
                    for term in SINGLES_AMPLITUDES_3:
                        value = context.contract(
                            term, {'i': ki, 'a': ki}, (1, -1))
                        if value is not None:
                            t1_3[ki] += value
                    denominator = _get_epq(
                        [0, nocc, ki, mo_e_o, nonzero_opadding],
                        [0, nvir, ki, mo_e_v, nonzero_vpadding],
                        fac=[1.0, -1.0])
                    t1_3[ki] /= denominator

        self.t1 = (t1_2, t1_3)
        self.t2 = (t2_1, t2_2)
        self.imds.t2_1_vvvv = t2_1_vvvv

    def correlation_energy(self, eris=None):
        r"""Contract doubles with integrals and store the correlation energy.

        With the stored, k-normalized integrals,

        .. math::
            E_c = N_k^{-1} \sum_{k_i k_j k_a,ijab}
            t_{ij}^{ab} [2(ia|jb) - (ja|ib)].

        ADC(2)/ADC(2)-x use first-order doubles; ADC(3) also includes their
        second-order correction. Only one orbital block is read at a time.

        Args:
            eris (_ChemistsERIs, optional): Active-space integrals. If
                amplitudes are absent they are computed first. With
                if_corr=False, the stored and returned contribution is zero.

        Returns:
            float: Correlation energy in Hartree per primitive cell.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> with mykradcbase.ao2mo() as eris:
            ...     mykradcbase.amplitudes(eris)
            ...     energy = mykradcbase.correlation_energy(eris)
            >>> round(energy, 8)
            -0.00416766
        """
        if eris is None:
            eris = self.ao2mo()
        if self.t2 is None:
            self.amplitudes(eris)
        if not self.if_corr:
            self.e_corr = 0.0
            return self.e_corr
        energy = 0.0
        for ki, kj, ka in kpts_helper.loop_kkk(self.nkpts):
            amplitude = self.t2[0][ki, kj, ka]
            if self.method == 'adc(3)':
                amplitude = amplitude + self.t2[1][ki, kj, ka]
            energy += 2 * einsum('ijab,iajb', amplitude, eris.ovov[ki, ka, kj])
            energy -= einsum('ijab,jaib', amplitude, eris.ovov[kj, ka, ki])
        self.e_corr = float(energy.real / self.nkpts)
        return self.e_corr

    def _finalize(self):
        r"""Log the computed reference correlation energy per primitive cell.

        Returns:
            KRADCBase: This object with e_corr stored in Hartree per cell.
        """
        logger.note(self, 'MP%s correlation energy per cell (Eh) = %.8f',
                    self.method[4], self.e_corr)
        return self

    def _mask_frozen(self, singles, doubles, kshift, const=0.0):
        r"""Return neutral vector blocks unchanged; charged classes mask
        padding.

        Args:
            singles (ndarray): Flattened singles block.
            doubles (ndarray): Doubles block with shape doubles_shape.
            kshift (int): Momentum sector.
            const (number): Replacement for masked entries in charged sectors.

        Returns:
            tuple[ndarray, ndarray]: Supplied blocks for EE. IP/EA override
            this method to replace inactive entries with const.
        """
        return singles, doubles

    def get_trans_moments(self, kshift):
        r"""Assemble the charged transition matrix for every active MO.

        Each row is the effective creation/annihilation-operator vector for
        one MO at kshift. It maps normalized ADC eigenvectors to spectroscopic
        amplitudes as :math:`X=T U`; this T does not yet contain a root axis.

        Args:
            kshift (int): Charged momentum sector and orbital k point.

        Returns:
            ndarray: Complex matrix of shape (nmo, vector_size()). MO rows use
            padded active occupied-then-virtual ordering. Perturbative terms
            follow method and approx_trans_moments.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> mykradcip.get_trans_moments(0).shape
            (2, 5)
        """
        nmo = self.nmo
        T = []
        for orb in range(nmo):
            T_a = self.get_trans_moments_orbital(orb, kshift)
            T.append(T_a)

        T = np.array(T)
        return T

    def _close_amp_scratch(self):
        r"""Release this instance's references to shared amplitude storage.

        Returns:
            None: Scratch storage closes after its last ADC owner releases it.
            Converted objects retain valid amplitudes when the source reruns
            or is deleted. Repeated cleanup is harmless.
        """
        self.__dict__.pop('_amp_h5file', None)
        self.__dict__.pop('_amp_tmpfile', None)
        self.__dict__.pop('_amp_storage', None)

    def _contract_ladder(self, t_amp, vvvv, ka, kb, kc):
        r"""Contract a ground-state doubles block with a particle-particle
        ladder.

        For fixed momenta, multiply the flattened occupied-pair/virtual-pair
        amplitudes by conjugated virtual-integral slices. Reconstruct at most
        chunk_size virtual rows at a time when the integrals come from DF
        factors.

        Args:
            t_amp (ndarray): Doubles block, shape (nocc, nocc, nvir, nvir).
            vvvv (ndarray or indexable dataset): An ndarray is interpreted as
                five-axis Lvv factors, shape (nkpts, nkpts, naux, nvir, nvir).
                Otherwise use stored normalized four-virtual blocks with three
                leading k axes. Stored dense NumPy ERIs use the caller's
                separate
                incore contraction path rather than this factor-based branch.
            ka, kb, kc (int): Particle momenta; kd = kconserv[ka, kc, kb].

        Returns:
            ndarray: Ladder contribution of shape (nocc, nocc, nvir, nvir).
            Reconstructed DF integrals are divided by nkpts; stored blocks
            already
            have that normalization. The input amplitude is not modified.
        """

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        kd = kconserv[ka, kc, kb]
        t_amp = np.ascontiguousarray(t_amp.reshape(nocc * nocc, nvir * nvir))
        t = np.zeros((nocc, nocc, nvir, nvir), dtype=t_amp.dtype)
        chunk_size = self.chunk_size
        chunk_size = min(chunk_size, nvir)
        a = 0
        if isinstance(vvvv, np.ndarray):
            vv1 = vvvv[kc, ka]
            vv2 = vvvv[kd, kb]
            for p in range(0, nvir, chunk_size):
                vvvv_p = _ChemistsERIs.vvvv_df(vv1, vv2, p, chunk_size) / nkpts
                vvvv_p = vvvv_p.reshape(-1, nvir * nvir)
                k = vvvv_p.shape[0]
                t += np.dot(t_amp[:, a:a + k], vvvv_p.conj()).reshape(
                    nocc, nocc, nvir, nvir)
                del vvvv_p
                a += k
        else:
            for p in range(0, nvir, chunk_size):
                vvvv_p = vvvv[ka, kb, kc, p:p + chunk_size, :, :, :].reshape(
                    -1, nvir * nvir)
                k = vvvv_p.shape[0]
                t += np.dot(t_amp[:, a:a + k], vvvv_p.conj()).reshape(
                    nocc, nocc, nvir, nvir)
                del vvvv_p
                a += k

        return t

    def __setattr__(self, name, value):
        r"""Assign an attribute, rejecting the removed sector-string settings.

        Args:
            name (str): Attribute name.
            value: New value.

        Raises:
            AttributeError: type_ or method_type is supplied. Construct
                KRADCIP, KRADCEA or KRADCEE to select the sector; type(obj)
                identifies it.
        """
        if name in ('type_', 'method_type'):
            raise AttributeError('Select a sector with KRADCIP, KRADCEA or '
                                 'KRADCEE; inspect it with type(obj).')
        super().__setattr__(name, value)

    def transform_integrals(self, backend=None):
        r"""Transform active integrals through the earlier method name.

        Args:
            backend (str, optional): Backend override passed to ao2mo(); None
                uses integral_backend. See AO2MO for supported storage choices.

        Returns:
            _ChemistsERIs: Owning normalized integral container. This is an
            alias
            operation for ao2mo(), whose examples and cleanup rules apply here.

        Warns:
            DeprecationWarning: Use ao2mo() for new calls. The same backend and
            integral-container ownership rules apply to both entry points.
        """
        warnings.warn(
            'transform_integrals is deprecated; use ao2mo() instead.',
            DeprecationWarning,
            stacklevel=2)
        return self.ao2mo(backend)

    def get_chnk_size(self):
        r"""Compute an automatic chunk estimate through the earlier method
        interface.

        Returns:
            int: Fresh estimate from kadc_ao2mo.chunk_size(), using active
            virtual
            dimensions and current memory usage. Unlike reading chunk_size,
            this
            method does not return or update the cached/manual property value.
        """
        return kadc_ao2mo.chunk_size(self)

    def _pad_orbitals(self):
        r"""Construct or reuse active-space coefficients and energies with KMP2
        padding.

        Returns:
            tuple[ndarray, ndarray, ndarray]: Cached (coefficients, occupied
            energies, virtual energies), shaped (nkpts, nao, nmo), (nkpts,
            nocc)
            and (nkpts, nvir). Energies are in Hartree. Arrays are shared
            cached
            objects; changing reference orbitals requires
            _invalidate_orbitals()
            before a fresh padding pass.
        """
        self._ensure_reference()
        if self._padded_orbitals is None:
            coeff, energy = _add_padding(self, self.mo_coeff, self.mo_energy)
            energy = np.asarray(energy)
            self._padded_orbitals = (coeff, energy[:, :self.nocc],
                                    energy[:, self.nocc:])
        return self._padded_orbitals

    def _orbital_data(self):
        r"""Return the cached padded orbital tuple through the earlier helper
        name.

        Returns:
            tuple[ndarray, ndarray, ndarray]: padded_coeff, e_occ and e_vir,
            with shapes (nkpts, nao, nmo), (nkpts, nocc) and (nkpts, nvir).
            New callers should read the individual properties, which share this
            same cache rather than recomputing padding.
        """
        return self._pad_orbitals()

    def _init_from_ground_state(self, adc):
        r"""Copy shared reference data and its amplitude-storage owner.

        Args:
            adc (KRADCBase): Reference or sector solver to convert.

        Returns:
            None: Preserve common reference/settings attributes and dimensional
            overrides. Copy mutable bookkeeping and share the amplitude-storage
            owner without retaining the source solver. Sector results, caches
            and FNO-only settings are excluded from the converted object.
        """
        for key in self._reference_keys:
            if hasattr(adc, key):
                setattr(self, key, getattr(adc, key))
        for key in ('_scf', '_nocc', '_nmo', '_nvir', '_chunk_size',
                    '_chunk_size_manual', 'stdout', 'verbose', 'max_memory'):
            setattr(self, key, getattr(adc, key))
        self.imds = copy.copy(adc.imds)
        self.frozen = copy.deepcopy(adc.frozen)
        self.density_gv = np.asarray(adc.density_gv).copy()
        self._amp_storage = getattr(adc, '_amp_storage', None)

    def get_init_guess(self, nroots=1, diag=None, ascending=True,
                       type_=None, ini=None, kshift=None, **legacy):
        r"""Build trial vectors through the deprecated common guess interface.

        Args:
            nroots (int): Number of guesses, from 1 to vector_size().
            diag (ndarray, optional): Preconditioner diagonal; with None, the
                sector helper obtains it for kshift.
            ascending (bool): Select low diagonal entries first, or high
                entries
                when False. Ignored for supplied vectors.
            type_ (str, optional): None for unit guesses; 'read' to validate
                ini.
            ini (array_like, optional): Supplied vectors, shape
                (nroots, vector_size()), copied before use.
            kshift (int, optional): Momentum sector, default zero.
            **legacy: The earlier keyword type is accepted as an alias for
            type_.
                Supplying both spellings or other keywords raises TypeError.

        Returns:
            list[ndarray]: One flat vector per requested root. The shared
            _init_guess helper masks inactive charged components; EE uses its
            full vector layout. Invalid shapes and zero norms raise
            ValueError.

        A DeprecationWarning is emitted. This entry point is intended for
        sector
        solvers; the kernel calls their protected helpers directly.

        Examples:
            >>> import numpy as np
            >>> import warnings
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = KRADCIP(kmf)
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter('always', DeprecationWarning)
            ...     guesses = mykradcip.get_init_guess(1, diag=np.arange(5.0))
            >>> np.shape(guesses), caught[0].category.__name__
            ((1, 5), 'DeprecationWarning')
        """
        warnings.warn('get_init_guess is deprecated; use _init_guess.',
                      DeprecationWarning, stacklevel=2)
        if 'type' in legacy:
            if type_ is not None:
                raise TypeError('Pass only one of type_ and the legacy type.')
            type_ = legacy.pop('type')
        if legacy:
            names = ', '.join(sorted(legacy))
            raise TypeError(f'Unexpected initial-guess keyword(s): {names}')
        return self._init_guess(nroots, diag, ascending, type_, ini, kshift)

    def ground_state_energy(self, eris=None):
        r"""Compute and store reference amplitudes and correlation energy.

        Args:
            eris (_ChemistsERIs, optional): Pretransformed active integrals.

        Returns:
            float: Correlation energy per primitive cell, in Hartree.
                The amplitudes are available as self.t1 and self.t2.
                This is the correlation contribution, not the SCF total.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> round(mykradcbase.ground_state_energy(), 8)
            -0.00416766
            >>> mykradcbase.t2[0].shape
            (2, 2, 2, 1, 1, 1, 1)
        """
        started = logger.process_clock(), logger.perf_counter()
        self._compute_reference(eris)
        logger.timer(self, 'ground-state ADC', *started)
        return self.e_corr

    def kernel_gs(self, eris=None):
        r"""Compute the reference through the deprecated tuple-returning
        interface.

        Args:
            eris (_ChemistsERIs, optional): Pretransformed active MO integrals.

        Returns:
            tuple: (e_corr, t1, t2), also stored on the driver. e_corr is the
            correlation contribution in Hartree per primitive cell. A
            DeprecationWarning is emitted; ground_state_energy() returns the
            scalar energy and leaves both amplitude tuples on the instance.
        """
        warnings.warn('kernel_gs is deprecated; use ground_state_energy() '
                      'and read t1 and t2 from the ADC instance.',
                      DeprecationWarning, stacklevel=2)
        self.ground_state_energy(eris)
        return self.e_corr, self.t1, self.t2

    def __del__(self):
        r"""Release owned amplitude scratch files during object finalization.

        Returns:
            None: Delegate to _close_amp_scratch(). Shared scratch arrays
            remain valid while another converted ADC retains their owner.
        """
        self._close_amp_scratch()

    def ip_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        r"""Solve IP-ADC using previously computed reference amplitudes.

        Args:
            nroots (int): Roots per selected momentum sector, capped at the
                sector vector dimension.
            guess (array_like, optional): Trial vectors with shape
                (len(kptlist), nroots, vector_size()), in requested sector
                order.
            eris (_ChemistsERIs, optional): Pretransformed active MO integrals;
                with None, the sector solver calls ao2mo().
            kptlist (sequence[int], optional): Sector indices, defaulting to
                every k point. Prepare amplitudes with ground_state_energy()
                before invoking this lower-level entry point.

        Returns:
            tuple: E, U, P, X and the sector solver. The first four entries
            follow kernel()'s charged shapes and units.
            Read T/converged and other sector-specific data from the returned
            solver. This helper does not attach it as the driver's _adc_es.

        Prefer kernel() and result attributes in user code; this low-level
        helper retains its return tuple for the common driver dispatch.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = KRADCIP(kmf).run(0)
            >>> excitation_results = mykradcip.kernel(1, kptlist=[0])
            >>> energies = mykradcip.E
            >>> energies.round(6).tolist()
            [[0.371589]]
        """
        from pyscf.pbc.adc import kadc_rhf_ip
        mykradcip = KRADCIP(self)
        e_exc, v_exc, spec_fac, x = mykradcip.kernel(
            nroots, guess, eris, kptlist)
        mykradcip.U = v_exc
        return e_exc, v_exc, spec_fac, x, mykradcip

    def ea_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        r"""Solve EA-ADC using previously computed reference amplitudes.

        Args:
            nroots (int): Roots per selected momentum sector, capped at the
                sector vector dimension.
            guess (array_like, optional): Trial vectors with shape
                (len(kptlist), nroots, vector_size()), in requested sector
                order.
            eris (_ChemistsERIs, optional): Pretransformed active MO integrals;
                with None, the sector solver calls ao2mo().
            kptlist (sequence[int], optional): Sector indices, defaulting to
                every k point. Prepare amplitudes with ground_state_energy()
                before invoking this lower-level entry point.

        Returns:
            tuple: E, U, P, X and the sector solver. The first four entries
            follow kernel()'s charged shapes and units.
            Read T/converged and other sector-specific data from the returned
            solver. This helper does not attach it as the driver's _adc_es.

        Prefer kernel() and result attributes in user code; this low-level
        helper retains its return tuple for the common driver dispatch.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcea = KRADCEA(kmf).run(0)
            >>> excitation_results = mykradcea.kernel(1, kptlist=[0])
            >>> energies = mykradcea.E
            >>> energies.round(6).tolist()
            [[1.558245]]
        """
        from pyscf.pbc.adc import kadc_rhf_ea
        mykradcea = KRADCEA(self)
        e_exc, v_exc, spec_fac, x = mykradcea.kernel(
            nroots, guess, eris, kptlist)
        mykradcea.U = v_exc
        return e_exc, v_exc, spec_fac, x, mykradcea

    def ee_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        r"""Solve EE-ADC using previously computed reference amplitudes.

        Args:
            nroots (int): Roots per selected momentum sector, capped at the
                sector vector dimension.
            guess (array_like, optional): Trial vectors with shape
                (len(kptlist), nroots, vector_size()), in requested sector
                order.
            eris (_ChemistsERIs, optional): Pretransformed active MO integrals;
                with None, the sector solver calls ao2mo().
            kptlist (sequence[int], optional): Sector indices, defaulting to
                every k point. Prepare amplitudes with ground_state_energy()
                before invoking this lower-level entry point.

        Returns:
            tuple: E, U, P, X and the sector solver. The first four entries
            follow kernel()'s neutral shapes and units.
            Read T/converged and other sector-specific data from the returned
            solver. This helper does not attach it as the driver's _adc_es.

        Prefer kernel() and result attributes in user code; this low-level
        helper retains its return tuple for the common driver dispatch.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = KRADCEE(kmf).run(0)
            >>> excitation_results = mykradcee.kernel(1, kptlist=[0])
            >>> energies = mykradcee.E
            >>> energies.round(6).tolist()
            [[1.756312]]
        """
        from pyscf.pbc.adc import kadc_rhf_ee
        mykradcee = KRADCEE(self)
        e_exc, v_exc, spec_fac, x = mykradcee.kernel(
            nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, mykradcee

    def density_fit(self, auxbasis=None, with_df=None):
        r"""Attach a density-fitting object for subsequent transformations.

        Args:
            auxbasis: Auxiliary basis passed to the periodic DF object.
                With None, inherit the SCF object's auxiliary basis.
            with_df (DF, optional): Existing DF object to attach directly.
                With None, create one with this driver's logging and memory
                settings.

        Returns:
            KRADCBase: This instance, for chained configuration calls.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> mykradcbase.density_fit(with_df=kmf.with_df) is mykradcbase
            True
            >>> mykradcbase.with_df is kmf.with_df
            True
        """
        if with_df is None:
            self.with_df = df.DF(self.cell, self.kpts)
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

    def _make_rdm1(self, root=None, kptlist=None, K_idx=None, if_ss=False):
        r"""Build active-space one-particle densities for saved charged roots.

        Copy and metric-normalize the saved eigenvectors before contracting
        them. Use make_rdm1() to restore frozen MOs or transform
        these active MO densities to the AO basis.

        Args:
            root (sequence[sequence[int]]): Root indices for each requested
                sector. Although the signature allows None, callers must
                provide the index lists; the driver prepares defaults.
            kptlist (sequence[int]): Sectors in the order used by saved U.
            K_idx (sequence[int]): Sectors to return, in the requested order.
                kptlist and K_idx must be supplied to this low-level method.
            if_ss (bool): Add an extra reference-density contribution for the
                state-specific/state-averaged FNO construction.

        Returns:
            list[ndarray]: One entry per requested sector, each with shape
            (number_of_requested_roots, nkpts, nmo, nmo). The MO axes retain
            active padding. No frozen-orbital restoration is performed here.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> excitation_results = mykradcip.kernel(1, kptlist=[0])
            >>> densities = mykradcip._make_rdm1([[0]], [0], [0])
            >>> np.shape(densities)
            (1, 1, 2, 2, 2)
            >>> round(float(
            ...     np.trace(densities[0][0], axis1=1, axis2=2).real.mean()
            ... ), 6)
            1.5
        """
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        rdm1 = []

        for kpt, kshift in enumerate(K_idx):
            k = kptlist.index(kshift)
            U = np.array(self.U[k]).T.copy()
            U = self.renormalize_eigenvectors(kshift, U, self.U.shape[1])
            rdm1_k = []
            for i in root[kpt]:
                rdm1_k.append(self.make_rdm1_eigenvectors(
                    U[:, i], U[:, i], kshift, if_ss))
            rdm1_k = np.stack(rdm1_k, axis=0)
            rdm1.append(rdm1_k)
        cput0 = log.timer_debug1('completed OPDM calculation', *cput0)
        return rdm1

    def make_rdm1(self, kptlist=None, root=None, K_idx=None,
                  with_frozen=True, ao_repr=False, if_ss=False):
        r"""Build charged-state one-particle densities after an IP/EA run.

        Args:
            kptlist (sequence[int], optional): Momentum sectors represented in
                saved U, in their original order. None uses the last kernel
                run.
            root (sequence[sequence[int]], optional): Root indices requested
                for each selected sector. With None, include all roots.
            K_idx (sequence[int], optional): Sectors to return; defaults to
                kptlist. Root counts may differ between selected sectors.
            with_frozen (bool): Restore frozen MOs and their occupied density.
            ao_repr (bool): Return AO matrices instead of padded active MO
                matrices (or full MO matrices when frozen MOs are restored).
            if_ss (bool): Include an additional reference-density contribution.

        Returns:
            list[ndarray] or ndarray: Densities with axes (sector, root, k, p,
            q). The final two dimensions are nao in AO representation,
            otherwise the corresponding active/full MO count. Neutral
            transition densities are available separately as X.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = KRADCIP(kmf)
            >>> excitation_results = mykradcip.kernel(1, kptlist=[0])
            >>> density = mykradcip.make_rdm1([0], root=[[0]], K_idx=[0])
            >>> np.shape(density)
            (1, 1, 2, 2, 2)
        """
        if not isinstance(self, (KRADCIP, KRADCEA)):
            raise NotImplementedError('State densities are available for IP/EA; '
                                      'EE transition densities are stored in X.')
        if self.U is None:
            raise ValueError('Run kernel() before requesting a state density.')
        if kptlist is None:
            kptlist = self._kptlist
        kptlist = list(kptlist)
        K_idx = kptlist if K_idx is None else list(K_idx)
        if root is None:
            root = [list(range(self.U.shape[1])) for kshift in K_idx]
        if len(root) != len(K_idx):
            raise ValueError('Supply one root list per selected momentum sector.')
        if any(kshift not in kptlist for kshift in K_idx):
            raise ValueError('Density sectors must be present in kptlist.')
        if any(len(roots) == 0 or any(not isinstance(i, (int, np.integer))
                               or not 0 <= i < self.U.shape[1] for i in roots)
               for roots in root):
            raise ValueError('Root lists must contain valid saved root indices.')
        densities = self._make_rdm1(root, kptlist, K_idx, if_ss)
        if with_frozen and self.frozen is not None:
            nmo = max(len(occ) for occ in self.mo_occ)
            mask = get_frozen_mask(self)
            active_indices = padding_k_idx(self, kind='joint')
            restored = []
            for sector_density in densities:
                dm = np.zeros((len(sector_density), self.nkpts, nmo, nmo),
                              dtype=sector_density.dtype)
                for kpt in range(self.nkpts):
                    occupied = np.flatnonzero(np.asarray(self.mo_occ[kpt]) > 0)
                    dm[:, kpt, occupied, occupied] = 4 if if_ss else 2
                    mo_indices = np.flatnonzero(mask[kpt])
                    padded = active_indices[kpt]
                    dm[:, kpt, mo_indices[:, None], mo_indices] = (
                        sector_density[:, kpt, padded[:, None], padded])
                restored.append(dm)
            densities = restored
            if ao_repr:
                coefficients = np.zeros((self.nkpts,
                                         self.mo_coeff[0].shape[0], nmo),
                                        dtype=np.result_type(*self.mo_coeff))
                for kpt, coeff in enumerate(self.mo_coeff):
                    coefficients[kpt, :, :coeff.shape[1]] = coeff
        elif ao_repr:
            coefficients = padded_mo_coeff(self, self.mo_coeff)
        if ao_repr:
            densities = [einsum('kpI,SkIJ,kqJ->Skpq', coefficients, dm,
                                np.asarray(coefficients).conj())
                         for dm in densities]
        # Distinct selected root counts remain a list rather than an object array.
        if len({len(dm) for dm in densities}) == 1:
            return np.asarray(densities)
        return densities

    def compute_amplitudes(self, eris):
        r"""Populate reference amplitudes through the deprecated tuple
        interface.

        Args:
            eris (_ChemistsERIs): Active-space MO integrals.

        Returns:
            tuple: t1, t2 and the first-order-doubles virtual-ladder
            intermediate.
            The same quantities remain on self and self.imds. Emit a
            DeprecationWarning; new callers use amplitudes(eris), which returns
            None, then read the stored attributes.
        """
        warnings.warn(
            'Use amplitudes() and the t1/t2 instance attributes.',
            DeprecationWarning,
            stacklevel=2)
        self.amplitudes(eris)
        return self.t1, self.t2, self.imds.t2_1_vvvv

    def compute_energy(self, t2, eris):
        r"""Evaluate a supplied doubles tuple through the deprecated energy
        interface.

        Args:
            t2 (tuple): First-/second-order doubles, in the layout of self.t2.
            eris (_ChemistsERIs): Active-space MO integrals matching those
                amplitudes.

        Returns:
            float: Correlation contribution in Hartree per primitive cell, with
            method and if_corr following the driver. Temporarily substitute t2
            while calling correlation_energy(), then restore the earlier t2 and
            e_corr even if an exception occurs. Emit a DeprecationWarning.
        """
        warnings.warn(
            'Use correlation_energy() and the e_corr attribute.',
            DeprecationWarning,
            stacklevel=2)
        saved_t2, saved_energy = self.t2, self.e_corr
        self.t2 = t2
        try:
            return self.correlation_energy(eris)
        finally:
            self.t2, self.e_corr = saved_t2, saved_energy

    def compute_amplitudes_energy(self, eris, verbose=None, if_corr=True):
        r"""Compute reference amplitudes and energy through the deprecated
        combined API.

        Args:
            eris (_ChemistsERIs): Active MO integral container.
            verbose: Retained for the earlier signature; logging uses
            self.verbose.
            if_corr (bool): Enable the correlation-energy contribution for this
                call. Amplitudes are computed either way; the driver flag is
                restored.

        Returns:
            tuple: e_corr, t1 and t2, also stored on the driver. The energy is
            in
            Hartree per primitive cell. Emit a DeprecationWarning; new code
            uses
            ground_state_energy() and reads amplitude attributes separately.
        """
        warnings.warn(
            'Use ground_state_energy() and instance attributes.',
            DeprecationWarning,
            stacklevel=2)
        self.amplitudes(eris)
        old_if_corr = self.if_corr
        self.if_corr = if_corr
        try:
            self.correlation_energy(eris)
        finally:
            self.if_corr = old_if_corr
        return self.e_corr, self.t1, self.t2

    @property
    def chunk_size(self):
        r"""Get the cached number of orbitals reconstructed per integral chunk.

        Returns:
            int: Explicitly assigned positive size, or an automatic estimate
            based on active virtual count, element size and remaining memory.
            The automatic estimate is cached until orbital data are
            invalidated;
            assign None to request a new estimate. A manual size is not clamped
            here, although consumers clamp it to the relevant orbital
            dimension.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> print(mykradcbase.chunk_size)
            1
            >>> mykradcbase.chunk_size = 2
            >>> mykradcbase.chunk_size
            2
        """
        if self._chunk_size is None:
            self._chunk_size = kadc_ao2mo.chunk_size(self)
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        r"""Set a manual integral chunk size or restore automatic estimation.

        Args:
            value (int or None): Positive orbital count, or None to clear the
                cached estimate and mark subsequent estimates as automatic.

        Returns:
            None: Update the cached size and its manual/automatic flag. Manual
            choices survive orbital-cache invalidation.

        Raises:
            ValueError: value is nonintegral or less than one.
        """
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or value < 1):
            raise ValueError('chunk_size must be a positive integer or None.')
        self._chunk_size = value
        self._chunk_size_manual = value is not None

    @property
    def chnk_size(self):
        r"""Read the earlier chnk_size spelling of the chunk_size property.

        Returns:
            int: Same cached or manually assigned orbital chunk size as
            chunk_size. Reading either spelling can trigger automatic
            estimation.
            New code should use chunk_size.
        """
        return self.chunk_size

    @chnk_size.setter
    def chnk_size(self, value):
        r"""Assign a chunk size through the earlier property spelling.

        Args:
            value (int or None): Positive orbital chunk count or None for
                automatic estimation, validated by the chunk_size setter.

        Returns:
            None: Update chunk_size and its manual-selection flag. This setter
            does not maintain a second independent chunk-size value.
        """
        self.chunk_size = value

    @property
    def nkpts(self):
        r"""Return the number of k points in the current reference mesh.

        Returns:
            int: len(kpts). This property reflects a changed mesh immediately;
            cached orbital and momentum data still need invalidation after
            changes.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> mykradcbase.nkpts
            2
        """
        return len(self.kpts)

    @property
    def nocc(self):
        r"""Return the padded active occupied count used by tensor storage.

        Returns:
            int or numpy.integer: KMP2 occupied dimension after frozen-orbital
            exclusion, normally
            the maximum active occupied count across the mesh. Padding
            represents
            k points with fewer active occupied MOs. See get_nocc() for per-k
            counts.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> int(mykradcbase.nocc)
            1
        """
        self._ensure_reference()
        return self.get_nocc()

    @property
    def nmo(self):
        r"""Return the padded active MO dimension used by ADC tensor storage.

        Returns:
            int or numpy.integer: KMP2 active MO dimension, including padding
            needed for both
            occupied and virtual spaces across k points. Frozen orbitals are
            excluded. This dimension may be smaller than the AO basis
            dimension.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> int(mykradcbase.nmo)
            2
        """
        self._ensure_reference()
        return self.get_nmo()

    @property
    def nvir(self):
        r"""Return the padded active virtual dimension.

        Returns:
            int or numpy.integer: nmo - nocc, excluding frozen virtual orbitals
            and including any
            padding required to represent the full mesh with common tensor
            shapes.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> int(mykradcbase.nvir)
            1
        """
        return self.nmo - self.nocc

    @property
    def singles_shape(self):
        r"""Describe the sector singles block before C-order flattening.

        Returns:
            tuple[int]: The shape supplied by a concrete IP, EA or EE class.

        Raises:
            NotImplementedError: A reference-only KRADCBase has no sector.
        """
        raise NotImplementedError('Choose KRADCIP, KRADCEA or KRADCEE.')

    @property
    def n_singles(self):
        r"""Return the number of entries in the sector singles block.

        Returns:
            int: Product of singles_shape, including active-space padding.
        """
        return int(np.prod(self.singles_shape))

    @property
    def n_doubles(self):
        r"""Return the number of entries in the sector doubles block.

        Returns:
            int: Product of doubles_shape, including active-space padding.
        """
        return int(np.prod(self.doubles_shape))

    @property
    def doubles_shape(self):
        r"""Describe the sector doubles block before C-order flattening.

        Returns:
            tuple[int]: Shape supplied by a concrete IP, EA or EE class.

        Raises:
            NotImplementedError: A reference-only KRADCBase has no sector.
        """
        raise NotImplementedError('Choose KRADCIP, KRADCEA or KRADCEE.')

    @property
    def padded_coeff(self):
        r"""Get active MO coefficients in the common padded k-point layout.

        Returns:
            ndarray: Cached coefficients, shape (nkpts, nao, nmo), with
            occupied
            columns before virtual columns. Frozen MOs are excluded and padding
            columns are zero. The returned object is shared with integral
            contexts.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> mykradcbase.padded_coeff.shape
            (2, 2, 2)
        """
        return self._pad_orbitals()[0]

    @property
    def e_occ(self):
        r"""Get cached active occupied orbital energies in Hartree.

        Returns:
            ndarray: Shape (nkpts, nocc), using the same padding/orbital
            ordering
            as padded_coeff. Amplitude routines combine these energies with
            padding masks when forming orbital-energy denominators.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> mykradcbase.e_occ.shape
            (2, 1)
        """
        return self._pad_orbitals()[1]

    @property
    def e_vir(self):
        r"""Get cached active virtual orbital energies in Hartree.

        Returns:
            ndarray: Shape (nkpts, nvir), corresponding to padded_coeff's
            columns
            nocc onward. Padding masks distinguish physical virtual entries
            from
            placeholders when constructing amplitude denominators.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCBase
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = KRADCBase(kmf)
            >>> mykradcbase.e_vir.shape
            (2, 1)
        """
        return self._pad_orbitals()[2]

    make_ref_rdm1 = make_ref_rdm1

    get_nocc = get_nocc

    get_nmo = get_nmo

    _charged_init_guess = _init_guess

    _neutral_init_guess = _init_guess


class KRADCIP(KRADCBase):
    r"""Solve restricted periodic ionization ADC in one-hole/particle sectors.

    Args:
        mf (KRHF, Cell or KRADCBase): Reference, periodic cell, or another ADC
            object. kernel() prepares reference amplitudes before solving
            roots. ADC conversion preserves common data and clears sector
            results.
        frozen (int or sequence, optional): KMP2 frozen-space override.
        mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
            orbital overrides. See KRADCBase.__init__ for inheritance and reset
            rules.
        kpts (array_like, optional): Cartesian inverse-Bohr mesh for a Cell;
            shape (nkpts, 3). None uses Gamma. See KRADCBase for shared
            settings.

    Attributes:
        method (str): 'adc(2)', 'adc(2)-x' or 'adc(3)', defaulting to ADC(2)
            for new references.
        E, P (ndarray): Energies in Hartree and dimensionless spectroscopic
            factors, each shaped (nsectors, nroots), after kernel().
        U (ndarray): Root vectors, shape (nsectors, nroots, vector_size()).
        X (ndarray): Spectroscopic amplitudes, shape (nsectors, nmo, nroots).
        converged (ndarray): Boolean convergence flags matching E.

    Vectors concatenate a one-hole block of length nocc and a C-flattened
    two-hole/one-particle tensor of shape (nkpts, nkpts, nvir, nocc, nocc).
    Results follow kptlist order. Converted objects share an amplitude-storage
    owner; recomputing either object preserves the other object's reference
    arrays.

    The inherited run() method returns this object for chaining. Use run(0) to
    prepare only the reference, or run(nroots) for all sectors. Use kernel(...,
    kptlist=...) when selecting individual momentum sectors.

    Examples:
        Create the two-k-point helium reference below; lattice vectors
        are in Angstrom and energies are in Hartree:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
        >>> excitation_results = mykradcip.kernel(nroots=1, kptlist=[0])
        >>> energies = mykradcip.E
        >>> vectors = mykradcip.U
        >>> strengths = mykradcip.P
        >>> amplitudes = mykradcip.X
        >>> energies.round(6).tolist()
        [[0.371589]]
        >>> vectors.shape, amplitudes.shape
        ((1, 1, 5), (1, 2, 1))
    """

    # __init__ and kernel are inherited from KRADCBase. The sector
    # routines below follow their calls through the shared driver.

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        r"""Normalize charged eigenvectors in the spin-adapted ADC metric.

        The metric is :math:`U_1^\dagger U_1+2U_2^\dagger U_2
        -U_2^\dagger P U_2`, where P exchanges the two
        holes, including their orbital and k indices.

        Args:
            kshift (int): Momentum sector fixing the exchange partner.
            U (ndarray): Writable floating or complex eigenvectors of shape
                (vector_size(), nroots). Columns must have nonzero metric norm.
                This routine scales U in place.
            nroots (int): Number of roots; must match the column count.

        Returns:
            ndarray: Normalized columns reshaped to (vector_size(), nroots).
            The result shares the mutated input data.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> vectors = np.zeros((mykradcip.vector_size(), 1), dtype=complex)
            >>> vectors[0, 0] = 2
            >>> normalized = mykradcip.renormalize_eigenvectors(0, vectors, 1)
            >>> float(abs(normalized[0, 0]))
            1.0
        """
        nkpts = self.nkpts
        nocc = self.t2[0].shape[3]
        n_singles = nocc
        nvir = self.nvir

        for I in range(U.shape[1]):
            U1 = U[:n_singles, I]
            U2 = U[n_singles:, I].reshape(nkpts, nkpts, nvir, nocc, nocc)
            UdotU = np.dot(U1.conj().ravel(), U1.ravel())
            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = self.khelper.kconserv[kj, kshift, kk]
                    UdotU += 2.0 * np.dot(
                        U2[ka, kj].conj().ravel(), U2[ka, kj].ravel()
                    ) - np.dot(
                        U2[ka, kj].conj().ravel(),
                        U2[ka, kk].transpose(0, 2, 1).ravel())
            U[:, I] /= np.sqrt(UdotU)

        U = U.reshape(-1, nroots)

        return U

    def _get_adc0_ham(self):
        r"""Return the zeroth-order IP orbital-energy intermediate.

        The ADC effective Hamiltonian is conventionally denoted M; see
        doi:10.1063/1.5131771. M_ij is its zeroth-order singles intermediate;
        the physical IP singles block uses its negative.

        Returns:
            ndarray: Diagonal canonical occupied energies in Hartree, shape
            (nkpts, nocc, nocc). Frozen orbitals are excluded and padding
            follows e_occ. Higher-order corrections are supplied by the other
            _get_adc*_ham helpers.
        """
        M_ij = np.zeros(
            (self.nkpts, self.nocc, self.nocc),
            dtype=self.padded_coeff.dtype)
        # Set the diagonal to the canonical zeroth-order orbital energies.
        indices = np.arange(self.nocc)
        M_ij[:, indices, indices] = self.e_occ
        return M_ij

    def _get_adc2_ham(self, eris, M=None):
        r"""Add the second-order IP singles correction.

        This helper contributes to the 1h-1h singles block of the ADC effective
        Hamiltonian M; see doi:10.1063/1.5131771. The IP convention uses minus
        the accumulated M_ij intermediate.

        Args:
            eris (_ChemistsERIs): Open active-space integrals in chemist
                notation.
            M (ndarray, optional): Accumulation target, shape (nkpts, nocc,
                nocc). None allocates a zero correction.

        Returns:
            ndarray: The correction alone, or the updated input M. The physical
            singles block is minus this intermediate. Requires first-order t2.
            ADC(2)-x uses the same singles correction as ADC(2).
        """
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv
        eris_ovov = eris.ovov
        t2_1 = self.t2[0]
        M_ij = np.zeros_like(self._get_adc0_ham()) if M is None else M
        for ki in range(nkpts):
            kj = ki
            for kl in range(nkpts):
                for kd in range(nkpts):
                    ke = kconserv[kj, kd, kl]
                    # Reuse one local block instead of retaining four slices.
                    t2_block = t2_1[ki, kl, kd]

                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'ilde,jdle->ij',
                        t2_block, eris_ovov[kj, kd, kl])
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'ilde,jeld->ij',
                        t2_block, eris_ovov[kj, ke, kl])
                    M_ij[ki] += 0.5 * einsum(
                        'ilde,jdle->ij',
                        t2_block, eris_ovov[kj, kd, kl])

                    t2_block = t2_1[kl, ki, kd]
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'lide,jdle->ij',
                        t2_block, eris_ovov[kj, kd, kl])
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'lide,jeld->ij',
                        t2_block, eris_ovov[kj, ke, kl])

                    t2_block = t2_1[kj, kl, kd]
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'jlde,idle->ij',
                        t2_block.conj(), eris_ovov[ki, kd, kl].conj())
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'jlde,ield->ij',
                        t2_block.conj(), eris_ovov[ki, ke, kl].conj())
                    M_ij[ki] += 0.5 * einsum(
                        'jlde,idle->ij',
                        t2_block.conj(), eris_ovov[ki, kd, kl].conj())

                    t2_block = t2_1[kl, kj, kd]
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'ljde,idle->ij',
                        t2_block.conj(), eris_ovov[ki, kd, kl].conj())
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'ljde,ield->ij',
                        t2_block.conj(), eris_ovov[ki, ke, kl].conj())
        return M_ij

    def _get_adc3_ham(self, eris, M=None):
        r"""Add only the third-order IP singles correction.

        This helper contributes to the 1h-1h singles block of the ADC effective
        Hamiltonian M; see doi:10.1063/1.5131771. The IP convention uses minus
        the accumulated M_ij intermediate.

        Args:
            eris (_ChemistsERIs): Open integrals including the required
                occupied/virtual blocks or their density-fitting factors.
            M (ndarray, optional): Accumulation target, shape (nkpts, nocc,
                nocc). None allocates a zero correction.

        Returns:
            ndarray: The correction alone, or M updated in place. Requires
            ADC(3) reference amplitudes and intermediates. Lower orders are
            supplied separately by _get_adc0_ham() and _get_adc2_ham(). The
            physical singles block is minus this intermediate.
        """
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv
        eris_ovov = eris.ovov
        M_ij = np.zeros_like(self._get_adc0_ham()) if M is None else M
        t1_2 = self.t1[0]
        eris_ovoo = eris.ovoo
        eris_ovvo = eris.ovvo
        eris_oovv = eris.oovv
        eris_oooo = eris.oooo

        for ki in range(nkpts):
            kj = ki
            for kl in range(nkpts):
                kd = kconserv[kj, ki, kl]
                M_ij[ki] += 2. * einsum(
                    'ld,ldji->ij',
                    t1_2[kl], eris_ovoo[kl, kd, kj])
                M_ij[ki] -= einsum(
                    'ld,jdli->ij',
                    t1_2[kl], eris_ovoo[kj, kd, kl])

                kd = kconserv[ki, kj, kl]
                M_ij[ki] += 2. * einsum(
                    'ld,ldij->ij',
                    t1_2[kl].conj(), eris_ovoo[kl, kd, ki].conj())
                M_ij[ki] -= einsum(
                    'ld,idlj->ij',
                    t1_2[kl].conj(), eris_ovoo[ki, kd, kl].conj())

                for kd in range(nkpts):
                    t2_1 = self.t2[0]

                    ke = kconserv[kj, kd, kl]
                    t2_2_ild = self.t2[1][ki, kl, kd]
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'ilde,jdle->ij',
                        t2_2_ild, eris_ovov[kj, kd, kl])
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'ilde,jeld->ij',
                        t2_2_ild, eris_ovov[kj, ke, kl])
                    M_ij[ki] += 0.5 * einsum(
                        'ilde,jdle->ij',
                        t2_2_ild, eris_ovov[kj, kd, kl])
                    del t2_2_ild

                    t2_2_lid = self.t2[1][kl, ki, kd]
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'lide,jdle->ij',
                        t2_2_lid, eris_ovov[kj, kd, kl])
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'lide,jeld->ij',
                        t2_2_lid, eris_ovov[kj, ke, kl])

                    t2_2_jld = self.t2[1][kj, kl, kd]
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'jlde,leid->ij',
                        t2_2_jld.conj(), eris_ovov[kl, ke, ki].conj())
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'jlde,ield->ij',
                        t2_2_jld.conj(), eris_ovov[ki, ke, kl].conj())
                    M_ij[ki] += 0.5 * einsum(
                        'jlde,leid->ij',
                        t2_2_jld.conj(), eris_ovov[kl, ke, ki].conj())

                    t2_2_ljd = self.t2[1][kl, kj, kd]
                    M_ij[ki] -= 0.5 * 0.5 * einsum(
                        'ljde,leid->ij',
                        t2_2_ljd.conj(), eris_ovov[kl, ke, ki].conj())
                    M_ij[ki] += 0.5 * 0.5 * einsum(
                        'ljde,ield->ij',
                        t2_2_ljd.conj(), eris_ovov[ki, ke, kl].conj())

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                t2_1 = self.t2[0]

                kl = kconserv[kd, km, ke]
                kf = kconserv[kj, kd, kl]
                temp_t2_v_1 = einsum(
                    'lmde,jldf->mejf',
                    t2_1[kl, km, kd].conj(), t2_1[kj, kl, kd])
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'mejf,ifem->ij',
                    temp_t2_v_1, eris_ovvo[ki, kf, ke])
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'meif,jfem->ij',
                    temp_t2_v_1.conj(), eris_ovvo[kj, kf, ke].conj())
                M_ij[ki] += 0.5 * einsum(
                    'mejf,imef->ij',
                    temp_t2_v_1, eris_oovv[ki, km, ke])
                M_ij[ki] += 0.5 * einsum(
                    'meif,jmef->ij',
                    temp_t2_v_1.conj(), eris_oovv[kj, km, ke].conj())

                temp_t2_v_new = einsum(
                    'mlde,ljdf->mejf',
                    t2_1[km, kl, kd].conj(), t2_1[kl, kj, kd])
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'mejf,ifem->ij',
                    temp_t2_v_new, eris_ovvo[ki, kf, ke])
                M_ij[ki] += 0.5 * einsum(
                    'mejf,imef->ij',
                    temp_t2_v_new, eris_oovv[ki, km, ke])
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'meif,jfem->ij',
                    temp_t2_v_new.conj(), eris_ovvo[kj, kf, ke].conj())
                M_ij[ki] += 0.5 * einsum(
                    'meif,jmef->ij',
                    temp_t2_v_new.conj(), eris_oovv[kj, km, ke].conj())
                del temp_t2_v_new

                temp_t2_v_2 = einsum(
                    'lmde,ljdf->mejf',
                    t2_1[kl, km, kd].conj(), t2_1[kl, kj, kd])
                M_ij[ki] += 0.5 * 4 * einsum(
                    'mejf,ifem->ij',
                    temp_t2_v_2, eris_ovvo[ki, kf, ke])
                M_ij[ki] += 0.5 * 4 * einsum(
                    'meif,jfem->ij',
                    temp_t2_v_2.conj(), eris_ovvo[kj, kf, ke].conj())
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'meif,jmef->ij',
                    temp_t2_v_2.conj(), eris_oovv[kj, km, ke].conj())
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'mejf,imef->ij',
                    temp_t2_v_2, eris_oovv[ki, km, ke])
                del temp_t2_v_2

                temp_t2_v_3 = einsum(
                    'mlde,jldf->mejf',
                    t2_1[km, kl, kd].conj(), t2_1[kj, kl, kd])
                M_ij[ki] += 0.5 * einsum(
                    'mejf,ifem->ij',
                    temp_t2_v_3, eris_ovvo[ki, kf, ke])
                M_ij[ki] += 0.5 * einsum(
                    'meif,jfem->ij',
                    temp_t2_v_3.conj(), eris_ovvo[kj, kf, ke].conj())
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'meif,jmef->ij',
                    temp_t2_v_3.conj(), eris_oovv[kj, km, ke].conj())
                M_ij[ki] -= 0.5 * 2 * einsum(
                    'mejf,imef->ij',
                    temp_t2_v_3, eris_oovv[ki, km, ke])
                del temp_t2_v_3

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):

                kl = kconserv[kd, km, ke]
                kf = kconserv[kl, kd, km]
                temp_t2_v_8 = einsum(
                    'lmdf,lmde->fe',
                    t2_1[kl, km, kd], t2_1[kl, km, kd].conj())
                M_ij[ki] += 3.0 * einsum(
                    'fe,jief->ij',
                    temp_t2_v_8, eris_oovv[kj, ki, ke])
                M_ij[ki] -= 1.5 * einsum(
                    'fe,jfei->ij',
                    temp_t2_v_8, eris_ovvo[kj, kf, ke])
                M_ij[ki] += einsum(
                    'ef,jief->ij',
                    temp_t2_v_8.T, eris_oovv[kj, ki, ke])
                M_ij[ki] -= 0.5 * einsum(
                    'ef,jfei->ij',
                    temp_t2_v_8.T, eris_ovvo[kj, kf, ke])
                del temp_t2_v_8

                temp_t2_v_9 = einsum(
                    'lmdf,mlde->fe',
                    t2_1[kl, km, kd], t2_1[km, kl, kd].conj())
                M_ij[ki] -= 1.0 * einsum(
                    'fe,jief->ij',
                    temp_t2_v_9, eris_oovv[kj, ki, ke])
                M_ij[ki] -= 1.0 * einsum(
                    'ef,jief->ij',
                    temp_t2_v_9.T, eris_oovv[kj, ki, ke])
                M_ij[ki] += 0.5 * einsum(
                    'fe,jfei->ij',
                    temp_t2_v_9, eris_ovvo[kj, kf, ke])
                M_ij[ki] += 0.5 * einsum(
                    'ef,jfei->ij',
                    temp_t2_v_9.T, eris_ovvo[kj, kf, ke])
                del temp_t2_v_9

                kl = kconserv[kd, km, ke]
                kn = kconserv[kd, kl, ke]
                temp_t2_v_10 = einsum(
                    'lnde,lmde->nm',
                    t2_1[kl, kn, kd], t2_1[kl, km, kd].conj())
                M_ij[ki] -= 3.0 * einsum(
                    'nm,jinm->ij',
                    temp_t2_v_10, eris_oooo[kj, ki, kn])
                M_ij[ki] -= 1.0 * einsum(
                    'mn,jinm->ij',
                    temp_t2_v_10.T, eris_oooo[kj, ki, kn])
                M_ij[ki] += 1.5 * einsum(
                    'nm,jmni->ij',
                    temp_t2_v_10, eris_oooo[kj, km, kn])
                M_ij[ki] += 0.5 * einsum(
                    'mn,jmni->ij',
                    temp_t2_v_10.T, eris_oooo[kj, km, kn])
                del temp_t2_v_10

                temp_t2_v_11 = einsum(
                    'lnde,mlde->nm',
                    t2_1[kl, kn, kd], t2_1[km, kl, kd].conj())
                M_ij[ki] += 1.0 * einsum(
                    'nm,jinm->ij',
                    temp_t2_v_11, eris_oooo[kj, ki, kn])
                M_ij[ki] -= 0.5 * einsum(
                    'nm,jmni->ij',
                    temp_t2_v_11, eris_oooo[kj, km, kn])
                M_ij[ki] -= 0.5 * einsum(
                    'mn,jmni->ij',
                    temp_t2_v_11.T, eris_oooo[kj, km, kn])
                M_ij[ki] += 1.0 * einsum(
                    'mn,jinm->ij',
                    temp_t2_v_11.T, eris_oooo[kj, ki, kn])
                del temp_t2_v_11

            for km, ke, kd in kpts_helper.loop_kkk(nkpts):
                t2_1 = self.t2[0]
                kl = kconserv[kd, km, ke]
                kn = kconserv[kd, ki, ke]
                temp_t2_v_12 = einsum(
                    'inde,lmde->inlm',
                    t2_1[ki, kn, kd].conj(), t2_1[kl, km, kd])
                M_ij[ki] += 0.5 * 1.25 * einsum(
                    'inlm,jlnm->ij',
                    temp_t2_v_12, eris_oooo[kj, kl, kn].conj())
                M_ij[ki] -= 0.5 * 0.25 * einsum(
                    'inlm,jmnl->ij',
                    temp_t2_v_12, eris_oooo[kj, km, kn].conj())

                M_ij[ki] += 0.5 * 1.25 * einsum(
                    'jnlm,ilnm->ij',
                    temp_t2_v_12.conj(), eris_oooo[ki, kl, kn])
                M_ij[ki] -= 0.5 * 0.25 * einsum(
                    'jnlm,imnl->ij',
                    temp_t2_v_12.conj(), eris_oooo[ki, km, kn])
                del temp_t2_v_12

                temp_t2_v_12_1 = einsum(
                    'nide,mlde->inlm',
                    t2_1[kn, ki, kd].conj(), t2_1[km, kl, kd])
                M_ij[ki] += 0.5 * 0.25 * einsum(
                    'inlm,jlnm->ij',
                    temp_t2_v_12_1, eris_oooo[kj, kl, kn].conj())
                M_ij[ki] -= 0.5 * 0.25 * einsum(
                    'inlm,jmnl->ij',
                    temp_t2_v_12_1, eris_oooo[kj, km, kn].conj())
                M_ij[ki] += 0.5 * 0.25 * einsum(
                    'jnlm,ilnm->ij',
                    temp_t2_v_12_1.conj(), eris_oooo[ki, kl, kn])
                M_ij[ki] -= 0.5 * 0.25 * einsum(
                    'jnlm,imnl->ij',
                    temp_t2_v_12_1.conj(), eris_oooo[ki, km, kn])

                temp_t2_v_13 = einsum(
                    'inde,mlde->inml',
                    t2_1[ki, kn, kd].conj(), t2_1[km, kl, kd])
                M_ij[ki] -= 0.5 * 0.5 * einsum(
                    'inml,jlnm->ij',
                    temp_t2_v_13, eris_oooo[kj, kl, kn].conj())
                M_ij[ki] += 0.5 * 0.25 * einsum(
                    'inml,jmnl->ij',
                    temp_t2_v_13, eris_oooo[kj, km, kn].conj())

                M_ij[ki] -= 0.5 * 0.5 * einsum(
                    'jnml,ilnm->ij',
                    temp_t2_v_13.conj(), eris_oooo[ki, kl, kn])
                M_ij[ki] += 0.5 * 0.25 * einsum(
                    'jnml,imnl->ij',
                    temp_t2_v_13.conj(), eris_oooo[ki, km, kn])
                del temp_t2_v_13

                temp_t2_v_13_1 = einsum(
                    'nide,lmde->inml',
                    t2_1[kn, ki, kd].conj(), t2_1[kl, km, kd])
                M_ij[ki] += 0.5 * 0.25 * einsum(
                    'inml,jmnl->ij',
                    temp_t2_v_13_1, eris_oooo[kj, km, kn].conj())
                M_ij[ki] += 0.5 * 0.25 * einsum(
                    'jnml,imnl->ij',
                    temp_t2_v_13_1.conj(), eris_oooo[ki, km, kn])
                del temp_t2_v_13_1
        return M_ij

    def get_diag(self, kshift, M_ij=None, eris=None):
        r"""Construct the ionization Davidson diagonal.

        Singles use the diagonal of -M_ij[kshift]. Doubles use the zeroth-order
        difference :math:`\epsilon_a-\epsilon_i-\epsilon_j`. Higher-order
        doubles interactions remain in matvec(). Inactive padded components
        receive a large positive energy so they do not seed low-energy roots.

        Args:
            kshift (int): Charged momentum sector.
            M_ij (ndarray, optional): Singles intermediate of shape (nkpts,
                nocc, nocc). None builds it with get_imds(eris=eris).
            eris (_ChemistsERIs, optional): Open active-space integrals
                forwarded when the singles intermediate must be built.

        Returns:
            ndarray: Flat singles-then-doubles diagonal in Hartree, of length
            vector_size(). Its dtype combines the singles intermediate and
            orbital energies, retaining real or complex and single or double
            precision as supplied. The doubles view has shape doubles_shape; no
            second full doubles array is allocated in the unpadded case.

        Examples:
            A two-k-point model makes the orbital-energy differences explicit
            without running SCF. Each k point has one occupied and one virtual
            orbital. The supplied singles intermediate includes an illustrative
            correction to the canonical energy:

            >>> import numpy as np
            >>> from types import SimpleNamespace
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> kconserv = np.fromfunction(
            ...     lambda p, q, r: (p - q + r) % 2, (2, 2, 2), dtype=int)
            >>> model = SimpleNamespace(
            ...     method='adc(2)', verbose=0, stdout=None, nkpts=2,
            ...     n_singles=1, doubles_shape=(2, 2, 1, 1, 1),
            ...     vector_size=lambda: 5,
            ...     e_occ=np.array([[-0.8], [-0.5]]),
            ...     e_vir=np.array([[0.2], [0.4]]),
            ...     khelper=SimpleNamespace(kconserv=kconserv),
            ...     frozen=None, nmo=2,
            ...     mo_coeff=np.repeat(np.eye(2)[None], 2, axis=0))
            >>> singles_ham = np.array([-0.75, -0.45]).reshape(2, 1, 1)
            >>> diagonal = KRADCIP.get_diag(model, 0, M_ij=singles_ham)
            >>> diagonal.round(2).tolist()
            [0.75, 1.8, 1.2, 1.7, 1.7]
        """
        log = logger.Logger(self.stdout, self.verbose)
        self.method = check_implemented(self.method)
        if M_ij is None:
            M_ij = self.get_imds(eris=eris)

        e_occ, e_vir = self.e_occ, self.e_vir
        diagonal = np.empty(self.vector_size(),
                            dtype=np.result_type(M_ij, e_occ, e_vir))
        # These views follow the same C-order layout as the matrix action.
        singles = diagonal[:self.n_singles]
        doubles = diagonal[self.n_singles:].reshape(self.doubles_shape)
        singles[:] = -np.diagonal(M_ij[kshift])
        kconserv = self.khelper.kconserv
        momenta = np.arange(self.nkpts)
        # Stream one virtual-k slab; broadcast over both occupied orbitals
        # and all first-hole momenta instead of looping over each k pair.
        for ka in range(self.nkpts):
            kj = kconserv[kshift, momenta, ka]
            doubles[ka] = (
                e_occ[:, None, :, None] + e_occ[kj][:, None, None, :])
            doubles[ka] -= e_vir[ka][None, :, None, None]
        np.negative(doubles, out=doubles)

        if self.frozen is not None or any(
                coeff.shape[1] != self.nmo for coeff in self.mo_coeff):
            singles[:], doubles[:] = self.mask_frozen_ip(
                singles, doubles, kshift, const=LARGE_DENOM)
        log.timer_debug1('Completed ip_diag calculation')
        return diagonal

    def matvec(self, kshift, M_ij=None, eris=None):
        r"""Return a callable applying the ionization effective Hamiltonian.

        The callable evaluates :math:`\sigma=M(k)r` with the requested ADC
        order, retaining independent k axes instead of forming a dense matrix.
        Its doubles block has shape (nkpts, nkpts, nvir, nocc, nocc); the
        independent k axes are
        (k_a, k_i), the orbital axes are (a, i, j), and k_j = kconserv[kshift,
        k_i, k_a].

        Args:
            kshift (int): Charged momentum sector.
            M_ij (ndarray, optional): Singles intermediate from get_imds().
                With None, build it internally using a new transformation.
            eris (_ChemistsERIs, optional): Integral container required when
                applying the returned action. Use gen_matvec() to arrange an
                automatic transformation and consistent intermediates.

        Returns:
            callable: Function accepting a one-dimensional vector of length
            vector_size() and returning an independent complex sigma vector
            of the same length. Supplied integral scratch files must stay open.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> with mykradcip.ao2mo() as eris:
            ...     action = mykradcip.matvec(
            ...         0, mykradcip.get_imds(eris), eris)
            ...     sigma = action(np.ones(mykradcip.vector_size()))
            >>> sigma.shape
            (5,)
        """
        self.method = check_implemented(self.method)

        method = self.method

        nkpts = self.nkpts
        nocc = self.nocc
        kconserv = self.khelper.kconserv
        n_singles = nocc
        nvir = self.nvir
        n_doubles = nkpts * nkpts * nvir * nocc * nocc

        s_singles = 0
        f_singles = n_singles
        s_doubles = f_singles
        f_doubles = s_doubles + n_doubles

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)

        if M_ij is None:
            M_ij = self.get_imds(eris=eris)

        # Calculate sigma vector
        def sigma_(r):
            r"""Apply the captured ionization Hamiltonian to one trial vector.

            Args:
                r (ndarray): Flat vector of length vector_size(), ordered as
                    one-hole amplitudes followed by the C-flattened
                    two-hole/one-particle block. The input is not modified.

            Returns:
                ndarray: Complex sigma vector :math:`M(k)r`, including the
                captured
                singles intermediate and the method-dependent coupling/doubles
                contractions. The k sector and integral owner are closure
                state.
            """
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles]
            r2 = r[s_doubles:f_doubles]

            r2 = r2.reshape(nkpts, nkpts, nvir, nocc, nocc)
            s2 = np.zeros(
                (nkpts, nkpts, nvir, nocc, nocc), dtype=np.complex128)
            cell = self.cell
            kpts = self.kpts
            madelung = tools.madelung(cell, kpts)

            eris_ovoo = eris.ovoo

    ############ ADC(2) ij block ############################

            s1 = einsum('ij,j->i', M_ij[kshift], r1)

    ########### ADC(2) i - kja block #########################
            for kj in range(nkpts):
                for kk in range(nkpts):
                    ka = kconserv[kk, kshift, kj]
                    ki = kconserv[kj, kk, ka]

                    s1 += 2. * einsum(
                        'jaki,ajk->i',
                        eris_ovoo[kj, ka, kk].conj(), r2[ka, kj])
                    s1 -= einsum(
                        'kaji,ajk->i',
                        eris_ovoo[kk, ka, kj].conj(), r2[ka, kj])
    #################### ADC(2) ajk - i block ############################

                    s2[ka, kj] += einsum(
                        'jaki,i->ajk',
                        eris_ovoo[kj, ka, kk], r1)

    ################# ADC(2) ajk - bil block ############################

                    s2[ka, kj] -= einsum('a,ajk->ajk', e_vir[ka], r2[ka, kj])
                    s2[ka, kj] += einsum('j,ajk->ajk', e_occ[kj], r2[ka, kj])
                    s2[ka, kj] += einsum('k,ajk->ajk', e_occ[kk], r2[ka, kj])

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

                            s2[ka, kj] -= 0.5 * einsum(
                                'kijl,ali->ajk',
                                eris_oooo[kk, ki, kj], r2[ka, kl])
                            s2[ka, kj] -= 0.5 * einsum(
                                'klji,ail->ajk',
                                eris_oooo[kk, kl, kj], r2[ka, ki])

                        for kl in range(nkpts):
                            kb = kconserv[ka, kk, kl]
                            s2[ka, kj] += 0.5 * einsum(
                                'klba,bjl->ajk',
                                eris_oovv[kk, kl, kb], r2[kb, kj])

                            kb = kconserv[kl, kj, ka]
                            s2[ka, kj] += 0.5 * einsum(
                                'jabl,bkl->ajk',
                                eris_ovvo[kj, ka, kb], r2[kb, kk])
                            s2[ka, kj] -= einsum(
                                'jabl,blk->ajk',
                                eris_ovvo[kj, ka, kb], r2[kb, kl])
                            kb = kconserv[ka, kj, kl]
                            s2[ka, kj] += 0.5 * einsum(
                                'jlba,blk->ajk',
                                eris_oovv[kj, kl, kb], r2[kb, kl])

                        for ki in range(nkpts):
                            kb = kconserv[ka, kk, ki]
                            s2[ka, kj] += 0.5 * einsum(
                                'kiba,bji->ajk',
                                eris_oovv[kk, ki, kb], r2[kb, kj])

                            kb = kconserv[ka, kj, ki]
                            s2[ka, kj] += 0.5 * einsum(
                                'jiba,bik->ajk',
                                eris_oovv[kj, ki, kb], r2[kb, ki])
                            s2[ka, kj] -= einsum(
                                'jabi,bik->ajk',
                                eris_ovvo[kj, ka, kb], r2[kb, ki])
                            kb = kconserv[ki, kj, ka]
                            s2[ka, kj] += 0.5 * einsum(
                                'jabi,bki->ajk',
                                eris_ovvo[kj, ka, kb], r2[kb, kk])

                if self.exxdiv is not None:
                    s2 += madelung * r2

            if (method == "adc(3)"):

                eris_ovoo = eris.ovoo

    # ################ ADC(3) i - kja block and ajk - i
    # ############################

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kj, kshift, kk]

                        for kb in range(nkpts):
                            kc = kconserv[kj, kb, kk]
                            t2_1 = self.t2[0]
                            temp_1 = einsum(
                                'jkbc,ajk->abc',
                                t2_1[kj, kk, kb], r2[ka, kj])
                            temp = 0.25 * einsum(
                                'jkbc,ajk->abc',
                                t2_1[kj, kk, kb], r2[ka, kj])
                            temp -= 0.25 * einsum(
                                'jkbc,akj->abc',
                                t2_1[kj, kk, kb], r2[ka, kk])
                            temp -= 0.25 * einsum(
                                'kjbc,ajk->abc',
                                t2_1[kk, kj, kb], r2[ka, kj])
                            temp += 0.25 * einsum(
                                'kjbc,akj->abc',
                                t2_1[kk, kj, kb], r2[ka, kk])
                            ki = kconserv[kc, ka, kb]
                            if eris.ovvv is None:
                                chunk_size = self.chunk_size
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[ki, kc],
                                        eris.Lvv[ka, kb],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    k = eris_ovvv.shape[0]
                                    s1[a:a + k] += einsum(
                                        'abc,icab->i',
                                        temp_1, eris_ovvv)
                                    s1[a:a + k] += einsum(
                                        'abc,icab->i',
                                        temp, eris_ovvv)
                                    del eris_ovvv
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[ki, kb],
                                        eris.Lvv[ka, kc],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    s1[a:a + k] -= einsum(
                                        'abc,ibac->i',
                                        temp, eris_ovvv)
                                    del eris_ovvv
                                    a += k
                            else:
                                eris_ovvv = eris.ovvv[:]
                                s1 += einsum(
                                    'abc,icab->i',
                                    temp_1, eris_ovvv[ki, kc, ka])
                                s1 += einsum(
                                    'abc,icab->i',
                                    temp, eris_ovvv[ki, kc, ka])
                                s1 -= einsum(
                                    'abc,ibac->i',
                                    temp, eris_ovvv[ki, kb, ka])
                                del eris_ovvv
                del temp
                del temp_1

                t2_1 = self.t2[0]

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kj, kshift, kk]
                        for kc in range(nkpts):
                            kb = kconserv[kj, kc, kk]
                            ki = kconserv[kb, ka, kc]
                            if eris.ovvv is None:
                                chunk_size = self.chunk_size
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[ki, kc],
                                        eris.Lvv[ka, kb],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    k = eris_ovvv.shape[0]
                                    temp = (einsum(
                                        'i,icab->cba',
                                        r1[a:a + k], eris_ovvv.conj()))
                                    del eris_ovvv
                                    a += k
                            else:
                                eris_ovvv = eris.ovvv[:]
                                temp = einsum(
                                    'i,icab->cba',
                                    r1, eris_ovvv[ki, kc, ka].conj())
                                del eris_ovvv
                            s2[ka, kj] += einsum(
                                'cba,jkbc->ajk',
                                temp, t2_1[kj, kk, kb].conj())
                del temp

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kj, kshift, kk]
                        for kb in range(nkpts):
                            kl = kconserv[ka, kj, kb]

                            t2_1 = self.t2[0]
                            temp = einsum(
                                'ljba,ajk->blk',
                                t2_1[kl, kj, kb], r2[ka, kj])
                            temp_2 = einsum(
                                'jlba,akj->blk',
                                t2_1[kj, kl, kb], r2[ka, kk])
                            del t2_1

                            t2_1_jla = self.t2[0][kj, kl, ka]
                            temp += einsum(
                                'jlab,ajk->blk',
                                t2_1_jla, r2[ka, kj])
                            temp -= einsum(
                                'jlab,akj->blk',
                                t2_1_jla, r2[ka, kk])

                            temp_1 = einsum(
                                'jlab,ajk->blk',
                                t2_1_jla, r2[ka, kj])
                            temp_1 -= einsum(
                                'jlab,akj->blk',
                                t2_1_jla, r2[ka, kk])
                            temp_1 += einsum(
                                'jlab,ajk->blk',
                                t2_1_jla, r2[ka, kj])
                            del t2_1_jla

                            t2_1_lja = self.t2[0][kl, kj, ka]
                            temp -= einsum(
                                'ljab,ajk->blk',
                                t2_1_lja, r2[ka, kj])
                            temp += einsum(
                                'ljab,akj->blk',
                                t2_1_lja, r2[ka, kk])

                            temp_1 -= einsum(
                                'ljab,ajk->blk',
                                t2_1_lja, r2[ka, kj])
                            del t2_1_lja

                            ki = kconserv[kk, kl, kb]
                            s1 += 0.5 * einsum(
                                'blk,lbik->i',
                                temp, eris_ovoo[kl, kb, ki])
                            s1 -= 0.5 * einsum(
                                'blk,iblk->i',
                                temp, eris_ovoo[ki, kb, kl])
                            s1 += 0.5 * einsum(
                                'blk,lbik->i',
                                temp_1, eris_ovoo[kl, kb, ki])
                            s1 -= 0.5 * einsum(
                                'blk,iblk->i',
                                temp_2, eris_ovoo[ki, kb, kl])
                            del temp
                            del temp_1
                            del temp_2

                        for kb in range(nkpts):
                            kl = kconserv[ka, kk, kb]

                            t2_1 = self.t2[0]

                            temp = -einsum(
                                'lkba,akj->blj',
                                t2_1[kl, kk, kb], r2[ka, kk])
                            temp_2 = -einsum(
                                'klba,ajk->blj',
                                t2_1[kk, kl, kb], r2[ka, kj])
                            del t2_1

                            t2_1_kla = self.t2[0][kk, kl, ka]
                            temp -= einsum(
                                'klab,akj->blj',
                                t2_1_kla, r2[ka, kk])
                            temp += einsum(
                                'klab,ajk->blj',
                                t2_1_kla, r2[ka, kj])
                            temp_1 = -2.0 * einsum(
                                'klab,akj->blj',
                                t2_1_kla, r2[ka, kk])
                            temp_1 += einsum(
                                'klab,ajk->blj',
                                t2_1_kla, r2[ka, kj])
                            del t2_1_kla

                            t2_1_lka = self.t2[0][kl, kk, ka]
                            temp += einsum(
                                'lkab,akj->blj',
                                t2_1_lka, r2[ka, kk])
                            temp -= einsum(
                                'lkab,ajk->blj',
                                t2_1_lka, r2[ka, kj])
                            temp_1 += einsum(
                                'lkab,akj->blj',
                                t2_1_lka, r2[ka, kk])
                            del t2_1_lka

                            ki = kconserv[kj, kl, kb]
                            s1 -= 0.5 * einsum(
                                'blj,lbij->i',
                                temp, eris_ovoo[kl, kb, ki])
                            s1 += 0.5 * einsum(
                                'blj,iblj->i',
                                temp, eris_ovoo[ki, kb, kl])
                            s1 -= 0.5 * einsum(
                                'blj,lbij->i',
                                temp_1, eris_ovoo[kl, kb, ki])
                            s1 += 0.5 * einsum(
                                'blj,iblj->i',
                                temp_2, eris_ovoo[ki, kb, kl])

                            del temp
                            del temp_1
                            del temp_2

                for kj in range(nkpts):
                    for kk in range(nkpts):
                        ka = kconserv[kk, kshift, kj]
                        for kl in range(nkpts):
                            kb = kconserv[kj, ka, kl]
                            ki = kconserv[kk, kl, kb]
                            temp_1 = einsum(
                                'i,lbik->kbl',
                                r1, eris_ovoo[kl, kb, ki].conj())
                            temp = einsum(
                                'i,lbik->kbl',
                                r1, eris_ovoo[kl, kb, ki].conj())
                            temp -= einsum(
                                'i,iblk->kbl',
                                r1, eris_ovoo[ki, kb, kl].conj())

                            t2_1 = self.t2[0]
                            s2[ka, kj] += einsum(
                                'kbl,ljba->ajk',
                                temp, t2_1[kl, kj, kb].conj())
                            s2[ka, kj] += einsum(
                                'kbl,jlab->ajk',
                                temp_1, t2_1[kj, kl, ka].conj())
                            s2[ka, kj] -= einsum(
                                'kbl,ljab->ajk',
                                temp_1, t2_1[kl, kj, ka].conj())

                            kb = kconserv[kk, ka, kl]
                            ki = kconserv[kj, kl, kb]
                            temp_2 = -einsum(
                                'i,iblj->jbl',
                                r1, eris_ovoo[ki, kb, kl].conj())
                            s2[ka, kj] += einsum(
                                'jbl,klba->ajk',
                                temp_2, t2_1[kk, kl, kb].conj())
                            del t2_1
            s2 = s2.reshape(-1)
            s = np.hstack((s1, s2))
            del s1
            del s2
            cput0 = log.timer_debug1(
                'completed sigma vector calculation', *cput0)
            s *= -1.0

            return s
        return sigma_

    def mask_frozen_ip(self, v1, v2, kshift, const=-LARGE_DENOM):
        r"""Replace inactive padding in charged vector blocks by a constant.

        The KMP2 padding maps identify the physical active indices remaining
        after frozen-orbital selection. Both orbital indices and the conserved
        k partner are checked for the two-hole/one-particle block.

        Args:
            v1 (ndarray): Singles vector of shape (nocc,).
            v2 (ndarray): Doubles tensor of shape (nkpts, nkpts, nvir, nocc,
                nocc).
            kshift (int): Sector determining the missing fourth momentum.
            const (scalar): Value for inactive components. The default is
                -LARGE_DENOM before the IP Hamiltonian sign reversal;
                use zero to mask input guesses.

        Returns:
            tuple[ndarray, ndarray]: Independent masked singles and doubles,
            with the original shapes and dtypes. Neither input is modified.
        """
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self)

        new_v1 = const * np.ones_like(v1)
        new_v2 = const * np.ones_like(v2)

        new_v1[nonzero_opadding[kshift]] = v1[nonzero_opadding[kshift]]
        for ka in range(nkpts):
            for ki in range(nkpts):
                kj = kconserv[kshift, ki, ka]
                idx = np.ix_(
                    [ka],
                    [ki],
                    nonzero_vpadding[ka],
                    nonzero_opadding[ki],
                    nonzero_opadding[kj])
                new_v2[idx] = v2[idx]

        return (new_v1, new_v2)

    def get_trans_moments_orbital(self, orb, kshift):
        r"""Build the effective ionization transition vector for one MO.

        Ground-state amplitudes dress the leading one-hole transition
        and introduce two-hole/one-particle components. Optional higher-order
        terms follow method and approx_trans_moments.

        Args:
            orb (int): Padded active MO index in [0, nmo), with occupied MOs
                first and virtual MOs beginning at nocc.
            kshift (int): Momentum sector and orbital k point.

        Returns:
            ndarray: Complex vector of length vector_size(). Stacking these
            vectors over orb produces the rows of get_trans_moments().

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> mykradcip.get_trans_moments_orbital(0, 0).shape
            (5,)
        """
        self.method = check_implemented(self.method)

        method = self.method
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nvir
        t2_1 = self.t2[0]

        idn_occ = np.identity(nocc)

        T1 = np.zeros(nocc, dtype=np.complex128)
        T2 = np.zeros((nkpts, nkpts, nvir, nocc, nocc), dtype=np.complex128)

    ######## ADC(2) 1h part  ############################################

        if orb < nocc:
            T1 += idn_occ[orb, :]
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kd = self.khelper.kconserv[kk, kc, kshift]
                    ki = self.khelper.kconserv[kc, kk, kd]
                    T1 += 0.25 * einsum(
                        'kdc,ikdc->i',
                        t2_1[kk, kshift, kd][:, orb, :, :].conj(),
                        t2_1[ki, kk, kd])
                    T1 -= 0.25 * einsum(
                        'kcd,ikdc->i',
                        t2_1[kk, kshift, kc][:, orb, :, :].conj(),
                        t2_1[ki, kk, kd])
                    T1 -= 0.25 * einsum(
                        'kdc,ikcd->i',
                        t2_1[kk, kshift, kd][:, orb, :, :].conj(),
                        t2_1[ki, kk, kc])
                    T1 += 0.25 * einsum(
                        'kcd,ikcd->i',
                        t2_1[kk, kshift, kc][:, orb, :, :].conj(),
                        t2_1[ki, kk, kc])
                    T1 -= 0.25 * einsum(
                        'kdc,ikdc->i',
                        t2_1[kshift, kk, kd][orb, :, :, :].conj(),
                        t2_1[ki, kk, kd])
                    T1 -= 0.25 * einsum(
                        'kcd,ikcd->i',
                        t2_1[kshift, kk, kc][orb, :, :, :].conj(),
                        t2_1[ki, kk, kc])
        else:
            if (self.approx_trans_moments is False or self.method == "adc(3)"):
                t1_2 = self.t1[0]
                T1 += t1_2[kshift][:, (orb - nocc)]

    ######## ADC(2) 2h-1p  part  ############################################

            for ki in range(nkpts):
                for kj in range(nkpts):
                    ka = self.khelper.kconserv[kj, kshift, ki]

                    t2_1_t = -t2_1[ki, kj, ka].transpose(2, 3, 1, 0)
                    T2[ka, kj] += t2_1_t[:, orb - nocc, :, :].conj()

            del t2_1_t
    ####### ADC(3) 2h-1p  part  ############################################

        if (self.method == 'adc(2)-x' and self.approx_trans_moments is False
            or self.method == 'adc(3)'):
            t2_2 = self.t2[1]

            if orb >= nocc:
                for ki in range(nkpts):
                    for kj in range(nkpts):
                        ka = self.khelper.kconserv[kj, kshift, ki]

                        t2_2_t = -t2_2[ki, kj, ka].transpose(2, 3, 1, 0)
                        T2[ka, kj] += t2_2_t[:, orb - nocc, :, :].conj()

    ######### ADC(3) 1h part  ############################################

        if (method == 'adc(3)'):
            if orb < nocc:
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        kd = self.khelper.kconserv[kk, kc, kshift]
                        ki = self.khelper.kconserv[kd, kk, kc]
                        T1 += 0.25 * einsum(
                            'kdc,ikdc->i',
                            t2_1[kk, ki, kd][:, orb, :, :].conj(),
                            t2_2[ki, kk, kd])
                        T1 -= 0.25 * einsum(
                            'kcd,ikdc->i',
                            t2_1[kk, ki, kc][:, orb, :, :].conj(),
                            t2_2[ki, kk, kd])
                        T1 -= 0.25 * einsum(
                            'kdc,ikcd->i',
                            t2_1[kk, ki, kd][:, orb, :, :].conj(),
                            t2_2[ki, kk, kc])
                        T1 += 0.25 * einsum(
                            'kcd,ikcd->i',
                            t2_1[kk, ki, kc][:, orb, :, :].conj(),
                            t2_2[ki, kk, kc])
                        T1 -= 0.25 * einsum(
                            'kdc,ikdc->i',
                            t2_1[ki, kk, kd][orb, :, :, :].conj(),
                            t2_2[ki, kk, kd])
                        T1 -= 0.25 * einsum(
                            'kcd,ikcd->i',
                            t2_1[ki, kk, kc][orb, :, :, :].conj(),
                            t2_2[ki, kk, kc])

                        T1 += 0.25 * einsum(
                            'ikdc,kdc->i',
                            t2_1[ki, kk, kd],
                            t2_2[kk, ki, kd][:, orb, :, :].conj())
                        T1 -= 0.25 * einsum(
                            'ikcd,kdc->i',
                            t2_1[ki, kk, kc],
                            t2_2[kk, ki, kd][:, orb, :, :].conj())
                        T1 -= 0.25 * einsum(
                            'ikdc,kcd->i',
                            t2_1[ki, kk, kd],
                            t2_2[kk, ki, kc][:, orb, :, :].conj())
                        T1 += 0.25 * einsum(
                            'ikcd,kcd->i',
                            t2_1[ki, kk, kc],
                            t2_2[kk, ki, kc][:, orb, :, :].conj())
                        T1 -= 0.25 * einsum(
                            'ikcd,kcd->i',
                            t2_1[ki, kk, kc],
                            t2_2[ki, kk, kc][orb, :, :, :].conj())
                        T1 -= 0.25 * einsum(
                            'ikdc,kdc->i',
                            t2_1[ki, kk, kd],
                            t2_2[ki, kk, kd][orb, :, :, :].conj())
            else:

                for kk in range(nkpts):
                    for kc in range(nkpts):
                        ki = self.khelper.kconserv[kshift, kk, kc]

                        T1 += 0.5 * einsum(
                            'kic,kc->i',
                            t2_1[kk, ki, kc][:, :, :, orb - nocc], t1_2[kk])
                        T1 -= 0.5 * einsum(
                            'ikc,kc->i',
                            t2_1[ki, kk, kc][:, :, :, orb - nocc], t1_2[kk])
                        T1 += 0.5 * einsum(
                            'kic,kc->i',
                            t2_1[kk, ki, kc][:, :, :, orb - nocc], t1_2[kk])

            del t2_2
        del t2_1

        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = self.khelper.kconserv[kj, kshift, ki]
                T2[ka, kj] += T2[ka, kj] - T2[ka, ki].transpose(0, 2, 1)

        T2 = T2.reshape(-1)
        T = np.hstack((T1, T2))

        return T

    def make_rdm1_eigenvectors_slow(self, L, R, kshift):
        r"""Evaluate the retained explicit-loop charged density expression.

        This reference path exposes the orbital and momentum contractions used
        for the active-space density. Production root calculations use the
        blocked make_rdm1_eigenvectors() implementation instead.

        Args:
            L, R (array_like): Left and right state vectors of length
                vector_size(). L is conjugated internally; supply the original
                ket coefficients, not an already conjugated bra.
            kshift (int): Charged momentum sector.

        Returns:
            ndarray: Complex active MO density of shape (nkpts, nmo, nmo).
            Ground amplitudes must already be present; inputs are copied.
            This legacy path has no additional if_ss reference-density flag.
        """
        L = np.array(L).ravel().conj()
        R = np.array(R).ravel()

        t2_ov = self.t1[0]
        t1_oovv = self.t2[0]

        kconserv = self.khelper.kconserv

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts

        n_singles = nocc
        n_doubles = nkpts * nkpts * nvir * nocc * nocc

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        kd_oc = np.identity(nocc)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        R2 = R2.reshape(nkpts, nkpts, nvir, nocc, nocc)

    # block- ij
        # 000
        for ki in range(nkpts):
            rdm1[ki][:nocc, :nocc] += 2 * einsum('ij,m,m->ij', kd_oc, L1, R1)
        rdm1[kshift][:nocc, :nocc] -= einsum('i,j->ij', L1, R1)
        # 101
        for ki in range(nkpts):
            for ka in range(nkpts):
                kj = kconserv[ka, ki, kshift]
                for kl in range(nkpts):
                    rdm1[kl][:nocc, :nocc] += 4 * einsum(
                        'aij,aij,IJ->IJ',
                        L2[ka][ki], R2[ka][ki], kd_oc)
                    rdm1[kl][:nocc, :nocc] -= 2 * einsum(
                        'aij,aji,IJ->IJ',
                        L2[ka][ki], R2[ka][kj], kd_oc)

                rdm1[kj][:nocc, :nocc] -= 2 * einsum(
                    'aJi,aIi->IJ',
                    L2[ka][kj], R2[ka][kj])
                rdm1[kj][:nocc, :nocc] += einsum(
                    'aJi,aiI->IJ',
                    L2[ka][kj], R2[ka][ki])
                rdm1[kj][:nocc, :nocc] += einsum(
                    'aiJ,aIi->IJ',
                    L2[ka][ki], R2[ka][kj])
                rdm1[kj][:nocc, :nocc] -= 2 * einsum(
                    'aiJ,aiI->IJ',
                    L2[ka][ki], R2[ka][ki])
        # 020
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kj]
                ki = kshift
                rdm1[ki][:nocc, :nocc] += einsum(
                    'J,i,Ijab,ijab->IJ',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ki][:nocc, :nocc] -= 1 / 2 * einsum(
                    'J,i,Ijab,ijba->IJ',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())
                rdm1[ki][:nocc, :nocc] += einsum(
                    'i,I,ijab,Jjab->IJ',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ki][:nocc, :nocc] -= 1 / 2 * einsum(
                    'i,I,ijab,Jjba->IJ',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())

                rdm1[kj][:nocc, :nocc] += 2 * einsum(
                    'i,j,Iiab,Jjab->IJ',
                    L1, R1, t1_oovv[kj][ki][ka], t1_oovv[kj][ki][ka].conj())
                rdm1[kj][:nocc, :nocc] -= einsum(
                    'i,j,Iiab,Jjba->IJ',
                    L1, R1, t1_oovv[kj][ki][ka], t1_oovv[kj][ki][kb].conj())
                for ki in range(nkpts):
                    kb = kconserv[ki, ka, kj]
                    rdm1[ki][:nocc, :nocc] -= 4 * einsum(
                        'i,i,Ijab,Jjab->IJ',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj()
                    )
                    rdm1[ki][:nocc, :nocc] += 2 * einsum(
                        'i,i,Ijab,Jjba->IJ',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj()
                    )
    # block- ab
        # 101
        for ki in range(nkpts):
            for ka in range(nkpts):
                kj = kconserv[ka, ki, kshift]
                rdm1[ka][nocc:, nocc:] += 2 * einsum(
                    'Aij,Bij->AB',
                    L2[ka][ki], R2[ka][ki])
                rdm1[ka][nocc:, nocc:] -= einsum(
                    'Aij,Bji->AB',
                    L2[ka][ki], R2[ka][kj])

        # 020
        for kk in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, kk]
                ki = kshift
                rdm1[ka][nocc:, nocc:] -= 2 * einsum(
                    'i,j,ikBa,jkAa->AB',
                    L1, R1, t1_oovv[ki][kk][ka], t1_oovv[ki][kk][ka].conj())
                rdm1[ka][nocc:, nocc:] += einsum(
                    'i,j,ikBa,jkaA->AB',
                    L1, R1, t1_oovv[ki][kk][ka], t1_oovv[ki][kk][kb].conj())
                rdm1[ka][nocc:, nocc:] += einsum(
                    'i,j,ikaB,jkAa->AB',
                    L1, R1, t1_oovv[ki][kk][kb], t1_oovv[ki][kk][ka].conj())
                rdm1[ka][nocc:, nocc:] -= 2 * einsum(
                    'i,j,ikaB,jkaA->AB',
                    L1, R1, t1_oovv[ki][kk][kb], t1_oovv[ki][kk][kb].conj())

                for kj in range(nkpts):
                    kb = kconserv[kj, ka, kk]
                    rdm1[ka][nocc:, nocc:] += 4 * einsum(
                        'i,i,jkBa,jkAa->AB',
                        L1, R1, t1_oovv[kj][kk][ka], t1_oovv[kj][kk][ka].conj()
                    )
                    rdm1[ka][nocc:, nocc:] -= 2 * einsum(
                        'i,i,jkBa,jkaA->AB',
                        L1, R1, t1_oovv[kj][kk][ka], t1_oovv[kj][kk][kb].conj()
                    )

    # block- ia
        for ki in range(nkpts):
            for kj in range(nkpts):
                kb = kshift
                ka = kconserv[ki, kb, kj]
                rdm1[kb][:nocc, nocc:] += einsum(
                    'aij,I,ijAa->IA',
                    L2[ka][ki], R1, t1_oovv[ki][kj][kb])
                rdm1[kb][:nocc, nocc:] -= 2 * einsum(
                    'aij,I,ijaA->IA',
                    L2[ka][ki], R1, t1_oovv[ki][kj][ka])
        for ka in range(nkpts):
            ki = kshift
            kj = kconserv[ka, ki, kshift]
            for kl in range(nkpts):
                km = kl
                rdm1[kl][:nocc, nocc:] -= 2 * einsum(
                    'aij,i,IjAa->IA',
                    L2[ka][ki], R1, t1_oovv[kl][kj][km])
                rdm1[kl][:nocc, nocc:] += 1 * einsum(
                    'aij,i,IjaA->IA',
                    L2[ka][ki], R1, t1_oovv[kl][kj][ka])
                rdm1[kl][:nocc, nocc:] += 4 * einsum(
                    'aij,j,IiAa->IA',
                    L2[ka][kj], R1, t1_oovv[kl][kj][km])
                rdm1[kl][:nocc, nocc:] -= 2 * einsum(
                    'aij,j,IiaA->IA',
                    L2[ka][kj], R1, t1_oovv[kl][kj][ka])
        for ka in range(nkpts):
            ki = kshift
            rdm1[ka][:nocc, nocc:] += 2 * einsum('i,AIi->IA', L1, R2[ka][ka])
            rdm1[ka][:nocc, nocc:] -= einsum('i,AiI->IA', L1, R2[ka][ki])

    # block- ai
        for ki in range(nkpts):
            rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

        if self.approx_trans_moments is False or self.method == "adc(3)":
            ki = kshift
            ### block- ia ###
            rdm1[ki][:nocc, nocc:] -= einsum('i,I,iA->IA', L1, R1, t2_ov[ki])
            for ki in range(nkpts):
                rdm1[ki][:nocc, nocc:] += 2 * einsum(
                    'i,i,IA->IA',
                    L1, R1, t2_ov[ki])
            ### block- ai ###
                rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

        return rdm1

    def make_rdm1_eigenvectors(self, L, R, kshift, if_ss):
        r"""Contract left/right charged vectors into an active MO density.

        The blocked contractions retain three independent ground-amplitude
        k axes and the sector-conserving doubles layout. Occupied-virtual
        ground-state terms follow approx_trans_moments and the ADC order.

        Args:
            L, R (array_like): Metric-normalized state vectors of length
                vector_size(). Each is copied; L is conjugated internally.
                For an ordinary state density, pass the same ket vector twice.
            kshift (int): Charged momentum sector.
            if_ss (bool): Add the reference-density contribution used in
                state-specific FNO generation; False gives the charged density.

        Returns:
            ndarray: Complex active MO density of shape (nkpts, nmo, nmo).
            The routine neither restores frozen orbitals nor transforms to AOs.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcip = kadc_rhf.KRADCIP(kmf).run(0)
            >>> excitation_results = mykradcip.kernel(1, kptlist=[0])
            >>> ket = mykradcip.renormalize_eigenvectors(
            ...     0, mykradcip.U[0].T.copy(), 1)
            >>> density = mykradcip.make_rdm1_eigenvectors(
            ...     ket[:, 0], ket[:, 0], 0, False)
            >>> density.shape
            (2, 2, 2)
        """
        L = np.array(L).ravel().conj()
        R = np.array(R).ravel()

        t2_ov = self.t1[0]
        t1_oovv = self.t2[0]

        kconserv = self.khelper.kconserv
        kconserv_s = kconserv[kshift, :, :]

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts

        t1_oovv_sjb = np.zeros_like(t1_oovv[0])
        t1_oovv_isb = np.zeros_like(t1_oovv[0])
        t1_oovv_sja = t1_oovv[kshift]
        t1_oovv_isa = t1_oovv[:, kshift, :, :, :, :]
        t1_oovv_ijs = t1_oovv[:, :, kshift, :, :, :]

        kj, ka = np.indices((nkpts, nkpts))
        kb = self.khelper.kconserv[kshift, ka, kj]
        t1_oovv_sjb[kj, ka] = t1_oovv_sja[kj, kb]
        t1_oovv_isb[kj, ka] = t1_oovv_isa[kj, kb]

        n_singles = nocc
        n_doubles = nkpts * nkpts * nvir * nocc * nocc
        occ_list = range(nocc)
        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        R2 = R2.reshape(nkpts, nkpts, nvir, nocc, nocc)
        R2_aji = np.zeros_like(R2)
        L2_aji = np.zeros_like(L2)
        L2_jia = np.zeros_like(L2)

        ka, ki = np.indices((nkpts, nkpts))
        kj = kconserv_s[ki, ka]
        R2_aji[ka, ki] = R2[ka, kj]
        L2_aji[ka, ki] = L2[ka, kj]
        L2_asj = L2[:, kshift, :, :, :]

        kconserv_s = kconserv[:, kshift, :]
        kj, ki = np.indices((nkpts, nkpts))
        ka = kconserv_s[ki, kj]
        L2_jia[kj, ki] = L2[ka, ki]

    # block- ij
        # 000
        rdm1[:, occ_list, occ_list] += 2 * einsum('m,m->', L1, R1)
        rdm1[kshift, :nocc, :nocc] -= einsum('i,j->ij', L1, R1)
        # 101
        rdm1[:, occ_list, occ_list] += 4 * einsum('kKaij,kKaij->', L2, R2)
        rdm1[:, occ_list, occ_list] -= 2 * einsum('kKaij,kKaji->', L2, R2_aji)
        rdm1[:, :nocc, :nocc] -= 2 * einsum('kKaJi,kKaIi->KIJ', L2, R2)
        rdm1[:, :nocc, :nocc] += einsum('kKaJi,kKaiI->KIJ', L2, R2_aji)
        rdm1[:, :nocc, :nocc] += einsum('kKaiJ,kKaIi->KIJ', L2_aji, R2)
        rdm1[:, :nocc, :nocc] -= 2 * einsum('kKaiJ,kKaiI->KIJ', L2_aji, R2_aji)
        # 020
        rdm1[kshift, :nocc, :nocc] += einsum(
            'J,i,kKIjab,kKijab->IJ',
            L1, R1, t1_oovv_sja, t1_oovv_sja.conj())
        rdm1[kshift, :nocc, :nocc] -= 1 / 2 * einsum(
            'J,i,kKIjab,kKijba->IJ',
            L1, R1, t1_oovv_sja, t1_oovv_sjb.conj())
        rdm1[kshift, :nocc, :nocc] += einsum(
            'i,I,kKijab,kKJjab->IJ',
            L1, R1, t1_oovv_sja, t1_oovv_sja.conj())
        rdm1[kshift, :nocc, :nocc] -= 1 / 2 * einsum(
            'i,I,kKijab,kKJjba->IJ',
            L1, R1, t1_oovv_sja, t1_oovv_sjb.conj())
        rdm1[:, :nocc, :nocc] += 2 * einsum(
            'i,j,KkIiab,KkJjab->KIJ',
            L1, R1, t1_oovv_isa, t1_oovv_isa.conj())
        rdm1[:, :nocc, :nocc] -= einsum(
            'i,j,KkIiab,KkJjba->KIJ',
            L1, R1, t1_oovv_isa, t1_oovv_isb.conj())

    # block- ab
        # 101
        rdm1[:, nocc:, nocc:] += 2 * einsum('KkAij,KkBij->KAB', L2, R2)
        rdm1[:, nocc:, nocc:] -= einsum('KkAij,KkBji->KAB', L2, R2_aji)
        # 020
        rdm1[:, nocc:, nocc:] -= 2 * einsum(
            'i,m,kKijBa,kKmjAa->KAB',
            L1, R1, t1_oovv_sja, t1_oovv_sja.conj())
        rdm1[:, nocc:, nocc:] += einsum(
            'i,m,kKijBa,kKmjaA->KAB',
            L1, R1, t1_oovv_sja, t1_oovv_sjb.conj())
        rdm1[:, nocc:, nocc:] += einsum(
            'i,m,kKijaB,kKmjAa->KAB',
            L1, R1, t1_oovv_sjb, t1_oovv_sja.conj())
        rdm1[:, nocc:, nocc:] -= 2 * einsum(
            'i,m,kKijaB,kKmjaA->KAB',
            L1, R1, t1_oovv_sjb, t1_oovv_sjb.conj())

    # block- ia
        rdm1[kshift, :nocc, nocc:] += einsum(
            'kKaij,I,KkijAa->IA',
            L2_jia, R1, t1_oovv_ijs)
        rdm1[:, :nocc, nocc:] += 2 * einsum('i,KKAIi->KIA', L1, R2)
        rdm1[:, :nocc, nocc:] -= einsum(
            'i,KAiI->KIA',
            L1, R2[:, kshift, :, :, :])

        del (t1_oovv_sjb)
        del (t1_oovv_sja)
        del (t1_oovv_isa)
        del (t1_oovv_ijs)
        del (t1_oovv_isb)
        del (R2_aji)
        del (L2_aji)

        kj, ka = np.indices((nkpts, nkpts))
        idx0 = np.arange(nkpts)[:, None]
        LR = np.dot(L1, R1)
        if if_ss:
            rdm1[:, :nocc, :nocc] += np.identity(nocc)
            LR += 1.
        t1_oovv_ijs_b = np.zeros_like(t1_oovv[0])
        oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
        vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
        ov = np.zeros((nkpts, nocc, nvir), dtype=np.complex128)
        for ki in range(nkpts):
            kb = self.khelper.kconserv[ki, ka, kj]
            t1_oovv_np = np.array(t1_oovv[ki])
            t1_oovv_ijb = t1_oovv_np[idx0, kb]
            t1_oovv_ijs_b[ki] = t1_oovv_ijb[:, kshift].copy()
    # block- ij
            oo[ki] -= 4 * einsum(
                'KkIjab,KkJjab->IJ',
                t1_oovv_np, t1_oovv_np.conj())
            oo[ki] += 2 * einsum(
                'KkIjab,KkJjba->IJ',
                t1_oovv_np, t1_oovv_ijb.conj())
    # block- ab
            vv += 4 * einsum(
                'kKijBa,kKijAa->KAB',
                t1_oovv_np, t1_oovv_np.conj())
            vv -= 2 * einsum(
                'kKijBa,kKijaA->KAB',
                t1_oovv_np, t1_oovv_ijb.conj())
    # block- ia
            ov[ki] -= 2 * einsum(
                'kaij,i,kIjAa->IA',
                L2_asj, R1, t1_oovv_np[:, ki])
            ov[ki] += 1 * einsum('kaij,i,kkIjaA->IA', L2_asj, R1, t1_oovv_np)
            ov[ki] += 4 * einsum(
                'kkaij,j,kIiAa->IA',
                L2, R1, t1_oovv_np[:, ki])
            ov[ki] -= 2 * einsum('kkaij,j,kkIiaA->IA', L2, R1, t1_oovv_np)
            del (t1_oovv_ijb)
            del (t1_oovv_np)

        oo *= LR
        vv *= LR
        rdm1[:, :nocc, :nocc] += oo
        rdm1[:, nocc:, nocc:] += vv
        rdm1[:, :nocc, nocc:] += ov
        del (oo)
        del (vv)
        del (ov)
        del (L2_asj)
    # block- ia
        rdm1[kshift, :nocc, nocc:] -= 2 * einsum(
            'kKaij,I,KkijaA->IA',
            L2_jia, R1, t1_oovv_ijs_b)
        del (L2_jia)
        del (t1_oovv_ijs_b)

    # block- ai
        rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

        if self.approx_trans_moments is False or self.method == "adc(3)":
            rdm1[kshift, :nocc, nocc:] -= einsum(
                'i,I,iA->IA',
                L1, R1, t2_ov[kshift])
            rdm1[:, :nocc, nocc:] += 2 * LR * t2_ov
            rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(
                0, 2, 1)

        return rdm1

    @property
    def singles_shape(self):
        r"""Describe the IP singles block before C-order flattening.

        Returns:
            tuple[int]: (nocc,), including active-space padding. The selected
            momentum sector fixes the one-hole momentum.
        """
        return (self.nocc,)

    @property
    def doubles_shape(self):
        r"""Return the IP doubles shape in the matrix-action axis order.

        Returns:
            tuple[int]: nkpts, nkpts, nvir, nocc, nocc,
            including padding; the sector fixes the unstored third momentum.
        """
        return (self.nkpts, self.nkpts, self.nvir, self.nocc, self.nocc)

    _mask_frozen = mask_frozen_ip


class KRADCEA(KRADCBase):
    r"""Solve restricted periodic electron-attachment ADC in one-hole/particle
    sectors.

    Args:
        mf (KRHF, Cell or KRADCBase): Reference, periodic cell, or another ADC
            object. kernel() prepares reference amplitudes before solving
            roots. ADC conversion preserves common data and clears sector
            results.
        frozen (int or sequence, optional): KMP2 frozen-space override.
        mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
            orbital overrides. See KRADCBase.__init__ for inheritance and reset
            rules.
        kpts (array_like, optional): Cartesian inverse-Bohr mesh for a Cell;
            shape (nkpts, 3). None uses Gamma. See KRADCBase for shared
            settings.

    Attributes:
        method (str): 'adc(2)', 'adc(2)-x' or 'adc(3)', defaulting to ADC(2)
            for new references.
        E, P (ndarray): Energies in Hartree and dimensionless spectroscopic
            factors, each shaped (nsectors, nroots), after kernel().
        U (ndarray): Root vectors, shape (nsectors, nroots, vector_size()).
        X (ndarray): Spectroscopic amplitudes, shape (nsectors, nmo, nroots).
        converged (ndarray): Boolean convergence flags matching E.

    Vectors concatenate a one-particle block of length nvir and a C-flattened
    two-particle/one-hole tensor of shape (nkpts, nkpts, nocc, nvir, nvir).
    Results follow kptlist order. Converted objects share an amplitude-storage
    owner; recomputing either object preserves the other object's reference
    arrays.

    The inherited run() method returns this object for chaining. Use run(0) to
    prepare only the reference, or run(nroots) for all sectors. Use kernel(...,
    kptlist=...) when selecting individual momentum sectors.

    Examples:
        Create the two-k-point helium reference below; lattice vectors
        are in Angstrom and energies are in Hartree:

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcea = kadc_rhf.KRADCEA(kmf).run(0)
        >>> excitation_results = mykradcea.kernel(nroots=1, kptlist=[0])
        >>> energies = mykradcea.E
        >>> vectors = mykradcea.U
        >>> strengths = mykradcea.P
        >>> amplitudes = mykradcea.X
        >>> energies.round(6).tolist()
        [[1.558245]]
        >>> vectors.shape, amplitudes.shape
        ((1, 1, 5), (1, 2, 1))
    """

    # __init__ and kernel are inherited from KRADCBase. The sector
    # routines below follow their calls through the shared driver.

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        r"""Normalize charged eigenvectors in the spin-adapted ADC metric.

        The metric is :math:`U_1^\dagger U_1+2U_2^\dagger U_2
        -U_2^\dagger P U_2`, where P exchanges the two
        particles, including their orbital and k indices.

        Args:
            kshift (int): Momentum sector fixing the exchange partner.
            U (ndarray): Writable floating or complex eigenvectors of shape
                (vector_size(), nroots). Columns must have nonzero metric norm.
                This routine scales U in place.
            nroots (int): Number of roots; must match the column count.

        Returns:
            ndarray: Normalized columns reshaped to (vector_size(), nroots).
            The result shares the mutated input data.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcea = kadc_rhf.KRADCEA(kmf).run(0)
            >>> vectors = np.zeros((mykradcea.vector_size(), 1), dtype=complex)
            >>> vectors[0, 0] = 2
            >>> normalized = mykradcea.renormalize_eigenvectors(0, vectors, 1)
            >>> float(abs(normalized[0, 0]))
            1.0
        """
        nkpts = self.nkpts
        nocc = self.t2[0].shape[3]
        nvir = self.nvir
        n_singles = nvir

        for I in range(U.shape[1]):
            U1 = U[:n_singles, I]
            U2 = U[n_singles:, I].reshape(nkpts, nkpts, nocc, nvir, nvir)
            UdotU = np.dot(U1.conj().ravel(), U1.ravel())
            for ka in range(nkpts):
                for kb in range(nkpts):
                    ki = self.khelper.kconserv[kb, kshift, ka]
                    UdotU += 2.0 * np.dot(
                        U2[ki, ka].conj().ravel(), U2[ki, ka].ravel()
                    ) - np.dot(
                        U2[ki, ka].conj().ravel(),
                        U2[ki, kb].transpose(0, 2, 1).ravel())
            U[:, I] /= np.sqrt(UdotU)

        U = U.reshape(-1, nroots)

        return U

    def _get_adc0_ham(self):
        r"""Return the zeroth-order EA orbital-energy intermediate.

        The ADC effective Hamiltonian is conventionally denoted M; see
        doi:10.1063/1.5131771. M_ab is its zeroth-order singles intermediate;
        the physical EA singles block uses the returned intermediate.

        Returns:
            ndarray: Diagonal canonical virtual energies in Hartree, shape
            (nkpts, nvir, nvir). Frozen orbitals are excluded and padding
            follows e_vir. Higher-order corrections are supplied by the other
            _get_adc*_ham helpers.
        """
        M_ab = np.zeros(
            (self.nkpts, self.nvir, self.nvir),
            dtype=self.padded_coeff.dtype)
        indices = np.arange(self.nvir)
        M_ab[:, indices, indices] = self.e_vir
        return M_ab

    def _get_adc2_ham(self, eris, M=None):
        r"""Add the second-order EA singles correction.

        This helper contributes to the 1p-1p singles block of the ADC effective
        Hamiltonian M; see doi:10.1063/1.5131771. The EA convention uses the
        accumulated M_ab intermediate directly.

        Args:
            eris (_ChemistsERIs): Open active-space integrals in chemist
                notation.
            M (ndarray, optional): Accumulation target, shape (nkpts, nvir,
                nvir). None allocates a zero correction.

        Returns:
            ndarray: The correction alone, or the updated input M. The physical
            singles block is this intermediate. Requires first-order t2.
            ADC(2)-x uses the same singles correction as ADC(2).
        """
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv
        eris_ovov = eris.ovov
        M_ab = np.zeros_like(self._get_adc0_ham()) if M is None else M
        for ka in range(nkpts):
            kb = ka
            for kl in range(nkpts):
                for km in range(nkpts):

                    kd = kconserv[kl, ka, km]
                    t2_1_mla = self.t2[0][km, kl, ka]
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'mlad,lbmd->ab',
                        t2_1_mla, eris_ovov[kl, kb, km])
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'mlad,ldmb->ab',
                        t2_1_mla, eris_ovov[kl, kd, km])

                    t2_1_lma = self.t2[0][kl, km, ka]
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'lmad,lbmd->ab',
                        t2_1_lma, eris_ovov[kl, kb, km])
                    M_ab[ka] -= 0.5 * einsum(
                        'lmad,lbmd->ab',
                        t2_1_lma, eris_ovov[kl, kb, km])
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'lmad,ldmb->ab',
                        t2_1_lma, eris_ovov[kl, kd, km])
                    del t2_1_lma

                    t2_1_lmb = self.t2[0][kl, km, kb]
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'lmbd,lamd->ab',
                        t2_1_lmb.conj(), eris_ovov[kl, ka, km].conj())
                    M_ab[ka] -= 0.5 * einsum(
                        'lmbd,lamd->ab',
                        t2_1_lmb.conj(), eris_ovov[kl, ka, km].conj())
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'lmbd,ldma->ab',
                        t2_1_lmb.conj(), eris_ovov[kl, kd, km].conj())
                    del t2_1_lmb

                    t2_1_mlb = self.t2[0][km, kl, kb]
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'mlbd,lamd->ab',
                        t2_1_mlb.conj(), eris_ovov[kl, ka, km].conj())
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'mlbd,ldma->ab',
                        t2_1_mlb.conj(), eris_ovov[kl, kd, km].conj())
                    del t2_1_mlb
        return M_ab

    def _get_adc3_ham(self, eris, M=None):
        r"""Add only the third-order EA singles correction.

        This helper contributes to the 1p-1p singles block of the ADC effective
        Hamiltonian M; see doi:10.1063/1.5131771. The EA convention uses the
        accumulated M_ab intermediate directly.

        Args:
            eris (_ChemistsERIs): Open integrals including the required
                occupied/virtual blocks or their density-fitting factors.
            M (ndarray, optional): Accumulation target, shape (nkpts, nvir,
                nvir). None allocates a zero correction.

        Returns:
            ndarray: The correction alone, or M updated in place. Requires
            ADC(3) reference amplitudes and intermediates. Lower orders are
            supplied separately by _get_adc0_ham() and _get_adc2_ham(). The
            physical singles block is this intermediate.
        """
        nkpts, nocc, nvir = self.nkpts, self.nocc, self.nvir
        kconserv = self.khelper.kconserv
        log = logger.Logger(self.stdout, self.verbose)
        cput0 = logger.process_clock(), logger.perf_counter()
        eris_ovov = eris.ovov
        M_ab = np.zeros_like(self._get_adc0_ham()) if M is None else M
        eris_oovv = eris.oovv
        eris_ovvo = eris.ovvo

        t1_2 = self.t1[0]

        for ka in range(nkpts):
            kb = ka
            for kl in range(nkpts):

                if eris.ovvv is None:
                    chunk_size = self.chunk_size
                    chunk_size = min(chunk_size, nocc)
                    a = 0
                    for p in range(0, nocc, chunk_size):
                        kd = kconserv[ka, kb, kl]
                        eris_ovvv = (_ChemistsERIs.ovvv_df(
                            eris.Lov[kl, kd],
                            eris.Lvv[ka, kb],
                            p,
                            chunk_size).reshape(
                            -1, nvir, nvir, nvir) / nkpts)
                        k = eris_ovvv.shape[0]
                        M_ab[ka] += 2.0 * einsum(
                            'ld,ldab->ab',
                            t1_2[kl][a:a + k], eris_ovvv)
                        del eris_ovvv
                        kd = kconserv[kb, ka, kl]
                        eris_ovvv = (_ChemistsERIs.ovvv_df(
                            eris.Lov[kl, kd],
                            eris.Lvv[kb, ka],
                            p,
                            chunk_size).reshape(
                            -1, nvir, nvir, nvir) / nkpts)
                        k = eris_ovvv.shape[0]
                        M_ab[ka] += 2.0 * einsum(
                            'ld,ldba->ab',
                            t1_2[kl][a:a + k].conj(), eris_ovvv.conj())
                        del eris_ovvv
                        kd = kconserv[ka, kb, kl]
                        eris_ovvv = (_ChemistsERIs.ovvv_df(
                            eris.Lov[kl, kb],
                            eris.Lvv[ka, kd],
                            p,
                            chunk_size).reshape(
                            -1, nvir, nvir, nvir) / nkpts)
                        k = eris_ovvv.shape[0]
                        M_ab[ka] -= einsum(
                            'ld,lbad->ab',
                            t1_2[kl][a:a + k], eris_ovvv)
                        del eris_ovvv
                        kd = kconserv[kb, ka, kl]
                        eris_ovvv = (_ChemistsERIs.ovvv_df(
                            eris.Lov[kl, ka],
                            eris.Lvv[kb, kd],
                            p,
                            chunk_size).reshape(
                            -1, nvir, nvir, nvir) / nkpts)
                        k = eris_ovvv.shape[0]
                        M_ab[ka] -= einsum(
                            'ld,labd->ab',
                            t1_2[kl, a:a + k].conj(), eris_ovvv.conj())
                        del eris_ovvv
                        a += k

                else:
                    eris_ovvv = eris.ovvv
                    kd = kconserv[ka, kb, kl]
                    M_ab[ka] += 2. * einsum(
                        'ld,ldab->ab',
                        t1_2[kl], eris_ovvv[kl, kd, ka])
                    M_ab[ka] -= einsum(
                        'ld,lbad->ab',
                        t1_2[kl], eris_ovvv[kl, kb, ka])
                    kd = kconserv[kb, ka, kl]
                    M_ab[ka] += 2.0 * einsum(
                        'ld,ldba->ab',
                        t1_2[kl].conj(), eris_ovvv[kl, kd, kb].conj())
                    M_ab[ka] -= einsum(
                        'ld,labd->ab',
                        t1_2[kl].conj(), eris_ovvv[kl, ka, kb].conj())

        cput0 = log.timer_debug1(
            'Completed M_ab ovvv ADC(3) calculation', *cput0)

        for ka in range(nkpts):
            kb = ka
            log.timer_debug1('start:t2_ijbc*v_acij||O(No^2*Nv^3*Nk^3)')
            for kl in range(nkpts):
                for km in range(nkpts):
                    kd = kconserv[km, ka, kl]

                    t2_1 = self.t2[0]
                    t2_2_lma = self.t2[1][kl, km, ka]
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'lmad,lbmd->ab',
                        t2_2_lma, eris_ovov[kl, kb, km])
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'lmad,ldmb->ab',
                        t2_2_lma, eris_ovov[kl, kd, km])
                    M_ab[ka] -= 0.5 * einsum(
                        'lmad,lbmd->ab',
                        t2_2_lma, eris_ovov[kl, kb, km])

                    t2_2_mla = self.t2[1][km, kl, ka]
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'mlad,lbmd->ab',
                        t2_2_mla, eris_ovov[kl, kb, km])
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'mlad,ldmb->ab',
                        t2_2_mla, eris_ovov[kl, kd, km])

                    t2_2_lmb = self.t2[1][kl, km, kb]
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'lmbd,lamd->ab',
                        t2_2_lmb.conj(), eris_ovov[kl, ka, km].conj())
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'lmbd,ldma->ab',
                        t2_2_lmb.conj(), eris_ovov[kl, kd, km].conj())
                    M_ab[ka] -= 0.5 * einsum(
                        'lmbd,lamd->ab',
                        t2_2_lmb.conj(), eris_ovov[kl, ka, km].conj())

                    t2_2_mlb = self.t2[1][km, kl, kb]
                    M_ab[ka] += 0.5 * 0.5 * einsum(
                        'mlbd,mdla->ab',
                        t2_2_mlb.conj(), eris_ovov[km, kd, kl].conj())
                    M_ab[ka] -= 0.5 * 0.5 * einsum(
                        'mlbd,ldma->ab',
                        t2_2_mlb.conj(), eris_ovov[kl, kd, km].conj())
                    del t2_2_mlb

            log.timer_debug1('start:temp1_ovov*v_ovov||O(No^3*Nv^3*Nk^3)')
            for kn, ke, kd in kpts_helper.loop_kkk(nkpts):

                kl = kconserv[ke, kn, kd]
                km = kconserv[kb, kl, kd]
                temp_t2_v_1 = einsum(
                    'lned,mlbd->nemb',
                    t2_1[kl, kn, ke], t2_1[km, kl, kb].conj())
                M_ab[ka] -= 0.5 * einsum(
                    'nemb,nmae->ab',
                    temp_t2_v_1, eris_oovv[kn, km, ka])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'nemb,neam->ab',
                    temp_t2_v_1, eris_ovvo[kn, ke, ka])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'nema,nebm->ab',
                    temp_t2_v_1, eris_ovvo[kn, ke, kb])
                M_ab[ka] -= 0.5 * einsum(
                    'nema,nmbe->ab',
                    temp_t2_v_1, eris_oovv[kn, km, kb])
                del temp_t2_v_1

                temp_t2_v_1 = einsum(
                    'nled,lmbd->mbne',
                    t2_1[kn, kl, ke], t2_1[kl, km, kb].conj())
                M_ab[ka] -= 0.5 * einsum(
                    'mbne,nmae->ab',
                    temp_t2_v_1, eris_oovv[kn, km, ka])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'mbne,neam->ab',
                    temp_t2_v_1, eris_ovvo[kn, ke, ka])
                M_ab[ka] -= 0.5 * einsum(
                    'mane,nmbe->ab',
                    temp_t2_v_1, eris_oovv[kn, km, kb])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'mane,nebm->ab',
                    temp_t2_v_1, eris_ovvo[kn, ke, kb])
                del temp_t2_v_1

                temp_t2_v_2 = einsum(
                    'nled,mlbd->nemb',
                    t2_1[kn, kl, ke], t2_1[km, kl, kb].conj())
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'nemb,nmae->ab',
                    temp_t2_v_2, eris_oovv[kn, km, ka])
                M_ab[ka] -= 0.5 * 4.0 * einsum(
                    'nemb,neam->ab',
                    temp_t2_v_2, eris_ovvo[kn, ke, ka])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'nema,nmbe->ab',
                    temp_t2_v_2, eris_oovv[kn, km, kb])
                M_ab[ka] -= 0.5 * 4.0 * einsum(
                    'nema,nebm->ab',
                    temp_t2_v_2, eris_ovvo[kn, ke, kb])
                del temp_t2_v_2

                temp_t2_v_3 = einsum(
                    'lned,lmbd->nemb',
                    t2_1[kl, kn, ke], t2_1[kl, km, kb].conj())
                M_ab[ka] -= 0.5 * einsum(
                    'nemb,neam->ab',
                    temp_t2_v_3, eris_ovvo[kn, ke, ka])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'nemb,nmae->ab',
                    temp_t2_v_3, eris_oovv[kn, km, ka])
                M_ab[ka] += 0.5 * 2.0 * einsum(
                    'nema,nmbe->ab',
                    temp_t2_v_3, eris_oovv[kn, km, kb])
                M_ab[ka] -= 0.5 * einsum(
                    'nema,nebm->ab',
                    temp_t2_v_3, eris_ovvo[kn, ke, kb])
                del temp_t2_v_3

            log.timer_debug1('start:temp2_oo*v_ovov||O(No^3*Nv^2*Nk^3)')
            for kn, ke, kd in kpts_helper.loop_kkk(nkpts):
                kl = kconserv[ke, kn, kd]
                km = kconserv[ke, kl, kd]
                temp_t2_v_8 = einsum(
                    'lned,mled->mn',
                    t2_1[kl, kn, ke], t2_1[km, kl, ke].conj())
                M_ab[ka] += 2. * einsum(
                    'mn,nmab->ab',
                    temp_t2_v_8, eris_oovv[kn, km, ka])
                M_ab[ka] -= einsum(
                    'mn,nbam->ab',
                    temp_t2_v_8, eris_ovvo[kn, kb, ka])
                del temp_t2_v_8

                temp_t2_v_9 = einsum(
                    'nled,mled->mn',
                    t2_1[kn, kl, ke], t2_1[km, kl, ke].conj())
                M_ab[ka] -= 4. * einsum(
                    'mn,nmab->ab',
                    temp_t2_v_9, eris_oovv[kn, km, ka])
                M_ab[ka] += 2. * einsum(
                    'mn,nbam->ab',
                    temp_t2_v_9, eris_ovvo[kn, kb, ka])
                del temp_t2_v_9

            log.timer_debug1('start:Mab vvvv||O(No^2*Nv^3*Nk^3)')
            for km in range(nkpts):
                for kl in range(nkpts):
                    kf = kconserv[km, kl, ka]
                    temp_t2 = self.imds.t2_1_vvvv[:]
                    t2_1_mla = self.t2[0][km, kl, ka]
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'mlaf,mlbf->ab',
                        t2_1_mla, temp_t2[km, kl, kb].conj())
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'mlaf,lmbf->ab',
                        t2_1_mla, temp_t2[kl, km, kb].conj())
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'mlaf,mlfb->ab',
                        t2_1_mla, temp_t2[km, kl, kf].conj())
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'mlaf,lmfb->ab',
                        t2_1_mla, temp_t2[kl, km, kf].conj())
                    M_ab[ka] -= 0.5 * einsum(
                        'mlaf,mlbf->ab',
                        t2_1_mla, temp_t2[km, kl, kb].conj())
                    del t2_1_mla

                    t2_1_lma = self.t2[0][kl, km, ka]
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'lmaf,mlbf->ab',
                        t2_1_lma, temp_t2[km, kl, kb].conj())
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'lmaf,lmbf->ab',
                        t2_1_lma, temp_t2[kl, km, kb].conj())
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'lmaf,mlfb->ab',
                        t2_1_lma, temp_t2[km, kl, kf].conj())
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'lmaf,lmfb->ab',
                        t2_1_lma, temp_t2[kl, km, kf].conj())
                    del t2_1_lma

                    kd = kconserv[km, ka, kl]
                    t2_1_mlb = self.t2[0][km, kl, kb]
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'mlad,mlbd->ab',
                        temp_t2[km, kl, ka].conj(), t2_1_mlb)
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'lmad,mlbd->ab',
                        temp_t2[kl, km, ka].conj(), t2_1_mlb)
                    M_ab[ka] -= 0.5 * einsum(
                        'mlad,mlbd->ab',
                        temp_t2[km, kl, ka].conj(), t2_1_mlb)

                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'lmad,mlbd->ab',
                        temp_t2[kl, km, ka].conj(), t2_1_mlb)
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'mlad,mlbd->ab',
                        temp_t2[km, kl, ka].conj(), t2_1_mlb)
                    del t2_1_mlb

                    t2_1_lmb = self.t2[0][kl, km, kb]
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'mlad,lmbd->ab',
                        temp_t2[km, kl, ka].conj(), t2_1_lmb)
                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'lmad,lmbd->ab',
                        temp_t2[kl, km, ka].conj(), t2_1_lmb)

                    M_ab[ka] -= 0.5 * 0.25 * einsum(
                        'lmad,lmbd->ab',
                        temp_t2[kl, km, ka].conj(), t2_1_lmb)
                    M_ab[ka] += 0.5 * 0.25 * einsum(
                        'mlad,lmbd->ab',
                        temp_t2[km, kl, ka].conj(), t2_1_lmb)
                    del temp_t2
                    del t2_1_lmb

                    for kd in range(nkpts):
                        kf = kconserv[km, kd, kl]
                        ke = kconserv[kb, ka, kf]
                        temp_t2_vv_1 = einsum(
                            'mldf,mled->ef',
                            t2_1[km, kl, kd], t2_1[km, kl, ke].conj())
                        temp_t2_vv_2 = einsum(
                            'mldf,lmed->ef',
                            t2_1[km, kl, kd], t2_1[kl, km, ke].conj())
                        temp_t2_vv_3 = einsum(
                            'lmdf,mled->ef',
                            t2_1[kl, km, kd], t2_1[km, kl, ke].conj())
                        temp_t2_vv_4 = einsum(
                            'lmdf,lmed->ef',
                            t2_1[kl, km, kd], t2_1[kl, km, ke].conj())
                        temp_t2_vv_5 = einsum(
                            'mlfd,mled->ef',
                            t2_1[km, kl, kf], t2_1[km, kl, ke].conj())
                        if eris.vvvv is None:
                            chunk_size = (self.chunk_size)
                            chunk_size = min(chunk_size, nvir)
                            a = 0
                            for p in range(0, nvir, chunk_size):
                                eris_vvvv = _ChemistsERIs.vvvv_df(
                                    eris.Lvv[kb, ka],
                                    eris.Lvv[ke, kf],
                                    p,
                                    chunk_size) / nkpts
                                k = eris_vvvv.shape[0]
                                M_ab[ka, a:a + k] -= einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_1, eris_vvvv)
                                M_ab[ka, a:a + k] += einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_2, eris_vvvv)
                                M_ab[ka, a:a + k] += einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_3, eris_vvvv)
                                M_ab[ka, a:a + k] -= einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_4, eris_vvvv)
                                M_ab[ka, a:a + k] += 2. * einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_5, eris_vvvv)
                                del eris_vvvv

                                eris_vvvv = _ChemistsERIs.vvvv_df(
                                    eris.Lvv[kb, kf],
                                    eris.Lvv[ke, ka],
                                    p,
                                    chunk_size) / nkpts
                                M_ab[ka, a:a + k] += 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_1, eris_vvvv)
                                M_ab[ka, a:a + k] -= 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_2, eris_vvvv)
                                M_ab[ka, a:a + k] -= 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_3, eris_vvvv)
                                M_ab[ka, a:a + k] += 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_4, eris_vvvv)
                                M_ab[ka, a:a + k] -= einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_5, eris_vvvv)
                                del eris_vvvv
                                a += k

                        elif isinstance(eris.vvvv, np.ndarray):
                            eris_vvvv = eris.vvvv
                            M_ab[ka] -= einsum(
                                'ef,aebf->ab',
                                temp_t2_vv_1, eris_vvvv[ka, ke, kb])
                            M_ab[ka] += einsum(
                                'ef,aebf->ab',
                                temp_t2_vv_2, eris_vvvv[ka, ke, kb])
                            M_ab[ka] += einsum(
                                'ef,aebf->ab',
                                temp_t2_vv_3, eris_vvvv[ka, ke, kb])
                            M_ab[ka] -= einsum(
                                'ef,aebf->ab',
                                temp_t2_vv_4, eris_vvvv[ka, ke, kb])
                            M_ab[ka] += 2.0 * einsum(
                                'ef,aebf->ab',
                                temp_t2_vv_5, eris_vvvv[ka, ke, kb])
                            M_ab[ka] += 0.5 * einsum(
                                'ef,aefb->ab',
                                temp_t2_vv_1, eris_vvvv[ka, ke, kf])
                            M_ab[ka] -= 0.5 * einsum(
                                'ef,aefb->ab',
                                temp_t2_vv_2, eris_vvvv[ka, ke, kf])
                            M_ab[ka] -= 0.5 * einsum(
                                'ef,aefb->ab',
                                temp_t2_vv_3, eris_vvvv[ka, ke, kf])
                            M_ab[ka] += 0.5 * einsum(
                                'ef,aefb->ab',
                                temp_t2_vv_4, eris_vvvv[ka, ke, kf])
                            M_ab[ka] -= einsum(
                                'ef,aefb->ab',
                                temp_t2_vv_5, eris_vvvv[ka, ke, kf])
                        else:
                            chunk_size = (self.chunk_size)
                            chunk_size = min(chunk_size, nvir)
                            a = 0
                            for p in range(0, nvir, chunk_size):
                                eris_vvvv = eris.vvvv[kb, ke, ka,
                                    p:p + chunk_size]
                                k = eris_vvvv.shape[0]
                                M_ab[ka, a:a + k] -= einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_1, eris_vvvv)
                                M_ab[ka, a:a + k] += einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_2, eris_vvvv)
                                M_ab[ka, a:a + k] += einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_3, eris_vvvv)
                                M_ab[ka, a:a + k] -= einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_4, eris_vvvv)
                                M_ab[ka, a:a + k] += 2. * einsum(
                                    'ef,aebf->ab',
                                    temp_t2_vv_5, eris_vvvv)
                                del eris_vvvv
                                eris_vvvv = eris.vvvv[ka, ke, kf,
                                    p:p + chunk_size]
                                M_ab[ka, a:a + k] += 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_1, eris_vvvv)
                                M_ab[ka, a:a + k] -= 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_2, eris_vvvv)
                                M_ab[ka, a:a + k] -= 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_3, eris_vvvv)
                                M_ab[ka, a:a + k] += 0.5 * einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_4, eris_vvvv)
                                M_ab[ka, a:a + k] -= einsum(
                                    'ef,aefb->ab',
                                    temp_t2_vv_5, eris_vvvv)
                                del eris_vvvv
                                a += k
                        del temp_t2_vv_1
                        del temp_t2_vv_2
                        del temp_t2_vv_3
                        del temp_t2_vv_4
                        del temp_t2_vv_5
        return M_ab

    def get_diag(self, kshift, M_ab=None, eris=None):
        r"""Construct the electron-attachment Davidson diagonal.

        Singles use the diagonal of M_ab[kshift]. Doubles use the zeroth-order
        difference :math:`\epsilon_a+\epsilon_b-\epsilon_i`. Higher-order
        doubles interactions remain in matvec(). Inactive padded components
        receive a large positive energy so they do not seed low-energy roots.

        Args:
            kshift (int): Charged momentum sector.
            M_ab (ndarray, optional): Singles intermediate of shape (nkpts,
                nvir, nvir). None builds it with get_imds(eris=eris).
            eris (_ChemistsERIs, optional): Open active-space integrals
                forwarded when the singles intermediate must be built.

        Returns:
            ndarray: Flat singles-then-doubles diagonal in Hartree, of length
            vector_size(). Its dtype combines the singles intermediate and
            orbital energies, retaining real or complex and single or double
            precision as supplied. The doubles view has shape doubles_shape; no
            second full doubles array is allocated in the unpadded case.

        Examples:
            A two-k-point model makes the orbital-energy differences explicit
            without running SCF. Each k point has one occupied and one virtual
            orbital. The supplied singles intermediate includes an illustrative
            correction to the canonical energy:

            >>> import numpy as np
            >>> from types import SimpleNamespace
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
            >>> kconserv = np.fromfunction(
            ...     lambda p, q, r: (p - q + r) % 2, (2, 2, 2), dtype=int)
            >>> model = SimpleNamespace(
            ...     method='adc(2)', verbose=0, stdout=None, nkpts=2,
            ...     n_singles=1, doubles_shape=(2, 2, 1, 1, 1),
            ...     vector_size=lambda: 5,
            ...     e_occ=np.array([[-0.8], [-0.5]]),
            ...     e_vir=np.array([[0.2], [0.4]]),
            ...     khelper=SimpleNamespace(kconserv=kconserv),
            ...     frozen=None, nmo=2,
            ...     mo_coeff=np.repeat(np.eye(2)[None], 2, axis=0))
            >>> singles_ham = np.array([0.3, 0.5]).reshape(2, 1, 1)
            >>> diagonal = KRADCEA.get_diag(model, 0, M_ab=singles_ham)
            >>> diagonal.round(2).tolist()
            [0.3, 1.2, 1.6, 1.1, 1.1]
        """
        log = logger.Logger(self.stdout, self.verbose)
        self.method = check_implemented(self.method)
        if M_ab is None:
            M_ab = self.get_imds(eris=eris)

        e_occ, e_vir = self.e_occ, self.e_vir
        diagonal = np.empty(self.vector_size(),
                            dtype=np.result_type(M_ab, e_occ, e_vir))
        # These views follow the same C-order layout as the matrix action.
        singles = diagonal[:self.n_singles]
        doubles = diagonal[self.n_singles:].reshape(self.doubles_shape)
        singles[:] = np.diagonal(M_ab[kshift])
        kconserv = self.khelper.kconserv
        momenta = np.arange(self.nkpts)
        # Stream one occupied-k slab and broadcast both virtual orbitals;
        # the second particle momentum is fixed by conservation.
        for ki in range(self.nkpts):
            kb = kconserv[kshift, momenta, ki]
            doubles[ki] = (
                e_vir[:, None, :, None] + e_vir[kb][:, None, None, :])
            doubles[ki] -= e_occ[ki][None, :, None, None]

        if self.frozen is not None or any(
                coeff.shape[1] != self.nmo for coeff in self.mo_coeff):
            singles[:], doubles[:] = self.mask_frozen_ea(
                singles, doubles, kshift, const=LARGE_DENOM)
        log.timer_debug1('Completed ea_diag calculation')
        return diagonal

    def matvec(self, kshift, M_ab=None, eris=None):
        r"""Return a callable applying the electron-attachment effective
        Hamiltonian.

        The callable evaluates :math:`\sigma=M(k)r` with the requested ADC
        order, retaining independent k axes instead of forming a dense matrix.
        Its doubles block has shape (nkpts, nkpts, nocc, nvir, nvir); the
        independent k axes are
        (k_i, k_a), the orbital axes are (i, a, b), and k_b = kconserv[kshift,
        k_a, k_i].

        Args:
            kshift (int): Charged momentum sector.
            M_ab (ndarray, optional): Singles intermediate from get_imds().
                With None, build it internally using a new transformation.
            eris (_ChemistsERIs, optional): Integral container required when
                applying the returned action. Use gen_matvec() to arrange an
                automatic transformation and consistent intermediates.

        Returns:
            callable: Function accepting a one-dimensional vector of length
            vector_size() and returning an independent complex sigma vector
            of the same length. Supplied integral scratch files must stay open.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcea = kadc_rhf.KRADCEA(kmf).run(0)
            >>> with mykradcea.ao2mo() as eris:
            ...     action = mykradcea.matvec(
            ...         0, mykradcea.get_imds(eris), eris)
            ...     sigma = action(np.ones(mykradcea.vector_size()))
            >>> sigma.shape
            (5,)
        """
        self.method = check_implemented(self.method)

        method = self.method

        nkpts = self.nkpts
        nocc = self.nocc
        kconserv = self.khelper.kconserv
        nvir = self.nvir
        n_singles = self.nmo - self.nocc
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        s_singles = 0
        f_singles = n_singles
        s_doubles = f_singles
        f_doubles = s_doubles + n_doubles

        mo_coeff, e_occ, e_vir = (self.padded_coeff,

            self.e_occ, self.e_vir)

        if M_ab is None:
            M_ab = self.get_imds(eris=eris)

        # Calculate sigma vector
        def sigma_(r):
            r"""Apply the captured electron-attachment Hamiltonian to one trial
            vector.

            Args:
                r (ndarray): Flat vector of length vector_size(), ordered as
                    one-particle amplitudes followed by the C-flattened
                    two-particle/one-hole block. The input is not modified.

            Returns:
                ndarray: Complex sigma vector :math:`M(k)r`, including the
                captured
                singles intermediate and the method-dependent coupling/doubles
                contractions. The k sector and integral owner are closure
                state.
            """
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles]
            r2 = r[s_doubles:f_doubles]

            r2 = r2.reshape(nkpts, nkpts, nocc, nvir, nvir)
            s2 = np.zeros(
                (nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)
            cell = self.cell
            kpts = self.kpts
            madelung = tools.madelung(cell, kpts)

    ############ ADC(2) ij block ############################

            s1 = einsum('ab,b->a', M_ab[kshift], r1)

    ########### ADC(2) coupling blocks #########################

            for kb in range(nkpts):
                for kc in range(nkpts):
                    ki = kconserv[kb, kshift, kc]
                    if eris.ovvv is None:
                        chunk_size = self.chunk_size
                        chunk_size = min(chunk_size, nocc)
                        a = 0
                        for p in range(0, nocc, chunk_size):
                            eris_ovvv = (_ChemistsERIs.ovvv_df(
                                eris.Lov[ki, kc],
                                eris.Lvv[kshift, kb],
                                p,
                                chunk_size).reshape(-1, nvir, nvir, nvir) /
                                nkpts)
                            k = eris_ovvv.shape[0]
                            s1 += 2. * einsum(
                                'icab,ibc->a',
                                eris_ovvv.conj(), r2[ki, kb, a:a + k])
                            s2[ki, kb, a:a + k] += einsum(
                                'icab,a->ibc',
                                eris_ovvv, r1)
                            del eris_ovvv

                            eris_ovvv = (_ChemistsERIs.ovvv_df(
                                eris.Lov[ki, kb],
                                eris.Lvv[kshift, kc],
                                p,
                                chunk_size).reshape(-1, nvir, nvir, nvir) /
                                nkpts)
                            s1 -= einsum(
                                'ibac,ibc->a',
                                eris_ovvv.conj(), r2[ki, kb, a:a + k])
                            del eris_ovvv
                            a += k
                    else:
                        direct = eris.ovvv[ki, kc, kshift]
                        exchange = eris.ovvv[ki, kb, kshift]
                        s1 += 2 * einsum(
                            'icab,ibc->a',
                            direct.conj(), r2[ki, kb])
                        s1 -= einsum(
                            'ibac,ibc->a',
                            exchange.conj(), r2[ki, kb])
                        s2[ki, kb] += einsum('icab,a->ibc', direct, r1)

    #########    ########## ADC(2) ajk - bil block ############################

                    s2[ki, kb] -= einsum('i,ibc->ibc', e_occ[ki], r2[ki, kb])
                    s2[ki, kb] += einsum('b,ibc->ibc', e_vir[kb], r2[ki, kb])
                    s2[ki, kb] += einsum('c,ibc->ibc', e_vir[kc], r2[ki, kb])

    ################ ADC(3) ajk - bil block ############################

            if (method == "adc(2)-x" or method == "adc(3)"):
                eris_oovv = eris.oovv
                eris_ovvo = eris.ovvo

                for kx in range(nkpts):
                    for ky in range(nkpts):
                        ki = kconserv[ky, kshift, kx]
                        for kz in range(nkpts):
                            kw = kconserv[kx, kz, ky]

                            if isinstance(eris.vvvv, np.ndarray):
                                eris_vvvv = eris.vvvv.reshape(
                                    nkpts,
                                    nkpts,
                                    nkpts,
                                    nvir * nvir,
                                    nvir * nvir)
                                r2_1 = r2.reshape(
                                    nkpts, nkpts, nocc, nvir * nvir)
                                s2[ki, kx] += np.dot(
                                    r2_1[ki, kw],
                                    eris_vvvv[kx, ky, kw].T.conj(),
                                ).reshape(nocc, nvir, nvir)
                            elif eris.vvvv is None:
                                s2[ki, kx] += self.ea_contract_r_vvvv(
                                    r2[ki, kw], eris.Lvv, kx, ky, kw)
                            else:
                                s2[ki, kx] += self.ea_contract_r_vvvv(
                                    r2[ki, kw], eris.vvvv, kx, ky, kw)
                        for kj in range(nkpts):
                            kz = kconserv[ky, ki, kj]
                            s2[ki, kx] -= 0.5 * einsum(
                                'iyzj,jzx->ixy',
                                eris_ovvo[ki, ky, kz], r2[kj, kz])
                            s2[ki, kx] += einsum(
                                'iyzj,jxz->ixy',
                                eris_ovvo[ki, ky, kz], r2[kj, kx])
                            s2[ki, kx] -= 0.5 * einsum(
                                'ijzy,jxz->ixy',
                                eris_oovv[ki, kj, kz], r2[kj, kx])

                            kz = kconserv[kj, ki, kx]
                            s2[ki, kx] -= 0.5 * einsum(
                                'ijzx,jzy->ixy',
                                eris_oovv[ki, kj, kz], r2[kj, kz])

                            kw = kconserv[kj, ki, kx]
                            s2[ki, kx] -= 0.5 * einsum(
                                'ijwx,jwy->ixy',
                                eris_oovv[ki, kj, kw], r2[kj, kw])

                            kw = kconserv[kj, ki, ky]
                            s2[ki, kx] -= 0.5 * einsum(
                                'ijwy,jxw->ixy',
                                eris_oovv[ki, kj, kw], r2[kj, kx])
                            s2[ki, kx] += einsum(
                                'iywj,jxw->ixy',
                                eris_ovvo[ki, ky, kw], r2[kj, kx])
                            s2[ki, kx] -= 0.5 * einsum(
                                'iywj,jwx->ixy',
                                eris_ovvo[ki, ky, kw], r2[kj, kw])

                if self.exxdiv is not None:
                    s2 += -madelung * r2
            if (method == "adc(3)"):

                eris_ovoo = eris.ovoo

    # ############### ADC(3) a - ibc block and ibc-a coupling blocks
    # ########################

                t2_1 = self.t2[0]

                for kz in range(nkpts):
                    for kw in range(nkpts):
                        kj = kconserv[kw, kshift, kz]

                        for kl in range(nkpts):
                            km = kconserv[kw, kl, kz]
                            t2_1_lmz = self.t2[0][kl, km, kz]
                            ka = kconserv[km, kj, kl]
                            temp_1 = einsum(
                                'lmzw,jzw->jlm',
                                t2_1_lmz, r2[kj, kz])
                            temp = 0.25 * einsum(
                                'lmzw,jzw->jlm',
                                t2_1_lmz, r2[kj, kz])
                            temp -= 0.25 * einsum(
                                'lmzw,jwz->jlm',
                                t2_1_lmz, r2[kj, kw])
                            del t2_1_lmz

                            t2_1_mlz = self.t2[0][km, kl, kz]
                            temp -= 0.25 * einsum(
                                'mlzw,jzw->jlm',
                                t2_1_mlz, r2[kj, kz])
                            temp += 0.25 * einsum(
                                'mlzw,jwz->jlm',
                                t2_1_mlz, r2[kj, kw])
                            del t2_1_mlz

                            s1 += einsum(
                                'jlm,lamj->a',
                                temp, eris_ovoo[kl, ka, km])
                            s1 -= einsum(
                                'jlm,malj->a',
                                temp, eris_ovoo[km, ka, kl])
                            s1 += einsum(
                                'jlm,lamj->a',
                                temp_1, eris_ovoo[kl, ka, km])

                            del temp
                            del temp_1

                for kx in range(nkpts):
                    for ky in range(nkpts):
                        ki = kconserv[ky, kshift, kx]
                        for kl in range(nkpts):
                            km = kconserv[kx, kl, ky]
                            kb = kconserv[km, ki, kl]
                            temp = einsum(
                                'b,lbmi->lmi',
                                r1, eris_ovoo[kl, kb, km].conj())
                            s2[ki, kx] += einsum(
                                'lmi,lmxy->ixy',
                                temp, t2_1[kl, km, kx].conj())
                            del temp

                for kz in range(nkpts):
                    for kw in range(nkpts):
                        kj = kconserv[kz, kshift, kw]
                        for kl in range(nkpts):
                            kd = kconserv[kj, kw, kl]
                            t2_1_jlw = self.t2[0][kj, kl, kw]

                            temp_s_a = einsum(
                                'jlwd,jzw->lzd',
                                t2_1_jlw, r2[kj, kz])
                            temp_s_a -= einsum(
                                'jlwd,jwz->lzd',
                                t2_1_jlw, r2[kj, kw])

                            temp_t2_r2_1 = einsum(
                                'jlwd,jzw->lzd',
                                t2_1_jlw, r2[kj, kz])
                            temp_t2_r2_1 -= einsum(
                                'jlwd,jwz->lzd',
                                t2_1_jlw, r2[kj, kw])
                            temp_t2_r2_1 += einsum(
                                'jlwd,jzw->lzd',
                                t2_1_jlw, r2[kj, kz])
                            del t2_1_jlw

                            t2_1_ljw = self.t2[0][kl, kj, kw]
                            temp_s_a -= einsum(
                                'ljwd,jzw->lzd',
                                t2_1_ljw, r2[kj, kz])
                            temp_s_a += einsum(
                                'ljwd,jwz->lzd',
                                t2_1_ljw, r2[kj, kw])
                            temp_s_a += einsum(
                                'ljdw,jzw->lzd',
                                t2_1[kl, kj, kd], r2[kj, kz])

                            temp_t2_r2_1 -= einsum(
                                'ljwd,jzw->lzd',
                                t2_1_ljw, r2[kj, kz])
                            del t2_1_ljw
                            temp_t2_r2_4 = einsum(
                                'jldw,jwz->lzd',
                                t2_1[kj, kl, kd], r2[kj, kw])

                            if eris.ovvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):
                                    ka = kconserv[kz, kd, kl]
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kd],
                                        eris.Lvv[kz, ka],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    k = eris_ovvv.shape[0]

                                    s1 += 0.5 * einsum(
                                        'lzd,ldza->a',
                                        temp_s_a[a:a + k], eris_ovvv)
                                    s1 += 0.5 * einsum(
                                        'lzd,ldza->a',
                                        temp_t2_r2_1[a:a + k], eris_ovvv)
                                    del eris_ovvv

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, ka],
                                        eris.Lvv[kz, kd],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    s1 -= 0.5 * einsum(
                                        'lzd,lazd->a',
                                        temp_s_a[a:a + k], eris_ovvv)
                                    s1 -= 0.5 * einsum(
                                        'lzd,lazd->a',
                                        temp_t2_r2_4[a:a + k], eris_ovvv)
                                    del eris_ovvv
                                    a += k
                            else:
                                ka = kconserv[kz, kd, kl]
                                eris_ovvv = eris.ovvv[:]
                                s1 += 0.5 * einsum(
                                    'lzd,ldza->a',
                                    temp_s_a, eris_ovvv[kl, kd, kz])
                                s1 += 0.5 * einsum(
                                    'lzd,ldza->a',
                                    temp_t2_r2_1, eris_ovvv[kl, kd, kz])

                                s1 -= 0.5 * einsum(
                                    'lzd,lazd->a',
                                    temp_s_a, eris_ovvv[kl, ka, kz])
                                s1 -= 0.5 * einsum(
                                    'lzd,lazd->a',
                                    temp_t2_r2_4, eris_ovvv[kl, ka, kz])
                                del eris_ovvv
                            del temp_s_a
                            del temp_t2_r2_1
                            del temp_t2_r2_4

                for kz in range(nkpts):
                    for kw in range(nkpts):
                        kj = kconserv[kz, kshift, kw]
                        for kl in range(nkpts):
                            kd = kconserv[kj, kz, kl]
                            t2_1_jlz = self.t2[0][kj, kl, kz]

                            temp_s_a_1 = -einsum(
                                'jlzd,jwz->lwd',
                                t2_1_jlz, r2[kj, kw])
                            temp_s_a_1 += einsum(
                                'jlzd,jzw->lwd',
                                t2_1_jlz, r2[kj, kz])
                            temp_t2_r2_2 = -einsum(
                                'jlzd,jwz->lwd',
                                t2_1_jlz, r2[kj, kw])
                            temp_t2_r2_2 += einsum(
                                'jlzd,jzw->lwd',
                                t2_1_jlz, r2[kj, kz])
                            temp_t2_r2_2 -= einsum(
                                'jlzd,jwz->lwd',
                                t2_1_jlz, r2[kj, kw])
                            del t2_1_jlz

                            t2_1_ljz = self.t2[0][kl, kj, kz]
                            temp_s_a_1 += einsum(
                                'ljzd,jwz->lwd',
                                t2_1_ljz, r2[kj, kw])
                            temp_s_a_1 -= einsum(
                                'ljzd,jzw->lwd',
                                t2_1_ljz, r2[kj, kz])
                            temp_s_a_1 -= einsum(
                                'ljdz,jwz->lwd',
                                t2_1[kl, kj, kd], r2[kj, kw])

                            temp_t2_r2_2 += einsum(
                                'ljzd,jwz->lwd',
                                t2_1_ljz, r2[kj, kw])

                            temp_t2_r2_3 = -einsum(
                                'ljzd,jzw->lwd',
                                t2_1_ljz, r2[kj, kz])
                            del t2_1_ljz

                            if eris.ovvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):

                                    ka = kconserv[kw, kd, kl]
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kd],
                                        eris.Lvv[kw, ka],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    k = eris_ovvv.shape[0]
                                    s1 -= 0.5 * einsum(
                                        'lwd,ldwa->a',
                                        temp_s_a_1[a:a + k], eris_ovvv)
                                    s1 -= 0.5 * einsum(
                                        'lwd,ldwa->a',
                                        temp_t2_r2_2[a:a + k], eris_ovvv)
                                    del eris_ovvv

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, ka],
                                        eris.Lvv[kw, kd],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    s1 += 0.5 * einsum(
                                        'lwd,lawd->a',
                                        temp_s_a_1[a:a + k], eris_ovvv)
                                    s1 += 0.5 * einsum(
                                        'lwd,lawd->a',
                                        temp_t2_r2_3[a:a + k], eris_ovvv)
                                    del eris_ovvv
                                    a += k

                            else:
                                ka = kconserv[kw, kd, kl]
                                eris_ovvv = eris.ovvv[:]
                                s1 -= 0.5 * einsum(
                                    'lwd,ldwa->a',
                                    temp_s_a_1, eris_ovvv[kl, kd, kw])
                                s1 += 0.5 * einsum(
                                    'lwd,lawd->a',
                                    temp_s_a_1, eris_ovvv[kl, ka, kw])

                                s1 -= 0.5 * einsum(
                                    'lwd,ldwa->a',
                                    temp_t2_r2_2, eris_ovvv[kl, kd, kw])
                                s1 += 0.5 * einsum(
                                    'lwd,lawd->a',
                                    temp_t2_r2_3, eris_ovvv[kl, ka, kw])
                                del eris_ovvv

                            del temp_s_a_1
                            del temp_t2_r2_2
                            del temp_t2_r2_3

                for kx in range(nkpts):
                    for ky in range(nkpts):
                        ki = kconserv[ky, kshift, kx]
                        for kl in range(nkpts):
                            temp = np.zeros(
                                (nocc, nvir, nvir), dtype=np.complex128)
                            temp_1_1 = np.zeros(
                                (nocc, nvir, nvir), dtype=np.complex128)
                            temp_2_1 = np.zeros(
                                (nocc, nvir, nvir), dtype=np.complex128)
                            if eris.ovvv is None:
                                chunk_size = (self.chunk_size)
                                chunk_size = min(chunk_size, nocc)
                                a = 0
                                for p in range(0, nocc, chunk_size):
                                    kd = kconserv[kl, kshift, kx]
                                    kb = kconserv[kl, kd, kx]

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kd],
                                        eris.Lvv[kx, kb],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    k = eris_ovvv.shape[0]
                                    temp_1_1[a:a + k] += einsum(
                                        'ldxb,b->lxd',
                                        eris_ovvv, r1)
                                    temp_2_1[a:a + k] += einsum(
                                        'ldxb,b->lxd',
                                        eris_ovvv, r1)
                                    del eris_ovvv

                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kb],
                                        eris.Lvv[kx, kd],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    temp_1_1[a:a + k] -= einsum(
                                        'lbxd,b->lxd',
                                        eris_ovvv, r1)
                                    del eris_ovvv

                                    kd = kconserv[kl, kshift, ky]
                                    kb = kconserv[ky, kd, kl]
                                    eris_ovvv = _ChemistsERIs.ovvv_df(
                                        eris.Lov[kl, kb],
                                        eris.Lvv[ky, kd],
                                        p,
                                        chunk_size).reshape(
                                        -1, nvir, nvir, nvir) / nkpts
                                    temp[a:a + k] -= einsum(
                                        'lbyd,b->lyd',
                                        eris_ovvv, r1)
                                    del eris_ovvv
                                    a += k
                            else:
                                kd = kconserv[ki, ky, kl]
                                kb = kconserv[kl, kd, kx]
                                eris_ovvv = eris.ovvv[:]
                                temp_1_1 += einsum(
                                    'ldxb,b->lxd',
                                    eris_ovvv[kl, kd, kx], r1)
                                temp_1_1 -= einsum(
                                    'lbxd,b->lxd',
                                    eris_ovvv[kl, kb, kx], r1)
                                temp_2_1 += einsum(
                                    'ldxb,b->lxd',
                                    eris_ovvv[kl, kd, kx], r1)

                                kd = kconserv[kl, kshift, ky]
                                kb = kconserv[ky, kd, kl]
                                temp -= einsum(
                                    'lbyd,b->lyd',
                                    eris_ovvv[kl, kb, ky], r1)
                                del eris_ovvv

                            kd = kconserv[kl, ky, ki]
                            s2[ki, kx] += einsum(
                                'lxd,lidy->ixy',
                                temp_1_1.conj(), t2_1[kl, ki, kd].conj())
                            s2[ki, kx] += einsum(
                                'lxd,ilyd->ixy',
                                temp_2_1.conj(), t2_1[ki, kl, ky].conj())
                            s2[ki, kx] -= einsum(
                                'lxd,ildy->ixy',
                                temp_2_1.conj(), t2_1[ki, kl, kd].conj())

                            kd = kconserv[kl, kx, ki]
                            s2[ki, kx] += einsum(
                                'lyd,lixd->ixy',
                                temp.conj(), t2_1[kl, ki, kx].conj())
                            del temp
                            del temp_1_1
                            del temp_2_1

            s2 = s2.reshape(-1)
            s = np.hstack((s1, s2))
            del s1
            del s2

            cput0 = log.timer_debug1(
                'completed sigma vector calculation', *cput0)

            return s
        return sigma_

    def mask_frozen_ea(self, v1, v2, kshift, const=LARGE_DENOM):
        r"""Replace inactive padding in charged vector blocks by a constant.

        The KMP2 padding maps identify the physical active indices remaining
        after frozen-orbital selection. Both orbital indices and the conserved
        k partner are checked for the two-particle/one-hole block.

        Args:
            v1 (ndarray): Singles vector of shape (nvir,).
            v2 (ndarray): Doubles tensor of shape (nkpts, nkpts, nocc, nvir,
                nvir).
            kshift (int): Sector determining the missing fourth momentum.
            const (scalar): Value for inactive components. The default is
                LARGE_DENOM for the EA preconditioner;
                use zero to mask input guesses.

        Returns:
            tuple[ndarray, ndarray]: Independent masked singles and doubles,
            with the original shapes and dtypes. Neither input is modified.
        """
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        # Get location of padded elements in occupied and virtual space
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self)

        new_v1 = const * np.ones_like(v1)
        new_v2 = const * np.ones_like(v2)

        new_v1[nonzero_vpadding[kshift]] = v1[nonzero_vpadding[kshift]]

        for ki in range(nkpts):
            for ka in range(nkpts):
                kb = kconserv[kshift, ka, ki]
                idx = np.ix_(
                    [ki],
                    [ka],
                    nonzero_opadding[ki],
                    nonzero_vpadding[ka],
                    nonzero_vpadding[kb])
                new_v2[idx] = v2[idx]

        return (new_v1, new_v2)

    def ea_contract_r_vvvv(self, r2, vvvv, ka, kb, kc):
        r"""Contract one EA doubles block with a virtual-particle ladder.

        For the selected momenta, evaluate
        :math:`s_i^{ab}=\sum_{cd} V_{ab,cd}^* r_i^{cd}`, where V is the
        stored particle-particle integral block. Reconstruct virtual
        slices from DF factors when those are supplied instead.

        Args:
            r2 (ndarray): Doubles block of shape (nocc, nvir, nvir).
            vvvv: Five-dimensional Lvv DF factors or an indexable normalized
                stored tensor with three leading k axes and four virtual axes.
            ka, kb, kc (int): Particle momenta; kd = kconserv[ka, kc, kb].
                Reconstruction is limited by this driver's chunk_size.

        Returns:
            ndarray: Contiguous ladder contribution of shape
            (nocc, nvir, nvir). DF reconstruction includes division by nkpts;
            stored integral blocks already contain this normalization.
        """
        nocc = r2.shape[0]
        nvir = r2.shape[1]
        nkpts = self.nkpts
        kconserv = self.khelper.kconserv

        kd = kconserv[ka, kc, kb]
        r2 = np.ascontiguousarray(r2.reshape(nocc, -1))
        r2_vvvv = np.zeros(
            (nvir, nvir, nocc), dtype=np.result_type(r2, self.mo_coeff[0]))
        chunk_size = self.chunk_size
        chunk_size = min(chunk_size, nvir)

        a = 0
        if isinstance(vvvv, np.ndarray) and vvvv.ndim == 5:
            vv1 = vvvv[ka, kc]
            vv2 = vvvv[kb, kd]
            for p in range(0, nvir, chunk_size):
                vvvv_p = _ChemistsERIs.vvvv_df(vv1, vv2, p, chunk_size) / nkpts
                k = vvvv_p.shape[0]
                vvvv_p = vvvv_p.reshape(-1, nvir * nvir)
                r2_vvvv[a:a + k] += np.dot(vvvv_p.conj(), r2.T).reshape(
                    -1, nvir, nocc)
                del vvvv_p
                a += k
        else:
            for p in range(0, nvir, chunk_size):
                vvvv_p = vvvv[ka, kb, kc][p:p + chunk_size].reshape(
                    -1, nvir * nvir)
                k = vvvv_p.shape[0]
                r2_vvvv[a:a + k] += np.dot(vvvv_p.conj(), r2.T).reshape(
                    -1, nvir, nocc)
                del vvvv_p
                a += k

        r2_vvvv = np.ascontiguousarray(r2_vvvv.transpose(2, 0, 1))

        return r2_vvvv

    def get_trans_moments_orbital(self, orb, kshift):
        r"""Build the effective electron-attachment transition vector for one
        MO.

        Ground-state amplitudes dress the leading one-particle transition
        and introduce two-particle/one-hole components. Optional higher-order
        terms follow method and approx_trans_moments.

        Args:
            orb (int): Padded active MO index in [0, nmo), with occupied MOs
                first and virtual MOs beginning at nocc.
            kshift (int): Momentum sector and orbital k point.

        Returns:
            ndarray: Complex vector of length vector_size(). Stacking these
            vectors over orb produces the rows of get_trans_moments().

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcea = kadc_rhf.KRADCEA(kmf).run(0)
            >>> mykradcea.get_trans_moments_orbital(0, 0).shape
            (5,)
        """
        self.method = check_implemented(self.method)

        method = self.method
        nkpts = self.nkpts
        nocc = self.nocc
        nvir = self.nvir
        t2_1 = self.t2[0]

        idn_vir = np.identity(nvir)

        T1 = np.zeros(nvir, dtype=np.complex128)
        T2 = np.zeros((nkpts, nkpts, nocc, nvir, nvir), dtype=np.complex128)

    ######## ADC(2) 1h part  ############################################

        if orb < nocc:

            if (self.approx_trans_moments is False or self.method == "adc(3)"):
                t1_2 = self.t1[0]
                T1 -= t1_2[kshift][orb, :]

            for kj in range(nkpts):
                for ka in range(nkpts):
                    kb = self.khelper.kconserv[kj, ka, kshift]
                    ki = self.khelper.kconserv[ka, kj, kb]

                    t2_1_t = t2_1[ki, kj, ka].transpose(1, 0, 2, 3)
                    T2[kj, ka] -= t2_1_t[:, orb, :, :].conj()

        else:

            T1 += idn_vir[(orb - nocc), :]
            for kk in range(nkpts):
                for kc in range(nkpts):
                    kl = self.khelper.kconserv[kc, kk, kshift]
                    ka = self.khelper.kconserv[kc, kl, kk]
                    T1 -= 0.25 * einsum(
                        'klc,klac->a',
                        t2_1[kk, kl, kshift][:, :, orb - nocc, :],
                        t2_1[kk, kl, ka].conj())
                    T1 -= 0.25 * einsum(
                        'lkc,lkac->a',
                        t2_1[kl, kk, kshift][:, :, orb - nocc, :],
                        t2_1[kl, kk, ka].conj())

                    T1 -= 0.25 * einsum(
                        'klc,klac->a',
                        t2_1[kk, kl, kshift][:, :, orb - nocc, :],
                        t2_1[kk, kl, ka].conj())
                    T1 += 0.25 * einsum(
                        'lkc,klac->a',
                        t2_1[kl, kk, kshift][:, :, orb - nocc, :],
                        t2_1[kk, kl, ka].conj())
                    T1 += 0.25 * einsum(
                        'klc,lkac->a',
                        t2_1[kk, kl, kshift][:, :, orb - nocc, :],
                        t2_1[kl, kk, ka].conj())
                    T1 -= 0.25 * einsum(
                        'lkc,lkac->a',
                        t2_1[kl, kk, kshift][:, :, orb - nocc, :],
                        t2_1[kl, kk, ka].conj())

    ######### ADC(3) 2p-1h  part  ############################################

        if (self.method == 'adc(2)-x' and self.approx_trans_moments is False
            or self.method == 'adc(3)'):
            t2_2 = self.t2[1]

            if orb < nocc:

                for kj in range(nkpts):
                    for ka in range(nkpts):
                        kb = self.khelper.kconserv[kj, ka, kshift]
                        ki = self.khelper.kconserv[ka, kj, kb]

                        t2_2_t = t2_2[ki, kj, ka].conj().transpose(1, 0, 2, 3)

                        T2[kj, ka] -= t2_2_t[:, orb, :, :].conj()

    ########### ADC(3) 1p part  ############################################

        if (method == 'adc(3)'):
            if orb < nocc:
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        ka = self.khelper.kconserv[kk, kc, kshift]
                        T1 += 0.5 * einsum(
                            'kac,ck->a',
                            t2_1[kk, kshift, kc][:, orb, :, :], t1_2[kc].T)
                        T1 -= 0.5 * einsum(
                            'kac,ck->a',
                            t2_1[kshift, kk, ka][orb, :, :, :], t1_2[kc].T)
                        T1 -= 0.5 * einsum(
                            'kac,ck->a',
                            t2_1[kshift, kk, ka][orb, :, :, :], t1_2[kc].T)

            else:
                for kk in range(nkpts):
                    for kc in range(nkpts):
                        kl = self.khelper.kconserv[kk, kc, kshift]
                        ka = self.khelper.kconserv[kl, kc, kk]

                        T1 -= 0.25 * einsum(
                            'klc,klac->a',
                            t2_1[kk, kl, kshift][:, :, orb - nocc, :],
                            t2_2[kk, kl, ka].conj())
                        T1 -= 0.25 * einsum(
                            'lkc,lkac->a',
                            t2_1[kl, kk, kshift][:, :, orb - nocc, :],
                            t2_2[kl, kk, ka].conj())
                        T1 -= 0.25 * einsum(
                            'klc,klac->a',
                            t2_1[kk, kl, kshift][:, :, orb - nocc, :],
                            t2_2[kk, kl, ka].conj())
                        T1 += 0.25 * einsum(
                            'klc,lkac->a',
                            t2_1[kk, kl, kshift][:, :, orb - nocc, :],
                            t2_2[kl, kk, ka].conj())
                        T1 += 0.25 * einsum(
                            'lkc,klac->a',
                            t2_1[kl, kk, kshift][:, :, orb - nocc, :],
                            t2_2[kk, kl, ka].conj())
                        T1 -= 0.25 * einsum(
                            'lkc,lkac->a',
                            t2_1[kl, kk, kshift][:, :, orb - nocc, :],
                            t2_2[kl, kk, ka].conj())

                        T1 -= 0.25 * einsum(
                            'klac,klc->a',
                            t2_1[kk, kl, ka].conj(),
                            t2_2[kk, kl, kshift][:, :, orb - nocc, :])
                        T1 -= 0.25 * einsum(
                            'lkac,lkc->a',
                            t2_1[kl, kk, ka].conj(),
                            t2_2[kl, kk, kshift][:, :, orb - nocc, :])
                        T1 -= 0.25 * einsum(
                            'klac,klc->a',
                            t2_1[kk, kl, ka].conj(),
                            t2_2[kk, kl, kshift][:, :, orb - nocc, :])
                        T1 += 0.25 * einsum(
                            'klac,lkc->a',
                            t2_1[kk, kl, ka].conj(),
                            t2_2[kl, kk, kshift][:, :, orb - nocc, :])
                        T1 += 0.25 * einsum(
                            'lkac,klc->a',
                            t2_1[kl, kk, ka].conj(),
                            t2_2[kk, kl, kshift][:, :, orb - nocc, :])
                        T1 -= 0.25 * einsum(
                            'lkac,lkc->a',
                            t2_1[kl, kk, ka].conj(),
                            t2_2[kl, kk, kshift][:, :, orb - nocc, :])

            del t2_2
        del t2_1

        for ka in range(nkpts):
            for kb in range(nkpts):
                ki = self.khelper.kconserv[kb, kshift, ka]
                T2[ki, ka] += T2[ki, ka] - T2[ki, kb].transpose(0, 2, 1)

        T2 = T2.reshape(-1)
        T = np.hstack((T1, T2))

        return T

    def make_rdm1_eigenvectors_slow(self, L, R, kshift):
        r"""Evaluate the retained explicit-loop charged density expression.

        This reference path exposes the orbital and momentum contractions used
        for the active-space density. Production root calculations use the
        blocked make_rdm1_eigenvectors() implementation instead.

        Args:
            L, R (array_like): Left and right state vectors of length
                vector_size(). L is conjugated internally; supply the original
                ket coefficients, not an already conjugated bra.
            kshift (int): Charged momentum sector.

        Returns:
            ndarray: Complex active MO density of shape (nkpts, nmo, nmo).
            Ground amplitudes must already be present; inputs are copied.
            This legacy path has no additional if_ss reference-density flag.
        """
        L = np.array(L).ravel().conj()
        R = np.array(R).ravel()

        t2_ov = self.t1[0]
        t1_oovv = self.t2[0]

        kconserv = self.khelper.kconserv

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts

        n_singles = nvir
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        kd_oc = np.identity(nocc)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        R2 = R2.reshape(nkpts, nkpts, nocc, nvir, nvir)

    # block- ij
        # 000
        for ki in range(nkpts):
            rdm1[ki][:nocc, :nocc] += 2 * einsum('ij,m,m->ij', kd_oc, L1, R1)
        # 101
        for ka in range(nkpts):
            for ki in range(nkpts):
                kb = kconserv[ki, ka, kshift]
                rdm1[ki][:nocc, :nocc] -= 2 * einsum(
                    'Jab,Iab->IJ',
                    L2[ki][ka], R2[ki][ka])
                rdm1[ki][:nocc, :nocc] += einsum(
                    'Jab,Iba->IJ',
                    L2[ki][ka], R2[ki][kb])
        for kj in range(nkpts):
            for ki in range(nkpts):
                for ka in range(nkpts):
                    kb = kconserv[ki, ka, kshift]
                    rdm1[kj][:nocc, :nocc] += 4 * einsum(
                        'iab,iab,IJ->IJ',
                        L2[ki][ka], R2[ki][ka], kd_oc)
                    rdm1[kj][:nocc, :nocc] -= 2 * einsum(
                        'iab,iba,IJ->IJ',
                        L2[ki][ka], R2[ki][kb], kd_oc)
        # 020
        for kj in range(nkpts):
            for ki in range(nkpts):
                for kb in range(nkpts):
                    kc = kconserv[kj, kb, ki]
                    rdm1[kj][:nocc, :nocc] -= 4 * einsum(
                        'a,a,Iibc,Jibc->IJ',
                        L1, R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kb].conj()
                    )
                    rdm1[kj][:nocc, :nocc] += 2 * einsum(
                        'a,a,Iibc,Jicb->IJ',
                        L1, R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kc].conj()
                    )
                kb = kshift
                kc = kconserv[kj, kb, ki]
                rdm1[kj][:nocc, :nocc] += 2 * einsum(
                    'a,b,Iiac,Jibc->IJ',
                    L1, R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kb].conj())
                rdm1[kj][:nocc, :nocc] -= einsum(
                    'a,b,Iiac,Jicb->IJ',
                    L1, R1, t1_oovv[kj][ki][kb], t1_oovv[kj][ki][kc].conj())
                rdm1[kj][:nocc, :nocc] -= einsum(
                    'a,b,Iica,Jibc->IJ',
                    L1, R1, t1_oovv[kj][ki][kc], t1_oovv[kj][ki][kb].conj())
                rdm1[kj][:nocc, :nocc] += 2 * einsum(
                    'a,b,Iica,Jicb->IJ',
                    L1, R1, t1_oovv[kj][ki][kc], t1_oovv[kj][ki][kc].conj())

    # block- ab
        # 000
        rdm1[kshift][nocc:, nocc:] += einsum('A,B->AB', L1, R1)
        # 101
        for ka in range(nkpts):
            for ki in range(nkpts):
                kb = kconserv[ki, ka, kshift]
                rdm1[kb][nocc:, nocc:] += 2 * einsum(
                    'iAa,iBa->AB',
                    L2[ki][kb], R2[ki][kb])
                rdm1[kb][nocc:, nocc:] -= einsum(
                    'iAa,iaB->AB',
                    L2[ki][kb], R2[ki][ka])
                rdm1[kb][nocc:, nocc:] -= einsum(
                    'iaA,iBa->AB',
                    L2[ki][ka], R2[ki][kb])
                rdm1[kb][nocc:, nocc:] += 2 * einsum(
                    'iaA,iaB->AB',
                    L2[ki][ka], R2[ki][ka])
        # 020
        for ki in range(nkpts):
            for kj in range(nkpts):
                ka = kshift
                kb = kconserv[ki, ka, kj]
                rdm1[ka][nocc:, nocc:] -= einsum(
                    'A,a,ijBb,ijab->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ka][nocc:, nocc:] += 1 / 2 * einsum(
                    'A,a,ijBb,ijba->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())
                rdm1[ka][nocc:, nocc:] -= einsum(
                    'a,B,ijab,ijAb->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ka][nocc:, nocc:] += 1 / 2 * einsum(
                    'a,B,ijab,ijbA->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())
                for ka in range(nkpts):
                    kb = kconserv[ki, ka, kj]
                    rdm1[ka][nocc:, nocc:] += 4 * einsum(
                        'a,a,ijBb,ijAb->AB',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj()
                    )
                    rdm1[ka][nocc:, nocc:] -= 2 * einsum(
                        'a,a,ijBb,ijbA->AB',
                        L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj()
                    )
                kb = kshift
                ka = kconserv[ki, kb, kj]
                rdm1[ka][nocc:, nocc:] -= 2 * einsum(
                    'a,b,ijBa,ijAb->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][ka].conj())
                rdm1[ka][nocc:, nocc:] += einsum(
                    'a,b,ijBa,ijbA->AB',
                    L1, R1, t1_oovv[ki][kj][ka], t1_oovv[ki][kj][kb].conj())

    # block- ia
        # 100
        for ki in range(nkpts):
            rdm1[ki][:nocc, nocc:] -= einsum('a,IAa->IA', L1, R2[ki][ki])
            ka = kshift
            rdm1[ki][:nocc, nocc:] += 2 * einsum('a,IaA->IA', L1, R2[ki][ka])
        # 110
        for ki in range(nkpts):
            for ka in range(nkpts):
                kj = kshift
                kb = kconserv[ki, ka, kj]
                rdm1[kj][:nocc, nocc:] -= 2 * einsum(
                    'iab,A,Iiab->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][ka])
                rdm1[kj][:nocc, nocc:] += einsum(
                    'iab,A,Iiba->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kb])
            for kj in range(nkpts):
                ka = kshift
                kb = kconserv[kj, kj, ki]
                rdm1[kj][:nocc, nocc:] += 4 * einsum(
                    'iab,a,IiAb->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kj])
                rdm1[kj][:nocc, nocc:] -= 2 * einsum(
                    'iab,a,IibA->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kb])
                rdm1[kj][:nocc, nocc:] -= 2 * einsum(
                    'iab,b,IiAa->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][kj])
                rdm1[kj][:nocc, nocc:] += einsum(
                    'iab,b,IiaA->IA',
                    L2[ki][ka], R1, t1_oovv[kj][ki][ka])

    # block- ai
        for ki in range(nkpts):
            rdm1[ki][nocc:, :nocc] = rdm1[ki][:nocc, nocc:].conj().T

        if self.approx_trans_moments is False or self.method == "adc(3)":
            ki = kshift
            ### block- ia ###
            rdm1[ki][:nocc, nocc:] -= einsum('a,A,Ia->IA', L1, R1, t2_ov[ki])
            for kj in range(nkpts):
                rdm1[kj][:nocc, nocc:] += 2 * einsum(
                    'a,a,IA->IA',
                    L1, R1, t2_ov[kj])
            ### block- ai ###
                rdm1[kj][nocc:, :nocc] = rdm1[kj][:nocc, nocc:].conj().T

        return rdm1

    def make_rdm1_eigenvectors(self, L, R, kshift, if_ss):
        r"""Contract left/right charged vectors into an active MO density.

        The blocked contractions retain three independent ground-amplitude
        k axes and the sector-conserving doubles layout. Occupied-virtual
        ground-state terms follow approx_trans_moments and the ADC order.

        Args:
            L, R (array_like): Metric-normalized state vectors of length
                vector_size(). Each is copied; L is conjugated internally.
                For an ordinary state density, pass the same ket vector twice.
            kshift (int): Charged momentum sector.
            if_ss (bool): Add the reference-density contribution used in
                state-specific FNO generation; False gives the charged density.

        Returns:
            ndarray: Complex active MO density of shape (nkpts, nmo, nmo).
            The routine neither restores frozen orbitals nor transforms to AOs.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEA
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcea = kadc_rhf.KRADCEA(kmf).run(0)
            >>> excitation_results = mykradcea.kernel(1, kptlist=[0])
            >>> ket = mykradcea.renormalize_eigenvectors(
            ...     0, mykradcea.U[0].T.copy(), 1)
            >>> density = mykradcea.make_rdm1_eigenvectors(
            ...     ket[:, 0], ket[:, 0], 0, False)
            >>> density.shape
            (2, 2, 2)
        """
        L = np.array(L).ravel().conj()
        R = np.array(R).ravel()

        t2_ov = self.t1[0]
        t1_oovv = self.t2[0]

        kconserv = self.khelper.kconserv
        kconserv_s = kconserv[kshift, :, :]

        nocc = self.nocc
        nmo = self.nmo
        nvir = nmo - nocc
        nkpts = self.nkpts
        occ_list = range(nocc)

        t1_oovv_sjb = np.zeros_like(t1_oovv[0])
        t1_oovv_sja = t1_oovv[kshift]
        t1_oovv_ijs = t1_oovv[:, :, kshift, :, :, :]

        kj, ka = np.indices((nkpts, nkpts))
        kb = self.khelper.kconserv[kshift, ka, kj]
        t1_oovv_sjb[kj, ka] = t1_oovv_sja[kj, kb]

        n_singles = nvir
        n_doubles = nkpts * nkpts * nocc * nvir * nvir

        s1 = 0
        f1 = n_singles
        s2 = f1
        f2 = s2 + n_doubles

        rdm1 = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)

        L1 = L[s1:f1]
        L2 = L[s2:f2]
        R1 = R[s1:f1]
        R2 = R[s2:f2]

        L2 = L2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        R2 = R2.reshape(nkpts, nkpts, nocc, nvir, nvir)
        R2_iba = np.zeros_like(R2)
        L2_iba = np.zeros_like(L2)

        ka, ki = np.indices((nkpts, nkpts))
        kb = kconserv_s[ka, ki]
        R2_iba[ki, ka] = R2[ki, kb]
        L2_iba[ki, ka] = L2[ki, kb]
        R2_isb = R2[:, kshift, :, :, :]
        L2_isb = L2[:, kshift, :, :, :]
    # block- ij
        # 000
        rdm1[:, occ_list, occ_list] += 2 * einsum('m,m->', L1, R1)
        # 101
        rdm1[:, :nocc, :nocc] -= 2 * einsum('KkJab,KkIab->KIJ', L2, R2)
        rdm1[:, :nocc, :nocc] += einsum('KkJab,KkIba->KIJ', L2, R2_iba)
        rdm1[:, occ_list, occ_list] += 4 * einsum('Kkiab,Kkiab->', L2, R2)
        rdm1[:, occ_list, occ_list] -= 2 * einsum('Kkiab,Kkiba->', L2, R2_iba)
        # 020
        rdm1[:, :nocc, :nocc] += 2 * einsum(
            'a,b,KkIiac,KkJibc->KIJ',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs.conj())

    # block- ab
        # 000
        rdm1[kshift, nocc:, nocc:] += einsum('A,B->AB', L1, R1)
        # 101
        rdm1[:, nocc:, nocc:] += 2 * einsum('kKiAa,kKiBa->KAB', L2, R2)
        rdm1[:, nocc:, nocc:] -= einsum('kKiAa,kKiaB->KAB', L2, R2_iba)
        rdm1[:, nocc:, nocc:] -= einsum('kKiaA,kKiBa->KAB', L2_iba, R2)
        rdm1[:, nocc:, nocc:] += 2 * einsum('kKiaA,kKiaB->KAB', L2_iba, R2_iba)
        # 020
        rdm1[kshift, nocc:, nocc:] -= einsum(
            'A,a,kKijBb,kKijab->AB',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs.conj())
        rdm1[kshift, nocc:, nocc:] -= einsum(
            'a,B,kKijab,kKijAb->AB',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs.conj())

    # block- ia
        # 100
        rdm1[:, :nocc, nocc:] -= einsum('a,KKIAa->KIA', L1, R2)
        rdm1[:, :nocc, nocc:] += 2 * einsum('a,KIaA->KIA', L1, R2_isb)
        # 110
        rdm1[kshift, :nocc, nocc:] -= 2 * einsum(
            'kKiab,A,kKIiab->IA',
            L2, R1, t1_oovv_sja)
        rdm1[kshift, :nocc, nocc:] += einsum(
            'kKiab,A,kKIiba->IA',
            L2, R1, t1_oovv_sjb)
        rdm1[:, :nocc, nocc:] += einsum(
            'kiab,b,KkIiaA->KIA',
            L2_isb, R1, t1_oovv_ijs)

        del (R2_iba)
        del (L2_iba)
        del (R2_isb)
        del (t1_oovv_sjb)
        del (t1_oovv_sja)

        kj, ka = np.indices((nkpts, nkpts))
        idx0 = np.arange(nkpts)[:, None]
        LR = np.dot(L1, R1)
        if if_ss:
            rdm1[:, :nocc, :nocc] += np.identity(nocc)
            LR += 1.
        t1_oovv_ijs_b = np.zeros_like(t1_oovv[0])
        oo = np.zeros((nkpts, nocc, nocc), dtype=np.complex128)
        vv = np.zeros((nkpts, nvir, nvir), dtype=np.complex128)
        ov = np.zeros((nkpts, nocc, nvir), dtype=np.complex128)
        for ki in range(nkpts):
            kb = self.khelper.kconserv[ki, ka, kj]
            t1_oovv_np = np.array(t1_oovv[ki])
            t1_oovv_ijb = t1_oovv_np[idx0, kb]
            t1_oovv_ijs_b[ki] = t1_oovv_ijb[:, kshift].copy()
    # block- ij
            oo[ki] -= 4 * einsum(
                'KkIibc,KkJibc->IJ',
                t1_oovv_np, t1_oovv_np.conj())
            oo[ki] += 2 * einsum(
                'KkIibc,KkJicb->IJ',
                t1_oovv_np, t1_oovv_ijb.conj())
    # block- ab
            vv += 4 * einsum(
                'kKijBb,kKijAb->KAB',
                t1_oovv_np, t1_oovv_np.conj())
            vv -= 2 * einsum(
                'kKijBb,kKijbA->KAB',
                t1_oovv_np, t1_oovv_ijb.conj())
    # block- ia
            ov[ki] += 4 * einsum(
                'kiab,a,kIiAb->IA',
                L2_isb, R1, t1_oovv_np[:, ki])
            ov[ki] -= 2 * einsum('kiab,a,kkIibA->IA', L2_isb, R1, t1_oovv_np)
            ov[ki] -= 2 * einsum(
                'kiab,b,kIiAa->IA',
                L2_isb, R1, t1_oovv_np[:, ki])
            del (t1_oovv_ijb)
            del (t1_oovv_np)

        oo *= LR
        vv *= LR
        rdm1[:, :nocc, :nocc] += oo
        rdm1[:, nocc:, nocc:] += vv
        rdm1[:, :nocc, nocc:] += ov
        del (oo)
        del (vv)
        del (ov)
        del (L2_isb)

        ki, ka = np.indices((nkpts, nkpts))
        kj = kconserv_s[ki, ka]
        t1_oovv_ijas = t1_oovv_ijs_b[idx0, kj]
        t1_oovv_ijsa = t1_oovv_ijs[idx0, kj]
    # block- ij
        rdm1[:, :nocc, :nocc] -= einsum(
            'a,b,KkIiac,KkJicb->KIJ',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs_b.conj())
        rdm1[:, :nocc, :nocc] -= einsum(
            'a,b,KkIica,KkJibc->KIJ',
            L1, R1, t1_oovv_ijs_b, t1_oovv_ijs.conj())
        rdm1[:, :nocc, :nocc] += 2 * einsum(
            'a,b,KkIica,KkJicb->KIJ',
            L1, R1, t1_oovv_ijs_b, t1_oovv_ijs_b.conj())
    # block- ab
        rdm1[kshift, nocc:, nocc:] += 1 / 2 * einsum(
            'A,a,kKijBb,kKijba->AB',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs_b.conj())
        rdm1[kshift, nocc:, nocc:] += 1 / 2 * einsum(
            'a,B,kKijab,kKijbA->AB',
            L1, R1, t1_oovv_ijs, t1_oovv_ijs_b.conj())
        rdm1[:, nocc:, nocc:] -= 2 * einsum(
            'a,b,kKijBa,kKijAb->KAB',
            L1, R1, t1_oovv_ijas, t1_oovv_ijas.conj())
        rdm1[:, nocc:, nocc:] += einsum(
            'a,b,kKijBa,kKijbA->KAB',
            L1, R1, t1_oovv_ijas, t1_oovv_ijsa.conj())
        del (t1_oovv_ijas)
        del (t1_oovv_ijsa)
        del (t1_oovv_ijs)
        del (t1_oovv_ijs_b)

    # block- ai
        rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(0, 2, 1)

        if self.approx_trans_moments is False or self.method == "adc(3)":
            rdm1[kshift, :nocc, nocc:] -= einsum(
                'a,A,Ia->IA',
                L1, R1, t2_ov[kshift])
            rdm1[:, :nocc, nocc:] += 2 * LR * t2_ov
            ### block- ai ###
            rdm1[:, nocc:, :nocc] = rdm1[:, :nocc, nocc:].conj().transpose(
                0, 2, 1)

        return rdm1

    @property
    def singles_shape(self):
        r"""Describe the EA singles block before C-order flattening.

        Returns:
            tuple[int]: (nvir,), including active-space padding. The selected
            momentum sector fixes the one-particle momentum.
        """
        return (self.nvir,)

    @property
    def doubles_shape(self):
        r"""Return the EA doubles shape in the matrix-action axis order.

        Returns:
            tuple[int]: nkpts, nkpts, nocc, nvir, nvir,
            including padding; the sector fixes the unstored third momentum.
        """
        return (self.nkpts, self.nkpts, self.nocc, self.nvir, self.nvir)

    _mask_frozen = mask_frozen_ea


class KRADCEE(KRADCBase):
    r"""Solve singlet periodic EE-ADC and compute momentum-resolved density
    response.

    Args:
        mf (KRHF, Cell or KRADCBase): Reference, periodic cell, or another ADC
            object. kernel() prepares reference amplitudes before solving
            roots. ADC conversion preserves common data and clears sector
            results.
        frozen (int or sequence, optional): KMP2 frozen-space override.
        mo_coeff, mo_occ, mo_energy (sequence[ndarray], optional): Per-k
            orbital overrides. See KRADCBase.__init__ for inheritance and reset
            rules.
        kpts (array_like, optional): Cartesian inverse-Bohr mesh for a Cell;
            shape (nkpts, 3). None uses Gamma. See KRADCBase for shared
            settings.

    Attributes:
        method (str): 'adc(2)', 'adc(2)-x' or 'adc(3)'.
        E (ndarray): Excitation energies in Hartree, shape (nsectors, nroots).
        U (ndarray): Singlet-metric eigenvectors, shape (nsectors, nroots,
            vector_size()).
        X (ndarray): Spin-free transition MO densities, shape (nsectors,
            nroots, nkpts, nmo, nmo).
        T (ndarray): Flattened view of X, shape (nsectors, nroots,
            nkpts*nmo*nmo), or None without properties.
        P (ndarray): Dimensionless density-response strengths per primitive
            cell, shape (nsectors, nroots).
        density_gv (ndarray): Reciprocal vector G, in Cartesian inverse Bohr.
            Properties use the operator exp[-i(q+G).r], q=-kpts[kshift].
        converged (ndarray): Boolean root-convergence flags with E's shape.

    Singles have axes (k_i, i, a); doubles have axes (k_i, k_j, k_a, i, j, a,
    b), with the remaining particle momentum fixed by k_i+k_j-k_a-k_b=kshift
    modulo a reciprocal vector. Vectors concatenate these blocks in C order.
    All saved sector axes follow kptlist order.

    The inherited run() method returns this object for chaining. Use run(0) to
    prepare only the reference, or run(nroots) for all sectors. Use kernel(...,
    kptlist=...) when selecting individual momentum sectors.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
        >>> excitation_results = mykradcee.kernel(2, kptlist=[0])
        >>> energies = mykradcee.E
        >>> vectors = mykradcee.U
        >>> strengths = mykradcee.P
        >>> densities = mykradcee.X
        >>> energies.round(6).tolist()
        [[1.756312, 2.102047]]
        >>> vectors.shape, strengths.shape, densities.shape
        ((1, 2, 10), (1, 2), (1, 2, 2, 2, 2))
    """

    # __init__ and kernel are inherited from KRADCBase. The sector
    # routines below follow their calls through the shared driver.

    def get_imds(self, eris=None, kshift=0):
        r"""Assemble the singles effective Hamiltonian through the ADC order.

        .. math::
            M_{11} = M^{(0+1)} + \Delta M^{(2)}
                     + \delta_{n,3}\Delta M^{(3)}.

        Args:
            eris (_ChemistsERIs, optional): Pretransformed MO integrals.
                With None, call ao2mo().
            kshift (int): Hole-minus-particle momentum sector.

        Returns:
            ndarray: A matrix of shape (n_singles, n_singles), using the
            flattened (k_I, I, D) and (k_L, L, A) ordering. All order
            builders accumulate into one tensor to avoid an extra dense
            Hamiltonian allocation per perturbative order.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee.get_imds(eris, kshift=1)
            >>> M.shape
            (2, 2)
        """
        self.method = check_implemented(self.method)
        if eris is None:
            eris = self.ao2mo()
        M_ph = self._get_adc1_ham(eris, kshift)
        self._get_adc2_ham(eris, kshift, M=M_ph)
        if self.method == 'adc(3)':
            self._get_adc3_ham(eris, kshift, M=M_ph)
        return M_ph.transpose(0, 2, 3, 1, 4, 5).reshape(
            self.n_singles, self.n_singles)

    def renormalize_eigenvectors(self, kshift, U, nroots=1):
        r"""Return copies of EE eigenvectors normalized in the singlet metric.

        The norm is :math:`U_1^\dagger U_1+U_2^\dagger U_2
        -\frac12 U_2^\dagger P_h U_2`. The hole exchange P_h swaps both
        occupied
        orbital and hole k indices. This convention matches the molecular limit
        while retaining the full, uncompressed doubles tensor.

        Args:
            kshift (int): Sector label accepted by the shared normalization
                API.
                The hole-exchange implementation itself does not need this
                index.
            U (ndarray): Columns of shape (vector_size(), nroots). Integer
                input
                is promoted to floating point. The original array is not
                modified.
            nroots (int): Number of columns to normalize; must match U's shape.

        Returns:
            ndarray: Independent normalized eigenvectors with U's shape.

        Raises:
            ValueError: U has the wrong shape or a nonpositive/zero metric
            norm.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf)
            >>> vectors = np.zeros((mykradcee.vector_size(), 1))
            >>> vectors[0, 0] = 2
            >>> normalized = mykradcee.renormalize_eigenvectors(0, vectors)
            >>> float(normalized[0, 0]), float(vectors[0, 0])
            (1.0, 2.0)
        """
        if U.shape != (self.vector_size(), nroots):
            raise ValueError(
                'Eigenvectors must have shape (vector_size, nroots).')
        vectors = np.array(U, dtype=np.result_type(U, float), copy=True)
        for root in range(nroots):
            singles = vectors[:self.n_singles, root]
            doubles = vectors[self.n_singles:, root].reshape(
                self.doubles_shape)
            exchanged = doubles.transpose(1, 0, 2, 4, 3, 5, 6)
            norm = (np.vdot(singles, singles) + np.vdot(doubles, doubles)
                    - 0.5 * np.vdot(doubles, exchanged)).real
            if norm <= LOOSE_ZERO_TOL:
                raise ValueError(
                    'Cannot normalize an EE vector with zero norm.')
            vectors[:, root] /= np.sqrt(norm)
        return vectors

    def get_properties(self, kshift, U, nroots=1):
        r"""Contract transition densities into density-response strengths per
        cell.

        For the Fourier density matrix rho and the spin-free transition density
        X,

        .. math::
            P_n(q+G)=\frac{2}{N_k}\left|\sum_{kpq}
                \rho_{kpq}(q+G)X_{nkpq}\right|^2.

        The factor of two follows the singlet convention. These are
        dimensionless
        density-response strengths, distinct from optical oscillator strengths.

        Args:
            kshift (int): Momentum sector; physical q=-kpts[kshift].
            U (ndarray): Eigenvector columns of shape (vector_size(), nroots).
                Use a normalized copy, preserving the supplied U.
            nroots (int): Number of roots, which must equal U's column count.

        Returns:
            tuple[ndarray, ndarray]: Real P of shape (nroots,) and transition
            MO
            densities X of shape (nroots, nkpts, nmo, nmo). Their flattened
            form
            T is cached by get_trans_moments() for this sector.

        Raises:
            ValueError: nroots does not match the supplied eigenvector columns.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> excitation_results = mykradcee.kernel(1, kptlist=[1])
            >>> strengths, densities = mykradcee.get_properties(
            ...     1, mykradcee.U[0].T, 1)
            >>> strengths.shape, densities.shape
            ((1,), (1, 2, 2, 2))
        """
        if U.shape[1] != nroots:
            raise ValueError('nroots does not match the eigenvector array.')
        T = self.get_trans_moments(kshift, U)
        X = T.reshape(nroots, self.nkpts, self.nmo, self.nmo)
        amplitudes = einsum(
            'kpq,nkpq->n',
            self.get_density_operator(kshift), X)
        P = (2.0 / self.nkpts) * abs(amplitudes)**2
        return P.real, X

    def _get_adc1_ham(self, eris, kshift=0):
        r"""Build the zeroth- and first-order effective Hamiltonian M matrix.

        M denotes the ADC effective Hamiltonian (terminology as in
        doi:10.1063/1.5131771). Here M_ph stores the neutral particle-hole
        singles block, with paired hole/particle row and column indices.

        In the spin-adapted singles basis, the orbital block is

        .. math::
            M^{(0+1)}_{ID,LA} = (\epsilon_D-\epsilon_I)
            \delta_{IL}\delta_{DA} + 2(LA|DI) - (IL|AD)^*.

        Args:
            eris (_ChemistsERIs): Chemists' MO integral blocks, normalized by
                the number of k points.
            kshift (int): Hole-minus-particle momentum sector.

        Returns:
            ndarray: M with axes (k_I, k_L, I, D, L, A), shape (nkpts, nkpts,
            nocc, nvir, nocc, nvir). The particle momenta follow the selected
            sector. No ground-state amplitudes are used.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> with mykradcee.ao2mo() as eris:
            ...     M_ph = mykradcee._get_adc1_ham(eris, kshift=1)
            >>> M_ph.shape
            (2, 2, 1, 1, 1, 1)
        """
        e_occ, e_vir = self.e_occ, self.e_vir
        ia, ai, doubles = self._momentum_maps(kshift)
        M_ph = np.zeros(self._singles_ham_shape, dtype=self.padded_coeff.dtype)
        for ki in range(self.nkpts):
            kd = ia[ki]
            # Prototype:
            # M[ki, ki] += einsum('A,AD,IL->IDLA', e_vir[kd],
            #                     np.eye(self.nvir), np.eye(self.nocc))
            # M[ki, ki] -= einsum('L,AD,IL->IDLA', e_occ[ki],
            #                     np.eye(self.nvir), np.eye(self.nocc))
            # A repeated-index view updates only the i-a diagonal, without
            # constructing identities or full four-index energy tensors.
            diagonal = np.einsum('idid->id', M_ph[ki, ki])
            diagonal += e_vir[kd][None, :] - e_occ[ki][:, None]
        for ki, kl in product(range(self.nkpts), repeat=2):
            kd, ka = ia[ki], ia[kl]
            M_ph[ki, kl] += 2 * eris.ovvo[kl, ka, kd].transpose(3, 2, 0, 1)
            M_ph[ki, kl] -= eris.oovv[ki, kl, ka].conj().transpose(0, 3, 1, 2)
        return M_ph

    def _get_adc2_ham(self, eris, kshift=0, M=None):
        r"""Accumulate the second-order correction to the singles M matrix.

        M denotes the ADC effective Hamiltonian (terminology as in
        doi:10.1063/1.5131771). Here M_ph stores the neutral particle-hole
        singles block, with paired hole/particle row and column indices.

        The correction contracts first-order doubles amplitudes with MO
        integrals. ADC(2)-x uses the same singles block as ADC(2).

        Args:
            eris (_ChemistsERIs): Active-space MO integral container.
            kshift (int): Hole-minus-particle momentum sector.
            M (ndarray, optional): Existing Hamiltonian tensor to update in
                place. With None, allocate a zero tensor instead.

        Returns:
            ndarray: The updated M, with axes (k_I, k_L, I, D, L, A). If M was
            omitted, this contains only :math:`\Delta M^{(2)}`.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> with mykradcee.ao2mo() as eris:
            ...     M_ph = mykradcee._get_adc1_ham(eris, kshift=1)
            ...     updated = mykradcee._get_adc2_ham(eris, 1, M=M_ph)
            >>> updated is M_ph, updated.shape
            (True, (2, 2, 1, 1, 1, 1))
        """
        M_ph = M
        nkpts, nocc, nvir = self.nkpts, self.nocc, self.nvir
        kconserv = self.khelper.kconserv
        kshift_ia, ai, doubles = self._momentum_maps(kshift)
        v_ovov = eris.ovov
        t1_oovv = t1d_oovv = self.t2[0]
        if M_ph is None:
            M_ph = np.zeros(self._singles_ham_shape, dtype=t1_oovv.dtype)
        occ_identity, vir_identity = np.eye(nocc), np.eye(nvir)
        # ADC(2)
        for kI in range(nkpts):
            kD = kshift_ia[kI]
            for kL in range(nkpts):
                kA = kshift_ia[kL]
                for ki in range(nkpts):
                    M_ph[kI, kL] += 2 * einsum(
                        'IiDa,LAia->IDLA',
                        t1_oovv[kI, ki, kD], v_ovov[kL, kA, ki])
                    M_ph[kI, kL] -= einsum(
                        'IiDa,iALa->IDLA',
                        t1_oovv[kI, ki, kD], v_ovov[ki, kA, kL])
                    M_ph[kI, kL] += 2 * einsum(
                        'LiAa,IDia->IDLA',
                        t1d_oovv[kL, ki, kA].conj(), v_ovov[kI, kD, ki].conj())
                    M_ph[kI, kL] -= einsum(
                        'LiAa,iDIa->IDLA',
                        t1d_oovv[kL, ki, kA].conj(), v_ovov[ki, kD, kI].conj())
                    M_ph[kI, kL] -= einsum(
                        'iIDa,LAia->IDLA',
                        t1_oovv[ki, kI, kD], v_ovov[kL, kA, ki])
                    M_ph[kI, kL] += 0.5 * einsum(
                        'iIDa,iALa->IDLA',
                        t1_oovv[ki, kI, kD], v_ovov[ki, kA, kL])
                    M_ph[kI, kL] -= einsum(
                        'iLAa,IDia->IDLA',
                        t1d_oovv[ki, kL, kA].conj(), v_ovov[kI, kD, ki].conj())
                    M_ph[kI, kL] += 0.5 * einsum(
                        'iLAa,iDIa->IDLA',
                        t1d_oovv[ki, kL, kA].conj(), v_ovov[ki, kD, kI].conj())
                if kI == kL:
                    M_IL = np.zeros((nocc, nocc), dtype=M_ph.dtype)
                    M_DA = np.zeros((nvir, nvir), dtype=M_ph.dtype)
                    for ki in range(nkpts):
                        for ka in range(nkpts):
                            kb = kconserv[kI, ka, ki]
                            M_IL -= einsum(
                                'Iiab,Laib->IL',
                                t1_oovv[kI, ki, ka], v_ovov[kL, ka, ki])
                            M_IL += 0.5 * einsum(
                                'Iiab,Lbia->IL',
                                t1_oovv[kI, ki, ka], v_ovov[kL, kb, ki])
                            M_IL -= einsum(
                                'Liab,Iaib->IL',
                                t1d_oovv[kL, ki, ka].conj(),
                                v_ovov[kI, ka, ki].conj())
                            M_IL += 0.5 * einsum(
                                'Liab,Ibia->IL',
                                t1d_oovv[kL, ki, ka].conj(),
                                v_ovov[kI, kb, ki].conj())
                        for kj in range(nkpts):
                            M_DA -= einsum(
                                'ijAa,iDja->DA',
                                t1d_oovv[ki, kj, kA].conj(),
                                v_ovov[ki, kD, kj].conj())
                            M_DA += 0.5 * einsum(
                                'ijAa,jDia->DA',
                                t1d_oovv[ki, kj, kA].conj(),
                                v_ovov[kj, kD, ki].conj())
                            M_DA -= einsum(
                                'ijDa,iAja->DA',
                                t1_oovv[ki, kj, kD], v_ovov[ki, kA, kj])
                            M_DA += 0.5 * einsum(
                                'ijDa,jAia->DA',
                                t1_oovv[ki, kj, kD], v_ovov[kj, kA, ki])
                    M_ph[kI, kL] += einsum('IL,DA->IDLA', M_IL, vir_identity)
                    M_ph[kI, kL] += einsum('IL,DA->IDLA', occ_identity, M_DA)

        return M_ph

    def _get_adc3_ham(self, eris, kshift=0, M=None):
        r"""Accumulate the third-order correction to the singles M matrix.

        M denotes the ADC effective Hamiltonian (terminology as in
        doi:10.1063/1.5131771). Here M_ph stores the neutral particle-hole
        singles block, with paired hole/particle row and column indices.

        Ground-state singles and first-/second-order doubles enter the
        contractions in SINGLES_3. MomentumContractions retains independent k
        axes and selects the conjugations for complex Bloch orbitals.

        Args:
            eris (_ChemistsERIs): Active-space MO integrals or DF factors.
            kshift (int): Hole-minus-particle momentum sector.
            M (ndarray, optional): Existing Hamiltonian tensor to update in
                place. With None, allocate a zero tensor instead.

        Returns:
            ndarray: The updated M, shape _singles_ham_shape. If M was omitted,
            the result contains only :math:`\Delta M^{(3)}`.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf)
            >>> mykradcee.method = 'adc(3)'
            >>> mykradcee = mykradcee.run(0)
            >>> with mykradcee.ao2mo() as eris:
            ...     M_ph = mykradcee._get_adc1_ham(eris, kshift=1)
            ...     updated = mykradcee._get_adc3_ham(eris, 1, M=M_ph)
            >>> updated is M_ph, updated.shape
            (True, (2, 2, 1, 1, 1, 1))
        """
        M_ph = M
        if M_ph is None:
            M_ph = np.zeros(self._singles_ham_shape, dtype=self.t2[0].dtype)
        context = MomentumContractions(self, eris, kshift)
        ia, ai, doubles = self._momentum_maps(kshift)
        for ki, kl in product(range(self.nkpts), repeat=2):
            momenta = dict(I=ki, D=ia[ki], L=kl, A=ia[kl])
            for term in kadc_ee_terms.SINGLES_3.get('ss', ()):
                output = term[1].split('->')[1]
                if len(output) == 2 and ki != kl:
                    continue
                signs = {'IDLA': (1, -1, -1, 1),
                         'IL': (1, -1), 'DA': (-1, 1)}[output]
                fixed = {x: momenta[x] for x in output}
                value = context.contract(term, fixed, signs)
                if value is None:
                    continue
                if output == 'IL':
                    value = einsum('IL,DA->IDLA', value, np.eye(self.nvir))
                elif output == 'DA':
                    value = einsum('IL,DA->IDLA', np.eye(self.nocc), value)
                M_ph[ki, kl] += value

        return M_ph

    def get_diag(self, kshift=0, M_ab=None, eris=None):
        r"""Build the diagonal used to precondition EE Davidson residuals.

        Singles use the real diagonal of the effective Hamiltonian M_11.
        Doubles use zeroth-order energy differences,
        :math:`\epsilon_a+\epsilon_b-\epsilon_i-\epsilon_j`, with the fourth k
        point fixed by the selected momentum sector. ADC(2)-x and ADC(3)
        doubles interaction corrections remain in matvec; this preconditioner
        omits them and is therefore an approximate diagonal.

        Args:
            kshift (int): Hole-minus-particle momentum sector, default zero.
            M_ab (ndarray, optional): Precomputed singles matrix, shape
                (n_singles, n_singles). With None, call get_imds().
            eris (_ChemistsERIs, optional): Integrals used if M_ab is absent.
                With both absent, get_imds() transforms the integrals.

        Returns:
            ndarray: Diagonal of length vector_size(), ordered as flattened
            singles followed by doubles. The dtype accommodates M_ab and t2[0],
            although the energy entries are real.

        Examples:
            Create the two-k-point helium reference below; lattice vectors
            are in Angstrom and energies are in Hartree:

            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> with mykradcee.ao2mo() as eris:
            ...     M = mykradcee.get_imds(eris, kshift=1)
            ...     diagonal = mykradcee.get_diag(1, M, eris)
            >>> M.shape, diagonal.shape
            ((2, 2), (10,))
            >>> bool(np.allclose(diagonal[:mykradcee.n_singles], M.diagonal()))
            True
        """
        self.method = check_implemented(self.method)
        if M_ab is None:
            M_ab = self.get_imds(eris, kshift)
        dtype = np.result_type(self.t2[0], M_ab)
        diagonal = np.empty(self.vector_size(), dtype=dtype)
        diagonal[:self.n_singles] = np.diagonal(M_ab).real
        diagonal[self.n_singles:] = self._doubles_diagonal(kshift).ravel()
        return diagonal

    def matvec(self, kshift, M_ab=None, eris=None):
        r"""Prepare a callable applying the EE effective Hamiltonian at one
        momentum.

        The action evaluates :math:`\sigma=M(k)r` through the selected ADC
        order,
        using full singles/doubles vectors without constructing a dense ADC or
        supercell matrix. A missing DF ovvv tensor is reconstructed once when
        preparing the closure; higher-order table contractions reuse a bounded
        MomentumContractions cache across trial vectors.

        Args:
            kshift (int): Hole-minus-particle momentum sector.
            M_ab (ndarray, optional): Precomputed singles matrix, shape
                (n_singles, n_singles). None builds it with get_imds(eris,
                kshift).
            eris (_ChemistsERIs, optional): Active integrals or DF factors.
                With
                None, transform them using ao2mo().

        Returns:
            callable: Function accepting a flat vector of length vector_size()
            and returning a new sigma vector of the same length. Integral
            scratch
            files supplied by the caller must remain open while it is used.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> with mykradcee.ao2mo() as eris:
            ...     action = mykradcee.matvec(0, eris=eris)
            ...     sigma = action(np.ones(mykradcee.vector_size()))
            >>> sigma.shape
            (10,)
        """

        self.method = check_implemented(self.method)

        nkpts, nocc, nvir = self.nkpts, self.nocc, self.nvir
        kconserv = self.khelper.kconserv
        f_singles = s_doubles = self.n_singles
        s_singles = 0
        f_doubles = self.vector_size()
        if eris is None:
            eris = self.ao2mo()
        doubles_diagonal = self._doubles_diagonal(kshift)

        v_ovoo = eris.ovoo
        if eris.ovvv is None:
            shape = (nkpts,) * 3 + (nocc,) + (nvir,) * 3
            v_ovvv = np.empty(shape, dtype=eris.Lov.dtype)
            for kp, kq, kr in kpts_helper.loop_kkk(nkpts):
                ks = kconserv[kp, kq, kr]
                v_ovvv[kp, kq, kr] = einsum(
                    'Lpq,Lrs->pqrs',
                    eris.Lov[kp, kq], eris.Lvv[kr, ks]) / nkpts
        else:
            v_ovvv = eris.ovvv

        kshift_ia, kshift_ai, kconserv_r2 = self._momentum_maps(kshift)

        if M_ab is None:
            M_ab = self.get_imds(eris, kshift)

        context = MomentumContractions(self, eris, kshift)

        # Calculate sigma vector
        def sigma_(r):
            r"""Apply the captured EE Hamiltonian to one trial vector.

            Args:
                r (ndarray): Length-vector_size() singles-then-doubles response
                    vector, with both blocks in C-flattened order. Input is not
                    modified.

            Returns:
                ndarray: Sigma vector in the same layout, including the singles
                matrix, zeroth-order doubles energies, coupling terms and the
                ADC(2)-x/ADC(3) corrections selected by the enclosing solver.
            """
            cput0 = (time.process_time(), time.time())
            log = logger.Logger(self.stdout, self.verbose)

            r1 = r[s_singles:f_singles].reshape(self.singles_shape)
            r2 = r[s_doubles:f_doubles].reshape(self.doubles_shape)

            s1 = (M_ab @ r[s_singles:f_singles]).reshape(self.singles_shape)
            s2 = np.empty(
                self.doubles_shape, dtype=np.result_type(self.t2[0], M_ab, r))

            # doubles rows: holes I, J at kI, kJ; particles C, D at kC, kD
            for kI, kJ, kC in kpts_helper.loop_kkk(nkpts):
                kD = kconserv_r2[kI, kC, kJ]

                s2[kI, kJ, kC] = doubles_diagonal[kI, kJ, kC] * r2[kI, kJ, kC]

                s2[kI, kJ, kC] += einsum(
                    'Ia,JDaC->IJCD',
                    r1[kI], v_ovvv[kJ, kD, kshift_ia[kI]].conj())
                s2[kI, kJ, kC] += einsum(
                    'Ja,ICaD->IJCD',
                    r1[kJ], v_ovvv[kI, kC, kshift_ia[kJ]].conj())
                s2[kI, kJ, kC] -= einsum(
                    'iC,JDIi->IJCD',
                    r1[kshift_ai[kC]], v_ovoo[kJ, kD, kI].conj())
                s2[kI, kJ, kC] -= einsum(
                    'iD,ICJi->IJCD',
                    r1[kshift_ai[kD]], v_ovoo[kI, kC, kJ].conj())

            # singles rows: hole I at kI, particle D at kD
            for kI in range(nkpts):
                kD = kshift_ia[kI]
                for ki in range(nkpts):
                    for ka in range(nkpts):
                        kb = kconserv_r2[kI, ka, ki]
                        s1[kI] -= einsum(
                            'Iiab,iaDb->ID',
                            r2[kI, ki, ka], v_ovvv[ki, ka, kD])
                        s1[kI] += 2 * einsum(
                            'Iiab,ibDa->ID',
                            r2[kI, ki, ka], v_ovvv[ki, kb, kD])
                    for kj in range(nkpts):
                        ka = kconserv_r2[ki, kD, kj]
                        s1[kI] -= 2 * einsum(
                            'ijDa,jaiI->ID',
                            r2[ki, kj, kD], v_ovoo[kj, ka, ki])
                        s1[kI] += einsum(
                            'ijDa,iajI->ID',
                            r2[ki, kj, kD], v_ovoo[ki, ka, kj])

            if self.method in ('adc(2)-x', 'adc(3)'):
                self._add_sigma_terms(
                    s1, s2, r1, r2, context, kadc_ee_terms.SIGMA_2X)
            if self.method == 'adc(3)':
                self._add_sigma_terms(
                    s1, s2, r1, r2, context, kadc_ee_terms.SIGMA_3)
            s = np.hstack((s1.reshape(-1), s2.reshape(-1)))

            log.timer_debug1('completed sigma vector calculation', *cput0)

            return s
        return sigma_

    def get_trans_moments(self, kshift, U=None):
        r"""Evaluate spin-free EE transition densities, flattened by root.

        Instead of constructing the full transition-operator matrix, contract
        the
        normalized response amplitudes with the ADC-order transition terms.
        Each
        MO pair connects hole momentum k to particle momentum ia[k] in this
        sector.
        The resulting T is the flattened density X for these particular roots.

        Args:
            kshift (int): Hole-minus-particle momentum sector.
            U (ndarray, optional): Eigenvector columns of shape
                (vector_size(), nroots). A normalized copy is used; U is
                unchanged.
                With None, read densities cached by an earlier property-enabled
                kernel run for this sector.

        Returns:
            ndarray: Complex/real transition densities of shape
            (nroots, nkpts*nmo*nmo), in k,p,q C order. The returned array is a
            view
            of the cached density tensor; treat it as read-only when reusing
            caches.
            Higher-order terms follow method and approx_trans_moments.

        Raises:
            ValueError: U is absent and a saved solver run/cached sector is
            missing.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf).run(0)
            >>> excitation_results = mykradcee.kernel(2, kptlist=[0])
            >>> mykradcee.get_trans_moments(0).shape
            (2, 8)
        """
        if U is None:
            if self.U is None or kshift not in self._transition_cache:
                raise ValueError('Supply eigenvectors or run kernel first.')
            return self._transition_cache[kshift].reshape(
                self._transition_cache[kshift].shape[0], -1)
        U = self.renormalize_eigenvectors(kshift, U, U.shape[1])
        context = MomentumContractions(self, kshift=kshift)
        tables = [kadc_ee_terms.TRANSITION_2]
        if self.method in ('adc(2)-x', 'adc(3)'):
            tables.append(kadc_ee_terms.TRANSITION_2X)
        if self.method == 'adc(3)':
            tables.append(kadc_ee_terms.TRANSITION_3)
        densities = np.zeros(
            (U.shape[1], self.nkpts, self.nmo, self.nmo),
            dtype=np.result_type(U, self.t2[0]))
        ia, ai, doubles = self._momentum_maps(kshift)
        occupied, virtual = slice(None, self.nocc), slice(self.nocc, None)
        for root in range(U.shape[1]):
            context.data.update(
                r1=U[:self.n_singles, root].reshape(self.singles_shape),
                r2=U[self.n_singles:, root].reshape(self.doubles_shape))
            for table in tables:
                for target, terms in table.items():
                    # Targets retain the molecular reference's block names.
                    rows = (virtual if target.startswith(
                        'TY[nocc:') else occupied)
                    cols = virtual if target.endswith('nocc:]') else occupied
                    for kpt in range(self.nkpts):
                        for term in terms:
                            output = term[1].split('->')[1]
                            fixed = dict(zip(output, (kpt, ia[kpt])))
                            value = context.contract(term, fixed, (1, -1))
                            if value is not None:
                                densities[root, kpt, rows, cols] += value
        self._transition_cache[kshift] = densities
        return densities.reshape(U.shape[1], -1)

    def get_density_operator(self, kshift):
        r"""Transform the Fourier density operator exp[-i(q+G).r] to active
        MOs.

        For q=-kpts[kshift] and G=density_gv, the AO-pair Fourier transform is
        sandwiched between the coefficients at k and ia[k]. G is Cartesian and
        uses inverse Bohr. At q+G=0 the operator is total particle number, so
        its
        inelastic response vanishes up to the approximation/numerical accuracy.

        Args:
            kshift (int): Sector fixing physical q and the particle momentum
                map.

        Returns:
            ndarray: Complex operator matrices of shape (nkpts, nmo, nmo), with
            bra MOs at k and ket MOs at ia[k]. No 1/nkpts strength
            normalization
            is included here; get_properties() applies it after contraction.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADCEE
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcee = kadc_rhf.KRADCEE(kmf)
            >>> mykradcee.get_density_operator(0).shape
            (2, 2, 2)
        """
        from pyscf.pbc.df import ft_ao
        coeff, e_occ, e_vir = (self.padded_coeff,
            self.e_occ, self.e_vir)
        ia, ai, doubles = self._momentum_maps(kshift)
        q = -self.kpts[kshift]
        operator = np.empty(
            (self.nkpts, self.nmo, self.nmo), dtype=np.complex128)
        for ki, ka in enumerate(ia):
            ao = ft_ao.ft_aopair(
                self.cell,
                self.density_gv.reshape(1, 3),
                kpti_kptj=np.array([self.kpts[ki], self.kpts[ka]]),
                q=q)[0]
            operator[ki] = coeff[ki].conj().T @ ao @ coeff[ka]
        return operator

    def _momentum_maps(self, kshift):
        r"""Construct or reuse the neutral-sector particle-momentum lookup
        tables.

        Args:
            kshift (int): Hole-minus-particle sector index, in [0, nkpts).

        Returns:
            tuple[ndarray, ndarray, ndarray]: ia, its inverse ai, and the
            doubles
            lookup. ia[ki] gives ka with ki-ka=kshift; ai[ka] gives the hole
            index. doubles[ki, ka, kj] gives kb with ki-ka+kj-kb=kshift, modulo
            a reciprocal vector. Shapes are (nkpts,), (nkpts,) and (nkpts,)*3.
            Maps are cached per sector until kernel() clears them.

        Raises:
            ValueError: The supplied mesh is not closed under the required
                momentum combinations within the lattice-coordinate tolerance.
        """
        if kshift not in self._ee_momentum:
            ia = self.khelper.kconserv[:, kshift, 0].copy()
            ai = np.empty_like(ia)
            ai[ia] = np.arange(self.nkpts)
            lattice = self.cell.lattice_vectors() / (2 * np.pi)
            kpts = self.kpts
            momenta = (kpts[:, None, None, :] - kpts[:, None, :]
                       + kpts - kpts[kshift])
            doubles = np.full((self.nkpts,) * 3, -1, dtype=int)
            for index, kpt in enumerate(kpts):
                scaled = einsum('wx,klmx->wklm', lattice, momenta - kpt)
                mask = np.sum(abs(scaled - np.rint(scaled)), axis=0) < 1e-9
                doubles[mask] = index
            if np.any(doubles < 0):
                raise ValueError('K-point mesh is not closed under momentum '
                                 'conservation for this excitation shift.')
            self._ee_momentum[kshift] = ia, ai, doubles
        return self._ee_momentum[kshift]

    def _doubles_diagonal(self, kshift):
        r"""Build the zeroth-order EE doubles energy differences for one
        sector.

        Args:
            kshift (int): Sector used to determine the unstored fourth
                momentum.

        Returns:
            ndarray: Shape doubles_shape, ordered as (k_i,k_j,k_a,i,j,a,b),
            with
            entries epsilon_a(k_a)+epsilon_b(k_b)-epsilon_i(k_i)-epsilon_j(k_j)
            in Hartree. The result includes padded slots and uses the real
            energy
            dtype. No integral or interaction correction is added here.
        """
        e_occ, e_vir = self.e_occ, self.e_vir
        ia, ai, kconserv = self._momentum_maps(kshift)
        diagonal = np.empty(
            self.doubles_shape, dtype=np.result_type(e_occ, e_vir))
        for ki, kj, ka in kpts_helper.loop_kkk(self.nkpts):
            kb = kconserv[ki, ka, kj]
            occupied = e_occ[ki][:, None] + e_occ[kj]
            virtual = e_vir[ka][:, None] + e_vir[kb]
            diagonal[ki, kj, ka] = (virtual[None, None, :, :]
                                     - occupied[:, :, None, None])
        return diagonal

    def _add_sigma_terms(self, s1, s2, r1, r2, context, table):
        r"""Accumulate table-driven higher-order singles and doubles matrix
        actions.

        Args:
            s1, s2 (ndarray): Writable sigma tensors shaped singles_shape and
                doubles_shape. Contributions are added in place.
            r1, r2 (ndarray): Response blocks in those same shapes. Stored in
                context.data so each contraction uses the current trial vector.
            context (MomentumContractions): Reference/integral context at the
                selected sector, with reusable ground-only contraction caches.
            table (dict): Lists of term triples under the s1/s2 output keys.
                See kadc_ee_terms for coefficient and orbital-label
                conventions.

        Returns:
            None: Update s1/s2 and the response entries of context.data. Each
            term
            sums only the momentum assignments allowed by the conserved sector.
        """
        context.data.update(r1=r1, r2=r2)
        ia, ai, doubles = self._momentum_maps(context.kshift)
        for ki in range(self.nkpts):
            for term in table.get('s1', ()):
                value = context.contract(term, {'I': ki, 'D': ia[ki]}, (1, -1))
                if value is not None:
                    s1[ki] += value
        for ki, kj, kc in kpts_helper.loop_kkk(self.nkpts):
            fixed = dict(I=ki, J=kj, C=kc, D=doubles[ki, kc, kj])
            # Contraction labels may differ in case; use the term's output.
            for term in table.get('s2', ()):
                output = term[1].split('->')[1]
                momenta = dict(zip(output, fixed.values()))
                value = context.contract(term, momenta, (1, 1, -1, -1))
                if value is not None:
                    s2[ki, kj, kc] += value

    @property
    def singles_shape(self):
        r"""Describe the neutral singles tensor before C-order flattening.

        Returns:
            tuple[int]: (nkpts, nocc, nvir), with axes (k_i, i, a). The
            selected
            kshift fixes k_a from k_i-k_a=kshift; it is not another independent
            tensor axis. See _momentum_maps() for the explicit lookup.
        """
        return (self.nkpts, self.nocc, self.nvir)

    @property
    def doubles_shape(self):
        r"""Describe the neutral doubles tensor before C-order flattening.

        Returns:
            tuple[int]: (nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir), with
            axes
            (k_i, k_j, k_a, i, j, a, b). The k_b momentum follows conservation
            in the selected sector. No orbital-pair permutation symmetry is
            used
            to compress this tensor.
        """
        return (self.nkpts,) * 3 + (self.nocc,) * 2 + (self.nvir,) * 2

    @property
    def _singles_ham_shape(self):
        r"""Describe the orbital tensor used while assembling the singles M
        matrix.

        Returns:
            tuple[int]: (nkpts, nkpts, nocc, nvir, nocc, nvir), whose axes are
            (k_i, k_l, i, a, l, b). get_imds() groups (k_i,i,a) and (k_l,l,b)
            into its row and column indices only after all order corrections.
        """
        return (self.nkpts, self.nkpts, self.nocc, self.nvir,
                self.nocc, self.nvir)


# Historical reference-only names; choose a concrete class for excitations.
KRADC = RADC = KRADCBase
