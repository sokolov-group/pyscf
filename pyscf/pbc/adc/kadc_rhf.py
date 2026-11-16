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

The public driver is KRADC. Ground-state correlation energies are in
Hartree per primitive cell; excitation roots are in Hartree and ordered
by the selected kptlist. Active coefficients and amplitudes use KMP2
padding to accommodate unequal orbital counts across the k mesh.

Each numerical example includes imports and builds its own two-k-point
helium reference ``kmf``. Ground amplitudes are
stored on the driver rather than returned by ground_state_energy().
"""
from types import SimpleNamespace
import tempfile
import time
import warnings

import h5py
import numpy as np

from pyscf import __config__, lib
from pyscf.adc import radc
from pyscf.data.nist import HARTREE2EV
from pyscf.lib import logger
from pyscf.lib.parameters import LOOSE_ZERO_TOL
from pyscf.pbc import df, scf, tools
from pyscf.pbc.adc import kadc_ao2mo
from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs
from pyscf.pbc.adc.lib import check_implemented, einsum, log_table, validate
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
    already be present for positive nroots; the KRADC driver prepares them.

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
        >>> from pyscf.pbc.adc.kadc_rhf import KRADC, kernel
        >>> from pyscf.pbc.adc.kadc_rhf_ee import KRADCEE
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
        >>> _ = kmf.kernel()
        >>> mykradc = KRADC(kmf)
        >>> _ = mykradc.ground_state_energy()
        >>> mykradcee = KRADCEE(mykradc)
        >>> _ = kernel(mykradcee, nroots=2, kptlist=[0])
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
        mykradc = getattr(adc, '_adc', adc)
        energy = KRADC.ground_state_energy(mykradc, eris)
        adc.t1, adc.t2, adc.e_corr = mykradc.t1, mykradc.t2, energy
        mykradc._clear_excitations()
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
    match adc.type_:
        case 'ee':
            X = np.zeros((*root_shape, nkpts, nmo, nmo), dtype=dtype)
            imds = None  # Neutral intermediates depend on kshift.
            initial_guess = adc._neutral_init_guess
        case 'ip' | 'ea':
            X = np.zeros((len(kptlist), nmo, nroots), dtype=dtype)
            imds = adc.get_imds(eris)
            initial_guess = adc._charged_init_guess
        case _:
            raise NotImplementedError(adc.type_)
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
        if adc.type_ == 'ee':
            U = adc.renormalize_eigenvectors(kshift, U, nroots)
            evecs[k] = U.T
        if adc.compute_properties:
            P[k], X[k] = adc.get_properties(kshift, U, nroots)

    adc.E, adc.U, adc.P, adc.X, adc.converged = evals, evecs, P, X, conv
    if adc.type_ == 'ee':
        adc.T = X.reshape(*root_shape, -1) if adc.compute_properties else None
    nfalse = conv.size - np.count_nonzero(conv)
    if nfalse:
        logger.warn(
            adc, f'Davidson iterations for {nfalse} root(s) did not converge.')
    data = {'Method': [], 'k': [], 'Root': [], 'Energy (Eh)': [],
            'Energy (eV)': []}
    strength_label = 'Density' if adc.type_ == 'ee' else 'Spec factor'
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
        adc (KRADC): Reference with previously computed t1 and t2.
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
        >>> from pyscf.pbc.adc.kadc_rhf import KRADC
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
        >>> _ = kmf.kernel()
        >>> mykradc = KRADC(kmf)
        >>> _ = mykradc.ground_state_energy()
        >>> density = mykradc.make_ref_rdm1()
        >>> density.shape
        (2, 2, 2)
        >>> round(float(np.trace(density, axis1=1, axis2=2).real.mean()), 6)
        2.0
        >>> mykradc.make_ref_rdm1(ao_repr=True).shape
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
        myadc (KRADC): Object supplying nkpts, mo_occ and frozen. Frozen
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
        myadc (KRADC): Object supplying the attributes used by mo_splitter
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


class KRADC(radc.RADC):
    r"""Restricted ADC for a periodic, k-sampled Hartree-Fock reference.

    Args:
        mf: Converged KRHF object.
        frozen: Frozen orbital count or orbital index lists, as in KMP2.
        mo_coeff, mo_occ, mo_energy: Optional orbitals replacing the SCF data.

    Attributes:
        method (str): ``adc(2)``, ``adc(2)-x`` or ``adc(3)``.
        type_ (str): ``ip`` (default), ``ea`` or singlet ``ee``.
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

    Examples:
        Two k points for a helium crystal with two s basis functions per
        atom. The lattice length is in Angstrom (PySCF's default unit):

        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc.kadc_rhf import KRADC
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
        >>> _ = kmf.kernel()
        >>> mykradc = KRADC(kmf)
        >>> ground_state_energy = mykradc.ground_state_energy()
        >>> round(ground_state_energy, 8)
        -0.00416766
        >>> amp_singles, amp_doubles = mykradc.t1, mykradc.t2
        >>> amp_doubles[0].shape
        (2, 2, 2, 1, 1, 1, 1)
        >>> mykradc.type_ = 'ee'
        >>> _ = mykradc.kernel(nroots=2, kptlist=[0])
        >>> excited_energies = mykradc.E
        >>> excited_energies.round(6).tolist()
        [[1.756312, 2.102047]]
        >>> mykradc.U.shape, mykradc.P.shape, mykradc.X.shape
        ((1, 2, 10), (1, 2), (1, 2, 2, 2, 2))
    """
    _keys = (
        radc.RADC._keys
        | {
            'kpts',
            'khelper',
            'exxdiv',
            'cell',
            'nkop_chk',
            'kop_npick',
            'chunk_size',
            'keep_exxdiv',
            'naux',
            'if_heri_eris',
            'eris',
            'if_naf',
            'thresh_naf',
            'if_corr',
            'integral_backend',
            'density_gv',
            'type_',
            'T',
            'U',
            'X',
            'P',
            'converged',
        }
    )

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
        mo_energy=None):

        r"""Initialize a restricted periodic ADC driver without running a
        calculation.

        Args:
            mf (KRHF): K-point restricted Hartree-Fock reference with cell,
                kpts,
                orbitals and an AO integral/DF transformation object.
            frozen (int or sequence, optional): KMP2 frozen-space
                specification:
                occupied count, a shared list of MO indices, or one list per k.
            mo_coeff (sequence[ndarray], optional): Coefficients C_k, each of
                shape
                (nao, nmo_k). None shares mf.mo_coeff.
            mo_occ (sequence[ndarray], optional): Occupations per k, with
                occupied
                entries positive and virtual entries zero. None shares
                mf.mo_occ.
            mo_energy (sequence[ndarray], optional): Orbital energies in
                Hartree,
                matching the coefficient columns. None shares mf.mo_energy.

        Returns:
            None: Set ADC(2)/IP defaults, solver tolerances, memory and
            property
            settings. Energies, amplitudes and excitation results are initially
            None. When using the SCF coefficients, invalid-energy FNO columns
            are
            removed together with their occupations and energies.

        See the class docstring for a complete executable calculation.
        """
        assert (isinstance(mf, scf.khf.KSCF))

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_energy is None:
            mo_energy = mf.mo_energy

        if (mo_coeff is mf.mo_coeff and np.any(
            np.asarray(mo_energy) == INVALID_ORBITAL_ENERGY)):
            valid = np.asarray(mo_energy) != INVALID_ORBITAL_ENERGY
            mo_coeff = [np.asarray(c)[:, v] for c, v in zip(mo_coeff, valid)]
            mo_occ = [np.asarray(o)[v] for o, v in zip(mo_occ, valid)]
            mo_energy = [e[v] for e, v in zip(mo_energy, valid)]

        self._scf = mf
        self.kpts = self._scf.kpts
        self.exxdiv = self._scf.exxdiv
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.method = "adc(2)"
        self.type_ = "ip"

        self.max_space = getattr(__config__, 'adc_kadc_RADC_max_space', 12)
        self.max_cycle = getattr(__config__, 'adc_kadc_RADC_max_cycle', 50)
        self.conv_tol = getattr(__config__, 'adc_kadc_RADC_conv_tol', 1e-07)
        self.tol_residual = getattr(
            __config__, 'adc_kadc_RADC_tol_res', 0.0001)
        self.scf_energy = mf.e_tot

        self.khelper = kpts_helper.KptsHelper(mf.cell, mf.kpts)
        self.cell = self._scf.cell
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mo_energy = mo_energy
        self.frozen = frozen
        self.compute_properties = True
        self.approx_trans_moments = True

        self._nocc = None
        self._nmo = None
        self._nvir = None
        self.nkop_chk = False
        self.kop_npick = False

        self.t1 = None
        self.t2 = None
        self.e_corr = None
        self._chunk_size = None
        self._chunk_size_manual = False
        self.imds = SimpleNamespace()

        self.keep_exxdiv = False
        self.naux = None
        self.if_heri_eris = False
        self.if_naf = False
        self.if_corr = True
        self.thresh_naf = 1e-2
        self.eris = None
        self._padded_orbitals = None
        self.integral_backend = 'auto'
        self.density_gv = np.zeros(3)
        self.T = None # Transition moments.
        self.U = None # Spectroscopic amplitudes.
        self.X = None
        self.P = None # Spectroscopic factors.

    make_ref_rdm1 = make_ref_rdm1

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> with mykradc.ao2mo('density_fit') as eris:
            ...     print(eris.ovov.shape)
            (2, 2, 2, 1, 1, 1, 1)
        """
        return kadc_ao2mo.ao2mo(self, backend)

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> print(mykradc.chunk_size)
            1
            >>> mykradc.chunk_size = 2
            >>> mykradc.chunk_size
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

    @property
    def method_type(self):
        r"""Read the earlier name for the excitation-sector setting type_.

        Returns:
            str: The current type_ value, normally 'ip', 'ea' or 'ee'. The
            alias
            is also used by inherited molecular PySCF utilities. Approximation
            order is set separately through method.
        """
        return self.type_

    @method_type.setter
    def method_type(self, value):
        r"""Assign the excitation sector through the earlier method_type name.

        Args:
            value (str): Sector string stored directly as type_. Case is
                normalized during driver validation, rather than in this
                setter.

        Returns:
            None: Update the same setting used by the IP/EA/EE solver dispatch.
        """
        self.type_ = value

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.nkpts
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> int(mykradc.nocc)
            1
        """
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> int(mykradc.nmo)
            2
        """
        return self.get_nmo()

    get_nocc = get_nocc
    get_nmo = get_nmo

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> int(mykradc.nvir)
            1
        """
        return self.nmo - self.nocc

    @property
    def n_singles(self):
        r"""Return the one-hole or one-particle charged-vector block length.

        Returns:
            int or numpy.integer: nocc for IP, otherwise nvir for the charged
            EA convention.
            KRADCEE overrides this property to include the full
            k-summed neutral
            particle-hole basis; the base value should not be used for EE
            vectors.
        """
        return self.nocc if self.type_ == 'ip' else self.nvir

    @property
    def doubles_shape(self):
        r"""Describe the charged doubles tensor before C-order flattening.

        Returns:
            tuple[int]: (nkpts, nkpts, nvir, nocc, nocc) for IP, whose
            independent
            momenta are (k_a, k_i); otherwise (nkpts, nkpts, nocc, nvir, nvir)
            for EA, with (k_i, k_a). The sector fixes the remaining momentum.
            KRADCEE supplies a separate neutral singles/doubles layout.
        """
        if self.type_ == 'ip':
            return (self.nkpts, self.nkpts, self.nvir, self.nocc, self.nocc)
        return (self.nkpts, self.nkpts, self.nocc, self.nvir, self.nvir)

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
        if self._padded_orbitals is None:
            coeff, energy = _add_padding(self, self.mo_coeff, self.mo_energy)
            energy = np.asarray(energy)
            self._padded_orbitals = (coeff, energy[:, :self.nocc],
                                    energy[:, self.nocc:])
        return self._padded_orbitals

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.padded_coeff.shape
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.e_occ.shape
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.e_vir.shape
            (2, 1)
        """
        return self._pad_orbitals()[2]

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

    def _init_from_ground_state(self, adc):
        r"""Share reference data with a sector solver without transferring file
        ownership.

        Args:
            adc (KRADC): Ground-state driver containing orbitals, amplitudes
                and
                settings to share. The solver keeps adc alive through _adc.

        Returns:
            None: Copy reference attributes except amplitude-file owners and
            the
            earlier excitation solver. Clear the local padded-orbital cache and
            register shared attributes in _keys. Amplitude arrays remain
            shared,
            while the original driver owns and closes their scratch files.
        """
        self.__dict__.update({key: value for key, value in adc.__dict__.items()
                              if key not in ('_amp_h5file', '_amp_tmpfile',
                                             '_adc_es')})
        self._adc = adc
        self._padded_orbitals = None
        self._keys = set(self.__dict__)

    def _charged_init_guess(self, nroots=1, diag=None, ascending=True,
                            type_=None, ini=None, kshift=None):
        r"""Build IP/EA unit guesses or validate supplied root vectors.

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> from pyscf.pbc.adc.kadc_rhf_ip import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradcip = KRADCIP(mykradc)
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
                    singles, doubles, kshift, const=0.0)
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
            list[ndarray]: One flat vector per requested root. IP/EA use
            _charged_init_guess and mask inactive components; EE uses
            _neutral_init_guess. Invalid shapes and zero norms raise
            ValueError.

        A DeprecationWarning is emitted. This entry point is intended for
        sector
        solvers; the kernel calls their protected helpers directly.

        Examples:
            >>> import numpy as np
            >>> import warnings
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> from pyscf.pbc.adc.kadc_rhf_ip import KRADCIP
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradcip = KRADCIP(mykradc)
            >>> with warnings.catch_warnings(record=True) as caught:
            ...     warnings.simplefilter('always', DeprecationWarning)
            ...     guesses = mykradcip.get_init_guess(1, diag=np.arange(5.0))
            >>> np.shape(guesses), caught[0].category.__name__
            ((1, 5), 'DeprecationWarning')
        """
        warnings.warn('get_init_guess is deprecated; use the sector-specific '
                      '_charged_init_guess or _neutral_init_guess helper.',
                      DeprecationWarning, stacklevel=2)
        if 'type' in legacy:
            if type_ is not None:
                raise TypeError('Pass only one of type_ and the legacy type.')
            type_ = legacy.pop('type')
        if legacy:
            names = ', '.join(sorted(legacy))
            raise TypeError(f'Unexpected initial-guess keyword(s): {names}')
        match self.type_:
            case 'ip' | 'ea':
                initial_guess = self._charged_init_guess
            case 'ee':
                initial_guess = self._neutral_init_guess
            case _:
                raise NotImplementedError(self.type_)
        return initial_guess(nroots, diag, ascending, type_, ini, kshift)

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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> round(mykradc.ground_state_energy(), 8)
            -0.00416766
            >>> mykradc.t2[0].shape
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> round(mykradc.kernel(nroots=0), 8)
            -0.00416766
            >>> mykradc.type_ = 'ee'
            >>> _ = mykradc.kernel(nroots=2, kptlist=[0])
            >>> mykradc.E.round(6).tolist()
            [[1.756312, 2.102047]]
        """
        if not isinstance(nroots, (int, np.integer)):
            raise TypeError('nroots must be an integer.')
        started = logger.process_clock(), logger.perf_counter()
        eris = self._compute_reference(eris)
        if nroots < 1:
            self._clear_excitations()
            logger.info(
                self,
                f'nroots = {nroots}; only the ground state was calculated.')
            logger.timer(self, 'ADC reference only', *started)
            return self.e_corr
        match self.type_:
            case 'ea':
                solve = self.ea_adc
            case 'ip':
                solve = self.ip_adc
            case 'ee':
                solve = self.ee_adc
            case _:
                raise NotImplementedError(self.type_)
        e_exc, v_exc, strengths, x, mykradc_es = solve(
            nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)
        self._adc_es = mykradc_es
        self.E, self.U, self.P, self.X = e_exc, v_exc, strengths, x
        self.converged = mykradc_es.converged
        self.T = mykradc_es.T if self.type_ == 'ee' else None
        logger.timer(self, 'complete ADC kernel', *started)
        return e_exc, v_exc, strengths, x

    def _close_amp_scratch(self):
        # close the HDF5 scratch file backing the on-disk amplitudes
        r"""Release scratch resources owning the disk-backed amplitude tensors.

        Returns:
            None: Attempt to close both the HDF5 file and its underlying
            temporary
            file. Ignore already-closed or unavailable handles so repeated
            cleanup
            and object finalization are safe. Existing dataset views must not
            be
            accessed after their owner is closed.
        """
        for attr in ('_amp_h5file', '_amp_tmpfile'):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except (AttributeError, OSError, ValueError):
                    pass

    def __del__(self):
        r"""Release owned amplitude scratch files during object finalization.

        Returns:
            None: Delegate to _close_amp_scratch(). Sector solvers do not copy
            amplitude-owner handles, so their destruction does not close the
            reference driver's shared amplitude datasets.
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.type_ = 'ip'
            >>> _ = mykradc.ground_state_energy()
            >>> _ = mykradc.kernel(1, kptlist=[0])
            >>> energies = mykradc.E
            >>> energies.round(6).tolist()
            [[0.371589]]
        """
        from pyscf.pbc.adc import kadc_rhf_ip
        mykradcip = kadc_rhf_ip.KRADCIP(self)
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.type_ = 'ea'
            >>> _ = mykradc.ground_state_energy()
            >>> _ = mykradc.kernel(1, kptlist=[0])
            >>> energies = mykradc.E
            >>> energies.round(6).tolist()
            [[1.558245]]
        """
        from pyscf.pbc.adc import kadc_rhf_ea
        mykradcea = kadc_rhf_ea.KRADCEA(self)
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.type_ = 'ee'
            >>> _ = mykradc.ground_state_energy()
            >>> _ = mykradc.kernel(1, kptlist=[0])
            >>> energies = mykradc.E
            >>> energies.round(6).tolist()
            [[1.756312]]
        """
        from pyscf.pbc.adc import kadc_rhf_ee
        mykradcee = kadc_rhf_ee.KRADCEE(self)
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
            KRADC: This instance, for chained configuration calls.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> mykradc.density_fit(with_df=kmf.with_df) is mykradc
            True
            >>> mykradc.with_df is kmf.with_df
            True
        """
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

    def make_rdm1(self, kptlist, root=None, K_idx=None, with_frozen=True,
        ao_repr=False, if_ss=False):
        r"""Build charged-state one-particle densities after an IP/EA run.

        Args:
            kptlist (sequence[int]): Momentum sectors represented in the
                saved solver eigenvectors, in their original order.
            root (sequence[sequence[int]], optional): Root indices requested
                for each selected sector. With None, include all roots.
            K_idx (sequence[int], optional): Sectors to return; defaults to
                kptlist. Use the same number of roots for each sector.
            with_frozen (bool): Restore frozen MOs and their occupied density.
            ao_repr (bool): Return AO matrices instead of padded active MO
                matrices (or full MO matrices when frozen MOs are restored).
            if_ss (bool): Include an additional reference-density contribution.

        Returns:
            list[ndarray] or ndarray: Densities with axes (sector, root,
            k, p, q). The final two dimensions are nao in AO representation,
            otherwise the corresponding active/full MO count. Neutral
            transition densities are available separately as X.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> _ = mykradc.kernel(1, kptlist=[0])
            >>> density = mykradc.make_rdm1([0], root=[[0]], K_idx=[0])
            >>> np.shape(density)
            (1, 1, 2, 2, 2)
        """
        if K_idx is None:
            K_idx = kptlist

        if root is None:
            nroots = [range(self._adc_es.U.shape[1]) for _ in K_idx]
        else:
            nroots = root

        rdm1 = self._adc_es.make_rdm1(nroots, kptlist, K_idx, if_ss)

        if with_frozen and self.frozen is not None:
            nmo = max(mo_occ_k.size for mo_occ_k in self.mo_occ)
            nocc = np.count_nonzero(self.mo_occ[0] > 0)
            mask = get_frozen_mask(self)
            dm = np.zeros(
                (len(K_idx), len(nroots), self.nkpts, nmo, nmo),
                dtype=np.complex128)
            occ_idx = np.arange(nocc)
            k0_idx = np.arange(len(K_idx))
            k1_idx = np.arange(self.nkpts)
            p_k_idx = padding_k_idx(self, kind='joint')
            s_idx = np.arange(len(nroots))
            K0, S, K1, O = np.meshgrid(
                k0_idx, s_idx, k1_idx, occ_idx, indexing='ij')
            if if_ss:
                dm[K0, S, K1, O, O] = 4
            else:
                dm[K0, S, K1, O, O] = 2
            for ki in k1_idx:
                moidx = np.where(mask[ki])[0]
                for i in range(len(nroots)):
                    for kj in k0_idx:
                        dm[kj][i][ki][moidx[:, None],
                            moidx] = rdm1[kj][i][ki][p_k_idx[ki][:, None],
                            p_k_idx[ki]]
            rdm1 = dm
            if ao_repr:
                mo = []
                for k in range(self.nkpts):
                    mo_k = np.asarray(self.mo_coeff[k])
                    npad = nmo - mo_k.shape[1]
                    if npad > 0:
                        mo_k = np.hstack((mo_k, np.zeros(
                            (mo_k.shape[0], npad), dtype=mo_k.dtype)))
                    mo.append(mo_k)
                mo = np.stack(mo)
                rdm1 = einsum('kpI,KSkIJ,kqJ->KSkpq', mo, rdm1, mo.conj())
        elif ao_repr:
            mo = padded_mo_coeff(self, self.mo_coeff)
            mo_H = [c.conj() for c in mo]
            rdm1 = einsum('kpI,KSkIJ,kqJ->KSkpq', mo, rdm1, mo_H)

        return rdm1


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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> with mykradc.ao2mo() as eris:
            ...     mykradc.amplitudes(eris)
            ...     print(mykradc.t2[0].shape)
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
        tf = tempfile.TemporaryFile()
        f = h5py.File(tf, 'a')
        # Keep scratch ownership until KRADC releases the amplitudes.
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
                 (self.method == 'adc(2)-x' or self.type_ == 'ee'))):
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
            >>> from pyscf.pbc.adc.kadc_rhf import KRADC
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
            >>> _ = kmf.kernel()
            >>> mykradc = KRADC(kmf)
            >>> with mykradc.ao2mo() as eris:
            ...     mykradc.amplitudes(eris)
            ...     energy = mykradc.correlation_energy(eris)
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


# Historical class name for code outside this package.
RADC = KRADC
