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

from types import SimpleNamespace
import tempfile
import time
import warnings

import h5py
import numpy as np

from pyscf import __config__, lib
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
import pyscf.adc.radc


# Note : All integrals are in Chemist's notation except for vvvv
#        Eg.of momentum conservation :
#        Chemist's  oovv(ijab) : ki - kj + ka - kb
#        Amplitudes t2(ijab)  : ki + kj - ka - kba

def kernel(adc, nroots=1, guess=None, eris=None, kptlist=None):
    """Solve one excitation sector; shared by IP, EA and EE solver classes.

    Positive nroots returns (E, U, P, X), with kptlist order on the first
    axis. Nonpositive nroots computes the reference only and returns e_corr.
    """
    if not isinstance(nroots, (int, np.integer)):
        raise TypeError('nroots must be an integer.')
    if nroots < 1:
        reference = getattr(adc, '_adc', adc)
        energy = KRADC.ground_state_energy(reference, eris)
        adc.t1, adc.t2, adc.e_corr = reference.t1, reference.t2, energy
        reference._clear_excitations()
        adc._clear_excitations()
        logger.info(adc, f'nroots = {nroots}; only the ground state '
                    'was calculated.')
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
    if not (kptlist and all(isinstance(k, (int, np.integer))
                           and 0 <= k < nkpts for k in kptlist)):
        raise ValueError('kptlist must contain valid k-point indices.')
    dtype = np.result_type(adc.t2[0])
    shape = len(kptlist), nroots
    evals = np.zeros(shape)
    evecs = np.zeros((*shape, vec_size), dtype=dtype)
    conv = np.zeros(shape, dtype=bool)
    P = np.zeros(shape)
    match adc.type_:
        case 'ee':
            X = np.zeros((*shape, nkpts, nmo, nmo), dtype=dtype)
            imds = None  # Neutral intermediates depend on kshift.
        case 'ip' | 'ea':
            X = np.zeros((len(kptlist), nmo, nroots), dtype=dtype)
            imds = adc.get_imds(eris)
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
                initial = adc.get_init_guess(
                    nroots, diag, ascending=True, type='read',
                    ini=guess_k[k], kshift=kshift)
            case None:
                initial = adc.get_init_guess(
                    nroots, diag, ascending=True, kshift=kshift)
        conv_k, evals_k, evecs_k = lib.linalg_helper.davidson_nosym1(
            lambda xs: [matvec(x) for x in xs], initial, diag,
            nroots=nroots, verbose=log, tol=adc.conv_tol,
            max_cycle=adc.max_cycle, max_space=adc.max_space,
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
        adc.T = X.reshape(*shape, -1) if adc.compute_properties else None
    nfalse = conv.size - np.count_nonzero(conv)
    if nfalse:
        logger.warn(adc, f'Davidson iterations for {nfalse} root(s) '
                    'did not converge.')
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

    adc.method = check_implemented(adc.method)

    t1 = adc.t1
    t2 = adc.t2
    t2_ce = t1[0]
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

        oo[ki] -= 2 * einsum('KkIiab,KkJiab->IJ', t1_oovv_np,
            t1_oovv_np.conj())
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
        OPDM[:, :nocc, nocc:] += t2_ce
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
                    OPDM[ki][:nocc, :nocc] -= 2 * einsum('Iiab,Jiab->IJ',
                        t1_oovv[ki][kj][ka], t2_oovv[ki][kj][ka].conj())
                    OPDM[ki][:nocc, :nocc] += einsum('Iiab,Jiba->IJ', t1_oovv[
                        ki][kj][ka], t2_oovv[ki][kj][kb].conj())
                    OPDM[ki][:nocc, :nocc] -= 2 * einsum('Jiab,Iiab->IJ',
                        t1_oovv[ki][kj][ka].conj(), t2_oovv[ki][kj][ka])
                    OPDM[ki][:nocc, :nocc] += einsum('Jiab,Iiba->IJ', t1_oovv[
                        ki][kj][ka].conj(), t2_oovv[ki][kj][kb])
                    ##### VIR-VIR ###
                    OPDM[ka][nocc:, nocc:] += 2 * einsum('ijBa,ijAa->AB',
                        t1_oovv[ki][kj][ka], t2_oovv[ki][kj][ka].conj())
                    OPDM[ka][nocc:, nocc:] -= einsum('ijBa,ijaA->AB', t1_oovv[
                        ki][kj][ka], t2_oovv[ki][kj][kb].conj())
                    OPDM[ka][nocc:, nocc:] += 2 * einsum('ijAa,ijBa->AB',
                        t1_oovv[ki][kj][ka].conj(), t2_oovv[ki][kj][ka])
                    OPDM[ka][nocc:, nocc:] -= einsum('ijAa,ijaB->AB', t1_oovv[
                        ki][kj][ka].conj(), t2_oovv[ki][kj][kb])

                ka = ki
                kb = kj
                ##### OCC-VIR ### ####
                OPDM[ki][:nocc, nocc:] += einsum('IiAa,ia->IA', t1_oovv[ki][kj]
                                                     [ka], t2_ce[kj].conj())
                OPDM[ki][:nocc, nocc:] -= 1 / 2 * \
                    einsum('IiaA,ia->IA', t1_oovv[ki][kj][kb],
                        t2_ce[kj].conj())
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
        p_k_idx = padding_k_idx(adc, kind="joint")
        dm[k_idx[:, None], occ_idx, occ_idx] = 2
        for k in k_idx:
            moidx = np.where(mask[k])[0]
            dm[k][moidx[:, None], moidx] = OPDM[k][p_k_idx[k][:, None],
                p_k_idx[k]]
        OPDM = dm
        if ao_repr:
            for k in k_idx:
                mo_k = np.asarray(adc.mo_coeff[k])
                npad = nmo - mo_k.shape[1]
                if npad > 0:
                    mo_k = np.hstack((mo_k, np.zeros((mo_k.shape[0], npad),
                        dtype=mo_k.dtype)))
                OPDM[k] = einsum('pI,IJ,qJ->pq', mo_k, OPDM[k], mo_k.conj())

    elif ao_repr:
        mo = padded_mo_coeff(adc, adc.mo_coeff)
        for k in k_idx:
            OPDM[k] = einsum('pI,IJ,qJ->pq', mo[k], OPDM[k], mo[k].conj())

    return OPDM


def mo_splitter(myadc):
    masks = []
    maskact = get_frozen_mask(myadc)
    for kpt in range(myadc.nkpts):
        maskact_k = maskact[kpt]
        maskocc_k = myadc.mo_occ[kpt] > 1e-6
        masks_k = [
            maskocc_k & ~maskact_k,    # frz occ
            maskocc_k & maskact_k,    # act occ
            ~maskocc_k & maskact_k,    # act vir
            ~maskocc_k & ~maskact_k,    # frz vir
        ]
        masks.append(masks_k)
    return masks


def orbital_number_lists(myadc):
    nocc_fr, nocc_ar, nvir_ar, nvir_fr, nremoved = [], [], [], [], []
    for kpt, (frz_occ, act_occ, act_vir,
        frz_vir) in enumerate(mo_splitter(myadc)):
        if getattr(myadc, 'mo_energy', None) is not None:
            valid_k = (np.asarray(myadc.mo_energy[
                kpt]) != INVALID_ORBITAL_ENERGY)
        else:
            valid_k = np.ones(frz_vir.size, dtype=bool)
        nocc_fr.append(np.count_nonzero(frz_occ))
        nocc_ar.append(np.count_nonzero(act_occ))
        nvir_ar.append(np.count_nonzero(act_vir))
        nvir_fr.append(np.count_nonzero(frz_vir & valid_k))
        nremoved.append(np.count_nonzero(~valid_k))
    return (np.array(nocc_fr), np.array(nocc_ar),
            np.array(nvir_ar), np.array(nvir_fr), np.array(nremoved))


class KRADC(pyscf.adc.radc.RADC):
    """Restricted ADC for a periodic, k-sampled Hartree-Fock reference.

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
        integral_backend (str): ``auto``, ``incore``, ``outcore`` or
            ``density_fit``; ``df`` is accepted for compatibility.
        chunk_size (int): Cached orbital chunk size; assigning None restores
            automatic selection, and a positive integer fixes it explicitly.

    Examples:
        >>> kadc = KRADC(kmf)
        >>> energy = kadc.ground_state_energy()
        >>> singles, doubles = kadc.t1, kadc.t2
        >>> kadc.type_ = 'ee'
        >>> energies, U, P, X = kadc.kernel(nroots=4)
    """
    _keys = (
        pyscf.adc.radc.RADC._keys
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

        assert (isinstance(mf, scf.khf.KSCF))

        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        if mo_energy is None:
            mo_energy = mf.mo_energy

        if (mo_coeff is mf.mo_coeff and np.any(np.asarray(
            mo_energy) == INVALID_ORBITAL_ENERGY)):
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
        self.conv_tol = getattr(__config__, 'adc_kadc_RADC_conv_tol', 1e-7)
        self.tol_residual = getattr(__config__, 'adc_kadc_RADC_tol_res', 1e-4)
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
        """Return active-space MO integrals using :class:`kadc_ao2mo.AO2MO`.

        Args:
            backend (str, optional): Override integral_backend for this call.

        Returns:
            _ChemistsERIs: The owner of arrays, DF factors and scratch files.
        """
        return kadc_ao2mo.ao2mo(self, backend)

    def transform_integrals(self, backend=None):
        """Compatibility method; use :meth:`ao2mo`."""
        return self.ao2mo(backend)

    @property
    def chunk_size(self):
        """Cached automatic chunk size, or the explicitly assigned size."""
        if self._chunk_size is None:
            self._chunk_size = kadc_ao2mo.chunk_size(self)
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        if value is not None and (not isinstance(value, (int, np.integer))
                                  or value < 1):
            raise ValueError('chunk_size must be a positive integer or None.')
        self._chunk_size = value
        self._chunk_size_manual = value is not None

    @property
    def chnk_size(self):
        """Compatibility alias for chunk_size."""
        return self.chunk_size

    @chnk_size.setter
    def chnk_size(self, value):
        self.chunk_size = value

    def get_chnk_size(self):
        """Compatibility method for chunk-size estimates."""
        return kadc_ao2mo.chunk_size(self)

    @property
    def method_type(self):
        """Compatibility alias for type_, also used by molecular base code."""
        return self.type_

    @method_type.setter
    def method_type(self, value):
        self.type_ = value

    @property
    def nkpts(self):
        return len(self.kpts)

    @property
    def nocc(self):
        return self.get_nocc()

    @property
    def nmo(self):
        return self.get_nmo()

    get_nocc = get_nocc
    get_nmo = get_nmo

    @property
    def nvir(self):
        return self.nmo - self.nocc

    @property
    def n_singles(self):
        return self.nocc if self.type_ == 'ip' else self.nvir

    @property
    def doubles_shape(self):
        if self.type_ == 'ip':
            return (self.nkpts, self.nkpts, self.nvir, self.nocc, self.nocc)
        return (self.nkpts, self.nkpts, self.nocc, self.nvir, self.nvir)

    def _pad_orbitals(self):
        """Cache the padded active coefficients and occupied/virtual
        energies."""
        if self._padded_orbitals is None:
            coeff, energy = _add_padding(self, self.mo_coeff, self.mo_energy)
            energy = np.asarray(energy)
            self._padded_orbitals = (coeff, energy[:, :self.nocc],
                                    energy[:, self.nocc:])
        return self._padded_orbitals

    @property
    def padded_coeff(self):
        """Active MO coefficients with KMP2 padding, shape (nk, nao, nmo)."""
        return self._pad_orbitals()[0]

    @property
    def e_occ(self):
        """Padded occupied orbital energies, shape (nk, nocc)."""
        return self._pad_orbitals()[1]

    @property
    def e_vir(self):
        """Padded virtual orbital energies, shape (nk, nvir)."""
        return self._pad_orbitals()[2]

    def _orbital_data(self):
        """Compatibility helper; use padded_coeff, e_occ and e_vir."""
        return self._pad_orbitals()

    def _invalidate_orbitals(self):
        self._padded_orbitals = None
        if not self._chunk_size_manual:
            self._chunk_size = None

    def _init_from_ground_state(self, adc):
        """Share reference data without taking ownership of its scratch
        file."""
        self.__dict__.update({key: value for key, value in adc.__dict__.items()
                              if key not in ('_amp_h5file', '_amp_tmpfile',
                                             '_adc_es')})
        self._adc = adc
        self._padded_orbitals = None
        self._keys = set(self.__dict__)

    def _charged_init_guess(self, nroots=1, diag=None, ascending=True,
                            type=None, ini=None, kshift=None):
        """Common IP/EA unit guesses and frozen-space input validation."""
        dim = self.vector_size()
        if not 1 <= nroots <= dim:
            raise ValueError('nroots must be between 1 and vector_size().')
        if type == 'read':
            g = np.asarray(ini).T.copy()
            if g.shape != (dim, nroots):
                raise ValueError(
                    f'Shape of guess each k point should be ({dim},{nroots})')
            for root in range(nroots):
                singles = g[:self.n_singles, root]
                doubles = g[self.n_singles:, root].reshape(self.doubles_shape)
                singles, doubles = self._mask_frozen(
                    singles, doubles, kshift, const=0.)
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

    def _dump_orbital_info(self):
        """Log active, frozen and removed orbital counts for each k point."""
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
        """Validate the reference, populate amplitudes and correlation
        energy."""
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
        """Compute and store reference amplitudes and correlation energy.

        Args:
            eris (_ChemistsERIs, optional): Pretransformed active integrals.

        Returns:
            float: Correlation energy per primitive cell, in Hartree.
                The amplitudes are available as self.t1 and self.t2.
                This is the correlation contribution, not the SCF total.

        Example:
            >>> e_corr = kadc.ground_state_energy()
            >>> first_order_doubles = kadc.t2[0]
        """
        started = logger.process_clock(), logger.perf_counter()
        self._compute_reference(eris)
        logger.timer(self, 'ground-state ADC', *started)
        return self.e_corr

    def kernel_gs(self, eris=None):
        """Deprecated tuple-returning wrapper for ground_state_energy."""
        warnings.warn('kernel_gs is deprecated; use ground_state_energy() '
                      'and read t1 and t2 from the ADC instance.',
                      DeprecationWarning, stacklevel=2)
        self.ground_state_energy(eris)
        return self.e_corr, self.t1, self.t2

    def _clear_excitations(self):
        """Discard stale excited-state results on a reference-only run."""
        self.E = self.U = self.P = self.X = self.T = None
        self.converged = None
        self._adc_es = None

    def kernel(self, nroots=1, guess=None, eris=None, kptlist=None):
        """Compute excitations, or just the reference when nroots < 1.

        Args:
            nroots (int): Roots per selected momentum sector. Zero or a
                negative value requests only the ground-state calculation.
            guess: Optional trial vectors grouped in kptlist order.
            eris (_ChemistsERIs, optional): Pretransformed integrals.
            kptlist: Momentum-sector indices, defaulting to every k point.

        Returns:
            tuple: E, U, P, X for an excited-state run; these are also stored
                on the driver. A reference-only run returns scalar e_corr.
        """
        if not isinstance(nroots, (int, np.integer)):
            raise TypeError('nroots must be an integer.')
        started = logger.process_clock(), logger.perf_counter()
        eris = self._compute_reference(eris)
        if nroots < 1:
            self._clear_excitations()
            logger.info(self, f'nroots = {nroots}; only the ground state '
                        'was calculated.')
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
        e_exc, v_exc, strengths, x, adc_es = solve(
            nroots=nroots, guess=guess, eris=eris, kptlist=kptlist)
        self._adc_es = adc_es
        self.E, self.U, self.P, self.X = e_exc, v_exc, strengths, x
        self.converged = adc_es.converged
        self.T = adc_es.T if self.type_ == 'ee' else None
        logger.timer(self, 'complete ADC kernel', *started)
        return e_exc, v_exc, strengths, x

    def _close_amp_scratch(self):
        # close the HDF5 scratch file backing the on-disk amplitudes
        for attr in ('_amp_h5file', '_amp_tmpfile'):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except (AttributeError, OSError, ValueError):
                    pass

    def __del__(self):
        self._close_amp_scratch()

    def ip_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ip
        adc_es = kadc_rhf_ip.RADCIP(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        adc_es.U = v_exc
        return e_exc, v_exc, spec_fac, x, adc_es

    def ea_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ea
        adc_es = kadc_rhf_ea.RADCEA(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        adc_es.U = v_exc
        return e_exc, v_exc, spec_fac, x, adc_es

    def ee_adc(self, nroots=1, guess=None, eris=None, kptlist=None):
        from pyscf.pbc.adc import kadc_rhf_ee
        adc_es = kadc_rhf_ee.RADCEE(self)
        e_exc, v_exc, spec_fac, x = adc_es.kernel(nroots, guess, eris, kptlist)
        return e_exc, v_exc, spec_fac, x, adc_es

    def density_fit(self, auxbasis=None, with_df=None):
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
            dm = np.zeros((len(K_idx), len(nroots), self.nkpts, nmo, nmo),
                dtype=np.complex128)
            occ_idx = np.arange(nocc)
            k0_idx = np.arange(len(K_idx))
            k1_idx = np.arange(self.nkpts)
            p_k_idx = padding_k_idx(self, kind="joint")
            s_idx = np.arange(len(nroots))
            K0, S, K1, O = np.meshgrid(k0_idx, s_idx, k1_idx, occ_idx,
                indexing='ij')
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
                        mo_k = np.hstack((mo_k, np.zeros((mo_k.shape[0], npad),
                            dtype=mo_k.dtype)))
                    mo.append(mo_k)
                mo = np.stack(mo)
                rdm1 = einsum('kpI,KSkIJ,kqJ->KSkpq', mo, rdm1, mo.conj())
        elif ao_repr:
            mo = padded_mo_coeff(self, self.mo_coeff)
            mo_H = [c.conj() for c in mo]
            rdm1 = einsum('kpI,KSkIJ,kqJ->KSkpq', mo, rdm1, mo_H)

        return rdm1


    def amplitudes(self, eris=None):
        """Populate self.t1, self.t2 and the ladder intermediate.

        Args:
            eris (_ChemistsERIs, optional): Active-space integral container.

        Returns:
            None: Results remain on the driver. Doubles are backed by the
            scratch file owned by this instance.
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
        t2_1 = f.create_dataset('t2_1', (nkpts, nkpts, nkpts, nocc, nocc, nvir,
            nvir), dtype=eris.ovov.dtype)

        mo_e_o, mo_e_v = self.e_occ, self.e_vir

        # Locate occupied and virtual elements that are not padding.
        nonzero_opadding, nonzero_vpadding = padding_k_idx(self, kind="split")

        kconserv = self.khelper.kconserv
        touched = np.zeros((nkpts, nkpts, nkpts), dtype=bool)

        for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
            if touched[ki, kj, ka]:
                continue

            kb = kconserv[ki, ka, kj]
            # For discussion of LARGE_DENOM, see t1new update above
            eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                           [0, nvir, ka, mo_e_v, nonzero_vpadding],
                           fac=[1.0, -1.0])

            ejb = _get_epq([0, nocc, kj, mo_e_o, nonzero_opadding],
                           [0, nvir, kb, mo_e_v, nonzero_vpadding],
                           fac=[1.0, -1.0])
            eijab = eia[:, None, :, None] + ejb[:, None, :]

            t2_1[ki, kj, ka] = eris.ovov[ki, ka, kj].conj().transpose((0, 2, 1,
                3)) / eijab

            if ka != kb:
                eijba = eijab.transpose(0, 1, 3, 2)
                t2_1[ki, kj, kb] = eris.ovov[ki, kb, kj].conj().transpose(
                    (0, 2, 1, 3),
                ) / eijba

            touched[ki, kj, ka] = touched[ki, kj, kb] = True

        cput0 = log.timer_debug1(
            'Completed t2_1 amplitude calculation',
            *cput0,
        )

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
                                    chunk_size,
                                ).reshape(
                                    -1,
                                    nvir,
                                    nvir,
                                    nvir,
                                ) / nkpts
                                k = eris_ovvv.shape[0]
                                t1_2[ki] += 1.5 * einsum('kdac,ikcd->ia',
                                    eris_ovvv, t2_1[ki, kk, kc, :, a:a + k])
                                t1_2[ki] -= 0.5 * einsum(
                                    'kdac,kicd->ia',
                                    eris_ovvv,
                                    t2_1[kk, ki, kc, a:a + k, :],
                                )
                                del eris_ovvv
                                eris_ovvv = _ChemistsERIs.ovvv_df(
                                    eris.Lov[kk, kc],
                                    eris.Lvv[ka, kd],
                                    p,
                                    chunk_size,
                                ).reshape(
                                    -1,
                                    nvir,
                                    nvir,
                                    nvir,
                                ) / nkpts
                                t1_2[ki] -= 0.5 * einsum('kcad,ikcd->ia',
                                    eris_ovvv, t2_1[ki, kk, kc, :, a:a + k])
                                t1_2[ki] += 0.5 * einsum(
                                    'kcad,kicd->ia',
                                    eris_ovvv,
                                    t2_1[kk, ki, kc, a:a + k, :],
                                )
                                del eris_ovvv
                                a += k
                        else:
                            eris_ovvv = eris.ovvv[:]
                            t1_2[ki] += 1.5 * einsum('kdac,ikcd->ia',
                                                         eris_ovvv[kk, kd,
                                                             ka], t2_1[ki, kk,
                                                             kc])
                            t1_2[ki] -= 0.5 * einsum('kdac,kicd->ia',
                                                         eris_ovvv[kk, kd,
                                                             ka], t2_1[kk, ki,
                                                             kc])
                            t1_2[ki] -= 0.5 * einsum('kcad,ikcd->ia',
                                                         eris_ovvv[kk, kc,
                                                             ka], t2_1[ki, kk,
                                                             kc])
                            t1_2[ki] += 0.5 * einsum('kcad,kicd->ia',
                                                         eris_ovvv[kk, kc,
                                                             ka], t2_1[kk, ki,
                                                             kc])
                            del eris_ovvv

                    for kl in range(nkpts):
                        kc = kconserv[kk, ki, kl]
                        ka = kconserv[kl, kc, kk]

                        t1_2[ki] -= 1.5 * einsum('lcki,klac->ia',
                                                     eris_ovoo[kl, kc,
                                                         kk], t2_1[kk, kl,
                                                         ka])
                        t1_2[ki] += 0.5 * einsum('lcki,lkac->ia',
                                                     eris_ovoo[kl, kc,
                                                         kk], t2_1[kl, kk,
                                                         ka])
                        t1_2[ki] -= 0.5 * einsum('kcli,lkac->ia',
                                                     eris_ovoo[kk, kc,
                                                         kl], t2_1[kl, kk,
                                                         ka])
                        t1_2[ki] += 0.5 * einsum('kcli,klac->ia',
                                                     eris_ovoo[kk, kc,
                                                         kl], t2_1[kk, kl,
                                                         ka])

            for ki in range(nkpts):
                ka = ki
                eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                               [0, nvir, ka, mo_e_v, nonzero_vpadding],
                               fac=[1.0, -1.0])
                t1_2[ki] = t1_2[ki] / eia

            cput0 = log.timer_debug1("Completed t1_2 amplitude calculation",
                *cput0)

        t2_2 = None
        t1_3 = None
        t2_1_vvvv = None

        if (self.method == 'adc(3)' or
                (not self.approx_trans_moments and
                 (self.method == 'adc(2)-x' or self.type_ == 'ee'))):
            # Compute second-order doubles t2 (tijab)
            t2_1_vvvv = f.create_dataset(
                't2_1_vvvv', (nkpts, nkpts, nkpts, nocc, nocc, nvir,
                    nvir), dtype=eris.ovov.dtype)
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
                        t2_1_vvvv[ki, kj, ka] += self._contract_ladder(t2_1[ki,
                            kj, kc], eris.Lvv, ka, kb, kc)
                    else:
                        t2_1_vvvv[ki, kj, ka] += self._contract_ladder(t2_1[ki,
                            kj, kc], eris.vvvv, kc, kd, ka)

            t2_2 = f.create_dataset('t2_2', (nkpts, nkpts, nkpts, nocc, nocc,
                                    nvir, nvir), dtype=eris.ovov.dtype)
            for ki, kj, ka in kpts_helper.loop_kkk(nkpts):
                amplitude = t2_1_vvvv[ki, kj, ka]
                if self.exxdiv is not None:
                    amplitude -= 2.0 * madelung * t2_1[ki, kj, ka]
                for kk in range(nkpts):

                    kc = kconserv[ki, ka, kk]
                    kb = kconserv[kj, kk, kc]
                    amplitude -= einsum(
                        'kjbc,kica->ijab',
                        eris_oovv[kk, kj, kb],
                        t2_1[kk, ki, kc],
                    )

                    kc = kconserv[kk, ka, kj]
                    kb = kconserv[ki, kk, kc]
                    amplitude -= einsum(
                        'kibc,jkca->ijab',
                        eris_oovv[kk, ki, kb],
                        t2_1[kj, kk, kc],
                    )

                    kc = kconserv[ka, kj, kk]
                    kb = kconserv[ki, kc, kk]
                    amplitude -= einsum(
                        'kjac,ikcb->ijab',
                        eris_oovv[kk, kj, ka],
                        t2_1[ki, kk, kc],
                    )

                    kc = kconserv[ka, ki, kk]
                    kb = kconserv[kk, kc, kj]
                    amplitude -= einsum(
                        'kiac,kjcb->ijab',
                        eris_oovv[kk, ki, ka],
                        t2_1[kk, kj, kc],
                    )

                for kl in range(nkpts):
                    kk = kconserv[kj, kl, ki]
                    amplitude += einsum(
                        'kilj,klab->ijab',
                        eris_oooo[kk, ki, kl],
                        t2_1[kk, kl, ka],
                    )

                for kk in range(nkpts):

                    kc = kconserv[ki, ka, kk]
                    kb = kconserv[kc, kk, kj]

                    amplitude += 2 * einsum('kcbj,kica->ijab',
                                                       eris_ovvo[kk, kc,
                                                           kb], t2_1[kk, ki,
                                                           kc])
                    kc = kconserv[kk, ka, ki]
                    amplitude -= einsum(
                        'kcbj,ikca->ijab',
                        eris_ovvo[kk, kc, kb],
                        t2_1[ki, kk, kc],
                    )

                    kc = kconserv[kk, ki, ka]
                    kb = kconserv[kk, kj, kc]
                    amplitude += 2 * einsum('kcai,kjcb->ijab',
                                                       eris_ovvo[kk, kc,
                                                           ka], t2_1[kk, kj,
                                                           kc])

                    kc = kconserv[kk, ki, ka]
                    kb = kconserv[kj, kk, kc]
                    amplitude -= einsum(
                        'kcai,jkcb->ijab',
                        eris_ovvo[kk, kc, ka],
                        t2_1[kj, kk, kc],
                    )

                kb = kconserv[ki, ka, kj]
                eia = _get_epq([0, nocc, ki, mo_e_o, nonzero_opadding],
                               [0, nvir, ka, mo_e_v, nonzero_vpadding],
                               fac=[1.0, -1.0])

                ejb = _get_epq([0, nocc, kj, mo_e_o, nonzero_opadding],
                               [0, nvir, kb, mo_e_v, nonzero_vpadding],
                               fac=[1.0, -1.0])
                eijab = eia[:, None, :, None] + ejb[:, None, :]

                t2_2[ki, kj, ka] = amplitude / eijab

            cput0 = log.timer_debug1("Completed t2_2 amplitude calculation",
                *cput0)

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
                            term,
                            {'i': ki, 'a': ki},
                            (1, -1),
                        )
                        if value is not None:
                            t1_3[ki] += value
                    denominator = _get_epq(
                        [0, nocc, ki, mo_e_o, nonzero_opadding],
                        [0, nvir, ki, mo_e_v, nonzero_vpadding],
                        fac=[1., -1.])
                    t1_3[ki] /= denominator

        self.t1 = (t1_2, t1_3)
        self.t2 = (t2_1, t2_2)
        self.imds.t2_1_vvvv = t2_1_vvvv

    def correlation_energy(self, eris=None):
        """Store and return the MP2/MP3 reference correlation contribution.

        Args:
            eris (_ChemistsERIs, optional): Active-space integrals. If
            amplitudes
                are absent they are computed first. With if_corr=False, the
                stored and returned contribution is zero.

        Returns:
            float: Correlation energy in Hartree per primitive cell.
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
                    nocc,
                    nocc,
                    nvir,
                    nvir,
                )
                del vvvv_p
                a += k
        else:
            for p in range(0, nvir, chunk_size):
                vvvv_p = vvvv[ka, kb, kc, p:p + chunk_size, :, :, :].reshape(
                    -1,
                    nvir * nvir,
                )
                k = vvvv_p.shape[0]
                t += np.dot(t_amp[:, a:a + k], vvvv_p.conj()).reshape(
                    nocc,
                    nocc,
                    nvir,
                    nvir,
                )
                del vvvv_p
                a += k

        return t

    def compute_amplitudes(self, eris):
        """Deprecated amplitude tuple API; use amplitudes and instance
        fields."""
        warnings.warn('Use amplitudes() and the t1/t2 instance attributes.',
                      DeprecationWarning, stacklevel=2)
        self.amplitudes(eris)
        return self.t1, self.t2, self.imds.t2_1_vvvv

    def compute_energy(self, t2, eris):
        """Deprecated pure-energy API; use correlation_energy in new code."""
        warnings.warn('Use correlation_energy() and the e_corr attribute.',
                      DeprecationWarning, stacklevel=2)
        saved_t2, saved_energy = self.t2, self.e_corr
        self.t2 = t2
        try:
            return self.correlation_energy(eris)
        finally:
            self.t2, self.e_corr = saved_t2, saved_energy

    def compute_amplitudes_energy(self, eris, verbose=None, if_corr=True):
        """Deprecated combined tuple API; use ground_state_energy."""
        warnings.warn('Use ground_state_energy() and instance attributes.',
                      DeprecationWarning, stacklevel=2)
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
