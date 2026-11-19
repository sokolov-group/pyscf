"""Contracts for the consolidated periodic ADC API."""
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import ast
import doctest
import io
import gc
import unittest
import weakref

import numpy as np

from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc import kadc_ao2mo, kadc_rhf, lib as adc_lib
from pyscf.pbc.scf.hf import INVALID_ORBITAL_ENERGY


def setUpModule():
    r"""Prepare the small two-k helium reference used by the API and doc tests.

    Returns:
        None: Store mf as a density-fitted KRHF reference with two s basis
        functions, one occupied and one virtual MO at each k point. This is the
        same reproducible system constructed by the KRADCBase class example.
    """
    global mf
    cell = gto.M(
        atom='He 0 0 0',
        a=np.eye(3) * 3,
        basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        verbose=0)
    mf = scf.KRHF(
        cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit().run()


class DriverAPI(unittest.TestCase):
    r"""Check reference-driver state, validation and compatibility contracts.

    The shared module fixture is the small two-k helium KRHF reference.
    Individual methods construct independent drivers to test scalar energies,
    root clearing, cached data, guess routing and AO densities.

    Run with ``python -m unittest test.test_kadc_api.DriverAPI``.

    Methods:
        test_periodic_sector_class_names: Check canonical class names and
            conversion from another sector.
        test_adc_conversion_reference_ownership: Convert between all periodic
            ADC classes without sharing sector state.
        test_lazy_cell_inputs_and_stream_methods: Accept raw Cells and unrun
            KRHF references with StreamObject chaining.
        test_typed_fno_construction: Select the FNO sector by concrete class,
            including converted inputs.
        test_charged_hamiltonian_order_helpers: Check correction-only and
            accumulation contracts for IP/EA helpers.
        test_charged_density_root_selection_and_ao: Restore frozen orbitals for
            several roots and arbitrary sector order.
        test_charged_diagonal_values_and_dtypes: Check charged diagonals
            against explicit orbital/momentum loops.
        test_charged_diagonal_padding_and_integral_reuse: Preserve frozen-
            padding sentinels and reuse supplied integral owners.
        test_flat_orbital_masks_and_per_k_counts: Check flat masks and orbital
            counts on unequal per-k MO spaces.
        test_ground_state_and_amplitudes: Check scalar reference energies,
            stored amplitudes and correlation control.
        test_reference_only_and_stale_results: Check reference-only kernels and
            stale-root clearing in every sector.
        test_fno_reference_only: Check the FNO zero-root scalar-energy path.
        test_direct_neutral_reference_only: Check a direct KRADCEE reference-
            only run without constructing an action.
        test_padding_and_chunking: Check cached padding, automatic chunking and
            manual overrides.
        test_input_validation: Check method normalization and invalid
            root/sector inputs.
        test_solver_uses_sector_guess_helpers: Check that each sector routes
            default and supplied guesses through _init_guess and reproduces
            converged roots from supplied vectors.
        test_sector_intermediates_and_transitions: Check intermediate reuse and
            root layout across two transfers.
        test_reference_density_ao: Check AO reference densities with complex
            phases and reduced MO spaces.
    """
    def test_periodic_sector_class_names(self):
        """Check canonical class names and conversion from another sector."""
        mykradc = adc.KRADCIP(mf)
        for canonical, name in ((adc.KRADCIP, 'KRADCIP'),
                                (adc.KRADCEA, 'KRADCEA'),
                                (adc.KRADCEE, 'KRADCEE')):
            self.assertEqual(canonical.__name__, name)
            self.assertIsInstance(canonical(mykradc), kadc_rhf.KRADCBase)
            self.assertIs(type(canonical(mykradc)), canonical)
            self.assertFalse(hasattr(canonical(mykradc), 'type_'))

    def test_adc_conversion_reference_ownership(self):
        r"""Convert between all periodic ADC classes without sharing sector
        state.

        Returns:
            None: Preserve reference settings and disk-backed amplitudes, copy
            mutable bookkeeping, clear every root/cache, and keep the old
            amplitude generation readable after its source recomputes.
        """
        classes = (adc.KRADCBase, adc.KRADCIP, adc.KRADCEA, adc.KRADCEE)
        for source_type in classes:
            source = source_type(mf).set(method='adc(3)',
                                         approx_trans_moments=False)
            source.chunk_size = 2
            source.density_gv = np.array([0.1, 0.2, 0.3])
            source.ground_state_energy()
            amplitude = np.asarray(source.t2[0]).copy()
            # Populate sector caches to detect accidental result sharing.
            source.E = source.P = np.ones((1, 1))
            source.U = source.X = source.T = np.ones((1, 1, 1))
            source.converged = np.ones((1, 1), dtype=bool)
            source._ee_momentum[0] = 'old momentum map'
            source._transition_cache[0] = 'old transition density'
            source.sector_specific_result = 'do not copy'
            converted = []
            for target_type in classes:
                with self.subTest(source=source_type, target=target_type):
                    target = target_type(source)
                    converted.append(target)
                    self.assertIs(type(target), target_type)
                    self.assertIs(target._scf, source._scf)
                    self.assertIs(target.mo_coeff, source.mo_coeff)
                    self.assertIs(target.t2, source.t2)
                    self.assertIs(target._amp_storage, source._amp_storage)
                    self.assertIsNot(target.imds, source.imds)
                    self.assertEqual(target.method, source.method)
                    self.assertEqual(target.e_corr, source.e_corr)
                    self.assertEqual(target.chunk_size, 2)
                    self.assertTrue(target._chunk_size_manual)
                    self.assertFalse(target.approx_trans_moments)
                    self.assertFalse(hasattr(target, 'type_'))
                    self.assertFalse(hasattr(target, 'sector_specific_result'))
                    self.assertNotIn('_amp_h5file', target.__dict__)
                    for name in ('E', 'U', 'P', 'X', 'T', 'converged'):
                        self.assertIsNone(getattr(target, name))
                    self.assertEqual(target._ee_momentum, {})
                    self.assertEqual(target._transition_cache, {})
                    target.density_gv[0] = 1.0
                    self.assertEqual(source.density_gv[0], 0.1)
            # Recomputing the source must leave old amplitudes readable.
            source.ground_state_energy()
            for target in converted:
                np.testing.assert_array_equal(target.t2[0], amplitude)
            # Orbital/frozen-space overrides cannot retain old amplitudes.
            changed = adc.KRADCEA(source, frozen=0)
            self.assertEqual(changed.frozen, 0)
            self.assertIsNone(changed.t2)
            self.assertIsNone(changed.e_corr)

        temporary = adc.KRADCIP(mf)
        temporary.ground_state_energy()
        amplitude = np.asarray(temporary.t2[0]).copy()
        retained = adc.KRADCEA(temporary)
        source_reference = weakref.ref(temporary)
        del temporary
        gc.collect()
        self.assertIsNone(source_reference())
        np.testing.assert_array_equal(retained.t2[0], amplitude)

    def test_lazy_cell_inputs_and_stream_methods(self):
        r"""Accept raw Cells and unrun KRHF references with StreamObject
        chaining.

        Returns:
            None: Construction/conversion performs no SCF, Cell inputs select
            Gamma or an explicit mesh, and reference-only run() returns the
            same object with a checked scalar energy. The base has no molecular
            RADC dependency and requires a concrete class for positive roots.
        """
        from pyscf import lib as pyscf_lib
        from pyscf.adc.radc import RADC as MolecularRADC

        mykradcbase = adc.KRADCBase(mf)
        self.assertIsInstance(mykradcbase, pyscf_lib.StreamObject)
        self.assertNotIsInstance(mykradcbase, MolecularRADC)
        with self.assertRaisesRegex(NotImplementedError, 'Choose KRADCIP'):
            mykradcbase.kernel(1)
        for solver_type in (adc.KRADCIP, adc.KRADCEA, adc.KRADCEE):
            pending = solver_type(mf.cell)
            self.assertIsNone(pending.mo_coeff)
            self.assertEqual(pending.kpts.shape, (1, 3))
            with mock.patch.object(
                    pending._scf, 'kernel',
                    side_effect=AssertionError('SCF in constructor')):
                converted = adc.KRADCEA(pending)
                self.assertIsNone(converted.mo_coeff)
                self.assertIsNone(converted._chunk_size)
            explicit = solver_type(mf.cell, kpts=mf.kpts)
            self.assertIs(explicit.set(compute_properties=False), explicit)
            self.assertIs(explicit.run(0), explicit)
            self.assertAlmostEqual(explicit.e_corr, -0.00416766, 8)
            self.assertIsNotNone(explicit.mo_coeff)
        unrun = scf.KRHF(mf.cell, mf.kpts, exxdiv=None).density_fit()
        mykradcee = adc.KRADCEE(unrun)
        self.assertIsNone(mykradcee.mo_coeff)
        mykradcee.kernel(1, kptlist=[0])
        self.assertAlmostEqual(mykradcee.E[0, 0], 1.756312, 6)
        with self.assertRaisesRegex(ValueError, 'only with a Cell'):
            adc.KRADCEE(mf, kpts=mf.kpts)
        with self.assertRaises(TypeError):
            adc.KRADCEE(object())

    def test_typed_fno_construction(self):
        r"""Select the FNO sector by concrete class, including converted
        inputs.

        Returns:
            None: Match every sector to its FNO subclass, preserve common
            settings and clear FNO/excitation results on conversion. Reject
            unavailable state-specific EE densities before running ADC.
        """
        for solver_type, fno_type in ((adc.KRADCIP, adc.KRADCIPFNO),
                                      (adc.KRADCEA, adc.KRADCEAFNO),
                                      (adc.KRADCEE, adc.KRADCEEFNO)):
            source = solver_type(mf).set(method='adc(3)')
            fno = adc.KRADC2FNO(source)
            self.assertIs(type(fno), fno_type)
            self.assertIsInstance(fno, solver_type)
            self.assertEqual(fno.method, 'adc(3)')
            fno.ref_state = [[0], [0]]
            fno.e_can = np.ones((1, 1))
            converted = adc.KRADC2FNO(fno)
            self.assertIs(type(converted), fno_type)
            self.assertIsNone(converted.ref_state)
            self.assertIsNone(converted.e_can)
        with self.assertRaisesRegex(NotImplementedError, 'ref_state=None'):
            adc.KRADCEEFNO(mf).set(ref_state=1).kernel(1)

    def test_charged_hamiltonian_order_helpers(self):
        r"""Check correction-only and accumulation contracts for IP/EA helpers.

        Returns:
            None: Verify orbital-energy signs, ADC(2)-x singles equivalence,
            independent output allocation, in-place accumulation into nonzero
            targets, and third-order composition without changing amplitudes.
        """
        for solver_type in (adc.KRADCIP, adc.KRADCEA):
            solver = solver_type(mf).set(method='adc(3)')
            solver.ground_state_energy()
            amplitudes = np.asarray(solver.t2[0]).copy()
            with solver.ao2mo() as eris:
                zeroth = solver._get_adc0_ham()
                energies = (solver.e_occ if solver_type is adc.KRADCIP
                            else solver.e_vir)
                np.testing.assert_allclose(
                    np.diagonal(zeroth, axis1=1, axis2=2), energies)
                second = solver._get_adc2_ham(eris)
                third = solver._get_adc3_ham(eris)
                for helper, correction in ((solver._get_adc2_ham, second),
                                            (solver._get_adc3_ham, third)):
                    target = np.full_like(correction, 0.7)
                    self.assertIs(helper(eris, target), target)
                    np.testing.assert_allclose(target, correction + 0.7,
                                               atol=1e-13)
                np.testing.assert_allclose(solver.get_imds(eris),
                                           zeroth + second + third, atol=1e-13)
                solver.method = 'adc(2)'
                adc2 = solver.get_imds(eris)
                solver.method = 'adc(2)-x'
                np.testing.assert_array_equal(solver.get_imds(eris), adc2)
                np.testing.assert_array_equal(solver.t2[0], amplitudes)

    def test_charged_density_root_selection_and_ao(self):
        r"""Restore frozen orbitals for several roots and arbitrary sector
        order.

        Returns:
            None: Check all saved roots when their count differs from the
            sector count, independent ragged selections, and AO transformation
            against explicit coefficient contractions for both charged sectors.
        """
        cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
                     basis=[[0, (1., 1.)], [0, (.5, 1.)], [0, (.2, 1.)]],
                     verbose=0)
        reference = scf.KRHF(cell, cell.make_kpts([1, 1, 2]),
                             exxdiv=None).density_fit().run()
        for solver_type in (adc.KRADCIP, adc.KRADCEA):
            solver = solver_type(reference, frozen=[2])
            solver.kernel(3, kptlist=[1, 0])
            density = solver.make_rdm1()
            self.assertEqual(density.shape, (2, 3, 2, 3, 3))
            ao_density = solver.make_rdm1(ao_repr=True)
            expected = np.einsum('kpi,skij,kqj->skpq',
                                  solver.mo_coeff, density[0],
                                  np.conj(solver.mo_coeff))
            np.testing.assert_allclose(ao_density[0], expected, atol=1e-12)
            selected = solver.make_rdm1(root=np.array([[0, 2], [1, 0]]))
            np.testing.assert_allclose(selected[0], density[0, [0, 2]])
            np.testing.assert_allclose(selected[1], density[1, [1, 0]])
            # Request unequal root counts in reverse saved-sector order.
            ragged = solver.make_rdm1(root=[[0], [1, 2]], K_idx=[0, 1])
            self.assertIsInstance(ragged, list)
            self.assertEqual([len(dm) for dm in ragged], [1, 2])
            np.testing.assert_allclose(ragged[0][0], density[1, 0])
            np.testing.assert_allclose(ragged[1], density[0, [1, 2]])

    def test_charged_diagonal_values_and_dtypes(self):
        r"""Check charged diagonals against explicit orbital/momentum loops.

        Returns:
            None: Verify C-order layout, every momentum sector, real/complex
            single/double precision, and unchanged input intermediates with
            unequal occupied/virtual dimensions. No SCF or integrals are
            needed.
        """
        nkpts, nocc, nvir = 3, 2, 3
        kconserv = np.fromfunction(
            lambda p, q, r: (p - q + r) % nkpts,
            (nkpts, nkpts, nkpts), dtype=int)
        random = np.random.default_rng(15)
        for solver_type in (adc.KRADCIP, adc.KRADCEA):
            ionization = solver_type is adc.KRADCIP
            n_singles = nocc if ionization else nvir
            shape = ((nkpts, nkpts, nvir, nocc, nocc) if ionization else
                     (nkpts, nkpts, nocc, nvir, nvir))
            vector_size = n_singles + int(np.prod(shape))
            for dtype in (np.float32, np.float64, np.complex64, np.complex128):
                real_dtype = np.empty((), dtype=dtype).real.dtype
                e_occ = random.uniform(-1.0, -0.2, (nkpts, nocc)).astype(
                    real_dtype)
                e_vir = random.uniform(0.1, 1.0, (nkpts, nvir)).astype(
                    real_dtype)
                intermediate = random.normal(
                    size=(nkpts, n_singles, n_singles)).astype(dtype)
                if np.issubdtype(dtype, np.complexfloating):
                    intermediate += 1j * random.normal(
                        size=intermediate.shape).astype(real_dtype)
                original = intermediate.copy()
                model = SimpleNamespace(
                    method='ADC(2)', verbose=0, stdout=None, nkpts=nkpts,
                    n_singles=n_singles, doubles_shape=shape,
                    vector_size=lambda: vector_size,
                    e_occ=e_occ, e_vir=e_vir,
                    khelper=SimpleNamespace(kconserv=kconserv),
                    nmo=nocc+nvir, frozen=None,
                    mo_coeff=[np.eye(nocc+nvir)] * nkpts)
                for kshift in range(nkpts):
                    with self.subTest(sector=solver_type.__name__,
                                      dtype=dtype, kshift=kshift):
                        expected = np.empty(vector_size, dtype=dtype)
                        expected[:n_singles] = np.diagonal(
                            intermediate[kshift]) * (-1 if ionization else 1)
                        doubles = expected[n_singles:].reshape(shape)
                        # Explicit loops are independent of the slab broadcast.
                        for k1, k2, p, q, r in np.ndindex(shape):
                            if ionization:
                                kj = kconserv[kshift, k2, k1]
                                doubles[k1, k2, p, q, r] = (
                                    e_vir[k1, p] - e_occ[k2, q] - e_occ[kj, r])
                            else:
                                kb = kconserv[kshift, k2, k1]
                                doubles[k1, k2, p, q, r] = (
                                    e_vir[k2, q] + e_vir[kb, r] - e_occ[k1, p])
                        actual = solver_type.get_diag(
                            model, kshift, intermediate)
                        self.assertEqual(actual.dtype, np.dtype(dtype))
                        tolerance = (5e-7 if real_dtype == np.dtype(np.float32)
                                     else 1e-13)
                        np.testing.assert_allclose(actual, expected,
                                                   atol=tolerance, rtol=0)
                np.testing.assert_array_equal(intermediate, original)

    def test_charged_diagonal_padding_and_integral_reuse(self):
        r"""Preserve frozen-padding sentinels and reuse supplied integral
        owners.

        Returns:
            None: Check unequal active virtual spaces, positive inactive
            diagonals, the saved block layout and eris forwarding when the
            singles intermediate is built on demand.
        """
        from pyscf.lib.parameters import LARGE_DENOM

        cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
                     basis=[[0, (1., 1.)], [0, (.5, 1.)], [0, (.2, 1.)]],
                     verbose=0)
        reference = scf.KRHF(cell, cell.make_kpts([1, 1, 2]),
                             exxdiv=None).density_fit().run()
        for solver_type in (adc.KRADCIP, adc.KRADCEA):
            solver = solver_type(reference, frozen=[[2], []]).run(0)
            with solver.ao2mo() as eris:
                intermediate = solver.get_imds(eris)
                with mock.patch.object(solver, 'get_imds',
                                       return_value=intermediate) as get_imds:
                    diagonal = solver.get_diag(0, eris=eris)
                get_imds.assert_called_once_with(eris=eris)
                physical = np.ones(solver.vector_size())
                singles, doubles = solver._mask_frozen(
                    physical[:solver.n_singles],
                    physical[solver.n_singles:].reshape(solver.doubles_shape),
                    0, const=0)
                mask = np.concatenate((singles, doubles.ravel())).astype(bool)
                self.assertTrue(np.any(~mask))
                np.testing.assert_array_equal(diagonal[~mask], LARGE_DENOM)
                self.assertTrue(np.all(diagonal[mask].real < LARGE_DENOM / 2))

    def test_flat_orbital_masks_and_per_k_counts(self):
        """Check flat masks and orbital counts on unequal per-k MO spaces.

        Returns:
            None: Verify each orbital belongs to exactly one selected-space
            mask, removed virtuals are counted separately, and invalid k-point
            selectors raise clear errors without altering occupations.
        """
        model = SimpleNamespace(
            nkpts=2, frozen=[[0, 5], [0, 3]],
            mo_occ=[np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
                    np.array([2.0, 2.0, 0.0, 0.0])],
            mo_energy=[np.array(
                [-2.0, -1.0, -0.5, 0.1, 0.2, INVALID_ORBITAL_ENERGY]),
                       np.array([-2.0, -1.0, 0.1, INVALID_ORBITAL_ENERGY])])
        occupations = [values.copy() for values in model.mo_occ]
        expected = ([[0], [1, 2], [3, 4], [5]], [[0], [1], [2], [3]])
        for kpt in range(model.nkpts):
            masks = kadc_rhf.mo_splitter(model, np.int64(kpt))
            self.assertEqual(len(masks), 4)
            for mask in masks:
                self.assertEqual(mask.ndim, 1)
                self.assertEqual(mask.dtype, np.dtype(bool))
                self.assertEqual(mask.shape, model.mo_occ[kpt].shape)
            np.testing.assert_array_equal(np.sum(masks, axis=0), 1)
            self.assertEqual(
                [np.flatnonzero(mask).tolist() for mask in masks],
                expected[kpt])
            np.testing.assert_array_equal(model.mo_occ[kpt], occupations[kpt])
        self.assertEqual(
            [x.tolist() for x in kadc_rhf.orbital_number_lists(model)],
            [[1, 1], [2, 1], [2, 1], [0, 0], [1, 1]])
        np.testing.assert_array_equal(
            kadc_rhf.mo_splitter(model), kadc_rhf.mo_splitter(model, 0))
        for invalid in (-1, model.nkpts):
            with self.assertRaises(IndexError):
                kadc_rhf.mo_splitter(model, invalid)
        with self.assertRaises(TypeError):
            kadc_rhf.mo_splitter(model, 0.5)

    def test_ground_state_and_amplitudes(self):
        r"""Check scalar reference energies, stored amplitudes and correlation
        control.

        Verify class construction, amplitude availability, unchanged doubles
        after repeated correlation contractions, and if_corr=False behavior.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADCIP(mf)
        self.assertIs(type(mykradc), adc.KRADCIP)
        energy = mykradc.ground_state_energy()
        self.assertIsInstance(energy, float)
        self.assertIsNotNone(mykradc.t2[0])
        with mykradc.ao2mo() as eris:
            self.assertIsNone(mykradc.amplitudes(eris))
            self.assertAlmostEqual(
                mykradc.correlation_energy(eris), energy, 12)
            amplitude = np.asarray(mykradc.t2[0]).copy()
            mykradc.correlation_energy(eris)
            np.testing.assert_array_equal(mykradc.t2[0], amplitude)
        mykradc.if_corr = False
        self.assertEqual(mykradc.ground_state_energy(), 0.0)
        self.assertIsNotNone(mykradc.t2[0])

    def test_reference_only_and_stale_results(self):
        r"""Check reference-only kernels and stale-root clearing in every
        sector.

        Compute IP/EA/EE roots, request zero/negative root counts, and assert
        that excitation solvers are bypassed while ground amplitudes remain.
        Old E/U/P/X/T/convergence data must be cleared.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        for solver_type in (adc.KRADCIP, adc.KRADCEA, adc.KRADCEE):
            with self.subTest(sector=solver_type.__name__):
                mykradc = solver_type(mf)
                expected = mykradc.ground_state_energy()
                mykradc.nroots = 2
                mykradc.kernel(kptlist=[0])
                self.assertEqual(mykradc.E.shape, (1, 2))
                with mock.patch.object(
                    mykradc,
                    'gen_matvec',
                    side_effect=AssertionError('solver called')):
                    actual = mykradc.kernel(0, kptlist=[])
                self.assertAlmostEqual(actual, expected, 12)
                self.assertEqual(mykradc.nroots, 2)
                for field in ('E', 'U', 'P', 'X', 'T', 'converged', '_adc_es'):
                    self.assertIsNone(getattr(mykradc, field))
                self.assertIsNotNone(mykradc.t2[0])
                self.assertAlmostEqual(mykradc.kernel(-2), expected, 12)

    def test_fno_reference_only(self):
        r"""Check the FNO zero-root scalar-energy path.

        Request ground-density FNOs with nroots=0, compare the returned energy
        with e_corr_fno, and verify that no excited-state solver remains
        attached.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc_fno = adc.KRADC2FNO(mf)
        energy = mykradc_fno.kernel(0, pct_occ=0.9)
        self.assertIsInstance(energy, float)
        self.assertEqual(energy, mykradc_fno.e_corr_fno)
        self.assertIsNone(mykradc_fno.E)
        self.assertIsNone(mykradc_fno._adc_es)

    def test_direct_neutral_reference_only(self):
        r"""Check a direct KRADCEE reference-only run without constructing an
        action.

        Patch gen_matvec to reject accidental use, compute zero roots and
        assert
        that reference amplitudes are shared correctly while excitation data
        clear.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADCIP(mf)
        expected = mykradc.ground_state_energy()
        mykradcee = adc.KRADCEE(mykradc)
        with mock.patch.object(
            mykradcee, 'gen_matvec', side_effect=AssertionError('matvec built')
        ):
            self.assertAlmostEqual(mykradcee.kernel(0), expected, 12)
        np.testing.assert_array_equal(mykradcee.t2[0], mykradc.t2[0])
        self.assertIsNone(mykradcee.E)

    def test_padding_and_chunking(self):
        """Check cached padding, automatic chunking and manual overrides."""
        mykradc = adc.KRADCIP(mf)
        for name in ('type_', 'method_type'):
            self.assertFalse(hasattr(mykradc, name))
            with self.assertRaisesRegex(AttributeError, 'Select a sector'):
                setattr(mykradc, name, 'ee')
        coeff = mykradc.padded_coeff
        self.assertIs(coeff, mykradc.padded_coeff)
        self.assertEqual(mykradc.e_occ.shape, (mykradc.nkpts, mykradc.nocc))
        self.assertEqual(mykradc.e_vir.shape, (mykradc.nkpts, mykradc.nvir))
        with mock.patch.object(
            kadc_ao2mo, 'chunk_size', return_value=1) as size:
            mykradc.chunk_size = None
            self.assertEqual(mykradc.chunk_size, 1)
            self.assertEqual(mykradc.chunk_size, 1)
            size.assert_called_once()
        mykradc.chunk_size = 2
        mykradc.ground_state_energy()
        self.assertEqual(mykradc.chunk_size, 2)
        mykradc.chunk_size = None
        self.assertGreaterEqual(mykradc.chunk_size, 1)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                mykradc.chunk_size = invalid

    def test_input_validation(self):
        r"""Check method normalization and invalid root/sector inputs.

        Reject unsupported ADC orders, accept uppercase supported methods,
        and reject empty/out-of-range/noninteger sector lists and noninteger
        root counts for an excitation calculation.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADCIP(mf)
        with self.assertRaises(NotImplementedError):
            mykradc.method = 'adc(4)'
            mykradc.ground_state_energy()
        mykradc.method = 'ADC(2)'
        self.assertIsInstance(mykradc.ground_state_energy(), float)
        for kptlist in ([], [-1], [mykradc.nkpts], ['0']):
            with self.assertRaises(ValueError):
                mykradc.kernel(1, kptlist=kptlist)
        with self.assertRaises(TypeError):
            mykradc.kernel(1.5)

    def test_solver_uses_sector_guess_helpers(self):
        """Check that each sector routes default and supplied guesses through
        _init_guess and reproduces converged roots from supplied vectors.
        """
        mykradc = adc.KRADCIP(mf)
        mykradc.compute_properties = False
        mykradc.ground_state_energy()
        for solver_type in (adc.KRADCIP, adc.KRADCEA, adc.KRADCEE):
            mykradc_es = solver_type(mykradc)
            helper = '_init_guess'
            with (
                self.subTest(sector=type(mykradc_es).__name__),
                mock.patch.object(
                    mykradc_es, helper, wraps=getattr(mykradc_es, helper)
                ) as guesses,
            ):
                # Reuse the converged vectors as guesses for the second run.
                energies, vectors, _, _ = mykradc_es.kernel(2, kptlist=[0])
                from_input, _, _, _ = mykradc_es.kernel(
                    2, guess=vectors, kptlist=[0])
                np.testing.assert_allclose(from_input, energies, atol=1e-08)
                self.assertEqual(guesses.call_count, 2)
                self.assertEqual(guesses.call_args.kwargs['type_'], 'read')

    def test_sector_intermediates_and_transitions(self):
        """Check intermediate reuse and root layout across two transfers."""
        for cls in (adc.KRADCIP, adc.KRADCEA, adc.KRADCEE):
            solver = cls(mf)
            with self.subTest(sector=cls.__name__), mock.patch.object(
                    solver, 'get_imds', wraps=solver.get_imds) as build:
                _, _, _, transitions = solver.kernel(2, kptlist=[1, 0])
                if cls is adc.KRADCEE:
                    shifts = [c.kwargs['kshift'] for c in build.call_args_list]
                    self.assertEqual(shifts, [1, 0])
                    self.assertTrue(np.shares_memory(solver.T, transitions))
                    self.assertEqual(solver.T.shape, (2, 2, solver.nkpts
                                                      * solver.nmo ** 2))
                else:
                    build.assert_called_once()
                    self.assertEqual(transitions.shape, (2, solver.nmo, 2))

    def test_reference_density_ao(self):
        r"""Check AO reference densities with complex phases and reduced MO
        spaces.

        Compare C D C-dagger explicitly with the density API for unfrozen,
        frozen-virtual and projected coefficient spaces. Check AO shape,
        Hermiticity and overlap-weighted electron number.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        cell = gto.M(
            atom='He 0 0 0',
            a=np.eye(3) * 3,
            basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)], [0, (0.2, 1.0)]],
            verbose=0)
        reference = scf.KRHF(
            cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit().run()
        for frozen in (None, [2], 0):
            options = {}
            if frozen == 0:
                # A projected orbital space also has fewer MOs than AOs
                # when the full (with_frozen=True) density is requested.
                options = {name: np.asarray(getattr(reference, name))[..., :2]
                           for name in ('mo_coeff', 'mo_occ', 'mo_energy')}
            mykradc = adc.KRADCIP(reference, frozen=frozen, **options)
            # Complex phases exercise the conjugation in C D C^dagger.
            mykradc.mo_coeff = np.asarray(mykradc.mo_coeff) * np.exp(
                1j * np.array([0.2, -0.3, 0.7])[:len(mykradc.mo_occ[0])])
            mykradc.ground_state_energy()
            for with_frozen in (False, True):
                with self.subTest(frozen=frozen, with_frozen=with_frozen):
                    density = mykradc.make_ref_rdm1(with_frozen=with_frozen)
                    ao = mykradc.make_ref_rdm1(
                        with_frozen=with_frozen, ao_repr=True)
                    if with_frozen and frozen is not None:
                        coeff = mykradc.mo_coeff
                    else:
                        coeff = mykradc.padded_coeff
                    expected = np.einsum(
                        'kpi,kij,kqj->kpq',
                        coeff, density, np.conj(coeff))
                    self.assertEqual(ao.shape, (2, 3, 3))
                    np.testing.assert_allclose(ao, expected, atol=1e-12)
                    np.testing.assert_allclose(
                        ao, ao.conj().transpose(0, 2, 1), atol=1e-12)
                    electrons = np.einsum(
                        'kpq,kqp->',
                        ao, reference.get_ovlp()) / 2
                    self.assertAlmostEqual(electrons.real, 2.0, 10)


class Documentation(unittest.TestCase):
    r"""Execute every worked docstring example in the periodic ADC package.

    Each example block runs in an empty namespace, using only the imports and
    reference setup shown in its example. Numerical energies and tensor
    shapes are checked, including public factories,
    FNO/integral workflows and pure numerical models.

    Run with ``python -m unittest test.test_kadc_api.Documentation``.

    Methods:
        test_worked_examples: Execute package examples in independent empty
            namespaces.
    """
    def test_worked_examples(self):
        r"""Execute package examples in independent empty namespaces.

        Read docstrings from the source AST and give each example block an
        empty namespace, including blocks separated by explanatory prose.
        This prevents module globals or another block from hiding missing
        imports, undefined functions or reference setup.
        """
        output = io.StringIO()
        runner = doctest.DocTestRunner()
        parser = doctest.DocTestParser()
        directory = Path(kadc_rhf.__file__).parent
        paths = sorted(
            [*directory.glob('*.py'), *(directory / 'test').glob('*.py')])
        documented_nodes = (ast.Module, ast.ClassDef, ast.FunctionDef,
                            ast.AsyncFunctionDef)
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, documented_nodes):
                    continue
                doc = ast.get_docstring(node)
                if not doc or not parser.get_examples(doc):
                    continue
                name = (f'{path.name}:{node.body[0].lineno}:'
                        f'{getattr(node, 'name', 'module')}')
                blocks = []
                current = []
                # Prose separates examples with independent setup code.
                for part in parser.parse(doc):
                    if isinstance(part, doctest.Example):
                        current.append(part)
                    elif part.strip() and current:
                        blocks.append(current)
                        current = []
                if current:
                    blocks.append(current)
                for index, examples in enumerate(blocks):
                    example = doctest.DocTest(
                        examples,
                        {},
                        name,
                        str(path),
                        node.body[0].lineno - 1,
                        doc)
                    with self.subTest(example=name, block=index):
                        result = runner.run(example, out=output.write)
                        self.assertEqual(result.failed, 0, output.getvalue())
        self.assertGreater(runner.tries, 50)


class IntegralAPI(unittest.TestCase):
    r"""Check shared integral ownership, lazy access and DF slice interfaces.

    Use the module helium reference for real transformation/storage checks,
    and synthetic complex factors for independent reconstruction comparisons.

    Run with ``python -m unittest test.test_kadc_api.IntegralAPI``.

    Methods:
        test_context_and_lazy_block_properties: Check AO2MO context metadata
            and nonexpanding integral block access.
        test_incore_memory_selection: Check the backend at a fixed RAM budget.
        test_chunked_ladder_reads: Check the slices read by the EA ladder.
        test_df_orbital_slices: Compare sliced DF reconstructions with full
            complex tensor contractions.
    """
    def test_context_and_lazy_block_properties(self):
        r"""Check AO2MO context metadata and nonexpanding integral block
        access.

        Compare a lazy ovvv block and orbital slice with an independent DF
        contraction, reject unsupported tensor-wide indices, and verify that
        reading either lazy accessor leaves raw ovvv/vvvv unmaterialized.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADCIP(mf)
        mykradc.method = 'adc(3)'
        context = kadc_ao2mo.AO2MO(mykradc, 'density_fit')
        self.assertEqual(context.nkpts, len(mykradc.kpts))
        self.assertIs(context.mo_coeff, mykradc.padded_coeff)
        with context() as eris:
            self.assertIsNone(eris.ovvv)
            self.assertIsNone(eris.vvvv)
            direct = np.einsum(
                'Lpq,Lrs->pqrs',
                eris.Lov[0, 0], eris.Lvv[0, 0]) / mykradc.nkpts
            with np.printoptions(precision=3):
                preview = repr(eris.ovvv_kpoint_normalized)
                self.assertIn(np.array2string(direct), preview)
                self.assertIn('block[0, 0, 0]', preview)
            np.testing.assert_allclose(
                eris.ovvv_kpoint_normalized[0, 0, 0], direct)
            np.testing.assert_allclose(
                eris.ovvv_kpoint_normalized[0, 0, 0, :1], direct[:1])
            with self.assertRaises(IndexError):
                eris.ovvv_kpoint_normalized[:]
            with self.assertRaises(IndexError):
                eris.vvvv_kpoint_normalized[mykradc.nkpts, 0, 0]
            # Reading either lazy property must not materialize its raw tensor.
            eris.vvvv_kpoint_normalized[0, 0, 0]
            self.assertIsNone(eris.ovvv)
            self.assertIsNone(eris.vvvv)

    def test_incore_memory_selection(self):
        """Check the extra virtual quartet storage and FFT grid workspace."""
        from pyscf.pbc.df import FFTDF

        solver = adc.KRADCEA(mf)
        solver.with_df = FFTDF(mf.cell, mf.kpts)
        solver.with_df.mesh = [9, 9, 9]
        ctx = kadc_ao2mo.AO2MO(solver)
        required = ctx._incore_memory_mb()
        virtual_mb = ctx.dtype.itemsize * ctx.nkpts**3 * ctx.nvir**4 / 1e6
        solver.max_memory = required + virtual_mb / 2
        with mock.patch.object(kadc_ao2mo.lib, 'current_memory',
                               return_value=(0., 0.)):
            self.assertEqual(kadc_ao2mo.AO2MO(solver).backend, 'incore')
            solver.method = 'adc(2)-x'
            ctx = kadc_ao2mo.AO2MO(solver)
            self.assertAlmostEqual(ctx._incore_memory_mb(),
                                   required + virtual_mb)
            self.assertEqual(ctx.backend, 'outcore')
            solver.with_df.mesh = [7, 7, 7]
            self.assertEqual(kadc_ao2mo.AO2MO(solver).backend, 'incore')

    def test_chunked_ladder_reads(self):
        """Request one disk slice per chunk and compare with a dense ladder."""
        nvir = 5
        rng = np.random.default_rng(8)
        tensor = rng.normal(size=(1, 1, 1) + (nvir,) * 4)
        disk = mock.MagicMock()
        disk.__getitem__.side_effect = tensor.__getitem__
        helper = SimpleNamespace(kconserv=np.zeros((1, 1, 1), dtype=int))
        model = SimpleNamespace(nkpts=1, mo_coeff=[np.eye(nvir)],
                                khelper=helper, chunk_size=2)
        r2 = rng.normal(size=(1, nvir, nvir))
        actual = adc.KRADCEA.ea_contract_r_vvvv(model, r2, disk, 0, 0, 0)
        np.testing.assert_allclose(actual, np.einsum(
            'abcd,icd->iab', tensor[0, 0, 0], r2))
        expected = [mock.call((0, 0, 0, slice(a, a + 2)))
                    for a in range(0, nvir, 2)]
        self.assertEqual(disk.__getitem__.call_args_list, expected)

    def test_df_orbital_slices(self):
        r"""Compare sliced DF reconstructions with full complex tensor
        contractions.

        Use independent random complex Lov/Lvv factors and compare occupied
        and virtual slice reconstruction with direct NumPy einsum results.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        rng = np.random.default_rng(9)
        Lov = rng.normal(size=(5, 3, 2)) + 1j * rng.normal(size=(5, 3, 2))
        Lvv = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
        expected = np.einsum('Lpq,Lrs->pqrs', Lov, Lvv)
        np.testing.assert_allclose(
            kadc_ao2mo._ChemistsERIs.ovvv_df(Lov, Lvv, 1, 1), expected[1:2])
        expected = np.einsum('Lpq,Lrs->prqs', Lvv, Lvv)
        np.testing.assert_allclose(
            kadc_ao2mo._ChemistsERIs.vvvv_df(Lvv, Lvv, 1, 1), expected[1:2])


class LoggingAPI(unittest.TestCase):
    r"""Check table width, field layout and logger integration.

    Use in-memory stdout to inspect table footers and integral/reference
    calculation summaries without relying on terminal formatting.

    Run with ``python -m unittest test.test_kadc_api.LoggingAPI``.

    Methods:
        test_tables_and_global_width: Check shared table formatting and
            calculation-summary logging.
    """
    def test_tables_and_global_width(self):
        r"""Check shared table formatting and calculation-summary logging.

        Patch the global width, test short-column padding and comma-separated
        rows, then capture a reference/excitation run to verify both common
        End of <title> footers are present.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        with mock.patch.object(adc_lib, 'LINE_WIDTH', 50):
            lines = adc_lib.table_lines({'k': [0, 1], 'E': ['1.0']}, 'ADC')
        self.assertTrue(all((len(line) == 50 for line in lines)))
        self.assertIn('End of ADC', lines[-1])
        self.assertIn(',', lines[1])
        mykradc = adc.KRADCIP(mf)
        mykradc.stdout, mykradc.verbose = io.StringIO(), 4
        mykradc.ground_state_energy()
        mykradc.kernel(1, kptlist=[0])
        output = mykradc.stdout.getvalue()
        self.assertIn('End of ADC orbital information', output)
        self.assertIn('End of ADC calculation summary', output)


if __name__ == '__main__':
    unittest.main()
