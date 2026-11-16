"""Contracts for the consolidated periodic ADC API."""
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
import ast
import doctest
import io
import unittest
import warnings

import numpy as np

from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc import (kadc_ao2mo, kadc_rhf, kadc_rhf_ea, kadc_rhf_ee,
                           kadc_rhf_ip, lib as adc_lib)
from pyscf.pbc.scf.hf import INVALID_ORBITAL_ENERGY


def setUpModule():
    r"""Prepare the small two-k helium reference used by the API and doc tests.

    Returns:
        None: Store mf as a density-fitted KRHF reference with two s basis
        functions, one occupied and one virtual MO at each k point. This is
        the same reproducible system constructed by the KRADC class example.
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
    root clearing, cached data, guess routing, aliases and AO densities.

    Run with ``python -m unittest test.test_kadc_api.DriverAPI``.
    """
    def test_periodic_sector_class_names(self):
        """Check canonical k-point class names and earlier import aliases.

        Returns:
            None: Verify each new KRADCIP/KRADCEA/KRADCEE class name and
            excitation sector. Earlier RADC spellings must remain aliases
            of the same class, preserving existing importing applications.
        """
        mykradc = adc.KRADC(mf)
        for module, suffix in ((kadc_rhf_ip, 'IP'), (kadc_rhf_ea, 'EA'),
                               (kadc_rhf_ee, 'EE')):
            canonical = getattr(module, 'KRADC' + suffix)
            self.assertEqual(canonical.__name__, 'KRADC' + suffix)
            self.assertIs(getattr(module, 'RADC' + suffix), canonical)
            self.assertIsInstance(canonical(mykradc), kadc_rhf.KRADC)
            self.assertEqual(canonical(mykradc).type_, suffix.lower())

    def test_transform_integrals_deprecation(self):
        r"""Check the earlier integral method warning and backend forwarding.

        Returns:
            None: Preserve the returned integral owner and backend selection
            while directing callers to ao2mo() with DeprecationWarning.
        """
        mykradc = adc.KRADC(mf)
        with mock.patch.object(
            mykradc, 'ao2mo', return_value=mock.sentinel.eris) as transform:
            with self.assertWarns(DeprecationWarning) as warning:
                eris = mykradc.transform_integrals('outcore')
        self.assertIs(eris, mock.sentinel.eris)
        transform.assert_called_once_with('outcore')
        self.assertEqual(warning.filename, __file__)

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
        mykradc = adc.KRADC(mf)
        self.assertIs(type(mykradc), kadc_rhf.KRADC)
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
        for sector in ('ip', 'ea', 'ee'):
            with self.subTest(sector=sector):
                mykradc = adc.KRADC(mf)
                mykradc.type_ = sector
                expected = mykradc.ground_state_energy()
                mykradc.kernel(1, kptlist=[0])
                self.assertIsNotNone(mykradc.E)
                with mock.patch.object(
                    mykradc,
                    f'{sector}_adc',
                    side_effect=AssertionError('solver called')):
                    actual = mykradc.kernel(0, kptlist=[])
                self.assertAlmostEqual(actual, expected, 12)
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
        mykradc = adc.KRADC(mf)
        mykradc.type_ = 'ee'
        expected = mykradc.ground_state_energy()
        mykradcee = kadc_rhf_ee.KRADCEE(mykradc)
        with mock.patch.object(
            mykradcee, 'gen_matvec', side_effect=AssertionError('matvec built')
        ):
            self.assertAlmostEqual(mykradcee.kernel(0), expected, 12)
        np.testing.assert_array_equal(mykradcee.t2[0], mykradc.t2[0])
        self.assertIsNone(mykradcee.E)

    def test_padding_chunking_and_legacy_names(self):
        r"""Check cached orbital/chunk properties and earlier API spellings.

        Verify active energy shapes, lazy automatic chunk estimation, retained
        manual overrides, invalid chunk rejection, aliases and deprecated
        ground-state tuple returns with their warning.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADC(mf)
        self.assertIs(kadc_rhf.RADC, kadc_rhf.KRADC)
        mykradc.method_type = 'EE'
        self.assertEqual(mykradc.type_, 'EE')
        mykradc.type_ = 'ip'
        self.assertEqual(mykradc.method_type, 'ip')
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
        mykradc.chnk_size = None
        self.assertGreaterEqual(mykradc.chunk_size, 1)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                mykradc.chunk_size = invalid
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always', DeprecationWarning)
            result = mykradc.kernel_gs()
        self.assertTrue(
            any((issubclass(w.category, DeprecationWarning) for w in recorded))
        )
        self.assertEqual(result, (mykradc.e_corr, mykradc.t1, mykradc.t2))

    def test_input_validation(self):
        r"""Check method normalization and invalid root/sector inputs.

        Reject unsupported ADC orders, accept uppercase supported methods,
        and reject empty/out-of-range/noninteger sector lists and noninteger
        root counts for an excitation calculation.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADC(mf)
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
        r"""Check that every sector kernel bypasses the deprecated guess
        adapter.

        Mock get_init_guess to fail and observe protected helper routing for
        unit and supplied guesses. Reusing converged root vectors must
        reproduce
        the original roots and pass type_=read to the helper.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADC(mf)
        mykradc.compute_properties = False
        mykradc.ground_state_energy()
        for solver_type in (kadc_rhf_ip.KRADCIP, kadc_rhf_ea.KRADCEA,
                            kadc_rhf_ee.KRADCEE):
            mykradc_es = solver_type(mykradc)
            helper = ('_neutral_init_guess' if mykradc_es.type_ == 'ee'
                      else '_charged_init_guess')
            with (
                self.subTest(sector=mykradc_es.type_),
                mock.patch.object(
                    mykradc_es,
                    'get_init_guess',
                    side_effect=AssertionError('deprecated')),
                mock.patch.object(
                    mykradc_es, helper, wraps=getattr(mykradc_es, helper)
                ) as guesses,
            ):
                energies, vectors, _, _ = mykradc_es.kernel(2, kptlist=[0])
                from_input, _, _, _ = mykradc_es.kernel(
                    2, guess=vectors, kptlist=[0])
                np.testing.assert_allclose(from_input, energies, atol=1e-08)
                self.assertEqual(guesses.call_count, 2)
                self.assertEqual(guesses.call_args.kwargs['type_'], 'read')

    def test_deprecated_initial_guess_adapter(self):
        r"""Check warning and keyword compatibility of the earlier guess
        interface.

        For IP/EA/EE, accept supplied vectors with type_ or legacy type,
        preserve
        the vector contents, and reject simultaneous spellings or unknown keys.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        mykradc = adc.KRADC(mf)
        mykradc.ground_state_energy()
        for solver_type in (kadc_rhf_ip.KRADCIP, kadc_rhf_ea.KRADCEA,
                            kadc_rhf_ee.KRADCEE):
            mykradc_es = solver_type(mykradc)
            supplied = np.eye(2, mykradc_es.vector_size())
            for keyword in ('type_', 'type'):
                with (self.subTest(sector=mykradc_es.type_, keyword=keyword),
                      self.assertWarns(DeprecationWarning)):
                    guesses = mykradc_es.get_init_guess(
                        2, ini=supplied, kshift=0, **{keyword: 'read'})
                np.testing.assert_array_equal(guesses, supplied)
            with (self.assertWarns(DeprecationWarning),
                  self.assertRaisesRegex(TypeError, 'only one')):
                mykradc_es.get_init_guess(type_='read', type='read')
            with (self.assertWarns(DeprecationWarning),
                  self.assertRaisesRegex(TypeError, 'Unexpected')):
                mykradc_es.get_init_guess(unknown='read')

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
            mykradc = adc.KRADC(reference, frozen=frozen, **options)
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
    r"""Execute every package and comparison-helper worked docstring example.

    Each example block runs in an empty namespace, using only the imports and
    reference setup shown in its example. Numerical energies and tensor
    shapes are checked, including public factories, compatibility functions,
    FNO/integral workflows and the comparison-script helpers.

    Run with ``python -m unittest test.test_kadc_api.Documentation``.
    """
    def test_worked_examples(self):
        """Execute package examples and comparison-table helper examples.

        Read docstrings from the source AST and give each example block an
        empty namespace, including blocks separated by explanatory prose.
        This prevents module globals or another block from hiding missing
        imports, undefined functions or reference setup. The comparison
        helpers use the explicit loader shown in their docs.
        """
        output = io.StringIO()
        runner = doctest.DocTestRunner()
        parser = doctest.DocTestParser()
        directory = Path(kadc_rhf.__file__).parent
        paths = sorted(
            [*directory.glob('*.py'), *(directory / 'test').glob('*.py')])
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node)
                if not doc or not parser.get_examples(doc):
                    continue
                name = (f'{path.name}:{node.body[0].lineno}:'
                        f'{getattr(node, 'name', 'module')}')
                blocks = []
                current = []
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
        mykradc = adc.KRADC(mf)
        mykradc.type_ = 'ee'
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
            np.testing.assert_allclose(eris.ovvv_blocks[0, 0, 0], direct)
            np.testing.assert_allclose(
                eris.ovvv_blocks[0, 0, 0, :1], direct[:1])
            with self.assertRaises(IndexError):
                eris.ovvv_blocks[:]
            with self.assertRaises(IndexError):
                eris.vvvv_blocks[mykradc.nkpts, 0, 0]
            # Reading either lazy property must not materialize its raw tensor.
            eris.vvvv_blocks[0, 0, 0]
            self.assertIsNone(eris.ovvv)
            self.assertIsNone(eris.vvvv)

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
        mykradc = adc.KRADC(mf)
        mykradc.stdout, mykradc.verbose = io.StringIO(), 4
        mykradc.ground_state_energy()
        mykradc.kernel(1, kptlist=[0])
        output = mykradc.stdout.getvalue()
        self.assertIn('End of ADC orbital information', output)
        self.assertIn('End of ADC calculation summary', output)


if __name__ == '__main__':
    unittest.main()
