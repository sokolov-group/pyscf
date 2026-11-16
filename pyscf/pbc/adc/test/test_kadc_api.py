"""Contracts for the consolidated periodic ADC API."""
from unittest import mock
import io
import unittest
import warnings

import numpy as np

from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc import kadc_ao2mo, kadc_rhf, kadc_rhf_ee, lib as adc_lib


def setUpModule():
    global mf
    cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
                 basis=[[0, (1., 1.)], [0, (.5, 1.)]], verbose=0)
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]),
                  exxdiv=None).density_fit().run()


class DriverAPI(unittest.TestCase):
    def test_ground_state_and_amplitudes(self):
        driver = adc.KRADC(mf)
        self.assertIs(type(driver), kadc_rhf.KRADC)
        energy = driver.ground_state_energy()
        self.assertIsInstance(energy, float)
        self.assertIsNotNone(driver.t2[0])
        with driver.ao2mo() as eris:
            self.assertIsNone(driver.amplitudes(eris))
            self.assertAlmostEqual(driver.correlation_energy(eris), energy, 12)
            amplitude = np.asarray(driver.t2[0]).copy()
            driver.correlation_energy(eris)
            np.testing.assert_array_equal(driver.t2[0], amplitude)
        driver.if_corr = False
        self.assertEqual(driver.ground_state_energy(), 0.)
        self.assertIsNotNone(driver.t2[0])

    def test_reference_only_and_stale_results(self):
        for sector in ('ip', 'ea', 'ee'):
            with self.subTest(sector=sector):
                driver = adc.KRADC(mf)
                driver.type_ = sector
                expected = driver.ground_state_energy()
                driver.kernel(1, kptlist=[0])
                self.assertIsNotNone(driver.E)
                with mock.patch.object(driver, f'{sector}_adc',
                                       side_effect=AssertionError('solver called')):
                    actual = driver.kernel(0, kptlist=[])
                self.assertAlmostEqual(actual, expected, 12)
                for field in ('E', 'U', 'P', 'X', 'T', 'converged', '_adc_es'):
                    self.assertIsNone(getattr(driver, field))
                self.assertIsNotNone(driver.t2[0])
                self.assertAlmostEqual(driver.kernel(-2), expected, 12)

    def test_fno_reference_only(self):
        driver = adc.KRADC2FNO(mf)
        energy = driver.kernel(0, pct_occ=0.9)
        self.assertIsInstance(energy, float)
        self.assertEqual(energy, driver.e_corr_fno)
        self.assertIsNone(driver.E)
        self.assertIsNone(driver._adc_es)

    def test_direct_neutral_reference_only(self):
        driver = adc.KRADC(mf)
        driver.type_ = 'ee'
        expected = driver.ground_state_energy()
        solver = kadc_rhf_ee.RADCEE(driver)
        with mock.patch.object(solver, 'gen_matvec',
                               side_effect=AssertionError('matvec built')):
            self.assertAlmostEqual(solver.kernel(0), expected, 12)
        np.testing.assert_array_equal(solver.t2[0], driver.t2[0])
        self.assertIsNone(solver.E)

    def test_padding_chunking_and_legacy_names(self):
        driver = adc.KRADC(mf)
        self.assertIs(kadc_rhf.RADC, kadc_rhf.KRADC)
        driver.method_type = 'EE'
        self.assertEqual(driver.type_, 'EE')
        driver.type_ = 'ip'
        self.assertEqual(driver.method_type, 'ip')
        coeff = driver.padded_coeff
        self.assertIs(coeff, driver.padded_coeff)
        self.assertEqual(driver.e_occ.shape, (driver.nkpts, driver.nocc))
        self.assertEqual(driver.e_vir.shape, (driver.nkpts, driver.nvir))
        with mock.patch.object(kadc_ao2mo, 'chunk_size', return_value=1) as size:
            driver.chunk_size = None
            self.assertEqual(driver.chunk_size, 1)
            self.assertEqual(driver.chunk_size, 1)
            size.assert_called_once()
        driver.chunk_size = 2
        driver.ground_state_energy()
        self.assertEqual(driver.chunk_size, 2)
        driver.chnk_size = None
        self.assertGreaterEqual(driver.chunk_size, 1)
        for invalid in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                driver.chunk_size = invalid
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always', DeprecationWarning)
            result = driver.kernel_gs()
        self.assertTrue(any(issubclass(w.category, DeprecationWarning)
                            for w in recorded))
        self.assertEqual(result, (driver.e_corr, driver.t1, driver.t2))

    def test_input_validation(self):
        driver = adc.KRADC(mf)
        with self.assertRaises(NotImplementedError):
            driver.method = 'adc(4)'
            driver.ground_state_energy()
        driver.method = 'ADC(2)'
        self.assertIsInstance(driver.ground_state_energy(), float)
        for kptlist in ([], [-1], [driver.nkpts], ['0']):
            with self.assertRaises(ValueError):
                driver.kernel(1, kptlist=kptlist)
        with self.assertRaises(TypeError):
            driver.kernel(1.5)


class IntegralAPI(unittest.TestCase):
    def test_context_and_lazy_block_properties(self):
        driver = adc.KRADC(mf)
        driver.type_ = 'ee'
        driver.method = 'adc(3)'
        context = kadc_ao2mo.AO2MO(driver, 'density_fit')
        self.assertEqual(context.nkpts, len(driver.kpts))
        self.assertIs(context.mo_coeff, driver.padded_coeff)
        with context() as eris:
            self.assertIsNone(eris.ovvv)
            self.assertIsNone(eris.vvvv)
            direct = np.einsum('Lpq,Lrs->pqrs',
                               eris.Lov[0, 0], eris.Lvv[0, 0]) / driver.nkpts
            np.testing.assert_allclose(eris.ovvv_blocks[0, 0, 0], direct)
            np.testing.assert_allclose(eris.ovvv_blocks[0, 0, 0, :1],
                                       direct[:1])
            with self.assertRaises(IndexError):
                eris.ovvv_blocks[:]
            with self.assertRaises(IndexError):
                eris.vvvv_blocks[driver.nkpts, 0, 0]
            # Reading either lazy property must not materialize its raw tensor.
            eris.vvvv_blocks[0, 0, 0]
            self.assertIsNone(eris.ovvv)
            self.assertIsNone(eris.vvvv)

    def test_df_orbital_slices(self):
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
    def test_tables_and_global_width(self):
        with mock.patch.object(adc_lib, 'LINE_WIDTH', 50):
            lines = adc_lib.table_lines({'k': [0, 1], 'E': ['1.0']}, 'ADC')
        self.assertTrue(all(len(line) == 50 for line in lines))
        self.assertIn('End of ADC', lines[-1])
        self.assertIn(',', lines[1])
        driver = adc.KRADC(mf)
        driver.stdout, driver.verbose = io.StringIO(), 4
        driver.ground_state_energy()
        driver.kernel(1, kptlist=[0])
        output = driver.stdout.getvalue()
        self.assertIn('End of ADC orbital information', output)
        self.assertIn('End of ADC calculation summary', output)


if __name__ == '__main__':
    unittest.main()
