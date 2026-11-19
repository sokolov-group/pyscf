"""Integral-container and backend consistency tests."""
import unittest

import numpy as np

from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs


class KnownValues(unittest.TestCase):
    r"""Check integral backend equivalence and owning-container behavior.

    Each test constructs its own helium reference. Charged comparisons cover
    all ADC orders; direct block comparisons use independent complex orbital
    phases on a three-k mesh and force the low-memory outcore path.

    Run with ``python -m unittest test.test_kadc_integrals.KnownValues``.

    Methods:
        test_charged_backend_selection: Compare charged energies and strengths
            across all integral backends.
        test_backends: Compare complex ERI blocks from DF, incore and forced-
            low-memory outcore.
    """
    def test_charged_backend_selection(self):
        r"""Compare charged energies and strengths across all integral
        backends.

        For IP/EA and every supported ADC order, compare density_fit, incore
        and outcore results on the two-k helium reference to 1e-10.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        cell = gto.M(
            atom='He 0 0 0',
            a=np.eye(3) * 3,
            basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            verbose=0)
        mf = scf.KRHF(
            cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit().run()
        for solver_type in (adc.KRADCIP, adc.KRADCEA):
            for method in ('adc(2)', 'adc(2)-x', 'adc(3)'):
                with self.subTest(sector=solver_type.__name__, method=method):
                    results = []
                    for backend in ('density_fit', 'incore', 'outcore'):
                        mykradc = solver_type(mf)
                        mykradc.method = method
                        mykradc.integral_backend = backend
                        results.append(mykradc.kernel(1, kptlist=[0]))
                    # Compare both explicit transforms with density fitting.
                    for energies, _, strengths, _ in results[1:]:
                        np.testing.assert_allclose(
                            energies, results[0][0], atol=1e-10)
                        np.testing.assert_allclose(
                            strengths, results[0][2], atol=1e-10)

    def test_backends(self):
        r"""Compare complex ERI blocks from DF, incore and forced-low-memory
        outcore.

        Apply independent MO phases on a three-k mesh and compare all stored
        occupied blocks and every lazy ovvv/vvvv k block to 1e-11. Close
        integral
        owners twice to exercise idempotent cleanup even after a failed
        assertion.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        cell = gto.Cell()
        cell.verbose = 0
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 3
        cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
        cell.build()
        mf = scf.KRHF(
            cell, cell.make_kpts([1, 1, 3]), exxdiv=None).density_fit().run()
        mykradc = adc.KRADCEE(mf)
        mykradc.method = 'adc(3)'
        # Complex MO phases catch conjugation/storage ordering mistakes.
        mykradc.mo_coeff = np.asarray(mykradc.mo_coeff, dtype=complex)
        phase = np.exp(1j * np.array([[0.3, 0.6], [0.8, -0.2], [-0.7, 0.2]]))
        mykradc.mo_coeff *= phase[:, None, :]
        integrals = []
        try:
            for backend in ('density_fit', 'incore', 'outcore'):
                mykradc.integral_backend = backend
                mykradc.max_memory = 0  # Exercise the low-memory vvvv path.
                eris = mykradc.ao2mo()
                self.assertIsInstance(eris, _ChemistsERIs)
                integrals.append(eris)
            reference = integrals[0]
            for candidate in integrals[1:]:
                for name in ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo'):
                    np.testing.assert_allclose(
                        np.asarray(getattr(candidate, name)),
                        np.asarray(getattr(reference, name)),
                        atol=1e-11)
                for kp in range(3):
                    for kq in range(3):
                        for kr in range(3):
                            np.testing.assert_allclose(
                                candidate.ovvv_kpoint_normalized[kp, kq, kr],
                                reference.ovvv_kpoint_normalized[kp, kq, kr],
                                atol=1e-11)
                            np.testing.assert_allclose(
                                candidate.vvvv_kpoint_normalized[kp, kq, kr],
                                reference.vvvv_kpoint_normalized[kp, kq, kr],
                                atol=1e-11)
        finally:
            for eris in integrals:
                eris.close()
                eris.close()


if __name__ == '__main__':
    unittest.main()
