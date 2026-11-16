"""Integral-container and backend consistency tests."""
import unittest

import numpy as np

from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc.kadc_ao2mo import _ChemistsERIs


class KnownValues(unittest.TestCase):
    def test_charged_backend_selection(self):
        cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
                     basis=[[0, (1., 1.)], [0, (.5, 1.)]], verbose=0)
        mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]),
                      exxdiv=None).density_fit().run()
        for sector in ('ip', 'ea'):
            for method in ('adc(2)', 'adc(2)-x', 'adc(3)'):
                with self.subTest(sector=sector, method=method):
                    results = []
                    for backend in ('density_fit', 'incore', 'outcore'):
                        ground = adc.KRADC(mf)
                        ground.method = method
                        ground.type_ = sector
                        ground.integral_backend = backend
                        results.append(ground.kernel(1, kptlist=[0]))
                    for energies, _, strengths, _ in results[1:]:
                        np.testing.assert_allclose(
                            energies, results[0][0], atol=1e-10)
                        np.testing.assert_allclose(
                            strengths, results[0][2], atol=1e-10)

    def test_backends(self):
        cell = gto.Cell()
        cell.verbose = 0
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 3
        cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
        cell.build()
        mf = scf.KRHF(cell, cell.make_kpts([1, 1, 3]),
                      exxdiv=None).density_fit().run()
        ground = adc.KRADC(mf)
        ground.method = 'adc(3)'
        ground.type_ = 'ee'
        # Complex MO phases catch conjugation/storage ordering mistakes.
        ground.mo_coeff = np.asarray(ground.mo_coeff, dtype=complex)
        phase = np.exp(1j * np.array([[0.3, 0.6], [0.8, -0.2], [-0.7, 0.2]]))
        ground.mo_coeff *= phase[:, None, :]
        integrals = []
        try:
            for backend in ('density_fit', 'incore', 'outcore'):
                ground.integral_backend = backend
                ground.max_memory = 0  # Exercise the low-memory vvvv path.
                eris = ground.ao2mo()
                self.assertIsInstance(eris, _ChemistsERIs)
                integrals.append(eris)
            for candidate in integrals[1:]:
                for name in ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo'):
                    np.testing.assert_allclose(
                        np.asarray(getattr(candidate, name)),
                        np.asarray(getattr(integrals[0], name)), atol=1e-11)
                for kp in range(3):
                    for kq in range(3):
                        for kr in range(3):
                            np.testing.assert_allclose(
                                candidate.ovvv_blocks[kp, kq, kr],
                                integrals[0].ovvv_blocks[kp, kq, kr], atol=1e-11)
                            np.testing.assert_allclose(
                                candidate.vvvv_blocks[kp, kq, kr],
                                integrals[0].vvvv_blocks[kp, kq, kr], atol=1e-11)
        finally:
            for eris in integrals:
                eris.close()
                eris.close()


if __name__ == '__main__':
    unittest.main()
