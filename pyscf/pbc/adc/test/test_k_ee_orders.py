"""Molecular-limit tests for all neutral ADC orders and transition densities."""
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf import adc as molecular_adc
from pyscf.adc import radc_ee, radc_amplitudes
from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc.kadc_rhf_ee import RADCEE
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools.pbc import super_cell


def setUpModule():
    global mf
    cell = gto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.atom = 'He 0 0 0; He 1.685068664391 1.685068664391 1.685068664391'
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 3.370137329
    cell.build()
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 1]), exxdiv=None).density_fit()
    mf.kernel()


class KnownValues(unittest.TestCase):
    def test_molecular_limit(self):
        for method in ('adc(2)', 'adc(2)-x', 'adc(3)'):
            with self.subTest(method=method):
                ground = adc.KRADC(mf)
                ground.type_ = 'ee'
                ground.method = method
                ground.approx_trans_moments = False
                ground.ground_state_energy()
                eris = ground.ao2mo()
                solver = RADCEE(ground)
                reference = SimpleNamespace(
                    method=method, _nocc=solver.nocc, _nvir=solver.nvir,
                    mo_energy=np.asarray(ground.mo_energy)[0].real,
                    t1=[None if x is None else np.asarray(x)[0].real
                        for x in ground.t1],
                    t2=[None if x is None else np.asarray(x)[0, 0, 0].real
                        for x in ground.t2])
                ref_eris = SimpleNamespace(**{
                    name: np.asarray(getattr(eris, name))[0, 0, 0].real
                    for name in ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo')})
                ref_eris.ovvv = eris.ovvv_blocks[0, 0, 0].real[
                    :, :, *np.tril_indices(solver.nvir)]
                ref_eris.vvvv = eris.vvvv_blocks[0, 0, 0].real.reshape(
                    solver.nvir**2, solver.nvir**2)
                reference.transform_integrals = lambda: ref_eris
                reference.verbose = 0
                reference.stdout = ground.stdout
                reference.method_type = 'ee'
                reference.approx_trans_moments = False
                ref_t1, ref_t2, _ = radc_amplitudes.compute_amplitudes(
                    reference, ref_eris)
                for actual_amp, expected_amp in zip(reference.t1, ref_t1):
                    if expected_amp is not None:
                        np.testing.assert_allclose(actual_amp, expected_amp,
                                                   atol=1e-12)
                for actual_amp, expected_amp in zip(reference.t2, ref_t2):
                    if expected_amp is not None:
                        np.testing.assert_allclose(actual_amp, expected_amp,
                                                   atol=1e-12)
                expected = radc_ee.get_imds(reference, ref_eris)
                actual = solver.get_imds(eris)
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                ref_action = radc_ee.matvec(reference, expected, ref_eris)
                action = solver.matvec(0, actual, eris)
                rng = np.random.default_rng(20)
                vector = rng.normal(size=solver.vector_size())
                np.testing.assert_allclose(action(vector), ref_action(vector),
                                           atol=1e-12)
                vectors = rng.normal(size=(solver.vector_size(), 2))
                for root in range(2):
                    block = vectors[solver.n_singles:, root].reshape(
                        solver.nocc, solver.nocc, solver.nvir, solver.nvir)
                    block[:] = block + block.transpose(1, 0, 3, 2)
                reference.U = vectors.copy()
                expected_t = radc_ee.get_trans_moments(reference).reshape(2, -1)
                actual_t = solver.get_trans_moments(0, vectors)
                np.testing.assert_allclose(actual_t, expected_t, atol=1e-12)
                eris.close()

    def test_supercell_density_response(self):
        self._check_supercell([1, 1, 2])

    def test_complex_kpoint_density_response(self):
        self._check_supercell([1, 1, 3])

    def _check_supercell(self, mesh):
        kpoints = mf.cell.make_kpts(mesh)
        kmf = scf.KRHF(mf.cell, kpoints, exxdiv=None).density_fit().run()
        scell = super_cell(mf.cell, mesh)
        smf = scf.RHF(scell, exxdiv=None).density_fit().run()
        for method in ('adc(2)', 'adc(2)-x', 'adc(3)'):
            with self.subTest(method=method):
                periodic = adc.KRADC(kmf)
                periodic.type_ = 'ee'
                periodic.method = method
                periodic.approx_trans_moments = False
                molecular = molecular_adc.RADC(smf)
                molecular.method_type = 'ee'
                molecular.method = method
                periodic.conv_tol = molecular.conv_tol = 1e-10
                periodic.tol_residual = molecular.tol_residual = 1e-8
                ke, _, kp, _ = periodic.kernel(4)
                me, _, _, mx = molecular.kernel(4 * len(kpoints))
                cutoff = min(ke.max(axis=1).min(), me.max()) - 1e-6
                np.testing.assert_allclose(
                    np.sort(ke[ke < cutoff]), np.sort(me[me < cutoff]),
                    atol=1e-6, rtol=0)
                for shift in range(len(kpoints)):
                    q = -kpoints[shift]
                    ao = ft_ao.ft_aopair(scell, q.reshape(1, 3))[0]
                    op = smf.mo_coeff.conj().T @ ao @ smf.mo_coeff
                    strength = (2 / len(kpoints)) * abs(
                        np.einsum('pq,npq->n', op, mx))**2
                    for energy in ke[shift][ke[shift] < cutoff]:
                        expected = strength[abs(me - energy) < 1e-6].sum()
                        actual = kp[shift][
                            abs(ke[shift] - energy) < 1e-6].sum()
                        np.testing.assert_allclose(actual, expected,
                                                   atol=1e-7, rtol=1e-5)

    def test_number_response(self):
        ground = adc.KRADC(mf)
        ground.type_ = 'ee'
        energies, vectors, weights, densities = ground.kernel(3)
        self.assertEqual(densities.shape, (1, 3, 1, ground.nmo, ground.nmo))
        np.testing.assert_allclose(weights, 0, atol=1e-12)
        np.testing.assert_array_equal(ground.T.reshape(densities.shape), densities)
        np.testing.assert_array_equal(ground.U, vectors)
        np.testing.assert_array_equal(ground.E, energies)


if __name__ == '__main__':
    unittest.main()
