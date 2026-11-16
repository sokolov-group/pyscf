"""Regression checks for periodic singlet EE-ADC(2)."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from pyscf.adc import radc_ee
from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc.kadc_rhf_ee import RADCEE


def setUpModule():
    global solver, eris
    cell = gto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.atom = 'He 0 0 0; He 1.685068664391 1.685068664391 1.685068664391'
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 3.370137329
    cell.build()
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
    mf.kernel()
    ground = adc.KRADC(mf)
    ground.type_ = 'ee'
    ground.compute_properties = False
    ground.ground_state_energy()
    eris = ground.ao2mo()
    solver = RADCEE(ground)


class KnownValues(unittest.TestCase):
    def test_energies(self):
        energies, vectors, _, _ = solver.kernel(4, eris=eris)
        np.testing.assert_allclose(
            np.sort(energies.ravel())[:6],
            [1.40629, 1.49977, 1.67110, 1.73389, 1.75824, 1.82029],
            atol=5e-6, rtol=0)
        for shift in range(solver.nkpts):
            action = solver.matvec(shift, eris=eris)
            for energy, vector in zip(energies[shift], vectors[shift]):
                residual = np.linalg.norm(action(vector) - energy * vector)
                self.assertLess(residual, solver.tol_residual)

    def test_diagonal_and_integrals(self):
        rng = np.random.default_rng(47)
        vector = (rng.normal(size=solver.vector_size())
                  + 1j * rng.normal(size=solver.vector_size()))
        # Compare the density-fitted reconstruction with stored ovvv tensors.
        stored = SimpleNamespace(**vars(eris))
        nk = solver.nkpts
        stored.ovvv = np.empty((nk,) * 3 + (solver.nocc,)
                                + (solver.nvir,) * 3, dtype=eris.Lov.dtype)
        for kp in range(nk):
            for kq in range(nk):
                for kr in range(nk):
                    ks = solver.khelper.kconserv[kp, kq, kr]
                    stored.ovvv[kp, kq, kr] = np.einsum(
                        'Lpq,Lrs->pqrs',
                        eris.Lov[kp, kq], eris.Lvv[kr, ks]) / nk
        for shift in range(nk):
            action, diagonal = solver.gen_matvec(shift, eris=eris)
            dense = np.column_stack([action(v) for v in
                                     np.eye(solver.vector_size())])
            np.testing.assert_allclose(diagonal, dense.diagonal(), atol=1e-12)
            np.testing.assert_allclose(action(vector), dense @ vector,
                                       atol=1e-12)
            np.testing.assert_allclose(
                action(vector), solver.matvec(shift, eris=stored)(vector),
                atol=1e-12)

    def test_normalization(self):
        rng = np.random.default_rng(13)
        vectors = (rng.normal(size=(solver.vector_size(), 2))
                   + 1j * rng.normal(size=(solver.vector_size(), 2)))
        original = vectors.copy()
        for shift in range(solver.nkpts):
            normalized = solver.renormalize_eigenvectors(shift, vectors, 2)
            np.testing.assert_allclose(
                solver.renormalize_eigenvectors(shift, normalized, 2),
                normalized, atol=1e-14)
            phase = np.exp(0.37j)
            np.testing.assert_allclose(
                solver.renormalize_eigenvectors(shift, phase * vectors, 2),
                phase * normalized, atol=1e-14)
        np.testing.assert_array_equal(vectors, original)
        with self.assertRaises(ValueError):
            solver.renormalize_eigenvectors(0, vectors * 0, 2)

    def test_molecular_normalization_limit(self):
        gamma = copy.copy(solver)
        gamma.kpts = solver.kpts[:1]
        rng = np.random.default_rng(19)
        vectors = rng.normal(size=(gamma.vector_size(), 2))
        # Physical singlet doubles obey simultaneous hole/particle exchange.
        for root in range(2):
            block = vectors[gamma.n_singles:, root].reshape(
                gamma.nocc, gamma.nocc, gamma.nvir, gamma.nvir)
            block[:] = block + block.transpose(1, 0, 3, 2)
        molecular = SimpleNamespace(U=vectors.copy(), _nocc=gamma.nocc,
                                    _nvir=gamma.nvir)
        expected = radc_ee.renormalize_eigenvectors(molecular, 2)
        np.testing.assert_allclose(
            gamma.renormalize_eigenvectors(0, vectors, 2),
            expected, atol=1e-14)

    def test_guesses_and_properties(self):
        guesses = np.asarray(solver.get_init_guess(2, kshift=1))
        self.assertEqual(guesses.shape, (2, solver.vector_size()))
        np.testing.assert_array_equal(
            solver.get_init_guess(2, type='read', ini=guesses), guesses)
        with self.assertRaises(ValueError):
            solver.get_init_guess(2, type='read', ini=guesses * 0)
        weights, densities = solver.get_properties(0, guesses.T, 2)
        self.assertEqual(densities.shape, (2, solver.nkpts,
                                          solver.nmo, solver.nmo))
        self.assertTrue(np.all(weights >= 0))


if __name__ == '__main__':
    unittest.main()
