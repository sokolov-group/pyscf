"""Regression checks for periodic singlet EE-ADC(2)."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from pyscf.adc import radc_ee
from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc.kadc_rhf import KRADCEE


def setUpModule():
    r"""Build the two-k helium EE-ADC(2) reference and shared integral
    container.

    Returns:
        None: Store a KRADCEE solver and its eris container as module globals.
        Reference amplitudes are computed once, with kernel property evaluation
        disabled so energy/action tests can reuse the same small system.
    """
    global mykradcee, eris
    cell = gto.Cell()
    cell.verbose = 0
    cell.unit = 'B'
    cell.atom = 'He 0 0 0; He 1.685068664391 1.685068664391 1.685068664391'
    cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
    cell.a = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) * 3.370137329
    cell.build()
    mf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit()
    mf.kernel()
    mykradc = adc.KRADCEE(mf)
    mykradc.compute_properties = False
    mykradc.ground_state_energy()
    eris = mykradc.ao2mo()
    mykradcee = KRADCEE(mykradc)


class KnownValues(unittest.TestCase):
    r"""Check EE-ADC(2) energies, tensor permutations and singlet conventions.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_k_ee.KnownValues``. The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.
    """
    def test_adc1_tensor_against_identity_prototype(self):
        r"""Compare the optimized first-order singles tensor with an
        independent prototype.

        Use unequal occupied/virtual counts, deterministic random complex ERIs
        and every sector. Explicit identity-tensor energy terms and einsum
        permutations must reproduce the optimized diagonal-view construction.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        # Unequal occupied/virtual dimensions catch permutation mistakes.
        model = copy.copy(mykradcee)
        model._nocc, model._nmo = 2, 5
        nk, no, nv = model.nkpts, model.nocc, model.nvir
        rng = np.random.default_rng(37)
        eo = rng.normal(size=(nk, no))
        ev = rng.normal(size=(nk, nv))
        model._padded_orbitals = (np.zeros((nk, 1, no + nv), complex), eo, ev)

        def complex_tensor(shape):
            r"""Generate a deterministic complex test tensor from the enclosing
            RNG.

            Args:
                shape (tuple[int]): Requested tensor dimensions.

            Returns:
                ndarray: Independent standard-normal real and imaginary
                components.
                No physical ERI symmetry is imposed, making axis mistakes
                observable.
            """
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        integrals = SimpleNamespace(
            ovvo=complex_tensor((nk,) * 3 + (no, nv, nv, no)),
            oovv=complex_tensor((nk,) * 3 + (no, no, nv, nv)))
        for shift in range(nk):
            with self.subTest(kshift=shift):
                ia, ai, doubles = model._momentum_maps(shift)
                expected = np.zeros((nk, nk, no, nv, no, nv), complex)
                for ki in range(nk):
                    kd = ia[ki]
                    expected[ki, ki] += np.einsum(
                        'a,ab,ij->iajb',
                        ev[kd], np.eye(nv), np.eye(no))
                    expected[ki, ki] -= np.einsum(
                        'j,ab,ij->iajb',
                        eo[ki], np.eye(nv), np.eye(no))
                    # Add the direct and exchange couplings for this sector.
                    for kj in range(nk):
                        ka = ia[kj]
                        expected[ki, kj] += 2 * np.einsum(
                            'laDi->iDla',
                            integrals.ovvo[kj, ka, kd])
                        expected[ki, kj] -= np.einsum(
                            'ilad->idla',
                            integrals.oovv[ki, kj, ka].conj())
                np.testing.assert_allclose(
                    model._get_adc1_ham(integrals, shift), expected, atol=1e-12
                )

    def test_energies(self):
        r"""Check low EE-ADC(2) energies and their Davidson residuals.

        Compare the six common low roots with stored helium energies and apply
        each sector matrix action to its returned eigenvectors. Residual norms
        must remain below the configured solver residual tolerance.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        energies, vectors, _, _ = mykradcee.kernel(4, eris=eris)
        np.testing.assert_allclose(
            np.sort(energies.ravel())[:6],
            [1.40629, 1.49977, 1.6711, 1.73389, 1.75824, 1.82029],
            atol=5e-06,
            rtol=0)
        for shift in range(mykradcee.nkpts):
            action = mykradcee.matvec(shift, eris=eris)
            for energy, vector in zip(energies[shift], vectors[shift]):
                residual = np.linalg.norm(action(vector) - energy * vector)
                self.assertLess(residual, mykradcee.tol_residual)

    def test_diagonal_and_integrals(self):
        r"""Compare the EE action, its ADC(2) diagonal and DF reconstruction.

        Assemble a small dense matrix from unit-vector actions and compare its
        diagonal and arbitrary complex-vector action with the public
        interfaces.
        Stored ovvv blocks must reproduce the DF-factor action.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        rng = np.random.default_rng(47)
        vector = (rng.normal(size=mykradcee.vector_size())
                  + 1j * rng.normal(size=mykradcee.vector_size()))
        # Compare the density-fitted reconstruction with stored ovvv tensors.
        stored = SimpleNamespace(**vars(eris))
        nk = mykradcee.nkpts
        stored.ovvv = np.empty(
            (nk,) * 3 + (mykradcee.nocc,) + (mykradcee.nvir,) * 3,
            dtype=eris.Lov.dtype)
        for kp in range(nk):
            for kq in range(nk):
                for kr in range(nk):
                    ks = mykradcee.khelper.kconserv[kp, kq, kr]
                    stored.ovvv[kp, kq, kr] = np.einsum(
                        'Lpq,Lrs->pqrs',
                        eris.Lov[kp, kq], eris.Lvv[kr, ks]) / nk
        for shift in range(nk):
            action, diagonal = mykradcee.gen_matvec(shift, eris=eris)
            # Unit-vector actions give each column of the dense matrix.
            dense = np.column_stack(
                [action(v) for v in np.eye(mykradcee.vector_size())])
            np.testing.assert_allclose(diagonal, dense.diagonal(), atol=1e-12)
            np.testing.assert_allclose(
                action(vector), dense @ vector, atol=1e-12)
            np.testing.assert_allclose(
                action(vector),
                mykradcee.matvec(shift, eris=stored)(vector),
                atol=1e-12)

    def test_normalization(self):
        r"""Check singlet normalization, phase covariance and zero-vector
        rejection.

        Random complex columns must normalize idempotently, retain their global
        phase, and leave the input unchanged. A zero-norm vector must raise
        ValueError rather than producing invalid normalized amplitudes.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        rng = np.random.default_rng(13)
        vectors = (rng.normal(size=(mykradcee.vector_size(), 2))
                   + 1j * rng.normal(size=(mykradcee.vector_size(), 2)))
        original = vectors.copy()
        for shift in range(mykradcee.nkpts):
            normalized = mykradcee.renormalize_eigenvectors(shift, vectors, 2)
            np.testing.assert_allclose(
                mykradcee.renormalize_eigenvectors(shift, normalized, 2),
                normalized,
                atol=1e-14)
            phase = np.exp(0.37j)
            np.testing.assert_allclose(
                mykradcee.renormalize_eigenvectors(shift, phase * vectors, 2),
                phase * normalized,
                atol=1e-14)
        np.testing.assert_array_equal(vectors, original)
        with self.assertRaises(ValueError):
            mykradcee.renormalize_eigenvectors(0, vectors * 0, 2)

    def test_molecular_normalization_limit(self):
        r"""Compare Gamma-only singlet normalization with molecular EE-ADC.

        Symmetrize the test doubles under simultaneous hole/particle exchange,
        then compare the periodic metric normalization with the molecular
        reference to numerical precision.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        gamma = copy.copy(mykradcee)
        gamma.kpts = mykradcee.kpts[:1]
        rng = np.random.default_rng(19)
        vectors = rng.normal(size=(gamma.vector_size(), 2))
        # Physical singlet doubles obey simultaneous hole/particle exchange.
        for root in range(2):
            block = vectors[gamma.n_singles:, root].reshape(
                gamma.nocc, gamma.nocc, gamma.nvir, gamma.nvir)
            block[:] = block + block.transpose(1, 0, 3, 2)
        molecular = SimpleNamespace(
            U=vectors.copy(), _nocc=gamma.nocc, _nvir=gamma.nvir)
        expected = radc_ee.renormalize_eigenvectors(molecular, 2)
        np.testing.assert_allclose(
            gamma.renormalize_eigenvectors(0, vectors, 2), expected, atol=1e-14
        )

    def test_guesses_and_properties(self):
        r"""Check neutral guess validation and density-property output shapes.

        Construct and reread sector guesses, reject zero supplied vectors,
        and verify nonnegative oscillator strengths and transition densities
        with the expected root/k/MO axes.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        guesses = np.asarray(mykradcee._init_guess(2, kshift=1))
        self.assertEqual(guesses.shape, (2, mykradcee.vector_size()))
        np.testing.assert_array_equal(
            mykradcee._init_guess(2, type_='read', ini=guesses),
            guesses)
        with self.assertRaises(ValueError):
            mykradcee._init_guess(2, type_='read', ini=guesses * 0)
        weights, densities = mykradcee.get_properties(
            0, guesses.T, 2, energies=[1., 2.])
        self.assertEqual(
            densities.shape, (2, mykradcee.nkpts, mykradcee.nmo, mykradcee.nmo)
        )
        self.assertTrue(np.all(weights >= 0))


if __name__ == '__main__':
    unittest.main()
