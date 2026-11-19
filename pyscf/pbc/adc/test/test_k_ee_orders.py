r"""Molecular-limit tests for neutral ADC orders and transition densities.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from pyscf import adc as molecular_adc
from pyscf.adc import radc_ee, radc_amplitudes
from pyscf.pbc import adc, gto, scf
from pyscf.pbc.adc.kadc_rhf import KRADCEE
from pyscf.pbc.df import ft_ao
from pyscf.pbc.tools.pbc import super_cell


def setUpModule():
    r"""Prepare the Gamma helium KRHF fixture for molecular and supercell
    checks.

    Returns:
        None: Store mf as a converged density-fitted k-point reference on the
        one-point mesh. Supercell tests construct larger meshes from this cell.
    """
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
    r"""Validate all three EE-ADC orders against molecular and supercell
    references.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_k_ee_orders.KnownValues``. The
    tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.

    Methods:
        test_molecular_limit: Compare all EE orders with molecular amplitudes,
            matrices and densities.
        test_gamma_oscillator_strengths: Compare four LiH strengths at Gamma.
        test_four_k_oscillator_strengths: Compare all eight example roots.
        test_supercell_oscillator_strengths: Check two-k EE energies and
            oscillator strengths against the Gamma supercell.
        test_complex_kpoint_oscillator_strengths: Check three-k EE strengths
            with complex Bloch orbitals.
        test_optical_transition: Compare a bright Gamma transition with the AO
            velocity integral.
    """
    def test_molecular_limit(self):
        r"""Compare all EE orders with molecular amplitudes, matrices and
        densities.

        For ADC(2), ADC(2)-x and ADC(3), compare full ground amplitudes,
        singles
        intermediates, arbitrary-vector actions and transition densities at
        Gamma.
        The molecular and periodic orbital conventions must agree to 1e-12.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        for method in ('adc(2)', 'adc(2)-x', 'adc(3)'):
            with self.subTest(method=method):
                mykradc = adc.KRADCEE(mf)
                mykradc.method = method
                mykradc.approx_trans_moments = False
                mykradc.ground_state_energy()
                eris = mykradc.ao2mo()
                mykradcee = KRADCEE(mykradc)
                # Strip the Gamma indices to build the molecular reference.
                reference = SimpleNamespace(
                    method=method, _nocc=mykradcee.nocc, _nvir=mykradcee.nvir,
                    mo_energy=np.asarray(mykradc.mo_energy)[0].real,
                    t1=[None if x is None else np.asarray(x)[0].real
                        for x in mykradc.t1],
                    t2=[None if x is None else np.asarray(x)[0, 0, 0].real
                        for x in mykradc.t2])
                ref_eris = SimpleNamespace(**{
                    name: np.asarray(getattr(eris, name))[0, 0, 0].real
                    for name in ('oooo', 'oovv', 'ovoo', 'ovov', 'ovvo')})
                ref_eris.ovvv = eris.ovvv_kpoint_normalized[0, 0, 0].real[
                    :, :, *np.tril_indices(mykradcee.nvir)]
                vvvv = eris.vvvv_kpoint_normalized[0, 0, 0].real
                ref_eris.vvvv = vvvv.reshape(mykradcee.nvir ** 2,
                                             mykradcee.nvir ** 2)
                reference.transform_integrals = lambda: ref_eris
                reference.verbose = 0
                reference.stdout = mykradc.stdout
                reference.method_type = 'ee'
                reference.approx_trans_moments = False
                ref_t1, ref_t2, _ = radc_amplitudes.compute_amplitudes(
                    reference, ref_eris)
                for actual_amp, expected_amp in zip(reference.t1, ref_t1):
                    if expected_amp is not None:
                        np.testing.assert_allclose(
                            actual_amp, expected_amp, atol=1e-12)
                for actual_amp, expected_amp in zip(reference.t2, ref_t2):
                    if expected_amp is not None:
                        np.testing.assert_allclose(
                            actual_amp, expected_amp, atol=1e-12)
                expected = radc_ee.get_imds(reference, ref_eris)
                actual = mykradcee.get_imds(eris)
                np.testing.assert_allclose(actual, expected, atol=1e-12)
                ref_action = radc_ee.matvec(reference, expected, ref_eris)
                action = mykradcee.matvec(0, actual, eris)
                rng = np.random.default_rng(20)
                vector = rng.normal(size=mykradcee.vector_size())
                np.testing.assert_allclose(
                    action(vector), ref_action(vector), atol=1e-12)
                # Singlet doubles have simultaneous hole-particle exchange.
                vectors = rng.normal(size=(mykradcee.vector_size(), 2))
                for root in range(2):
                    block = vectors[mykradcee.n_singles:, root].reshape(
                        mykradcee.nocc,
                        mykradcee.nocc,
                        mykradcee.nvir,
                        mykradcee.nvir)
                    block[:] = block + block.transpose(1, 0, 3, 2)
                reference.U = vectors.copy()
                expected_t = radc_ee.get_trans_moments(
                    reference).reshape(2, -1)
                actual_t = mykradcee.get_trans_moments(0, vectors)
                np.testing.assert_allclose(actual_t, expected_t, atol=1e-12)
                eris.close()

    def test_gamma_oscillator_strengths(self):
        """Compare four nondegenerate LiH excitations at all three orders."""
        cell = gto.M(atom='Li 0 0 0; H 0 0 1.6', a=np.diag([6., 7., 8.]),
                     unit='Angstrom', basis='sto-3g', verbose=0)
        self._check_supercell(cell, [1, 1, 1], compare_all=True)

    def test_four_k_oscillator_strengths(self):
        """Match the example's eight roots, including its degenerate pairs."""
        self._check_supercell(mf.cell, [1, 2, 2], methods=('adc(2)',),
                              nroots=2, compare_all=True)

    def test_supercell_oscillator_strengths(self):
        r"""Check two-k EE energies and oscillator strengths against the Gamma
        supercell.

        Invoke the common supercell comparison on a 1x1x2 mesh for all EE
        orders,
        including sums of response strengths within degenerate eigenspaces.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self._check_supercell(mf.cell, [1, 1, 2])

    def test_complex_kpoint_oscillator_strengths(self):
        r"""Check three-k EE strengths with complex Bloch orbitals.

        Invoke the same supercell comparison on a 1x1x3 mesh, where complex
        orbital phases exercise momentum orientation and integral conjugation.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        self._check_supercell(mf.cell, [1, 1, 3])

    def _check_supercell(self, cell, mesh,
                         methods=('adc(2)', 'adc(2)-x', 'adc(3)'),
                         nroots=4, compare_all=False):
        r"""Compare periodic EE energies and oscillator strengths with a
        supercell.

        Args:
            cell (Cell): Primitive cell used for the k-point calculation.
            mesh (sequence[int]): Replication counts defining the k mesh and
                its commensurate Gamma-point supercell.
            methods (tuple[str]): Approximation orders to check.
            nroots (int): Number of roots requested per momentum sector.
            compare_all (bool): Require both solvers to cover the same roots,
                including the upper edge of the computed spectrum.

        Returns:
            None: For all EE orders, assert agreement of roots below the shared
            cutoff and summed oscillator strengths in degenerate subspaces.
            Compare directly with molecular ADC pole energies and strengths.
        """
        kpoints = cell.make_kpts(mesh)
        kmf = scf.KRHF(cell, kpoints, exxdiv=None).density_fit().run()
        scell = super_cell(cell, mesh)
        smf = scf.RHF(scell, exxdiv=None).density_fit().run()
        for method in methods:
            with self.subTest(method=method):
                mykradc = adc.KRADCEE(kmf)
                mykradc.method = method
                molecular = molecular_adc.RADC(smf)
                molecular.method_type = 'ee'
                molecular.method = method
                mykradc.approx_trans_moments = molecular.approx_trans_moments
                mykradc.conv_tol = molecular.conv_tol = 1e-10
                mykradc.tol_residual = molecular.tol_residual = 1e-8
                ke, _, kp, _ = mykradc.kernel(nroots)
                me, _, mp, _ = molecular.kernel(nroots * len(kpoints))
                # Restrict the comparison to roots covered by both solves.
                cutoff = min(ke.max(axis=1).min(), me.max()) - 1e-6
                if compare_all:
                    cutoff = np.inf
                np.testing.assert_allclose(
                    np.sort(ke[ke < cutoff]),
                    np.sort(me[me < cutoff]),
                    atol=1e-06,
                    rtol=0)
                # Molecular roots may mix degenerate states from different
                # transfers. Their summed oscillator strength is invariant.
                for energy in me[me < cutoff]:
                    expected = mp[abs(me - energy) < 1e-6].sum()
                    actual = kp[abs(ke - energy) < 1e-6].sum()
                    np.testing.assert_allclose(
                        actual, expected, atol=1e-6, rtol=1e-5)

    def test_optical_transition(self):
        """Compare a bright Gamma transition with the AO velocity integral."""
        cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 5, unit='B',
                     basis=[[0, (1., 1.)], [1, (.5, 1.)]], verbose=0)
        kmf = scf.KRHF(cell, exxdiv=None).density_fit().run()
        mykradc = adc.KRADCEE(kmf)
        mykradc.oscillator_strength_mode = 'periodic'
        energies, vectors, weights, densities = mykradc.kernel(3)
        coeff = kmf.mo_coeff[0]
        velocity = coeff.conj().T @ cell.pbc_intor('int1e_ipovlp') @ coeff
        moment = np.einsum('xpq,npq->nx', velocity, densities[0, :, 0])
        reference = 4 * (abs(moment)**2).sum(axis=1) / (3 * energies[0])
        self.assertGreater(reference.max(), 1e-3)
        np.testing.assert_allclose(weights[0], reference, atol=1e-12)
        np.testing.assert_array_equal(
            mykradc.T.reshape(densities.shape), densities)
        np.testing.assert_array_equal(mykradc.U, vectors)
        np.testing.assert_array_equal(mykradc.E, energies)

        # A reciprocal lattice vector probes a finite transfer at Gamma.
        mykradc.density_gv = cell.reciprocal_vectors()[0]
        strengths, _ = mykradc.get_properties(0, vectors[0].T, 3)
        ao = ft_ao.ft_aopair(cell, mykradc.density_gv.reshape(1, 3))[0]
        rho = coeff.conj().T @ ao @ coeff
        amplitudes = np.einsum('pq,npq->n', rho, densities[0, :, 0])
        q2 = mykradc.density_gv @ mykradc.density_gv
        np.testing.assert_allclose(strengths,
                                   4 * energies[0] * abs(amplitudes)**2 / q2)
        with self.assertRaisesRegex(ValueError, 'matching'):
            mykradc.get_properties(0, vectors[0, ::-1].T, 3)


if __name__ == '__main__':
    unittest.main()
