import pyscf.pbc.tools.make_test_cell as make_test_cell
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, adc
import unittest

def setUpModule():
    r"""Prepare the diffuse-cell KRHF reference shared by the basic IP tests.

    Returns:
        None: Store cell and kmf as module globals. Use the N3 diffuse test
        cell, a Gamma-containing 1x1x2 mesh, and tight SCF energy/gradient
        tolerances before any ADC roots are computed.
    """
    global cell, kmf, mycc, eris
    cell = make_test_cell.test_cell_n3_diffuse()
    kmf = scf.KRHF(
        cell,
        kpts=cell.make_kpts([1, 1, 2], with_gamma_point=True),
        exxdiv=None).density_fit()
    kmf.conv_tol = 1e-10
    kmf.conv_tol_grad = 1e-6
    kmf.verbose = 0
    kmf.scf()

def tearDownModule():
    r"""Release the diffuse-cell fixture and close its log stream.

    Returns:
        None: Close cell.stdout and delete the module's cell/kmf references
        after the test case has completed.
    """
    global cell, kmf
    cell.stdout.close()
    del cell, kmf

class KnownValues(unittest.TestCase):
    r"""Check diffuse-cell reference energies and momentum-selected IP roots.

    The module fixture supplies the reference system shared by these tests.
    Each method exercises a separate numerical or API contract; no production
    solver state is returned by the test case.

    Run with ``python -m unittest test.test_kadc.KnownValues``. The tests
    raise unittest assertion failures on a regression rather than returning
    energies or pass/fail flags.
    """
    def test_n3_diffuse(self):

        # Running HF and MP2 and MP3 with 1x1x2 Monkhorst-Pack k-point mesh
        r"""Check diffuse-cell HF, MP2 and momentum-resolved IP-ADC roots.

        Verify the stored HF/MP2 values, compare three IP-ADC(2) roots in each
        selected sector, and check the IP-ADC(3) roots at sector zero. Energies
        are checked to four decimal places, including the single-sector
        ordering.

        Returns:
            None: unittest reports an assertion failure if the stated contract
            fails.
        """
        ehf2 = kmf.e_tot
        self.assertAlmostEqual(ehf2, -6.1729762695142645, 4)

        mykradc  = adc.KRADC(kmf)
        e_mp2 = mykradc.ground_state_energy()
        t1 = mykradc.t1
        t2 = mykradc.t2
        self.assertAlmostEqual(e_mp2, -0.11464808675651168, 4)

        # IP-ADC(2)
        mykradc.method = 'adc(2)'
        e, v, p, x = mykradc.kernel(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e[0][0], -1.1684386160049567, 4)
        self.assertAlmostEqual(e[0][1], -1.0944210679002468, 4)
        self.assertAlmostEqual(e[0][2], -1.0944210623845938, 4)

        # Ensure kptlist behaves correctly
        e, v, p, x = mykradc.kernel(nroots=3, kptlist=[1])
        self.assertAlmostEqual(e[0][0], -1.0362468821854942, 4)
        self.assertAlmostEqual(e[0][1], -1.0015050525294895, 4)
        self.assertAlmostEqual(e[0][2], -1.0015050525294888, 4)

        # IP-ADC(3)
        mykradc.method = 'adc(3)'
        e, v, p, x = mykradc.kernel(nroots=3, kptlist=[0])
        self.assertAlmostEqual(e[0][0], -1.1740951190044178, 4)
        self.assertAlmostEqual(e[0][1], -1.1536065878793587, 4)
        self.assertAlmostEqual(e[0][2], -1.153606584783398, 4)

if __name__ == '__main__':
    print('kadc_ip_basic tests')
    unittest.main()
