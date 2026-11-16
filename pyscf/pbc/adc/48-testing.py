#!/usr/bin/env python
#
# Authors: 
#       Ziqiu Wang <sgwzq0810@gmail.com>
#       Samragni Banerjee <samragnibanerjee4@gmail.com>
#

# --- Options ---
type_ = 'ee'
method = 'adc(2)'
# --- End of Options ---

'''
This example shows the use of periodic EA/IP-ADC with k-point sampling.
Also shown are molecular EA/IP-ADC Gamma-point calculations using a supercell.
'''

import numpy as np
from pyscf.pbc import gto, scf, adc
from pyscf import adc as mol_adc
from pyscf.pbc.tools.pbc import super_cell


cell = gto.Cell()
cell.verbose = 3
cell.unit = 'B'

#
# Helium crystal
#
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000
'''
cell.build()

ncopies = [1, 1, 2]
kptlist = np.arange(np.prod(ncopies)) # [0, 1, 2, 3] if ncopies = [1, 2, 2]
nroots_test = 4

# KRHF
kpoints = cell.make_kpts(ncopies)
kmf = scf.KRHF(cell, kpts=kpoints, exxdiv=None).density_fit()
ekrhf = kmf.kernel()

# KADC
mykradc  = adc.KRADC(kmf)

# EE-KRADC
mykradc.type_ = type_
mykradc.method = method
mykradc.approx_trans_moments = False
mykradc.kernel(nroots_test, kptlist=kptlist)
k_e_ee, v_e_ee = mykradc.E, mykradc.U
k_p_ee, k_x_ee = mykradc.P, mykradc.X

# Supercell
scell = super_cell(cell, ncopies)

# PBC Gamma-point RHF based on supercell
mf = scf.RHF(scell, exxdiv=None).density_fit()
erhf = mf.kernel()

# Molecular ADC
myadc = mol_adc.RADC(mf)

# Molecular EE-ADC
myadc.method_type = type_
myadc.method = method
mol_e_ee,mol_v_ee,mol_p_ee,mol_x_ee = myadc.kernel(
    nroots_test * np.prod(ncopies))

def tprint(data: dict, title=None, num_header=None) -> None:
    r"""Print a numbered comparison table to standard output.

    Args:
        data (dict[str, sequence]): Nonempty ordered mapping from column
            headings to values. Unequal columns are extended with None.
            Floating values use five digits after the decimal point; other
            values are converted with str().
        title (str, optional): Centered star-filled heading and matching
            End of <title> footer. Omit both when title is false.
        num_header (str, optional): Heading for the one-based row-number
            column. Its width is at least three; data columns use width eleven.

    Returns:
        None: Print comma-separated centered headings and rows. Widths are
        minimum field widths, so long headings or values can extend a line.
        This helper is local to the regression example; production ADC tables
        are formatted by pyscf.pbc.adc.lib.table_lines().

    Examples:
        The helper lives in the executable 48-testing.py example. Its first
        import also runs the small helium regression; capture that diagnostic
        output before formatting a separate table below.

        >>> import importlib
        >>> import io
        >>> from contextlib import redirect_stdout
        >>> with redirect_stdout(io.StringIO()):
        ...     comparison = importlib.import_module(
        ...         'pyscf.pbc.adc.48-testing')
        >>> tprint = comparison.tprint
        >>> output = io.StringIO()
        >>> with redirect_stdout(output):
        ...     tprint({'E': [1.0, 2.0]}, title='ADC')
        >>> print(output.getvalue(), end='')  # doctest: +NORMALIZE_WHITESPACE
        ******ADC*******
           ,     E
         1 ,  1.00000
         2 ,  2.00000
        ***End of ADC***
        >>> lines = output.getvalue().splitlines()
        >>> lines[0], lines[-1], len(lines)
        ('******ADC*******', '***End of ADC***', 5)
        >>> [cell.strip() for cell in lines[2].split(',')]
        ['1', '1.00000']
    """
    COL_WIDTH = 11
    NUM_COL_WIDTH = 3
    FLOAT_PRECISION = 5
    if isinstance(num_header, str):
        NUM_COL_WIDTH = max(NUM_COL_WIDTH, len(num_header))
    # No., Col1, Col2, ...
    LINE_WIDTH = (
            + NUM_COL_WIDTH+1                   # Plus 1 for delimeters ','.
            + (COL_WIDTH+1)*len(data) 
            )
    # Title: start of table.
    if title: print(f'{title:*^{LINE_WIDTH}}')
    # Print headers.
    if not isinstance(num_header, str): num_header = ''
    header = (
            f'{num_header:^{NUM_COL_WIDTH}},'
            + ','.join(f'{str_:^{COL_WIDTH}}' for str_ in data.keys())
            )
    print(header)
    # Calc how many rows in total.
    len_each_col = [len(arr) for arr in data.values()]
    maxlen = max(len_each_col)
    # Generate a row for each data. row_index col first, then all other cols.
    for row_idx in range(maxlen):
        row_out = f'{row_idx+1:^{NUM_COL_WIDTH}},'   # Start counting from 1
        # Generate each line of table and print.
        for col_idx, arr in enumerate(data.values()):
            val = arr[row_idx] if row_idx < len_each_col[col_idx] else None
            # Get string for each cell. This could be defined explicitly.
            if isinstance(val, float): 
                val_str = f'{val:^ {COL_WIDTH}.{FLOAT_PRECISION}f}' 
            else:
                val_str = f'{str(val):^{COL_WIDTH}}'
            # Add delimiter.
            if col_idx < len(data.values()) - 1:
                val_str += ','
            row_out += val_str
        print(row_out)
    # Title: end of table.
    if title: print(f"{'End of ' + title:*^{LINE_WIDTH}}")

def tprint_compare(col_title1, arr1, col_title2, arr2, title=None) -> None:
    r"""Print two numerical arrays and their elementwise second-minus-first
    difference.

    Args:
        col_title1 (str): Heading for the first numerical column.
        arr1 (ndarray): First one-dimensional comparison array.
        col_title2 (str): Heading for the second numerical column.
        arr2 (ndarray): Second array, with shape compatible with arr1.
        title (str, optional): Heading/footer passed to tprint().

    Returns:
        None: Print the two arrays and a Diff column containing arr2-arr1.
        Inputs must support subtraction; numeric formatting and row numbering
        follow tprint(). No tolerance or numerical assertion is applied here;
        the regression script checks energies separately.

    Examples:
        The helper lives in the executable 48-testing.py example. Its first
        import also runs the small helium regression; capture that diagnostic
        output before formatting a separate table below.

        >>> import importlib
        >>> import io
        >>> import numpy as np
        >>> from contextlib import redirect_stdout
        >>> with redirect_stdout(io.StringIO()):
        ...     comparison = importlib.import_module(
        ...         'pyscf.pbc.adc.48-testing')
        >>> tprint_compare = comparison.tprint_compare
        >>> output = io.StringIO()
        >>> with redirect_stdout(output):
        ...     tprint_compare(
        ...         'PBC', np.array([1.0]), 'Mol', np.array([1.0]), title='ADC'
        ...     )
        >>> print(output.getvalue(), end='')  # doctest: +NORMALIZE_WHITESPACE
        ******************ADC*******************
           ,    PBC    ,    Mol    ,   Diff
         1 ,  1.00000  ,  1.00000  ,  0.00000
        ***************End of ADC***************
        >>> row = output.getvalue().splitlines()[2]
        >>> [cell.strip() for cell in row.split(',')]
        ['1', '1.00000', '1.00000', '0.00000']
    """
    mydict = {
            col_title1: arr1,
            col_title2: arr2,
            'Diff': arr2 - arr1
            }
    tprint(mydict, title)

# Print and Compare
# Only states below every run's highest root are present in both lists.
# The 1e-6 margin keeps a degenerate set at the cutoff out of both lists.
cutoff = min(np.max(k_e_ee, axis=1).min(), np.max(mol_e_ee))
k_e_ee_low = np.sort(k_e_ee.ravel())
k_e_ee_low = k_e_ee_low[k_e_ee_low < cutoff - 1e-6]
mol_e_ee_low = np.sort(mol_e_ee)
mol_e_ee_low = mol_e_ee_low[mol_e_ee_low < cutoff - 1e-6]
n_compare = min(len(k_e_ee_low), len(mol_e_ee_low))
tabtitle = (type_ + '-' + method).upper()
tprint_compare(
    'PBC', k_e_ee_low[:n_compare], 'Mol', mol_e_ee_low[:n_compare], tabtitle)
print(f'Compared the {n_compare} states below {cutoff:.6f} Eh, the lowest highest root of all runs.')
if len(k_e_ee_low) != len(mol_e_ee_low):
    print(f'FLAG: different number of states below the cutoff: PBC {len(
        k_e_ee_low)}, Mol {len(mol_e_ee_low)}')

# Fail the regression check when the supercell comparison disagrees.
assert n_compare > 0, 'No common EE roots were available for comparison.'
assert len(k_e_ee_low) == len(mol_e_ee_low), 'EE state counts differ.'
np.testing.assert_allclose(k_e_ee_low, mol_e_ee_low, atol=1e-06, rtol=0)
