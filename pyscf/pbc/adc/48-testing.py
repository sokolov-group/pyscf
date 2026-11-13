#!/usr/bin/env python
#
# Authors: 
#       Ziqiu Wang <sgwzq0810@gmail.com>
#       Samragni Banerjee <samragnibanerjee4@gmail.com>
#

# --- Options ---
method_type = 'ee'
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
nroots_test = 2

# KRHF
kpoints = cell.make_kpts(ncopies)
kmf = scf.KRHF(cell, kpts=kpoints, exxdiv=None).density_fit()
ekrhf = kmf.kernel()

# KADC
kadc  = adc.KRADC(kmf)

# EE-KRADC
kadc.method_type = method_type
kadc.method = method
k_e_ee, v_e_ee, k_p_ee, k_x_ee = kadc.kernel(nroots_test,kptlist=kptlist)

# Supercell
scell = super_cell(cell, ncopies)

# PBC Gamma-point RHF based on supercell
mf = scf.RHF(scell, exxdiv=None).density_fit()
erhf = mf.kernel()

# Molecular ADC
myadc = mol_adc.RADC(mf)

# Molecular EE-ADC
myadc.method_type = method_type
myadc.method = method
mol_e_ee,mol_v_ee,mol_p_ee,mol_x_ee = myadc.kernel(nroots_test*np.prod(ncopies))

def tprint(data: dict, title=None, num_header=None) -> None:
    '''
    Print Tables
    Input: 
            data: dict. consisting of { 'col_title': content: array_like }
            title: str. The title of the whole table.
            num_header: str, The title of the No. column.
            wlist: list, The width per column.
    Print: A table separated with comma ','
    Examples:
    >>> title = 'Sample'
    >>> header_1 = 'Name'; arr_1 = ['Amy', 'Tom']
    >>> header_2 = 'Age'; arr_2 = [19, 20]
    >>> tprint(data = {
            header_1: arr_1,
            header_2: arr_2,
            }, 
            title=title,
            num_header=''
            )
    ***********Sample***********
       ,   Name    ,    Age
     1 ,    Amy    ,    19
     2 ,    Tom    ,    20
    *******End of Sample********
    '''
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
    if title: print(f'{'End of '+title :*^{LINE_WIDTH}}')

def tprint_compare(col_title1, arr1, col_title2, arr2, title=None) -> None:
    '''
    Compare two arrays and print list.
    '''
    mydict = {
            col_title1: arr1,
            col_title2: arr2,
            'Diff': arr2 - arr1
            }
    tprint(mydict, title)

# Print and Compare
k_e_ee_ravel = np.sort(k_e_ee.ravel())
tabtitle = (method_type + '-' + method).upper()
tprint_compare('PBC', k_e_ee_ravel, 'Mol', mol_e_ee, tabtitle)
