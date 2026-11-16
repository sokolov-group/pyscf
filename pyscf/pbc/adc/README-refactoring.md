# Restricted periodic ADC

The production modules share integral storage, backend selection,
optimized tensor contractions and charged-sector initial guesses. The
primary reference class is `kadc_rhf.KRADC`, returned by `adc.KRADC(kmf)`.
IP and EA numerical routines belong to `KRADCIP` and `KRADCEA`. Molecular
implementations under `pyscf/adc` retain their own APIs and are unchanged.

## Reference calculations and API names

```python
mykradc = adc.KRADC(kmf)
mykradc.type_ = 'ee'  # ip, ea, or ee
energy = mykradc.ground_state_energy()
singles, doubles = mykradc.t1, mykradc.t2
assert energy == mykradc.e_corr
```

`ground_state_energy()` returns the correlation contribution per primitive
cell in Hartree. The total reference energy is `scf_energy + e_corr`.
`amplitudes(eris)` stores `t1`, `t2` and the ladder intermediate and returns
None. `correlation_energy(eris)` stores and returns `e_corr`, reading large
doubles arrays one block at a time. No combined amplitude/energy tuple is
used internally.

`kernel(nroots=0)` and negative root counts compute the reference only,
return scalar `e_corr`, and clear stale E/U/P/X/T and convergence results.
They do not enter the excitation solver or validate excitation-only
arguments such as kptlist. Positive root counts retain the E/U/P/X tuple
interface for compatibility. New user code should call kernel() and read
E, U, P and X from driver attributes instead of unpacking its return value.
FNO's `ground_state_energy()` returns the active-space
correlation energy (or a list for multiple thresholds); additive
corrections remain available through `correct_corr`.

Compatibility entry points remain for external callers:

| Earlier name | Primary API |
| --- | --- |
| `kadc_rhf.RADC` | `kadc_rhf.KRADC` |
| `RADCIP` / `RADCEA` / `RADCEE` | `KRADCIP` / `KRADCEA` / `KRADCEE` |
| `method_type` | `type_` |
| `kernel_gs()` | `ground_state_energy()` |
| `transform_integrals()` | `ao2mo()` |
| `chnk_size` / `get_chnk_size()` | `chunk_size` property |
| `_orbital_data()` | `padded_coeff`, `e_occ`, `e_vir` properties |
| `compute_amplitudes()` | `amplitudes()` and instance attributes |
| `compute_energy()` | `correlation_energy()` |
| `get_init_guess(..., type=...)` | Sector helper with `type_`, described below |

The earlier base `kernel_gs()` still returns its tuple with a
DeprecationWarning; callers should migrate to the scalar API and instance
fields. The earlier `compute_*` methods and the driver
`transform_integrals()` method also emit DeprecationWarning. Earlier sector
class names remain aliases of their canonical K-point class names. Periodic
examples and tests use the new names. Molecular reference calls keep their
existing PySCF names.

The shared solver calls `_charged_init_guess` for IP/EA and
`_neutral_init_guess` for EE. Both use `type_=None` for diagonal unit guesses
and `type_='read'` for supplied vectors. The inherited `get_init_guess`
adapter emits a DeprecationWarning and accepts the earlier `type` keyword.
User guesses have shape `(nsectors, nroots, vector_size)` at the kernel
boundary, and `(nroots, vector_size)` for one sector's helper.

## Integral transformation and layout

`KRADC.ao2mo()` creates a `kadc_ao2mo.AO2MO` context. Its initializer shares
padded coefficients, dimensions, momenta, dtype and the AO transformation
function. Calling the context selects one backend and returns the integral
container; a Python initializer itself does not return the backend result.

```python
from pyscf.pbc.adc.kadc_ao2mo import AO2MO

context = AO2MO(mykradc, backend='density_fit')
with context() as eris:
    block = eris.ovvv_blocks[0, 0, 0]
    virtual_block = eris.vvvv_blocks[0, 0, 0]
```

Backend methods are `incore()`, `outcore()` and `density_fit()`.
`integral_backend='auto'` uses GDF when available, otherwise considering
available memory. `'df'` remains an alias for `'density_fit'`. The selected
backend setting stays stable across kernel calls and FNO rotations.
`chunk_size` is a cached property with an optional positive integer
override; assign None to restore automatic estimation. Automatic chunk
estimation accounts for real or complex element size.

`kadc_ao2mo._ChemistsERIs` owns arrays, density-fitting factors and at most
one HDF5 integral scratch file. Its `ovvv_blocks` and `vvvv_blocks`
properties provide indexable, lazy access. Three integer k indices are
required, followed by optional orbital slices. Reading these properties
does not expand a dense tensor. The raw `ovvv` and `vvvv` attributes remain
None for DF storage so existing backend decisions stay explicit.
`close()` and context-manager support provide scratch cleanup.

DF reconstruction helpers belong to `_ChemistsERIs` as `ovvv_df` and
`vvvv_df`.

Four-index blocks have three leading k axes and use chemists' ordering,
except `vvvv`, which uses particle-particle ordering for ladder
contractions. Stored and reconstructed blocks have consistent k-point
normalization. Tests include forced low-memory outcore transformations,
complex orbital phases and charged-excitation backend comparisons.

## Common helpers and logging

`lib.py` contains optimized `einsum`, `MomentumContractions`, approximation
validation, and comma-separated table formatting. Tensor identifiers use
occupied/virtual labels (o/v) and dimensions use nocc/nvir throughout the
periodic implementation. Repeated independent k loops use product loops.
`mo_splitter(mykradc, kpt=0)` returns a flat list of four one-dimensional
boolean masks for one selected k point: frozen occupied, active occupied,
active virtual and frozen virtual. Call it separately for each k point;
this supports different MO counts across the mesh. Orbital counting and
FNO construction use the same interface.

Short einsum calls stay on one line. Longer calls put the contraction
subscripts on their own line, followed by the operands together when they
fit, or one operand per line otherwise. Reference, orbital-partition and
logging docstrings include executable examples with locally checked values.

```python
from pyscf.pbc.adc import lib as adc_lib
adc_lib.LINE_WIDTH = 99
```

LINE_WIDTH is the minimum table width; headings and data can require a
wider row. Its import-time default is the PySCF configuration option
`pbc_adc_line_width` or 79. Orbital and excitation tables use matching
`End of <title>` footers and respect the PySCF logger verbosity. Local log
messages use formatted strings; numerical modulo operations are retained.

## Neutral excitations

`KRADCEE` supports singlet `adc(2)`, `adc(2)-x`, and `adc(3)`. The ADC(2)
loops remain specialized. Higher-order contractions are listed in
`kadc_ee_terms.py`, with coefficients combined from the molecular
`pyscf.adc.radc_ee` reference. Third-order ground-state singles needed for
full transition moments are also implemented.

The singles effective Hamiltonian is assembled by `_get_adc1_ham`,
`_get_adc2_ham` and `_get_adc3_ham`. The first includes zeroth-/first-order
terms; the latter two return corrections or accumulate into an existing
tensor. The orbital-energy term updates a writable diagonal view instead
of constructing identity matrices and four-index energy temporaries.
The EE Davidson preconditioner uses the full singles diagonal and
zeroth-order doubles energy differences, so it omits higher-order doubles
interactions in ADC(2)-x and ADC(3).

`lib.MomentumContractions` assigns momenta to orbital labels,
enforces each tensor's conservation law, and orients complex conjugates
so contracted orbital phases cancel. Some real-orbital `ovvo` expressions
must be written with the last electron pair reversed as `ovov` for Bloch
orbitals. Response-independent intermediates are cached across Davidson
vectors, with bounded tensor and momentum-plan caches. Second-order doubles
amplitudes are accumulated one orbital block at a
time and retained on disk; the incore ladder path no longer reads the
entire first-order doubles dataset inside its inner loop. Four-index
integrals and amplitudes keep their three independent k axes; a dense
supercell tensor or ADC matrix is not built.

These changes reduce redundant work and storage. They do not establish a
wall-time advantage for every system; large-system performance remains a
separate benchmarking task.

## Density-response properties

For full transition moments, explicitly set `approx_trans_moments=False`.
The periodic default remains `True` for compatibility with IP/EA usage.
With that approximation, absent higher-order amplitudes are omitted,
following the molecular reference's behavior.

```python
mykradc = adc.KRADC(kmf)
mykradc.type_ = 'ee'
mykradc.method = 'adc(3)'
mykradc.approx_trans_moments = False
mykradc.compute_properties = True
mykradc.density_gv = np.zeros(3)  # Cartesian reciprocal vector, inverse Bohr
mykradc.kernel(nroots=4, kptlist=[0, 1])
energies, U, P, X = mykradc.E, mykradc.U, mykradc.P, mykradc.X
T = mykradc.T
```

For `nq` requested sectors, `nr` roots, `nk` k points, and padded active
`nmo` orbitals per k point:

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `E` | `(nq, nr)` | Excitation energies in Hartree |
| `U` | `(nq, nr, vector_size)` | EE eigenvectors in the singlet metric |
| `X` | `(nq, nr, nk, nmo, nmo)` | Spin-free transition densities |
| `T` | `(nq, nr, nk*nmo*nmo)` | Flattened view of `X` |
| `P` | `(nq, nr)` | Density-response strengths per primitive cell |

The first axis follows `kptlist`, rather than indexing all possible
sectors. Results are stored on both the ground-state driver and its
`_adc_es` solver. With properties disabled, `T` is `None` and the returned
property arrays retain the shared solver's zero-filled convention.

The existing momentum convention is hole minus particle = kshift.
Accordingly, the physical transfer represented here is
`q = -kpts[kshift]`, using the supplied mesh representatives. For each k,
`X` connects that k point to its singles particle momentum. The operator
is the analytic Fourier AO-pair matrix of `exp[-i(q+G).r]`, transformed to
the active MOs; `G = density_gv`. For spin-free transition density `X_n`,

```
P_n(q+G) = (2/nk) * |sum_kpq rho_kpq(q+G) X_nkpq|**2
```

The factor two is for singlets; division by `nk` gives the primitive-cell
normalization. These are density-response strengths, as requested, not
optical oscillator strengths. At `q+G=0`, the operator is total particle
number and inelastic strengths vanish. Equivalent crystal-momentum
sectors represented by different reciprocal vectors can have different
form factors.

## Validation

Every function and class in the active Python modules, comparison script
and tests has a docstring, including constructors, property setters,
compatibility adapters and nested contraction helpers. The scientific
interfaces document equations, tensor shapes and ordering, k-point
normalization, units, mutation and scratch-file ownership. Class docstrings
describe their configuration and result attributes.

Each worked docstring example includes its imports and setup. Numerical
examples construct their own two-k-point helium `kmf`; pure helpers supply
their own small input arrays or models. No example needs variables from
another docstring or an existing SCF calculation.

Use a Python environment with NumPy and this built PySCF checkout, including
`pyscf.pbc.adc`, importable. Enter the code after `>>>` and `...` in order;
unprefixed lines show expected output. Each example block, including blocks
separated by explanatory prose, can start in a fresh Python session. The
comparison-script helpers show
their explicit loader; the first import of that script also runs its small
helium regression, with diagnostic output captured in the example.

The documentation test reads the source docstrings and runs every example
block in an empty namespace. It supplies no module globals, imported
functions, NumPy alias or shared reference. Run it on its own with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
  test.test_kadc_api.Documentation -v
```

Run the focused tests with one BLAS/OpenMP thread for reproducibility:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest \
  test.test_kadc_api test.test_k_ee test.test_k_ee_orders \
  test.test_kadc_integrals test.test_kadc test.test_ground_state_energy \
  test.test_k_vs_supercell_ea test.test_k_vs_supercell_ip \
  test.test_k_ea test.test_k_ip test.test_kadc_fno
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 48-testing.py
```

The separate `test_k_vs_supercell_{ea,ip}_highcost` modules are optional
larger-system benchmarks and are not included in this command.

API tests check scalar energies, state-only amplitudes, legacy warnings,
zero-root behavior, lazy integral properties, cached chunking and logging.
They execute the worked docstring examples and check reference-density AO
conversion with frozen orbitals, complex MO phases and rectangular active
coefficient matrices. AO conversion now allocates an AO-sized result and
works without a frozen-space branch. The optimized first-order EE block is
also checked against the identity-tensor prototype with unequal occupied
and virtual dimensions.
The numerical tests compare Gamma amplitudes, singles blocks, arbitrary
matrix-vector products, and transition densities against molecular implementations for
all three EE methods. Supercell checks compare energies and summed
strengths in degenerate subspaces for two- and three-k-point meshes,
including complex Bloch orbitals. Backend tests compare DF, incore, and
forced low-memory outcore integrals after complex MO phase rotations.
Existing ground-state, IP, EA, and FNO regression tests exercise the shared
refactor. `48-testing.py` now asserts its energy comparison and uses full
transition moments.

## References and scope

The default EE Davidson starting guesses can still miss the lowest roots
in symmetric crystals while reporting convergence. A separate dense-matrix
audit of ADC(2), four roots, at Gamma for diamond C (a = 3.567 Angstrom)
and Si (a = 5.431 Angstrom), using GTH-SZV/GTH-PADE, Gaussian density
fitting and exxdiv=None, found maximum differences of 2.61e-4 and 6.43e-3
Hartree, respectively. The existing regression suite passes but does not
cover these root-completeness cases. A robust initial-guess strategy and
permanent dense-versus-Davidson regressions remain to be implemented.

The molecular working equations in `pyscf/adc/radc_ee.py` and
`radc_amplitudes.py` are the direct implementation references. The supplied
literature provides the broader ADC and periodic charged-excitation
context:

- [Third-order EA/IP-ADC and Green's functions](https://doi.org/10.1063/1.5131771).
- [Efficient EA/IP-ADC implementation](https://doi.org/10.1063/5.0040317).
- [Non-Dyson ADC for charged excitations in solids](https://doi.org/10.1021/acs.jctc.2c00565).
- [Multireference ADC formulation](https://doi.org/10.1063/1.5055380).

These papers do not by themselves validate the new periodic neutral
implementation. The molecular and supercell comparisons above are the
numerical checks. Triplet states, optical velocity/dipole response, and
large-solid scaling benchmarks are outside this change.
