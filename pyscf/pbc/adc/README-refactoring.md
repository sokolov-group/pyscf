# Restricted periodic ADC

`kadc_rhf.py` defines `KRADCBase` and its `KRADCIP`, `KRADCEA` and
`KRADCEE` subclasses. The base inherits `lib.StreamObject` directly and
shares reference amplitudes, integral handling, solver settings, guesses,
and charged-property helpers. Methods start with `__init__` and `kernel`,
then follow their call dependencies; sector classes inherit those entry
points. The earlier sector modules re-export these
classes and retain functional adapters for importing applications.
Molecular implementations under `pyscf/adc` retain their own APIs.

## Reference calculations and API names

```python
import numpy as np
from pyscf.pbc import adc, gto, scf

cell = gto.M(atom='He 0 0 0', a=np.eye(3) * 3,
             basis=[[0, (1., 1.)], [0, (.5, 1.)]], verbose=0)
kmf = scf.KRHF(cell, cell.make_kpts([1, 1, 2]), exxdiv=None).density_fit().run()
mykradcbase = adc.KRADCBase(kmf)
energy = mykradcbase.ground_state_energy()
singles, doubles = mykradcbase.t1, mykradcbase.t2
assert energy == mykradcbase.e_corr

mykradcee = adc.KRADCEE(mykradcbase)
excitation_results = mykradcee.kernel(2, kptlist=[0])
energies = mykradcee.E
mykradcea = adc.KRADCEA(mykradcee)
assert type(mykradcea) is adc.KRADCEA
assert mykradcea.e_corr == mykradcee.e_corr
assert mykradcea.E is None
```

Each concrete class also accepts a raw periodic `Cell`. It creates a
density-fitted KRHF with `exxdiv=None`, defaulting to Gamma; pass `kpts=`
for a different mesh. SCF is deferred until orbital data are needed.
`KRADCEE(cell)` and `KRADCEA(KRADCEE(cell))` therefore construct without
running SCF. Existing KRHF references can be converged or unrun.

Select the sector by constructing its class and inspect it with `type(obj)`.
The former `type_` and `method_type` attributes are removed; assigning them
raises a migration error. `KRADCBase` supports reference calculations and
`kernel(nroots=0)`, while positive roots require a concrete sector class.
The inherited `.set(...)` and `.run(...)` support PySCF chaining; positional
arguments to `.run()` are forwarded to `kernel()`. Use `.run(0)` when only
reference preparation is needed. Docstring examples use this form instead
of capturing unused energy values. A returned energy is still shown when
the example uses it.

ADC-to-ADC conversion preserves common reference settings, orbital data,
reference amplitudes and `e_corr`. It clears roots, property results and
sector caches. The amplitude-storage owner is shared, so recomputing either
object does not invalidate the other object's saved reference arrays.
Orbital or frozen-space overrides discard copied amplitudes. `None` for
`frozen` inherits an ADC source setting; set `.frozen = None` afterward to
unfreeze it explicitly.

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
| `KRADC` / `kadc_rhf.RADC` | Reference-only `KRADCBase` |
| `RADCIP` / `RADCEA` / `RADCEE` | `KRADCIP` / `KRADCEA` / `KRADCEE` |
| `type_` / `method_type` sector attributes | Construct `KRADCIP`, `KRADCEA` or `KRADCEE` |
| `kernel_gs()` | `ground_state_energy()` |
| `transform_integrals()` | `ao2mo()` |
| `chnk_size` / `get_chnk_size()` | `chunk_size` property |
| `_orbital_data()` | `padded_coeff`, `e_occ`, `e_vir` properties |
| `compute_amplitudes()` | `amplitudes()` and instance attributes |
| `compute_energy()` | `correlation_energy()` |
| `get_init_guess(..., type=...)` | Shared `_init_guess(..., type_=...)` helper |

The earlier base `kernel_gs()` still returns its tuple with a
DeprecationWarning; callers should migrate to the scalar API and instance
fields. The earlier `compute_*` methods and the driver
`transform_integrals()` method also emit DeprecationWarning. Earlier sector
class names remain aliases of their canonical K-point class names. Periodic
examples and tests use the new names. Molecular reference calls keep their
existing PySCF names.

The shared solver calls `_init_guess` for every sector. The earlier
`_charged_init_guess` and `_neutral_init_guess` spellings alias this helper.
Its `type_=None` argument selects diagonal unit guesses and `type_='read'`
selects supplied vectors; this argument describes the guess, not an ADC
sector attribute. The inherited `get_init_guess`
adapter emits a DeprecationWarning and accepts the earlier `type` keyword.
User guesses have shape `(nsectors, nroots, vector_size)` at the kernel
boundary, and `(nroots, vector_size)` for one sector's helper.

## Integral transformation and layout

`KRADCBase.ao2mo()` creates a `kadc_ao2mo.AO2MO` context. Its initializer shares
padded coefficients, dimensions, momenta, dtype and the AO transformation
function. Calling the context selects one backend and returns the integral
container; a Python initializer itself does not return the backend result.

```python
from pyscf.pbc.adc.kadc_ao2mo import AO2MO

context = AO2MO(mykradcee, backend='density_fit')
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

IP and EA singles intermediates use separate `_get_adc0_ham()`,
`_get_adc2_ham(eris, M=None)` and `_get_adc3_ham(eris, M=None)` helpers.
The latter two return corrections alone when `M` is omitted, or accumulate
into `M` in place. `get_imds()` combines the orders requested by `method`;
ADC(2)-x shares ADC(2)'s singles block. IP retains the orbital-energy sign
in the intermediate, with the physical singles Hamiltonian equal to its
negative. EA uses the intermediate directly. Names `M_ij`, `M_ab` and `M_ph`
identify the IP, EA and neutral particle-hole singles blocks, respectively.
These are blocks/contributions of the effective Hamiltonian M in the ADC
terminology of [Banerjee and Sokolov](https://doi.org/10.1063/1.5131771).

Charged Davidson diagonals use the solver's singles/doubles dimensions and
fill views of one flat result array. Orbital-energy differences are
broadcast one k-point slab at a time, retaining independent momentum axes.
The input energies and singles block determine real/complex precision;
inactive padded components retain a large positive sentinel. Small
`SimpleNamespace` examples show the actual IP/EA diagonal values without
requiring SCF. The IP second-order builder reuses one orbital-block local
in place of deleting four separately named slices; explicit releases of
large intermediates elsewhere retain their existing memory behavior.

FNO workflows have concrete `KRADCIPFNO`, `KRADCEAFNO` and `KRADCEEFNO`
classes. `KRADC2FNO(adc_object)` preserves the source sector; raw Cells,
KRHF references and reference-only ADC inputs default to IP. For example,
`adc.KRADC2FNO(adc.KRADCEA(kmf))` constructs an EA FNO driver. The earlier
`RADC2FNO` name aliases `KRADCIPFNO`. EE FNOs use the reference density;
state-specific EE densities are not implemented.

## Density-response properties

For full transition moments, explicitly set `approx_trans_moments=False`.
The periodic default remains `True` for compatibility with IP/EA usage.
With that approximation, absent higher-order amplitudes are omitted,
following the molecular reference's behavior.

```python
mykradc = adc.KRADCEE(kmf)
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
sectors. Results are stored directly on the concrete solver. With properties disabled, `T` is `None` and the returned
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

Docstrings stay within 79 source characters per line. Each worked example
includes its imports and setup. Numerical examples construct their own
two-k-point helium `kmf`; pure helpers supply their own small input arrays
or models. No example needs variables from
another docstring or an existing SCF calculation.

Use a Python environment with NumPy and this built PySCF checkout, including
`pyscf.pbc.adc`, importable. Enter the code after `>>>` and `...` in order;
unprefixed lines show expected output. Each example block, including blocks
separated by explanatory prose, can start in a fresh Python session.
`48-testing.py` uses `lib.log_table()` for its numbered PBC/molecular
comparison instead of maintaining local printing helpers.

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

API tests check concrete sector classes, ADC-to-ADC conversion, shared
amplitude ownership across reference recomputation, deferred SCF, typed FNO
construction, charged Hamiltonian orders, ragged root selection,
scalar energies, state-only amplitudes, legacy warnings,
zero-root behavior, lazy integral properties, cached chunking and logging.
They execute the worked docstring examples and check reference-density AO
conversion with frozen orbitals, complex MO phases and rectangular active
coefficient matrices. AO conversion now allocates an AO-sized result and
works without a frozen-space branch. The optimized first-order EE block is
also checked against the identity-tensor prototype with unequal occupied
and virtual dimensions. Charged diagonal tests independently enumerate
orbital and momentum indices for every sector and four numerical dtypes,
and check unequal virtual spaces and integral-container reuse.
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
