# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Terrence Stahl <terrencestahl@gmail.com>
#         Ning-Yuan Chen <cny003@outlook.com>
#         Alexander Sokolov <alexander.y.sokolov@gmail.com>
#

"""Tensor contractions, validation and logging shared by periodic ADC.

``einsum`` uses optimized contraction paths. ``MomentumContractions``
keeps k-point conservation explicit without dense supercell tensors.
``LINE_WIDTH`` controls the common summary and orbital-information tables.
"""
from collections import Counter, OrderedDict
from functools import lru_cache
from itertools import product

import numpy as np

from pyscf import __config__, lib as pyscf_lib
from pyscf.lib import logger


LINE_WIDTH = getattr(__config__, 'pbc_adc_line_width', 79)
IMPLEMENTED_METHODS = ('adc(2)', 'adc(2)-x', 'adc(3)')


def einsum(subscripts, *operands, **kwargs):
    r"""Contract orbital tensors with PySCF's optimized Einstein summation.

    Args:
        subscripts (str): Einstein input/output labels, for example
            'ij,jk->ik'. Ellipses and omitted output labels follow PySCF's
            einsum interface.
        *operands (array_like): Tensors in the order specified by subscripts.
        **kwargs: Options forwarded to pyscf.lib.einsum. optimize defaults
            to True; an explicit value supplied by the caller is retained.

    Returns:
        ndarray or scalar: Contracted result with output axes in subscripts
        order. Result shape and dtype follow the selected contraction.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc.adc.lib import einsum
        >>> einsum('ij,j->i', np.eye(2), np.array([2.0, 3.0])).tolist()
        [2.0, 3.0]
    """
    kwargs.setdefault('optimize', True)
    return pyscf_lib.einsum(subscripts, *operands, **kwargs)


def check_implemented(method, supported=IMPLEMENTED_METHODS):
    r"""Normalize an ADC approximation name and check whether it is supported.

    Args:
        method (str): Case-insensitive approximation name. The spelling,
            parentheses and hyphen must match one of the supported entries.
        supported (sequence[str]): Lowercase allowed names; defaults to
            ('adc(2)', 'adc(2)-x', 'adc(3)').

    Returns:
        str: Lowercase approximation name. No surrounding whitespace is
        removed.

    Raises:
        NotImplementedError: The normalized name is not in supported.

    Examples:
        >>> from pyscf.pbc.adc.lib import check_implemented
        >>> check_implemented('ADC(2)-X')
        'adc(2)-x'
        >>> check_implemented('adc(4)')
        Traceback (most recent call last):
            ...
        NotImplementedError: adc(4)
    """
    method = method.lower()
    if method not in supported:
        raise NotImplementedError(method)
    return method


def validate(adc):
    r"""Normalize the ADC approximation and run verbosity-aware sanity checks.

    Args:
        adc (KRADCBase): Driver with method and verbose attributes, and a
            check_sanity() method. Only the method is checked against the
            implemented list here.

    Returns:
        None: The lowercase method is stored back on adc. Invoke
        check_sanity() only when verbose is at least logger.WARN.

    Raises:
        NotImplementedError: The requested ADC approximation is unsupported.

    Examples:
        >>> from pyscf.pbc.adc.lib import validate
        >>> from types import SimpleNamespace
        >>> model = SimpleNamespace(method='ADC(2)', verbose=0)
        >>> validate(model)
        >>> model.method
        'adc(2)'
    """
    adc.method = check_implemented(adc.method)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()


def table_lines(data, title=None, width=None):
    r"""Format columns as a comma-separated table, without writing to stdout.

    Args:
        data (dict): Ordered column headings and sequences of formatted cells.
        title (str, optional): Heading; the footer reads ``End of <title>``.
        width (int, optional): Minimum table width; defaults to LINE_WIDTH.

    Returns:
        list[str]: Printable lines. Short columns are padded with empty cells.

    Examples:
        Format and print every line of a fixed-width two-column table:

        >>> from pyscf.pbc.adc.lib import table_lines
        >>> lines = table_lines(
        ...     {'k': [0, 1], 'Energy': ['1.0', '2.0']}, title='ADC', width=24)
        >>> for line in lines:  # doctest: +NORMALIZE_WHITESPACE
        ...     print(line)
        **********ADC***********
            k    ,    Energy
            0    ,      1.0
            1    ,      2.0
        *******End of ADC*******
        >>> [[cell.strip() for cell in row.split(',')] for row in lines[1:-1]]
        [['k', 'Energy'], ['0', '1.0'], ['1', '2.0']]
        >>> all(len(line) == 24 for line in lines)
        True
        >>> table_lines({})
        []

        A missing cell is left empty; width is a minimum and grows when
        the headings or formatted values need more space:

        >>> from pyscf.pbc.adc.lib import table_lines
        >>> rows = table_lines({'k': [0, 1], 'E': ['1.0']}, width=5)
        >>> [cell.strip() for cell in rows[-1].split(',')]
        ['1', '']
    """
    if not data:
        return []
    columns = [[str(x) for x in values] for values in data.values()]
    headings = [str(x) for x in data]
    # Spread unused table width evenly after sizing columns to their contents.
    widths = [max(map(len, [heading, *column]))
              for heading, column in zip(headings, columns)]
    line_width = max(
        LINE_WIDTH if width is None else width,
        sum(widths) + 2 * (len(widths) - 1))
    for index in range(line_width - sum(widths) - 2 * (len(widths) - 1)):
        widths[index % len(widths)] += 1

    def row(values):
        r"""Center one row within the column widths captured by table_lines().

        Args:
            values (sequence[str]): Formatted cells, one per captured column
                width.

        Returns:
            str: Comma-space-separated row. Each cell is centered and padded to
            its assigned width; this helper performs no numeric conversion.
        """
        return ', '.join(
            f'{value:^{size}}' for value, size in zip(values, widths))

    lines = [f'{title:*^{line_width}}'] if title else []
    lines.append(row(headings))
    for index in range(max(map(len, columns), default=0)):
        lines.append(
            row([col[index] if index < len(col) else '' for col in columns]))
    if title:
        lines.append(f"{'End of ' + title:*^{line_width}}")
    return lines


def log_table(adc, data, title=None):
    r"""Write a formatted ADC table through the PySCF INFO logger.

    Args:
        adc: Object supplying stdout and verbose to the PySCF logger.
        data (dict): Ordered headings and formatted cell sequences, as in
            table_lines(). Unequal column lengths are padded with empty cells.
        title (str, optional): Centered title and matching End of <title>
            footer.

    Returns:
        None: Emit table lines when INFO logging is enabled. Width follows
        the module-level LINE_WIDTH setting and expands to fit cell contents.

    Examples:
        >>> import io
        >>> from pyscf.lib import logger
        >>> from pyscf.pbc.adc.lib import log_table
        >>> from types import SimpleNamespace
        >>> output = io.StringIO()
        >>> model = SimpleNamespace(stdout=output, verbose=logger.INFO)
        >>> log_table(model, {'k': [0], 'E': ['1.0']}, 'ADC')
        >>> 'End of ADC' in output.getvalue()
        True
    """
    for line in table_lines(data, title):
        logger.info(adc, line)


_SIGNATURES = {
    't2_1': (1, 1, -1, -1), 't2_2': (1, 1, -1, -1),
    't1_2': (1, -1), 't1_3': (1, -1),
    'r1': (1, -1), 'r2': (1, 1, -1, -1),
    'eo': (0,), 'ev': (0,),
    'oooo': (-1, 1, -1, 1), 'oovv': (-1, 1, -1, 1),
    'ovov': (-1, 1, -1, 1),
    'ovoo': (-1, 1, -1, 1), 'ovvo': (-1, 1, -1, 1),
    'ovvv': (-1, 1, -1, 1), 'vvvv': (-1, -1, 1, 1),
}


@lru_cache(maxsize=None)
def _conjugations(equation, names, output_signs):
    r"""Choose operand conjugations consistent with the desired external
    phases.

    Each tensor name has signed orbital phases in _SIGNATURES. Enumerate
    conjugations of ground tensors until every internal label has net phase
    zero and each output label has its requested sign. Response tensors and
    orbital energies retain their orientation. Successful results are cached.

    Args:
        equation (str): Explicit-output orbital einsum equation, with one
            input string per operand and unique external labels.
        names (tuple[str]): Tensor names represented by those inputs.
        output_signs (tuple[int]): Required phase sign for each output label.
            Inputs must be hashable because this function uses lru_cache.

    Returns:
        tuple[bool]: One conjugation flag per operand; True means conjugate.
        The first valid orientation is returned when several are possible.

    Raises:
        ValueError: No permitted conjugation pattern satisfies phase balance.

    Examples:
        >>> from pyscf.pbc.adc.lib import _conjugations
        >>> _conjugations('Ia,La->IL', ('t1_2', 't1_2'), (1, -1))
        (False, True)
    """
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    labels = set(''.join(inputs))
    choices = [(1,) if name.startswith('r') or name in ('eo', 'ev')
               else (1, -1) for name in names]
    # Internal orbital phases must cancel in a valid contraction.
    for flips in product(*choices):
        weights = Counter()
        for indices, name, flip in zip(inputs, names, flips):
            for label, sign in zip(indices, _SIGNATURES[name]):
                weights[label] += flip * sign
        expected = dict(zip(output, output_signs))
        if all(weights[x] == expected.get(x, 0) for x in labels):
            return tuple(flip < 0 for flip in flips)
    raise ValueError(f'Cannot orient EE contraction {equation}: {names}')


class MomentumContractions:
    r"""Evaluate spin-free EE equations while keeping momentum axes
    independent.

    The context maps each orbital label to a k index, solves the conservation
    constraints of every operand, orients complex conjugations and contracts
    only the allowed orbital blocks. It also caches ground-only blocks and
    intermediates across repeated Davidson-vector applications.

    Args:
        adc (KRADCBase): Reference supplying the k mesh, active energies,
            amplitudes
            and max_memory. A momentum mesh with index zero as its origin is
            assumed by the addition/negation tables.
        eris (_ChemistsERIs, optional): Integral blocks or DF factors. It can
            be omitted for contractions involving amplitudes and energies only.
        kshift (int): External EE momentum sector, default zero.
        t1, t2 (tuple, optional): Amplitude overrides. None uses adc.t1/adc.t2.

    Attributes:
        data (dict): eo/ev energies and t1_2/t1_3/t2_1/t2_2 amplitudes. Sigma
            callers add r1/r2 tensors for the current trial vector. A missing
            perturbative order is represented by None.
        kconserv, add, neg (ndarray): Momentum conservation, addition and
            inversion lookup tables over mesh indices.
        eris: Shared integral container; this context does not own its files.
        kshift, nkpts (int): Momentum sector and mesh size.

    Tensor cache storage is bounded by min(128 MB, 10% of max_memory), with a
    1 MB floor. Momentum-plan storage has a separate quarter-sized budget.
    The context must be rebuilt after changing its reference orbitals or mesh.

    Examples:
        >>> import numpy as np
        >>> from pyscf.pbc import gto, scf
        >>> from pyscf.pbc.adc import kadc_rhf
        >>> from pyscf.pbc.adc.lib import MomentumContractions
        >>> cell = gto.M(
        ...     atom='He 0 0 0',
        ...     a=np.eye(3) * 3,
        ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
        ...     verbose=0)
        >>> kmf = scf.KRHF(
        ...     cell, cell.make_kpts([1, 1, 2]),
        ...     exxdiv=None).density_fit().run()
        >>> mykradcbase = kadc_rhf.KRADCBase(kmf).run(0)
        >>> context = MomentumContractions(mykradcbase)
        >>> term = (1., 'I->I', ('eo',))
        >>> value = context.contract(term, {'I': 0}, (0,))
        >>> value.shape, bool(np.allclose(value, mykradcbase.e_occ[0]))
        ((1,), True)
    """

    def __init__(self, adc, eris=None, kshift=0, t1=None, t2=None):
        r"""Bind reference tensors and initialize bounded momentum/tensor
        caches.

        Args:
            adc (KRADCBase): Reference with initialized amplitude tuples and
                energies.
            eris (_ChemistsERIs, optional): Shared integrals; required only for
                contraction terms that contain integral operands.
            kshift (int): Sector used by response-tensor conservation
                equations.
            t1, t2 (tuple, optional): Override singles/doubles amplitude
                tuples.
                Missing higher-order entries may be None.

        Returns:
            None: Retain tensor references, momentum lookup tables and empty
            LRU
            caches. No full supercell tensor is constructed and no files are
            owned.
        """
        self.adc = adc
        self.eris = eris
        self.kshift = kshift
        self.nkpts = adc.nkpts
        # Use the zero k index as the origin for mesh addition and inversion.
        self.kconserv = adc.khelper.kconserv
        self.add = self.kconserv[:, 0, :]
        self.neg = self.kconserv[0, :, 0]
        # Use the same padded orbital axes for energies and amplitudes.
        if hasattr(adc, 'padded_coeff'):
            coefficients, eo, ev = adc.padded_coeff, adc.e_occ, adc.e_vir
        else:
            from pyscf.pbc.mp.kmp2 import _add_padding
            coefficients, energy = _add_padding(
                adc, adc.mo_coeff, adc.mo_energy)
            eo = np.asarray(energy)[:, :adc.nocc]
            ev = np.asarray(energy)[:, adc.nocc:]
        t1 = adc.t1 if t1 is None else t1
        t2 = adc.t2 if t2 is None else t2
        self.data = {'eo': eo, 'ev': ev, 't2_1': t2[0],
                     't2_2': t2[1], 't1_2': t1[0],
                     't1_3': t1[1] if len(t1) > 1 else None}
        # Budget k-point assignments separately from cached orbital tensors.
        self._plans = OrderedDict()
        self._plans_bytes = 0
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = int(
            min(128, max(1, adc.max_memory * 0.1)) * 1000000.0)

    def _sum(self, terms):
        r"""Evaluate an integer linear combination of k indices modulo the
        mesh.

        Args:
            terms (iterable[tuple[int, int]]): Pairs (k_index, coefficient),
                with
                integer coefficients. Negative coefficients use the inversion
                table.

        Returns:
            int: Mesh index of the sum, starting from origin index zero.
            Repeated
            table addition supports coefficients with magnitude greater than
            one.
        """
        value = 0
        for index, coefficient in terms:
            for term_index in range(abs(coefficient)):
                value = self.add[value, index if coefficient > 0
                                 else self.neg[index]]
        return value

    def _plan(self, equation, names, fixed):
        r"""Enumerate allowed operand momenta for fixed external orbital
        labels.

        For each tensor, its signed k sum equals zero for ground operands or
        kshift for response operands. Propagate constraints with a single
        unknown
        label, then enumerate remaining independent k indices recursively.

        Args:
            equation (str): Orbital einsum expression with explicit output
                labels.
            names (tuple[str]): Operand names defining conservation signatures.
            fixed (dict[str, int]): Known orbital-label to k-index assignments.

        Returns:
            list[tuple[tuple[int]]]: Allowed plans, each holding one complete k
            tuple per operand in input-label order. Contradictory fixed indices
            give an empty list. Plans are cached when they fit the plan budget.
        """
        key = equation, names, tuple(sorted(fixed.items()))
        if key in self._plans:
            plans, cost = self._plans.pop(key)
            self._plans[key] = plans, cost
            return plans
        # Combine repeated orbital labels into one mesh equation per operand.
        inputs = equation.split('->')[0].split(',')
        equations = []
        for indices, name in zip(inputs, names):
            coefficients = Counter()
            for label, sign in zip(indices, _SIGNATURES[name]):
                coefficients[label] += sign
            coefficients = {x: n for x, n in coefficients.items() if n}
            if coefficients:
                rhs = self.kshift if name.startswith('r') else 0
                equations.append((coefficients, rhs))
        labels = set(''.join(inputs))
        plans = []

        def solve(known):
            r"""Propagate one partial k assignment and append its consistent
            completions.

            Args:
                known (dict[str, int]): Current orbital-label assignments.
                    Constraint
                    propagation updates this mapping; recursive branches
                    receive copies.

            Returns:
                None: Append complete operand assignments to the enclosing
                plans list.
                Return immediately if a fully assigned conservation equation
                fails.
                The enclosing scope supplies equations, labels and tensor
                signatures.
            """
            # Solve single unknowns before branching over coupled indices.
            while True:
                changed = False
                for coefficients, rhs in equations:
                    missing = [x for x in coefficients if x not in known]
                    total = self._sum((known[x], n) for x, n in
                                      coefficients.items() if x in known)
                    if not missing:
                        if total != rhs:
                            return
                    elif len(missing) == 1:
                        label = missing[0]
                        sign = coefficients[label]
                        if abs(sign) != 1:
                            continue
                        value = self.add[rhs, self.neg[total]]
                        known[label] = value if sign > 0 else self.neg[value]
                        changed = True
                if not changed:
                    break
            # Branch first on the unknown appearing in the most constraints.
            missing = labels - known.keys()
            if missing:
                label = max(sorted(missing), key=lambda x: sum(
                    x in coefficients for coefficients, operands in equations))
                for kpt in range(self.nkpts):
                    solve(dict(known, **{label: kpt}))
            else:
                plans.append(tuple(
                    tuple((known[x] for x in indices)) for indices in inputs))

        solve(dict(fixed))
        # Retain small plans, evicting the least recently used ones first.
        cost = sum(
            80 + sum((80 + 8 * len(ks) for ks in plan)) for plan in plans)
        limit = self._cache_limit // 4
        if cost <= limit:
            while self._plans and self._plans_bytes + cost > limit:
                equation_key, (path, old_cost) = self._plans.popitem(
                    last=False)
                self._plans_bytes -= old_cost
            self._plans[key] = plans, cost
            self._plans_bytes += cost
        return plans

    def _cached(self, key, build):
        r"""Read or construct a tensor using the bounded least-recently-used
        cache.

        Args:
            key (hashable): Tensor/intermediate identity within this context.
            build (callable): Zero-argument constructor returning an ndarray or
                None.

        Returns:
            ndarray or None: Cached or newly built result. None and arrays
            larger
            than the cache budget are returned without caching. Otherwise evict
            oldest entries as needed. Returned cached arrays are shared
            references
            and should be treated as read-only by callers.
        """
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = build()
        if value is None or value.nbytes > self._cache_limit:
            return value
        # Release older blocks until the new tensor fits the memory budget.
        while (self._cache and
               self._cache_bytes + value.nbytes > self._cache_limit):
            cache_key, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        self._cache[key] = value
        self._cache_bytes += value.nbytes
        return value

    def _block(self, name, kpts):
        r"""Obtain one orbital block, caching only response-independent
        operands.

        Args:
            name (str): Tensor key in data or an ERI block name.
            kpts (tuple[int]): One k index per orbital label of the operand.

        Returns:
            ndarray or None: Selected orbital data. Response names beginning
            with
            r bypass the cache so a later Davidson vector is read immediately;
            missing perturbative amplitudes remain None.
        """
        if name.startswith('r'):
            return self._read_block(name, kpts)
        key = ('block', name, kpts)
        return self._cached(key, lambda: self._read_block(name, kpts))

    def _read_block(self, name, kpts):
        r"""Select an energy, amplitude, response or integral orbital block.

        Args:
            name (str): eo/ev, r1/r2, t1_2/t1_3/t2_1/t2_2, or an integral name.
            kpts (tuple[int]): Operand k assignments. Energies and singles use
                the first index; doubles and ERIs use the first three
                independent
                indices. Any fourth index is fixed by the caller's momentum
                plan.

        Returns:
            ndarray or None: Orbital tensor without leading k axes. ovvv/vvvv
            use
            the owning ERI container's lazy block accessors. None propagates an
            absent perturbative amplitude. This reader does not cache its
            result.
        """
        if name in self.data:
            tensor = self.data[name]
            if tensor is None:
                return None
            if name in ('eo', 'ev'):
                return tensor[kpts[0]]
            if name in ('r1', 't1_2', 't1_3'):
                return tensor[kpts[0]]
            return tensor[kpts[:3]]
        # These virtual blocks may need reconstruction from DF factors.
        if name == 'ovvv':
            return self.eris.ovvv_kpoint_normalized[kpts[:3]]
        if name == 'vvvv':
            return self.eris.vvvv_kpoint_normalized[kpts[:3]]
        return getattr(self.eris, name)[kpts[:3]]

    def contract(self, term, fixed, output_signs):
        r"""Evaluate a term with fixed external momenta and sum allowed
        internal k's.

        Args:
            term (tuple): (coefficient, equation, names), as in kadc_ee_terms.
                equation uses orbital labels only; names is a tuple of tensor
                keys.
            fixed (dict[str, int]): K indices for external labels in the term.
            output_signs (sequence[int]): Desired orbital phase signs, in the
                equation's output order. These select complex conjugations.

        Returns:
            ndarray or scalar or None: Coefficient-weighted orbital
            contraction,
            with axes in output-label order and internal k indices summed.
            Return
            None if no allowed plan contributes or a required tensor is absent.
            Terms linear in one response operand reuse ground-only
            intermediates.

        Examples:
            >>> import numpy as np
            >>> from pyscf.pbc import gto, scf
            >>> from pyscf.pbc.adc import kadc_rhf
            >>> from pyscf.pbc.adc.lib import MomentumContractions
            >>> cell = gto.M(
            ...     atom='He 0 0 0',
            ...     a=np.eye(3) * 3,
            ...     basis=[[0, (1.0, 1.0)], [0, (0.5, 1.0)]],
            ...     verbose=0)
            >>> kmf = scf.KRHF(
            ...     cell, cell.make_kpts([1, 1, 2]),
            ...     exxdiv=None).density_fit().run()
            >>> mykradcbase = kadc_rhf.KRADCBase(kmf).run(0)
            >>> context = MomentumContractions(mykradcbase)
            >>> context.contract((2.0, 'I->I', ('eo',)), {'I': 0}, (0,)).shape
            (1,)
        """
        coefficient, equation, names = term
        conjugate = _conjugations(equation, names, tuple(output_signs))
        plans = self._plan(equation, names, fixed)
        response = [i for i, name in enumerate(names) if name.startswith('r')]
        # Factor linear terms so Davidson vectors can share ground tensors.
        if len(response) == 1 and len(names) > 1:
            return self._linear_contract(
                term, fixed, conjugate, plans, response[0])
        # Sum weighted orbital contractions over allowed k-point assignments.
        value = None
        for momenta in plans:
            operands = [self._block(name, ks)
                        for name, ks in zip(names, momenta)]
            if any(operand is None for operand in operands):
                continue
            operands = [x.conj() if conj else x for x, conj in
                        zip(operands, conjugate)]
            block = einsum(equation, *operands)
            if value is None:
                value = coefficient * block
            else:
                value += coefficient * block
        return value

    def _linear_contract(self, term, fixed, conjugate, plans, response):
        r"""Reuse a ground-only intermediate for a term linear in one response
        tensor.

        First sum ground contractions over momenta not carried by the response
        operand, then contract that intermediate with the current response
        block.
        Cache keys include the equation, conjugations, fixed external momenta
        and
        response k tuple, so repeated Davidson vectors can reuse the ground
        work.

        Args:
            term (tuple): Coefficient, einsum equation and operand-name tuple.
            fixed (dict[str, int]): Fixed external momentum assignments.
            conjugate (sequence[bool]): Precomputed operand conjugation flags.
            plans (list[tuple]): Allowed momentum assignments from _plan().
            response (int): Position of the sole r-prefixed operand in term.

        Returns:
            ndarray or scalar or None: Coefficient-weighted linear action with
            output-label axes, or None when no nonmissing intermediate
            contributes.
            Ground intermediates obey the same bounded cache policy as blocks.
        """
        coefficient, equation, names = term
        inputs, output = equation.split('->')
        inputs = inputs.split(',')
        ground = [i for i in range(len(names)) if i != response]
        # Keep the ground indices needed by the response or final result.
        external = set(inputs[response] + output)
        labels = ''.join(dict.fromkeys(''.join((inputs[i] for i in ground))))
        labels = ''.join(x for x in labels if x in external)
        ground_eq = ','.join(inputs[i] for i in ground) + '->' + labels
        final_eq = inputs[response] + ',' + labels + '->' + output
        # Sum internal momenta once for each distinct response block.
        groups = {}
        for momenta in plans:
            groups.setdefault(momenta[response], []).append(momenta)
        value = None
        for response_kpts, assignments in groups.items():
            key = ('intermediate', equation, names, tuple(conjugate),
                   tuple(sorted(fixed.items())), response_kpts)

            def build():
                r"""Sum the captured ground-operand assignments for one
                response k tuple.

                Returns:
                    ndarray or None: Ground-only intermediate over the captured
                    surviving
                    labels. Skip assignments with absent perturbative operands
                    and return
                    None if all are absent. Enclosing variables supply
                    equations, signs,
                    operand positions and allowed momentum assignments.
                """
                # Sum the ground factors for this response momentum tuple.
                intermediate = None
                for momenta in assignments:
                    operands = [self._block(
                        names[i], momenta[i]) for i in ground]
                    if any(x is None for x in operands):
                        continue
                    operands = [x.conj() if conjugate[i] else x for i, x in
                                zip(ground, operands)]
                    block = einsum(ground_eq, *operands)
                    if intermediate is None:
                        intermediate = block.copy()
                    else:
                        intermediate += block
                return intermediate

            # Apply the trial vector to the cached ground intermediate.
            intermediate = self._cached(key, build)
            if intermediate is None:
                continue
            vector = self._block(names[response], response_kpts)
            block = einsum(final_eq, vector, intermediate)
            if value is None:
                value = coefficient * block
            else:
                value += coefficient * block
        return value
