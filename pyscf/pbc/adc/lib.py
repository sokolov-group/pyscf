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
    """Contract orbital tensors with optimized paths by default."""
    kwargs.setdefault('optimize', True)
    return pyscf_lib.einsum(subscripts, *operands, **kwargs)


def check_implemented(method, supported=IMPLEMENTED_METHODS):
    """Normalize an ADC method name and reject unsupported approximations.

    Args:
        method (str): Requested method, case insensitive.
        supported (tuple[str]): Implemented approximations.

    Returns:
        str: The normalized method name.
    """
    method = method.lower()
    if method not in supported:
        raise NotImplementedError(method)
    return method


def validate(adc):
    """Apply common method validation and PySCF's verbosity-aware sanity
    check."""
    adc.method = check_implemented(adc.method)
    adc.type_ = adc.type_.lower()
    if adc.verbose >= logger.WARN:
        adc.check_sanity()


def table_lines(data, title=None, width=None):
    """Format columns as a comma-separated table, without writing to stdout.

    Args:
        data (dict): Ordered column headings and sequences of formatted cells.
        title (str, optional): Heading; the footer reads ``End of <title>``.
        width (int, optional): Minimum table width; defaults to LINE_WIDTH.

    Returns:
        list[str]: Printable lines. Short columns are padded with empty cells.
    """
    if not data:
        return []
    columns = [[str(x) for x in values] for values in data.values()]
    headings = [str(x) for x in data]
    widths = [max(map(len, [heading, *column]))
              for heading, column in zip(headings, columns)]
    line_width = max(LINE_WIDTH if width is None else width,
                     sum(widths) + 2 * (len(widths) - 1))
    for index in range(line_width - sum(widths) - 2 * (len(widths) - 1)):
        widths[index % len(widths)] += 1

    def row(values):
        return ', '.join(f'{value:^{size}}'
                         for value, size in zip(values, widths))

    lines = [f'{title:*^{line_width}}'] if title else []
    lines.append(row(headings))
    for index in range(max(map(len, columns), default=0)):
        lines.append(row([col[index] if index < len(col) else ''
                          for col in columns]))
    if title:
        lines.append(f'{"End of " + title:*^{line_width}}')
    return lines


def log_table(adc, data, title=None):
    """Write a common ADC table through PySCF's logger."""
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
    """Orient bra/ket operands so every internal orbital cancels."""
    inputs, output = equation.split('->')
    inputs = inputs.split(',')
    labels = set(''.join(inputs))
    choices = [(1,) if name.startswith('r') or name in ('eo', 'ev')
               else (1, -1) for name in names]
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
    """Execute spin-free equations on one ADC object's k-point blocks."""

    def __init__(self, adc, eris=None, kshift=0, t1=None, t2=None):
        self.adc = adc
        self.eris = eris
        self.kshift = kshift
        self.nkpts = adc.nkpts
        self.kconserv = adc.khelper.kconserv
        self.add = self.kconserv[:, 0, :]
        self.neg = self.kconserv[0, :, 0]
        if hasattr(adc, 'padded_coeff'):
            _, eo, ev = (adc.padded_coeff,
                adc.e_occ, adc.e_vir)
        else:
            from pyscf.pbc.mp.kmp2 import _add_padding
            _, energy = _add_padding(adc, adc.mo_coeff, adc.mo_energy)
            eo = np.asarray(energy)[:, :adc.nocc]
            ev = np.asarray(energy)[:, adc.nocc:]
        t1 = adc.t1 if t1 is None else t1
        t2 = adc.t2 if t2 is None else t2
        self.data = {'eo': eo, 'ev': ev, 't2_1': t2[0],
                     't2_2': t2[1], 't1_2': t1[0],
                     't1_3': t1[1] if len(t1) > 1 else None}
        self._plans = OrderedDict()
        self._plans_bytes = 0
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = int(min(128, max(1, adc.max_memory * 0.1)) * 1e6)

    def _sum(self, terms):
        value = 0
        for index, coefficient in terms:
            for _ in range(abs(coefficient)):
                value = self.add[value, index if coefficient > 0
                                 else self.neg[index]]
        return value

    def _plan(self, equation, names, fixed):
        key = equation, names, tuple(sorted(fixed.items()))
        if key in self._plans:
            plans, cost = self._plans.pop(key)
            self._plans[key] = plans, cost
            return plans
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
            missing = labels - known.keys()
            if missing:
                label = max(sorted(missing), key=lambda x: sum(
                    x in coefficients for coefficients, _ in equations))
                for kpt in range(self.nkpts):
                    solve(dict(known, **{label: kpt}))
            else:
                plans.append(tuple(tuple(known[x] for x in indices)
                                   for indices in inputs))

        solve(dict(fixed))
        cost = sum(80 + sum(80 + 8 * len(ks) for ks in plan)
                   for plan in plans)
        limit = self._cache_limit // 4
        if cost <= limit:
            while self._plans and self._plans_bytes + cost > limit:
                _, (_, old_cost) = self._plans.popitem(last=False)
                self._plans_bytes -= old_cost
            self._plans[key] = plans, cost
            self._plans_bytes += cost
        return plans

    def _cached(self, key, build):
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = build()
        if value is None or value.nbytes > self._cache_limit:
            return value
        while (self._cache and
               self._cache_bytes + value.nbytes > self._cache_limit):
            _, old = self._cache.popitem(last=False)
            self._cache_bytes -= old.nbytes
        self._cache[key] = value
        self._cache_bytes += value.nbytes
        return value

    def _block(self, name, kpts):
        if name.startswith('r'):
            return self._read_block(name, kpts)
        key = ('block', name, kpts)
        return self._cached(key, lambda: self._read_block(name, kpts))

    def _read_block(self, name, kpts):
        if name in self.data:
            tensor = self.data[name]
            if tensor is None:
                return None
            if name in ('eo', 'ev'):
                return tensor[kpts[0]]
            if name in ('r1', 't1_2', 't1_3'):
                return tensor[kpts[0]]
            return tensor[kpts[:3]]
        if name == 'ovvv':
            return self.eris.ovvv_blocks[kpts[:3]]
        if name == 'vvvv':
            return self.eris.vvvv_blocks[kpts[:3]]
        return getattr(self.eris, name)[kpts[:3]]

    def contract(self, term, fixed, output_signs):
        """Evaluate a term for specified external orbital k points."""
        coefficient, equation, names = term
        conjugate = _conjugations(equation, names, tuple(output_signs))
        plans = self._plan(equation, names, fixed)
        response = [i for i, name in enumerate(names) if name.startswith('r')]
        if len(response) == 1 and len(names) > 1:
            return self._linear_contract(term, fixed, conjugate, plans,
                                         response[0])
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
        """Reuse ground-only intermediates across Davidson vectors.

        Internal k sums and orbital sums that do not touch the response
        vector are evaluated once, subject to the bounded cache budget.
        """
        coefficient, equation, names = term
        inputs, output = equation.split('->')
        inputs = inputs.split(',')
        ground = [i for i in range(len(names)) if i != response]
        external = set(inputs[response] + output)
        labels = ''.join(dict.fromkeys(''.join(inputs[i] for i in ground)))
        labels = ''.join(x for x in labels if x in external)
        ground_eq = ','.join(inputs[i] for i in ground) + '->' + labels
        final_eq = inputs[response] + ',' + labels + '->' + output
        groups = {}
        for momenta in plans:
            groups.setdefault(momenta[response], []).append(momenta)
        value = None
        for response_kpts, assignments in groups.items():
            key = ('intermediate', equation, names, tuple(conjugate),
                   tuple(sorted(fixed.items())), response_kpts)

            def build():
                intermediate = None
                for momenta in assignments:
                    operands = [self._block(names[i],
                        momenta[i]) for i in ground]
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
