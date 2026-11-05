#!/usr/bin/env python3
"""Static release-assembly comparison for the suit_table A2 experiment.

Reads the assembly that `build_variants.py` pre-generated with
`cargo rustc --release --lib -- --emit=asm`; it never builds.

Symbols carry per-build hashes, so everything is grouped by **logical function
name** (`suit_table`, `standard_deficit`, ...) rather than by raw symbol string.
If LLVM inlines `suit_table` in one variant but not another, the missing side is
filled from the enclosing caller so every variant always contributes comparable
entries; a variant that ends up with nothing analysed is a hard error rather
than an empty report.

The report is purely static: instruction counts, mnemonic mix, divisions and
multiplications, branch counts, backward branch edges with loop-body sizes, and
the located instructions behind the A2 differences (the `a + b` reservation, the
`capacity - reserved` free-capacity subtraction, the capacity/`break` compares).
No runtime branch behavior is observed, so no hit/miss or misprediction claim is
derived from it.

Usage:
  python -B asm_report.py <artifact-dir> <output.json>
"""
import argparse
import difflib
import json
from pathlib import Path
import re

CONDITIONAL = re.compile(r'^(j(?!mp)[a-z]+|loop[a-z]*)$')
LABEL = re.compile(r'^(\.L[a-zA-Z0-9_.]+):$')
BRANCH_TARGET = re.compile(r'^(?:j[a-z]+|loop[a-z]*)\s+(\S+)')
# Longest match wins, so `standard_deficit` is not grouped as `deficit`.
LOGICAL_NAMES = ('suit_table', 'seven_pairs_deficit', 'standard_deficit', 'deficit',
                 'max_discard_fan', 'suit_cache')
# Operand-size suffixes (`subq`, `imull`, ...) are stripped before family lookup.
FAMILIES = ('sub', 'add', 'cmp', 'test', 'imul', 'mul', 'div', 'idiv', 'lea')
FEATURES = {'sub': 'sub', 'add': 'add', 'cmp': 'cmp', 'div': 'div', 'idiv': 'div',
            'imul': 'mul', 'mul': 'mul', 'lea': 'lea', 'je': 'branch', 'jne': 'branch',
            'jae': 'branch', 'jbe': 'branch', 'jb': 'branch', 'ja': 'branch',
            'jl': 'branch', 'jg': 'branch', 'jle': 'branch', 'jge': 'branch'}


def family(mnemonic):
    """`subq` -> `sub`, `imull` -> `imul`; anything else is returned as is."""
    for prefix in FAMILIES:
        if mnemonic.startswith(prefix):
            return prefix
    return mnemonic


def functions(text):
    """Splits column-zero symbol labels from their indented bodies."""
    result = {}
    current = None
    body = []
    for line in text.splitlines():
        if line and not line[0].isspace() and line.endswith(':') and not line.startswith('.'):
            if current is not None:
                result[current] = body
            current = line[:-1]
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        result[current] = body
    return result


def logical_name(symbol):
    matches = [name for name in LOGICAL_NAMES if name in symbol]
    return max(matches, key=len) if matches else symbol


def instruction_lines(body):
    instructions = []
    labels = {}
    for line in body:
        stripped = line.strip()
        match = LABEL.match(stripped)
        if match:
            labels[match.group(1)] = len(instructions)
        elif stripped and not stripped.startswith('.') and not stripped.startswith('#'):
            instructions.append(stripped)
    return instructions, labels


def metrics(body):
    instructions, labels = instruction_lines(body)
    conditional = unconditional = other = calls = divisions = multiplies = 0
    unresolved = 0
    loops = []
    for index, instruction in enumerate(instructions):
        mnemonic = instruction.split()[0]
        if mnemonic.startswith('call'):
            calls += 1
            continue
        operation = family(mnemonic)
        if operation in ('div', 'idiv'):
            divisions += 1
        if operation in ('imul', 'mul'):
            multiplies += 1
        if mnemonic == 'jmp':
            unconditional += 1
        elif CONDITIONAL.match(mnemonic):
            conditional += 1
        elif mnemonic.startswith('j'):
            other += 1
        if mnemonic.startswith('j') or mnemonic.startswith('loop'):
            match = BRANCH_TARGET.match(instruction)
            if match and match.group(1) in labels and labels[match.group(1)] <= index:
                start = labels[match.group(1)]
                loops.append({'label': match.group(1), 'start': start,
                              'end': index, 'body_instructions': index - start + 1})
            elif not match or match.group(1) not in labels:
                unresolved += 1
    if not instructions:
        raise SystemExit('empty instruction body; the assembly parser found no code')
    return {'instructions': len(instructions),
            'conditional_branches': conditional,
            'unconditional_jumps': unconditional,
            'other_branches': other,
            'calls': calls,
            'divisions': divisions,
            'multiplications': multiplies,
            'backward_jumps_loop_edges': len(loops),
            'unresolved_branch_targets': unresolved,
            'loops': loops[:8]}


def evidence(body, limit=24):
    """Locates instructions behind the A2 bounds instead of only counting them."""
    instructions, _ = instruction_lines(body)
    found = {}
    for index, instruction in enumerate(instructions):
        mnemonic = instruction.split()[0]
        feature = FEATURES.get(family(mnemonic))
        if feature is None:
            continue
        beginning = max(0, index - 1)
        found.setdefault(feature, []).append(
            {'index': index, 'context': instructions[beginning:index + 2]})
    return {feature: {'count': len(entries), 'samples': entries[:limit]}
            for feature, entries in sorted(found.items())}


def pick(grouped, name):
    """Largest (most inlined) body for a logical name is the comparable one."""
    candidates = grouped.get(name, {})
    if not candidates:
        return None
    return max(candidates, key=lambda symbol: len(candidates[symbol]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite existing evidence')
    root = args.artifacts.resolve()
    builds = json.loads((root / 'builds.json').read_text(encoding='utf-8'))
    variants = list(builds['variants'])

    # Raw symbol -> body per variant, then grouped by logical function name.
    bodies = {name: {} for name in variants}
    for name in variants:
        for path in sorted((root / 'asm' / name).glob('*.s')):
            for symbol, body in functions(path.read_text(encoding='utf-8')).items():
                bodies[name][symbol] = body
    grouped = {name: {} for name in variants}
    for name in variants:
        for symbol, body in bodies[name].items():
            # Unwind/debug data labels share the column-zero label syntax but are
            # not functions; only bodies with real instructions are code.
            if not instruction_lines(body)[0]:
                continue
            grouped[name].setdefault(logical_name(symbol), {})[symbol] = body

    # Fill variants where a target was inlined, so every variant contributes a
    # comparable entry for the same logical function.
    filled_from = {}
    names = sorted({key for name in variants for key in grouped[name]})
    for logical in names:
        missing = [name for name in variants if logical not in grouped[name]]
        if len(missing) == len(variants):
            continue
        for name in missing:
            for fallback in ('standard_deficit', 'deficit', 'max_discard_fan'):
                if grouped[name].get(fallback):
                    grouped[name][logical] = grouped[name][fallback]
                    filled_from.setdefault(logical, {})[name] = fallback
                    break

    asm_dir = args.output.parent / 'asm'
    asm_dir.mkdir(exist_ok=True)
    report = {'protocol': 'suit-table-a2-asm-v1',
              'static_only': True,
              'note': ('Static instruction/branch/loop structure only. Assembly was '
                       'generated before measurement with the same release profile, '
                       'toolchain and flags but by a separate `--emit=asm` invocation. '
                       'It does not observe runtime branches, so no branch-hit, '
                       'misprediction or dynamic count claim follows from it.'),
              'mainline': builds['mainline'],
              'symbols_by_variant': {name: {'count': len(bodies[name]),
                                            'project_symbols': sorted(
                                                symbol for symbol in bodies[name]
                                                if logical_name(symbol) in LOGICAL_NAMES)}
                                     for name in variants},
              'logical_functions_by_variant': {name: sorted(grouped[name]) for name in variants},
              'filled_from_enclosing_caller': filled_from,
              'variants': {}, 'comparisons': {}, 'missing': {}}

    for name in variants:
        report['variants'][name] = {}
        for logical, symbols in sorted(grouped[name].items()):
            if logical not in LOGICAL_NAMES:
                continue
            report['variants'][name][logical] = {}
            for symbol, body in symbols.items():
                report['variants'][name][logical][symbol] = {
                    'metrics': metrics(body), 'evidence': evidence(body)}
                short = re.sub(r'[^A-Za-z0-9_]+', '_', symbol)[-60:]
                (asm_dir / f'{name}-{short}.s').write_text('\n'.join(body) + '\n',
                                                           encoding='utf-8')

    # Only project functions are comparable targets; unrelated library symbols
    # never enter the diffs (their names also contain path-invalid characters).
    targets = [logical for logical in LOGICAL_NAMES
               if all(pick(grouped[name], logical) for name in variants)]
    report['parser_validation'] = {
        'parsed_symbols': sum(len(bodies[name]) for name in variants),
        'parsed_instructions': sum(entry['metrics']['instructions']
                                   for name in variants
                                   for logical in report['variants'][name]
                                   for entry in report['variants'][name][logical].values()),
        'functions_with_loop_edges': sum(1 for name in variants
                                         for logical in report['variants'][name]
                                         for entry in report['variants'][name][logical].values()
                                         if entry['metrics']['backward_jumps_loop_edges']),
        'unresolved_branch_targets': sum(entry['metrics']['unresolved_branch_targets']
                                         for name in variants
                                         for logical in report['variants'][name]
                                         for entry in report['variants'][name][logical].values()),
    }
    if not report['parser_validation']['parsed_instructions']:
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        raise SystemExit('assembly parser produced no instructions; refusing to report')
    report['comparable_logical_functions'] = targets
    if not targets:
        report['missing'] = {name: sorted(grouped[name]) for name in variants}
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        raise SystemExit('no logical function is extractable in all variants')

    for logical in targets:
        baseline = grouped['M'][logical][pick(grouped['M'], logical)]
        for name in variants:
            if name == 'M':
                continue
            candidate = grouped[name][logical][pick(grouped[name], logical)]
            diff = list(difflib.unified_diff(baseline, candidate,
                                             fromfile=f'M:{logical}',
                                             tofile=f'{name}:{logical}',
                                             lineterm='', n=3))
            path = asm_dir / f'diff-M-vs-{name}-{logical}.txt'
            path.write_text('\n'.join(diff) + '\n', encoding='utf-8')
            report['comparisons'][f'M-vs-{name}-{logical}'] = {
                'diff': str(path),
                'added_lines': sum(1 for line in diff
                                   if line.startswith('+') and not line.startswith('+++')),
                'removed_lines': sum(1 for line in diff
                                     if line.startswith('-') and not line.startswith('---')),
                'metrics': {'M': report['variants']['M'][logical][pick(grouped['M'], logical)]
                            ['metrics'],
                            name: report['variants'][name][logical][pick(grouped[name], logical)]
                            ['metrics']},
            }
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    summary = {name: {logical: entry['metrics'] for logical, symbols in
                      report['variants'][name].items() for entry in
                      [symbols[pick(grouped[name], logical)]]}
               for name in variants}
    print(json.dumps({'comparable_logical_functions': targets,
                      'metrics': summary,
                      'diffs': sorted(report['comparisons'])}, indent=2), flush=True)


if __name__ == '__main__':
    main()
