#!/usr/bin/env python3
"""Isolated microbenchmark and counting model for the `suit_table` experiment.

Timing calls the unmodified production function text extracted from a measured
variant's `src/policy/analysis.rs`, so nothing is instrumented and the inputs
and the returned table are passed through `std::hint::black_box`.

Counting uses the same text with counters injected at two statement sites that
exist verbatim in all three variants; that model is never timed and also
reproduces the full `SuitTable`, so it cannot drift away from the real function.

The counting sites are definition-sensitive and must be stated explicitly:

* `accepted` — immediately before `let extra = ...`, i.e. every transition that
  passed every validity check. Identical across variants by construction.
* `considered` — at the loop that enumerates the *leaf* transition candidate,
  i.e. the innermost loop whose body contains the `accepted` site. `M` and `A2`
  enumerate `sets -> ... -> sequences -> triplets -> pair`, so their leaf loop is
  the `pair` loop; archived full `A` enumerates `pair -> sequences -> triplets`,
  so its leaf loop is the `triplets` loop. Counting any enclosing loop instead
  compares different units and is not comparable across variants.

The same injection also tallies every enclosing transition loop (`rank`,
`active` state, `pair`, `sequences`, `triplets`) so the per-layer enumeration
can be read back. All counters are `u64`, never timed, and printed after the run.

No production file, public API or benchmark driver is added.

Build phase:
  python -B microbench.py --prepare <artifact-dir>
  python -B microbench.py --prepare-counts <artifact-dir>   # corrected counting only

Measurement phase (serial, nothing else running):
  python -B microbench.py --measure <artifact-dir> <output.json>
  python -B microbench.py --count <artifact-dir> <output.json> <01-microbench.json>
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

VARIANTS = ('M', 'MA', 'A2')
# Pre-fixed interleaved slot order; the same rotation is used for timing and
# recorded for every invocation.
SLOT_ORDERS = {
    1: ('M', 'MA', 'A2'),
    2: ('MA', 'A2', 'M'),
    3: ('A2', 'M', 'MA'),
    4: ('M', 'MA', 'A2'),
    5: ('MA', 'A2', 'M'),
}
INPUTS = [
    ('empty-cap4-s4', [0, 0, 0, 0, 0, 0, 0, 0, 0], [4] * 9, 4),
    ('typical-cap4-s4', [1, 1, 1, 1, 1, 1, 1, 0, 0], [4] * 9, 4),
    ('dense-cap4-s4', [4, 4, 4, 0, 0, 0, 0, 0, 0], [4] * 9, 4),
    ('wide-cap4-s4', [2, 1, 2, 1, 2, 1, 2, 1, 2], [4] * 9, 4),
    ('empty-cap4-s3', [0, 0, 0, 0, 0, 0, 0, 0, 0], [4] * 9, 3),
    ('typical-cap4-s2', [1, 1, 1, 1, 1, 1, 1, 0, 0], [4] * 9, 2),
    ('typical-cap4-s1', [1, 1, 1, 1, 1, 1, 1, 0, 0], [4] * 9, 1),
    ('typical-cap4-s0', [1, 1, 1, 1, 1, 1, 1, 0, 0], [4] * 9, 0),
    ('peng-cap1-s4', [1, 1, 1, 1, 0, 0, 0, 0, 0], [1, 4, 4, 4, 4, 4, 4, 4, 4], 4),
    ('gang-cap0-front-s4', [0, 1, 1, 1, 1, 1, 1, 0, 0], [0, 4, 4, 4, 4, 4, 4, 4, 4], 4),
    ('tail-pressure-s4', [1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 4, 4, 4, 4, 4, 4], 4),
    ('missing-suit-cap0-s4', [1, 1, 1, 1, 1, 1, 1, 1, 1], [0] * 9, 4),
]

SIGNATURE = ('fn suit_table(hand: &[u8; 9], capacity: &[u8; 9], target_sets: usize) '
             '-> SuitTable {')
ACCEPT_ANCHOR = 'let extra = (count as u8).saturating_sub(hand[rank]);'
LOOP_HEADER = re.compile(r'for\s+(?:&)?(\w+)\s+in\s+[^\n{]*\{')
# Per-layer tallies; `considered` is placed at the leaf (innermost) loop.
LOOP_FIELDS = {
    'rank': 'ranks',
    'active_state': 'states',
    'sets': 'sets',
    'pair': 'pairs',
    'sequences': 'sequences',
    'triplets': 'triplets',
}

PREAMBLE = '''\
use std::hint::black_box;
use std::time::Instant;

const INF: u8 = u8::MAX;
type SuitTable = [[u8; 2]; 5];

#[allow(dead_code)]
struct Counters {
    ranks: u64,
    states: u64,
    sets: u64,
    pairs: u64,
    sequences: u64,
    triplets: u64,
    accepted: u64,
    considered: u64,
}

fn digest(table: &SuitTable) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for entry in table.iter().flatten() {
        hash ^= u64::from(*entry);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct Input {
    name: &'static str,
    hand: [u8; 9],
    capacity: [u8; 9],
    sets: usize,
}

static INPUTS: &[Input] = &[
'''

TIMING_MAIN = '''
fn main() {
    let mut iters = 10_000usize;
    let mut rounds = 5usize;
    let mut warmup = 1_000usize;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let value = args[index + 1].parse::<usize>().expect("numeric option value");
        match args[index].as_str() {
            "--iters" => iters = value,
            "--rounds" => rounds = value,
            "--warmup" => warmup = value,
            other => panic!("unknown option {other}"),
        }
        index += 2;
    }
    println!("mode timing iters {iters} rounds {rounds} warmup {warmup}");
    let mut sink = 0u64;
    for input in INPUTS {
        println!("digest {} {:016x}",
                 input.name,
                 digest(&suit_table(&input.hand, &input.capacity, input.sets)));
        for _ in 0..warmup {
            sink ^= digest(&suit_table(black_box(&input.hand), black_box(&input.capacity),
                                      black_box(input.sets)));
        }
        for round in 1..=rounds {
            let start = Instant::now();
            for _ in 0..iters {
                let table = suit_table(black_box(&input.hand), black_box(&input.capacity),
                                       black_box(input.sets));
                black_box(&table);
            }
            let elapsed = start.elapsed().as_nanos() as f64;
            println!("timing {} {} {:.4}", input.name, round, elapsed / iters as f64);
        }
    }
    black_box(sink);
}
'''

COUNT_MAIN = '''
fn main() {
    for input in INPUTS {
        let mut counters = Counters { ranks: 0, states: 0, sets: 0, pairs: 0, sequences: 0,
                                      triplets: 0, accepted: 0, considered: 0 };
        let table = suit_table(&input.hand, &input.capacity, input.sets, &mut counters);
        println!("count {} {} {} {:016x} ranks={} states={} sets={} pairs={} sequences={} triplets={}",
                 input.name, counters.accepted, counters.considered, digest(&table),
                 counters.ranks, counters.states, counters.sets, counters.pairs,
                 counters.sequences, counters.triplets);
        print!("table {}", input.name);
        for entry in table.iter().flatten() {
            print!(" {}", entry);
        }
        println!();
    }
}
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_suit_table(source_path):
    """Returns the exact `fn suit_table` text of a measured variant."""
    text = source_path.read_text(encoding='utf-8')
    if text.count('fn suit_table(') != 1:
        raise SystemExit(f'{source_path}: expected exactly one suit_table definition')
    start = text.index('fn suit_table(')
    end = text.index('\n}\n', start) + len('\n}')
    body = text[start:end]
    if body.count(ACCEPT_ANCHOR) != 1:
        raise SystemExit(f'{source_path}: accept-anchor not unique in suit_table')
    if body.count(SIGNATURE) != 1:
        raise SystemExit(f'{source_path}: unexpected suit_table signature')
    return body


def brace_depths(text):
    """Brace depth after each character; ignores line comments and literals."""
    depths = [0] * (len(text) + 1)
    depth = 0
    index = 0
    in_line_comment = in_string = in_char = False
    while index < len(text):
        char = text[index]
        if in_line_comment:
            if char == '\n':
                in_line_comment = False
        elif in_string:
            if char == '\\':
                index += 1
            elif char == '"':
                in_string = False
        elif in_char:
            if char == '\\':
                index += 1
            elif char == "'":
                in_char = False
        elif char == '/' and text[index + 1:index + 2] == '/':
            in_line_comment = True
        elif char == '"':
            in_string = True
        elif char == "'":
            in_char = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        depths[index + 1] = depth
        index += 1
    return depths


def instrument(body):
    """Injects the counters, `considered` at the leaf transition loop.

    The leaf loop is the innermost `for` loop whose body contains the accept
    anchor, so the unit stays one enumerated transition candidate regardless of
    each variant's loop order. Every enclosing transition loop also gets its own
    tally. Returns `(instrumented_text, leaf_loop_field)`.
    """
    accept_at = body.index(ACCEPT_ANCHOR)
    depths = brace_depths(body)
    loops = []
    for match in LOOP_HEADER.finditer(body):
        if match.end() > accept_at:
            continue
        field = LOOP_FIELDS.get(match.group(1))
        if field is None:
            raise SystemExit(f'unmapped transition loop `{match.group(1)}` before the anchor')
        loops.append((match.end(), field))
    if not loops:
        raise SystemExit('no transition loop before the accept anchor')
    leaf_depth = depths[accept_at]
    leaf_ends = [end for end, _ in loops if depths[end] == leaf_depth]
    if not leaf_ends:
        raise SystemExit('no loop opens the block holding the accept anchor')
    leaf_end = max(leaf_ends)
    leaf_fields = [field for end, field in loops if end == leaf_end]
    if len(leaf_fields) != 1:
        raise SystemExit(f'ambiguous leaf loop: {leaf_fields}')

    inserts = {}
    for end, field in loops:
        inserts.setdefault(end, []).append(f' counters.{field} += 1;')
    inserts.setdefault(leaf_end, []).append(' counters.considered += 1;')
    instrumented = body
    for position in sorted(inserts, reverse=True):
        instrumented = (instrumented[:position] + ''.join(inserts[position])
                        + instrumented[position:])

    accept_at = instrumented.index(ACCEPT_ANCHOR)
    instrumented = (instrumented[:accept_at] + 'counters.accepted += 1; '
                    + instrumented[accept_at:])
    if instrumented.count(SIGNATURE) != 1:
        raise SystemExit('unexpected suit_table signature count')
    instrumented = instrumented.replace(
        SIGNATURE,
        'fn suit_table(hand: &[u8; 9], capacity: &[u8; 9], target_sets: usize, '
        'counters: &mut Counters) -> SuitTable {')
    return instrumented, leaf_fields[0]


COUNTER_FIELDS = ('ranks', 'states', 'sets', 'pairs', 'sequences', 'triplets')


def parse_count_line(line):
    """Parses both the positional count line and its `key=value` tallies."""
    parts = line.split()
    if len(parts) < 5 or parts[0] != 'count':
        return None
    record = {'accepted': int(parts[2]), 'considered': int(parts[3]), 'digest': parts[4]}
    for token in parts[5:]:
        if '=' in token:
            key, value = token.split('=', 1)
            record[key] = int(value)
    return parts[1], record


def parse_table_line(line):
    parts = line.split()
    if len(parts) != 12 or parts[0] != 'table':
        return None
    return parts[1], [int(value) for value in parts[2:]]


def harness(body, counting):
    inputs = ''.join(
        f'    Input {{ name: "{name}", hand: {hand!r}, capacity: {capacity!r}, sets: {sets} }},\n'.replace("'", '')
        for name, hand, capacity, sets in INPUTS
    )
    return (PREAMBLE + inputs + '];\n\n' + body + '\n'
            + (COUNT_MAIN if counting else TIMING_MAIN))


def prepare(root):
    builds = json.loads((root / 'builds.json').read_text(encoding='utf-8'))
    source_dir = root / 'micro'
    source_dir.mkdir()
    prepare_log = {'rustc': subprocess.check_output(['rustc', '-V'], text=True), 'variants': {}}
    extension = '.exe' if (root / 'M.exe').exists() else ''
    for name in VARIANTS:
        body = extract_suit_table(root / 'sources' / f'{name}.analysis.rs')
        entry = {'suit_table_sha256': hashlib.sha256(body.encode()).hexdigest(),
                 'binary_sha256': {}, 'commands': {}}
        count_text, leaf = instrument(body)
        entry['considered_site'] = leaf
        for mode, text in (('timing', harness(body, False)), ('count', harness(count_text, True))):
            path = source_dir / f'{name}-{mode}.rs'
            path.write_text(text, encoding='utf-8')
            binary = root / f'micro-{name}-{mode}{extension}'
            command = ['rustc', '--edition=2021', '-O', str(path), '-o', str(binary)]
            subprocess.run(command, check=True)
            entry['commands'][mode] = command
            entry['binary_sha256'][mode] = sha256(binary)
        prepare_log['variants'][name] = entry
        print(f'prepared microbench {name}', flush=True)
    prepare_log['builds_mainline'] = builds['mainline']
    (source_dir / 'prepare.json').write_text(
        json.dumps(prepare_log, indent=2) + '\n', encoding='utf-8')


def measure(root, output_path):
    if output_path.exists():
        raise SystemExit('refusing to overwrite existing evidence')
    prepared = json.loads((root / 'micro' / 'prepare.json').read_text(encoding='utf-8'))
    # Fail closed: the timed binaries must be the ones prepared and hashed.
    for name, entry in prepared['variants'].items():
        for mode, expected in entry['binary_sha256'].items():
            actual = sha256(root / f'micro-{name}-{mode}.exe')
            if actual != expected:
                raise SystemExit(f'micro-{name}-{mode}.exe hash {actual} differs from '
                                 f'prepare.json {expected}; refusing to measure')
    document = {'protocol': 'suit-table-a2-bench-v1',
                'mainline': prepared['builds_mainline'],
                'rustc': prepared['rustc'],
                'codegen_note': ('`rustc -O` is the project release opt-level (3); this is a '
                                 'standalone single-function build, so only the '
                                 'across-variant comparison is meaningful.'),
                'iters': 10_000, 'rounds': 5, 'warmup': 1_000,
                'inputs': [name for name, *_ in INPUTS],
                'slot_orders': {str(k): list(v) for k, v in SLOT_ORDERS.items()},
                'variants': prepared['variants'],
                'environment': {'cpu_count': __import__('os').cpu_count(),
                                'platform': __import__('platform').platform()},
                'runs': []}

    def save():
        output_path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    def invoke(name, mode, extra):
        binary = root / ('micro-' + name + '-' + mode + ('.exe' if (root / 'M.exe').exists() else ''))
        command = [str(binary)] + extra
        record = {'variant': name, 'mode': mode, 'command': command,
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        document['runs'].append(record)
        save()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        record.update(returncode=completed.returncode,
                      stdout=completed.stdout, stderr=completed.stderr)
        save()
        if completed.returncode != 0:
            raise SystemExit('failed run retained; experiment aborted, no automatic retry')
        return completed.stdout

    timing = {name: {} for name in VARIANTS}
    digests = {name: {} for name in VARIANTS}
    for slot, order in SLOT_ORDERS.items():
        for name in order:
            text = invoke(name, 'timing',
                          ['--iters', '10000', '--rounds', '5', '--warmup', '1000'])
            for line in text.splitlines():
                parts = line.split()
                if parts[0] == 'timing' and len(parts) == 4:
                    timing[name].setdefault(parts[1], []).append(float(parts[3]))
                elif parts[0] == 'digest' and len(parts) == 3:
                    digests[name].setdefault(parts[1], set()).add(parts[2])
    counts = {}
    for name in VARIANTS:
        counts[name] = {}
        for line in invoke(name, 'count', []).splitlines():
            parsed = parse_count_line(line)
            if parsed is None:
                continue
            input_name, record = parsed
            counts[name][input_name] = record

    summary = {'timing': {}, 'counts': counts, 'checks': {}, 'table_digests': {}}
    for name in VARIANTS:
        summary['timing'][name] = {
            input_name: {'median_ns_per_call': statistics.median(samples),
                         'min_ns_per_call': min(samples),
                         'samples': len(samples)}
            for input_name, samples in timing[name].items()
        }
    summary['relative_to_M_percent'] = {
        name: {input_name: 100 * (values['median_ns_per_call']
                                 / summary['timing']['M'][input_name]['median_ns_per_call'] - 1)
               for input_name, values in summary['timing'][name].items()
               if input_name in summary['timing']['M']}
        for name in VARIANTS if name != 'M'
    }
    summary['table_digests'] = {name: {k: sorted(v) for k, v in digests[name].items()}
                                for name in VARIANTS}
    failures = []
    for input_name in document['inputs']:
        digest_values = {name: counts[name].get(input_name, {}).get('digest') for name in VARIANTS}
        if len(set(digest_values.values())) != 1:
            failures.append(f'table digest differs for {input_name}: {digest_values}')
        for name in VARIANTS:
            if set(digests[name].get(input_name, ())) != {digest_values[name]}:
                failures.append(f'{name} instrumented model digest differs for {input_name}')
        accepted = {name: counts[name].get(input_name, {}).get('accepted') for name in VARIANTS}
        if len(set(accepted.values())) != 1:
            failures.append(f'accepted transition count differs for {input_name}: {accepted}')
    summary['checks'] = {'failures': failures, 'ok': not failures}
    document['summary'] = summary
    document['verdict'] = 'invalid' if failures else 'consistent'
    save()
    print(json.dumps({'verdict': document['verdict'],
                      'relative_to_M_percent': summary['relative_to_M_percent'],
                      'counts': counts, 'checks': summary['checks']}, indent=2), flush=True)
    if failures:
        raise SystemExit('counting invariant failed; evidence retained')


def prepare_counts(root):
    """Compiles only the corrected counting harness; timing artifacts untouched."""
    source_dir = root / 'micro'
    log_path = source_dir / 'prepare-counts.json'
    if log_path.exists():
        raise SystemExit(f'{log_path} exists; refusing to overwrite counting evidence')
    builds = json.loads((root / 'builds.json').read_text(encoding='utf-8'))
    extension = '.exe' if (root / 'M.exe').exists() else ''
    prepare_log = {'purpose': ('corrected counting model: `considered` is injected at the leaf '
                               'transition loop, so the unit is one enumerated candidate'),
                   'rustc': subprocess.check_output(['rustc', '-V'], text=True),
                   'variants': {}}
    for name in VARIANTS:
        body = extract_suit_table(root / 'sources' / f'{name}.analysis.rs')
        count_text, leaf = instrument(body)
        path = source_dir / f'{name}-count-v2.rs'
        path.write_text(harness(count_text, True), encoding='utf-8')
        binary = root / f'micro-{name}-count-v2{extension}'
        command = ['rustc', '--edition=2021', '-O', str(path), '-o', str(binary)]
        subprocess.run(command, check=True)
        prepare_log['variants'][name] = {
            'suit_table_sha256': hashlib.sha256(body.encode()).hexdigest(),
            'considered_site': leaf,
            'layers_counted': list(COUNTER_FIELDS),
            'count_source': str(path),
            'count_source_sha256': sha256(path),
            'count_binary': str(binary),
            'count_binary_sha256': sha256(binary),
            'command': command,
        }
        print(f'prepared corrected count model {name} (considered at `{leaf}` loop)', flush=True)
    prepare_log['builds_mainline'] = builds['mainline']
    log_path.write_text(json.dumps(prepare_log, indent=2) + '\n', encoding='utf-8')


# Known answer of the corrected definition for the full-capacity input, read back
# from the loop bounds rather than from any timing; guards against re-counting an
# enclosing loop.
KNOWN_LEAF_COUNTS = {
    'empty-cap4-s4': {'M': 3036, 'MA': 1183, 'A2': 1938},
}


def corrected_counts(root, output_path, timed_path):
    """Runs the corrected counting harness once per variant, serially, untimed."""
    if output_path.exists():
        raise SystemExit('refusing to overwrite existing evidence')
    prepared = json.loads((root / 'micro' / 'prepare-counts.json').read_text(encoding='utf-8'))
    extension = '.exe' if (root / 'M.exe').exists() else ''
    for name, entry in prepared['variants'].items():
        actual = sha256(root / f'micro-{name}-count-v2{extension}')
        if actual != entry['count_binary_sha256']:
            raise SystemExit(f'micro-{name}-count-v2 hash {actual} differs from '
                             f'prepare-counts.json; refusing to count')
    timed = json.loads(timed_path.read_text(encoding='utf-8'))
    timed_digests = timed['summary']['table_digests']
    old_counts = timed['summary']['counts']
    document = {
        'protocol': 'suit-table-a2-counts-v2',
        'purpose': ('Corrected, untimed counting evidence. Supersedes the `considered` numbers '
                    'of 01-microbench.json only; 01 timing stays valid and was not re-run.'),
        'supersedes': {
            'file': timed_path.name,
            'field': 'summary.counts[*][*].considered',
            'reason': ('01 injected `considered` at the last `pair` loop before the accept '
                       'anchor. In M and A2 the pair loop is the innermost transition loop, '
                       'but archived full A enumerates pair -> sequences -> triplets, so its '
                       '573 counted outer pair candidates, not leaf (pair, seq, trip) '
                       'candidates. `accepted` (1183) exceeding that `considered` (573) is the '
                       'signature of the mismatch. The 01 `considered` values are retained '
                       'unchanged as raw evidence and are not comparable across variants.'),
            'retained': ('01-microbench.json is kept byte-for-byte: all timing samples, digests '
                         'and the old counter field remain raw evidence; nothing was re-run, '
                         'discarded or overwritten.'),
        },
        'counting_definition': {
            'accepted': ('incremented immediately before `let extra = ...`; one unit per '
                         'transition that passed every validity check; must be equal across '
                         'variants'),
            'considered': ('incremented at the innermost `for` loop whose body contains the '
                           'accept anchor, i.e. one unit per enumerated leaf transition '
                           'candidate: M/A2 pair loop (sequences -> triplets -> pair), '
                           'MA triplets loop (pair -> sequences -> triplets)'),
            'layers': ('each enclosing transition loop also tallied: ' + ', '.join(COUNTER_FIELDS)
                       + ' (`sets` tallies a pre-accept `sets` loop and stays 0 in all '
                         'three variants, whose only `sets` loop assembles the result table '
                         'after the accept site)'),
            'never_timed': True,
        },
        'mainline': prepared['builds_mainline'],
        'rustc': prepared['rustc'],
        'inputs': [{'name': name, 'hand': hand, 'capacity': capacity, 'sets': sets}
                   for name, hand, capacity, sets in INPUTS],
        'variants': prepared['variants'],
        'run_order': list(VARIANTS),
        'runs': [],
        'counts': {},
        'tables': {},
        'superseded_01_considered': {name: {input_name: old_counts[name][input_name]['considered']
                                           for input_name, *_ in INPUTS}
                                     for name in VARIANTS},
        'checks': {},
    }

    def save():
        output_path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    counts = {}
    tables = {}
    for name in VARIANTS:
        binary = root / f'micro-{name}-count-v2{extension}'
        command = [str(binary)]
        record = {'variant': name, 'command': command,
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        document['runs'].append(record)
        save()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        record.update(returncode=completed.returncode,
                      stdout=completed.stdout, stderr=completed.stderr)
        save()
        if completed.returncode != 0:
            raise SystemExit('failed count run retained; experiment aborted, no retry')
        counts[name] = {}
        tables[name] = {}
        for line in completed.stdout.splitlines():
            parsed = parse_count_line(line)
            if parsed is not None:
                input_name, entry = parsed
                counts[name][input_name] = entry
            parsed_table = parse_table_line(line)
            if parsed_table is not None:
                input_name, values = parsed_table
                tables[name][input_name] = values
    document['counts'] = counts
    document['tables'] = tables

    failures = []
    checks = {'accepted_equal': {}, 'considered_superset_of_accepted': {},
              'considered_matches_leaf_layer': {}, 'counted_table_equal': {},
              'digest_matches_timed_table': {}, 'digest_matches_01_count_model': {},
              'known_leaf_counts': {}}
    for input_name, *_ in INPUTS:
        accepted = {name: counts[name].get(input_name, {}).get('accepted') for name in VARIANTS}
        checks['accepted_equal'][input_name] = accepted
        if len(set(accepted.values())) != 1 or None in accepted.values():
            failures.append(f'accepted differs for {input_name}: {accepted}')
        superset = {}
        for name in VARIANTS:
            entry = counts[name].get(input_name, {})
            superset[name] = (entry.get('accepted') is not None
                              and entry['accepted'] <= entry.get('considered', -1))
            leaf = prepared['variants'][name]['considered_site']
            if entry.get('considered') != entry.get(leaf):
                failures.append(f'{name} {input_name}: considered {entry.get("considered")} != '
                                f'`{leaf}` layer count {entry.get(leaf)}')
        checks['considered_superset_of_accepted'][input_name] = superset
        if not all(superset.values()):
            failures.append(f'leaf enumeration does not cover accepted for {input_name}: {superset}')
        checks['considered_matches_leaf_layer'][input_name] = {
            name: counts[name].get(input_name, {}).get('considered') for name in VARIANTS}
        table_values = {name: tables[name].get(input_name) for name in VARIANTS}
        checks['counted_table_equal'][input_name] = {
            'all_equal': len({tuple(v) if v is not None else None
                              for v in table_values.values()}) == 1,
            'M': table_values['M']}
        if not checks['counted_table_equal'][input_name]['all_equal']:
            failures.append(f'instrumented tables differ for {input_name}: {table_values}')
        digest_values = {name: counts[name].get(input_name, {}).get('digest') for name in VARIANTS}
        timed_values = {name: (timed_digests.get(name, {}).get(input_name) or [None])[0]
                        for name in VARIANTS}
        checks['digest_matches_timed_table'][input_name] = {
            name: {'timed': timed_values[name], 'corrected_count_model': digest_values[name],
                   'match': timed_values[name] == digest_values[name]}
            for name in VARIANTS}
        if len(set(digest_values.values())) != 1:
            failures.append(f'counted-model digests differ for {input_name}: {digest_values}')
        for name in VARIANTS:
            if timed_values[name] != digest_values[name]:
                failures.append(f'{name} {input_name}: corrected model digest {digest_values[name]} '
                                f'!= timed table digest {timed_values[name]}')
        checks['digest_matches_01_count_model'][input_name] = {
            name: {'01_count_model': old_counts[name][input_name]['digest'],
                   'corrected_count_model': digest_values[name],
                   'match': old_counts[name][input_name]['digest'] == digest_values[name]}
            for name in VARIANTS}
        for name in VARIANTS:
            if old_counts[name][input_name]['digest'] != digest_values[name]:
                failures.append(f'{name} {input_name}: corrected table digest differs from the 01 '
                                'instrumented model')
        if input_name in KNOWN_LEAF_COUNTS:
            observed = {name: counts[name].get(input_name, {}).get('considered')
                        for name in VARIANTS}
            checks['known_leaf_counts'][input_name] = {
                'expected': KNOWN_LEAF_COUNTS[input_name], 'observed': observed}
            if observed != KNOWN_LEAF_COUNTS[input_name]:
                failures.append(f'known leaf counts differ for {input_name}: {observed}')
    checks['failures'] = failures
    checks['ok'] = not failures
    document['checks'] = checks
    document['summary'] = {
        'leaf_considered': {input_name: {name: counts[name][input_name]['considered']
                                         for name in VARIANTS}
                            for input_name, *_ in INPUTS},
        'accepted': {input_name: {name: counts[name][input_name]['accepted']
                                  for name in VARIANTS}
                     for input_name, *_ in INPUTS},
        'per_layer_iterations': {
            input_name: {name: {field: counts[name][input_name].get(field)
                                for field in COUNTER_FIELDS}
                         for name in VARIANTS}
            for input_name, *_ in INPUTS},
    }
    document['verdict'] = 'invalid' if failures else 'consistent'
    save()
    print(json.dumps({'verdict': document['verdict'],
                      'leaf_considered': document['summary']['leaf_considered'],
                      'checks': {'ok': checks['ok'], 'failures': failures}}, indent=2),
          flush=True)
    if failures:
        raise SystemExit('corrected counting check failed; evidence retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', type=Path, metavar='ARTIFACT_DIR')
    parser.add_argument('--prepare-counts', type=Path, metavar='ARTIFACT_DIR')
    parser.add_argument('--measure', nargs=2, metavar=('ARTIFACT_DIR', 'OUTPUT_JSON'))
    parser.add_argument('--count', nargs=3,
                        metavar=('ARTIFACT_DIR', 'OUTPUT_JSON', 'TIMED_JSON'))
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    elif args.prepare_counts:
        prepare_counts(args.prepare_counts.resolve())
    elif args.measure:
        measure(Path(args.measure[0]).resolve(), Path(args.measure[1]).resolve())
    elif args.count:
        corrected_counts(Path(args.count[0]).resolve(), Path(args.count[1]).resolve(),
                         Path(args.count[2]).resolve())
    else:
        parser.error('choose --prepare, --prepare-counts, --measure or --count')



if __name__ == '__main__':
    main()
