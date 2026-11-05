#!/usr/bin/env python3
"""Formal three-variant comparison for the suit_table A2 experiment.

Uses only the existing `heuristic-benchmark-v1` executable and workload
(`--games 2000 --seed {7,10007} --warmup 20`). Each round runs the three
variants once in a pre-fixed order; the order rotates per round and seed 10007
uses the reversed rotation of seed 7. Rounds 1-3 are mandatory; if the
`A2`-vs-`M` verdict is uncertain, rounds 4 and 5 are added for both seeds and
the final verdict uses every observation. Nothing is retried or discarded.

Verdicts reuse the archived conservative range criterion and its helper
functions from the archived Change, unmodified.

Usage:
  python -B performance.py <artifact-dir> <output.json>
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

VARIANTS = ('M', 'MA', 'A2')
COMPARISONS = {'A-vs-M': ('M', 'MA'), 'A2-vs-M': ('M', 'A2')}
BASE_ORDERS = {7: ('M', 'MA', 'A2'), 10007: ('A2', 'MA', 'M')}
SEEDS = (7, 10007)
TIMING_FIELDS = {'total_time_seconds', 'policy_time_seconds', 'rules_time_seconds',
                 'games_per_sec', 'decisions_per_sec', 'mean_policy_us_per_decision'}
ARCHIVE_PERFORMANCE = (Path(__file__).resolve().parents[2] / 'archive'
                       / 'heuristic-analysis-optimization' / 'performance.py')


def load_archive():
    spec = importlib.util.spec_from_file_location('archived_performance', ARCHIVE_PERFORMANCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def order(seed, round_number):
    base = BASE_ORDERS[seed]
    shift = (round_number - 1) % len(base)
    return base[shift:] + base[:shift]


def project(rows, baseline, candidate):
    """Projects three-variant rows onto the archived helper's two roles.

    The archived helper keys observations by `pair`; here one round is one
    observation, so the round number is exposed as that key.
    """
    role = {baseline: 'baseline', candidate: 'candidate'}
    return [dict(row, role=role[row['variant']], pair=row['round'])
            for row in rows if row['variant'] in (baseline, candidate)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite existing evidence')
    root = args.artifacts.resolve()
    builds = json.loads((root / 'builds.json').read_text(encoding='utf-8'))
    archive = load_archive()
    extension = '.exe' if os.name == 'nt' else ''
    binaries = {name: root / (name + extension) for name in VARIANTS}
    # Fail closed: only the binaries recorded by the build manifest are measured.
    hashes = {}
    for name, path in binaries.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = builds['variants'][name]['binary_sha256']
        if actual != expected:
            raise SystemExit(f'{path.name} hash {actual} differs from builds.json '
                             f'{expected}; refusing to measure')
        hashes[name] = actual
    document = {
        'comparison': 'suit-table-a2',
        'protocol': 'heuristic-benchmark-v1',
        'mainline': builds['mainline'],
        'verified_binary_sha256': hashes,
        'variants': {name: {'binary_sha256': hashes[name],
                            'patch_sha256': builds['variants'][name]['patch_sha256'],
                            'source_sha256': builds['variants'][name]['source_sha256']}
                     for name, path in binaries.items()},
        'a2_patch': builds['patches']['A2'],
        'rustc': builds['rustc'],
        'cargo': builds['cargo'],
        'build_environment': builds['environment'],
        'measurement_environment': {
            'platform': __import__('platform').platform(),
            'cpu_count': os.cpu_count(),
            'machine': builds.get('machine'),
            'rustflags': os.environ.get('RUSTFLAGS'),
            'cargo_encoded_rustflags': os.environ.get('CARGO_ENCODED_RUSTFLAGS'),
        },
        'rounds_mandatory': 3,
        'rounds_extension': 2,
        'extension_trigger': 'A2-vs-M uncertain after the mandatory rounds',
        'runs': [],
    }

    def save():
        args.output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    def run_rounds(first, last):
        for seed in SEEDS:
            for round_number in range(first, last + 1):
                round_fields = {}
                for name in order(seed, round_number):
                    command = [str(binaries[name]), '--games', '2000',
                               '--seed', str(seed), '--warmup', '20']
                    print(f'seed={seed} round={round_number} {name}', flush=True)
                    record = {'seed': seed, 'round': round_number, 'variant': name,
                              'command': command,
                              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    document['runs'].append(record)
                    save()
                    completed = subprocess.run(command, capture_output=True, text=True, check=False)
                    record.update(returncode=completed.returncode, stdout=completed.stdout,
                                  stderr=completed.stderr)
                    save()
                    if completed.returncode != 0:
                        raise SystemExit('failed run retained; experiment aborted, '
                                         'no automatic retry')
                    fields = dict(line.split(': ', 1) for line in completed.stdout.splitlines())
                    for field in TIMING_FIELDS:
                        if field in fields:
                            fields[field] = float(fields[field])
                    record['fields'] = fields
                    round_fields[name] = fields
                    save()
                stable = [{k: v for k, v in round_fields[name].items()
                           if k not in TIMING_FIELDS} for name in VARIANTS]
                if stable[0] != stable[1] or stable[0] != stable[2]:
                    document['error'] = f'non-time batch fields differ at seed {seed} ' \
                                        f'round {round_number}'
                    save()
                    raise SystemExit(document['error'])

    def verdicts():
        return {name: archive.verdict(project(document['runs'], *roles))
                for name, roles in COMPARISONS.items()}

    run_rounds(1, 3)
    document['initial_verdicts'] = verdicts()
    document['extension_triggered'] = document['initial_verdicts']['A2-vs-M'] == 'uncertain'
    save()
    if document['extension_triggered']:
        run_rounds(4, 5)
    document['verdicts'] = verdicts()
    document['comparisons'] = {
        name: archive.summary(project(document['runs'], *roles))
        for name, roles in COMPARISONS.items()
    }
    save()
    print(json.dumps({'verdicts': document['verdicts'],
                      'initial_verdicts': document['initial_verdicts'],
                      'extension_triggered': document['extension_triggered']}, indent=2),
          flush=True)


if __name__ == '__main__':
    main()
