#!/usr/bin/env python3
"""Change-local orchestration; uses the unmodified heuristic-benchmark-v1 executable.

Run only after all builds/tests finish, with no other benchmark running.
Each comparison starts at three pairs per seed and extends to five only when
its conservative range verdict is uncertain. All successful/failed attempts
are persisted, and an error aborts rather than silently retrying.
"""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import subprocess


def verdict(rows):
    directions = []
    for seed in (7, 10007):
        baseline = [r['fields']['total_time_seconds'] for r in rows
                    if r['seed'] == seed and r['role'] == 'baseline']
        candidate = [r['fields']['total_time_seconds'] for r in rows
                     if r['seed'] == seed and r['role'] == 'candidate']
        if max(candidate) < min(baseline):
            directions.append('benefit')
        elif max(baseline) < min(candidate):
            directions.append('regression')
        else:
            directions.append('uncertain')
    return directions[0] if directions[0] == directions[1] else 'uncertain'


def summary(rows):
    result = {'verdict': verdict(rows), 'seeds': {}}
    for seed in (7, 10007):
        records = [r for r in rows if r['seed'] == seed]
        roles = {}
        for role in ('baseline', 'candidate'):
            selected = [r['fields'] for r in records if r['role'] == role]
            roles[role] = {
                field: {'median': statistics.median(values),
                        'min': min(values), 'max': max(values)}
                for field in ('total_time_seconds', 'policy_time_seconds',
                              'games_per_sec', 'decisions_per_sec')
                for values in [[r[field] for r in selected]]
            }
        pairs = []
        for pair in sorted({r['pair'] for r in records}):
            by_role = {r['role']: r['fields'] for r in records if r['pair'] == pair}
            pairs.append({
                'pair': pair,
                'total_change_percent': 100 * (
                    by_role['candidate']['total_time_seconds'] /
                    by_role['baseline']['total_time_seconds'] - 1),
                'policy_change_percent': 100 * (
                    by_role['candidate']['policy_time_seconds'] /
                    by_role['baseline']['policy_time_seconds'] - 1),
            })
        result['seeds'][str(seed)] = {'roles': roles, 'pairs': pairs}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing evidence')
    binaries = {'baseline': args.baseline.resolve(), 'candidate': args.candidate.resolve()}
    import hashlib
    document = {'comparison': args.name, 'protocol': 'heuristic-benchmark-v1',
                'binaries': {role: {'path': str(path),
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                             for role, path in binaries.items()}, 'runs': []}

    def save():
        args.output.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')

    def run_pairs(first, last):
        for seed in (7, 10007):
            for pair in range(first, last + 1):
                # Opposite initial order for the two seed groups; continue the
                # same alternation for any extension, never reorder by results.
                baseline_first = (pair % 2 == 1) == (seed == 7)
                order = ('baseline', 'candidate') if baseline_first else ('candidate', 'baseline')
                pair_fields = {}
                for role in order:
                    command = [str(binaries[role]), '--games', '2000', '--seed', str(seed), '--warmup', '20']
                    print(f'{args.name} seed={seed} pair={pair} {role}', flush=True)
                    record = {'seed': seed, 'pair': pair, 'role': role,
                              'command': command, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
                    document['runs'].append(record)
                    save()
                    try:
                        completed = subprocess.run(command, capture_output=True, text=True, check=False)
                    except OSError as error:
                        record['error'] = repr(error)
                        save()
                        raise
                    record.update(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
                    save()
                    if completed.returncode != 0:
                        raise SystemExit('failed run retained; experiment aborted, no automatic retry')
                    fields = dict(line.split(': ', 1) for line in completed.stdout.splitlines())
                    for field in ('total_time_seconds', 'policy_time_seconds', 'rules_time_seconds',
                                  'games_per_sec', 'decisions_per_sec'):
                        fields[field] = float(fields[field])
                    record['fields'] = fields
                    pair_fields[role] = fields
                    save()
                timing = {'total_time_seconds', 'policy_time_seconds', 'rules_time_seconds',
                          'games_per_sec', 'decisions_per_sec', 'mean_policy_us_per_decision'}
                stable = [{k: v for k, v in pair_fields[role].items() if k not in timing}
                          for role in ('baseline', 'candidate')]
                if stable[0] != stable[1]:
                    document['error'] = 'non-time batch fields differ'
                    save()
                    raise SystemExit(document['error'])

    run_pairs(1, 3)
    document['initial_three_pair_verdict'] = verdict(document['runs'])
    save()
    if document['initial_three_pair_verdict'] == 'uncertain':
        run_pairs(4, 5)
    document['summary'] = summary(document['runs'])
    save()
    print(json.dumps(document['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
