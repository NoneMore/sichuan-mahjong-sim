#!/usr/bin/env python3
"""Rebuild isolated production variants from baseline plus retained patches.

No changes are made to the caller's checkout. Output must be a new directory.
Builds and optional trace capture finish BEFORE performance.py is invoked.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASELINE = 'c9202e33ac78cfb819def7339c3986c58cc3d13f'
VARIANTS = {'baseline': '', 'A': 'A', 'B': 'B', 'C': 'C',
            'ABC': 'ABC', 'ABCD': 'ABCD', 'BD': 'BD'}


def run(args, **kwargs):
    subprocess.run([str(x) for x in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tree = output / 'source'
    run(['git', '-C', repo, 'worktree', 'add', '--detach', tree, BASELINE])
    extension = '.exe' if os.name == 'nt' else ''
    manifest = {
        'baseline': BASELINE,
        'build_command': ['cargo', 'build', '--release', '--example', 'heuristic_bench'],
        'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
        'cargo': subprocess.check_output(['cargo', '-V'], text=True),
        'environment': {name: os.environ.get(name) for name in (
            'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET',
            'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_LTO',
            'CARGO_PROFILE_RELEASE_CODEGEN_UNITS')},
        'variants': {},
    }
    for name in args.variants:
        # Only this newly created detached worktree is reset between variants.
        run(['git', '-C', tree, 'reset', '--hard', BASELINE])
        for candidate in VARIANTS[name]:
            run(['git', '-C', tree, 'apply', here / 'patches' / f'{candidate}.patch'])
        patch = subprocess.check_output(['git', '-C', str(tree), 'diff', '--binary'])
        (output / f'{name}.patch').write_bytes(patch)
        run(['cargo', 'build', '--release', '--example', 'heuristic_bench'], cwd=tree)
        binary = output / (name + extension)
        shutil.copy2(tree / 'target/release/examples' / ('heuristic_bench' + extension), binary)
        manifest['variants'][name] = {'patch_sha256': hashlib.sha256(patch).hexdigest(),
                                      'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
        collector = here / 'tools' / 'trace_corpus.rs'
        if collector.exists():
            executable = output / ('collector-' + name + extension)
            run(['rustc', '--edition=2021', '-O', collector,
                 '--extern', 'sichuan_mahjong_sim=' + str(next((tree / 'target/release/deps').glob('libsichuan_mahjong_sim-*.rlib'))),
                 '-L', 'dependency=' + str(tree / 'target/release/deps'), '-o', executable])
            trace_path = output / f'{name}.trace'
            with trace_path.open('wb') as trace:
                run([executable, '--revision', BASELINE + '+' + name], stdout=trace)
            def normalized(path):
                return '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                                 if not line.startswith('revision: '))
            reference = here / 'baseline' / 'trace-corpus.txt'
            if normalized(trace_path) != normalized(reference):
                raise RuntimeError(f'{name}: behavioral corpus differs from the frozen baseline')
            manifest['variants'][name]['trace_matches_frozen_baseline'] = True
            manifest['variants'][name]['trace_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    (output / 'builds.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Built {", ".join(args.variants)}; artifacts: {output}', flush=True)


if __name__ == '__main__':
    main()
