#!/usr/bin/env python3
"""Build the three measured suit_table variants from one fixed mainline.

M:  mainline `5ad7540` unchanged (original suit_table + retained seven-pairs B)
MA: same mainline + archived PR12 `patches/A.patch` (full A + B)
A2: same mainline + this Change's A2 patch (A2 + B)

Everything happens in one freshly created detached worktree, serially. The
script also pre-generates each variant's release library assembly (so a later
regression can be inspected statically without another build) and runs the
reused frozen trace collector, requiring every variant to reproduce the frozen
64-game behavioral corpus.

Usage:
  python -B build_variants.py <new-artifact-dir> [--a2-patch PATH]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

MAINLINE_PREFIX = '5ad7540'
ARCHIVE = 'specs/archive/heuristic-analysis-optimization'
VARIANTS = {'M': [], 'MA': ['A'], 'A2': ['A2']}


def run(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def output(args):
    return subprocess.check_output([str(x) for x in args], text=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_patch(data, notes):
    """Strip a UTF-8 BOM and CRLF line endings so patch context matches LF blobs."""
    if data.startswith(b'\xef\xbb\xbf'):
        data = data[3:]
        notes.append('stripped UTF-8 BOM')
    if b'\r\n' in data:
        data = data.replace(b'\r\n', b'\n')
        notes.append('normalized CRLF to LF')
    return data


def touched_paths(patch):
    paths = set()
    for line in patch.decode('utf-8').splitlines():
        if line.startswith('+++ b/'):
            paths.add(line[len('+++ b/'):])
    return paths


def powershell(command):
    """Lightweight machine query; output is forced to UTF-8, never fatal."""
    try:
        completed = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; ' + command],
            capture_output=True, check=True, timeout=60)
        return completed.stdout.decode('utf-8', 'replace').strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f'unavailable: {error!r}'


def machine():
    return {
        'cpu': powershell('(Get-CimInstance -ClassName Win32_Processor).Name'),
        'os_caption': powershell('(Get-CimInstance -ClassName Win32_OperatingSystem).Caption'),
        'os_version': powershell('(Get-CimInstance -ClassName Win32_OperatingSystem).Version'),
        'physical_memory_mib': powershell(
            '[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB)'),
        'platform': platform.platform(),
        'logical_cpus': os.cpu_count(),
        'python': platform.python_version(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--a2-patch', type=Path,
                        default=Path(__file__).resolve().parent / 'patches' / 'A2.patch')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    output_ = args.output.resolve()
    output_.mkdir(parents=True, exist_ok=False)

    mainline = output(['git', '-C', repo, 'rev-parse', MAINLINE_PREFIX]).strip()
    if not mainline.startswith(MAINLINE_PREFIX):
        raise SystemExit(f'unexpected mainline {mainline}')

    # Patch bytes are taken from git blobs (debug LF) or normalized workspace files.
    a_patch = subprocess.check_output(
        ['git', '-C', repo, 'cat-file', 'blob', f'{mainline}:{ARCHIVE}/patches/A.patch'])
    a2_notes = []
    a2_patch = normalize_patch(args.a2_patch.read_bytes(), a2_notes)
    a2_touched = touched_paths(a2_patch)
    if not a2_touched or any(not p.startswith('src/policy/analysis') for p in a2_touched):
        raise SystemExit(f'A2 patch must only touch src/policy/analysis*: {sorted(a2_touched)}')

    (output_ / 'patches').mkdir()
    (output_ / 'patches' / 'A.patch').write_bytes(a_patch)
    (output_ / 'patches' / 'A2.patch').write_bytes(a2_patch)

    collector = repo / ARCHIVE / 'tools' / 'trace_corpus.rs'
    frozen = repo / ARCHIVE / 'baseline' / 'trace-corpus.txt'
    extension = '.exe' if os.name == 'nt' else ''
    tree = output_ / 'source'
    run(['git', '-C', repo, 'worktree', 'add', '--detach', tree, mainline])
    (output_ / 'sources').mkdir()
    (output_ / 'asm').mkdir()

    manifest = {
        'change': 'suit-table-a2',
        'protocol': 'suit-table-a2-v1',
        'mainline': mainline,
        'variants': {name: {'patches': patches} for name, patches in VARIANTS.items()},
        'patches': {
            'A': {'source': f'{mainline}:{ARCHIVE}/patches/A.patch',
                  'sha256': hashlib.sha256(a_patch).hexdigest()},
            'A2': {'source': str(args.a2_patch),
                   'sha256_original': sha256(args.a2_patch),
                   'sha256_applied': hashlib.sha256(a2_patch).hexdigest(),
                   'normalizations': a2_notes,
                   'touched_paths': sorted(a2_touched)},
        },
        'build_command': ['cargo', 'build', '--release', '--example', 'heuristic_bench'],
        'asm_command': ['cargo', 'rustc', '--release', '--lib', '--', '--emit=asm'],
        'collector': {'source': str(collector), 'sha256': sha256(collector)},
        'frozen_trace': {'source': str(frozen), 'sha256': sha256(frozen)},
        'rustc': output(['rustc', '-Vv']),
        'cargo': output(['cargo', '-V']),
        'machine': machine(),
        'environment': {name: os.environ.get(name) for name in (
            'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET',
            'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_LTO',
            'CARGO_PROFILE_RELEASE_CODEGEN_UNITS', 'CARGO_INCREMENTAL')},
    }

    def normalized_trace(path):
        return '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                         if not line.startswith('revision: '))

    reference = normalized_trace(frozen)
    deps = tree / 'target/release/deps'
    for name in VARIANTS:
        # Only this newly created detached worktree is reset between variants.
        run(['git', '-C', tree, 'reset', '--hard', mainline])
        for patch in VARIANTS[name]:
            run(['git', '-C', tree, 'apply', '--check', output_ / 'patches' / f'{patch}.patch'])
            run(['git', '-C', tree, 'apply', output_ / 'patches' / f'{patch}.patch'])
        patch = subprocess.check_output(['git', '-C', tree, 'diff', '--binary'])
        (output_ / f'{name}.patch').write_bytes(patch)
        source = (tree / 'src/policy/analysis.rs').read_bytes()
        (output_ / 'sources' / f'{name}.analysis.rs').write_bytes(source)
        run(['cargo', 'build', '--release', '--example', 'heuristic_bench'], cwd=tree)
        binary = output_ / (name + extension)
        shutil.copy2(tree / 'target/release/examples' / ('heuristic_bench' + extension), binary)
        entry = {'patch_sha256': hashlib.sha256(patch).hexdigest(),
                 'binary_sha256': sha256(binary),
                 'source_sha256': hashlib.sha256(source).hexdigest()}

        # Pre-generate the release assembly; no build is needed to inspect it later.
        before = {p: p.stat().st_mtime_ns for p in deps.glob('*.s')}
        run(['cargo', 'rustc', '--release', '--lib', '--', '--emit=asm'], cwd=tree)
        fresh = [p for p in deps.glob('*.s') if before.get(p) != p.stat().st_mtime_ns]
        library_asm = [p for p in fresh if 'sichuan_mahjong_sim' in p.name]
        if not library_asm:
            raise SystemExit(f'{name}: no library assembly emitted')
        (output_ / 'asm' / name).mkdir()
        entry['asm'] = {}
        for path in library_asm:
            target = output_ / 'asm' / name / path.name
            shutil.copy2(path, target)
            entry['asm'][path.name] = sha256(target)

        rlib = max(deps.glob('libsichuan_mahjong_sim-*.rlib'), key=lambda p: p.stat().st_mtime_ns)
        collector_exe = output_ / ('collector-' + name + extension)
        run(['rustc', '--edition=2021', '-O', collector, '--extern',
             'sichuan_mahjong_sim=' + str(rlib), '-L', 'dependency=' + str(deps),
             '-o', collector_exe])
        trace_path = output_ / f'{name}.trace'
        with trace_path.open('wb') as trace:
            run([collector_exe, '--revision', f'{mainline}+{name}'], stdout=trace)
        if normalized_trace(trace_path) != reference:
            raise SystemExit(f'{name}: behavioral corpus differs from the frozen baseline')
        entry['trace_sha256'] = sha256(trace_path)
        entry['trace_normalized_sha256'] = hashlib.sha256(
            normalized_trace(trace_path).encode()).hexdigest()
        entry['trace_matches_frozen_baseline'] = True
        manifest['variants'][name].update(entry)
        print(f'built {name}', flush=True)

    (output_ / 'builds.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Built {", ".join(VARIANTS)}; artifacts: {output_}', flush=True)


if __name__ == '__main__':
    main()
