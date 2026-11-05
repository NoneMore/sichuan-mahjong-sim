#!/usr/bin/env bash
# Rebuild the isolated behavioral trace collector against one checkout's
# library and capture the frozen corpus for this Change.
#
# Usage:
#   specs/archive/heuristic-analysis-optimization/tools/collect_trace.sh [REPO] [OUT_FILE]
#
# REPO defaults to the current git work tree; OUT_FILE defaults to the Change's
# baseline artifact. The collector source is compiled directly with rustc
# against `cargo build --release --lib`, so no file outside the Change is
# touched and no game driving is added: the corpus comes only from
# Runner::run_new. The collector itself takes no required arguments; it writes
# the frozen seeds 7..=38 and 10007..=10038 (32 games each) to stdout.
#
# The recorded `revision:` header is HEAD, with a marker for tracked changes.
set -euo pipefail

repo=${1:-$(git rev-parse --show-toplevel)}
change_dir=$(cd "$(dirname "$0")/.." && pwd)
out_file=${2:-$change_dir/baseline/trace-corpus.txt}
# Resolve output relative to the caller before changing to the source checkout.
mkdir -p "$(dirname "$out_file")"
out_file=$(cd "$(dirname "$out_file")" && pwd)/$(basename "$out_file")
revision=$(git -C "$repo" rev-parse HEAD)
if ! git -C "$repo" diff --quiet HEAD --; then
  revision="$revision+working-tree"
fi
if [[ -e "$out_file" ]]; then
  echo "refusing to overwrite existing corpus: $out_file" >&2
  exit 2
fi
collector=$change_dir/tools/trace_corpus.rs

cd "$repo"
cargo build --release --lib --locked

rlib=$(ls -t target/release/deps/libsichuan_mahjong_sim-*.rlib | head -n 1)
binary=target/release/trace_corpus
rustc --edition 2021 -O -D warnings \
  --extern "sichuan_mahjong_sim=$rlib" \
  -L dependency=target/release/deps \
  -o "$binary" \
  "$collector"

HEURISTIC_TRACE_REVISION="$revision" "$binary" > "$out_file"
echo "wrote $out_file"
