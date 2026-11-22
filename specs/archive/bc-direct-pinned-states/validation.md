# BC training states gathered into pinned slots

## Epoch timing

| Pair | Order | Baseline (s) | Direct gather (s) | Time reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | Baseline first | 10.964 | 7.479 | 31.8% |
| 2 | Direct gather first | 10.913 | 7.298 | 33.1% |
| 3 | Baseline first | 11.065 | 7.581 | 31.5% |
| 4 | Direct gather first | 11.252 | 7.539 | 33.0% |
| Median | | 11.014 | 7.509 | 32.4% |

All four pairs passed the speed criterion. Both checkpoint payloads matched
the first baseline trial in all seven comparisons, including model, optimizer,
RNG and validation metrics. The validation NLL was 1.14440491176093. PyTorch's
peak CUDA allocation was 145.2 MiB on the baseline and 150.6–151.0 MiB with
direct gather.

## Workload and measurement

The prepared cache contains 50,000 teacher games. Each measured epoch visited
1,729,386 eligible training decisions in 1,689 batches and evaluated 210,162
eligible validation decisions. The cache ID is
`6ab386e2ee421fd9fa0c6f733ca6300d1e7f9a9f9317ec53521325be847e97b7`.
The corpus manifest SHA-256 is
`e8ab7f62525ebe60722f3a6d7cc7f9657e981fc684fc6c886d1120054267c26e`.

The baseline was `a45b460d0614900bd9b83674732765ac64e0bda2` and the direct
gather revision was `2cc6ead2a7a2905e0cb06359fed28e7b3a2f5433`. Both used
the default 256/128 ReLU policy, AdamW, seed 7, batch size 1024, learning rate
0.001, weight decay 0.01, zero loader workers, four PyTorch CPU threads and two
interop threads. Both used RAM storage and CUDA prefetch, with RAM and pinned
budgets of 14 GiB and 64 MiB.

Measurements ran on Windows 10 with a Ryzen 7 3700X, an RTX 2070, NVIDIA
driver 591.74, Python 3.12.10, NumPy 2.5.4 and PyTorch 2.14.1+cu126.
Each trial used a fresh process with the selected revision's archived Python
package on `PYTHONPATH`. The four pairs alternated baseline-first and
direct-first. Each process loaded the cache, warmed up with one full epoch,
then trained a fresh model. Timing included validation and checkpoint
publication. Cache verification, split loading and warmup were outside the
timed interval.

The acceptance criteria were faster execution in each pair, a median paired
time reduction of at least 5%, and exact equality of the logical `last.pt`
and `best.pt` payloads across the measured runs.

Run the same [profiling tool](../../../python/tools/profile_training.py) with
`PYTHONPATH` set to each source snapshot's `python` directory. Use separate
output directories and pass the first measured trial as the `--reference-run`
for the others.

```text
python python/tools/profile_training.py --dataset <cache> --output <output> --storage ram --ram-limit-mib 14336 --batch-transfer cuda-prefetch --pinned-limit-mib 64 --seed 7 --batch-size 1024 --learning-rate 0.001 --weight-decay 0.01 --threads 4 --interop-threads 2
```

## Diagnostic runs

Host spans and CUDA traces each used one separate run per revision, with no
full-epoch warmup. The CUDA capture waited 160 training steps, warmed up for
16 and recorded the next 96.

| Training host span | Baseline median (ms) | Direct gather median (ms) |
| --- | ---: | ---: |
| Gather | 4.932 | 2.539 |
| Gather states and targets | 3.133 | 0.674 |
| Stage | 0.507 | 0.229 |
| Acquire producer slot | 0.018 | 0.809 |
| Take from ready queue | 1.826 | 0.006 |

The spans are inclusive per call. In the direct path, the producer acquires
the slot before gathering states.

| CUDA trace over 96 training steps | Baseline | Direct gather |
| --- | ---: | ---: |
| Observed window (ms) | 752.0 | 541.5 |
| Kernel busy fraction | 21.8% | 26.3% |
| Host-to-device copy time (ms) | 42.4 | 41.8 |

Both checkpoints matched the timing reference for all four diagnostic runs.
