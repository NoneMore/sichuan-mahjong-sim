# Offline behavior-cloning training specification

## Outcome and scope

Prepare the existing heuristic corpus, train a PyTorch policy, evaluate imitation,
and recover training from checkpoints without requiring PyO3. Rules, teacher
behavior, and encoding v1 remain unchanged. Closed-loop play belongs to the
[companion Change](../../active/python-policy-evaluation/design.md); RL training is outside
this Change. [Design](design.md) explains the structural choices.

## Data contract

- Consume one captured committed manifest and its listed shards, strictly
  validating the corpus. Later appends do not change the dataset or training run.
- Preserve the existing seed-based splits, feature values, canonical candidate
  order, episode identity, and all decisions, including forced choices.
- Produce split-specific sharded `.npy` arrays and a JSON manifest. For `D`
  decisions and `C` total candidates per shard:

  | Array | Content |
  | --- | --- |
  | `states` | `D x 1013` float32 features |
  | `candidate_features` | `C x 39` float32 packed candidate features |
  | `candidate_offsets` | `D + 1` offsets delimiting each decision's candidates |
  | `targets` | `D` local candidate indices |
  | Metadata | Episode identity, decision index, actor seat, stage |

  Offsets span `[0, C]`, each candidate range is nonempty, and each target indexes
  its own range.
- Record input snapshot/checksums, encoding/preparation versions, array layout,
  output inventory/checksums, and split counts. Retain collection, preparation,
  and training source provenance separately. Cache loading rejects unsupported
  encoding or preparation format versions.
- Prepare with bounded episode/output buffers and publish only after validation
  completes. Interrupted preparation restarts; completed caches support memory
  mapping independently of the original corpus.

## Policy and training

The shared Python policy accepts state, candidates, and a valid-candidate mask,
and returns logits. Use float32 pointwise MLPs without dropout or batch-dependent
normalization. Candidate position is not an input: permutation permutes logits.
Pad to the batch maximum and mask padding before loss or selection. Support greedy
selection and categorical sampling; greedy ties choose canonical-first.

Only decisions with more than one candidate enter training and primary metrics.
Train with mean masked cross-entropy, weighting each decision equally. Each epoch
visits every eligible training decision once in seeded shuffled order, including
the final partial batch. Retain the natural distribution across stages, passes,
seats, and outcomes; apply neither reweighting nor augmentation.

Select `best` by validation mean NLL and retain `last` for recovery. Evaluate
complete eligible validation/test sets, aggregating by decision. Report overall
NLL/top-1 accuracy, per-stage NLL/top-1 accuracy/support, and support/top-1 accuracy
grouped by the teacher's chosen action kind. Report forced counts separately;
zero-support groups have undefined metrics. Test evaluates the frozen model;
online game scores do not participate in checkpoint selection.

## Run identity and recovery

Record dataset identity, resolved configuration, source/dependency versions,
training seed, device, and metrics. Checkpoints contain encoding version,
model/checkpoint format versions, model configuration and weights, optimizer/
scheduler state when used, completed epoch/step, and random states. The shared
policy loader rejects unsupported versions. Publish atomically and restore the
last completed epoch, replaying any interrupted epoch. Resume requires the same
dataset identity, model/training configuration, device type, and dependency
versions; output/log paths may change.
Support single-local-GPU training and CPU smoke tests.

## Acceptance

- Corpus/cache fixtures preserve splits, features, candidate/target alignment, and
  metadata. Appends leave the captured dataset unchanged; failed preparation does
  not publish a cache.
- Padding leaves valid logits and loss unchanged. Candidate permutation with
  remapped targets permutes logits and preserves loss and unique-maximum selection;
  tie checks use canonical candidate order.
- A tiny CPU training fixture reduces BC loss; metric and traversal checks verify
  non-forced denominators, complete epochs, and validation-only selection.
- Under a fixed CPU runtime/configuration, epoch-boundary restart matches the
  uninterrupted run's subsequent sample order and step counts exactly, and model/
  optimizer states within numerical tolerances explicitly declared by the test.
  Interrupted checkpoint writes preserve the last usable checkpoint.
- A small valid corpus completes preparation, training, validation, checkpoint
  save/resume, and frozen-model test evaluation with recorded artifacts. A separate
  single-GPU smoke run verifies training and checkpoint reload on CUDA.
