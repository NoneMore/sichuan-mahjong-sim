# Behavior-cloning teacher data collection design

Rationale for the [collection specification](spec.md).

## Start with teacher self-play

Four-seat teacher self-play tests basic imitation on a single distribution.
Mixing exploration or student trajectories immediately would make poor imitation
harder to distinguish from distribution changes.

The trade-off is sparse coverage of rare responses and states reached after
student mistakes. Opportunity and selection counts distinguish missing evidence
from a teacher's rejection. The behavior policy and teacher coincide initially,
so one choice serves both execution and labeling.

## Let collection own the drive loop

A neural-side collector coordinates Game, teacher, encoding, and persistence
through existing interfaces. A recording `Policy` wrapper is less suitable:
`choose` cannot return storage errors or observe submission success. Adding
Runner callbacks solely for collection would spread those responsibilities.

## Preserve the raw behavior distribution

Hard labels reproduce the teacher's final choice and tie-breaks. Shanten, ukeire,
and fan are not a common action-value scale from which to derive probabilities.
Suit permutation can also change index-based tie-breaks.

Winner-only filtering conditions demonstrations on outcomes; dropping `Pass`
erases refusals. Complete collection leaves training weights adjustable. The
initial training baseline samples multi-candidate decisions naturally;
single-candidate records support trace checks but carry no choice-learning signal.

## Keep identity and storage simple

One committed source revision identifies the simulator and teacher. Source enters
episode identity but not split membership, keeping the same deal in the same split
across teacher versions.

Batch complete episodes into configurable shards rather than creating one file
per game. Streaming keeps memory independent of batch size; per-shard manifest
entries keep inventory growth proportional to shard count. Consecutive seed ranges
provide the resume position without scanning historical episodes.

Larger shards reduce file and commit overhead but increase uncommitted work to
replay after failure. Pilot storage size and I/O cost should guide `shard_games`.
Immutable shards, atomic manifest replacement, and one OS-locked committer provide
a commit boundary without concurrent inventory updates.

## Let evidence determine expansion

If coverage exists but agreement is poor, inspect the model, training, and label
pipeline. Extend natural self-play while it adds useful support. Consider separate,
teacher-labeled supplemental data only when further collection mostly repeats
common decisions; preserve seed grouping and the natural-distribution test set.

Offline agreement measures imitation on teacher trajectories; student games on
reserved seeds expose accumulated errors and distribution shift. These results
and learning curves determine useful corpus scale.
