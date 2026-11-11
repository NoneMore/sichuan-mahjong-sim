# Teacher collection: design rationale

Begin with four-seat HeuristicPolicy self-play so behavior and labels come
from one stable distribution. Keep all decisions, including passes,
forced choices and losing trajectories: filtering by outcome would alter
the target behavior, and forced choices remain useful for replay.

The collector drives Game and teacher through public interfaces. A
recording Policy wrapper cannot capture submission success and report
persistence errors cleanly; Runner callbacks would expand the execution
API for a data-production concern.

Bind corpus identity to the committed teacher/core/encoding revision while
assigning splits by seed alone, so the same deal stays in the same split
across teacher revisions. Publish complete immutable episode shards,
with an atomic manifest as the commit boundary. Larger shards reduce
storage overhead but enlarge the recovery unit.

Start with natural teacher trajectories. Expand to supplemental,
teacher-labeled states when coverage analysis shows that additional
self-play mainly repeats common decisions.
