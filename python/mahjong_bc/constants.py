"""Frozen encoding, cache, policy, and checkpoint constants.

The encoding and split constants mirror the Rust neural crate. A change to any
of them requires a matching version change on both sides; this package rejects
caches or checkpoints that declare an unsupported version.
"""

#: Decision encoding version implemented by ``sichuan-mahjong-neural``.
ENCODING_VERSION = 1

#: Width of the flattened v1 decision state.
STATE_WIDTH = 1013

#: Width of one flattened v1 candidate action.
CANDIDATE_WIDTH = 39

#: Number of one-hot action-kind features at the start of a candidate.
ACTION_KIND_WIDTH = 9

#: Number of decision stages in v1 encoding order.
STAGE_COUNT = 5

#: Number of action kinds in v1 encoding order.
ACTION_KIND_COUNT = 9

#: Stable v1 stage names, indexed by wire stage id.
STAGE_NAMES = (
    "exchange",
    "missing_suit",
    "turn",
    "discard_response",
    "added_kong_response",
)

#: Stable v1 action-kind names, indexed by wire kind id.
ACTION_KIND_NAMES = (
    "exchange",
    "choose_missing",
    "discard",
    "peng",
    "open_kong",
    "concealed_kong",
    "added_kong",
    "hu",
    "pass",
)

#: Candidate count of a forced decision.
FORCED_CANDIDATE_COUNT = 1

#: Prepared cache manifest format.
DATASET_FORMAT = "mahjong-bc-dataset"

#: Prepared cache preparation-format and preparation versions.
PREPARATION_FORMAT_VERSION = 1
PREPARATION_VERSION = 1

#: Seed split algorithm recorded by the prepared cache.
SPLIT_ALGORITHM = "seed-sha256-80-10-10-v1"

#: Checkpoint container format. Format 3 adds a recoverable epoch-boundary
#: ``training`` payload to the format 2 policy container. Format 2 checkpoints
#: (reserved ``training: null``) remain readable for evaluation but cannot be
#: resumed.
CHECKPOINT_FORMAT_VERSION = 3

#: Flat MLP candidate-scoring serialization format.
MODEL_FORMAT_VERSION = 1

#: Suit/rank residual candidate-scoring serialization format.
SUIT_RANK_RESNET_MODEL_FORMAT_VERSION = 2

#: Dataset split names accepted by this package.
SPLIT_NAMES = ("train", "validation", "test")

#: Checkpoint role names written by this package.
ROLE_POLICY = "policy"
ROLE_BEST = "best"
ROLE_LAST = "last"
ROLES = (ROLE_POLICY, ROLE_BEST, ROLE_LAST)

#: Top-level checkpoint format marker.
CHECKPOINT_KIND = "mahjong-bc-checkpoint"

#: Package version recorded in checkpoints and run metadata.
PACKAGE_VERSION = "0.1.0"
