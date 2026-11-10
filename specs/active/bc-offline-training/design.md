# Offline behavior-cloning training design

Rationale for the [specification](spec.md). This Change produces the policy used
by [closed-loop evaluation](../python-policy-evaluation/design.md), while keeping
offline training independent of environment bindings.

## Ownership and data flow

```text
corpus snapshot -> Rust preparation -> NumPy cache
               -> Python Dataset -> PyTorch policy -> metrics / checkpoint
```

Rust reuses the neural crate's streaming validation and feature semantics. Python
owns batching, the network, and optimization. This avoids a second corpus parser
and keeps Python bindings out of the offline path.

The captured snapshot separates training from ongoing collection. Sharded NumPy
arrays provide memory mapping; packed candidates avoid global padding and repeated
JSONL parsing. Rebuildable caches make preparation recovery machinery unnecessary.

## Candidate-scoring model

```text
state -> State MLP -> h
concat(h, candidate) -> shared nonlinear MLP -> logit
```

Compute the state representation once per decision and share the scorer across
candidates and stages. Nonlinear interaction is essential: a purely additive state
term cancels in softmax. Returning logits lets BC, online evaluation, and future
RL share the model while choosing their own loss or action-selection method.

## Baseline choices

Equal-weight imitation makes errors easier to interpret. Forced choices carry zero
candidate cross-entropy. Suit permutation can conflict with the teacher's
index-based tie-breaks.

Offline imitation measures agreement on teacher trajectories; game scores expose
accumulated errors and distribution shift. Epoch-boundary recovery avoids tracking
in-flight batches or loader prefetch.

Network widths, optimizer settings, batch size, loader workers, and corpus scale
remain experimental parameters guided by learning curves and resource use.
