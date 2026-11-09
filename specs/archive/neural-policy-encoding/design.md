# Neural policy encoding design

Proposed encoding for BC of `HeuristicPolicy`, with later self-play using the
same policy interface. The [specification](spec.md) defines the contract.

## Boundary

```text
Game -> Decision -> encoding -> state + legal candidates -> network logits
  ^                                                            |
  +---------------- semantic Action <- action selection -------+
```

[`Decision`](../../../src/game.rs) already supplies player-visible information
and legality. Use that boundary rather than privileged Game state. Encoding
produces ordinary numeric data; framework adaptation owns batching and devices,
while training and action selection consume the logits. This keeps feature
semantics shared between BC and inference without coupling the simulator to a
learning framework. Start with one concrete encoder, not a plugin abstraction.

## Observation choices

- **Memoryless snapshot.** The current teacher uses only `Decision`, so a first
  student does not need recurrent state or event replay. Preserve current hand,
  public player state, meld structure, discard counts, and response context.
- **Structured groups.** Keep hand, players, melds, and context separate until
  the MLP adapter flattens them. This supports inspection and later state-network
  changes without reconstructing structure from an opaque vector.
- **Meld structure and discard ownership.** Meld kinds and sources preserve
  public payment relationships. Discard categories distinguish historical offers
  from tiles still owned by records, allowing visibility counts without counting
  meld-claimed tiles twice.
- **Counts rather than history.** Drop event order and per-record winner links.
  This is a compressed observation, not a complete player information state.
  History can become an explicit input if experiments justify it.

The student learns from the raw hand and public facts, not exact shanten, ukeire,
fan, or teacher rankings. Supplying those would retain the analysis dependency
that BC is intended to replace. They can still help offline diagnosis.

## Actions and model

Score legal candidates instead of assigning every possible action a global
output index. This fits ordinary semantic actions and unordered exchange
multisets. It costs variable-length batching and additional exchange candidates,
but avoids a large exchange output vocabulary or autoregressive selection with
intermediate multiplicity constraints.

Action features describe intent parameters, not simulated post-action hands.
The state identifies the opportunity for parameterless response actions. Candidate
position is not a feature; the semantic action provides label and output identity.

Use a State MLP once per decision, then concatenate its representation with each
action's features and apply a shared nonlinear scoring MLP. A purely additive
state score plus action score would cancel the state term in softmax. Network
sizes remain experimental; CNNs or event-sequence models are not prerequisites.

## BC and self-play

BC matches the teacher's semantic action to a candidate and trains cross-entropy.
Teacher preferences supply labels, not legality filters: retaining all legal
choices lets later self-play explore beyond the teacher. Avoid default suit
permutation augmentation because the teacher breaks some ties by suit index.

Logits support either greedy selection or sampling. Later self-play may add a
value head and trajectory handling outside encoding; it does not require a new
policy representation. Model and data compatibility depends on feature meanings,
not just tensor shapes, so the encoding has an explicit version.

## Remaining evidence

First verify that the encoding preserves teacher choices through input-projection
and collision checks. Then measure imitation by decision stage, separating forced
choices from genuine decisions. Whether a small network learns the policy well,
and whether history improves later self-play, remain experimental questions.
