# Offline behavior-cloning training: design rationale

The offline pipeline separates Rust preparation from Python training.
Rust verifies the collected corpus and publishes a fixed NumPy snapshot;
Python consumes that snapshot without embedding the simulator or duplicating
its corpus parser. Packed candidate arrays support memory mapping without
padding every decision to the largest action count.

A state embedding is computed once per decision, then combined with each
candidate by a shared nonlinear scorer. A purely additive state and
candidate score would cancel the state term in softmax. Candidate logits
serve both imitation training and frozen online inference.

Equal-weight imitation gives each eligible teacher decision the same
training weight. Forced choices contribute no candidate cross-entropy;
online scores measure the effect of accumulated decisions beyond teacher
agreement. Suit permutation is not an automatic augmentation because
the teacher uses suit-index tie-breaks.

Validation NLL selects the best model. Recovery occurs at epoch boundaries,
keeping restart state small and making an interrupted epoch replayable.
Network size, corpus scale and optimizer parameters remain training choices.
