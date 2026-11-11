# Neural encoding: design rationale

Use the player-visible Decision as the boundary between Game and learning.
A first behavior-cloning policy can use a memoryless snapshot because the
teacher chooses from the same Decision; historical action order can be
introduced by a later model design.

Preserve the acting hand, public meld structure, discard ownership and
response context. Counts compress public history while retaining tile
visibility and payment-relevant relationships. Keep the representation
inspectable even when an MLP later flattens it.

Score the legal candidate list rather than indexing all possible actions
in a global output vocabulary. This handles exchange multisets and
variable legal choices without an autoregressive exchange decoder.
Candidate features encode action meaning; candidate position is not an
input. The learner receives raw observations rather than the teacher's
exact deficit, ukeire or ranking computations.

Train on the teacher's selected semantic action while retaining every
legal candidate. Version feature meaning and ordering together; matching
tensor dimensions alone do not establish compatibility.
