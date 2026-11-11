# Current architecture

The [fixed gameplay rules](../baseline-rules.md) define game semantics.
These guides describe the implemented subsystem boundaries:

| Subsystem | Guide | Responsibility |
| --- | --- | --- |
| Rule foundation | [Foundation](foundation.md) | Tile model, pure shape and scoring facts |
| Authoritative hand | [Game](game.md) | Turn state, legal decisions and settlement |
| Policies | [Policy](policy.md) | Player-visible action choice and optional analysis cache |
| Execution | [Runner](execution.md) | Driving games, traces and batch reports |
| Experiments | [Research tools](experiments.md) | Paired policy comparisons and profiling |
| Neural pipeline | [Neural](neural.md) | Encoding, collection, preparation, training and online evaluation |

The core exposes Game → player-scoped Decision → Policy → Action → Game.
Runner drives this interaction without interpreting rules. Rule helpers are
pure; Game owns state changes and hidden information.

Neural collection uses the same public Game/Decision boundary, while offline
training consumes prepared immutable caches. The online PyO3 environment
exposes encoded decisions to Python without giving Python direct access to
Game internals. The core depends on neither neural nor Python packages.

[Development](../development.md) ·
[Runnable workflows](../README.md) ·
[Unimplemented research methods](../research-decisions.md)
