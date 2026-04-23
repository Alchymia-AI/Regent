# Recipes

Reproducible recipes for training Regent on a specific domain. Same 4-phase pipeline, same stability constraints, same scaling rules. Domain-specific variation is data and a few config fields.

| Recipe | Target | Status |
|---|---|---|
| [code-model.md](code-model.md) | Python code | Validated at 72M on M3 Ultra |
| [chat-model.md](chat-model.md) | General conversation | Spec only, no reference run yet |

## Adding a recipe

1. Copy `code-model.md` as a template.
2. Identify HuggingFace datasets per phase; record licenses.
3. Decide Phase 1 data mix fractions.
4. Adjust config deltas (sequence length, d_state, vocab size).
5. Run a 72M–500M validation; record the trajectory.
6. Submit alongside any new scripts or configs.

Every recipe must be runnable end-to-end from a clean clone against its declared hardware and compute budget.
