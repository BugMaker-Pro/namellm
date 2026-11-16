# Sparse2 Lockbox

This directory is the only accepted standard for deployment-oriented
experiments in Name That LLM.

## Protocol

- Five outer folds are grouped by `prompt_id`.
- Each seed samples two validation rows from every held-out prompt.
- Prompt-dependent operations must run after that sampling step. A static
  branch is valid only when its features and predictions do not depend on
  other rows from the same prompt.
- Hyperparameters are selected only inside outer-train prompts.
- Report mean, standard deviation, fifth percentile, paired delta against a
  parent, seed win rate and a prompt-bootstrap interval.

Run the standard protocol with:

```powershell
python scripts/run_sparse2_lockbox.py `
  --baseline-static parent=features/example_parent.npy `
  --candidate-static candidate=features/example_candidate.npy `
  --baseline parent `
  --output D:/tmp/sparse2_results.json `
  --per-seed-output D:/tmp/sparse2_per_seed.csv
```

Prompt-dependent candidates use `NAME=MODULE:FUNCTION` with:

```python
def function(train_df, valid_df, seed, fold):
    # valid_df already contains exactly two rows per held-out prompt.
    return probabilities  # shape: (len(valid_df), 18)
```

## Calibration

`lb_calibration.csv` stores matched sparse2 and public-LB observations.
Until at least five matched submissions exist, use paired deltas only. A
candidate does not count as successful until a real submission exceeds the
target public score.
