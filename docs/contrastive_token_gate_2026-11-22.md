# Contrastive Token Gate

Updated 2026-11-23 after the overnight capacity, seed-ensemble and sparse2
checks.

## Leakage Correction

The earlier `features/qwen3_seq_fold0` hidden sequences were produced with
the fold-0 QLoRA adapter, not the unadapted base model:

- layer-36 mean absolute difference to the fold-0 adapter states:
  `0.014-0.017`;
- layer-36 mean absolute difference to base Qwen3-4B:
  `0.502-0.535`.

The sequence metadata train/valid masks also match fold 0 exactly. Therefore
the earlier multi-seed contrastive OOF `0.7395` was not strictly fold-safe for
folds 1-4 and is retained only as a diagnostic, not as a deployment estimate.

The corrected run uses:

- `features/qwen3_base_hidden_layers_8_36/layer8`;
- `features/qwen3_base_hidden_layers_8_36/layer36`;
- no QLoRA adapter;
- width 64, 4 heads, dropout 0.40, 14 epochs;
- one prompt-grouped seed, with per-fold model checkpoints.

Corrected results:

| Metric | Value |
| --- | ---: |
| Standalone OOF | **0.60833** |
| Parent-gated OOF at `w=0.18` | **0.73505** |
| Sparse2, seed start 20261166 | **0.73632** |
| Sparse2, seed start 20261216 | **0.73380** |
| Paired deltas versus parent | **+0.09655 / +0.09330** |
| Seed win rate | 50/50 in both runs |
| Prompt bootstrap 95% | [+0.08989,+0.10311] / [+0.08579,+0.10103] |

The corrected fixed candidate is
`validation/layerwise_contrastive_clean_e14_5fold/blends/gated_w0.18.npy`.
No public submission was created. The existing **0.703** submission remains
unchanged.

## Pre-Correction Candidate (Not Fold-Safe)

The strongest prompt-grouped and sparse2 branch uses:

- Qwen3-4B hidden token states at layers 8 and 36;
- QKV token signatures at layers 2, 3, 20, 23 and 35;
- a local token Transformer plus independent QKV layer attention;
- `d_model=64`, 4 heads, dropout 0.40;
- 10 epochs, 18-class CE, vendor6 CE, group3 CE, hard-pair CE and SupCon;
- three 64-dimensional seeds:
  - standard dropout 0.25, seed base 20261322;
  - standard dropout 0.25, seed base 20261522;
  - dropout 0.40, seed base 20261322;
- equal arithmetic probability averaging;
- application only when parent top-1 is not `gpt-5.6-terra` or
  `claude-fable-5`;
- fixed blend weight **0.30**.

The candidate is a diagnostic result, not a deployable submission artifact.
No public submission was created or replaced. The account-owned best remains
the existing **0.703** submission.

## Main Results

Prompt-grouped 5-fold balanced accuracy:

| Branch | Standalone OOF | Parent OOF after fixed gate |
| --- | ---: | ---: |
| Previous two-seed 256-wide branch | 0.3667 | 0.6531 at `w=0.22` |
| 64-wide standard seed 1 | 0.5194 | 0.6947 at `w=0.18` |
| 64-wide standard seeds 1+2 | **0.6028** | 0.7324 at `w=0.26` |
| 64-wide seeds 1+2+dropout0.40 | **0.6358** | **0.7395 at `w=0.30`** (not fold-safe) |

The fixed final candidate is
`validation/layerwise_contrastive_micro3_5fold/blends/gated_w0.30.npy`.
Its OOF balanced accuracy is 0.73950 versus parent 0.63941.

## Pre-Correction Sparse2 Replication

The final candidate was checked on three independent 50-seed sparse2 runs.
Each run samples two rows per prompt before evaluation.

| Seed start | Parent | Candidate | Paired delta | Win rate | Prompt bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | --- |
| 20261116 | 0.64470 | **0.74679** | +0.10209 | 50/50 | [+0.09502, +0.10892] |
| 20261166 | 0.63977 | **0.74377** | +0.10400 | 50/50 | [+0.09786, +0.11103] |
| 20261216 | 0.64051 | **0.74082** | +0.10031 | 50/50 | [+0.09303, +0.10764] |

All three runs pass the project small-increment, new-reader and
fallback-reader acceptance gates.

## Capacity Diagnosis

All capacity comparisons below use the same fold 0, seed base 20261322 and
10 epochs.

| Width | Balanced accuracy |
| --- | ---: |
| 384 / 6 heads | 0.3333 |
| 256 / 4 heads | 0.3778 |
| 192 / 4 heads | 0.3852 |
| 128 / 4 heads | 0.4722 |
| 64 / 4 heads | **0.5148** |
| 32 / 4 heads | 0.4981 |

The original 256-dimensional model was over-parameterized for 2,699 official
rows. Capacity reduction was the dominant improvement, not a wider head.

Matched fold-0 regularisation checks at width 64:

| Variant | Balanced accuracy |
| --- | ---: |
| dropout 0.25 | 0.5148 |
| dropout 0.40 | **0.5611** |
| dropout 0.50 | 0.5296 |
| batch 16 instead of 8 | 0.5426 |

Dropout 0.40 is the useful setting. The larger contrastive batch was positive
but weaker and was not expanded to five folds.

## Rejected Experiments

- A third 256-wide seed lowered the ensemble from 0.3667 to 0.3630; equal
  averaging diluted the two stronger seeds.
- Giving that weak seed 10% weight recovered standalone OOF to 0.3674, but the
  gated blend did not beat the two-seed candidate.
- Excluding parent class `gpt-5.6-luna` improved OOF by about 0.00185, but the
  direct sparse2 delta over the previous candidate was only
  `+0.0011~+0.0021`, with 52%-68% seed wins. It failed the direct acceptance
  check and was not adopted.
- Confidence and parent-margin gates did not beat applying the blend to every
  eligible row.
- Geometric seed averaging improved standalone OOF but degraded the gated
  parent blend.
- Adding weaker 192/256-wide seeds to the two 64-wide seeds reduced the final
  gated OOF despite increasing standalone ensemble diversity.

## Reproduction

Training:

```powershell
.\.venv\Scripts\python.exe scripts\train_layerwise_contrastive_gpu.py `
  --fold 0 --epochs 10 --seed-base 20261322 `
  --d-model 64 --heads 4 --dropout 0.40 `
  --output validation/layerwise_contrastive_micro_dropout40_fold0
```

Merge:

```powershell
.\.venv\Scripts\python.exe scripts/merge_contrastive_seed_ensemble.py `
  --seed-template "validation/layerwise_contrastive_micro_fold{fold}" `
  --seed-template "validation/layerwise_contrastive_micro_seed2_fold{fold}" `
  --seed-template "validation/layerwise_contrastive_micro_dropout40_fold{fold}" `
  --weights 0.26,0.30 `
  --output-dir validation/layerwise_contrastive_micro3_5fold
```

Sparse2:

```powershell
.\.venv\Scripts\python.exe scripts/run_sparse2_lockbox.py `
  --baseline-static stack=features/qwen3_embedding_stack_blend_oof.npy `
  --candidate-static micro3_w030=validation/layerwise_contrastive_micro3_5fold/blends/gated_w0.30.npy `
  --baseline stack --seed-start 20261116 `
  --output validation/layerwise_contrastive_micro3_5fold/sparse2_final_20261116.json `
  --per-seed-output validation/layerwise_contrastive_micro3_5fold/sparse2_final_20261116_per_seed.csv
```

## Packaging Blocker

The corrected training run now writes per-fold model checkpoints. The
repository still lacks a runtime that accepts arbitrary `test.csv` input for
the exact hidden/QKV feature stack.

Before any submission:

1. Parameterize hidden-layer 8/36 and QKV 2/3/20/23/35 extraction for `test.csv`
   or the hidden rerun input.
2. Reproduce the fixed gate and weight 0.18 from raw parent and clean token
   probabilities.
3. Validate finite probabilities, row ordering and class coverage.
4. Run a public ablation without replacing the existing 0.703 submission
   until the result is at least 0.704.

## Primary Artifacts

- `scripts/train_layerwise_contrastive_gpu.py`
- `scripts/extract_base_hidden_token_layers.py`
- `scripts/audit_token_feature_provenance.py`
- `scripts/merge_contrastive_seed_ensemble.py`
- `scripts/tune_contrastive_confidence_gate.py`
- `validation/token_feature_provenance.json`
- `validation/layerwise_contrastive_clean_e14_5fold/`
- `validation/layerwise_contrastive_clean_e14_5fold/blends/gated_w0.18.npy`
- `validation/layerwise_contrastive_clean_e14_5fold/sparse2_clean_20261166.json`
- `validation/layerwise_contrastive_clean_e14_5fold/sparse2_clean_20261216.json`
- `validation/layerwise_contrastive_micro3_5fold/`
- `validation/layerwise_contrastive_micro3_5fold/blends/gated_w0.30.npy`
- `validation/layerwise_contrastive_micro3_5fold/sparse2_final_20261116.json`
- `validation/layerwise_contrastive_micro3_5fold/sparse2_final_20261166.json`
- `validation/layerwise_contrastive_micro3_5fold/sparse2_final_20261216.json`
