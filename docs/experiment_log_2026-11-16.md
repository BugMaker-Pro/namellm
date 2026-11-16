# Name That LLM Experiment Log - 2026-11-16

## Context

- Existing deployable submission: public LB 0.409.
- Existing full-group OOF: 0.6394, which did not transfer to the hidden test.
- Public MIT reproduction: grouped CV 0.418, public LB 0.389.
- Public scored fallback: 0.696 notebook built from public data, models,
  adapters, and kernels.

## Sparse Prompt Validation

The expanded OOF probabilities were resampled at 1, 2, 4, and 8 rows per
prompt using 100 seeds. Reapplying centering and cluster balancing to the
same OOF matrix does not remove the prompt-dependent information already
present in those predictions.

Selected two-row results:

| Branch | Raw | Center | Balance |
| --- | ---: | ---: | ---: |
| Qwen embedding fusion | 0.5562 | 0.5450 | 0.5441 |
| Cluster-balanced fusion | 0.6296 | 0.5944 | 0.5930 |
| Rich stack | 0.6297 | 0.5948 | 0.5936 |
| Stack blend | 0.6436 | 0.6175 | 0.6168 |

The key finding is not that the raw numbers are deployable. It is that
sampling rows after the OOF predictions were created cannot undo the
full-group information used while fitting those branches. New branches must
be trained without `group_relative`, prompt centering, or group balancing.

## Style Encoders

### StyleDistance

- Model: `StyleDistance/styledistance`, RoBERTa-base, 768 dimensions.
- Official train only:
  - PCA256: OOF 0.2997.
  - PCA256 plus absolute style features: OOF 0.3397.
- Official train plus Open Pool v14, 1,500 rows per label:
  - OOF 0.3682 across five prompt-grouped folds.
  - Per-fold: 0.3481, 0.3759, 0.3704, 0.3907, 0.3557.

The external pool adds about 2.9 points over the same official-only branch.
The branch is cheap and group-independent, but it is not strong enough to
replace the public fallback.

### LUAR

- Model: `rrivera1849/LUAR-MUD`, episode embedding, 512 dimensions.
- Official train plus absolute style features: OOF 0.3219.
- A 50:50 LUAR and StyleDistance blend reaches about 0.3642 before external
  data, showing representation diversity but limited standalone quality.

### Rejected Controls

- A diagonal Gaussian classifier with group-relative features only on
  inferred 2-5-row groups reached OOF 0.1612 on StyleDistance embeddings.
  The partial-group transform was too noisy at this representation quality.
- RidgeFT with PCA256, absolute style features, and 2,048 random features
  reached OOF 0.1019. The official implementation remains useful as a
  method reference, but this configuration is rejected.

## Blend Diagnostics

StyleDistance external OOF blended with existing branches:

| Base | Base OOF | Best style weight | Blended OOF |
| --- | ---: | ---: | ---: |
| Compact TF-IDF | 0.3208 | 0.40 | 0.4163 |
| Summary/funnel | 0.5268 | 0.30 | 0.5368 |
| Qwen embedding fusion | 0.5545 | 0.10 | 0.5575 |
| Existing rich stack blend | 0.6394 | 0.03 | 0.6390 |

The result improves weak bases but has almost no effect on the existing rich
stack. Given the hidden-test discrepancy, the stack number is not considered
deployable evidence.

## Deployment

- The exact public 0.696 notebook was fetched from version 28 with all
  dataset, kernel, and model sources.
- Kaggle commit version 2 completed with all 35 inputs present and no mount
  failures.
- Code submission `58397368` was launched from that exact kernel version.
- Submission result: **public LB 0.696**. The account is around rank 501 of
  6,063 teams; the current leader is 0.775.

## Next Gate

Qwen3-14B QLoRA fold 0 is trained on external pool data, then on official
train with a two-row-per-prompt validation subset. The branch is retained
only if sparse validation reaches at least 0.60 and its errors are
complementary to the public fallback.

Qwen14 checkpoints are saved every 50 optimizer steps. The first two
portable backups are:

| Step | SHA256 |
| ---: | --- |
| 100 | `6EFAED2488F953EB216DA08A3F6E1431AE9AEA71FB54D5B78A838B83AA2F88C8` |
| 200 | `BC0EEACB1D68AD25BD9D3C5B57A73C50EF18941850B09193B79C328AEAC721C3` |
| 300 | `F0639F14E6AE32899279831C77F59D4F181FA5EE00B476ED421F16F22D842E28` |

Resume command:

```bash
cd /content/namellm
export HF_HOME=/content/hf_cache
python -u scripts/train_external_qwen14.py \
  --model-dir Qwen/Qwen3-14B-Base \
  --pool data/style_pool_1500.parquet \
  --output-dir /content/runs/qwen14_fold0_resumeable \
  --resume-stage 1 \
  --stage1-max-steps 800 \
  --stage2-max-steps 600 \
  --checkpoint-every 50
```

## Remote Access Caveat

The Colab runtime was launched with a resumable Qwen14 process. The CLI
OAuth refresh later timed out because the local proxy lost its upstream
connection. The remote training process was still active at the last
successful probe (stage1 step 420, loss around 2.0-2.3). Local portable
backups through step 400 are retained in `artifacts/qwen14_backup/`.

To take over the running session after re-authentication:

```bash
colab status -s namellm
colab ssh -s namellm -- bash -lc "tail -50 /content/namellm/qwen14_fold0_resumeable.log"
```
