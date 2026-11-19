# Experiment Log 2026-11-20

## Public State

- Current best public submission remains `58438557` at LB `0.702`.
- The public `0.704` notebook is the current strongest scored public
  lineage, but the saved best submission is not replaced until an ablation
  completes.
- Public kernels newer than `0.704` were audited but have no verified
  higher LB. Their unverified marks are not treated as scores.

## Learned Fable / GLM Gate

The gate is a prompt-independent three-class logistic model over:

- Base probabilities for Claude Fable, GLM-5.3, and GLM-5.3-Flash.
- The GLM-160 likelihood member.
- The 36 existing style features.
- Base and GLM entropies, margins, mass, differences, top-class agreement.

Training uses `GroupKFold(groups=prompt_id)` with the same outer and inner
split policy as `scripts/train_family_tree_gate.py`.

At fixed strength `0.5`, margin `0.20`:

- Full prompt-grouped OOF: `0.639411 -> 0.663905`.
- Sparse2: `0.644696 -> 0.669232`.
- Paired sparse2 delta: `+0.024535`.
- Seed win rate: `50/50`.
- Prompt-bootstrap interval: `[+0.021675, +0.026989]`.
- Active rows: `355 / 2699`; changed rows: `124 / 2699`.
- Target recall:
  - Fable: `0.4094 -> 0.5436`.
  - GLM-5.3: `0.3867 -> 0.5133`.
  - GLM-5.3-Flash: `0.3667 -> 0.5467`.

The gate remains positive after applying the public 0.704 GLM-to-Fable
margin routing. It is not only duplicating that heuristic.

Rejected variants:

- Binary Fable-vs-GLM gate with base GLM mass split afterward:
  `+0.0045` full OOF, far below the three-class gate.
- Add all 144 mined structure/function-word features:
  the gate dropped to `+0.0133` at its best fixed strength.
- Add the external text boundary reader:
  full OOF fell below the plain learned gate.
- Qwen14 conditional GPT/Gemini gates had no usable activation because the
  18-class base rarely puts those families first.
- Applying Q35 separately inside the ClaudeGLM, GPTKimiQwen, and Gemini
  groups was weaker than the global Q35 blend. The best context-robust
  group version was only about `+0.005` on the full OOF.
- A LogisticRegression stack over base/Q35 logits, differences, products,
  and entropy features underperformed the direct log blend.
- Adding Q35 probabilities as extra features to the Fable family gate
  reached at most `+0.0197`, below the direct conservative Q35 blend with
  the learned gate.
- Q35 probability TTA over the 1/3/6-row OOF contexts was tested. The best
  50-seed quick mean was about `+0.0305`, effectively tied with the single
  3-row context result and not enough to justify a more complex runtime.
- The new `wenimohamed/name-that-llm-stylometry-features-v1` dataset has
  64 numeric features, but adding it to the Fable family gate changed OOF
  from `0.658330` to only `0.658700` at its best `C`.
- The v87 "exact family mass" change is algebraically equivalent to the
  existing `logsumexp(z[:, columns])` refinement. It is not a real model
  gain by itself.
- A fold-safe 22-marker typographic prior changed the
  Fable+Q35 context3 OOF from `0.671685` to `0.672056`, inside noise.
- The full 21-component Q35 bundle was worse than the Q35-only subset.
  At 3-row context its best Fable blend was only about `+0.0115`, versus
  `+0.0306` for the Q35-only conservative branch.
- Component-level additions on top of Q35 were also weak. The best extra
  component was `deberta` at weight `0.05`, with a minimum context delta of
  only `+0.0015`.
- A nested class-bias/temperature calibration over Fable+Q35 did not beat
  the raw blend. The best 3-row result was `0.67464` versus `0.67528`
  before calibration.

## Public Q35 OOF Bundle

The public `ntl-oof-q35-heads` bundle is inside
`lightningv08/ntl-olmo2-native24-final-inference-assets`.

- All bundle checksums were verified against `bundle_manifest.json`.
- `oof_predictions.npz` has `2699` rows and `37` components.
- Scenario 6 calibrated nested OOF is `0.574912`.
- A weighted Q35-only subset (using the bundle's published Q35 weights)
  reaches `0.549734`.

Q35 is weak standalone but highly complementary:

| Candidate | Full OOF |
| --- | ---: |
| Qwen stack | 0.639411 |
| Stack + Q35(w=0.4, full context) | 0.663132 |
| Learned Fable gate | 0.663905 |
| Fable gate + Q35(w=0.4, full context) | 0.686127 |

Sparse2 for the Q35-only weighted component:

- Stack + Q35(w=0.4): `0.665497`.
- Paired delta: `+0.020801`.
- Seed win rate: `45/50`.
- Interval: `[+0.014986, +0.026395]`.

Sparse2 for the combined two-stage signal:

- Fable + Q35(w=0.4): `0.687475`.
- Paired delta versus stack: `+0.042778`.
- Seed win rate: `50/50`.
- Interval: `[+0.036562, +0.048738]`.

Q35 center heads are prompt-dependent. Re-testing with the bundle's 1-row
and 3-row context scenarios exposed the full-context result as optimistic:

| Q35 context | Fable + Q35(w=0.4) OOF | Sparse2 delta | Seed wins |
| --- | ---: | ---: | ---: |
| 1 row | 0.655377 | +0.014301 | 0.76 |
| 3 rows | 0.672804 | +0.029776 | 1.00 |

At the safer weight `0.15`:

| Q35 context | Fable + Q35(w=0.15) OOF | Sparse2 delta | Seed wins |
| --- | ---: | ---: | ---: |
| 1 row | 0.666122 | +0.025325 | 0.96 |
| 3 rows | 0.671685 | +0.030584 | 0.98 |

Deployment therefore uses `Q35_WEIGHT=0.15`, not `0.4`.

## Candidate Kernels

### Fable only

- Local directory: `kaggle_fable_gate_0704`.
- Kaggle kernel: `yyyyhhhhwww/ntl-0704-fable-learned-gate`.
- Derived from the exact public 0.704 version 12.
- Version 2 uses gate strength `0.5`, margin `0.20`.
- If the gate does not activate, behavior matches the public 0.704 path.

### Fable plus Q35

- Local directory: `kaggle_fable_q35_0704`.
- Kaggle kernel: `yyyyhhhhwww/ntl-0704-fable-q35`.
- Q35 is blended into `FINAL_LOGITS` at log weight `0.15` before the existing
  GLM likelihood and Apex stages.
- The learned Fable gate runs after the GLM/Apex routing.
- Q35 failure keeps the original 0.704 logits and still allows the Fable
  gate to run, so the two additions degrade independently.

## Rejected External Text Reader

The char TF-IDF plus mined-feature boundary reader used external rows and
official prompt-grouped folds:

- External three-class balanced accuracy: about `0.429`.
- Official target OOF folds: `0.4111`, `0.3444`, `0.3444`, `0.5111`,
  `0.3966`.
- Mean: about `0.4015`.

It is far below the learned GLM/style gate and did not improve the gate as
an additional feature block. Reject this branch.

## Next Public Ablation

1. Submit the Fable-only version 2 if its commit stays mount-valid.
2. Submit the combined Fable plus Q35 version 1 only after its commit
   completes and preserves the fallback.
3. Keep `58438557` as the fallback until a public score exceeds `0.702`.
4. Record the exact public score and sparse2-to-LB delta in
   `validation/lb_calibration.csv`.

## PNASF Overnight Experiment

The new PNASF reader was trained without using the public kernel decision
chain or the q35 public heads.

- External paired sample: `20,000` rows from `8,856` same-prompt groups.
- Frozen backbone: 4-bit `Qwen3-4B-Base`, layers `24,36`.
- Feature width: `20,480` hidden-summary dimensions plus 36 style values.
- Encoder: prompt-adversarial contrastive encoder, 256-D normalized output.
- Objectives: supervised contrastive, prompt-centroid contrastive, and
  gradient-reversal prompt adversary.
- Official validation: five prompt-grouped folds with a balanced logistic
  head.

Best encoder variant:

- `1200` steps, `adv_weight=0.15`, batch `128`, `C=1.0`.
- Standalone official OOF: `0.4323`.
- Raw style-transfer baseline: `0.2456`.
- Raw Qwen-feature prototype transfer: `0.1460`.

PNASF is complementary rather than dominant:

| Candidate | Full OOF | Sparse2 mean | Sparse2 delta vs stack |
| --- | ---: | ---: | ---: |
| Stack | 0.639411 | 0.644696 | 0 |
| Fable learned gate | 0.663905 | 0.669232 | +0.024535 |
| Fable + q35(0.15) | 0.671685 | 0.675280 | +0.030584 |
| Fable + q35(0.15) + PNASF(0.10) | **0.685391** | **0.689318** | **+0.044621** |

Integration gate:

- Sparse2 seed wins: `50/50`.
- Prompt-bootstrap interval: `[+0.039979, +0.049239]`.
- This is local validation evidence, not a public LB score.

Saved artifacts:

- `features/pnasf_external_qwen4_l24_36.npz`
- `features/pnasf_official_qwen4_l24_36.npz`
- `validation/pnasf_run3_adv015/pnasf_encoder.pt`
- `validation/pnasf_run3_adv015/pnasf_official_oof.npy`
- `validation/pnasf_run3_best_report.json`
- `scripts/extract_pnasf_features.py`
- `scripts/train_pnasf.py`
- `scripts/predict_pnasf_kernel.py`

## PNASF Kernel Launch

- `yyyyhhhhwww/ntl-0704-pnasf` version 2 was pushed on 2026-11-20 and entered
  `KernelWorkerStatus.RUNNING`.
- A deployment parity check found that the kernel inference copy of the style
  extractor counted straight quotes in the curly-double-quote feature. The
  training implementation counts only `“` and `”`.
- The inference copy was corrected before launch. All 2,699 official training
  texts now match `namellm.style.style_frame` exactly with `max_abs=0.0`.
- The embedded inference script compiles and the notebook passes
  `nbformat.validate`.
- The deployed dataset is `yyyyhhhhwww/ntl-pnasf-run3`, containing
  `pnasf_encoder.pt`, `pnasf_classifier.joblib`, and `pnasf_config.json`.
- Submission `58438557` at public LB `0.702` remains the fallback. Do not
  replace it until the PNASF kernel finishes and its submission is evaluated.

## DeepSeek-OCR Query Flow Diagnostic

The DeepSeek-OCR 2 query idea was tested as a prompt-independent sequence
reader over the existing fold-0 Qwen layer-36 PCA-128 token states.

- Content tokens form fixed memory.
- A learnable query bank reads memory through cross-attention.
- Query-to-query self-attention can be causal or symmetric.
- Only query outputs and the existing Qwen summary PCA features reach the
  classifier.

The matched Attention MIL baseline reached fold-0 balanced accuracy `0.4481`.

Single-model query results:

| Variant | Balanced accuracy |
| --- | ---: |
| Causal K=8, one layer | 0.4315 |
| Symmetric K=8, one layer | 0.4352 |
| Causal K=16, one layer | 0.4352 |
| Symmetric K=16, one layer | 0.4370 |
| Causal K=16, two layers | 0.4333 |
| Causal plus 36-style reconstruction w=0.05 | 0.4296 |
| Causal plus 36-style reconstruction w=0.20 | 0.4222 |
| Causal plus prompt adversary w=0.05, seed 42 | 0.4407 |

A four-seed geometric query ensemble reached `0.4519`. Its fixed-weight
log blend with Attention MIL reached `0.4593` at query weight `0.35`.
A two-seed ordinary Transformer control also raised Attention MIL to
`0.4574`, so much of the apparent gain is generic ensemble diversity rather
than query-specific behavior.

The mandatory sparse2 check used 50 seeds, two rows per validation prompt,
and 500 prompt-bootstrap replicates:

- Attention MIL baseline mean: `0.4490`.
- Best query blend at weight `0.40`: `0.4682`.
- Paired delta: `+0.0192`.
- Seed win rate: `0.66`, below the required `0.80`.
- Prompt-bootstrap interval: `[+0.0131, +0.0245]`.
- Reader gate: rejected.

Prompt adversary over four seeds produced `0.4407`, `0.4241`, `0.4148`, and
`0.4333`. Its geometric ensemble was `0.4481`, and its Attention MIL blend
was only `0.4574`; correlation with Attention MIL also increased to about
`0.90`.

As a diagnostic only, the query ensemble raised the prompt-dependent
cluster-balanced fusion on the fold-0 validation rows from `0.6185` to
`0.6315` at weight `0.10`. It did not improve the current Qwen stack blend.

Conclusion: reject the causal-query branch as a deployment candidate. Keep
`scripts/train_sequence_downstream.py --model query_flow` and the saved
diagnostics for reproducibility, but do not expand it to five folds or add it
to submission `58438557`.

## PNASF Submission

- Code submission `58578599` uses
  `yyyyhhhhwww/ntl-0704-pnasf` version 2.
- It combines the public 0.704 path, the learned Fable/GLM gate, Q35 at
  weight `0.15`, and PNASF at weight `0.10`.
- Status at launch was `PENDING`; public LB was recorded as `TBD`.
