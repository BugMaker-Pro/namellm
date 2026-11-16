# Name That LLM Project Memory

## User Preferences

- Competition: https://www.kaggle.com/competitions/name-that-llm
- Open Kaggle with Microsoft Edge and the normal logged-in profile.
- Use the repository environment at `D:\namellm\.venv`.
- Do not create another virtual environment.
- Reproduce or synchronize dependencies with `uv sync --frozen`.
- The existing `D:\rsna\.venv` was moved into the repository on 2026-11-15.

## Competition Status

- The user has joined the competition and accepted its rules.
- The Data, Rules, and Submit Prediction pages are available in the logged-in
  Edge session.
- Public description: identify which of 18 language models generated a
  self-contained paragraph.
- Metric: balanced accuracy.
- Train: 2,699 rows and 150 prompts, normally 18 rows per prompt.
- Interactive test: 3 example rows. The hidden rerun has about 4,000 passages
  whose prompts do not appear in the training set.
- Hidden-test class counts are not guaranteed to be equal. Per-prompt quota or
  Hungarian assignment is only a diagnostic, not a safe final prediction rule.
- Submission columns: `id`, `model`.

## Raw Data

Downloaded from the logged-in Kaggle page and extracted to `data/raw`.

- `train.csv`: SHA256
  `40A22CE8D88C5F46FEB4637A099DB815BEF5F713FFE53E89DBE1B7B21DE77707`
- `test.csv`: SHA256
  `4545AA3A3459F4C0724F97AD5BD06C4CAA3506BDFEA5FAB905E8203184FC0C57`
- `sample_submission.parquet`: SHA256
  `CCF86796B80FD7F83D8463664603FD1208E17559ABCF416F198776B819646188`
- The original ZIP is
  `C:\Users\Administrator\Downloads\name-that-llm.zip`.

## Models And Validation

- Always validate with prompt-level grouping because hidden-test prompts are
  unseen. Use `GroupKFold` with `groups=prompt_id`; use
  `StratifiedGroupKFold` only when fold class balance needs explicit control.
- Never use random row-level validation for reported scores.
- Balanced classes make balanced accuracy the primary model-selection metric.

Observed prompt-grouped 5-fold results:

| Experiment | Balanced accuracy |
| --- | ---: |
| Word TF-IDF + LogisticRegression, C=3 | 0.2771 |
| Char-wb TF-IDF + LogisticRegression, C=3 | 0.3060 |
| Word + char-wb + LogisticRegression, C=3 | 0.3100 |
| BGE-small CLS embeddings + LogisticRegression | about 0.153 |
| DeBERTa-v3-small fine-tune, best prompt split | 0.2352 |
| Preserved-case `char(2,6)` + LinearSVC, C=1 | 0.3204 |
| Preserved-case `char(1,6)` + LinearSVC, C=1 | 0.3234 |
| Preserved-case `char(1,6)` + LinearSVC, C=2 | 0.3249 |
| TF-IDF + Qwen3-4B 4-bit relative token stats, stacked | 0.3886 |
| Full feature union + three-group funnel | 0.4705 |
| Enhanced Qwen token summaries + downstream classifier | 0.4694 |
| Equal summary/funnel blend | 0.5231 |
| Prompt-grouped power-calibrated blend | 0.5268 |
| Qwen3-Embedding-4B, layers 16/24/32/36 | 0.4797 |
| Qwen3-Embedding-4B seven-layer BiGRU | 0.4797 |
| Nested three-way embedding fusion | 0.5512 |
| Nested seven-layer embedding fusion | 0.5545 |
| Prompt-centered seven-layer fusion | 0.5742 |
| Nested prompt-centered + Sinkhorn balance | 0.6242 |
| Fixed cluster-balanced fusion | 0.6264 |
| Rich LogisticRegression stack | 0.5583 |
| Rich stack + cluster balance | 0.6320 |
| Nested rich stack + cluster balance | 0.6250 |
| 0.45 stack blend with cluster-balanced fusion | **0.6394** |

The current best prompt-grouped OOF result is the cluster-balanced
Qwen3-Embedding fusion blended with a rich LogisticRegression stack at
**0.6394**. The fixed primary cluster-balanced branch alone is 0.6264.
Lowercasing and BGE semantic embeddings discard useful style signals, so
preserve case and punctuation.

Prompt-level validation details:

- A fixed 120/30 prompt split gave plain balanced accuracy 0.3111.
- BGE clustering recovered all 30 validation prompt groups exactly.
- Hungarian assignment raised that single holdout from 0.3111 to 0.4130.
- Grouped 5-fold recovery plus Hungarian averaged 0.3830, but this is diagnostic
  because the hidden test may have unequal classes per prompt.

## Important Implementation Notes

- Preserve case, punctuation, and accents for character features:
  `lowercase=False`, `strip_accents=None`.
- Transformer checkpoints may load in half precision. When fine-tuning
  DeBERTa, explicitly pass `dtype=torch.float32`; BF16/FP16 caused NaNs.
- A combined sparse character matrix plus dense style features made Liblinear
  converge too slowly. Prefer separate style and text models, then blend their
  out-of-fold probabilities or decision scores.
- DeBERTa-small trained for four epochs but only reached 0.235 balanced
  accuracy. Do not spend more time on that exact configuration without a
  materially different setup.
- A hierarchical 6-family model underperformed the flat classifier.

## Public Solution Findings

- Public reference: https://github.com/stewari1210/name-that-llm
- Reference lockbox result: E004 three-group funnel reaches about 0.486.
- Main gain comes from a rich feature union, not the funnel itself:
  TF-IDF, blind-cluster-relative TF-IDF, 37 style features, and frozen-LM token
  statistics from Qwen3-4B-Base.
- Stage 1 predicts ClaudeGLM / Gemini / GPTKimiQwen with about 0.94 accuracy.
- Stage 2 trains one specialist per group.
- Public notebooks around 0.54-0.62 use LoRA-fine-tuned discriminative models.
- The leaderboard top is around 0.77, so the next serious step is a stronger
  discriminative model, likely 4B QLoRA or a comparable transformer.

## Current Assets

- `models/Qwen3-4B-Base`: ModelScope copy, all three weight hashes verified.
- `features/e002_qwen3-4b-base-4bit.parquet`: 2,702 rows of frozen Qwen token
  statistics generated on the local RTX 5060 Ti.
- `scripts/eval_funnel.py`: TF-IDF + relative + style + LM-stat feature funnel.
  Current OOF balanced accuracy is 0.4705.
- `notebooks/train_qlora.ipynb`: interactive 4B QLoRA notebook with per-step
  loss, epoch validation, confusion matrix, and OOF artifact export.
- The Jupyter kernel is installed as `namellm` in the repository `.venv`.
- The full fold-0 one-epoch QLoRA run completed 270/270 steps and reached
  validation balanced accuracy 0.1722, so it needs more epochs and/or better
  hyperparameters before it is competitive.
- Fold-0 adapter and metrics are in `artifacts/qlora_fold0`; validation OOF
  probabilities are in `features/qlora_fold0_oof.parquet`.
- The notebook falls back to `D:/tmp/namellm_qlora_outputs` if the execution
  sandbox blocks writes inside the repository.
- `features/qwen3_token_features_fold0.npz` contains Qwen3 tokenizer hidden
  sequence summaries from layers 24 and 36: token mean, std, last token, and
  adjacent-token delta. Feature dimension is 20,480.
- `features/qlora_token_downstream_fold0_probs.npy` contains the fold-0
  downstream LogisticRegression validation probabilities.

## Token-Level Downstream Result

- Tokenizer: `models/Qwen3-4B-Base` through `AutoTokenizer`.
- Hidden states: Qwen3 layers 24 and 36, 2,560 dimensions per token.
- Downstream model: StandardScaler + PCA(256) + LogisticRegression(C=0.1).
- Fold-0 validation balanced accuracy:
  - original end-to-end QLoRA head: 0.1722
  - token-sequence downstream classifier: 0.4222
  - feature funnel on the same fold: 0.4648
  - 0.2 token model + 0.8 funnel blend: 0.5111
- This validates the two-stage approach: fine-tune representation, then train a
  separate downstream model from token-level features.
- Full fold-0 token sequences from Qwen layer 36 are in
  `features/qwen3_seq_fold0`; PCA-128 token sequences are in
  `features/qwen3_seq_fold0_pca128`.
- Sequence-model fold-0 validation:
  - BiGRU + attention: 0.3278
  - CNN + BiGRU: 0.3278
  - small Transformer: 0.3111
  - lightweight gated DeltaNet: about 0.3019 at epoch 4 before interruption
- Adding these sequence probabilities to the 0.5111 summary+funnel blend did
  not improve it. The sequence models currently overfit and do not add
  complementary signal.
- Raw 2,560-dimensional token sequences passed directly into BiGRU reached only
  0.2741 on fold 0. Adding PCA preprocessing improved it to 0.3278, and adding
  the summary branch improved it to 0.3981. The current representation is too
  noisy for an unregularized raw-token downstream model.
- A 1D residual Conv1D block before BiGRU reached 0.1426 on raw tokens and
  0.3870 on PCA-128 plus summary features. It did not beat the 0.3981
  summary+BiGRU model, but it remains a useful ensemble candidate.
- Fold-0 fusion from the 0.5111 summary+funnel base:
  - adding summary+BiGRU at weight 0.15: 0.5296
  - adding summary+ResNet-BiGRU at weight 0.10: 0.5167
  - the ResNet model adds no complementary signal beyond BiGRU.
- Gated Attention MIL reached 0.2870 on raw token sequences and 0.4278 on
  PCA-128 sequences plus summary features.
- A PCA-512 summary classifier with raw and blind-cluster-relative style and
  Qwen LM statistics reached 0.4722 on fold 0. LDA reached 0.4630, Ridge
  0.4167, and HistGradientBoosting 0.4056, so LogisticRegression remains best.
- ArcFace on the same summary/feature space overfit badly and reached only
  0.2278.
- Fold-0 blend search across funnel, enhanced summary, summary-BiGRU and
  Attention MIL peaked around 0.5278. The simpler 0.15 summary-BiGRU blend
  remains slightly better at 0.5296, inside fold-level noise.
- End-to-end QLoRA with all-token gated attention pooling and mean/max/last
  pooling completed 270/270 steps and reached 0.2481 on fold 0. It improved
  over the original 0.1722 single-vector classifier but remains well below the
  summary-feature downstream model.

## Complete 5-Fold Results

- Fold-specific QLoRA adapters for folds 1-4 are in `artifacts/qlora_folds`;
  fold 0 remains in `artifacts/qlora_fold0`.
- Five token-summary files are in `features/token_folds`.
- Enhanced summary downstream fold scores:
  - fold 0: 0.4722
  - fold 1: 0.4519
  - fold 2: 0.4889
  - fold 3: 0.4815
  - fold 4: 0.4526
- Enhanced summary OOF balanced accuracy: **0.4694**.
- Feature-funnel OOF balanced accuracy: 0.4705.
- Best equal-weight blend: **0.5231**.
- Nested prompt-grouped power calibration improves the blend to **0.5268**.
  All five outer folds selected summary power 0.75, funnel power 0.50, and
  equal model weights.
- A 36-dimensional LogisticRegression stack on log probabilities reaches only
  0.5079, so it is a negative control rather than the final combiner.
- Saved arrays:
  - `features/summary_oof.npy`
  - `features/funnel_oof.npy`
  - `features/summary_funnel_blend.npy`
  - `features/summary_funnel_power_blend.npy`
- `scripts/train_meta_fusion.py` reproduces the nested power calibration and
  the LogisticRegression control.

## Qwen3-Embedding-4B Branch

- `Qwen/Qwen3-Embedding-4B` runs fully unquantized on the free Colab T4:
  FP16 weights, no bitsandbytes, peak CUDA allocation about 7.6 GB.
- The embedding model is used only as a frozen feature extractor. Full
  parameter training does not fit; any later representation training on T4
  must use LoRA.
- Token statistics are mask-aware mean/std/last/adjacent-delta from layers
  selected in multiple passes. Extraction takes about 5 minutes for 2,699 rows
  and batch size 1.
- Blind-cluster-relative embedding features are essential: PCA-only OOF was
  0.3012, while adding group-relative and existing style/LM features raised the
  branch to 0.4705-0.4916.
- PCA 256 and PCA 512 have complementary fusion behavior even though PCA 256
  has the stronger standalone score.
- Layer averaging underperformed. Preserving the ordered layer sequence with a
  BiGRU is better:
  - four-layer BiGRU OOF: 0.4630
  - seven-layer BiGRU OOF: 0.4797
  - the seven-layer branch supersedes the four-layer branch in nested fusion
- Nested fusion selects approximately base 0.55, PCA512 0.05-0.15,
  PCA256 0.00-0.15, layer-concat 0.05-0.10, seven-layer BiGRU 0.15-0.25.
- Feature files:
  - `features/qwen3_embedding_4b_token_features.npz`: layers 24/36
  - `features/qwen3_embedding_4b_layers16_32.npz`: layers 16/32
  - `features/qwen3_embedding_4b_layers8_20_28.npz`: layers 8/20/28
  - `features/qwen3_embedding_4b_layers16_24_32_36.npz`: four-layer concat
  - `features/qwen3_embedding_4b_layers7.npz`: seven-layer concat
  - `features/qwen3_embedding_4b_layer_sequence7_oof.npy`: BiGRU OOF
  - `features/qwen3_embedding_fusion7_oof.npy`: final nested fusion OOF
- Reproduction scripts:
  - `scripts/extract_embedding_token_features.py`
  - `scripts/combine_layer_features.py`
  - `scripts/train_embedding_downstream.py`
  - `scripts/tune_embedding_downstream.py`
  - `scripts/train_layer_sequence.py`
  - `scripts/train_embedding_fusion.py`

## Decision Layer

- The final decision rule operates on the five existing OOF probability
  branches; it does not add another encoder.
- Blind clusters are recomputed inside each outer validation fold, matching
  hidden-test inference where all test prompts are available at once.
- Prompt-local log-probability centering with alpha 0.75 alone raises OOF from
  0.5545 to 0.5742.
- A soft Sinkhorn-style class-margin balance raises it further. The fixed,
  risk-controlled rule uses:
  - center alpha: 0.50
  - balance strength: 0.625
  - iterations: 5
  - minimum cluster size: 8
- Fixed cluster-balanced OOF is **0.6264**. Nested selection of center alpha,
  strength, and iterations gives 0.6242, with every outer fold above 0.6204.
- This is a soft prior, not Hungarian assignment. A robustness test randomly
  retained 70%-90% of each prompt's rows, creating unequal class counts.
  Soft balancing beat centering-only in 20/20 seeds at every retention rate:
  - keep 90%: +0.0507
  - keep 80%: +0.0437
  - keep 70%: +0.0326
- `scripts/apply_cluster_balance.py` applies the final rule to OOF validation
  or to all hidden-test rows.
- Final saved OOF:
  `features/qwen3_embedding_cluster_balanced_fusion_oof.npy`.
- Rejected downstream experiments:
  - candidate-wise LogisticRegression reranker: about 0.5605
  - candidate reranker after centering: 0.5668
  - hierarchical group/class stacking: 0.5431, 0.5601 after centering
  - neural MLP stacking: 0.5157
  - learned branch gate: 0.5142
  - center every branch before nested weight selection: 0.5668
  - BiLSTM layer sequence: 0.4708
  - causal residual LSTM: 0.4597
  - causal residual + layer-delta LSTM: 0.4367
  - gated DeltaNet layer sequence: 0.4323
  - Set Transformer: 0.5849, 0.6079 after cluster balance; blending with the
    current best only reduced OOF
- `scripts/evaluate_cluster_balance_robustness.py` reproduces the imbalance
  stress test and `scripts/analyze_downstream_errors.py` reports error modes.

## Logistic Stacking

- Row-level LogisticRegression stacking uses five OOF branch probability
  matrices. The rich feature set contains:
  - raw probabilities
  - log probabilities
  - prompt-centered log probabilities
  - per-branch top-class one-hot features
  - per-branch maximum probability and entropy
  - per-branch probability mass for the three model groups
- The rich stack alone scores 0.5583. Its C value is selected inside each outer
  fold from `[0.001, 0.003, 0.01, 0.03, 0.1]`.
- Applying the same prompt cluster balance raises the rich stack to 0.6320.
  Nested C selection plus balance gives 0.6250.
- The cluster-balanced primary fusion and the nested rich stack are
  complementary. A nested blend-weight search selects stack weight 0.45 in all
  five outer folds and reaches **0.6394**.
- Reproduction scripts:
  - `scripts/train_logistic_stack.py`
  - `scripts/train_stack_blend.py`
- Saved probabilities:
  - `features/qwen3_embedding_cluster_balanced_fusion_oof.npy`
  - `features/qwen3_embedding_rich_stack_balanced_oof.npy`
  - `features/qwen3_embedding_stack_blend_oof.npy`

## Sequence And Set Models

- `scripts/train_layer_sequence.py` supports GRU, LSTM, and gated DeltaNet,
  plus causal/bidirectional mode, residual skip, and layer-delta input.
- The seven-layer BiGRU remains strongest at 0.4797. LSTM and DeltaNet do worse
  on this short seven-step layer sequence:
  - BiLSTM: 0.4708
  - residual LSTM: 0.4597
  - residual + layer-delta LSTM: 0.4367
  - gated DeltaNet: 0.4323
- DeltaNet is designed for long sequence memory. Its fixed-size recurrent state
  does not provide an advantage over seven ordered layer summaries.
- Online research notes:
  - Gated DeltaNet-2 and related 2026 papers target long-context language
    modeling and retrieval, not seven-step layer sequences.
  - A 2026 cross-layer routing study found that forwarding lower-layer
    delta-rule write error into the next layer did not improve matched
    baselines.
  - Recent authorship-attribution work supports contrastive author
    representations, stylometric feature augmentation, and set-level context.
  - A Set Transformer over prompt clusters reached 0.5849 and did not improve
    the current blend, likely because only about 150 training prompts exist.
- Sequence-model rejected OOF artifacts:
  - `features/qwen3_embedding_layer_bilstm_oof.npy`
  - `features/qwen3_embedding_layer_res_lstm_oof.npy`
  - `features/qwen3_embedding_layer_res_delta_lstm_oof.npy`
  - `features/qwen3_embedding_layer_deltanet_oof.npy`
  - `features/qwen3_embedding_set_transformer_oof.npy`

## Immediate Next Work

1. Treat the public `alexchilton/name-that-llm-0-696-all-public-what-failed`
   kernel version 28 as the safe baseline. It is mirrored privately at
   `yyyyhhhhwww/name-that-llm-0696-fallback`; commit version 2 passed all 35
   input-mount checks and submission `58397368` scored **0.696** public LB.
2. Use `scripts/evaluate_sparse_prompt_validation.py` for prompt-sparse
   diagnostics. Reapplying center/balance to existing full-group OOF is a
   negative control, not evidence of hidden-test performance.
3. Evaluate new branches without `group_relative`, prompt centering, or
   group balancing. StyleDistance plus absolute style features reaches 0.3397
   on official train and 0.3682 after Open Pool v14 external training.
4. Qwen3-14B QLoRA is the only expensive branch worth continuing. Train fold
   0 with resumable stage checkpoints; retain it only if two-row sparse
   validation reaches at least 0.60 and the blend gate shows complementary
   errors.
5. Submission `58397368` scored 0.696 and the account is around rank 501 of
   6,063 teams. Do not replace it with an unvalidated experiment.

## Submission And Session Status

- The actual best prior public LB is **0.409** (submission `58393311`).
  The full-group OOF 0.6394 does not transfer and is no longer a deployment
  estimate.
- Submission `58397368` uses the public 0.696 fallback and is COMPLETE with
  public score **0.696**.
- The MIT competition reproduction reports grouped CV 0.418 and public LB
  0.389, matching the observed local-to-LB failure mode.
- Kaggle rejected a direct parquet upload because this competition accepts
  submissions from Notebooks only.
- A temporary private Kaggle artifact dataset and inference kernel were
  created during setup, then deleted on 2026-11-16.
- All WeChat/Feishu bridge experiments were removed at the user's request:
  - AnyBot source and data
  - CLI-WeChat-Bridge globals and config
  - Lark bridge globals and local config
  - mini Codex WeChat bridge
  - scheduled startup tasks and bridge processes
  - copied temporary Kaggle credentials
- The repository still has the original Codex, Claude Code, and Pi global
  tools, but no WeChat/Feishu autostart or background bridge.
- The notebook optional environment was restored after removing `lark-oapi`.

### 2026-11-16 Later State

- Public kernel `alexchilton/name-that-llm-0-701-all-public-what-failed`
  reached 0.701 by adding a mass-preserving sibling residual and three
  typographic invariants on top of the 0.696 line.
- New dataset `nakagawaren0805/nllm-glm-logits-train` provides aligned
  160-token GLM likelihoods. Its multiclass GLM-pair OOF is 0.6600 and the
  GLM-internal pairwise OOF is 0.8867 (fixed `C=0.5`: 0.8933).
- `configs/glm_likelihood_model_160.json` and
  `configs/glm_pairwise_model_160.json` are the refreshed deployable models.
  On the local stack blend, reference weight 1.25 gives OOF 0.6664.
- The private experiment kernel
  `yyyyhhhhwww/name-that-llm-0701-glm-pairwise` derives from the 0.701
  notebook, replaces `ll_model.json` inside the V1 payload, inserts the
  pairwise gate between V1 and the sibling residual, and preserves the
  non-GLM 0.701 post-processing.
- Public Gemini frozen DeBERTa features were audited and rejected: their
  benchmark champion logits reached only about 0.27 within-Gemini balanced
  accuracy.
- Qwen3-14B QLoRA stage1 completed 800 steps on Colab and stage2 is active.
  Remote stage2 checkpoints exist through step 100; epoch-1 probabilities
  have not been written yet.

### Test-Like Validation Standard

- Use `validation/sparse2_protocol.json` and
  `scripts/run_sparse2_lockbox.py` for candidate screening.
- Protocol: 5 prompt-grouped outer folds, 50 seeds, 2 sampled rows per
  validation prompt before any prompt-dependent transform, and 500
  prompt-bootstrap replicates.
- Static probability branches are valid only when their features are
  prompt-independent. Prompt-dependent candidates must expose a callable
  receiving `(train_df, valid_df, seed, fold)`.
- Do not use posthoc sampling of complete 18-row OOF as hidden-test evidence.
- Correct pairwise GLM OOF must train on true-GLM rows but predict every
  outer-validation row.
- Strict V1 160-token GLM sparse2 result uses the prompt-independent
  StyleDistance external branch: **0.4047**, versus 0.3717, paired delta
  **+0.0330**, 49/50 seeds, bootstrap interval **[+0.0300, +0.0359]**.
- Independent sparse2 replication: 0.4040 versus 0.3728, paired delta
  **+0.0312**, 50/50 seeds, bootstrap interval **[+0.0284, +0.0340]**.
- The stack-blend sparse2 diagnostic is 0.6447 -> 0.6710, but its parent has
  prompt-dependent full-group operations, so it is not an accepted strict
  baseline.
- Track every public submission in `validation/lb_calibration.csv`; use
  matched sparse2-to-LB deltas rather than claiming absolute OOF equals LB.
- `scripts/evaluate_qwen14_sparse2.py` applies the same two-row gate to
  fold-0 stage2 probabilities and returns explicit new-reader/fallback
  decisions.

## Browser Notes

- Launch Edge directly, for example:
  `& "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" <url>`
- `Start-Process` may be blocked by the current execution policy.
- The Xunlei Edge extension intercepts Kaggle downloads. If it is enabled,
  disable it temporarily from `edge://extensions`, download with Edge, then
  re-enable it.
- The Xunlei Download Support extension has been re-enabled after the initial
  competition-data download.

## Execution Environment Notes

- Repository `.venv`: `D:\namellm\.venv\Scripts\python.exe`.
- Lockfile: `uv.lock`; project dependencies: `pyproject.toml`.
- Python 3.12 and CUDA PyTorch 2.11 with CUDA 12.8 are pinned.
- Installed direct packages include Transformers, scikit-learn, pandas,
  scipy, pyarrow, sentencepiece, accelerate, bitsandbytes, and Kaggle.
- The Kaggle 2.2.5 package omits a runtime dependency on `six`; `six` is
  therefore declared explicitly in `pyproject.toml`.
- In the Codex command sandbox, Python processes may receive write-permission
  errors in `D:\namellm` even though PowerShell can write there. If this
  recurs, write generated artifacts under `D:\tmp`, then copy them into the
  repository with PowerShell.

## WSL And Google Colab CLI

- WSL2 Ubuntu is installed for running the official Google Colab CLI.
- WSL Linux user: `namellm`.
- WSL home: `/home/namellm`.
- Windows workspace from WSL: `/mnt/d/namellm`.
- The current WSL config is `C:\Users\Administrator\.wslconfig`:
  - mirrored networking
  - automatic Windows proxy
  - 4GB memory
  - 4 processors
- `google-colab-cli` is installed in WSL with `keyrings.alt` so OAuth survives
  between commands.
- The CLI is available as `/home/namellm/.local/bin/colab`.
- Google OAuth must use Google Chrome, not Edge.
- The current active Colab session is named `namellm` and uses a T4 GPU.
- Remote paths:
  - code: `/content/namellm`
  - data: `/content/data`
  - models: `/content/models`
- The remote Qwen3-4B-Base weights were downloaded from Hugging Face because
  ModelScope from the US Colab node was much slower.
- The remote Qwen3-4B-Base download is complete (about 7.9 GB); do not
  redownload it unless the Colab runtime is replaced.
- Remote Qwen3-Embedding-4B is at
  `/content/models/Qwen3-Embedding-4B`. Its unquantized extraction is complete;
  it can be redownloaded in about one minute if the runtime is replaced.
- `scripts/train_qlora.py` and `scripts/train_attention_pool_qlora.py` now
  choose FP16 on pre-Ampere GPUs such as T4 and BF16 on Ampere or newer GPUs.
- The scripts use FP16 gradient scaling and do not advance the scheduler when
  an optimizer update is skipped because of overflow.
- A full remote fold-0, one-epoch QLoRA run completed on T4 with balanced
  accuracy 0.2111, compared with 0.1722 for the earlier local run.
- The remote adapter and validation probabilities are archived under
  `artifacts/qlora_colab_fold0_ep1/`.
- Unquantized Qwen3-4B FP16 LoRA also fits T4 with peak allocation 7.65 GB;
  it needs FP32 trainable adapter parameters so GradScaler can unscale them.
- ModernBERT-base is fully unquantized and fast on T4. Completed fold scores
  are 0.2519 and 0.2889 for folds 0 and 1; that experiment was paused after
  the stronger Qwen3-Embedding branch was adopted.
- `colab exec` has a 30-second silence timeout by default. For long commands
  use `--timeout 1800` or run through `colab ssh`.
- Useful commands:

```bash
colab status -s namellm
colab exec -s namellm --timeout 1800
colab ssh -s namellm -- bash -lc "COMMAND"
colab download -s namellm REMOTE_PATH LOCAL_PATH
colab stop -s namellm
```
