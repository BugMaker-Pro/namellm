# Name That LLM Experiment Log - 2026-11-18

## Preserved Deployment State

- Do not replace public submission `58438557` at public LB **0.702**.
- Keep both Qwen14 six-epoch Kaggle kernels running:
  - `yyyyhhhhwww/name-that-llm-qwen14-e6-lr8e5`
  - `yyyyhhhhwww/name-that-llm-qwen14-e6-lr4e5`
- Final status check at 2026-11-18 02:54: both are still
  `KernelWorkerStatus.RUNNING`.
- Kaggle GPU quota at the same check: 17.83h used, **12.17h remaining**.
- The local RTX 5060 Ti is occupied by the interactive
  hierarchical notebook runs. Do not launch another full local training
  process while that kernel is active.

### ModernBERT Contrastive Run

- `modernbert_contrastive8` completed fold 0.
- Best validation balanced accuracy was **0.4000** at epoch 4.
- Validation balanced accuracy then fell to 0.3963 at epoch 5 and 0.3722 at
  epoch 6.
- Train loss continued from 1.7211 to 0.9067 while validation loss rose from
  2.2211 to 2.7147, so early stopping correctly selected epoch 4.
- The notebook automatically continued into `modernbert_no_supcon8`.

## Related Competition Review

The closest public analogue is the Kaggle competition
`map-charting-student-math-misunderstandings`.

- First-place writeup:
  `https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution`
- The winning architecture is candidate scoring rather than a plain
  paragraph classifier.
- A shared prefix is concatenated with each of 8-12 candidate suffixes.
  FlexAttention isolates candidates while preserving shared prefix attention.
- The last hidden state for each candidate feeds a one-logit head, followed
  by cross entropy over candidates.
- The final solution used Qwen3-32B and GLM-Z1-32B, three seeds each, after
  Qwen3-8B five-fold and multi-seed development.
- Multi-seed ensembles were substantially more stable than the best single
  seed. The final two-family ensemble reached MAP@3 **0.9496**.

### Mapping To This Competition

- Shared text: the passage to attribute.
- Candidate set: the 18 model identities.
- Candidate representation: a learned candidate embedding initialized from
  the candidate model-name tokens.
- Prototype head: normalized passage projection dotted with normalized
  candidate embeddings, followed by a scale and bias.

The first implementation is deliberately lighter than the first-place
FlexAttention architecture. It tests whether candidate scoring is a useful
inductive bias on the 8 GB local GPU before paying the cost of one encoder
pass per candidate.

## Explicit Sentence Statistics

Implementation:

- `src/namellm/sentence_style.py`
- `scripts/evaluate_sentence_features.py`

The representation uses prompt-independent features only:

- explicit sentence count and line structure;
- sentence word/character length distribution;
- short, medium, long and very long sentence fractions;
- first/last sentence length and sentence-position slope;
- sentence-initial articles, pronouns, transitions, conditionals and
  questions;
- punctuation frequency per sentence and per 100 words;
- terminal punctuation, fragment and repeated-start ratios;
- bullet/numbered-line indicators and per-sentence lexical diversity.

Prompt-grouped five-fold OOF results:

| Feature set | Logistic | HistGB | ExtraTrees |
| --- | ---: | ---: | ---: |
| Sentence only | **0.2552** | 0.2208 | 0.2186 |
| Sentence + paragraph style | **0.3197** | 0.2749 | 0.2745 |
| StyleDistance external + sentence | **0.3915** | 0.3749 | 0.3771 |
| Cluster-balanced fusion + sentence | 0.6057 | 0.5872 | **0.6094** |

The strict parent is StyleDistance external, OOF **0.3682**. The
sentence-enhanced LogisticRegression is **0.3915**, a full-OOF gain of
2.33 points.

Strict sparse2 result on the standard 50 seeds:

| Branch | Sparse2 balanced accuracy |
| --- | ---: |
| StyleDistance external | 0.3717 |
| StyleDistance external + sentence | **0.3961** |
| Paired delta | **+0.0244** |
| Seed wins | 42 / 50 |
| Prompt-bootstrap 95% interval | **[+0.0170, +0.0324]** |

This passes all sparse2 acceptance thresholds. It is the first new
prompt-independent branch in this session to do so.

Replacing the logistic sentence model with an RBF SVM raises its standalone
OOF from 0.3915 to 0.3971, but mixing RBF into the TF-IDF sentence branch
gives sparse2 0.4260 versus 0.4266 for the logistic version and wins only
22/50 seeds. Reject the RBF replacement.

The strongest standardized sentence coefficients beyond the parent
probabilities are:

- last sentence length relative to the passage mean;
- per-sentence type-token ratio;
- demonstrative, conditional and preposition sentence starts;
- late-minus-early sentence length.

This indicates that the gain comes from passage organization and closing
sentence behavior, not merely from counting punctuation.

The cluster-balanced branch is prompt-dependent and remains diagnostic only.
Its sentence-enhanced variants underperformed the original 0.6264 OOF.

## Candidate Scoring Prototype

`scripts/train_hierarchical_encoder.py` now supports
`head_type="candidate_cosine"` in addition to the original hierarchical
classifier.

- Candidate embeddings are initialized by mean-pooling the tokenizer
  embeddings for each model name.
- The passage projection and candidate embeddings are L2-normalized.
- Model logits are scaled cosine similarities plus a candidate bias.
- The existing family auxiliary loss and sibling SupCon loss remain
  unchanged.
- The original hierarchical behavior is the default, so existing
  checkpoints and configs remain compatible.
- `configs/hierarchical_modernbert_candidate8.json` and the
  `modernbert_candidate8` notebook option run the fold-0 comparison.

A 2-update local RTX 5060 Ti smoke test completed successfully with 208
training rows, 64 validation rows, and 150.4M trainable parameters.

The interactive notebook completed both hierarchical comparisons:

- SupCon: best epoch 4, balanced accuracy **0.4000**.
- No SupCon: best epoch 4, balanced accuracy **0.4074**.

The paired fold-0 delta is +0.0074, but a 1,000-replicate prompt bootstrap
interval is **[-0.0241, +0.0407]** with 66.9% positive draws. The no-SupCon
result is promising but not statistically decisive from one fold.

Candidate-cosine training completed after the notebook released the GPU:

| Head / objective | Best epoch | Balanced accuracy |
| --- | ---: | ---: |
| Hierarchical CE + family + SupCon | 4 | 0.4000 |
| Hierarchical CE + family, no SupCon | 4 | **0.4074** |
| Candidate cosine + family + SupCon | 4 | 0.3556 |

The candidate-cosine head then fell to 0.3426 at epoch 5 and 0.3167 at epoch
6, with validation loss rising to 2.7562. The lightweight candidate-embedding
head is rejected relative to the free hierarchical classifier.

Reloading the best checkpoint gives 0.3500 raw model accuracy and 0.3463
after the family joint, so the family mixture is not the cause of the loss.

This does not reject the first-place candidate-scoring architecture itself.
It rejects the cheap cosine-table approximation. A future candidate scorer
should use actual candidate-token suffixes with shared-prefix attention, not
only a learned label embedding.

## Sentence-Position TF-IDF

`scripts/evaluate_sentence_segments.py` evaluates first sentence, last
sentence, first/last pairs, body, first-two/last-two edges, and the full
sentence reconstruction with preserved-case character and word TF-IDF.

| View | Balanced accuracy |
| --- | ---: |
| First sentence | 0.1689 |
| Last sentence | 0.1752 |
| First + last | 0.2389 |
| Body | 0.2085 |
| First two + last two | 0.2923 |
| All sentences, C=2 | 0.3163 |
| All sentences, C=3 | **0.3197** |

The C=2 and C=3 TF-IDF scores are close. Direct sparse2 comparison favored
C=3 by only 0.0033 with 30/50 seed wins, so C=2 remains the conservative
choice for the established blend.

A nested log-probability blend of the sentence-statistics branch and all-text
TF-IDF raises OOF from 0.3915 to **0.4189**. Sparse2 rises from 0.3717 to
**0.4266**, a paired delta of **+0.0548**, with 48/50 seed wins and bootstrap
interval **[+0.0465, +0.0629]**.

Within-family balanced accuracy improves for every sibling family:

| Family | Style external | Sentence + TF-IDF |
| --- | ---: | ---: |
| Claude | 0.5298 | **0.5685** |
| GPT | 0.4733 | **0.4760** |
| Gemini | 0.4650 | **0.5067** |
| GLM | 0.5167 | **0.5767** |

This supports using the branch as a family-level residual source even though
it does not add to the current Qwen prompt-dependent stack.

The C=3 variant reaches OOF 0.4237 and sparse2 0.4299, but its direct gain over
C=2 is not stable enough to supersede the C=2 result.

## Strict Equal Stack

A prompt-independent equal log-probability stack was evaluated over:

- compact character/word TF-IDF;
- StyleDistance external + absolute style;
- StyleDistance LDA;
- explicit sentence statistics;
- sentence-position TF-IDF;
- LUAR + absolute style;
- 160-token GLM likelihood.

The full-group OOF is **0.4475**. On sparse2 it reaches **0.4511** versus the
StyleDistance parent at 0.3717, a paired delta of **+0.0794**, with **50/50**
seed wins and bootstrap interval **[+0.0720, +0.0880]**.

The nested learned branch weights score only 0.4327, below equal weighting.
This agrees with the SemEval class-balanced soft-voting finding: the branches
are correlated enough that a low-capacity equal vote is more stable than
re-estimating global weights.

The strict equal stack is not close to the 0.702 public submission and does
not replace it. It is a low-weight candidate member or sibling residual source.

## Layer-Geometry Probe

Inspired by topology-aware authorship attribution, the seven Qwen embedding
layers were reduced to 368 prompt-independent geometry features:

- layer-vector norms;
- pairwise cosine and Euclidean distances;
- adjacent deltas and second deltas;
- delta-direction curvature;
- cosine and centered Gram eigenvalues;
- complete-graph MST length.

Standalone geometry reaches only 0.2137 OOF; geometry plus paragraph style
reaches 0.2956. A nested blend with the sentence/TF-IDF branch raises full OOF
from 0.4189 to 0.4241 and sparse2 from 0.4266 to 0.4295.

The direct geometry increment is only +0.0029, wins 30/50 seeds, and has a
bootstrap interval crossing zero. It is rejected as an independent accepted
reader. The result is retained as a weak diagnostic source for future family
residuals.

## Sentence Residual

`src/namellm/sentence_residual.py` implements the public family
mass-preserving residual rule, but uses the explicit sentence and paragraph
style features:

- Claude, GPT, Gemini and GLM families are refined independently.
- Probability mass inside each family is preserved.
- Low-margin rows receive more style evidence.
- Strength is selected with nested prompt-grouped folds.

Nested sentence residual on the existing stack blend scores 0.6398 versus
0.6394, an increment within noise. Applying the strict equal stack as family
evidence gives 0.6413 at a fixed strength but 0.6376 under nested selection.
The existing cluster-balanced branch improves from 0.6264 to 0.6294 under a
diagnostic nested refinement, but that parent contains prompt-dependent
operations.

Decision: do not promote a sentence residual into the current 0.702 path
without an end-to-end Kaggle ablation.

A closer reproduction of the public family residual was also tested:
per-family character/word TF-IDF models generated 18-column evidence on every
outer-validation row. At the best fixed strength it moves the stack blend from
0.6394 to 0.6402 and leaves the cluster-balanced branch unchanged. This is
within noise and is rejected.

## Frozen Sentence Embeddings

`scripts/extract_sentence_embeddings.py` encodes every sentence independently
with frozen ModernBERT-base and aggregates passage-level mean/std/min/max/
first/last vectors. Full extraction covers 13,086 sentences and saves a
2,699 x 4,608 matrix.

Prompt-grouped OOF:

| Representation | Balanced accuracy |
| --- | ---: |
| Sentence aggregate | 0.2941 |
| Sentence aggregate + paragraph style | 0.3467 |

The aggregate alone is weaker than StyleDistance external, but the two
representations are complementary. A nested log-prob blend improves
StyleDistance external from 0.3682 to **0.3919** OOF. Sparse2 improves from
0.3717 to **0.3933**, a paired delta of **+0.0215**, with 46/50 seed wins and
bootstrap interval **[+0.0159, +0.0267]**.

A nested blend with the sentence-statistics/TF-IDF branch reaches 0.4274 OOF
and 0.4302 sparse2. The direct increment over that branch is +0.0036 with
30/50 seed wins, so the embedding is a valid standalone complement but not a
reliable incremental reader over the sentence/TF-IDF blend.

Adding this branch to the seven-member strict equal stack raises OOF from
0.4475 to **0.4549** and sparse2 from 0.4511 to **0.4561**. The direct
eight-versus-seven comparison averages +0.0050 with 35/50 wins, so the
eight-member version remains exploratory rather than an accepted replacement.

An entropy-weighted vote on the seven strict branches reaches 0.4586 OOF and
0.4612 sparse2 at fixed `gamma=-1`. The paired sparse2 delta is +0.0100, but
only 32/50 seeds improve. Nested gamma selection falls to 0.4552 and 29/50
wins. Reject the entropy weighting as an accepted decision rule despite the
higher mean.

## Open Pool Sibling Specialists

The 0.702 public line leaves most residual error inside model families. The
Open Pool contains enough external passages to train a separate balanced
char+word TF-IDF specialist for Claude, GPT and Gemini. The specialist is fit
only on external rows; it never sees official train labels during fitting. At
application time its centered log-probs are added only to the lab's columns
and the lab's total log-mass is preserved.

We exported the actual `crtproduction/name-that-llm-open-pool` parquet through
a CPU-only Kaggle kernel. The export removes rows flagged for train-prompt
regeneration or near-train prompts, served-model mismatch, proxy/metadata
flags, exact official-text duplicates, and caps each model at 12,000 rows.
The resulting pool has 143,138 usable rows.

Full prompt-grouped OOF on existing parents:

| Parent | Parent OOF | W=0.25 sibling OOF | Delta |
| --- | ---: | ---: | ---: |
| `qwen3_embedding_fusion7_oof` | 0.5545 | **0.5861** | +0.0316 |
| `qwen3_embedding_stack_blend_oof` | 0.6394 | **0.6617** | +0.0223 |

The specialist's standalone within-lab accuracies are 0.6729 for Claude,
0.5333 for GPT and 0.5883 for Gemini. Against the prompt-sparse protocol with
100 seeds:

| Parent | 2 rows/prompt parent | 2 rows/prompt sibling | Paired delta | Seed wins |
| --- | ---: | ---: | ---: | ---: |
| fusion7 | 0.5562 | **0.5881** | +0.0318 | 97/100 |
| stack blend | 0.6436 | **0.6647** | +0.0212 | 89/100 |

The same improvement is present at 4 and 8 rows per prompt. This is the
strongest new reader signal in the current investigation. A private 0.702
kernel adding the specialist between the GLM stage and the final sibling
residual is mirrored locally at
`kaggle_pool_siblings_0702/name-that-llm-0702-pool-siblings.ipynb`. Kernel
version 2 passed the smoke path and was submitted as `58491685`; it was
`PENDING` at the 2026-11-18 status check.

### Qwen3-14B Six-Epoch Follow-Up

Both six-epoch Qwen14 kernels completed:

- `e6_lr8e5`: best plain validation balanced accuracy 0.4611; standalone
  two-row sparse2 0.4615.
- `e6_lr4e5`: best plain validation balanced accuracy 0.4333; standalone
  two-row sparse2 0.4322.

The six-epoch training budget improves the ordinary fold metric, but neither
run reaches the 0.60 standalone sparse2 reader gate. Do not scale this exact
Qwen14 configuration or replace the 0.702 submission with it.

### Incremental Gate Recheck

The 0.60 standalone threshold is appropriate for a new independent reader,
but too strict for a post-processing specialist that only reallocates mass
inside a family. The operative gate should be the incremental effect on the
actual parent:

- `sparse2` paired delta >= +0.004
- seed win rate >= 0.75
- positive direction in at least 3/5 prompt folds
- bootstrap lower bound near or above zero
- no material degradation of the parent fallback

Incremental checks against the family-specialist parent:

- GLM TF-IDF specialist: full-stack OOF 0.6394 -> 0.6401, reject.
- Six top pair-specific specialists: added only +0.0022 on two-row sparse2
  with 55% seed wins, reject.
- Family-specific weight and margin gates: nested selection fell below fixed
  weight 0.25, reject.
- Dense sentence/style family specialist: full OOF +0.0019 and sparse2
  +0.0019 with 57%-59% wins, reject.
- Qwen14 `lr4e5` after family specialists: +0.0163 in the sampled
  diagnostic, but only 62% seed wins; `lr8e5` was weaker. Keep out of the
  current kernel.

The fixed Open Pool family specialists at weight 0.25 remain the best
validated post-processing layer. The next expensive test should be a
family-specific fine-tuned encoder, not more TF-IDF rerankers.

## Gemini Pair Encoder

A ModernBERT-base binary encoder was trained on the filtered Open Pool for
`gemini-3.7-flash` versus `gemini-3.8-flash`. External prompt groups were
split 90/10; official train was never used for fitting or checkpoint
selection.

Training history:

| Epoch | External valid BA | Official pair BA |
| ---: | ---: | ---: |
| 1 | 0.617 | 0.547 |
| 2 | **0.649** | **0.577** |
| 3 | 0.643 | 0.560 |
| 4 | 0.607 | 0.550 |

The best external-validation checkpoint was epoch 2, but it did not transfer:
the official pair score was only 0.577. Applying the encoder inside the
Gemini family did not improve the family stack; on the weaker fusion parent
the best small-weight gain was about +0.002, with no stable sparse2 support.

Decision: reject this external-only encoder. A future encoder experiment must
train with official prompt-grouped folds and use the official held-out
prompts in the objective, rather than selecting on an external-only split.

## Official + External Family Specialists

The strongest new result comes from adding official same-family rows to each
TF-IDF specialist, while keeping prompt-grouped folds strict. For every outer
fold, the specialist is fit on:

- the filtered Open Pool rows for that family, excluding exact official text;
- official rows from the same family in the other four folds.

The held-out prompt fold is never used for fitting. This closes the
external-to-official distribution gap without prompt leakage.

Full prompt-grouped OOF on `qwen3_embedding_stack_blend_oof`:

| Weight | Overall OOF | Claude | GPT | Gemini |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.6394 | 0.7085 | 0.6267 | 0.6867 |
| 0.20 | 0.6743 | 0.7633 | 0.6827 | 0.7050 |
| 0.35 | **0.6798** | **0.7901** | 0.6760 | 0.7050 |
| 0.50 | 0.6747 | 0.7915 | 0.6680 | 0.6900 |

Prompt-sparse two-row validation:

| Branch | Mean BA |
| --- | ---: |
| Base stack | 0.6422 |
| External-only family specialists | 0.6622 |
| Official + external family specialists | **0.6811** |

The direct official-vs-external paired delta is **+0.0189**, with **96/100
seed wins**. The direct official-vs-base delta is **+0.0389**, with 97/100
seed wins. This is the first new branch in this session to clear the
incremental gate by a wide margin.

Private kernel `yyyyhhhhwww/name-that-llm-0702-pool-siblings` version 3 adds
official same-family rows to the runtime specialist and uses weight 0.35.
The smoke path passed with 1,400 Gemini, 1,750 GPT and 1,749 Claude training
rows.
