# Name That LLM Research Sources

## Direct Competition Evidence

### Public 0.696 fallback

- URL: https://www.kaggle.com/code/alexchilton/name-that-llm-0-696-all-public-what-failed
- Version used: 28
- License shown by the upstream lineage: Apache-2.0
- Why it matters: complete public-data pipeline with Qwen LoRA artifacts,
  Qwen3-Embedding-8B experts, sparse style, ATLAS and GLM likelihood.
- Decision: keep as the safe fallback. Do not alter its scored code before
  obtaining a leaderboard score.

### Open Pool v14

- URL: https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool
- Rows in the local copy: 226,191.
- Why it matters: public discussion consistently identifies external data as
  the largest transferable gain.
- Risk: the dataset license is not explicit. Use only after provenance and
  competition-rule review.

### Public and MIT competition repositories

- https://github.com/stewari1210/name-that-llm
- https://github.com/pedrohbenetton/kaggle-name-that-llm
- Useful code: prompt-grouped lockbox validation, sparse-prompt simulation,
  capped blind clustering, typographic invariants and configurable TF-IDF.
- Main lesson: group-dependent CV gains transfer poorly. The MIT project
  reports grouped CV 0.418 and public LB 0.389.

## Directly Reusable Models

### StyleDistance

- Paper: https://arxiv.org/abs/2410.12757
- Model: https://huggingface.co/StyleDistance/styledistance
- License: MIT
- Architecture: RoBERTa-base, 768-dimensional style embedding.
- Local result: official-only plus style features OOF 0.3397; Open Pool v14
  external training OOF 0.3682.
- Decision: cheap and content-independent, but only a low-weight ensemble
  candidate.

### LUAR

- Repository: https://github.com/llnl/LUAR
- Model: https://huggingface.co/rrivera1849/LUAR-MUD
- License: Apache-2.0
- Local result: official train plus absolute style features OOF 0.3219.
- Decision: useful representation diversity, not a standalone replacement.

### LLMDet

- Paper: https://arxiv.org/abs/2305.15004
- Code: https://github.com/TrustedLLM/LLMDet
- License: MIT
- Method: model-specific proxy perplexity from salient n-gram probabilities.
- Decision: candidate complementary likelihood feature. Do not add if it
  correlates strongly with existing Qwen and GLM likelihood heads.

### RidgeFT

- Paper: https://arxiv.org/abs/2606.05626
- Code: https://github.com/Vincent-HKUSTGZ/RidgeFT-Lifelong-MGT
- License: MIT
- Method: covariance calibration, fixed random features and class-balanced
  analytic ridge updates.
- Local control: PCA256 plus style and 2,048 random features reached OOF
  0.1019. Reject this configuration.

## Attribution Methods and Benchmarks

### ATLAS

- Paper: https://arxiv.org/abs/2610.19583
- Code: https://github.com/komseok0109/ATLAS
- Benchmark: https://huggingface.co/datasets/ming9999/maabench
- Method: learn one source control embedding and compare token likelihood
  across sources with clipped token-level evidence.
- License status: no clear repository license. Use the paper and public
  competition artifacts; do not copy repository code into a final submission.

### OpenTuringBench

- Paper: https://arxiv.org/abs/2504.11369
- Data: https://huggingface.co/datasets/MLNTeam-Unical/OpenTuringBench
- License: CC BY-NC 4.0.
- Method: OTBDetector contrastive learning for detection and attribution.
- Decision: benchmark and representation-pretraining candidate only after
  competition-rule review.

### M4GT-Bench

- Paper: https://arxiv.org/abs/2402.11175
- Code: https://github.com/mbzuai-nlp/M4GT-Bench
- Contents: multilingual multi-generator detection and attribution baselines.
- Decision: evaluation and diagnostic reference.

### MultiGhostBench

- Paper: https://arxiv.org/abs/2609.02379
- Code: https://github.com/GrecoMT/MultiGhostBench
- Finding: no single attribution method wins across domain, author, and
  language shifts.
- Decision: use its distribution-shift framing to reject brittle
  prompt-group-only gains.

### Authorship Attribution in the Era of LLMs

- Survey: https://arxiv.org/abs/2408.08946
- Paper list: https://github.com/baixianghuang/survey-authorship
- Use: terminology and task taxonomy. Do not conflate binary AI-text
  detection with 18-way model attribution.

## License and Integration Rules

- MIT/Apache-2.0 code and models may be used with attribution.
- CC BY-NC datasets and repositories without an explicit license require
  competition-rule review before final training.
- External datasets must remain publicly and equally accessible under the
  competition rules.
