# Baseline Evaluation Protocol

## Shared benchmark

- Dataset: `pipeline/output/task_split/test_corrected_streaming.jsonl`
- Submitted tasks: 70
- Metric denominator: the 66 tasks whose reference execution is valid
- Decoding: deterministic (`do_sample=False`)
- Output limit: 448 new tokens
- Step limit: `max(10, 2 * reference_action_count)`
- Seed: 42
- Metrics: plan success, execution success, global/local Jaccard, global/local LCS ratio, token count, and model inference time

## Controller profiles

`raw` disables model retries, global replanning, room normalization, task-plan repair, task-coverage completion, and scripted stalled-action repair. Parsing, plan validation, and symbolic execution remain enabled because they define whether a prediction is valid and executable.

`bounded` enables the execution controller and limits semantic global-plan repair to four edits.

`full` enables the complete recovery configuration.

Text-only Qwen and DeepSeek rows intended to measure raw model planning must use `--controller-profile raw`. A protocol-matched controller row should remain in the paper to separate controller effects from representation and model effects.

DeepSeek-R1 distill checkpoints force a `<think>` prefix in their chat template. With the shared 448-token limit, use `--reasoning-mode suppress` to prefill an empty reasoning block and reserve the measured output budget for the JSON plan. This is an inference configuration, not controller repair, and must be reported with the result. Native reasoning results require a larger, separately reported token budget.

## External methods

External methods are evaluated as complete published systems, not as raw language models. Their native search, planner, simulator, feedback, and learned components must be reported explicitly. They must not be presented as if they shared HSG-RTP's global/local model interface. Results use one of the following labels:

- `official reproduction`: unmodified official method code and released weights, evaluated with only a documented dataset adapter.
- `official-code adaptation`: official code is retained, but the benchmark schema, model provider, training data, or output interface is adapted.
- `paper-based reimplementation`: no usable official implementation is available and the method is reconstructed from the paper.
- `paper-reported`: copied from the original paper and never placed in a metric column beside results from this benchmark.

### DELTA

- Official repository: `https://github.com/boschresearch/DELTA.git`
- Audited commit: `fd334fe608f33c759d9be83cc7ade5a93d490f25`
- License: AGPL-3.0
- Environment: Python 3.8.17, PyTorch 2.3, Transformers 4.44.2, PDDLGym, an external PDDLGym planner installation, and the VAL validator. The local integration currently uses `pddlgym==0.0.7` and `pddlgym_planners` commit `060d1dd632df5f101d36493116d5377cc2f48fbd` in the `gra` environment; DELTA's bundled Fast Downward reports version 24.06+.
- Model access: Azure/OpenAI APIs or separately downloaded gated Llama 3.1 weights; DELTA does not release a task-specific checkpoint.
- Interface mismatch: DELTA generates and decomposes PDDL problems and invokes an automated planner. It does not expose HSG-RTP's global-plan/local-action interface.
- Current result label: `paper-based adaptation (Fast Downward planner)`. The current adapter uses the published DELTA decomposition idea and the released Fast Downward planner, but does not invoke the upstream `delta.py` pipeline. It is not an official reproduction. A stronger `official-code adaptation` result requires invoking and documenting the upstream DELTA code path itself.
- Upstream adapter status: `evaluation/delta_upstream.py` converts HSG-RTP scenes to DELTA's native `rooms/items/neighbor/agent` schema and is regression-tested against the upstream scene pruning functions. The runnable policy now uses DELTA's upstream `sg_2_pddl_problem` prompt, validates every generated problem through upstream `planner.query`, applies the upstream decomposition prompt, parses mapped PDDL sub-goals, and plans each sub-goal through the same upstream planner. Domain generation remains fixed to the documented HSG benchmark domain, matching DELTA's supported standalone `problem`/`decompose` experiment modes.
- Runnable upstream evaluator: `scripts/evaluate_external_baseline.py --method delta_upstream` adapts the upstream code to Qwen3-8B and the HSG task schema. It may be labeled `official-code adaptation` only after its smoke and 70-task runs complete and their logs confirm that both model-driven upstream stages and planner validation were exercised.
- Queue order: `scripts/run_delta_upstream_after_grid.sh` waits for the strict GRID summary, runs a three-task smoke evaluation, and only then launches the 70-task upstream DELTA evaluation. Problem generation uses an 8,192-token input budget and 2,048-token output budget because a complete PDDL problem cannot reliably fit the shared 448-token action-generation limit. The result must therefore remain in a separately labeled external-method block.

### SayPlan

- Project page: `https://sayplan.github.io/`
- Official code, license, and checkpoint: not linked or released on the project page or arXiv record as of 2026-08-08.
- Available substitute: `third_party/DELTA/baselines/sayplan.py` explicitly identifies itself as a self-implemented baseline based on the SayPlan appendix.
- Interface mismatch: the method includes semantic graph search, classical path planning, simulator feedback, and iterative replanning rather than a single interchangeable model call.
- Allowed label on this benchmark: `paper-based reimplementation`. The DELTA implementation must not be called official SayPlan code.
- The earlier `sayplan_qwen3_8b_adaptation_final70_20260808` run used `test.jsonl`, whose task-level hash differs from the corrected benchmark, and is excluded from the paper. The corrected run uses `test_corrected_streaming.jsonl`, Qwen3-8B, deterministic decoding, a 4,096/448 input/output budget, and the raw external controller.
- Corrected result: `evaluation_results/sayplan_qwen3_8b_corrected_final70_20260810`, with 19.70% Plan SR, 10.61% Exec SR, and task hash `643322a6d7f3ae1bfcbf88e4e94cdb6230bb33e8996180861fe35b97da039142`.

### SayCan

- Official repository: `https://github.com/google-research/google-research/tree/master/saycan`
- Audited SayCan path commit: `a0080d35561b0a02504bf303edc4ba7f8011b5f8`
- License: Apache-2.0 as part of `google-research`.
- Released artifact: a self-contained tabletop notebook that scores every candidate skill by `exp(language log-probability) * affordance` and repeatedly executes the highest-scoring skill until `done()`.
- Unreleased artifacts: the original mobile-manipulator skill policies, value functions, training data, and task-specific checkpoints are not included in the notebook.
- Adaptation: `evaluation/saycan_baseline.py` retains candidate-wise conditional language scoring and multiplicative affordance filtering. Qwen3-8B replaces the original language-model API, and exact symbolic action preconditions replace the unavailable learned value functions. The skill library uses the benchmark's `goto`, `scan`, `pick`, `place`, `press`, and `wait` actions plus `finish`.
- Evaluation protocol: strict direct skill rollout with no HSG-RTP retry, repair, or global-plan controller. SayCan does not emit a room-level plan, so Plan SR and global sequence metrics are reported as not applicable. The evaluator is `scripts/evaluate_saycan_adaptation.py`.
- Allowed label on this benchmark: `official-code adaptation (notebook; symbolic affordance adapter)`. This label does not claim reproduction of the unreleased mobile-robot value functions.
- Completed result: `evaluation_results/saycan_qwen3_8b_corrected_final70_20260810`, with 42.42% Exec SR, 0.224 local Jaccard, 0.221 local LCS ratio, and the corrected task hash. Delivery Exec SR is 0%, while guidance and tidying reach 60.00% and 66.67%, respectively.

### GRID

- Official repository: `https://github.com/jackyzengl/GRID.git`
- Verified HEAD: `167fba3512ebd4fc150d4c91d605b888e5041446`
- License: Apache-2.0
- Environment: Python 3.8.16, PyTorch/Lightning, PyTorch Geometric, CUDA 11.7-era packages, and a vendored INSTRUCTOR embedding implementation using `hkunlp/instructor-xl`.
- Released artifacts: training and evaluation code plus a separate mini-dataset repository (`jackyzengl/GRID_Dataset`).
- Missing artifacts: the official README still marks checkpoints and standalone inference code as unreleased; the dataset repository marks the full 70-object dataset as unreleased.
- Local author-development repository: `/home/swzz/disk2T/grid/ridsg`, commit `d4b595c7133b8dc69e19ef2029e9f8702edb34b3`. Its history is authored by Zhe Ni (`Leib-Niz`) and predates/extends the public GRID release. It contains datasets, inference code, 363 checkpoints, and approximately 15.1 GiB of checkpoint files, but has no configured remote or local license file.
- Verified local checkpoint: `logs/sg_cos_similarity_scaled/version_0/checkpoints/epoch=499.ckpt`, with the adjacent CLIP-RN50 configuration. It loads successfully in `/home/swzz/anaconda3/grid` when that environment's `lib` directory is placed first in `LD_LIBRARY_PATH`; the model has 2,795,013 parameters.
- Interface mismatch: GRID predicts action-object subtasks from its own scene/robot graph schema and requires task-specific supervised training.
- Action mismatch: the local checkpoint predicts GRID's ten actions (`move`, `pick`, `place_to`, `finish`, and open/close variants) and does not directly represent HSG-RTP's `scan`, `press`, or `wait` actions.
- Completed dataset adapter: `scripts/convert_hsg_rtp_to_grid.py` converts the 280-task training split into 6,951 action-object samples and the 70-task corrected test split into 1,625 samples. It preserves four robot nodes, independent room/object/condition nodes, target node IDs, and all seven adapted actions (`goto`, `pick`, `place`, `press`, `scan`, `wait`, `finish`).
- Active retraining protocol: the local author-development CLIP-RN50 branch is retrained with a seven-class action head after the component-ablation queue. Preprocessing, action encoding, and a `(1, 7)` action / `(1, 87)` object forward pass have passed smoke tests. The queued launcher is `scripts/run_grid_after_ablations.sh`.
- Evaluation protocol: GRID is evaluated as its native direct action-object policy with strict state-manager execution. It must emit `finish` after reaching the task goal. Because GRID does not emit room-level plans, Plan SR and global sequence metrics are reported as not applicable rather than filled from an oracle plan. The evaluator is `scripts/evaluate_grid_author_checkpoint.py`.
- Allowed label on this benchmark: `local author-code adaptation (RN50, retrained)`. It is not an official reproduction and is not called an `official-code adaptation`, because the executable author-development repository has no configured remote or local license file and differs from the public release.

SayPlan remains a paper-based reimplementation, while SayCan is limited to the released official notebook algorithm with a symbolic affordance adapter. Neither is presented as a reproduction of unreleased robot policies. All direct-policy rows keep non-applicable global columns explicit.

## Model identity

Record the exact model repository and parameter count. The locally available DeepSeek checkpoint is `DeepSeek-R1-Distill-Qwen-14B`; it must not be reported as an 8B model. If an 8B row is required, acquire and evaluate an explicitly identified 8B checkpoint separately.

## Trained component ablations

All trained component ablations use the same Qwen3-8B base model, LoRA rank, four-stage data schedule, optimizer settings, routed global/local evaluation, deterministic decoding, and bounded controller as the full system. Only the named component is removed.

Before component training begins, the queue re-evaluates the released full HSG routed checkpoints with the current evaluator and corrected task file. This result is written to `evaluation_results/full_hsg_bounded_final70_20260809` and is the authoritative full-system row for component comparisons.

- `no_hsge`: removes all HSG tokens and freezes the unused graph encoder. It is trained independently from the Qwen3-8B base model; it must not resume from a full HSG checkpoint.
- `no_global_topology`: retains room, floor, macro-zone, agent, and local object tokens, but bypasses and freezes the room-level GAT and its post-GNN projection.
- `no_object_tokens`: retains the global HSG and local room/agent tokens, but removes independent local object tokens and freezes object projection/attention parameters.
- `no_graph_updates_history`: removes completed/pending history and trains on static views. Each task reuses its first global view and the first observed local view for each room. Dataset manifests record source/output SHA-256 hashes.
- `no recovery`: uses the trained full model with retries, replanning, semantic task-plan repair, task-coverage completion, and scripted stalled-action repair disabled. It does not require retraining.

Static-view datasets and manifests are generated by `scripts/build_static_history_ablation_dataset.py`. Component training is launched by `scripts/run_component_ablation_training.sh`; each completed run writes the routed global/local checkpoint paths to `checkpoints.env`.

## Paper result audit

Before any result is copied into the paper, run `scripts/audit_paper_results.py`. The audit requires 70 submitted tasks, the 66-task valid denominator, the declared controller and ablation labels, and task-level equivalence to `test_corrected_streaming.jsonl`. Dataset equivalence is computed from the instruction, task metadata, execution summary, and scene name, so older files that differ only in streaming augmentation metadata remain valid for task-level comparisons. GRID additionally must report `plan_sr: null` with an explicit reason and use the strict direct-action rollout profile.

## Verified raw runs

Both runs below use the 70-task file, 66 valid-task denominator, seed 42, deterministic decoding, 448 new tokens, and the `raw` controller profile.

| Model | Reasoning | Plan SR | Exec SR | Global Jaccard | Global LCS | Local Jaccard | Local LCS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | native | 34.85 | 16.67 | 0.255 | 0.247 | 0.071 | 0.066 |
| DeepSeek-R1-Distill-Qwen-14B | suppressed | 30.30 | 10.61 | 0.285 | 0.264 | 0.062 | 0.055 |

Result directories:

- `evaluation_results/qwen3_8b_raw_final70_20260808`
- `evaluation_results/deepseek_r1_distill_qwen_14b_raw_suppress_final70_20260808`

The DeepSeek run used both GPUs while GPU 1 also hosted unrelated vision services. Its success and sequence metrics are valid, but its inference-time measurement is not protocol-matched and should not be copied into the paper comparison table.
