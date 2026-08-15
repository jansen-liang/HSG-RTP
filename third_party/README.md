# Third-Party Baselines

Keep upstream baseline implementations isolated in this directory and pin every checkout to a commit. Do not copy external source files into the main evaluation package.

Expected layout:

```text
third_party/
  DELTA/
  GRID/
```

Repository-specific environments and generated assets should remain inside the corresponding directory or outside the repository. HSG-RTP adapters belong under `scripts/` or `evaluation/` and must document whether they reproduce official code or adapt a method from its paper.

Verified upstream source:

- DELTA: `https://github.com/boschresearch/DELTA.git`
- Audited commit: `fd334fe608f33c759d9be83cc7ade5a93d490f25`
- License: AGPL-3.0

The current benchmark adapter does not invoke the upstream DELTA pipeline; its results must be labeled as a paper-based adaptation using the bundled Fast Downward planner until that is corrected.
- GRID: `https://github.com/jackyzengl/GRID.git`
- Audited commit: `167fba3512ebd4fc150d4c91d605b888e5041446`
- License: Apache-2.0

SayPlan is intentionally not vendored because no official implementation or redistribution license has been released. DELTA includes a non-official SayPlan reimplementation, which must be labeled as a paper-based reimplementation if used.

SayCan is released as a self-contained notebook under the Apache-2.0-licensed Google Research repository (`google-research/saycan`, audited path commit `a0080d35561b0a02504bf303edc4ba7f8011b5f8`). The benchmark adapter ports its candidate scoring rule but does not vendor the 139 KB notebook or claim access to the unreleased mobile-robot skill value functions.

GRID is vendored for a documented schema-conversion and retraining experiment. The public release contains training code and a mini-dataset but no checkpoint, full dataset, or standalone inference code. A separate local author-development checkout at `/home/swzz/disk2T/grid/ridsg` contains checkpoints and inference code but no configured remote or local license file. Results from those checkpoints must be labeled as local author-checkpoint adaptations; retrained results using the public source are official-code adaptations.
