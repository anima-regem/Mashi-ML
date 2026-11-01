# Measured initial campaign — 2026-11-01

The learned model improved missing-appearance overlap on held-out digital fonts, but **nearest-font retrieval still performs better on the test set**. Keep this model as research. No real handwriting or photograph was supplied, and no writer-generalization claim is supported.

## Corpus and leakage controls

We acquired **14 unique Malayalam families**, 19 original font binaries, and rendered **28 weight instances** from official Google Fonts/SMC repositories at pinned commits. There are **711 candidate exact Unicode sequences**, 19,817 valid instance-sequence renders out of 19,908 attempts, and **698 trained sequences**. 13 candidates failed at least one training instance's canonical-canvas/coverage checks and are excluded from the checkpoint. Holdout font support never determines training vocabulary. Every rendered record has shaping glyph IDs/clusters or an explicit exclusion status.

| Split | Families | Conservative lineage groups | Weight instances | Families |
|---|---:|---:|---:|---|
| train | 8 | 6 | 15 | anekmalayalam, gayathri, anjalioldlipi, karumbi, dyuthi, rachana, meera, malini |
| validation | 3 | 3 | 6 | baloochettan2, manjari, uroob |
| test | 3 | 2 | 7 | chilanka, notosansmalayalam, notoserifmalayalam |

All weights and synthetic variants inherit the split. AnjaliOldLipi/Karumbi are conservatively grouped by their common designers; Rachana/Meera are grouped as related serif/sans counterparts. Noto Sans/Serif Malayalam share one test group. Shared shaping-rule history does not alone prove identical outlines; the retained upstream notes document the grouping rationale. No acquired derivative was split across train/validation/test. Different authors/designs sharing engineering can still be correlated: 14 families are not 14 independent writers.

The trained inventory comprises {'basic': 65, 'vowel-context': 428, 'virama': 36, 'conjunct': 159, 'legacy-joiner': 5, 'explicit-nonjoiner': 5}. This is an explicit research repertoire, not the complete Malayalam writing system or a native expert's certification. It includes base letters/digits, consonant-vowel contexts, virama forms, curated conjuncts and exact joiner labels. No one-codepoint-per-component assumption is used. Legacy ZWJ chillus equivalent to an observed atomic chillu are excluded from missing-target scores. The full candidate list, source usage decisions, notices, hashes and excluded sources are in `manifests/` and `licenses/`.

RIT discovery was partly audited but GitLab HTTP429 prevented complete pinned acquisition; **zero RIT binaries are used**. RaghuMalayalamSans/Suruma were excluded from the selected OFL corpus after inspecting their exact GPL notices. Nupuram required a source build and was outside this bounded campaign. No arbitrary web-font scraping or noncommercial research modules were used.

## Rendering and model

HarfBuzz 12.3.2 shapes exact Malayalam Unicode sequences (`Mlym`, `ml`, ltr); FreeType 2.13.2 renders unhinted glyphs at 3x supersampling. Canonical canvas: 96x64, 32-pixel em, y=45 baseline, horizontal ink centering with saved translation. Font scale/baseline are fixed; individual glyphs are not independently stretched to a box. `.notdef`, missing cmap entries, empty or clipped forms are excluded. `mlm2`/`mlym` are shaper-version tags, not orthography styles.

Each mask becomes a clipped signed Euclidean distance field (SDF), positive inside ink, divided by 6 pixels. The original NumPy/SciPy model learns a per-sequence mean and joint cross-sequence style axes using family-weighted PCA. Observations fit shared style coefficients by regularized least squares. Missing targets use those coefficients and their learned target-specific axes. This is genuine fitted statistical learning, with no neural epochs and no pretrained handwriting model.

The first model is rank 2 with full-canvas inference. We fitted clean and mildly augmented conditions and evaluated **48 fixed validation settings**: ranks 2/4/8/12 × ridge 0.0001/0.001/0.01 × full-canvas/near-ink weighting × clean/augmented. Augmentation supplies one ±0.035 shear and ±0.06 normalized SDF offset per training instance; it is synthetic geometry, never real handwriting. Family weights prevent a multiweight family dominating the training mean/covariance.

Selection uses only Regular instances of three validation families, macro-averaged across 5/10/20 observations and three fixed reference permutations. The best clean setting scores **0.4012 IoU**; the selected augmented setting scores **0.4056**, against **0.3954** for nearest-font retrieval on the same selection protocol. The selected configuration is `{'method': 'pca', 'rank': 8, 'ridge': 0.0001, 'band': True, 'blend': 1.0}`. With all validation weights, the model scores 0.4147 and retrieval 0.4186; selection on Regular forms does not guarantee robustness across weights.

## Held-out evaluation

All seven held-out weight instances of Chilanka, Noto Sans Malayalam and Noto Serif Malayalam are evaluated. Each condition has **63 episodes and 42,921 missing-target predictions** across three nested reference budgets and three permutations. Scores average within family before averaging families. Repeated permutations and weights are not independent samples; the test represents only **two conservative lineages**. No writer-level confidence intervals are warranted.

| Method | Test macro IoU ↑ | Normalized SDF MAE ↓ |
|---|---:|---:|
| fixed-template | 0.2183 | 0.0641 |
| mean | 0.1753 | 0.0762 |
| nearest-template | 0.4070 | 0.0442 |
| template-offset | 0.4070 | 0.0441 |
| first-trained | 0.3137 | 0.0471 |
| selected | 0.3862 | 0.0587 |

The learned overlap gain is **0.0725 absolute IoU** over the fixed first learned model. It remains **0.0208 below retrieval**. Its SDF MAE is worse than both the first model and retrieval. Do not describe this as a universal quality improvement or production success. The template-offset baseline did not improve binary overlap.

| Held-out family | First trained | Selected | Nearest-font baseline |
|---|---:|---:|---:|
| chilanka | 0.2555 | 0.3023 | 0.3410 |
| notosansmalayalam | 0.3536 | 0.4464 | 0.4599 |
| notoserifmalayalam | 0.3321 | 0.4099 | 0.4201 |

| Observed reference count | First trained | Selected | Nearest-font baseline |
|---|---:|---:|---:|
| 5 | 0.3139 | 0.3897 | 0.4169 |
| 10 | 0.3134 | 0.3845 | 0.4069 |
| 20 | 0.3139 | 0.3844 | 0.3972 |

The target pool is the supported inventory minus observed appearances, so it varies slightly with reference count. These rows describe completion episodes; they do not isolate a causal effect of reference count. The extra synthetic observation-alignment stress scores **0.3623 IoU**, with clean digital target fields. This is a segmentation stress test, not handwriting validation. Detailed per-tier, per-instance and per-sequence errors are in `evaluations-v2.json.gz`; compact comparisons are in `test-compact.json`.

Protocol v1 had a legacy-chillu alias scoring issue found during review. We preserved its reports, corrected the predetermined appearance aliases, and recomputed the same grid/evaluation. The chosen configuration did not change. This corrected evaluation is **not a second independent test set**; prior v1 test scores had been viewed. No architecture, hyperparameter candidates, font splits or training vocabulary were tuned against those scores. See `PROTOCOL_CORRECTION.md` and `protocol-v1-audit.json.gz`.

## Qualitative review and failure modes

Six deterministic grids include fixed first/middle missing sequences from each repertoire tier and the 12 lowest-overlap selected predictions per test family. Columns show ground truth, mean, nearest font, first trained model, selected model and error overlays. They expose lost ink, broken contours, geometry averaging and conjunct mismatches; the selected model often looks worse than retrieval, especially for Chilanka. Pixel overlap is sensitive to baseline, width and stroke thickness and cannot certify character identity or readability.

* [Chilanka fixed tier examples](figures/chilanka-fixed.png) and [worst targets](figures/chilanka-worst.png)
* [Noto Sans fixed examples](figures/notosansmalayalam-fixed.png) and [worst targets](figures/notosansmalayalam-worst.png)
* [Noto Serif fixed examples](figures/notoserifmalayalam-fixed.png) and [worst targets](figures/notoserifmalayalam-worst.png)

## Resources and reproducibility

Hardware: local Apple M1 Pro, 16 GiB RAM, eight CPU cores; about 25 GiB disk free initially. At most two BLAS/OpenMP threads, no GPU/paid compute, no sleep/security/system-setting changes. The small corpus justified CPU linear algebra instead of a large neural model. The corrected clean fit took 0.449s and augmented fit 4.971s; fitting plus all validation/checkpoint writes took 78.20s, peak process RSS **1.73 GiB**. Rendering took 24.22s. The original scoring pass adds 78.45s; four real PCA fits were executed across the two protocol passes, not artificial epochs. Acquisition, package installation and reporting are additional wall time.

The exact Python 3.11.17 dependency environment is in `requirements.lock`, and hardware/library settings in `environment.json`. Network acquisition is separate from offline rendering, training, evaluation and inference. Checkpoint byte hashes include run timing metadata, so identical learned arrays may serialize to different hashes after a fresh fit. Corpus hashes and all model artifact hashes/sizes are in `manifests/artifacts.json`.

Selected local checkpoint: `artifacts/best-validation.npz`, **61,793,153 bytes**, SHA256 `0165690674d47818bd66055cf66f7b8861bdbb557f964bea56154b33180ea045`. It contains 12 learned axes with rank 8 selected at inference, means and training templates for baseline diagnostics. The first learned checkpoint and preserved application pilot remain local. Binaries/checkpoints/rendered corpus are excluded from ordinary Git history. Model-weight distribution is not presumed licensed merely because font sources use OFL; no checkpoint is published.

The historical 55-form pilot was independently reproduced using its read-only original environment: IoU 0.703976 vs mean 0.690513 and matching checkpoint hash. Those scores use bounding-box normalization, isolated letters, a smaller/easier target inventory and a single holdout; **they are not comparable to this campaign's scores**.

## Integration and next justified work

[Application integration contract](../docs/APP_INTEGRATION.md) separates photo/CV → approved observed samples from missing-appearance inference, followed by shared shaping/font assembly. An end-to-end local synthetic request produced 20 generated suggestions with hashes/provenance; it is not a real handwriting test. The app's earlier 64x64 bounding-box pilot needs a deliberate alignment adapter for this 96x64 fixed-baseline checkpoint. No application workspace was edited.

Stop here because global low-rank averages fragment contours and the independent-family corpus is small; more epochs cannot fix those assumptions. Next: acquire additional fully audited independent families when RIT rate limits clear; split them before experimentation; test a local/outline-aware conditional model against retrieval on a fresh untouched family set. Before claiming handwriting quality, obtain consented real writer pages and human-approved labels/baselines, reserve writers before training, check native Malayalam readability and conjunct behavior, and measure segmentation/alignment errors separately. Keep every generated proposal labelled generated throughout review, tracing and export.

[Post-training audit: matched evaluation, uncertainty and licensing distinctions](AUDIT.md).
