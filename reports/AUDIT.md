# Post-training audit

Audit scope: inspect frozen models/reports and source code; no extra training, model selection or test-score optimization. `scripts/audit.py` reproduces the numerical checks in `audit.json` without modifying model artifacts.

## Comparable evaluation, with a disclosed correction

The reported **0.313727 → 0.386190** comparison evaluates the fixed first inference configuration and validation-selected model against identical fields, supported targets, exclusions, weight instances, observed sequences, counts and random seeds. Each method has 63 episodes / 42,921 missing-target predictions. All episode keys and denominators match exactly across the first model, selected model and all baselines.

The 48 hyperparameter candidates were fixed before opening the first test results. Selection uses validation only, Regular forms of three families, averaged over 5/10/20 references and seeds 1103/2207/3301. The test loader is not called by `campaign.py train`. The first model is a fixed rank-2, ridge=0.001, full-canvas baseline; the selected setting is augmented rank 8, ridge=0.0001, near-ink weighting.

**Important caveat:** the first test results were viewed before correcting an atomic/legacy-chillu appearance-alias scoring issue. Protocol v2 recomputed the same grid and matched evaluation, excluding aliases of observed appearances. The candidates, architecture, source split and trained inventory did not change; the winning configuration is identical in both passes. Therefore the comparison is matched and validation-selected, but the correction is **not a second independent test**. The overly broad frozen-file phrase “no test scores read” is explained in `PROTOCOL_CORRECTION.md`; that immutable file is retained because its hash binds the test report. Test scores were not used to select a different candidate.

| Test family | First | Selected | Nearest training font |
|---|---:|---:|---:|
| Chilanka | 0.255479 | 0.302284 | 0.340971 |
| Noto Sans Malayalam | 0.353597 | 0.446377 | 0.459915 |
| Noto Serif Malayalam | 0.332106 | 0.409909 | 0.420130 |
| Family macro | 0.313727 | 0.386190 | 0.407005 |

All three families improve over the fixed first model and underperform retrieval. SDF MAE regresses from 0.047080 to 0.058666; retrieval scores 0.044177. The overlap gain is not a universal geometric/readability gain.

There are only **two conservative test lineages**: Chilanka and Noto Malayalam. A paired cluster resampling (20,000 draws, seed 20261101) has only three possible outcomes. Its descriptive selected-IoU range is 0.302284–0.428143; the learned-model gain range is +0.046805–+0.085291, and its deficit to retrieval is −0.038687–−0.011880. Do **not** present these as reliable population confidence intervals: two clusters and synthetic font renderings cannot quantify uncertainty over actual writers.

## Baseline provenance and eligibility

Nearest-font retrieval chooses among **15 original training-instance SDF templates**, using mean squared distance over only the supplied reference fields, then copies that training instance's corresponding missing-sequence fields. No target truth enters template selection. The stored templates are bytewise identical between first/selected checkpoints and their IDs exactly match training instances; no held-out font/weight/lineage is a candidate.

The same 5/10/20 reference budgets, exact reference sequences, seeds and target masks apply to every method. For 5/10/20 references, selected IoU is 0.389731/0.384463/0.384376 and nearest-font IoU is 0.416934/0.406868/0.397214. As more references are supplied, the missing-target set changes slightly; these rows do not isolate a causal reference-count effect. The fixed-template baseline always uses the first training instance (Anek Malayalam weight 200); the offset baseline fits its adjustment from references only. All source assets have the selected OFL option and retained notices. These baselines are legitimate comparisons for digital-font appearance, not successful handwriting reconstruction.

## Split, filtering and normalization

All weights/augmentation inherit frozen family splits. Training: 8 families / 6 groups; validation: 3 / 3; test: 3 / 2. AnjaliOldLipi/Karumbi and Rachana/Meera are conservatively grouped; Noto Sans/Serif share a test group. There is no cross-split declared lineage group or source binary. Common designers/shared shaping history still create correlations not eliminated by these group assignments; there is no exhaustive historical genealogy proof.

Training vocabulary filtering uses `available[train].all(axis=0)`, admitting 698 of 711 candidate sequences. No validation/test availability determines that vocabulary. The 13 training exclusions are recorded. Test availability within the trained vocabulary is 697 for Chilanka; Noto Sans weights 200/400/700: 698/698/678; Noto Serif: 698/698/693. Missing/clipped targets are excluded from metric denominators and disclosed in `coverage.json`.

There is **no learned/fitted normalization transform**: canvas 96x64, em=32, baseline y=45, fixed threshold and SDF divisor 6 are constants set before fitting. Each raster, including holdouts, is horizontally ink-centered using its own bounds and records that translation. This is standardization of a known font render, not prediction of the missing glyph's bearings. It assumes perfect canonical scale/baseline and removes horizontal placement information. Only means, PCA axes and family weights are learned, using training fields exclusively. Arbitrary photo crops do not satisfy this preprocessing assumption.

## Integration and visual-asset audit

The 20-suggestion fixture executes the actual `scripts/predict.py` path: hash-bound checkpoint load, observed/confirmed gates, exact-label and normalization checks, PNG→SDF, `Model.infer`, raster generation and provenance manifest. Input sample hashes remain unchanged; outputs are disjoint from observed labels, hash-checked and `model-generated`, initially unconfirmed, with uncalibrated uncertainty. This does **not** exercise the photo/CV engine, real human confirmation, application import, tracing or font assembly. Its input project ID explicitly says `synthetic-integration-fixture-NOT-HANDWRITING`.

Every comparison grid is titled by the source font and captioned as digital font samples, with ground-truth/mean/nearest/first/selected columns. Fixed inventory examples and explicit worst-error grids expose failures. Reports/card consistently disclaim real handwriting evaluation and never assert ownership of upstream glyphs. Source copyright notices and exact asset provenance remain present. “Ground truth” denotes the held-out digital raster only. The Git tree has no personal samples, credentials, font binaries, dataset tensors or checkpoints.

## Licence distinction and publication scope

All acquired font sources offer OFL-1.1; explicit dual-licence sources use their OFL option. The [official OFL text](https://openfontlicense.org/open-font-license-official-text/) governs redistribution of Font Software, retention of notices and reserved names. SIL's [FAQ 1.25–1.26](https://openfontlicense.org/ofl-faq/) specifically addresses AI-generated fonts and distinguishes graphical output. The release restriction on **generated fonts** is documented; checkpoint distribution is a separate unresolved classification/packaging question here.

The checkpoint contains 15 × 698 recoverable training-font SDF templates plus means/bases; it is not merely opaque learned coefficients. No source notice explicitly bans ML training or all model-weight redistribution. We have not established a compliant distribution classification for that mixed geometry payload and do not assign it a blanket MIT licence. Keeping it local is a release decision, not a demonstrated legal prohibition. The distinct-name guardrail is conservative project policy; OFL's specific name restriction concerns declared Reserved Font Names, none found in the acquired notice headers.

Approved publication scope: original MIT engine/acquisition/rendering/training/evaluation code, configs, pinned manifests/hashes/metadata, metrics and audit logs, factual prose, small comparison figures as explanatory graphics, original font notices under their existing terms, and the MIT application-pilot archive with attribution. This scope does not transfer ownership of source type designs. Excluded from Git/publication: checkpoints, font binaries, rendered dataset tensors, raw reusable glyph collections, personal data and secrets. The model card and application contract carry these distinctions forward.
