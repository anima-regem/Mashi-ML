# Completion engine contract

Application: [anima-regem/Mashi](https://github.com/anima-regem/Mashi). This repository implements missing-appearance inference. The photo/CV engine supplies observed samples; shared shaping, tracing and font assembly remain application responsibilities. No application files were edited during this campaign.

The interface is an **offline, opt-in research adapter**. Model predictions are generated proposals and must remain distinguishable from user-observed handwriting after review, editing and font export. Acceptance may change review status, never historical provenance.

## Input

Run `PYTHONPATH=src python scripts/predict.py --checkpoint artifacts/best-validation.npz --request example/request.json --output example/suggestions`.

```json
{
  "schema": "mashi-completion-request/v1",
  "checkpoint_sha256": "<full SHA256 of locally approved checkpoint>",
  "project": {"id": "project-id", "revision": 4},
  "observations": [{
    "id": "confirmed-region-id", "label": "ക",
    "file": "samples/u0d15.png", "sha256": "<sample SHA256>",
    "provenance": "observed", "confirmed": true,
    "normalization": {
      "width": 96, "height": 64, "em_pixels": 32,
      "baseline_y": 45, "ink": "black"
    }
  }],
  "targets": ["മ", "ക്ക", "കു"]
}
```

Labels are exact Unicode sequences: preserve joiners and combining signs. One approved sample per sequence; unsupported/duplicate labels and observed targets are rejected. Images live inside the request directory and are hash-checked. The engine reads local PNGs and does no network access. Checkpoint loading uses `allow_pickle=False`.

## Alignment is a real dependency

The checkpoint expects a 96x64 raster with 32-pixel em, baseline at y=45, and horizontally centered ink. The photo engine must estimate/confirm baseline and shared scale, retaining the source-to-canonical transform. **An arbitrary crop stretched into the canvas is not equivalent.** The application's earlier 64x64 bounding-box pilot cannot directly use this checkpoint. Do not silently invent em/baseline for a photograph. A crop alignment workflow and real writer evaluation are outstanding application work.

For approved vectors, rasterize with the same canvas/transform, preserving separate width/advance metadata. This model predicts appearance only: its horizontally centered raster does not recover left/right bearings, advance width, kerning or font layout rules.

## Output and review

`suggestions.json` includes request/project revision, checkpoint and sample hashes, source observation IDs, exact labels, raster hashes, and diagnostics. Every prediction has `provenance: "model-generated"`, `confirmed: false`, `not_observed_in_photo: true`, `requires_human_review: true`, and `uncertainty: "uncalibrated"`. A reference residual is a diagnostic, **not a confidence probability**. Empty outputs are flagged. The application should preview source references alongside generated proposals, expose reject/edit/accept actions, and never replace an observed sample.

Check project revision before import. Preserve source hashes and provenance through tracing. Any raster-to-vector conversion must retain its method/resolution and remain a generated artifact. Avoid direct export until a human reviews content and font licence obligations. Normal font assembly must map sequences to appropriate glyph names/GSUB/GPOS; a codepoint-per-component mapping is invalid for Malayalam.

## Licensing and supported scope

Original engine code is MIT. Font licences are per-source OFL notices (some sources offer dual licensing, with OFL selected). The local checkpoint has **no asserted unrestricted redistribution licence**; it is not uploaded in this campaign. Training on OFL fonts does not by itself settle licensing of ML weights or generated font geometry. Review source lineage/reserved names before packaging derivative fonts. Keep generated family names distinct from source font families.

The inventory in `manifests/repertoire.json` contains candidates; the actual trained sequence list is stored in the checkpoint, selected using training coverage only. Unknown sequences require a new model/repertoire. Digital-font evaluation cannot establish the quality of actual handwriting completion. This engine is not an observed-handwriting reconstruction service.

Generated-font assembly must apply the OFL output conditions described in the [licence audit](../reports/AUDIT.md). The unresolved checkpoint release question is separate from those documented font-output conditions.

## Checkpoint profiles and capabilities

The original **format 2** checkpoints remain readable and unchanged; they contain mean/basis arrays and training templates. New local **format 3 / `pca-inference-v1`** exports omit both `templates` and `template_ids`. They retain PCA/mean inference and explicitly disable template baselines and nearest-template diagnostics. Removing templates does **not** settle rights in the learned means/bases; no weights are published.

```sh
PYTHONPATH=src python scripts/export_checkpoint.py \
  --checkpoint artifacts/best-validation.npz \
  --output artifacts/inference-only-v3.npz
```

Export refuses to overwrite any checkpoint. Keep the printed source/export checksum record separately with corpus/licence provenance. The exported payload excludes `fit_seconds` and other run metadata; the original experimental checkpoint still includes them. Artifact checksums verify exact files. Retraining should be checked for numerical equivalence under a stated tolerance; identical hashes are not promised across BLAS/platform/library differences. Exports from identical arrays/configuration have identical payload hashes even when run timings differ.

Request/result schema remains `v1`. The request must approve the **new export's checksum**. Results now include `model_capabilities`; diagnostics carry the same flags. UI actions must use those flags rather than assume retrieval is available. `nearest_training_template` is optional: PCA no longer reads templates by default. An original full model may request it explicitly with `include_template_diagnostics=True`; a template-free model rejects that request or any template baseline with a clear capability error. Existing PCA numeric predictions are unchanged.
