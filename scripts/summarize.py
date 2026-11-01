"""Generate auditable report tables/cards from the measured campaign artifacts."""
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def load(name):
    return json.loads((REPORTS / (name + ".json")).read_text())


def macro(rows, key="iou"):
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r[key])
    return sum(sum(v) / len(v) for v in by_family.values()) / len(by_family)


def main():
    corpus = load("corpus")
    selection = load("selection-frozen")
    test = load("test-summary")["results"]
    runs = load("runs")
    instances = json.loads((ROOT / "manifests/instances.json").read_text())
    repertoire = json.loads((ROOT / "manifests/repertoire.json").read_text())
    with np.load(ROOT / "data/rendered/corpus.npz", allow_pickle=False) as arrays:
        supported, available = arrays["supported"], arrays["available"]
    coverage = []
    for i, instance in enumerate(instances):
        coverage.append({"instance_id": instance["id"], "family": instance["family"], "split": instance["split"],
                         "trained_sequences": int(supported.sum()), "available_trained_sequences": int(available[i, supported].sum()),
                         "excluded_trained_sequences": [r["sequence"] for j, r in enumerate(repertoire) if supported[j] and not available[i, j]],
                         "by_tier": {tier: {"eligible": sum(bool(supported[j] and r["tier"] == tier) for j, r in enumerate(repertoire)), "available": sum(bool(supported[j] and available[i, j] and r["tier"] == tier) for j, r in enumerate(repertoire))} for tier in sorted({r["tier"] for r in repertoire})}})
    (REPORTS / "coverage.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")
    regular = {r["id"] for r in instances if r["is_regular"]}
    validation = {}
    for name, filename in [("fixed-template", "validation-fixed-template"), ("mean", "validation-mean"), ("nearest-template", "validation-nearest-template"), ("template-offset", "validation-template-offset"), ("selected", "validation-best-all-weights")]:
        result = load(filename)
        validation[name] = {"regular_macro_iou": macro([r for r in result["rows"] if r["instance_id"] in regular]), "all_weights": result["macro"]}
    clean_best = max((r for r in runs if r["training"] == "clean"), key=lambda r: r["validation_regular"]["macro"]["iou"])
    first_setting = next(r for r in runs if r["id"] == "clean-r2-l0.001-b0")
    compact = {"protocol": "v2-known-appearance-aliases-excluded", "validation": validation, "best_clean": clean_best, "first_setting_validation": first_setting, "test": test,
               "test_synthetic_observation_stress": {k: load("test-synthetic-observation-stress")[k] for k in ["macro", "by_family", "episodes", "target_predictions"]}}
    (REPORTS / "test-compact.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n")
    # Publish complete metrics/logs in a compact inspectable archive, rather than discard per-glyph failures.
    raw = {p.name: json.loads(p.read_text()) for p in sorted(REPORTS.glob("*.json")) if p.name.startswith(("validation-", "test-")) and p.name not in ["test-compact.json", "test-summary.json"]}
    with gzip.GzipFile(filename=str(REPORTS / "evaluations-v2.json.gz"), mode="wb", mtime=0) as handle:
        handle.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode())
    for name in ["training", "test"]:
        shutil.copyfile(REPORTS / (name + ".log"), REPORTS / (name + "-console.txt"))
    with (REPORTS / "runs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["id", "training", "rank", "ridge", "near_ink_weighting", "regular_validation_macro_iou", "regular_validation_sdf_mae"])
        for r in runs:
            c, m = r["config"], r["validation_regular"]["macro"]
            writer.writerow([r["id"], r["training"], c["rank"], c["ridge"], c["band"], m["iou"], m["sdf_mae"]])
    records = []
    for path in sorted((ROOT / "artifacts").glob("*.npz")):
        records.append({"path": "artifacts/" + path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "distribution": "local only; excluded from Git", "licence": "No unrestricted redistribution licence asserted; source notices retained"})
    for rel in ["data/rendered/corpus.npz", "data/rendered/shaping.jsonl.gz", "manifests/corpus.lock.json", "requirements.lock"]:
        path = ROOT / rel
        records.append({"path": rel, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "distribution": "source metadata in Git; rendered corpus local only"})
    (ROOT / "manifests/artifacts.json").write_text(json.dumps(records, indent=2) + "\n")
    checkpoint = next(r for r in records if r["path"] == "artifacts/best-validation.npz")
    table = "\n".join(f"| {name} | {test[name]['macro']['iou']:.4f} | {test[name]['macro']['sdf_mae']:.4f} |" for name in ["fixed-template", "mean", "nearest-template", "template-offset", "first-trained", "selected"])
    family_table = "\n".join(f"| {family} | {test['first-trained']['by_family'][family]['iou']:.4f} | {test['selected']['by_family'][family]['iou']:.4f} | {test['nearest-template']['by_family'][family]['iou']:.4f} |" for family in test["selected"]["by_family"])
    count_table = "\n".join(f"| {k} | {test['first-trained']['by_reference_count_iou'][k]:.4f} | {test['selected']['by_reference_count_iou'][k]:.4f} | {test['nearest-template']['by_reference_count_iou'][k]:.4f} |" for k in ["5", "10", "20"])
    split_table = "\n".join(f"| {split} | {len(corpus['split_families'][split])} | {len({i['lineage_group'] for i in instances if i['split'] == split})} | {sum(i['split'] == split for i in instances)} | {', '.join(corpus['split_families'][split])} |" for split in ["train", "validation", "test"])
    valid_count = sum(r["counts"].get("ok", 0) for r in corpus["coverage"])
    excluded = corpus["candidate_sequences"] - corpus["train_supported_sequences"]
    best_config = selection["selected"]["config"]
    stress = compact["test_synthetic_observation_stress"]["macro"]["iou"]
    report = f"""# Measured initial campaign — 2026-11-01

The learned model improved missing-appearance overlap on held-out digital fonts, but **nearest-font retrieval still performs better on the test set**. Keep this model as research. No real handwriting or photograph was supplied, and no writer-generalization claim is supported.

## Corpus and leakage controls

We acquired **14 unique Malayalam families**, 19 original font binaries, and rendered **28 weight instances** from official Google Fonts/SMC repositories at pinned commits. There are **{corpus['candidate_sequences']} candidate exact Unicode sequences**, {valid_count:,} valid instance-sequence renders out of {28 * corpus['candidate_sequences']:,} attempts, and **{corpus['train_supported_sequences']} trained sequences**. {excluded} candidates failed at least one training instance's canonical-canvas/coverage checks and are excluded from the checkpoint. Holdout font support never determines training vocabulary. Every rendered record has shaping glyph IDs/clusters or an explicit exclusion status.

| Split | Families | Conservative lineage groups | Weight instances | Families |
|---|---:|---:|---:|---|
{split_table}

All weights and synthetic variants inherit the split. AnjaliOldLipi/Karumbi are conservatively grouped by their common designers; Rachana/Meera are grouped as related serif/sans counterparts. Noto Sans/Serif Malayalam share one test group. Shared shaping-rule history does not alone prove identical outlines; the retained upstream notes document the grouping rationale. No acquired derivative was split across train/validation/test. Different authors/designs sharing engineering can still be correlated: 14 families are not 14 independent writers.

The trained inventory comprises {corpus['supported_by_tier']}. This is an explicit research repertoire, not the complete Malayalam writing system or a native expert's certification. It includes base letters/digits, consonant-vowel contexts, virama forms, curated conjuncts and exact joiner labels. No one-codepoint-per-component assumption is used. Legacy ZWJ chillus equivalent to an observed atomic chillu are excluded from missing-target scores. The full candidate list, source usage decisions, notices, hashes and excluded sources are in `manifests/` and `licenses/`.

RIT discovery was partly audited but GitLab HTTP429 prevented complete pinned acquisition; **zero RIT binaries are used**. RaghuMalayalamSans/Suruma were excluded from the selected OFL corpus after inspecting their exact GPL notices. Nupuram required a source build and was outside this bounded campaign. No arbitrary web-font scraping or noncommercial research modules were used.

## Rendering and model

HarfBuzz 12.3.2 shapes exact Malayalam Unicode sequences (`Mlym`, `ml`, ltr); FreeType 2.13.2 renders unhinted glyphs at 3x supersampling. Canonical canvas: 96x64, 32-pixel em, y=45 baseline, horizontal ink centering with saved translation. Font scale/baseline are fixed; individual glyphs are not independently stretched to a box. `.notdef`, missing cmap entries, empty or clipped forms are excluded. `mlm2`/`mlym` are shaper-version tags, not orthography styles.

Each mask becomes a clipped signed Euclidean distance field (SDF), positive inside ink, divided by 6 pixels. The original NumPy/SciPy model learns a per-sequence mean and joint cross-sequence style axes using family-weighted PCA. Observations fit shared style coefficients by regularized least squares. Missing targets use those coefficients and their learned target-specific axes. This is genuine fitted statistical learning, with no neural epochs and no pretrained handwriting model.

The first model is rank 2 with full-canvas inference. We fitted clean and mildly augmented conditions and evaluated **48 fixed validation settings**: ranks 2/4/8/12 × ridge 0.0001/0.001/0.01 × full-canvas/near-ink weighting × clean/augmented. Augmentation supplies one ±0.035 shear and ±0.06 normalized SDF offset per training instance; it is synthetic geometry, never real handwriting. Family weights prevent a multiweight family dominating the training mean/covariance.

Selection uses only Regular instances of three validation families, macro-averaged across 5/10/20 observations and three fixed reference permutations. The best clean setting scores **{clean_best['validation_regular']['macro']['iou']:.4f} IoU**; the selected augmented setting scores **{selection['selected']['validation_regular']['macro']['iou']:.4f}**, against **{validation['nearest-template']['regular_macro_iou']:.4f}** for nearest-font retrieval on the same selection protocol. The selected configuration is `{best_config}`. With all validation weights, the model scores {validation['selected']['all_weights']['iou']:.4f} and retrieval {validation['nearest-template']['all_weights']['iou']:.4f}; selection on Regular forms does not guarantee robustness across weights.

## Held-out evaluation

All seven held-out weight instances of Chilanka, Noto Sans Malayalam and Noto Serif Malayalam are evaluated. Each condition has **{test['selected']['episodes']} episodes and {test['selected']['target_predictions']:,} missing-target predictions** across three nested reference budgets and three permutations. Scores average within family before averaging families. Repeated permutations and weights are not independent samples; the test represents only **two conservative lineages**. No writer-level confidence intervals are warranted.

| Method | Test macro IoU ↑ | Normalized SDF MAE ↓ |
|---|---:|---:|
{table}

The learned overlap gain is **{test['selected']['macro']['iou'] - test['first-trained']['macro']['iou']:.4f} absolute IoU** over the fixed first learned model. It remains **{test['nearest-template']['macro']['iou'] - test['selected']['macro']['iou']:.4f} below retrieval**. Its SDF MAE is worse than both the first model and retrieval. Do not describe this as a universal quality improvement or production success. The template-offset baseline did not improve binary overlap.

| Held-out family | First trained | Selected | Nearest-font baseline |
|---|---:|---:|---:|
{family_table}

| Observed reference count | First trained | Selected | Nearest-font baseline |
|---|---:|---:|---:|
{count_table}

The target pool is the supported inventory minus observed appearances, so it varies slightly with reference count. These rows describe completion episodes; they do not isolate a causal effect of reference count. The extra synthetic observation-alignment stress scores **{stress:.4f} IoU**, with clean digital target fields. This is a segmentation stress test, not handwriting validation. Detailed per-tier, per-instance and per-sequence errors are in `evaluations-v2.json.gz`; compact comparisons are in `test-compact.json`.

Protocol v1 had a legacy-chillu alias scoring issue found during review. We preserved its reports, corrected the predetermined appearance aliases, and recomputed the same grid/evaluation. The chosen configuration did not change. This corrected evaluation is **not a second independent test set**; prior v1 test scores had been viewed. No architecture, hyperparameter candidates, font splits or training vocabulary were tuned against those scores. See `PROTOCOL_CORRECTION.md` and `protocol-v1-audit.json.gz`.

## Qualitative review and failure modes

Six deterministic grids include fixed first/middle missing sequences from each repertoire tier and the 12 lowest-overlap selected predictions per test family. Columns show ground truth, mean, nearest font, first trained model, selected model and error overlays. They expose lost ink, broken contours, geometry averaging and conjunct mismatches; the selected model often looks worse than retrieval, especially for Chilanka. Pixel overlap is sensitive to baseline, width and stroke thickness and cannot certify character identity or readability.

* [Chilanka fixed tier examples](figures/chilanka-fixed.png) and [worst targets](figures/chilanka-worst.png)
* [Noto Sans fixed examples](figures/notosansmalayalam-fixed.png) and [worst targets](figures/notosansmalayalam-worst.png)
* [Noto Serif fixed examples](figures/notoserifmalayalam-fixed.png) and [worst targets](figures/notoserifmalayalam-worst.png)

## Resources and reproducibility

Hardware: local Apple M1 Pro, 16 GiB RAM, eight CPU cores; about 25 GiB disk free initially. At most two BLAS/OpenMP threads, no GPU/paid compute, no sleep/security/system-setting changes. The small corpus justified CPU linear algebra instead of a large neural model. The corrected clean fit took {load('first-checkpoint')['metadata']['fit_seconds']:.3f}s and augmented fit {selection['model_metadata']['fit_seconds']:.3f}s; fitting plus all validation/checkpoint writes took {selection['training_wall_seconds_including_validation']:.2f}s, peak process RSS **{selection['max_rss_bytes'] / 2**30:.2f} GiB**. Rendering took {corpus['render_seconds']:.2f}s. The original scoring pass adds 78.45s; four real PCA fits were executed across the two protocol passes, not artificial epochs. Acquisition, package installation and reporting are additional wall time.

The exact Python 3.11.17 dependency environment is in `requirements.lock`, and hardware/library settings in `environment.json`. Network acquisition is separate from offline rendering, training, evaluation and inference. Checkpoint byte hashes include run timing metadata, so identical learned arrays may serialize to different hashes after a fresh fit. Corpus hashes and all model artifact hashes/sizes are in `manifests/artifacts.json`.

Selected local checkpoint: `artifacts/best-validation.npz`, **{checkpoint['bytes']:,} bytes**, SHA256 `{checkpoint['sha256']}`. It contains 12 learned axes with rank 8 selected at inference, means and training templates for baseline diagnostics. The first learned checkpoint and preserved application pilot remain local. Binaries/checkpoints/rendered corpus are excluded from ordinary Git history. Model-weight distribution is not presumed licensed merely because font sources use OFL; no checkpoint is published.

The historical 55-form pilot was independently reproduced using its read-only original environment: IoU 0.703976 vs mean 0.690513 and matching checkpoint hash. Those scores use bounding-box normalization, isolated letters, a smaller/easier target inventory and a single holdout; **they are not comparable to this campaign's scores**.

## Integration and next justified work

[Application integration contract](../docs/APP_INTEGRATION.md) separates photo/CV → approved observed samples from missing-appearance inference, followed by shared shaping/font assembly. An end-to-end local synthetic request produced 20 generated suggestions with hashes/provenance; it is not a real handwriting test. The app's earlier 64x64 bounding-box pilot needs a deliberate alignment adapter for this 96x64 fixed-baseline checkpoint. No application workspace was edited.

Stop here because global low-rank averages fragment contours and the independent-family corpus is small; more epochs cannot fix those assumptions. Next: acquire additional fully audited independent families when RIT rate limits clear; split them before experimentation; test a local/outline-aware conditional model against retrieval on a fresh untouched family set. Before claiming handwriting quality, obtain consented real writer pages and human-approved labels/baselines, reserve writers before training, check native Malayalam readability and conjunct behavior, and measure segmentation/alignment errors separately. Keep every generated proposal labelled generated throughout review, tracing and export.
"""
    report += "\n[Post-training audit: matched evaluation, uncertainty and licensing distinctions](AUDIT.md).\n"
    (REPORTS / "CAMPAIGN.md").write_text(report)
    card = f"""# Model card: Mashi-ML SDF completion v0.1

**Status: local research; not ready for handwriting/font production.** The validation-selected learned model underperforms nearest-font retrieval on held-out families. No actual handwriting or photo evaluation was performed.

* Architecture: original family-weighted joint-sequence SDF PCA; 12 stored axes, rank {best_config['rank']} selected, ridge {best_config['ridge']}, near-ink observation weighting, coefficients clipped to ±4.
* Training: 8 families / 6 conservative lineages, 15 real weight instances, 30 including mild synthetic style variants. {corpus['train_supported_sequences']} supported exact sequences, 96x64 canvas, em=32, baseline y=45. Trained target identities are known across training families; this is missing appearance for known sequence content, not zero-shot Unicode learning.
* Selection: 48 fixed settings, only validation Regular instances of Baloo Chettan 2, Manjari, Uroob. Selected validation IoU {selection['selected']['validation_regular']['macro']['iou']:.4f}; clean best {clean_best['validation_regular']['macro']['iou']:.4f}.
* Holdout: Chilanka + Noto Sans/Serif Malayalam, 3 families / 2 groups / 7 weights. Macro IoU {test['selected']['macro']['iou']:.4f}; first learned model {test['first-trained']['macro']['iou']:.4f}; nearest-font baseline {test['nearest-template']['macro']['iou']:.4f}. SDF MAE {test['selected']['macro']['sdf_mae']:.4f}, worse than first/retrieval. Three reference budgets, three permutations. Small correlated corpus; no writer-accuracy claim.
* Intended use: inspect local generated raster proposals from user-confirmed, exactly labelled, canonically aligned observed samples, with human review and generated provenance preserved.
* Unsupported: arbitrary unaligned crops; complete Malayalam repertoire; unseen target content; recovery of observed handwriting; automated export; font metrics/spacing/GSUB/GPOS; calibrated confidence; guaranteed readable conjuncts.
* Failure modes: contours fragment, missing strokes, averaged geometry, template-like bias, clipping limits, limited unusual-style generalization; low-resolution overlap does not establish readability.
* Data/rights: official Google Fonts/SMC commits and exact OFL notices retained. No personal handwriting/photos. No restricted research implementation. Original code MIT; font/dataset/model/output licensing remain separate. No unrestricted checkpoint distribution licence is asserted, and weights are not published.
* Provenance: `{checkpoint['sha256']}`; local `artifacts/best-validation.npz`, {checkpoint['bytes']:,} bytes. See `manifests/artifacts.json` and [full campaign](reports/CAMPAIGN.md).
* Evaluation audit: known legacy-chillu aliases excluded from missing-target scores in protocol v2. Initial v1 reports are retained; v2 recomputation is not a new independent test set.

Every output must remain `provenance: model-generated`, `confirmed: false` initially, `not_observed_in_photo: true`, and `uncertainty: uncalibrated`. Human acceptance does not turn generated geometry into observed source handwriting. [Integration contract](docs/APP_INTEGRATION.md).
"""
    card += "\n[Audit and current font-output/checkpoint licensing distinction](reports/AUDIT.md).\n"
    (ROOT / "MODEL_CARD.md").write_text(card)
    readme = f"""# Mashi-ML

Local missing-Malayalam-sequence appearance research for [Mashi, the open-source handwriting-to-font studio](https://github.com/anima-regem/Mashi). The photo/CV engine supplies confirmed observed samples; this engine proposes missing appearances. Shared font shaping/assembly stays separate.

**Actual Mac training completed:** whole corpus **14 licensed families / 28 weight instances**, 711 candidate / 698 trained exact sequences. Training used **8 families / 6 lineages / 15 real instances** (30 with augmentation); validation **3 / 3 / 6**; test **3 families / 2 lineages / 7 instances**. Clean and augmented PCA fits were compared across 48 validation settings. Held-out IoU improved **{test['first-trained']['macro']['iou']:.4f} → {test['selected']['macro']['iou']:.4f}**, while nearest-font retrieval scored **{test['nearest-template']['macro']['iou']:.4f}**. The learned model remains a research result. No real handwriting generalization is claimed.

* [Measured campaign and failure cases](reports/CAMPAIGN.md)
* [Model card](MODEL_CARD.md)
* [Application integration contract](docs/APP_INTEGRATION.md)
* [Pinned corpus/legal manifest](manifests/corpus.lock.json) and [separate licensing layers](licenses/README.md)
* [48-run metrics](reports/runs.csv), [test comparisons](reports/test-compact.json), [full compressed metrics](reports/evaluations-v2.json.gz), [console log](reports/training-console.txt)
* [Local artifact hashes](manifests/artifacts.json), [environment](reports/environment.json), [protocol correction audit](reports/PROTOCOL_CORRECTION.md)

## Reproduce on Python 3.11

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
.venv/bin/python scripts/acquire.py
.venv/bin/python scripts/render_corpus.py
.venv/bin/python scripts/campaign.py train
.venv/bin/python scripts/campaign.py test
PYTHONPATH=src:scripts .venv/bin/python scripts/qualitative.py
.venv/bin/python scripts/predict.py --checkpoint artifacts/best-validation.npz --request artifacts/demo/request.json --output artifacts/demo/suggestions
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/summarize.py
```

Only acquisition needs network access. Training/evaluation/inference run offline. The original source lock is immutable: replay requires the exact config/lock hashes and explicit per-asset decisions in `manifests/source-review.json`. New `--create-lock --output-lock <new-file>` acquisition is PENDING until reviewed, never automatic licence/RFN approval. See [acquisition review](docs/ACQUISITION_REVIEW.md). Expect below 2 GiB peak training RSS in this campaign and under 1 GiB task artifacts including duplicate checkpoint conditions. Keep at least several GiB free for reproduction. No paid services, credentials or personal files are needed.

The local selected checkpoint is **{checkpoint['bytes']:,} bytes**, SHA256 `{checkpoint['sha256']}`. Font binaries, raster corpus and weights stay outside Git; no weight redistribution licence is presumed from OFL. Reproduction generates the checkpoint locally. Original source code is MIT; the complete font notices and model licensing caveats remain separate.

The integration demo uses rendered-font fixtures, not a person's handwriting. No app files were modified; the 64x64 pilot requires an explicit scale/baseline adapter for this 96x64 checkpoint. All predictions are uncertain **generated** proposals, never recovered/observed handwriting. See the model card before use.
"""
    readme += "\n[Post-training audit, uncertainty and publication scope](reports/AUDIT.md).\n"
    (ROOT / "README.md").write_text(readme.rstrip() + "\n")
    print(json.dumps({"checkpoint": checkpoint, "selected_test": test["selected"]["macro"], "nearest_test": test["nearest-template"]["macro"], "valid_renders": valid_count}, indent=2))


if __name__ == "__main__":
    main()
