"""Read-only post-training audit: no fit, inference or hyperparameter changes."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / "reports"
    test = json.loads((reports / "test-summary.json").read_text())
    results = test["results"]
    with gzip.open(reports / "evaluations-v2.json.gz", "rt") as handle:
        evaluations = json.load(handle)
    with gzip.open(reports / "protocol-v1-audit.json.gz", "rt") as handle:
        original = json.load(handle)
    selection = json.loads((reports / "selection-frozen.json").read_text())
    new_runs = json.loads((reports / "runs.json").read_text())
    keys = ["instance_id", "reference_count", "seed", "targets", "observed_sequences"]
    episodes = [[row[k] for k in keys] for row in evaluations["test-selected.json"]["rows"]]
    matched = {name: episodes == [[r[k] for k in keys] for r in evaluations[f"test-{name}.json"]["rows"]] for name in ["first-trained", "nearest-template", "fixed-template", "mean", "template-offset"]}
    names = ["chilanka", "notosansmalayalam", "notoserifmalayalam"]
    family = {f: {name: results[name]["by_family"][f]["iou"] for name in ["first-trained", "selected", "nearest-template"]} for f in names}
    # Resample two lineage clusters, preserving all constituent font families.
    # Only 3 distinct bootstrap outcomes exist: this is descriptive, not a trustworthy CI.
    clusters = [["chilanka"], ["notosansmalayalam", "notoserifmalayalam"]]
    rng = np.random.default_rng(20261101)
    draws = {name: [] for name in ["selected", "delta_first", "delta_nearest"]}
    for _ in range(20000):
        fonts = [f for i in rng.integers(0, 2, size=2) for f in clusters[i]]
        selected = float(np.mean([family[f]["selected"] for f in fonts]))
        draws["selected"].append(selected)
        draws["delta_first"].append(selected - float(np.mean([family[f]["first-trained"] for f in fonts])))
        draws["delta_nearest"].append(selected - float(np.mean([family[f]["nearest-template"] for f in fonts])))
    with np.load(ROOT / "artifacts/best-validation.npz", allow_pickle=False) as selected, np.load(ROOT / "artifacts/first-trained.npz", allow_pickle=False) as first:
        instances = json.loads((ROOT / "manifests/instances.json").read_text())
        train_ids = {r["id"] for r in instances if r["split"] == "train"}
        template_audit = {"only_training_instances": set(selected["template_ids"].tolist()) == train_ids,
                          "identical_between_first_and_selected": bool(np.array_equal(selected["templates"], first["templates"])),
                          "shape": list(selected["templates"].shape)}
    output = {"matched_test_episodes": matched, "per_family_iou": family,
              "hyperparameter_grid_unchanged_after_v1_test": [(r["training"], r["config"]) for r in original["runs.json"]] == [(r["training"], r["config"]) for r in new_runs],
              "winner_unchanged_after_scoring_correction": original["selection-frozen.json"]["selected"]["id"] == selection["selected"]["id"],
              "selection_hash_matches_test": hashlib.sha256((reports / "selection-frozen.json").read_bytes()).hexdigest() == test["selection_file_sha256"],
              "template_audit": template_audit,
              "descriptive_two_lineage_bootstrap": {"seed": 20261101, "resamples": 20000, "warning": "Only two conservative lineages; these percentiles are descriptive sensitivity ranges, not reliable population confidence intervals or writer-generalization uncertainty.", "statistics": {key: {"percentiles_2p5_97p5": np.percentile(values, [2.5, 97.5]).tolist(), "distinct_outcomes": np.unique(np.round(values, 10)).tolist()} for key, values in draws.items()}}}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
