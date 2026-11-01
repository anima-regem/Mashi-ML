"""Bounded validation-only selection. Test evaluation is a separate, frozen step."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from mashi_ml.model import Model
from mashi_ml.rendering import sdf
from mashi_ml.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


def load_split(split):
    instances = json.loads((ROOT / "manifests/instances.json").read_text())
    all_rep = json.loads((ROOT / "manifests/repertoire.json").read_text())
    with np.load(ROOT / "data/rendered/corpus.npz", allow_pickle=False) as data:
        supported = data["supported"]
        selected = [i for i, row in enumerate(instances) if row["split"] == split]
        masks = data["masks"][selected][:, supported]
        valid = data["available"][selected][:, supported]
    fields = np.zeros(masks.shape, np.float32)
    for i, stack in enumerate(masks):
        for j, mask in enumerate(stack):
            if valid[i, j]:
                fields[i, j] = sdf(mask)
            else:
                fields[i, j] = -1
    return fields, [instances[i] for i in selected], valid, [r for j, r in enumerate(all_rep) if supported[j]]


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def train():
    start = time.perf_counter()
    train_fields, train_instances, _, repertoire = load_split("train")
    val_fields, val_instances, val_valid, _ = load_split("validation")
    labels = [r["sequence"] for r in repertoire]
    training_families = [r["family"] for r in train_instances]
    template_ids = [r["id"] for r in train_instances]
    runs, best = [], None
    reportdir = ROOT / "reports"
    for augment in [False, True]:
        fit_start = time.perf_counter()
        model = Model.fit(labels, train_fields, training_families, template_ids, max_rank=12, augment=augment)
        name = "augmented" if augment else "clean"
        model.save(ROOT / f"artifacts/{name}-full.npz")
        if not augment:
            first_config = {"method": "pca", "rank": 2, "ridge": 0.001, "band": False, "blend": 1.0}
            first_hash = model.save(ROOT / "artifacts/first-trained.npz", first_config)
            dump(reportdir / "first-checkpoint.json", {"sha256": first_hash, "config": first_config, "metadata": model.metadata})
            for method in ["fixed-template", "mean", "nearest-template", "template-offset"]:
                result = evaluate(model, {"method": method}, val_fields, val_instances, val_valid, repertoire)
                dump(reportdir / ("validation-" + method + ".json"), result)
                print("BASELINE", method, result["macro"], flush=True)
        # 4 ranks x 3 ridges x 2 observation weightings = 24 settings/model, 48 total.
        for rank in [2, 4, 8, 12]:
            if rank > len(model.basis):
                continue
            for ridge in [0.0001, 0.001, 0.01]:
                for band in [False, True]:
                    config = {"method": "pca", "rank": rank, "ridge": ridge, "band": band, "blend": 1.0}
                    result = evaluate(model, config, val_fields, val_instances, val_valid, repertoire, regular_only=True)
                    record = {"id": f"{name}-r{rank}-l{ridge}-b{int(band)}", "training": name, "config": config, "validation_regular": {k: result[k] for k in ["macro", "by_family", "by_reference_count_iou", "episodes", "target_predictions"]}}
                    runs.append(record)
                    print("RUN", record["id"], result["macro"], flush=True)
                    if best is None or result["macro"]["iou"] > best["validation_regular"]["macro"]["iou"]:
                        best = record
                        dump(reportdir / "selection-in-progress.json", record)
        if best["training"] == name:
            best_hash = model.save(ROOT / "artifacts/best-validation.npz", best["config"])
            dump(reportdir / "selection-in-progress.json", {**best, "checkpoint_sha256": best_hash})
        dump(reportdir / "runs.json", runs)
        print("FIT", name, "seconds", model.metadata["fit_seconds"], "stage_wall", time.perf_counter() - fit_start, flush=True)
        del model
    selected = Model.load(ROOT / "artifacts/best-validation.npz")
    selected_config = selected.metadata["inference_config"]
    complete_validation = evaluate(selected, selected_config, val_fields, val_instances, val_valid, repertoire, detail=True)
    dump(reportdir / "validation-best-all-weights.json", complete_validation)
    dump(reportdir / "validation-synthetic-observation-stress.json", evaluate(selected, selected_config, val_fields, val_instances, val_valid, repertoire, counts=[10], stress=True))
    selection = {"status": "FROZEN BEFORE TEST", "selected": best, "checkpoint_sha256": hashlib.sha256((ROOT / "artifacts/best-validation.npz").read_bytes()).hexdigest(), "selection_metric": "Mean IoU macro over validation families, Regular only; average 5/10/20 references x 3 fixed reference permutations; test scores are not used for selection. A protocol correction audit, if present, records any prior test exposure.",
                 "candidate_count": len(runs), "training_wall_seconds_including_validation": time.perf_counter() - start, "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "model_metadata": selected.metadata,
                 "test_protocol": "Evaluate selected model, fixed first trained model and four baselines once, all held-out test weights. No post-test tuning. Report all families including failures.", "corpus_lock_sha256": hashlib.sha256((ROOT / "manifests/corpus.lock.json").read_bytes()).hexdigest()}
    dump(reportdir / "selection-frozen.json", selection)
    print("FROZEN", json.dumps(selection), flush=True)


def test():
    selection_path = ROOT / "reports/selection-frozen.json"
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    if selection["status"] != "FROZEN BEFORE TEST":
        raise ValueError("Freeze selection first")
    checkpoint = ROOT / "artifacts/best-validation.npz"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    fields, instances, valid, repertoire = load_split("test")
    model = Model.load(checkpoint)
    results = {}
    for method in ["fixed-template", "mean", "nearest-template", "template-offset", "selected"]:
        config = model.metadata["inference_config"] if method == "selected" else {"method": method}
        result = evaluate(model, config, fields, instances, valid, repertoire, detail=True)
        dump(ROOT / ("reports/test-" + method + ".json"), result)
        results[method] = {k: result[k] for k in ["macro", "by_family", "by_reference_count_iou", "episodes", "target_predictions"]}
        print("TEST", method, result["macro"], flush=True)
    del model
    first = Model.load(ROOT / "artifacts/first-trained.npz")
    result = evaluate(first, first.metadata["inference_config"], fields, instances, valid, repertoire, detail=True)
    dump(ROOT / "reports/test-first-trained.json", result)
    results["first-trained"] = {k: result[k] for k in ["macro", "by_family", "by_reference_count_iou", "episodes", "target_predictions"]}
    del first
    model = Model.load(checkpoint)
    dump(ROOT / "reports/test-synthetic-observation-stress.json", evaluate(model, model.metadata["inference_config"], fields, instances, valid, repertoire, counts=[10], stress=True))
    dump(ROOT / "reports/test-summary.json", {"selection_file_sha256": hashlib.sha256(selection_bytes).hexdigest(), "results": results, "note": "Generated digital-font geometry only; not handwriting or photographic reconstruction accuracy. Family count is small; reference permutations are not independent writers."})
    print("TEST COMPLETE", json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["train", "test"])
    args = parser.parse_args()
    train() if args.phase == "train" else test()
