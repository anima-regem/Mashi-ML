"""Offline missing-appearance suggestions from explicitly approved observed samples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from mashi_ml.model import Model
from mashi_ml.rendering import HEIGHT, WIDTH, EM, BASELINE, sdf


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(checkpoint, request_path, output):
    request = json.loads(request_path.read_text())
    if request.get("schema") != "mashi-completion-request/v1":
        raise ValueError("Unsupported request schema")
    digest = sha256(checkpoint)
    if request.get("checkpoint_sha256") != digest:
        raise ValueError("Checkpoint hash does not match the approved request")
    model = Model.load(checkpoint)
    references, fields, source_records = [], [], []
    seen = set()
    for item in request["observations"]:
        if item.get("confirmed") is not True or item.get("provenance") != "observed":
            raise ValueError("Only user-confirmed observed samples may condition completion")
        label = item["label"]
        if label not in model.labels or label in seen:
            raise ValueError("Unsupported or duplicate exact sequence label")
        seen.add(label)
        normalization = item.get("normalization", {})
        if normalization != {"width": WIDTH, "height": HEIGHT, "em_pixels": EM, "baseline_y": BASELINE, "ink": "black"}:
            raise ValueError("Samples must explicitly declare the checkpoint's canonical normalization")
        path = (request_path.parent / item["file"]).resolve()
        if not path.is_relative_to(request_path.parent.resolve()):
            raise ValueError("Observation file must be within the request directory")
        if path.stat().st_size > 2_000_000 or sha256(path) != item["sha256"]:
            raise ValueError("Observation size or hash mismatch")
        with Image.open(path) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError("Observation canvas must be exactly 96 by 64 pixels")
            mask = np.asarray(image.convert("L")) < 128
        fields.append(sdf(mask))
        references.append(model.labels.index(label))
        source_records.append({k: item[k] for k in ["id", "label", "sha256", "provenance", "confirmed"]})
    if not references:
        raise ValueError("At least one approved supported observation required")
    targets = request["targets"]
    if len(targets) != len(set(targets)) or any(t not in model.labels or t in seen for t in targets):
        raise ValueError("Targets must be unique, missing, supported exact sequences")
    predictions, diagnostics = model.infer(references, np.stack(fields), **model.metadata["inference_config"])
    output.mkdir(parents=True, exist_ok=True)
    suggestions = []
    for label in targets:
        name = "-".join(f"u{ord(c):04x}" for c in label) + "-MODEL-GENERATED.png"
        mask = predictions[model.labels.index(label)] >= 0
        Image.fromarray((~mask).astype(np.uint8) * 255).save(output / name)
        suggestions.append({"label": label, "codepoints": [f"U+{ord(c):04X}" for c in label], "raster": name,
                            "raster_sha256": sha256(output / name), "provenance": "model-generated", "confirmed": False,
                            "not_observed_in_photo": True, "uncertainty": "uncalibrated", "empty_prediction": not bool(mask.any()),
                            "requires_human_review": True, "source_observation_ids": [s["id"] for s in source_records]})
    result = {"schema": "mashi-completion-result/v1", "project": request.get("project"), "checkpoint_sha256": digest,
              "request_sha256": sha256(request_path), "observations": source_records, "diagnostics": diagnostics,
              "normalization": {"width": WIDTH, "height": HEIGHT, "em_pixels": EM, "baseline_y": BASELINE, "ink": "black"},
              "suggestions": suggestions, "status": "generated-research-suggestions-awaiting-review",
              "licensing": "OFL-derived font outputs require OFL compliance. Local checkpoint distribution remains unclassified; see reports/AUDIT.md. Keep source lineage/notices.",
              "limitations": ["No real handwriting evaluation", "No shaping rules or font metrics inferred", "No calibrated confidence probability", "No automatic font export"]}
    (output / "suggestions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = predict(args.checkpoint, args.request, args.output)
    print(json.dumps({"status": result["status"], "suggestions": len(result["suggestions"])}))
