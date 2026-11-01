"""Readable deterministic examples and explicitly worst-error grids; no cherry picking."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mashi_ml.model import Model
from mashi_ml.evaluation import references, metric, missing_targets
from mashi_ml.rendering import HEIGHT, WIDTH, BASELINE, EM
from campaign import load_split

ROOT = Path(__file__).resolve().parents[1]


def grid(path, instance, target, available, repertoire, selected, first, *, worst=False):
    refs = references(repertoire, available, 10, 1103)
    prediction, _ = selected.infer(refs, target[refs], **selected.metadata["inference_config"])
    initial, _ = first.infer(refs, target[refs], **first.metadata["inference_config"])
    nearest, _ = selected.infer(refs, target[refs], method="nearest-template")
    targets = missing_targets(repertoire, available, refs)
    scores = metric(prediction, target, targets)["iou"]
    if worst:
        chosen = targets[np.argsort(scores)[:12]]
    else:
        chosen = []
        for tier in ["basic", "vowel-context", "virama", "conjunct", "legacy-joiner", "explicit-nonjoiner"]:
            candidates = [j for j in targets if repertoire[j]["tier"] == tier]
            if candidates:
                # Evenly spaced inventory examples, fixed before examining scores.
                chosen += [candidates[0], candidates[len(candidates) // 2]]
        chosen = np.array(list(dict.fromkeys(chosen)), int)
    scale, margin, row_height = 2, 16, 152
    width = 6 * WIDTH * scale + 2 * margin
    image = Image.new("RGB", (width, 210 + len(chosen) * row_height), "#f5f3ee")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    draw.text((margin, 12), instance["family"] + (" | 12 WORST MISSING TARGETS" if worst else " | fixed tier examples"), fill="#17241f", font=font)
    draw.text((margin, 38), "Digital font samples. Generated suggestions; not real handwriting. 10 observed references:", fill="#17241f", font=small)
    for k, j in enumerate(refs):
        tile = Image.fromarray(np.uint8(~(target[j] >= 0)) * 255).convert("RGB")
        image.paste(tile, (margin + k * 100, 60))
    columns = ["Ground truth", "Mean baseline", "Nearest font", "First trained", "Selected model", "Error overlay"]
    for col, label in enumerate(columns):
        draw.text((margin + col * WIDTH * scale, 154), label, fill="#17241f", font=small)
    for row, j in enumerate(chosen):
        y = 192 + row * row_height
        label = repertoire[j]
        codes = " ".join(label["codepoints"])
        draw.text((margin, y - 12), codes + " | " + label["tier"], fill="#17241f", font=small)
        masks = [target[j] >= 0, selected.mean[j] >= 0, nearest[j] >= 0, initial[j] >= 0, prediction[j] >= 0]
        for col, mask in enumerate(masks):
            tile = Image.fromarray(np.uint8(~mask) * 255).convert("RGB").resize((WIDTH * scale, HEIGHT * scale), Image.Resampling.NEAREST)
            image.paste(tile, (margin + col * WIDTH * scale, y + 6))
        gt, pred = masks[0], masks[-1]
        overlay = np.full((HEIGHT, WIDTH, 3), 255, np.uint8)
        overlay[gt & pred] = [28, 46, 38]
        overlay[gt & ~pred] = [37, 110, 213]
        overlay[~gt & pred] = [222, 72, 51]
        image.paste(Image.fromarray(overlay).resize((WIDTH * scale, HEIGHT * scale), Image.Resampling.NEAREST), (margin + 5 * WIDTH * scale, y + 6))
    draw.text((margin, image.height - 16), "Overlay: blue = missing target ink; red = extra generated ink. Canonical em 32px, baseline y=45.", fill="#17241f", font=small)
    image.save(path)
    return chosen


def main():
    selected = Model.load(ROOT / "artifacts/best-validation.npz")
    first = Model.load(ROOT / "artifacts/first-trained.npz")
    fields, instances, available, repertoire = load_split("test")
    figure_records = []
    for i, instance in enumerate(instances):
        if not instance["is_regular"]:
            continue
        for worst in [False, True]:
            name = instance["family"] + ("-worst.png" if worst else "-fixed.png")
            chosen = grid(ROOT / "reports/figures" / name, instance, fields[i], available[i], repertoire, selected, first, worst=worst)
            figure_records.append({"file": "reports/figures/" + name, "selection": "lowest selected IoU with fixed 10 references" if worst else "first and middle missing sequence per tier, deterministic", "instance_id": instance["id"], "sequences": [repertoire[j]["sequence"] for j in chosen]})
    (ROOT / "reports/figures/manifest.json").write_text(json.dumps(figure_records, ensure_ascii=False, indent=2) + "\n")
    # End-to-end synthetic integration fixture, excluded from Git; not a human handwriting sample.
    i = next(i for i, r in enumerate(instances) if r["family"] == "chilanka")
    refs = references(repertoire, available[i], 10, 1103)
    folder = ROOT / "artifacts/demo"
    (folder / "samples").mkdir(parents=True, exist_ok=True)
    observations = []
    for j in refs:
        filename = "samples/" + "-".join(repertoire[j]["codepoints"]).replace("+", "") + ".png"
        path = folder / filename
        Image.fromarray(np.uint8(~(fields[i, j] >= 0)) * 255).save(path)
        observations.append({"id": "synthetic-font-fixture-" + str(j), "label": repertoire[j]["sequence"], "file": filename,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": "observed", "confirmed": True,
                             "normalization": {"width": WIDTH, "height": HEIGHT, "em_pixels": EM, "baseline_y": BASELINE, "ink": "black"}})
    targets = [r["sequence"] for j, r in enumerate(repertoire) if j not in refs and available[i, j]][:20]
    request = {"schema": "mashi-completion-request/v1", "checkpoint_sha256": hashlib.sha256((ROOT / "artifacts/best-validation.npz").read_bytes()).hexdigest(), "project": {"id": "synthetic-integration-fixture-NOT-HANDWRITING", "revision": 1}, "observations": observations, "targets": targets}
    (folder / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    print("Created", len(figure_records), "qualitative grids and synthetic integration request.")


if __name__ == "__main__":
    main()
