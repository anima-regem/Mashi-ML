"""Offline rendering; split assignment is checked before any font is opened."""
import gzip
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont

from mashi_ml.rendering import Renderer, HEIGHT, WIDTH, EM, BASELINE
from mashi_ml.repertoire import inventory
from mashi_ml.acquisition import load_approved_corpus

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.perf_counter()
    lock_bytes = (ROOT / "manifests/corpus.lock.json").read_bytes()
    rows = load_approved_corpus(ROOT)["fonts"]
    groups = {}
    for r in rows:
        if r["lineage_group"] in groups and groups[r["lineage_group"]] != r["split"]:
            raise ValueError("Cross-split derivative group: " + r["lineage_group"])
        groups[r["lineage_group"]] = r["split"]
    repertoire = inventory()
    (ROOT / "manifests/repertoire.json").write_text(json.dumps(repertoire, ensure_ascii=False, indent=2) + "\n")
    instances, masks, available = [], [], []
    coverage = []
    out = ROOT / "data/rendered"
    out.mkdir(exist_ok=True, parents=True)
    with gzip.open(out / "shaping.jsonl.gz", "wt", encoding="utf-8") as shaping:
        for r in rows:
            for name in r["font_filenames"]:
                asset = next(a for a in r["assets"] if a["local_path"].endswith("/" + name))
                path = ROOT / asset["local_path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
                    raise ValueError("Font hash mismatch")
                with TTFont(path) as tt:
                    cmap = set(tt.getBestCmap())
                    variations = [{}]
                    if "fvar" in tt:
                        axis = next((a for a in tt["fvar"].axes if a.axisTag == "wght"), None)
                        if axis:
                            variations = [{"wght": v} for v in sorted({axis.defaultValue, max(axis.minValue, 200), min(axis.maxValue, 700)})]
                    scripts = sorted({s.ScriptTag for tag in ["GSUB", "GPOS"] if tag in tt for s in tt[tag].table.ScriptList.ScriptRecord})
                for axes in variations:
                    instance = {"id": r["family"] + "/" + name + "/" + json.dumps(axes, sort_keys=True), "family": r["family"], "split": r["split"], "lineage_group": r["lineage_group"], "asset_sha256": asset["sha256"], "axes": axes,
                                "is_regular": (axes.get("wght") == 400 if axes else ("Regular" in name or len(r["font_filenames"]) == 1)), "script_tags": scripts}
                    renderer = Renderer(path, axes)
                    stack, valid, counts = [], [], Counter()
                    for label in repertoire:
                        s = label["sequence"]
                        if any(ord(c) not in cmap for c in s if c not in "\u200c\u200d"):
                            mask, record = None, {"status": "cmap-missing", "glyphs": []}
                        else:
                            mask, record = renderer.render(s)
                        counts[record["status"]] += 1
                        stack.append(np.zeros((HEIGHT, WIDTH), bool) if mask is None else mask)
                        valid.append(mask is not None)
                        shaping.write(json.dumps({"instance_id": instance["id"], "sequence": s, **record}, ensure_ascii=False) + "\n")
                    instances.append(instance)
                    masks.append(np.stack(stack))
                    available.append(valid)
                    coverage.append({"instance_id": instance["id"], "family": r["family"], "split": r["split"], "counts": dict(counts)})
                    print(r["family"], axes, dict(counts), flush=True)
    masks = np.stack(masks)
    available = np.asarray(available, bool)
    # Inventory selection only consults training fonts. Holdout absence never changes training targets.
    train = [i for i, r in enumerate(instances) if r["split"] == "train"]
    supported = available[train].all(axis=0)
    np.savez_compressed(out / "corpus.npz", masks=masks, available=available, supported=supported)
    (ROOT / "manifests/instances.json").write_text(json.dumps(instances, indent=2) + "\n")
    summary = {"candidate_sequences": len(repertoire), "train_supported_sequences": int(supported.sum()), "supported_by_tier": dict(Counter(r["tier"] for j, r in enumerate(repertoire) if supported[j])),
               "train_excluded_sequences": [r for j, r in enumerate(repertoire) if not supported[j]], "unique_families": len(rows), "instances": len(instances), "split_families": {s: [r["family"] for r in rows if r["split"] == s] for s in ["train", "validation", "test"]}, "coverage": coverage,
               "manifest_sha256": hashlib.sha256(lock_bytes).hexdigest(), "source_review_sha256": hashlib.sha256((ROOT / "manifests/source-review.json").read_bytes()).hexdigest(), "canvas": {"height": HEIGHT, "width": WIDTH, "em": EM, "baseline_y": BASELINE, "horizontal_alignment": "ink-centered with recorded translation", "vertical_alignment": "fixed font baseline, no per-glyph fit"}, "render_seconds": time.perf_counter() - started,
               "shaping_note": "HarfBuzz ml/Mlym ltr; exact Unicode clusters recorded. mlm2/mlym tags are shaper versions, not orthography labels. Missing cmap, glyph 0, empty and clipped forms are excluded and counted."}
    (ROOT / "reports/corpus.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["unique_families", "instances", "candidate_sequences", "train_supported_sequences", "supported_by_tier", "render_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
