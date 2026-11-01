"""Fixed reference/target protocol and family-macro metrics."""
from collections import defaultdict

import numpy as np
from scipy.ndimage import affine_transform


COUNTS = [5, 10, 20]
SEEDS = [1103, 2207, 3301]

# Exact input labels remain distinct, but legacy/atomic chillus must not leak
# an already observed appearance into the missing-target score.
ALIASES = {c + "്\u200d": atomic for c, atomic in zip("ണനരലള", "ൺൻർൽൾ")}


def appearance_group(sequence):
    return ALIASES.get(sequence, sequence)


def missing_targets(repertoire, valid, observed):
    groups = {appearance_group(repertoire[i]["sequence"]) for i in observed}
    return np.asarray([i for i, r in enumerate(repertoire) if valid[i] and appearance_group(r["sequence"]) not in groups], int)


def metric(prediction, target, indices):
    p, t = prediction[indices], target[indices]
    a, b = p >= 0, t >= 0
    intersection = np.logical_and(a, b).sum(axis=(1, 2))
    union = np.logical_or(a, b).sum(axis=(1, 2))
    return {"iou": intersection / np.maximum(union, 1), "sdf_mae": np.abs(p - t).mean(axis=(1, 2)), "empty_prediction": ~a.any(axis=(1, 2))}


def references(repertoire, valid, count, seed):
    candidates = [i for i, r in enumerate(repertoire) if r["tier"] == "basic" and len(r["sequence"]) == 1 and r["sequence"] not in "൦൧൨൩൪൫൬൭൮൯" and valid[i]]
    rng = np.random.default_rng(seed)
    return np.asarray(rng.permutation(candidates)[:count], int)


def corrupt_observations(fields):
    # Deliberately small segmentation/alignment stress. Synthetic only, never called handwriting.
    matrix = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0.025, 1.]])
    return affine_transform(fields, matrix, offset=[0, -0.5, -0.5], order=1, mode="constant", cval=-1, prefilter=False).astype(np.float32)


def evaluate(model, config, fields, instances, available, repertoire, *, counts=COUNTS, seeds=SEEDS, regular_only=False, stress=False, detail=False):
    results, per_glyph = [], []
    for n, (target, instance, valid) in enumerate(zip(fields, instances, available)):
        if regular_only and not instance["is_regular"]:
            continue
        for count in counts:
            for seed in seeds:
                observed = references(repertoire, valid, count, seed)
                target_indices = missing_targets(repertoire, valid, observed)
                if len(observed) != count or not len(target_indices):
                    continue
                obs = target[observed]
                if stress:
                    obs = corrupt_observations(obs)
                pred, diag = model.infer(observed, obs, **config)
                scores = metric(pred, target, target_indices)
                row = {"family": instance["family"], "instance_id": instance["id"], "reference_count": count, "seed": seed, "targets": len(target_indices),
                       "iou": float(scores["iou"].mean()), "sdf_mae": float(scores["sdf_mae"].mean()), "empty_predictions": int(scores["empty_prediction"].sum()), "observed_sequences": [repertoire[j]["sequence"] for j in observed], "diagnostics": diag, "by_tier": {}}
                for tier in sorted({repertoire[j]["tier"] for j in target_indices}):
                    select = [k for k, j in enumerate(target_indices) if repertoire[j]["tier"] == tier]
                    row["by_tier"][tier] = {"targets": len(select), "iou": float(scores["iou"][select].mean()), "sdf_mae": float(scores["sdf_mae"][select].mean())}
                results.append(row)
                if detail and count == 10 and seed == seeds[0]:
                    for k, j in enumerate(target_indices):
                        per_glyph.append({"family": instance["family"], "instance_id": instance["id"], "sequence": repertoire[j]["sequence"], "tier": repertoire[j]["tier"], "iou": float(scores["iou"][k]), "sdf_mae": float(scores["sdf_mae"][k])})
    by_family = defaultdict(list)
    for row in results:
        by_family[row["family"]].append(row)
    family_metrics = {family: {key: float(np.mean([r[key] for r in rows])) for key in ["iou", "sdf_mae"]} for family, rows in by_family.items()}
    macro = {key: float(np.mean([r[key] for r in family_metrics.values()])) for key in ["iou", "sdf_mae"]}
    by_count = {}
    for count in counts:
        family_scores = []
        for family in by_family:
            select = [r["iou"] for r in by_family[family] if r["reference_count"] == count]
            family_scores.append(np.mean(select))
        by_count[str(count)] = float(np.mean(family_scores))
    return {"protocol": "v2-known-appearance-aliases-excluded", "macro": macro, "by_family": family_metrics, "by_reference_count_iou": by_count, "episodes": len(results), "target_predictions": sum(r["targets"] for r in results), "stress": stress, "rows": results, "per_glyph": per_glyph}
