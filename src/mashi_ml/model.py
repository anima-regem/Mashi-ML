"""Original weighted low-rank cross-sequence SDF model (no research-code imports)."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform

from .rendering import HEIGHT, WIDTH


def augmented(fields):
    """One mild, global synthetic style per font; not a sample of any writer."""
    result = []
    for i, x in enumerate(fields):
        shear = 0.035 if i % 2 else -0.035
        transform = np.array([[1., 0., 0.], [0., 1., 0.], [0., -shear, 1.]])
        y = affine_transform(x, transform, offset=[0, 0, shear * 45], order=1, mode="constant", cval=-1, prefilter=False)
        y = np.clip(y + (0.06 if i % 2 else -0.06), -1, 1)
        result.append(y.astype(np.float32))
    return np.stack(result)


class Model:
    def __init__(self, labels, mean, basis, templates, template_ids, metadata):
        self.labels = list(labels)
        self.mean, self.basis, self.templates = mean, basis, templates
        self.template_ids, self.metadata = list(template_ids), dict(metadata)

    @property
    def capabilities(self):
        present = self.templates is not None
        return {"pca_inference": True, "mean_inference": True,
                "template_baselines": present, "nearest_template_diagnostic": present}

    @classmethod
    def fit(cls, labels, fields, families, template_ids, max_rank=12, augment=False):
        started = time.perf_counter()
        fields = np.asarray(fields, np.float32)
        if fields.ndim != 4 or fields.shape[1:] != (len(labels), HEIGHT, WIDTH):
            raise ValueError("Expected fonts x labels x 64 x 96")
        if len(set(families)) < 3:
            raise ValueError("At least three independent training families required")
        original = fields
        weights = np.array([1 / (len(set(families)) * families.count(f)) for f in families], np.float32)
        if augment:
            fields = np.concatenate([fields, augmented(fields)])
            weights = np.concatenate([weights, weights]) / 2
        flat = fields.reshape(len(fields), -1)
        mean = weights @ flat
        centered = (flat - mean) * np.sqrt(weights[:, None])
        gram = centered @ centered.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        order = [i for i in order if eigenvalues[i] > 1e-5][:max_rank]
        # PCA axes scaled to one empirical standard deviation. No heldout family enters fit.
        basis = eigenvectors[:, order].T @ centered
        variance = eigenvalues[order] / max(float(np.trace(gram)), 1e-9)
        shape = (len(labels), HEIGHT, WIDTH)
        return cls(labels, mean.reshape(shape).astype(np.float32), basis.reshape(-1, *shape).astype(np.float32), original,
                   template_ids, {"architecture": "family-weighted joint-sequence SDF PCA with regularized style inference", "training_families": sorted(set(families)), "training_instances": len(original), "fitted_instances_including_synthetic": len(fields), "max_rank": len(basis), "augmentation": augment,
                                  "explained_variance_ratio": variance.tolist(), "fit_seconds": time.perf_counter() - started})

    def infer(self, reference_indices, reference_fields, *, method="pca", rank=4, ridge=0.001, band=False, blend=1.0, include_template_diagnostics=False):
        indices = np.asarray(reference_indices, dtype=int)
        reference_fields = np.asarray(reference_fields, np.float32)
        if len(indices) < 1 or len(set(indices.tolist())) != len(indices):
            raise ValueError("Distinct, labelled observations required")
        if reference_fields.shape != (len(indices), HEIGHT, WIDTH) or not np.isfinite(reference_fields).all():
            raise ValueError("Observation field shape or values invalid")
        if np.any(indices < 0) or np.any(indices >= len(self.labels)):
            raise ValueError("Observation outside model inventory")
        residual = (reference_fields - self.mean[indices]).reshape(-1)
        diagnostics = {"uncertainty": "uncalibrated", "provenance": "model-generated", "confirmed": False,
                       "capabilities": self.capabilities, "template_diagnostics_included": False}
        template_method = method in ["fixed-template", "nearest-template", "template-offset"]
        if (template_method or include_template_diagnostics) and self.templates is None:
            raise ValueError("Checkpoint lacks template capability; use PCA/mean without template diagnostics")
        if method in ["nearest-template", "template-offset"] or include_template_diagnostics:
            distances = ((self.templates[:, indices] - reference_fields) ** 2).mean(axis=(1, 2, 3))
            nearest = int(np.argmin(distances))
            diagnostics["nearest_training_template"] = self.template_ids[nearest]
            diagnostics["template_diagnostics_included"] = True
        if method == "mean":
            return self.mean.copy(), diagnostics
        if method in ["fixed-template", "nearest-template", "template-offset"]:
            template = self.templates[0 if method == "fixed-template" else nearest].copy()
            diagnostics["selected_training_template"] = self.template_ids[0 if method == "fixed-template" else nearest]
            if method == "template-offset":
                near = np.abs(template[indices]) < 0.6
                delta = float(np.median((reference_fields - template[indices])[near]))
                template = np.clip(template + np.clip(delta, -0.25, 0.25), -1, 1)
                diagnostics["sdf_offset"] = delta
            return template, diagnostics
        if method != "pca":
            raise ValueError("Unknown inference method")
        rank = min(rank, len(self.basis))
        basis = self.basis[:rank, indices].reshape(rank, -1)
        # Downweight far-background pixels; coefficients still learned jointly over all sequence identities.
        weights = np.where(np.abs(reference_fields).reshape(-1) < 0.6, 1., 0.05).astype(np.float32) if band else np.ones(len(residual), np.float32)
        norm = float(weights.sum())
        lhs = (basis * weights) @ basis.T / norm
        rhs = basis @ (weights * residual) / norm
        coeff = np.linalg.solve(lhs + ridge * np.eye(rank), rhs)
        coeff = np.clip(coeff, -4, 4)
        prediction = self.mean + blend * np.tensordot(coeff, self.basis[:rank], axes=1)
        diagnostics.update(style_coefficients=coeff.tolist(), reference_sdf_rmse=float(np.sqrt(np.mean((prediction[indices] - reference_fields) ** 2))), coefficient_clipped=bool(np.any(np.abs(coeff) >= 4)))
        return np.clip(prediction, -1, 1).astype(np.float32), diagnostics

    def save(self, path, inference_config=None):
        if self.templates is None:
            raise ValueError("Inference-only models must use export_inference; format 2 requires templates")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "inference_config": inference_config or {"method": "pca", "rank": 2, "ridge": 0.001, "band": False, "blend": 1.0}}
        np.savez_compressed(path, format_version=np.array(2), labels=np.array(self.labels), mean=self.mean, basis=self.basis,
                            templates=self.templates, template_ids=np.array(self.template_ids), metadata=np.array(json.dumps(metadata)))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def export_inference(self, path):
        """Deterministic inference payload; excludes templates and run timing metadata.

        Removing templates does not settle rights in the learned mean/basis.
        Source-checkpoint provenance belongs in the caller's separate export record.
        """
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Inference export requires an explicit .npz filename")
        if path.exists():
            raise ValueError("Refusing to overwrite an existing checkpoint")
        config = self.metadata.get("inference_config", {"method": "pca", "rank": 2, "ridge": 0.001, "band": False, "blend": 1.0})
        if config.get("method", "pca") not in ["pca", "mean"] or config.get("include_template_diagnostics", False):
            raise ValueError("Inference profile requires PCA/mean without template diagnostics")
        metadata = {"profile": "pca-inference-v1", "capabilities": {"pca_inference": True, "mean_inference": True,
                    "template_baselines": False, "nearest_template_diagnostic": False}, "inference_config": config,
                    "licensing": "Local research only; template removal does not resolve mean/basis redistribution rights"}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, format_version=np.array(3), labels=np.array(self.labels), mean=self.mean, basis=self.basis,
                            metadata=np.array(json.dumps(metadata, sort_keys=True)))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            version = data["format_version"].item()
            metadata = json.loads(data["metadata"].item())
            if version == 3:
                if set(data.files) != {"format_version", "labels", "mean", "basis", "metadata"} or metadata.get("profile") != "pca-inference-v1":
                    raise ValueError("Invalid inference-only payload/profile")
                model = cls(data["labels"].tolist(), data["mean"], data["basis"], None, [], metadata)
                if metadata.get("capabilities") != model.capabilities:
                    raise ValueError("Checkpoint capability flags contradict payload")
                return model
            if version != 2:
                raise ValueError("Unsupported checkpoint format")
            return cls(data["labels"].tolist(), data["mean"], data["basis"], data["templates"], data["template_ids"].tolist(), metadata)
