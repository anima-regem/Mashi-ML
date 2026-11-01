import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from mashi_ml.acquisition import digest, validate_replay, verify_local_assets
from mashi_ml.model import Model
from mashi_ml.rendering import HEIGHT, WIDTH

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("acquire", ROOT / "scripts/acquire.py")
acquire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquire)


def encode(obj):
    return json.dumps(obj, sort_keys=True).encode()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = {"schema": 2, "fonts": [{"family": "fixture", "repository": "example/fonts", "commit": "a" * 40,
            "font_filenames": ["font.ttf"], "source_paths": ["font.ttf", "OFL.txt"], "license_filename": "OFL.txt", "split": "train", "lineage_group": "fixture", "lineage_evidence": "test-only fixture"}]}
        self.raw = {"font.ttf": b"test font bytes", "OFL.txt": b"This Font Software is licensed under the SIL Open Font License, Version 1.1.\n"}
        assets = [{"source_path": p, "url": "https://raw.githubusercontent.com/example/fonts/" + "a" * 40 + "/" + p,
                   "local_path": "data/acquisitions/fixture/fixture/" + p, "sha256": digest(b), "bytes": len(b)} for p, b in self.raw.items()]
        self.lock = {"schema": 2, "source_config_schema": 2, "source_config_sha256": digest(encode(self.spec)), "data_root": "data/acquisitions/fixture", "fonts": [{**self.spec["fonts"][0], "assets": assets}]}
        self.review = {"schema": 1, "source_config_schema": 2, "lock_schema": 2, "source_config_sha256": digest(encode(self.spec)), "lock_sha256": digest(encode(self.lock)),
            "decisions": [{"id": "review", "status": "approved-for-local-research", "selected_license": "OFL-1.1", "local_rendering_and_research": True,
                "notice_evidence": {"source_path": "OFL.txt", "sha256": digest(self.raw["OFL.txt"]), "affirmative_grant_excerpt": self.raw["OFL.txt"].decode().strip(), "reviewer": "test reviewer", "grant_review": "affirmative-grant-reviewed"},
                "reserved_font_names": {"status": "not-assessed-for-export", "names": None}}],
            "assets": [{"asset_id": "fixture:" + a["source_path"], "family": "fixture", "repository": "example/fonts", "commit": "a" * 40,
                        "source_path": a["source_path"], "sha256": a["sha256"], "bytes": a["bytes"], "status": "approved-for-local-research", "decision_id": "review"} for a in assets]}

    def bind(self):
        self.review["source_config_sha256"] = digest(encode(self.spec))
        self.review["lock_sha256"] = digest(encode(self.lock))

    def replay(self, fetcher):
        with contextlib.redirect_stdout(io.StringIO()):
            acquire.replay(self.root, encode(self.spec), encode(self.lock), encode(self.review), fetcher)

    def test_exact_reviewed_replay_and_cache(self):
        fetch = Mock(side_effect=lambda url: self.raw[url.rsplit("/", 1)[-1]])
        self.replay(fetch)
        self.assertEqual(fetch.call_count, 2)
        self.replay(Mock(side_effect=AssertionError("cache should suffice")))

    def test_extra_asset_fails_before_network_even_if_config_binding_updated(self):
        self.spec["fonts"][0]["source_paths"].append("new.ttf")
        self.bind()
        fetch = Mock()
        with self.assertRaises(ValueError):
            self.replay(fetch)
        fetch.assert_not_called()

    def test_config_commit_change_fails_before_network(self):
        self.spec["fonts"][0]["commit"] = "b" * 40
        self.lock["source_config_sha256"] = digest(encode(self.spec))
        self.bind()
        fetch = Mock()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.replay(fetch)
        fetch.assert_not_called()

    def test_lock_asset_missing_fails_even_with_rebound_review(self):
        self.lock["fonts"][0]["assets"].pop()
        self.lock["source_config_sha256"] = digest(encode(self.spec))
        self.bind()
        with self.assertRaisesRegex(ValueError, "asset membership"):
            self.replay(Mock())

    def test_missing_review_and_wrong_asset_hash_are_rejected(self):
        original = copy.deepcopy(self.review)
        self.review["assets"].pop()
        with self.assertRaisesRegex(ValueError, "explicit asset review"):
            self.replay(Mock())
        self.review = original
        self.review["assets"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "explicit asset review"):
            self.replay(Mock())

    def test_unbound_config_and_unknown_versions_rejected(self):
        self.spec["fonts"][0]["lineage_evidence"] = "changed"
        with self.assertRaisesRegex(ValueError, "binding"):
            self.replay(Mock())
        self.bind()
        self.lock["schema"] = 999
        self.bind()
        with self.assertRaisesRegex(ValueError, "schema"):
            self.replay(Mock())

    def test_bad_download_never_installs_unverified_bytes(self):
        with self.assertRaisesRegex(ValueError, "Downloaded source hash"):
            self.replay(Mock(return_value=b"changed"))
        self.assertFalse((self.root / self.lock["fonts"][0]["assets"][0]["local_path"]).exists())

    def test_grant_evidence_checked_not_heading(self):
        self.review["decisions"][0]["notice_evidence"]["affirmative_grant_excerpt"] = "a grant absent from the actual notice"
        with self.assertRaisesRegex(ValueError, "grant excerpt absent"):
            self.replay(Mock(side_effect=lambda url: self.raw[url.rsplit("/", 1)[-1]]))

    def test_missing_affirmative_review_is_not_inferred_from_notice(self):
        del self.review["decisions"][0]["notice_evidence"]["grant_review"]
        fetch = Mock()
        with self.assertRaisesRegex(ValueError, "notice evidence"):
            self.replay(fetch)
        fetch.assert_not_called()

    def test_unassessed_rfn_cannot_claim_an_empty_name_list(self):
        self.review["decisions"][0]["reserved_font_names"]["names"] = []
        with self.assertRaisesRegex(ValueError, "RFNs must remain unknown"):
            self.replay(Mock())

    def test_new_heading_only_source_stays_pending_without_guessed_rfn(self):
        notice = b"SIL OPEN FONT LICENSE Version 1.1\nReserved Font Name Fixture\nNo affirmative grant supplied."
        raw = {**self.raw, "OFL.txt": notice}
        output = self.root / "pending.json"
        with contextlib.redirect_stdout(io.StringIO()):
            lock = acquire.stage_pending(self.root, encode(self.spec), output, lambda url: raw[url.rsplit("/", 1)[-1]])
        row = lock["fonts"][0]
        self.assertEqual(row["review_status"], "pending-manual-review")
        self.assertIsNone(row["approved_usage"])
        self.assertNotIn("license_selected", row)
        self.assertEqual(row["reserved_font_names"], {"status": "pending-manual-review", "names": None})
        with self.assertRaises(ValueError):
            validate_replay(encode(self.spec), output.read_bytes(), encode(self.review))

    def test_create_lock_never_overwrites_existing_lock(self):
        path = self.root / "lock.json"
        path.write_bytes(b"original")
        fetch = Mock()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            acquire.stage_pending(self.root, encode(self.spec), path, fetch)
        fetch.assert_not_called()
        self.assertEqual(path.read_bytes(), b"original")


class InferenceProfileTests(unittest.TestCase):
    def model(self):
        rng = np.random.default_rng(7)
        mean = rng.uniform(-0.3, 0.3, (2, HEIGHT, WIDTH)).astype(np.float32)
        basis = rng.uniform(-0.2, 0.2, (1, 2, HEIGHT, WIDTH)).astype(np.float32)
        return Model(["ക", "മ"], mean, basis, mean[None].copy(), ["training-only-fixture"], {"fit_seconds": 1, "inference_config": {"method": "pca", "rank": 1, "ridge": 0.001}})

    def test_pca_does_not_access_templates_by_default(self):
        model = self.model()
        class Forbidden:
            def __getitem__(self, key):
                raise AssertionError("PCA must not read template data")
        model.templates = Forbidden()
        _, diagnostics = model.infer([0], model.mean[:1], rank=1)
        self.assertNotIn("nearest_training_template", diagnostics)
        self.assertFalse(diagnostics["template_diagnostics_included"])

    def test_v2_compatibility_and_template_free_v3_equivalence(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            full, exported = Path(tmp) / "full.npz", Path(tmp) / "inference.npz"
            model.save(full, model.metadata["inference_config"])
            loaded = Model.load(full)
            loaded.export_inference(exported)
            stripped = Model.load(exported)
            expected, _ = loaded.infer([0], model.mean[:1], rank=1)
            actual, info = stripped.infer([0], model.mean[:1], rank=1)
            np.testing.assert_array_equal(expected, actual)
            self.assertIsNone(stripped.templates)
            self.assertFalse(info["capabilities"]["template_baselines"])
            with np.load(exported, allow_pickle=False) as data:
                self.assertNotIn("templates", data.files)
                self.assertNotIn("template_ids", data.files)
                self.assertNotIn("fit_seconds", json.loads(data["metadata"].item()))
            for method in ["nearest-template", "fixed-template", "template-offset"]:
                with self.assertRaisesRegex(ValueError, "lacks template capability"):
                    stripped.infer([0], model.mean[:1], method=method)
            with self.assertRaisesRegex(ValueError, "lacks template capability"):
                stripped.infer([0], model.mean[:1], include_template_diagnostics=True)
            _, diagnostics = loaded.infer([0], model.mean[:1], rank=1, include_template_diagnostics=True)
            self.assertIn("nearest_training_template", diagnostics)

    def test_export_hash_ignores_run_timing_but_not_arrays(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            a = model.export_inference(Path(tmp) / "a.npz")
            model.metadata["fit_seconds"] = 999
            b = model.export_inference(Path(tmp) / "b.npz")
            self.assertEqual(a, b)
            model.mean = model.mean + 0.01
            c = model.export_inference(Path(tmp) / "c.npz")
            self.assertNotEqual(a, c)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                model.export_inference(Path(tmp) / "a.npz")

    def test_contradictory_profile_flags_or_hidden_templates_rejected(self):
        model = self.model()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "good.npz"
            model.export_inference(path)
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            metadata = json.loads(arrays["metadata"].item())
            metadata["capabilities"]["template_baselines"] = True
            arrays["metadata"] = np.array(json.dumps(metadata))
            np.savez(Path(tmp) / "bad.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "capability flags"):
                Model.load(Path(tmp) / "bad.npz")
            arrays["templates"] = model.templates
            np.savez(Path(tmp) / "hidden.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "payload/profile"):
                Model.load(Path(tmp) / "hidden.npz")


if __name__ == "__main__":
    unittest.main()
