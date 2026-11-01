import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mashi_ml.model import Model
from mashi_ml.rendering import HEIGHT, WIDTH, sdf
from mashi_ml.repertoire import inventory
from mashi_ml.evaluation import missing_targets

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("predict", ROOT / "scripts/predict.py")
predict = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict)


class EngineTests(unittest.TestCase):
    def test_legacy_alias_of_observed_chillu_is_not_a_missing_target(self):
        rep = [{"sequence": s} for s in ["ൻ", "ന്\u200d", "ക", "ന്\u200c"]]
        self.assertEqual(missing_targets(rep, np.ones(4, bool), [0]).tolist(), [2, 3])
    def test_exact_sequences_and_no_duplicate_inventory(self):
        labels = [r["sequence"] for r in inventory()]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertIn("ക്ക", labels)
        self.assertIn("ന്\u200d", labels)
        self.assertIn("ന്\u200c", labels)
        self.assertNotEqual(labels.index("ന്\u200d"), labels.index("ന്\u200c"))

    def test_train_holdout_lineages_are_disjoint(self):
        groups = {}
        rows = json.loads((ROOT / "configs/sources.json").read_text())["fonts"]
        for row in rows:
            self.assertEqual(groups.setdefault(row["lineage_group"], row["split"]), row["split"])
        self.assertEqual(len({r["family"] for r in rows}), 14)

    def test_style_conditions_an_unobserved_target_and_checkpoint_roundtrip(self):
        # Known low-rank content/style law, with target 2 absent from inference references.
        rng = np.random.default_rng(42)
        content = rng.uniform(-0.2, 0.2, (3, HEIGHT, WIDTH)).astype(np.float32)
        style = rng.uniform(-0.1, 0.1, content.shape).astype(np.float32)
        fields = np.stack([content + a * style for a in [-2, -1, 0, 1, 2]])
        model = Model.fit(["ക", "മ", "ക്ക"], fields, list("abcde"), list("abcde"), max_rank=1)
        expected = content + 0.6 * style
        prediction, _ = model.infer([0, 1], expected[:2], rank=1, ridge=1e-8)
        self.assertLess(float(np.abs(prediction[2] - expected[2]).mean()), 1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            model.save(path)
            restored = Model.load(path)
            np.testing.assert_array_equal(model.basis, restored.basis)

    def test_empty_ink_and_duplicate_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            sdf(np.zeros((HEIGHT, WIDTH), bool))
        a = np.zeros((2, HEIGHT, WIDTH), np.float32)
        model = Model(["ക", "മ"], a, np.ones((1, 2, HEIGHT, WIDTH), np.float32), a[None], ["fixture"], {})
        with self.assertRaises(ValueError):
            model.infer([0, 0], a)

    def test_unconfirmed_request_fails_before_prediction(self):
        from PIL import Image
        import hashlib
        a = np.zeros((2, HEIGHT, WIDTH), np.float32)
        model = Model(["ക", "മ"], a, np.ones((1, 2, HEIGHT, WIDTH), np.float32), a[None], ["fixture"], {})
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            checkpoint = tmp / "model.npz"
            model.save(checkpoint)
            request = {"schema": "mashi-completion-request/v1", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "observations": [{"label": "ക", "provenance": "model-generated", "confirmed": False}], "targets": ["മ"]}
            path = tmp / "request.json"
            path.write_text(json.dumps(request))
            with self.assertRaisesRegex(ValueError, "confirmed observed"):
                predict.predict(checkpoint, path, tmp / "out")


if __name__ == "__main__":
    unittest.main()
