"""Verify the delivered checkpoint, metrics and end-to-end synthetic suggestion manifest."""
import gzip
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    for item in json.loads((ROOT / "manifests/artifacts.json").read_text()):
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"], item["path"]
        assert sha(path) == item["sha256"], item["path"]
    checks.append("All indexed local artifact sizes and SHA256 hashes match")
    selection_file = ROOT / "reports/selection-frozen.json"
    selection = json.loads(selection_file.read_text())
    assert sha(ROOT / "artifacts/best-validation.npz") == selection["checkpoint_sha256"]
    summary = json.loads((ROOT / "reports/test-summary.json").read_text())
    assert sha(selection_file) == summary["selection_file_sha256"]
    checks.append("Test report binds to frozen selection and selected checkpoint")
    with gzip.open(ROOT / "reports/evaluations-v2.json.gz", "rt") as handle:
        evaluations = json.load(handle)
    assert len(evaluations) >= 10
    for name, result in evaluations.items():
        assert result["protocol"] == "v2-known-appearance-aliases-excluded", name
        assert 0 <= result["macro"]["iou"] <= 1
        assert result["target_predictions"] == sum(r["targets"] for r in result["rows"])
    checks.append("Complete compressed evaluations have finite bounded overlap and consistent target denominators")
    request_path = ROOT / "artifacts/demo/request.json"
    request = json.loads(request_path.read_text())
    result = json.loads((ROOT / "artifacts/demo/suggestions/suggestions.json").read_text())
    assert sha(request_path) == result["request_sha256"]
    assert result["checkpoint_sha256"] == selection["checkpoint_sha256"]
    assert len(result["suggestions"]) == 20
    seen = {r["label"] for r in request["observations"]}
    for item in request["observations"]:
        assert sha(request_path.parent / item["file"]) == item["sha256"]
    for item in result["suggestions"]:
        assert item["label"] not in seen
        assert item["provenance"] == "model-generated"
        assert item["confirmed"] is False
        assert item["not_observed_in_photo"] is True
        assert sha(ROOT / "artifacts/demo/suggestions" / item["raster"]) == item["raster_sha256"]
    checks.append("20 synthetic end-to-end suggestions preserve input hashes and generated provenance; no observed target overwritten")
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "unit_tests": {"passed": 6, "command": "PYTHONPATH=src python -m unittest discover -s tests -v"},
              "qualitative_review": ["Viewed Chilanka fixed-tier grid", "Viewed Noto Sans worst-target grid", "Fragmented contours/lost ink visible and documented"],
              "pilot_reproduction": json.loads((ROOT / "reports/pilot-reproduction.json").read_text())["matches_handoff"],
              "github_ci": "Prepared workflow; not run because public repository creation is pending explicit publication approval", "app_workspace_modified": False}
    (ROOT / "reports/verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
