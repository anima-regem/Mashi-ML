"""Fail-closed source identity and explicit review checks; raw text is not approval."""
import hashlib
import json
import re
from pathlib import PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique(rows, key, description):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError("Duplicate " + description + ": " + value)
        result[value] = row
    return result


def validate_spec(spec):
    if type(spec.get("schema", 1)) is not int or spec.get("schema", 1) not in [1, 2]:
        raise ValueError("Unsupported source configuration schema")
    rows = unique(spec["fonts"], "family", "family")
    for family, row in rows.items():
        allowed = {"family", "repository", "commit", "font_filenames", "source_paths", "license_filename", "split", "lineage_group", "lineage_evidence"}
        if set(row) != allowed or row["split"] not in ["train", "validation", "test"] or not row["lineage_group"]:
            raise ValueError("Unknown/missing source configuration fields or invalid split")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", family) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["repository"]):
            raise ValueError("Invalid source identity")
        if not re.fullmatch(r"[a-f0-9]{40}", row["commit"]):
            raise ValueError("A complete immutable Git commit is required")
        paths = row["source_paths"]
        if len(paths) != len(set(paths)) or len({PurePosixPath(p).name for p in paths}) != len(paths):
            raise ValueError("Duplicate source path or colliding filename")
        for path in paths:
            p = PurePosixPath(path)
            if p.is_absolute() or ".." in p.parts or str(p) != path or "\\" in path:
                raise ValueError("Unsafe source path")
        if not set(row["font_filenames"] + [row["license_filename"]]).issubset({PurePosixPath(p).name for p in paths}):
            raise ValueError("Declared font/notice absent from source paths")
    return rows


def validate_replay(spec_bytes, lock_bytes, review_bytes):
    """Check all config/lock/review bindings before network access or writes."""
    spec, lock, review = [json.loads(b) for b in [spec_bytes, lock_bytes, review_bytes]]
    configured = validate_spec(spec)
    if type(lock.get("schema")) is not int or lock.get("schema") not in [1, 2] or type(review.get("schema")) is not int or review.get("schema") != 1:
        raise ValueError("Unsupported lock/review schema")
    if review.get("lock_sha256") != digest(lock_bytes) or review.get("source_config_sha256") != digest(spec_bytes):
        raise ValueError("Review binding mismatch: config or lock changed")
    if type(review.get("source_config_schema")) is not int or type(review.get("lock_schema")) is not int or review.get("source_config_schema") != spec.get("schema", 1) or review.get("lock_schema") != lock["schema"]:
        raise ValueError("Review schema binding mismatch")
    if lock["schema"] == 2 and (lock.get("source_config_sha256") != digest(spec_bytes) or type(lock.get("source_config_schema")) is not int or lock.get("source_config_schema") != spec.get("schema", 1)):
        raise ValueError("Lock configuration fingerprint mismatch")
    locked = unique(lock["fonts"], "family", "locked family")
    if list(configured) != list(locked):
        raise ValueError("Config/lock family membership mismatch")
    reviewed = unique(review["assets"], "asset_id", "reviewed asset")
    decisions = unique(review["decisions"], "id", "review decision")
    actual_ids, local_paths = set(), set()
    for family, row in configured.items():
        old = locked[family]
        if any(old.get(key) != value for key, value in row.items()):
            raise ValueError("Config/lock identity mismatch: " + family)
        assets = unique(old["assets"], "source_path", "locked source asset")
        if list(assets) != row["source_paths"]:
            raise ValueError("Config/lock asset membership mismatch: " + family)
        for source_path, asset in assets.items():
            asset_id = family + ":" + source_path
            actual_ids.add(asset_id)
            url = f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{source_path}"
            data_root = "data/fonts" if lock["schema"] == 1 else lock["data_root"]
            expected_path = data_root + "/" + family + "/" + PurePosixPath(source_path).name
            p = PurePosixPath(expected_path)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != "data" or "\\" in expected_path:
                raise ValueError("Unsafe local asset path")
            if asset.get("url") != url or asset.get("local_path") != expected_path or expected_path in local_paths:
                raise ValueError("Locked URL/path mismatch")
            local_paths.add(expected_path)
            if not re.fullmatch(r"[a-f0-9]{64}", asset["sha256"]) or type(asset["bytes"]) is not int or asset["bytes"] < 0:
                raise ValueError("Invalid locked hash/size")
            record = reviewed.get(asset_id)
            identity = {"family": family, "repository": row["repository"], "commit": row["commit"], "source_path": source_path, "sha256": asset["sha256"], "bytes": asset["bytes"]}
            if record is None or any(record.get(k) != v for k, v in identity.items()):
                raise ValueError("Missing or mismatched explicit asset review: " + asset_id)
            decision = decisions.get(record.get("decision_id"))
            if record.get("status") != "approved-for-local-research" or not decision or decision.get("status") != "approved-for-local-research":
                raise ValueError("Asset review is pending or not approved: " + asset_id)
            if decision.get("selected_license") != "OFL-1.1" or decision.get("local_rendering_and_research") is not True:
                raise ValueError("No explicit local-research licence decision")
            evidence = decision.get("notice_evidence", {})
            notice_path = next(p for p in row["source_paths"] if PurePosixPath(p).name == row["license_filename"])
            if evidence.get("source_path") != notice_path or evidence.get("sha256") != assets[notice_path]["sha256"] or not evidence.get("affirmative_grant_excerpt") or not evidence.get("reviewer") or evidence.get("grant_review") != "affirmative-grant-reviewed":
                raise ValueError("Incomplete reviewed notice evidence")
            names = decision.get("reserved_font_names", {})
            if names.get("status") not in ["not-assessed-for-export", "reviewed"]:
                raise ValueError("Missing explicit RFN review status")
            if names["status"] == "not-assessed-for-export" and names.get("names") is not None:
                raise ValueError("Unassessed RFNs must remain unknown, not an inferred name list")
            if names["status"] == "reviewed" and (not isinstance(names.get("names"), list) or not names.get("evidence")):
                raise ValueError("A reviewed RFN finding requires names and evidence")
    if set(reviewed) != actual_ids:
        raise ValueError("Review/config asset membership mismatch")
    return spec, lock, review


def verify_local_assets(root, lock, review):
    decisions = {r["id"]: r for r in review["decisions"]}
    reviewed = {r["asset_id"]: r for r in review["assets"]}
    checked = set()
    for row in lock["fonts"]:
        for asset in row["assets"]:
            data = (root / asset["local_path"]).read_bytes()
            if len(data) != asset["bytes"] or digest(data) != asset["sha256"]:
                raise ValueError("Local source hash/size mismatch: " + asset["local_path"])
            decision = decisions[reviewed[row["family"] + ":" + asset["source_path"]]["decision_id"]]
            if decision["id"] in checked:
                continue
            evidence = decision["notice_evidence"]
            notice = next(a for a in row["assets"] if a["source_path"] == evidence["source_path"])
            notice_bytes = (root / notice["local_path"]).read_bytes()
            if digest(notice_bytes) != evidence["sha256"]:
                raise ValueError("Reviewed notice hash mismatch")
            if " ".join(evidence["affirmative_grant_excerpt"].split()) not in " ".join(notice_bytes.decode("utf-8-sig").split()):
                raise ValueError("Reviewed grant excerpt absent from exact notice")
            checked.add(decision["id"])


def load_approved_corpus(root):
    result = validate_replay((root / "configs/sources.json").read_bytes(), (root / "manifests/corpus.lock.json").read_bytes(), (root / "manifests/source-review.json").read_bytes())
    verify_local_assets(root, result[1], result[2])
    return result[1]
