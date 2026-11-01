"""Replay an explicitly reviewed corpus, or stage a new PENDING source lock."""
import argparse
import json
import urllib.request
from pathlib import Path

from mashi_ml.acquisition import digest, validate_spec, validate_replay, verify_local_assets

ROOT = Path(__file__).resolve().parents[1]


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mashi-ML"}), timeout=60) as response:
        return response.read()


def replay(root, spec_bytes, lock_bytes, review_bytes, fetcher=fetch):
    _, lock, review = validate_replay(spec_bytes, lock_bytes, review_bytes)
    for row in lock["fonts"]:
        for asset in row["assets"]:
            dest = root / asset["local_path"]
            data = dest.read_bytes() if dest.exists() else None
            if data is None or digest(data) != asset["sha256"] or len(data) != asset["bytes"]:
                data = fetcher(asset["url"])
                if digest(data) != asset["sha256"] or len(data) != asset["bytes"]:
                    raise ValueError("Downloaded source hash/size mismatch: " + asset["url"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
    verify_local_assets(root, lock, review)
    print("Verified exact config/lock/review identity, all locked bytes and explicit notice evidence.")


def stage_pending(root, spec_bytes, output, fetcher=fetch):
    """Acquisition is not licence clearance. Never overwrite an existing lock."""
    if output.exists():
        raise ValueError("Refusing to overwrite an existing lock")
    spec = json.loads(spec_bytes)
    validate_spec(spec)
    data_root = "data/acquisitions/" + digest(spec_bytes)[:16]
    rows = []
    for configured in spec["fonts"]:
        row = dict(configured)
        assets = []
        for source_path in row["source_paths"]:
            url = f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{source_path}"
            data = fetcher(url)
            local_path = data_root + "/" + row["family"] + "/" + Path(source_path).name
            dest = root / local_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            assets.append({"source_path": source_path, "url": url, "local_path": local_path, "sha256": digest(data), "bytes": len(data)})
        row.update(assets=assets, review_status="pending-manual-review", approved_usage=None,
                   reserved_font_names={"status": "pending-manual-review", "names": None},
                   raw_notice_asset=next(a["source_path"] for a in assets if Path(a["source_path"]).name == row["license_filename"]))
        rows.append(row)
    lock = {"schema": 2, "source_config_schema": spec.get("schema", 1), "source_config_sha256": digest(spec_bytes), "data_root": data_root,
            "review_status": "pending-manual-review", "fonts": rows,
            "policy": "Raw acquisition only. No heading/filename automatically approves a licence, usage or RFN finding. Every asset needs an explicit hash-bound source-review decision before rendering/research."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    print("Staged PENDING lock; no licence/research/export approval granted: " + str(output))
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-lock", action="store_true", help="Stage a PENDING new lock; never approve assets")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sources.json")
    parser.add_argument("--lock", type=Path, default=ROOT / "manifests/corpus.lock.json")
    parser.add_argument("--review-manifest", type=Path, default=ROOT / "manifests/source-review.json")
    parser.add_argument("--output-lock", type=Path)
    args = parser.parse_args()
    if args.create_lock:
        if args.output_lock is None:
            parser.error("--create-lock requires an explicit new --output-lock; the campaign lock is immutable")
        stage_pending(ROOT, args.config.read_bytes(), args.output_lock)
    else:
        replay(ROOT, args.config.read_bytes(), args.lock.read_bytes(), args.review_manifest.read_bytes())


if __name__ == "__main__":
    main()
