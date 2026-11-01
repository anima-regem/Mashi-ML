"""Fetch only explicit official, pinned source assets; verify lock hashes on replay."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-lock", action="store_true")
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/sources.json").read_text())
    lockpath = ROOT / "manifests/corpus.lock.json"
    old = {} if args.create_lock else json.loads(lockpath.read_text())
    locked = {r["family"]: r for r in old.get("fonts", [])}
    rows = []
    for row in spec["fonts"]:
        row = dict(row)
        family = row["family"]
        folder = ROOT / "data/fonts" / family
        folder.mkdir(parents=True, exist_ok=True)
        assets = []
        for path in row["source_paths"]:
            url = f"https://raw.githubusercontent.com/{row['repository']}/{row['commit']}/{path}"
            filename = Path(path).name
            dest = folder / filename
            expected = next((r for r in locked.get(family, {}).get("assets", []) if r["source_path"] == path), None)
            if dest.exists() and (expected is None or sha256(dest.read_bytes()) == expected["sha256"]):
                data = dest.read_bytes()
            else:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mashi-ML"}), timeout=60) as response:
                    data = response.read()
                dest.write_bytes(data)
            if expected and sha256(data) != expected["sha256"]:
                raise ValueError("Source hash mismatch: " + url)
            assets.append({"source_path": path, "url": url, "local_path": f"data/fonts/{family}/{filename}", "sha256": sha256(data), "bytes": len(data)})
        notice = (folder / row["license_filename"]).read_text()
        if "SIL OPEN FONT LICENSE Version 1.1" not in notice:
            raise ValueError("Unapproved font licence: " + family)
        (ROOT / "licenses/fonts" / (family + ".txt")).write_text(notice)
        header = notice.split("PREAMBLE")[0]
        row.update(assets=assets, license_selected="OFL-1.1", notice_header=header,
                   reserved_font_names_review="No project-specific Reserved Font Name declaration found before PREAMBLE; inspect retained notice and embedded name records before exporting a font.",
                   approved_usage={"local_rendering_and_research": True, "redistribute_original_font_binary": "Only together with retained copyright/licence notices under OFL; binaries excluded from Git.", "derivative_fonts": "OFL conditions and any reserved names apply; Mashi-generated font naming must not reuse source family names.", "model_weights": "Local research only in this campaign. OFL does not explicitly resolve ML-weight distribution; no unrestricted model licence asserted and no weights published."})
        rows.append(row)
        print(f"{family}: {row['split']}, {len(assets)} pinned assets", flush=True)
    lock = {"schema": 1, "acquired_utc": "2026-11-01", "fonts": rows,
            "split_policy": "Frozen family/derivative-group assignment before rendering or tuning; all weights, variable instances and augmentation inherit family split. Same-name SMC/RIT derivatives would share a group; no RIT binary acquired.",
            "licence_policy": "Exact per-asset notices retained. Dual licences use the offered OFL option. No font binaries in Git; no assertion of unrestricted weight/output licensing."}
    rendered = json.dumps(lock, ensure_ascii=False, indent=2) + "\n"
    if args.create_lock:
        lockpath.write_text(rendered)
    else:
        print("Verified all locked source hashes.")


if __name__ == "__main__":
    main()
