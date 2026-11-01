"""One-off source audit; acquisition uses the resulting pinned lock, never HEAD."""
import concurrent.futures
import json
import urllib.request
import urllib.parse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mashi-ML-research"}), timeout=45) as response:
        data = response.read()
    return data


def inspect(project):
    api = "https://gitlab.com/api/v4/projects/" + str(project["id"])
    commit = json.loads(get(api + "/repository/commits/" + project["default_branch"]))["id"]
    tree = []
    for page in range(1, 8):
        part = json.loads(get(api + f"/repository/tree?recursive=true&per_page=100&page={page}&ref={commit}"))
        tree += part
        if len(part) < 100:
            break
    paths = [r["path"] for r in tree if r["type"] == "blob"]
    license_path = next((p for p in paths if p.lower() in ["ofl.txt", "license.txt", "license", "copying"]), None)
    if not license_path:
        return {"name": project["path"], "commit": commit, "error": "no top-level licence", "paths": paths}
    raw = api + "/repository/files/"
    def file(path):
        return get(raw + urllib.parse.quote(path, safe="") + "/raw?ref=" + commit)
    notice = file(license_path).decode("utf-8")
    readme_path = next((p for p in paths if p.lower() in ["readme.md", "readme"]), None)
    readme = file(readme_path).decode("utf-8") if readme_path else ""
    folder = ROOT / "data" / "audit" / project["path"]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "license.txt").write_text(notice)
    (folder / "readme.md").write_text(readme)
    return {"name": project["path"], "id": project["id"], "commit": commit,
            "license_path": license_path, "raw_base": raw,
            "ofl": "SIL OPEN FONT LICENSE Version 1.1" in notice,
            "font_paths": [p for p in paths if p.lower().endswith((".ttf", ".otf"))],
            "copyright": notice.split("PREAMBLE")[0][:1500]}


if __name__ == "__main__":
    cached = ROOT / "data/source-page-0.txt"
    projects = json.loads(cached.read_text()) if cached.exists() else json.loads(get("https://gitlab.com/api/v4/groups/rit-fonts/projects?per_page=100"))
    names = {"Bahadur", "Thaara", "Lekha", "Lasya", "Keram", "Indira", "Ala", "chingam", "karuna", "Kutty", "MeeraNew", "rit-uroob", "rit-keraleeyam", "Panmana", "ezhuthu", "RIT-Rachana", "tnjoy", "Sundar"}
    rows = []
    for project in [p for p in projects if p["path"] in names]:
        try:
            row = inspect(project)
        except Exception as exc:
            row = {"name": project["path"], "error": str(exc)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (ROOT / "data/rit-discovery.json").write_text(json.dumps(rows, indent=2))
        if "429" in row.get("error", ""):
            print("Stopping audit at server rate limit; remaining sources are unacquired.", flush=True)
            break
        time.sleep(1)
    (ROOT / "data/rit-discovery.json").write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(json.dumps(row), flush=True)
