"""Download optional experiment data from a pinned archive, verifying SHA-256."""

import argparse
import hashlib
import json
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fetch(path, expected, base_url):
    target = HERE / "artifacts" / path
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected["sha256"]:
        return
    with urllib.request.urlopen(f"{base_url}/{path}", timeout=60) as response:
        data = response.read()
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-only", action="store_true", help="Download only data for a new run"
    )
    args = parser.parse_args()
    manifest = json.loads((HERE / "artifact_manifest.json").read_text())
    files = {
        path: metadata
        for path, metadata in manifest["files"].items()
        if not args.corpus_only or path.startswith("fixtures/")
    }
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: fetch(*item, manifest["base_url"]), files.items()))
    print(f"Verified {len(files)} artifacts in {HERE / 'artifacts'}")


if __name__ == "__main__":
    main()
