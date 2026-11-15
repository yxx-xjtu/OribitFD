"""Download the 16 official CWRU 48k drive-end & normal baseline .mat files.

Verifies DE channel integrity and writes ``registry.json`` with SHA-256 digests
used by ``scripts/prepare_benchmarks.py`` for task groups T3, T6, and T7.
"""
from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from scipy.io import loadmat

BASE = "https://engineering.case.edu"


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[str, str]] = []
        self.href: str | None = None
        self.label: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = ""

    def handle_data(self, data: str) -> None:
        if self.href:
            self.label += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.href:
            self.rows.append((self.label.strip(), urljoin(BASE, self.href)))
            self.href = None


def fetch(url: str) -> bytes:
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 OrbitFD-Reproducibility"})
            with urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as exc:
            print(f"Download retry {attempt + 1}/4 for {url}: {type(exc).__name__}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official CWRU 48k benchmark subset.")
    parser.add_argument("--out", default="data/CWRU48_official", help="Destination folder.")
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    registry = []
    for page in ("normal-baseline-data", "48k-drive-end-bearing-fault-data"):
        url = f"{BASE}/bearingdatacenter/{page}"
        html = fetch(url)
        (root / f"{page}.html").write_bytes(html)
        link_parser = Links()
        link_parser.feed(html.decode("utf-8", errors="ignore"))
        for label, link in link_parser.rows:
            if not label.startswith(("Normal_", "IR007_", "B007_", "OR007@6_")):
                continue
            load = int(label.rsplit("_", 1)[1])
            cls = (
                0
                if label.startswith("Normal")
                else (1 if label.startswith("IR") else (2 if label.startswith("OR") else 3))
            )
            dest = root / (label.replace("@", "_") + ".mat")
            if not dest.exists():
                dest.write_bytes(fetch(link))
            mat = loadmat(dest)
            keys = [k for k in mat if k.endswith("_DE_time")]
            expected = "X" + str(int(Path(link).stem)).zfill(3) + "_DE_time"
            channel = expected if expected in keys else (keys[0] if len(keys) == 1 else None)
            if channel is None:
                raise ValueError(f"Cannot resolve DE channel for {label}: {keys}")
            x = mat[channel].reshape(-1)
            registry.append(
                {
                    "file": dest.name,
                    "label": cls,
                    "condition": load,
                    "severity": "healthy" if cls == 0 else "007",
                    "sampling_rate": 48000,
                    "raw_points": int(len(x)),
                    "channel": channel,
                    "official_url": link,
                    "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                }
            )
            print(f"Verified {dest.name:20s} | points={len(x):7d} | channel={channel}", flush=True)

    if len(registry) != 16 or len({(r["label"], r["condition"]) for r in registry}) != 16:
        raise RuntimeError(f"Expected 16 unique (class, load) files, got {len(registry)}.")
    (root / "registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Successfully prepared CWRU48 registry at {root / 'registry.json'}")


if __name__ == "__main__":
    main()
