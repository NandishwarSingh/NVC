#!/usr/bin/env python3
"""Download real-video corpora for NVC-TinySR training.

The mini preset intentionally uses small WebM previews from Xiph's public test
media. The xiph-3gb preset downloads larger Y4M sources for a more meaningful
temporary training pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path


DERF_URL = "https://media.xiph.org/video/derf"
WEBM_URL = f"{DERF_URL}/webm"
Y4M_URL = f"{DERF_URL}/y4m"
XIPH_MINI_CORPUS = [
    {
        "name": "FourPeople",
        "url": f"{WEBM_URL}/FourPeople_1280x720_60.webm",
        "license_note": "Xiph Derf test media WebM preview",
    },
    {
        "name": "Johnny",
        "url": f"{WEBM_URL}/Johnny_1280x720_60.webm",
        "license_note": "Xiph Derf test media, listed as public domain",
    },
    {
        "name": "KristenAndSara",
        "url": f"{WEBM_URL}/KristenAndSara_1280x720_60.webm",
        "license_note": "Xiph Derf test media WebM preview",
    },
    {
        "name": "vidyo1",
        "url": f"{WEBM_URL}/vidyo1_720p_60fps.webm",
        "license_note": "Xiph Derf test media, listed as public domain",
    },
    {
        "name": "vidyo3",
        "url": f"{WEBM_URL}/vidyo3_720p_60fps.webm",
        "license_note": "Xiph Derf test media, listed as public domain",
    },
    {
        "name": "vidyo4",
        "url": f"{WEBM_URL}/vidyo4_720p_60fps.webm",
        "license_note": "Xiph Derf test media, listed as public domain",
    },
]

XIPH_3GB_CORPUS = [
    {
        "name": "FourPeople",
        "url": f"{Y4M_URL}/FourPeople_1280x720_60.y4m",
        "size_bytes": 830_826_065,
        "license_note": "Xiph Derf test media Y4M source",
    },
    {
        "name": "Johnny",
        "url": f"{Y4M_URL}/Johnny_1280x720_60.y4m",
        "size_bytes": 830_826_065,
        "license_note": "Xiph Derf test media Y4M source",
    },
    {
        "name": "KristenAndSara",
        "url": f"{Y4M_URL}/KristenAndSara_1280x720_60.y4m",
        "size_bytes": 830_826_065,
        "license_note": "Xiph Derf test media Y4M source",
    },
    {
        "name": "BlueSky",
        "url": f"{Y4M_URL}/blue_sky_1080p25.y4m",
        "size_bytes": 674_958_138,
        "license_note": "Xiph Derf test media Y4M source",
    },
]

PRESETS = {
    "mini": XIPH_MINI_CORPUS,
    "xiph-3gb": XIPH_3GB_CORPUS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NVC Xiph mini training corpus")
    parser.add_argument("--out", type=Path, default=Path("datasets/xiph-mini"))
    parser.add_argument("--preset", choices=sorted(PRESETS), default="mini")
    parser.add_argument("--limit", type=int, help="Number of corpus videos to download")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    corpus = PRESETS[args.preset]
    limit = len(corpus) if args.limit is None else args.limit
    selected = corpus[: max(0, min(limit, len(corpus)))]
    if not selected:
        raise SystemExit("--limit selected zero videos")

    for entry in selected:
        url = entry["url"]
        filename = url.rsplit("/", 1)[-1]
        target = args.out / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"skip existing {target}")
            continue
        print(f"download {url}")
        download(url, target)
        print(f"wrote {target} ({target.stat().st_size:,} bytes)")

    manifest = {
        "name": f"nvc-{args.preset}",
        "source": "https://media.xiph.org/video/derf/",
        "description": "Real-video corpus for NVC-TinySR-v0 training.",
        "preset": args.preset,
        "downloaded_bytes_estimate": sum(int(item.get("size_bytes", 0)) for item in selected),
        "items": selected,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out / "sources.txt").write_text(
        "\n".join(str(args.out / item["url"].rsplit("/", 1)[-1]) for item in selected) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {args.out / 'manifest.json'}")
    print(f"sources: {args.out / 'sources.txt'}")


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", "0"))
        with tempfile.NamedTemporaryFile(delete=False, dir=target.parent, suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
            copied = 0
            next_report = 64 * 1024 * 1024
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                copied += len(chunk)
                if copied >= next_report:
                    if total > 0:
                        percent = copied * 100 / total
                        print(f"  {copied / 1024 / 1024:.0f} MiB / {total / 1024 / 1024:.0f} MiB ({percent:.1f}%)")
                    else:
                        print(f"  {copied / 1024 / 1024:.0f} MiB")
                    next_report += 64 * 1024 * 1024
    tmp_path.replace(target)


if __name__ == "__main__":
    main()
