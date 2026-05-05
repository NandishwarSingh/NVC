#!/usr/bin/env python3
"""Lightweight benchmark helper for NVC artifacts.

This intentionally uses only the Python standard library plus FFmpeg tools.
Heavy perceptual metrics such as VMAF and LPIPS can be added later without
blocking the basic size/speed/quality smoke loop.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an NVC benchmark artifact")
    parser.add_argument("source", type=Path, help="Original source video")
    parser.add_argument("nvc", type=Path, help="Encoded .nvc file")
    parser.add_argument("--decoded", type=Path, help="Decoded comparison video")
    parser.add_argument("--nvc-bin", type=Path, default=Path("./zig-out/bin/nvc"), help="Path to the nvc CLI")
    parser.add_argument("--metrics", default="psnr,ssim", help="Comma-separated FFmpeg metrics: psnr,ssim,none")
    return parser.parse_args()


def run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def ffprobe(path: Path) -> dict[str, float | int | str]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or 0)
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": parse_rate(stream.get("r_frame_rate") or "0/1"),
        "duration": duration,
    }


def parse_rate(text: str) -> float:
    if "/" not in text:
        return float(text or 0)
    num, den = text.split("/", 1)
    denominator = float(den or 1)
    return 0.0 if denominator == 0 else float(num or 0) / denominator


def nvc_info(nvc_bin: Path, nvc: Path) -> dict[str, str]:
    result = run([str(nvc_bin), "info", str(nvc)])
    info: dict[str, str] = {}
    for line in result.stderr.splitlines() + result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()
    return info


def ffmpeg_metric(source: Path, decoded: Path, metric: str) -> str | None:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(decoded), "-i", str(source), "-lavfi", metric, "-f", "null", "-"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    if metric == "psnr":
        match = re.search(r"average:([0-9.]+)", result.stderr)
        return match.group(1) if match else None
    if metric == "ssim":
        match = re.search(r"All:([0-9.]+)", result.stderr)
        return match.group(1) if match else None
    return None


def main() -> None:
    args = parse_args()
    source_probe = ffprobe(args.source)
    nvc_bytes = args.nvc.stat().st_size
    duration = float(source_probe["duration"] or 0)
    bitrate_kbps = (nvc_bytes * 8 / duration / 1000) if duration > 0 else 0
    info = nvc_info(args.nvc_bin, args.nvc)

    print("NVC benchmark")
    print(f"source={args.source}")
    print(f"nvc={args.nvc}")
    print(f"nvc_bytes={nvc_bytes}")
    print(f"bitrate_kbps={bitrate_kbps:.2f}")
    print(f"source_width={source_probe['width']}")
    print(f"source_height={source_probe['height']}")
    print(f"source_fps={source_probe['fps']:.3f}")
    for key in ["profile", "frames", "base_codec", "base_coded_bytes", "feature_format", "feature_coded_bytes", "model_id"]:
        if key in info:
            print(f"{key}={info[key]}")

    if not args.decoded:
        print("decoded=not provided")
        return

    decoded_probe = ffprobe(args.decoded)
    print(f"decoded={args.decoded}")
    print(f"decoded_width={decoded_probe['width']}")
    print(f"decoded_height={decoded_probe['height']}")
    requested = {item.strip().lower() for item in args.metrics.split(",") if item.strip()}
    if "none" in requested:
        return
    if "psnr" in requested:
        print(f"psnr_avg={ffmpeg_metric(args.source, args.decoded, 'psnr') or 'unavailable'}")
    if "ssim" in requested:
        print(f"ssim_all={ffmpeg_metric(args.source, args.decoded, 'ssim') or 'unavailable'}")


if __name__ == "__main__":
    main()
