#!/usr/bin/env python3
"""Run a MOD0 file's TinySR weights on (LR, HR) video pairs and report PSNR.

Used to compare a per-clip distilled MOD0 against the generic default on the
same VP9-decoded base, without going through the browser.

Usage:
    python3 ml/eval_modl.py --modl PATH --lr-video LR.mp4 --hr-video HR.mp4
"""
from __future__ import annotations
import argparse
import json
import math
import struct
import subprocess
from pathlib import Path


def read_mod0(path: Path) -> tuple[dict, list[float]]:
    data = path.read_bytes()
    assert data[:4] == b"MOD0"
    metadata_len = struct.unpack("<I", data[8:12])[0]
    weights_len = struct.unpack("<Q", data[12:20])[0]
    metadata = json.loads(data[20:20 + metadata_len])
    raw = data[20 + metadata_len:20 + metadata_len + weights_len]
    floats = list(struct.unpack(f"<{weights_len // 4}f", raw))
    return metadata, floats


def extract_frames(path: Path, w: int, h: int, max_frames: int):
    import torch
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={w}:{h}:flags=bicubic,format=rgb24",
        "-frames:v", str(max_frames),
        "-f", "rawvideo", "pipe:1",
    ]
    res = subprocess.run(cmd, capture_output=True, check=True)
    frame_bytes = w * h * 3
    n = len(res.stdout) // frame_bytes
    raw = res.stdout[:n * frame_bytes]
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    return t.reshape(n, h, w, 3).permute(0, 3, 1, 2).float().div(255.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modl", type=Path, required=True)
    ap.add_argument("--lr-video", type=Path, required=True)
    ap.add_argument("--hr-video", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--lr-width", type=int, default=128)
    ap.add_argument("--lr-height", type=int, default=72)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    metadata, floats = read_mod0(args.modl)
    layers = {layer["name"]: layer for layer in metadata["layers"]}

    def tensor_for(layer_name: str, kind: str) -> torch.Tensor:
        layer = layers[layer_name]
        offset = layer[f"{kind}_offset"] // 4
        shape = layer["weight_shape" if kind == "weight" else "bias_shape"]
        count = math.prod(shape)
        return torch.tensor(floats[offset:offset + count], dtype=torch.float32).reshape(shape)

    class TinySR(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv0 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 12, 3, padding=1)

        def forward(self, x):
            x = F.relu(self.conv0(x))
            x = F.relu(self.conv1(x))
            return F.pixel_shuffle(self.conv2(x), 2).clamp(0, 1)

    model = TinySR()
    model.conv0.weight.data = tensor_for("conv0", "weight")
    model.conv0.bias.data = tensor_for("conv0", "bias")
    model.conv1.weight.data = tensor_for("conv1", "weight")
    model.conv1.bias.data = tensor_for("conv1", "bias")
    model.conv2.weight.data = tensor_for("conv2", "weight")
    model.conv2.bias.data = tensor_for("conv2", "bias")
    model.eval()

    lr = extract_frames(args.lr_video, args.lr_width, args.lr_height, args.frames)
    hr = extract_frames(args.hr_video, args.lr_width * 2, args.lr_height * 2, args.frames)
    n = min(lr.shape[0], hr.shape[0])
    lr = lr[:n]
    hr = hr[:n]

    with torch.no_grad():
        sr = model(lr)

    mse = ((sr - hr) ** 2).mean().item()
    psnr = -10 * math.log10(mse + 1e-12)
    l1 = (sr - hr).abs().mean().item()
    print(f"modl: {args.modl}")
    print(f"  frames evaluated: {n}")
    print(f"  L1 vs HR:         {l1:.4f}")
    print(f"  PSNR vs HR:       {psnr:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
