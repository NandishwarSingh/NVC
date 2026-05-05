#!/usr/bin/env python3
"""Train or export NVC-TinySR-v0.

The default path has no third-party dependencies: it exports a deterministic
nearest-neighbor TinySR model as a self-contained MOD0 artifact. If PyTorch is
installed, `--train` enables a small supervised training loop over paired tensor
files for future experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Iterable


MODEL_ID = "NVC-TinySR-v0"
MODL_MAGIC = b"MOD0"
VERSION_MAJOR = 0
VERSION_MINOR = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or export NVC-TinySR-v0")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train"))
    parser.add_argument("--val", type=Path, default=Path("datasets/val"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export", type=Path, default=Path("ml/exports/nvc-tinysr-v0.modl"))
    parser.add_argument("--train", action="store_true", help="Run PyTorch training before export")
    parser.add_argument("--source-video", type=Path, help="Train from a normal video file using FFmpeg-extracted frames")
    parser.add_argument("--source-dir", type=Path, help="Train from every supported video file in a directory")
    parser.add_argument(
        "--source-list",
        type=Path,
        help="Train from a newline-delimited text file of video paths",
    )
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum source-video frames to train on")
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=32,
        help="Maximum frames to extract from each video when using --source-dir or --source-list",
    )
    parser.add_argument("--train-width", type=int, default=256, help="Training crop/frame width; must be even")
    parser.add_argument("--train-height", type=int, default=144, help="Training crop/frame height; must be even")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="PyTorch training device",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate paths and print the planned run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("NVC-TinySR-v0")
    print(f"dataset: {args.dataset}")
    print(f"validation: {args.val}")
    print(f"epochs: {args.epochs}")
    print(f"batch size: {args.batch_size}")
    print(f"learning rate: {args.lr}")
    if args.source_video:
        print(f"source video: {args.source_video}")
        print(f"training frames: {args.max_frames} at {args.train_width}x{args.train_height}")
    if args.source_dir:
        print(f"source dir: {args.source_dir}")
        print(f"frames per video: {args.frames_per_video} at {args.train_width}x{args.train_height}")
    if args.source_list:
        print(f"source list: {args.source_list}")
        print(f"frames per video: {args.frames_per_video} at {args.train_width}x{args.train_height}")
    print(f"device: {args.device}")
    print(f"export: {args.export}")
    print(f"mode: {'train+export' if args.train else 'deterministic export'}")

    if args.dry_run:
        print("dry run complete")
        return

    random.seed(args.seed)
    metadata, weights = initialized_tinysr()

    if args.train:
        metadata, weights = train_with_torch(args, metadata, weights)

    write_mod0(args.export, metadata, weights)
    print(f"wrote {args.export} ({args.export.stat().st_size:,} bytes)")
    print(f"model id: {metadata['model_id']}")
    print(f"parameters: {sum(layer['parameter_count'] for layer in metadata['layers']):,}")


def initialized_tinysr() -> tuple[dict, list[float]]:
    """Create deterministic TinySR weights.

    Architecture:
    - conv 3 -> 16, 3x3, ReLU
    - conv 16 -> 16, 3x3, ReLU
    - conv 16 -> 12, 3x3
    - pixel shuffle x2

    The initialized model copies RGB through the hidden layers and fills the
    four pixel-shuffle subpixels with the same RGB value. It is simple, but it
    is a real neural model artifact with executable weights.
    """

    layers: list[dict] = []
    weights: list[float] = []

    conv0 = zeros(16 * 3 * 3 * 3)
    bias0 = zeros(16)
    for channel in range(3):
        conv0[index4(channel, channel, 1, 1, 3, 3, 3)] = 1.0
    add_layer(layers, weights, "conv0", "conv2d", [16, 3, 3, 3], conv0, bias0)

    conv1 = zeros(16 * 16 * 3 * 3)
    bias1 = zeros(16)
    for channel in range(3):
        conv1[index4(channel, channel, 1, 1, 16, 3, 3)] = 1.0
    add_layer(layers, weights, "conv1", "depthwiseish_conv2d", [16, 16, 3, 3], conv1, bias1)

    conv2 = zeros(12 * 16 * 3 * 3)
    bias2 = zeros(12)
    for subpixel in range(4):
        for channel in range(3):
            out_channel = channel * 4 + subpixel
            conv2[index4(out_channel, channel, 1, 1, 16, 3, 3)] = 1.0
    add_layer(layers, weights, "conv2", "conv2d", [12, 16, 3, 3], conv2, bias2)

    metadata = {
        "format": "MOD0",
        "model_id": MODEL_ID,
        "architecture": "tiny_cnn_pixel_shuffle_x2",
        "scale": 2,
        "input": {"name": "lr_rgb", "layout": "nchw", "channels": 3, "range": "0..1"},
        "output": {"name": "sr_rgb", "layout": "nchw", "channels": 3, "range": "0..1"},
        "activation": "relu_after_conv0_and_conv1",
        "weights_dtype": "float32le",
        "layers": layers,
        "notes": [
            "Deterministic bootstrap weights perform nearest-neighbor x2 upscaling.",
            "Trained weights can replace this artifact without changing the MOD0 format.",
        ],
    }
    return metadata, weights


def train_with_torch(args: argparse.Namespace, metadata: dict, weights: list[float]) -> tuple[dict, list[float]]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        print("PyTorch is not installed, so training is skipped.")
        print(f"import error: {exc}")
        return metadata, weights

    class TinySR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv0 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 12, 3, padding=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = F.relu(self.conv0(x))
            x = F.relu(self.conv1(x))
            x = self.conv2(x)
            return F.pixel_shuffle(x, 2).clamp(0, 1)

    torch.manual_seed(args.seed)
    device = select_device(args.device, torch)
    model = TinySR().to(device)
    load_initialized_weights(model, weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    video_sources = list_video_sources(args)
    source_batches = make_training_batches(args, torch, F, device, video_sources)
    if not source_batches:
        print("No video or .pt training pairs found, exporting deterministic bootstrap weights.")
        print("Pass --source-video input.mp4, or provide paired files like clip_001.lr.pt and clip_001.hr.pt.")
        return metadata, weights

    final_loss = 0.0
    for epoch in range(args.epochs):
        random.shuffle(source_batches)
        total_loss = 0.0
        for lr, hr in source_batches:
            lr = lr.to(device)
            hr = hr.to(device)
            pred = model(lr)
            loss = F.l1_loss(pred, hr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
        final_loss = total_loss / max(len(source_batches), 1)
        print(f"epoch {epoch + 1}/{args.epochs} l1={final_loss:.6f}")

    trained = extract_weights(model)
    metadata["notes"] = ["Weights were exported after PyTorch training."]
    metadata["trained"] = True
    metadata["training"] = {
        "source_video": str(args.source_video) if args.source_video else None,
        "source_dir": str(args.source_dir) if args.source_dir else None,
        "source_list": str(args.source_list) if args.source_list else None,
        "video_sources": [str(path) for path in video_sources],
        "dataset": str(args.dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": str(device),
        "max_frames": args.max_frames,
        "frames_per_video": args.frames_per_video,
        "train_width": args.train_width,
        "train_height": args.train_height,
        "final_l1": final_loss,
    }
    return metadata, trained


def select_device(requested: str, torch: object) -> object:
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_training_batches(
    args: argparse.Namespace,
    torch: object,
    functional: object,
    device: object,
    video_sources: list[Path],
) -> list[tuple[object, object]]:
    if video_sources:
        hr_tensors = []
        for path in video_sources:
            frame_limit = args.max_frames if args.source_video and len(video_sources) == 1 else args.frames_per_video
            hr_tensors.append(extract_video_tensor(path, frame_limit, args, torch))
        hr = torch.cat(hr_tensors, dim=0)
        lr = functional.interpolate(hr, scale_factor=0.5, mode="bilinear", align_corners=False)
        print(f"training corpus frames: {hr.shape[0]}")
        return batch_tensors(lr, hr, args.batch_size, device)

    pairs = list_tensor_pairs(args.dataset)
    if not pairs:
        return []

    batches: list[tuple[object, object]] = []
    for lr_path, hr_path in pairs:
        lr = torch.load(lr_path, map_location="cpu").float()
        hr = torch.load(hr_path, map_location="cpu").float()
        if lr.ndim == 3:
            lr = lr.unsqueeze(0)
        if hr.ndim == 3:
            hr = hr.unsqueeze(0)
        batches.append((lr, hr))
    return batches


def list_video_sources(args: argparse.Namespace) -> list[Path]:
    if args.source_video:
        return [args.source_video]

    paths: list[Path] = []
    if args.source_dir:
        if not args.source_dir.exists():
            raise SystemExit(f"source dir not found: {args.source_dir}")
        for suffix in ["*.mp4", "*.mov", "*.mkv", "*.webm", "*.y4m"]:
            paths.extend(sorted(args.source_dir.glob(suffix)))

    if args.source_list:
        if not args.source_list.exists():
            raise SystemExit(f"source list not found: {args.source_list}")
        for line in args.source_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            paths.append(Path(stripped))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def extract_video_tensor(video_path: Path, max_frames: int, args: argparse.Namespace, torch: object) -> object:
    if args.train_width <= 0 or args.train_height <= 0 or args.train_width % 2 != 0 or args.train_height % 2 != 0:
        raise SystemExit("--train-width and --train-height must be positive even numbers")
    if max_frames <= 0:
        raise SystemExit("--max-frames and --frames-per-video must be positive")
    if not video_path.exists():
        raise SystemExit(f"source video not found: {video_path}")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("FFmpeg is required for --source-video training")

    frame_size = args.train_width * args.train_height * 3
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps=15,scale={args.train_width}:{args.train_height}:flags=bicubic,format=rgb24",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"ffmpeg frame extraction failed:\n{err}")
    if len(result.stdout) < frame_size:
        raise SystemExit("ffmpeg did not return enough RGB frame data")

    frame_count = len(result.stdout) // frame_size
    raw = result.stdout[: frame_count * frame_size]
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    tensor = tensor.reshape(frame_count, args.train_height, args.train_width, 3)
    tensor = tensor.permute(0, 3, 1, 2).float().div(255.0)
    print(f"extracted {frame_count} training frames from {video_path}")
    return tensor


def batch_tensors(lr: object, hr: object, batch_size: int, device: object) -> list[tuple[object, object]]:
    if batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    batches: list[tuple[object, object]] = []
    for start in range(0, lr.shape[0], batch_size):
        end = min(start + batch_size, lr.shape[0])
        batches.append((lr[start:end].to(device), hr[start:end].to(device)))
    return batches


def list_tensor_pairs(dataset: Path) -> list[tuple[Path, Path]]:
    if not dataset.exists():
        return []
    pairs: list[tuple[Path, Path]] = []
    for lr_path in sorted(dataset.glob("*.lr.pt")):
        hr_path = lr_path.with_name(lr_path.name.replace(".lr.pt", ".hr.pt"))
        if hr_path.exists():
            pairs.append((lr_path, hr_path))
    return pairs


def load_initialized_weights(model: object, weights: list[float]) -> None:
    import torch

    tensors = [
        ("conv0.weight", [16, 3, 3, 3]),
        ("conv0.bias", [16]),
        ("conv1.weight", [16, 16, 3, 3]),
        ("conv1.bias", [16]),
        ("conv2.weight", [12, 16, 3, 3]),
        ("conv2.bias", [12]),
    ]
    state = {}
    offset = 0
    for name, shape in tensors:
        count = math.prod(shape)
        state[name] = torch.tensor(weights[offset : offset + count], dtype=torch.float32).reshape(shape)
        offset += count
    model.load_state_dict(state)


def extract_weights(model: object) -> list[float]:
    params: list[float] = []
    state = model.state_dict()
    for name in ["conv0.weight", "conv0.bias", "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
        params.extend(float(x) for x in state[name].detach().cpu().reshape(-1).tolist())
    return params


def write_mod0(path: Path, metadata: dict, weights: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    weight_values = list(weights)
    weight_bytes = struct.pack(f"<{len(weight_values)}f", *weight_values)
    with path.open("wb") as fh:
        fh.write(MODL_MAGIC)
        fh.write(struct.pack("<HHIQ", VERSION_MAJOR, VERSION_MINOR, len(metadata_bytes), len(weight_bytes)))
        fh.write(metadata_bytes)
        fh.write(weight_bytes)


def add_layer(
    layers: list[dict],
    weights: list[float],
    name: str,
    kind: str,
    weight_shape: list[int],
    weight_values: list[float],
    bias_values: list[float],
) -> None:
    weight_offset = len(weights) * 4
    weights.extend(weight_values)
    bias_offset = len(weights) * 4
    weights.extend(bias_values)
    parameter_count = len(weight_values) + len(bias_values)
    layers.append(
        {
            "name": name,
            "kind": kind,
            "weight_shape": weight_shape,
            "bias_shape": [len(bias_values)],
            "weight_offset": weight_offset,
            "bias_offset": bias_offset,
            "parameter_count": parameter_count,
        }
    )


def zeros(count: int) -> list[float]:
    return [0.0 for _ in range(count)]


def index4(o: int, i: int, y: int, x: int, in_channels: int, kernel_h: int, kernel_w: int) -> int:
    return (((o * in_channels + i) * kernel_h + y) * kernel_w) + x


if __name__ == "__main__":
    main()
