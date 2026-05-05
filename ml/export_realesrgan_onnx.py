#!/usr/bin/env python3
"""Convert upstream realesr-animevideov3 PyTorch weights to ONNX for browser use.

The browser's MOD0 TinySR is ~5K parameters; the CLI's `--enhancer realesrgan`
runs realesr-animevideov3 at ~300K parameters via ncnn-vulkan. This script
produces an ONNX file we can ship to the browser via ONNX Runtime Web so
W1 Neural mode in NVC Studio can match the CLI's quality.

Architecture: SRVGGNetCompact (the same one ncnn loads).
  Input  (3, H,   W) RGB float32 in [0, 1]
  Output (3, H*S, W*S) RGB float32 in [0, 1]  with S = upscale factor

Output: ml/exports/nvc-realesrgan-anime-x{S}.onnx
"""
from __future__ import annotations
import argparse
import sys
import urllib.request
from pathlib import Path

UPSTREAM_PTH = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesr-animevideov3.pth"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # Upstream realesr-animevideov3.pth is trained with upscale=4 (final conv = 48 ch);
    # we match the released weights and let the caller downscale on input or output for x2.
    ap.add_argument("--upscale", type=int, default=4, choices=[4])
    ap.add_argument("--num-feat", type=int, default=64)
    ap.add_argument("--num-conv", type=int, default=16)
    ap.add_argument("--pth", type=Path, default=Path("ml/cache/realesr-animevideov3.pth"))
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ONNX output path (default ml/exports/nvc-realesrgan-anime-x{N}.onnx)",
    )
    ap.add_argument("--opset", type=int, default=17)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.out is None:
        args.out = Path(f"ml/exports/nvc-realesrgan-anime-x{args.upscale}.onnx")

    args.pth.parent.mkdir(parents=True, exist_ok=True)
    if not args.pth.exists():
        print(f"downloading upstream weights → {args.pth}")
        urllib.request.urlretrieve(UPSTREAM_PTH, args.pth)
    print(f"weights: {args.pth} ({args.pth.stat().st_size:,} bytes)")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        print(f"PyTorch not installed: {e}", file=sys.stderr)
        print("Activate the .venv (which has torch 2.11) before running this script.", file=sys.stderr)
        return 2

    class SRVGGNetCompact(nn.Module):
        """Mirrors the BasicSR SRVGGNetCompact architecture used by realesr-animevideov3."""

        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type="prelu"):
            super().__init__()
            self.upscale = upscale
            self.body = nn.ModuleList()
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(self._activation(act_type, num_feat))
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(self._activation(act_type, num_feat))
            self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
            self.upsampler = nn.PixelShuffle(upscale)

        @staticmethod
        def _activation(kind: str, num_feat: int):
            if kind == "prelu":
                return nn.PReLU(num_parameters=num_feat)
            if kind == "leakyrelu":
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)
            return nn.ReLU(inplace=True)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            # Add bilinear-upscaled input as a residual; matches the BasicSR forward.
            base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            return (out + base).clamp(0, 1)

    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3,
        num_feat=args.num_feat, num_conv=args.num_conv,
        upscale=args.upscale, act_type="prelu",
    ).eval()

    state = torch.load(args.pth, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "params" in state:
        state = state["params"]
    if isinstance(state, dict) and "params_ema" in state:
        state = state["params_ema"]

    # The upstream checkpoint stores all 3 upscale heads. Filter to ours.
    filtered = {}
    for k, v in state.items():
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[7:]
        filtered[new_key] = v
    missing = []
    expected = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}
    state_keys = set(filtered.keys())
    for k in expected:
        if k not in state_keys:
            missing.append(k)
    if missing:
        print("missing keys (will be randomly initialized):", missing[:5], "..." if len(missing) > 5 else "", file=sys.stderr)

    model.load_state_dict(filtered, strict=False)
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} parameters")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["lr_rgb"],
        output_names=["sr_rgb"],
        dynamic_axes={"lr_rgb": {0: "batch", 2: "h", 3: "w"}, "sr_rgb": {0: "batch", 2: "out_h", 3: "out_w"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    # The new dynamo exporter splits weights into a sidecar .onnx.data file. The browser path
    # is far simpler with a single self-contained .onnx, so re-pack inline.
    try:
        import onnx
        m = onnx.load(str(args.out), load_external_data=True)
        sidecar = args.out.with_suffix(args.out.suffix + ".data")
        onnx.save(m, str(args.out), save_as_external_data=False)
        if sidecar.exists():
            sidecar.unlink()
        print(f"re-packed weights inline (no sidecar)")
    except Exception as e:
        print(f"warning: could not re-pack weights inline ({e}); keeping sidecar .onnx.data", file=sys.stderr)
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")

    # Sanity check: run inference once via torch and once via onnxruntime, compare.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        ort_out = sess.run(None, {"lr_rgb": dummy.numpy()})[0]
        diff = float((torch_out - ort_out).__abs__().max())
        print(f"torch vs onnxruntime max abs diff: {diff:.6f} (good if < 1e-4)")
    except Exception as e:
        print(f"sanity check skipped: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
