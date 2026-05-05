# NVC: Neural Video Codec

NVC is an experimental custom video codec with its own `.nvc` file type. The goal is to compress video by storing a small low-resolution base stream plus neural data that helps reconstruct a visually accurate full-resolution video.

This repo is an alpha implementation. It already creates and reads native `.nvc` files, includes a Zig CLI, includes a browser demo, and lays down the format/spec structure for the neural codec work.

## What Is NVC?

NVC means Neural Video Codec.

The big idea is simple:

1. Take a normal video.
2. Downscale it.
3. Remove or simplify expensive details like grain and color noise.
4. Store a compact base stream in a `.nvc` file.
5. Store neural reconstruction data in the same `.nvc` file.
6. Decode it later with the bundled model/data to recreate a visually accurate video.

NVC is not trying to be mathematically lossless. It is trying to look right while using fewer bytes.

## Who Is This For?

This project is for:

- Beginners who want to learn how a video codec is built.
- Codec researchers experimenting with neural compression.
- Web video developers interested in WebGPU playback.
- AI compression builders who want a product-style format, CLI, and player.

## What Works Today

Current alpha status:

- Native `.nvc` chunked file container: working.
- Per-chunk CRC checks: working.
- `nvc encode`: working for a custom `BAS5` packetized Huffman-coded motion-compensated tiled transform base stream.
- `nvc decode`: working for alpha preview video output.
- `nvc info`: working.
- `nvc inspect`: working.
- Browser `.nvc` parser/player demo: working for metadata, chunk info, drag-and-drop, range-loaded sample playback, playable `PRVW` preview streams with play/pause/seek, BAS5 packet-index loading, GOP packet range fetches, packet byte cache, decoded GOP frame cache for Codec/Neural seek, alpha Codec playback across GOP packet boundaries, codec-base detail guidance, `FET1` feature residuals, `COL1`/`GRN1` reconstruction tuning in Neural mode, legacy color first-frame `BAS0`/`BAS1`/`BAS2`/`BAS3`/`BAS4` preview, and `MOD0` TinySR neural reconstruction through WebGPU with CPU fallback.
- `NVC-TinySR-v0` model export: working as a self-contained `MOD0` artifact.
- Optional PyTorch training path: working from normal source videos or paired `.pt` tensors.
- Feature residuals: working as compact alpha `FET1` luma correction vectors inside `FEAT`.
- `nvc bench`: working for encode/decode timing, `.nvc` size, approximate bitrate, encode FPS, and decode FPS. Perceptual metrics are still planned.

The alpha encoder uses FFmpeg to read normal video files and extract raw frames. It does not put MP4, AV1, H.264, H.265, or VP9 video streams inside `.nvc`.

## Install Requirements

You need:

- Zig `0.15.2+`
- Bun `1.3+`
- Python `3.10+`
- FFmpeg
- Optional: CUDA or Apple MPS for future model training

Check your tools:

```bash
zig version
bun --version
python3 --version
ffmpeg -version
```

## Quick Start

Build the CLI:

```bash
zig build
```

Generate a tiny sample video:

```bash
ffmpeg -y -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 2 samples/input.mp4
```

Export the bootstrap neural model:

```bash
python3 ml/train_tinysr.py --export ml/exports/nvc-tinysr-v0.modl
```

Train a tiny model from the sample video:

```bash
python3 ml/train_tinysr.py --train --source-video samples/input.mp4 --max-frames 16 --epochs 3 --train-width 256 --train-height 144 --batch-size 4 --device cpu --export ml/exports/nvc-tinysr-v0-trained.modl
```

Encode it to NVC:

```bash
./zig-out/bin/nvc encode samples/input.mp4 samples/output.nvc --profile w1 --model ml/exports/nvc-tinysr-v0.modl
```

Inspect the file:

```bash
./zig-out/bin/nvc info samples/output.nvc
./zig-out/bin/nvc inspect samples/output.nvc
```

Decode it back to a normal video:

```bash
./zig-out/bin/nvc decode samples/output.nvc samples/reconstructed.mp4
```

Run the browser demo:

```bash
cd web
bun install
bun run dev
```

Open the URL printed by Bun, then drag `samples/output.nvc` into the page.

Set up Python tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python ml/train_tinysr.py --help
python ml/benchmark.py --help
```

## For Total Beginners

An **encoder** turns a normal video into a compressed file.

A **decoder** turns the compressed file back into something you can watch.

A **container** is the file structure. For NVC, the container is the `.nvc` file.

A **profile** is a mode. NVC has two public profiles:

- `NVC-W1` for web playback.
- `NVC-XC` for extreme compression.

**Bitrate** means how many bits are used per second of video. Lower bitrate usually means smaller files.

**WebGPU** lets the browser run GPU code. NVC will use it for neural reconstruction.

**Model weights** are the numbers inside a neural network. NVC stores model data in the `.nvc` file so the file is self-contained.

## Profiles

### NVC-W1

Use this for realtime web playback.

```bash
./zig-out/bin/nvc encode input.mp4 output.nvc --profile w1
```

The alpha W1 encoder uses a half-resolution base stream. A 1920x1080 video becomes a 960x540 base stream.

### NVC-XC

Use this for smaller files when slower encoding is okay.

```bash
./zig-out/bin/nvc encode input.mp4 output.nvc --profile xc
```

The alpha XC encoder uses a quarter-resolution base stream. Future XC builds will add stronger neural and feature-level compression.

## Project Structure

```text
core/    Zig codec library and CLI
web/     Bun + TypeScript browser player
ml/      Python model training and benchmark tools
spec/    NVC file format and profile docs
samples/ sample files and fixtures
```

## How Playback Works

The final NVC playback pipeline is:

1. Parse the `.nvc` file.
2. Read the low-resolution `BASE` stream.
3. Load the bundled neural model from `MODL`.
4. Apply motion/context data from `MOTN`.
5. Apply feature residuals from `FEAT`.
6. Restore color using `COLR`.
7. Add synthetic grain using `GRAN`.
8. Draw the final frame to a canvas.

The alpha browser player parses the `.nvc` file and uses `PVW1` data inside `PRVW` for instant preview playback with play, pause, and seek controls. URL playback uses HTTP Range requests to load the header, `TOC0`, `PRVW`, `MODL`, `SEEK`, `FEAT`, `COLR`, `GRAN`, and the small BAS5 packet index first instead of downloading the whole file before the first frame. Codec and Neural modes then range-load only the BAS5 GOP packet needed for the current seek position. Neural mode runs TinySR, guides luma/detail from the decoded codec base so it does not throw away BAS5 detail, then applies `FET1` luma feature residuals, `COL1` color tuning, and `GRN1` deterministic grain. It still supports direct color first-frame decode for older `BAS0`, `BAS1`, `BAS2`, `BAS3`, and `BAS4` files that do not have `PRVW`, plus older still-frame `PVW0` files. It can run the embedded `MOD0` TinySR model through WebGPU, with a smaller CPU fallback if WebGPU is unavailable or too slow.

## How Compression Works

The intended NVC compression pipeline is:

1. Decode the source video.
2. Convert frames to YUV.
3. Downscale frames.
4. Remove or simplify grain.
5. Encode a custom low-resolution base stream with block prediction, previous-frame motion modes, RLE-compressed mode data, 4x4 transform blocks, quantization, zigzag scanning, zero-run coding, and signed varints.
6. Store neural latents and feature residuals.
7. Bundle neural model weights.
8. Save everything into one `.nvc` file.

The alpha implementation currently does steps 1, 2, 3, 5, 6, 7, and 8, plus source/base-stat `COL1` color tuning and basic `GRN1` grain synthesis. `FEAT` currently stores `FET1`, a compact per-frame tile luma correction stream. Step 4 is still basic.

## Training A Model

The model exporter lives in `ml/`. By default it writes deterministic bootstrap weights, so it works even when PyTorch is not installed.

Export the current model artifact:

```bash
python3 ml/train_tinysr.py --export ml/exports/nvc-tinysr-v0.modl
```

To train directly from a normal MP4, install PyTorch in your virtual environment, then run:

```bash
python3 ml/train_tinysr.py --train --source-video samples/input.mp4 --max-frames 16 --epochs 3 --train-width 256 --train-height 144 --batch-size 4 --device cpu --export ml/exports/nvc-tinysr-v0-trained.modl
```

This extracts small RGB frames with FFmpeg, creates low-resolution/high-resolution training pairs, trains the TinySR x2 model, and writes a `MOD0` artifact.

To download a larger temporary Xiph corpus, train, and then remove the corpus:

```bash
python3 ml/download_xiph_corpus.py --preset xiph-3gb --out datasets/xiph-3gb-temp
python3 ml/train_tinysr.py --train --source-dir datasets/xiph-3gb-temp --frames-per-video 180 --epochs 20 --train-width 384 --train-height 216 --batch-size 8 --device mps --export ml/exports/nvc-tinysr-v0-xiph-3gb.modl
rm -rf datasets/xiph-3gb-temp
```

The output file uses the `MOD0` format and can be embedded with:

```bash
./zig-out/bin/nvc encode input.mp4 output.nvc --profile w1 --model ml/exports/nvc-tinysr-v0-trained.modl
```

Optional PyTorch training expects paired tensor files:

```text
datasets/
  train/
    clip_001.lr.pt
    clip_001.hr.pt
```

Run training when PyTorch is installed:

```bash
python3 ml/train_tinysr.py --train --dataset datasets/train --epochs 5 --export ml/exports/nvc-tinysr-v0-trained.modl
```

`NVC-TinySR-v0` is a small x2 super-resolution CNN. The browser can run the embedded `MOD0` weights through WebGPU or a CPU fallback.

## Benchmarking

Run:

```bash
./zig-out/bin/nvc bench samples/input.mp4 --profiles w1,xc --frames 60 --model ml/exports/nvc-tinysr-v0-xiph-3gb.modl
python ml/benchmark.py samples/input.mp4 samples/output.nvc
```

The Zig benchmark command writes timestamped artifacts into `.nvc-bench/` and prints a CSV-style table with file size, bitrate, encode time, decode time, encode FPS, and decode FPS. The Python helper inspects an existing `.nvc` file and can run FFmpeg PSNR/SSIM if you pass a decoded MP4 with `--decoded`.

Important metrics:

- **VMAF**: visual quality score that often matches human judgment better than PSNR.
- **MS-SSIM**: structural similarity score.
- **PSNR**: simple pixel error score.
- **LPIPS**: neural perceptual difference score.
- **Decode FPS**: how fast playback runs.
- **File size**: how small the compressed file is.

## Common Errors

### `ffmpeg` not found

Install FFmpeg and make sure it is in your terminal path.

```bash
ffmpeg -version
```

### WebGPU unavailable

Use a modern Chrome or Edge build with WebGPU enabled for the fastest neural reconstruction. The demo still parses `.nvc` files and can fall back to the CPU `MOD0` path for small codec-base frames.

### Unsupported `.nvc` version

The alpha reader expects the `NVCF` magic and v0-style chunks.

### Corrupt chunks

Every chunk has a CRC32 checksum. If a file was cut off or edited incorrectly, `nvc info` may fail with a CRC error.

### Model mismatch

Future NVC files will include a model architecture version. The player must support that architecture.

## Roadmap

1. Alpha container and CLI.
2. Custom `BAS1` transform base codec.
3. Browser parser and player.
4. Custom `BAS2` predictive transform base codec.
5. Custom `BAS3` motion-compensated transform base codec.
6. Custom `BAS4` Huffman entropy-coded motion transform base codec.
7. `NVC-TinySR-v0` training and `MOD0` export.
8. `PRVW` instant browser preview stream with play/pause/seek.
9. WebGPU neural model path.
10. HTTP Range loading for URL preview playback.
11. `BAS5` packetized `BASE` GOP format.
12. Codec/Neural seek by BAS5 GOP range request.
13. Continuous Neural playback optimization from BAS5 packets.
14. Alpha `COL1` color restoration and `GRN1` grain synthesis.
15. Alpha `FET1` feature residual stream.
16. `NVC-XC` extreme compression mode.
17. Audio support later.

## FAQ

### Is this MP4?

No. NVC uses its own `.nvc` file type.

### Does it use AV1?

No for v0. FFmpeg may be used to read normal videos and write normal decoded output videos, but the `.nvc` file does not embed AV1.

### Can browsers play it natively?

No. Use the NVC web player.

### Is it lossless?

No. NVC targets visual accuracy, not exact pixel-perfect reconstruction.

### Why bundle model weights?

So every `.nvc` file can be self-contained. The player provides the NVC runtime, and the file carries the video-specific data it needs.
