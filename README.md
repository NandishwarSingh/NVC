# NVC: Neural Video Codec

## By the numbers
<<<<<<< HEAD

Current alpha benchmark snapshots:

- **16 hours of 1024x576 natural video can fit in roughly 325 MB** at the measured XC rate of about 5.6 KB/s.
- **21x smaller than a source MP4** on the WhatsApp test clip: 6.78 MB source to 314 KB `.nvc`.
- **9x smaller W1 base stream** in the BAS5 to BAS6 transition benchmark, with VMAF around 85 on synthetic test content.
- **+10.6 VMAF from `--enhancer realesrgan`** on W1 natural content in the FourPeople benchmark.
- **12 fps base to 30 fps output** is supported with `--interpolate-rife` for smoother XC CLI decode.

See [STATS.md](STATS.md) for methodology, exact commands, and the reproducibility script.

NVC is an experimental custom video codec with its own `.nvc` file type. The goal is to compress video by storing a small low-resolution base stream plus neural data that helps reconstruct a visually accurate full-resolution video.

This repo is an alpha implementation. It already creates and reads native `.nvc` files, includes a Zig CLI, includes a browser demo, and lays down the format/spec structure for the neural codec work. New here? Start with [HOW_TO_SETUP_FROM_SCRATCH.md](HOW_TO_SETUP_FROM_SCRATCH.md).
=======

Current alpha benchmark snapshots:

- **16 hours of 1024x576 natural video can fit in roughly 325 MB** at the measured XC rate of about 5.6 KB/s.
- **21x smaller than a source MP4** on the WhatsApp test clip: 6.78 MB source to 314 KB `.nvc`.
- **9x smaller W1 base stream** in the BAS5 to BAS6 transition benchmark, with VMAF around 85 on synthetic test content.
- **+10.6 VMAF from `--enhancer realesrgan`** on W1 natural content in the FourPeople benchmark.
- **12 fps base to 30 fps output** is supported with `--interpolate-rife` for smoother XC CLI decode.

See [STATS.md](STATS.md) for methodology, exact commands, and the reproducibility script.

NVC is an experimental neural-augmented video codec with its own `.nvc` file type. The goal is to compress video by storing a downscaled VP9 base stream plus neural reconstruction data that helps recover a visually accurate full-resolution video.

This repo is an alpha implementation. It already creates and reads native `.nvc` files, includes a Zig CLI, includes a browser demo, and lays down the format/spec structure for the neural codec work. New here? Start with [HOW_TO_SETUP_FROM_SCRATCH.md](HOW_TO_SETUP_FROM_SCRATCH.md).

The current alpha base codec is `BAS6`: a libvpx-vp9 IVF bitstream embedded inside the `BASE` chunk, encoded in CRF mode at the profile's downscaled resolution and frame rate. Earlier alpha base codecs (`BAS0`–`BAS5`) remain readable for backwards compatibility.
>>>>>>> 4fa184d (Replace BAS5 with BAS6 VP9 base codec, add Real-ESRGAN and RIFE enhancers, per-clip distillation)

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
- `nvc encode`: working for `BAS6`, a libvpx-vp9 IVF bitstream stored inside the `BASE` chunk. Older `BAS0`–`BAS5` files remain readable.
- `nvc decode`: working for full reconstruction at source resolution.
- `nvc decode --enhancer realesrgan`: working when `realesrgan-ncnn-vulkan` is on `PATH`. Decoded base frames are upscaled with the bundled `realesr-animevideov3` model (Apple Silicon Vulkan/Metal supported via MoltenVK), then muxed with `libx264 -preset slow -crf 23` by default. Use `--crf 18` when you want a larger, higher-quality MP4.
- `nvc info`: working.
- `nvc inspect`: working.
- Browser NVC Studio app: working for normal video upload to downloadable `.nvc`, `.nvc` upload to downloadable MP4, local `.nvc` playback, metadata, chunk info, drag-and-drop, and range-loaded sample playback. Newer files use a single VP9 IVF base, decoded in the browser via WebCodecs `VideoDecoder`. Older `BAS0`–`BAS5` files still play through the existing custom decode path. `MOD0` TinySR neural reconstruction runs through WebGPU with a CPU fallback.
- `NVC-TinySR-v0` model export: working as a self-contained `MOD0` artifact.
- Optional PyTorch training path: working from normal source videos or paired `.pt` tensors.
- Feature residuals: working as compact alpha `FET1` luma correction vectors inside `FEAT`.
- `nvc bench`: working for encode/decode timing, `.nvc` size, approximate bitrate, encode FPS, and decode FPS. Perceptual metrics are still planned.

The alpha encoder uses FFmpeg to read normal video files and extract raw frames, then shells out to libvpx-vp9 (also via FFmpeg) to compress the downscaled base stream. The resulting VP9 bitstream lives inside the `.nvc` container; NVC does not embed MP4, AV1, H.264, or H.265.

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

By default, `nvc encode` encodes the full input video. Use `--frames 60` only when you want a short test encode.

Inspect the file:

```bash
./zig-out/bin/nvc info samples/output.nvc
./zig-out/bin/nvc inspect samples/output.nvc
```

Decode it back to a normal video:

```bash
./zig-out/bin/nvc decode samples/output.nvc samples/reconstructed.mp4
```

Optional: upscale decoded base frames with `realesrgan-ncnn-vulkan` before muxing the MP4. Install the binary first (download the macOS release zip from `https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases` and place it on your `PATH`, with the `models/` directory reachable via a wrapper that injects `-m`).

```bash
./zig-out/bin/nvc decode samples/output.nvc samples/reconstructed.mp4 \
  --enhancer realesrgan \
  --realesrgan-model realesr-animevideov3 \
  --crf 18
```

The enhancer picks `-s 2`, `-s 3`, or `-s 4` automatically based on the base-to-source ratio; any remaining factor is filled in with a final lanczos scale before muxing.

Run NVC Studio:

```bash
cd web
bun install
bun run dev
```

Open the URL printed by Bun. The web app can:

- Upload a normal video and download a `.nvc` file. The web app defaults to `NVC-XC`; leave frame limit blank to encode the full video.
- Upload a `.nvc` file and download a decoded MP4.
- Upload or drag a `.nvc` file and play it directly in Preview, Codec, or Neural mode.

The interface follows the ElevenLabs UI style of code-owned, shadcn-like primitives: restrained cards, compact controls, status surfaces, and editable local components.

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

The alpha W1 encoder downscales to a half-resolution base stream, caps the coded stream at 30 fps, and encodes the base with libvpx-vp9 at CRF 36. A 1920x1080 video becomes a 960x540 VP9 base stream.

### NVC-XC

Use this for smaller files when slower encoding is okay.

```bash
./zig-out/bin/nvc encode input.mp4 output.nvc --profile xc
```

The alpha XC encoder downscales to a one-quarter-resolution base stream, caps the coded stream at 12 fps, encodes the base with libvpx-vp9 at CRF 44, and stores only a tiny sampled preview. A 1920x1080 video becomes a 480x270 VP9 base stream. This is much smaller than W1, but it is more lossy and depends more on neural reconstruction.

## Project Structure

```text
core/    Zig codec library and CLI
web/     Bun + TypeScript NVC Studio web app and browser player
ml/      Python model training and benchmark tools
spec/    NVC file format and profile docs
samples/ sample files and fixtures
```

## How Playback Works

The final NVC playback pipeline is:

1. Parse the `.nvc` file.
2. Decode the low-resolution VP9 `BASE` stream (browser via WebCodecs `VideoDecoder`, CLI via libvpx-vp9 through FFmpeg).
3. Load the bundled neural model from `MODL`.
4. Apply motion/context data from `MOTN`.
5. Apply feature residuals from `FEAT`.
6. Restore color using `COLR`.
7. Add synthetic grain using `GRAN`.
8. Draw the final frame to a canvas.

NVC Studio adds browser upload/download workflows on top of the player. The Bun server accepts video uploads at `/api/encode`, calls the Zig CLI, and returns a `.nvc` download. It accepts `.nvc` uploads at `/api/decode`, calls the Zig decoder, and returns a decoded MP4 download. The playback surface parses `.nvc` files in the browser and uses compact sampled `PVW2` data inside `PRVW` for instant preview playback with play, pause, and seek controls. `PVW2` keeps the real source duration but stores roughly one tiny RGB preview frame per second, capped by profile, so the preview track does not dominate XC files. For new `BAS6` files the browser eagerly demuxes the embedded VP9 IVF and decodes every frame through WebCodecs `VideoDecoder` into an ImageData cache once the file loads, so Preview, Codec, and Neural modes can seek synchronously from there. Older `BAS0`–`BAS5` files keep the existing custom decode path with `PVW2` previews, BAS5 GOP packet range fetches, and packet byte caching. Neural mode runs TinySR, guides luma/detail from the decoded codec base so it does not throw away VP9 detail, then applies `FET1` luma feature residuals, `COL1` color tuning, and `GRN1` deterministic grain. It can run the embedded `MOD0` TinySR model through WebGPU, with a smaller CPU fallback if WebGPU is unavailable or too slow.

## How Compression Works

The intended NVC compression pipeline is:

1. Decode the source video.
2. Convert frames to YUV.
3. Downscale frames to the profile's base resolution and frame rate.
4. Remove or simplify grain.
5. Encode the low-resolution base stream with libvpx-vp9 in CRF mode (`BAS6`).
6. Store neural latents and feature residuals.
7. Bundle neural model weights.
8. Save everything into one `.nvc` file.

The alpha implementation currently does steps 1, 2, 3, 5, 6, 7, and 8, plus source/base-stat `COL1` color tuning and basic `GRN1` grain synthesis. `FEAT` currently stores `FET1`, a compact per-frame tile luma correction stream. Step 4 is still basic.

The earlier alpha base codecs (`BAS0`–`BAS5`) implemented progressively richer custom transform-and-entropy stacks (RLE, 4x4 Hadamard, predictive, motion-compensated, Huffman, then packetized GOPs). They are still readable, but a benchmark across five synthetic content types showed plain libvpx-vp9 produced 3–13x smaller `BASE` chunks at comparable or better VMAF than `BAS5`, so new encodes always write `BAS6`. The custom-codec work remains useful as a reference and as a fallback if a future profile needs container-only decoding without libvpx.

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

Done in alpha:

1. Alpha container and CLI.
2. Custom `BAS1`–`BAS5` transform/entropy/motion base codecs (kept readable for old files).
3. Browser parser and player.
4. `NVC-TinySR-v0` training and `MOD0` export.
5. `PRVW` instant browser preview stream with play/pause/seek.
6. WebGPU neural model path.
7. HTTP Range loading for URL preview playback.
8. Alpha `COL1` color restoration and `GRN1` grain synthesis.
9. Alpha `FET1` feature residual stream.
10. `NVC-XC` extreme compression mode.
11. `BAS6` libvpx-vp9 base codec via FFmpeg subprocess (current default).
12. Browser WebCodecs `VideoDecoder` path for `BAS6`.
13. `nvc decode --enhancer realesrgan` post-decode upscale.
14. Better configurable `libx264` CRF MP4 mux on `nvc decode`.

Planned:

15. Continuous Neural playback optimization from VP9 base frames.
16. Tighter VP9 rate control per profile (target-bitrate mode in addition to CRF).
17. Replace `realesr-animevideov3` with a smaller WebGPU ONNX model for in-browser SR.
18. Audio support.

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
