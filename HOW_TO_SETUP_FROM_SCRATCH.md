# How To Set Up NVC From Scratch

This guide assumes you are starting fresh.

## What You'll Have At The End

You will have the NVC CLI built, a `.nvc` file created from a normal video, a decoded MP4, and the local web app running. You will also know how to run the first benchmark and where the training tools live.

## Prerequisites By OS

NVC needs four tools:

- Zig `0.15.2+`
- Bun `1.3+`
- Python `3.10+`
- FFmpeg

Optional tools:

- `realesrgan-ncnn-vulkan` for high-quality CLI upscaling.
- `rife-ncnn-vulkan` for frame interpolation.

### macOS

```bash
brew install zig bun python ffmpeg
zig version
bun --version
python3 --version
ffmpeg -version
```

Expected output starts like:

```text
0.15.2
1.3.x
Python 3.10.x
ffmpeg version ...
```

### Linux

```bash
sudo apt update
sudo apt install -y python3 python3-venv ffmpeg curl unzip
curl -fsSL https://bun.sh/install | bash
zig version
bun --version
python3 --version
ffmpeg -version
```

If `zig` is not available through your package manager, download it from:

```text
https://ziglang.org/download/
```

### Windows

Use PowerShell:

```powershell
winget install Oven-sh.Bun
winget install Python.Python.3.12
winget install Gyan.FFmpeg
```

Install Zig from:

```text
https://ziglang.org/download/
```

Then check:

```powershell
zig version
bun --version
python --version
ffmpeg -version
```

## Get The Code

```bash
git clone https://github.com/NandishwarSingh/NVC.git
cd NVC
```

If you already have the repo:

```bash
git pull
```

## Build The CLI

```bash
zig build
```

Expected result:

```text
zig-out/bin/nvc
```

Check the CLI:

```bash
./zig-out/bin/nvc help
```

You should see commands like:

```text
nvc encode
nvc decode
nvc info
nvc inspect
nvc bench
```

## First Encode/Decode

### Create A Tiny Test Video

```bash
ffmpeg -y -f lavfi -i testsrc2=size=1280x720:rate=30 -t 2 sample.mp4
```

Expected result:

```text
sample.mp4
```

### Encode A Video To `.nvc`

```bash
./zig-out/bin/nvc encode sample.mp4 sample.nvc --profile w1
```

Expected output includes:

```text
encoded sample.mp4 -> sample.nvc
profile=NVC-W1
```

### Inspect The `.nvc` File

```bash
./zig-out/bin/nvc info sample.nvc
./zig-out/bin/nvc inspect sample.nvc
```

You should see metadata like:

```text
profile=NVC-W1
model_format=MOD0
feature_format=FET1
chunks: 13
```

### Decode It Back To MP4

```bash
./zig-out/bin/nvc decode sample.nvc sample-decoded.mp4
```

Expected result:

```text
sample-decoded.mp4
```

Open it with your normal video player.

## Decode With Quality Enhancers

Install `realesrgan-ncnn-vulkan` from:

```text
https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases
```

Then run:

```bash
./zig-out/bin/nvc decode sample.nvc sample-hq.mp4 --enhancer realesrgan
```

This decodes the low-resolution base frames, upscales them with Real-ESRGAN, then writes a normal MP4.

Install `rife-ncnn-vulkan` from:

```text
https://github.com/nihui/rife-ncnn-vulkan/releases
```

Then run:

```bash
./zig-out/bin/nvc decode sample.nvc sample-smooth.mp4 --enhancer realesrgan --interpolate-rife
```

This adds frame interpolation. It is most useful for `NVC-XC`, where the base stream is capped at 12 fps.

## Run The Web Player

```bash
cd web
bun install
bun run dev
```

Expected output:

```text
NVC web player: http://localhost:5173
```

Open that URL. You can:

- Upload a normal video and download a `.nvc`.
- Upload a `.nvc` and download an MP4.
- Drop a `.nvc` onto the canvas and play it.
- Switch between Preview, Codec, and Neural modes.

## Set Up Python And Per-Clip Distillation

Go back to the repo root:

```bash
cd ..
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install PyTorch:

```bash
pip install torch torchvision
```

Export the bootstrap model:

```bash
python3 ml/train_tinysr.py --export ml/exports/nvc-tinysr-v0.modl
```

Train a tiny model from one video:

```bash
python3 ml/train_tinysr.py \
  --train \
  --source-video sample.mp4 \
  --max-frames 16 \
  --epochs 3 \
  --train-width 256 \
  --train-height 144 \
  --batch-size 4 \
  --device cpu \
  --export ml/exports/nvc-tinysr-v0-trained.modl
```

### Per-Clip Distillation

Per-clip distillation trains a tiny `MOD0` model on the exact degradation pattern of one clip.

```bash
tools/distill.sh sample.mp4 sample-distilled.nvc --profile w1 --epochs 30
```

What happens:

1. NVC encodes the source once.
2. The script extracts the compressed base stream.
3. TinySR trains on pairs of low-resolution codec frames and original frames.
4. NVC re-encodes the file with the distilled model.

## Benchmarking

Run the normal benchmark:

```bash
./zig-out/bin/nvc bench sample.mp4 --profiles w1,xc --frames 60
```

Run the stats CSV script:

```bash
tools/bench/run_stats.sh sample.mp4
```

Read the results:

```bash
cat .nvc-stats/stats.csv
```

## Common Errors And Fixes

### `ffmpeg not found`

Install FFmpeg and check:

```bash
ffmpeg -version
```

### `realesrgan-ncnn-vulkan: command not found`

Download the release zip, unzip it, and put the binary on your `PATH`.

You can also pass the path directly:

```bash
./zig-out/bin/nvc decode sample.nvc sample-hq.mp4 \
  --enhancer realesrgan \
  --realesrgan-bin /path/to/realesrgan-ncnn-vulkan
```

### Real-ESRGAN models are missing

Some release zips do not include the model directory you need. Keep the `models/` folder next to the binary, or use a wrapper script that passes `-m /path/to/models`.

### `WebGPU unavailable`

Use a modern Chromium-based browser. The player can still fall back to CPU TinySR for small frames, but it will be slower.

### `Unsupported .nvc version`

Your file may come from an older or newer alpha format. Run:

```bash
./zig-out/bin/nvc info file.nvc
```

### `Corrupt chunks`

Every chunk has a CRC. If a file was cut off during upload/download, NVC will reject it.

### Paths with spaces fail

Quote paths:

```bash
./zig-out/bin/nvc encode "my video.mp4" "my output.nvc" --profile w1
```

### VP9 WebCodecs error in browser

Try Chrome or Edge first. Safari support can vary by version.

## Where To Go Next

- Read [README.md](README.md) for the project overview.
- Read [STATS.md](STATS.md) for benchmark claims and reproduction commands.
- Read [spec/nvc-v0.md](spec/nvc-v0.md) for the container format.
- Look at open GitHub issues for the roadmap.
