# NVC Stats

This file collects the headline numbers used in the README and web app. NVC is alpha software, so treat these as benchmark snapshots, not permanent promises. When the codec changes, rerun the scripts and update this file.

## Headline Claims

| Claim | Measurement | Source / math |
|---|---:|---|
| 16 hours of 1024x576 natural video can fit in roughly 325 MB | 5.6 KB/s measured XC rate | `16 h * 3600 s * 5.6 KB/s = 322,560 KB`, rounded to about 325 MB |
| 21x smaller than source MP4 | 6.78 MB to 314 KB | `6.78 MB / 0.314 MB = 21.6x` |
| 9x smaller W1 base stream after BAS6 | 1.59 MB BAS5 to 174 KB BAS6 W1 | BAS5 to BAS6 transition benchmark |
| +10.6 VMAF from Real-ESRGAN on W1 | 80.77 plain to 91.40 enhanced | FourPeople natural-content benchmark |
| 12 fps base to 30 fps output | XC coded fps to source-style output fps | `nvc decode --interpolate-rife` |
| Self-contained `.nvc` files | `MODL` plus `BASE` plus side chunks in one file | Container design in `spec/nvc-v0.md` |

## Reproduce Local Stats

Run the project benchmark on any video:

```bash
tools/bench/run_stats.sh samples/input.mp4
```

The script writes a CSV file at:

```text
.nvc-stats/stats.csv
```

Each row includes:

- Source file size, duration, dimensions, and fps.
- Profile (`w1` or `xc`).
- Encoded `.nvc` file size.
- Compression ratio versus the source file.
- Approximate `.nvc` bitrate.
- Hours of video that would fit in 500 MB at that measured rate.
- Decode MP4 size.
- PSNR and SSIM when FFmpeg can compute them.

Run a longer benchmark:

```bash
tools/bench/run_stats.sh path/to/source.mp4 --frames 900 --profiles w1,xc
```

Run with a specific model:

```bash
tools/bench/run_stats.sh path/to/source.mp4 \
  --model ml/exports/nvc-tinysr-v0-xiph-3gb.modl
```

Run an enhanced decode pass too:

```bash
tools/bench/run_stats.sh path/to/source.mp4 --enhanced
```

Enhanced mode needs `realesrgan-ncnn-vulkan` on `PATH`. Add RIFE smoothing:

```bash
tools/bench/run_stats.sh path/to/source.mp4 --enhanced --rife
```

## CSV Columns

```text
source,source_bytes,duration_seconds,width,height,fps,profile,frames,nvc_path,nvc_bytes,decoded_path,decoded_bytes,compression_ratio,nvc_bitrate_kbps,hours_per_500mb,psnr_avg,ssim_all,enhanced,decode_flags
```

## Methodology

1. Build the Zig CLI with `zig build`.
2. Encode the source with `nvc encode`.
3. Decode the `.nvc` file with `nvc decode`.
4. Inspect `.nvc` metadata with `nvc info`.
5. Compare source and decoded MP4 with `ml/benchmark.py`.
6. Save one CSV row per profile.

The script uses FFmpeg/FFprobe for video metadata and PSNR/SSIM. VMAF and LPIPS are not built into the lightweight script yet; when used, record the exact FFmpeg/libvmaf command and model version beside the result.

## Updating This File

When benchmark numbers change:

1. Rerun `tools/bench/run_stats.sh`.
2. Copy the strongest rows into this file.
3. Update the README `By the numbers` section.
4. Update the stats panel in `web/index.html` if the public claims changed.
5. Keep every claim tied to a command or math line.
