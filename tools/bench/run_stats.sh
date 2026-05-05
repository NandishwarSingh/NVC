#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
  echo "usage: tools/bench/run_stats.sh <source-video> [--frames N] [--profiles w1,xc] [--model PATH] [--out-dir DIR] [--enhanced] [--rife]" >&2
  exit 2
fi
shift || true

FRAMES="60"
PROFILES="w1,xc"
MODEL=""
OUT_DIR=".nvc-stats"
ENHANCED="0"
RIFE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --frames)
      FRAMES="${2:?missing --frames value}"
      shift 2
      ;;
    --profiles)
      PROFILES="${2:?missing --profiles value}"
      shift 2
      ;;
    --model)
      MODEL="${2:?missing --model value}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?missing --out-dir value}"
      shift 2
      ;;
    --enhanced)
      ENHANCED="1"
      shift
      ;;
    --rife)
      RIFE="1"
      shift
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

NVC="${NVC_BIN:-./zig-out/bin/nvc}"
PYTHON="${PYTHON:-python3}"
CSV="$OUT_DIR/stats.csv"

if [ ! -f "$SOURCE" ]; then
  echo "source video not found: $SOURCE" >&2
  exit 2
fi

command -v ffprobe >/dev/null || { echo "ffprobe not found; install FFmpeg" >&2; exit 2; }
command -v ffmpeg >/dev/null || { echo "ffmpeg not found; install FFmpeg" >&2; exit 2; }

if [ ! -x "$NVC" ]; then
  echo "building nvc..."
  zig build
fi

mkdir -p "$OUT_DIR"

source_bytes="$(stat -f%z "$SOURCE" 2>/dev/null || stat -c%s "$SOURCE")"
duration="$(ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 "$SOURCE")"
width="$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of default=nk=1:nw=1 "$SOURCE" | head -n1)"
height="$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nk=1:nw=1 "$SOURCE" | head -n1)"
fps="$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nk=1:nw=1 "$SOURCE" | head -n1)"

if [ ! -f "$CSV" ]; then
  echo "source,source_bytes,duration_seconds,width,height,fps,profile,frames,nvc_path,nvc_bytes,decoded_path,decoded_bytes,compression_ratio,nvc_bitrate_kbps,hours_per_500mb,psnr_avg,ssim_all,enhanced,decode_flags" > "$CSV"
fi

IFS=',' read -r -a profile_list <<< "$PROFILES"
for profile in "${profile_list[@]}"; do
  profile="$(echo "$profile" | tr -d '[:space:]')"
  [ -n "$profile" ] || continue

  stem="$(basename "$SOURCE")"
  stem="${stem%.*}-$profile-$FRAMES"
  nvc_path="$OUT_DIR/$stem.nvc"
  decoded_path="$OUT_DIR/$stem.mp4"

  encode_args=("$NVC" encode "$SOURCE" "$nvc_path" --profile "$profile" --frames "$FRAMES")
  if [ -n "$MODEL" ]; then
    encode_args+=(--model "$MODEL")
  fi

  echo "encode ${profile} -> ${nvc_path}" >&2
  "${encode_args[@]}" >/dev/null

  decode_args=("$NVC" decode "$nvc_path" "$decoded_path")
  flags="plain"
  if [ "$ENHANCED" = "1" ]; then
    decode_args+=(--enhancer realesrgan)
    flags="realesrgan"
  fi
  if [ "$RIFE" = "1" ]; then
    decode_args+=(--interpolate-rife)
    flags="${flags}+rife"
  fi

  echo "decode ${profile} -> ${decoded_path}" >&2
  "${decode_args[@]}" >/dev/null

  nvc_bytes="$(stat -f%z "$nvc_path" 2>/dev/null || stat -c%s "$nvc_path")"
  decoded_bytes="$(stat -f%z "$decoded_path" 2>/dev/null || stat -c%s "$decoded_path")"

  metrics="$("$PYTHON" ml/benchmark.py "$SOURCE" "$nvc_path" --decoded "$decoded_path" 2>/dev/null || true)"
  psnr="$(printf '%s\n' "$metrics" | awk -F= '/^psnr_avg=/{print $2; exit}')"
  ssim="$(printf '%s\n' "$metrics" | awk -F= '/^ssim_all=/{print $2; exit}')"
  psnr="${psnr:-unavailable}"
  ssim="${ssim:-unavailable}"

  row="$("$PYTHON" - "$SOURCE" "$source_bytes" "$duration" "$width" "$height" "$fps" "$profile" "$FRAMES" "$nvc_path" "$nvc_bytes" "$decoded_path" "$decoded_bytes" "$psnr" "$ssim" "$ENHANCED" "$flags" <<'PY'
import csv
import sys

(
    source,
    source_bytes,
    duration,
    width,
    height,
    fps,
    profile,
    frames,
    nvc_path,
    nvc_bytes,
    decoded_path,
    decoded_bytes,
    psnr,
    ssim,
    enhanced,
    flags,
) = sys.argv[1:]

source_bytes_i = int(source_bytes)
nvc_bytes_i = int(nvc_bytes)
duration_f = float(duration or 0)
compression_ratio = (source_bytes_i / nvc_bytes_i) if nvc_bytes_i else 0
bitrate_kbps = (nvc_bytes_i * 8 / duration_f / 1000) if duration_f > 0 else 0
bytes_per_second = (nvc_bytes_i / duration_f) if duration_f > 0 else 0
hours_per_500mb = ((500 * 1024 * 1024) / bytes_per_second / 3600) if bytes_per_second > 0 else 0

writer = csv.writer(sys.stdout)
writer.writerow([
    source,
    source_bytes_i,
    f"{duration_f:.3f}",
    width,
    height,
    fps,
    profile,
    frames,
    nvc_path,
    nvc_bytes_i,
    decoded_path,
    int(decoded_bytes),
    f"{compression_ratio:.2f}",
    f"{bitrate_kbps:.2f}",
    f"{hours_per_500mb:.2f}",
    psnr,
    ssim,
    enhanced,
    flags,
])
PY
)"
  printf '%s\n' "$row" >> "$CSV"
done

echo "wrote $CSV"
