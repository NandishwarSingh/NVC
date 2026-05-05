#!/usr/bin/env python3
"""Extract the BAS6 VP9 IVF bitstream from a .nvc file.

Used by the per-clip distillation pipeline so the trainer can decode the
realistic codec-degraded base via FFmpeg without going through the full
nvc decode -> lanczos-upscale -> mp4 path.

Usage:
    python3 ml/extract_base.py input.nvc output.ivf
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


BAS6_HEADER_LEN = 56


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract BAS6 VP9 IVF from an .nvc file")
    parser.add_argument("nvc", type=Path)
    parser.add_argument("out_ivf", type=Path)
    args = parser.parse_args()

    data = args.nvc.read_bytes()
    if len(data) < 20 or data[:4] != b"NVCF":
        print(f"not an .nvc file: {args.nvc}", file=sys.stderr)
        return 2

    file_header_len = struct.unpack("<I", data[8:12])[0]
    offset = file_header_len
    while offset + 20 <= len(data):
        chunk_id = data[offset:offset + 4]
        payload_len = struct.unpack("<Q", data[offset + 4:offset + 12])[0]
        payload_start = offset + 20
        payload_end = payload_start + payload_len
        if payload_end > len(data):
            print(f"truncated chunk at offset {offset}", file=sys.stderr)
            return 2
        if chunk_id == b"BASE":
            payload = data[payload_start:payload_end]
            if payload[:4] != b"BAS6":
                print(f"BASE chunk magic is {payload[:4]!r}, expected BAS6", file=sys.stderr)
                return 2
            coded_size = struct.unpack("<Q", payload[36:44])[0]
            ivf = payload[BAS6_HEADER_LEN:BAS6_HEADER_LEN + coded_size]
            args.out_ivf.write_bytes(ivf)
            print(f"wrote {args.out_ivf} ({len(ivf):,} bytes)")
            return 0
        offset = payload_end
    print("BASE chunk not found", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
