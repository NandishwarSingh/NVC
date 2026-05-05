# NVC v0 Format

NVC is a custom neural video codec container with the `.nvc` file extension. Version `0.1` is an alpha format used to stabilize the file structure, CLI, and browser player before the neural model and feature residual streams become production quality.

## Design Goals

- Use a native `.nvc` file type from day one.
- Keep every `.nvc` self-contained.
- Avoid embedding AV1, H.264, H.265, VP9, or MP4 video streams inside `.nvc` v0.
- Support two public profiles:
  - `NVC-W1`: realtime WebGPU playback target.
  - `NVC-XC`: slower encode, higher compression target.
- Use alpha neural reconstruction chunks now: `MOD0` model data, `FET1` feature residuals, `COL1` color tuning, and `GRN1` grain synthesis.

## File Header

All integers are little-endian.

```text
offset  size  name
0       4     magic = "NVCF"
4       2     major version
6       2     minor version
8       4     header length, currently 20
12      8     file flags, currently 0
```

## Chunk Header

Every chunk uses the same fixed 20-byte header.

```text
offset  size  name
0       4     chunk id
4       8     payload byte length
12      4     CRC32 of payload
16      4     chunk flags
20      N     payload
```

## Standard Chunks

- `HEAD`: text metadata for profile, source dimensions, base dimensions, fps, frame count, and alpha status.
- `TOC0`: streamable table of contents for later chunks.
- `PRVW`: compact browser preview stream for instant playback while the full `BASE` stream decoder initializes.
- `MODL`: bundled neural model data. Current alpha supports `MOD0`.
- `BASE`: NVC-native low-resolution base stream.
- `MOTN`: motion/context latent stream.
- `FEAT`: feature residual latent stream. Current alpha supports `FET1`.
- `GRAN`: deterministic grain synthesis parameters. Current alpha supports `GRN1`.
- `COLR`: color restoration parameters. Current alpha supports `COL1`.
- `ENTR`: entropy tables.
- `SEEK`: seek metadata.
- `META`: encoder metadata.
- `AUDI`: reserved for future audio support.

## MOD0 Model Artifact

`MOD0` is the current self-contained model artifact stored inside the `MODL` chunk.

```text
offset  size  name
0       4     magic = "MOD0"
4       2     major version
6       2     minor version
8       4     metadata JSON byte count
12      8     weight byte count
20      M     UTF-8 JSON metadata
20+M    N     float32 little-endian weights
```

The first model is `NVC-TinySR-v0`, a small x2 CNN with two ReLU convolution layers, a final convolution, and pixel shuffle. The exporter can write deterministic nearest-neighbor-like bootstrap weights or train from source-video frames with PyTorch; either artifact uses the same self-contained `MOD0` layout.

## PRVW Preview Chunk

`PRVW` is a small streamable preview chunk used by the browser player for immediate playback, seeking, and UI smoke tests. It is not a replacement for the `BASE` stream.

Current alpha files write `PVW1`, an RGB888 preview-video stream. The preview is intentionally tiny and decoder-friendly; production NVC playback still uses `BASE`, `MODL`, and later latent chunks.

```text
offset  size  name
0       4     magic = "PVW1"
4       2     version = 1
6       2     flags = 0
8       4     preview width
12      4     preview height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      4     RGB bytes per frame
32      N     packed RGB888 frames
```

Older alpha files may contain `PVW0`, a single still-frame preview:

```text
offset  size  name
0       4     magic = "PVW0"
4       2     version = 1
6       2     flags = 0
8       4     preview width
12      4     preview height
16      4     RGB byte count
20      N     packed RGB888 pixels
```

## COL1 Color Chunk

`COL1` is the first alpha color restoration payload inside `COLR`. The current encoder estimates luma bias, contrast, and saturation from source base frames versus decoded lossy base frames, and the browser applies the transform during Neural reconstruction.

```text
offset  size  name
0       4     magic = "COL1"
4       2     version = 1
6       2     flags = 0
8       4     float32 luma scale
12      4     float32 luma bias
16      4     float32 saturation scale
20      4     float32 contrast scale
```

## GRN1 Grain Chunk

`GRN1` is the first deterministic grain synthesis payload inside `GRAN`. Grain is generated from the seed, pixel coordinate, and frame index so every decoder can recreate the same texture without storing grain pixels.

```text
offset  size  name
0       4     magic = "GRN1"
4       2     version = 1
6       2     flags = 0
8       4     uint32 seed
12      4     float32 grain intensity
16      4     luma-only flag
```

## FET1 Feature Residual Chunk

`FET1` is the first alpha feature residual payload inside `FEAT`. It stores one entropy-coded signed luma correction per frame tile. The encoder computes each correction from the source base frame minus the decoded lossy base frame, quantizes it, and the browser applies it after neural reconstruction. This gives Neural mode compact per-video correction data without storing full residual images.

```text
offset  size  name
0       4     magic = "FET1"
4       2     version = 1
6       2     flags = 0
8       4     base width
12      4     base height
16      4     frame count
20      4     tile size
24      4     luma residual quantization step
28      4     tile grid width
32      4     tile grid height
36      8     raw residual byte count
44      8     coded residual byte count
52      N     `HUF0` entropy-coded residual bytes
```

Residual bytes are stored as `signed_value + 128`. The reconstructed luma delta is:

```text
delta = (byte - 128) * quant_step
```

## Alpha BASE Streams

The current alpha encoder writes `BAS5`, a packetized custom Huffman entropy-coded motion-compensated tiled transform stream. The decoder still accepts older `BAS0`, `BAS1`, `BAS2`, `BAS3`, and `BAS4` files created by earlier milestones.

### BAS0 Legacy RLE Stream

```text
offset  size  name
0       4     magic = "BAS0"
4       2     version
6       2     flags
8       4     base width
12      4     base height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      8     raw YUV420 byte count
36      8     RLE byte count
44      N     RLE byte pairs: count, value
```

### BAS1 Legacy Tiled Transform Stream

`BAS1` stores each YUV420 plane as 4x4 blocks. Each block is centered around 128, transformed with a 4x4 integer Hadamard transform, quantized, zigzag scanned, then entropy-coded with zero-run and signed varint tokens.

```text
offset  size  name
0       4     magic = "BAS1"
4       2     version = 1
6       2     flags
8       4     base width
12      4     base height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      8     raw YUV420 byte count
36      8     coded byte count
44      4     block size, currently 4
48      4     luma quantization step
52      4     chroma quantization step
56      N     coefficient token stream
```

Coefficient token stream:

- `0, count`: a zero run of `count` coefficients.
- `zigzag_signed(coeff) + 1`: one non-zero coefficient.

This is still an alpha base codec. It gives NVC a real lossy transform path, but future versions need prediction, motion compensation, stronger entropy coding, and neural reconstruction before it becomes competitive with established codecs.

### BAS2 Predictive Tiled Transform Stream

`BAS2` predicts every 4x4 block from already reconstructed boundary pixels, transforms only the residual, then uses the same quantized coefficient token stream as `BAS1`.

```text
offset  size  name
0       4     magic = "BAS2"
4       2     version = 2
6       2     flags
8       4     base width
12      4     base height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      8     raw YUV420 byte count
36      8     coded byte count
44      4     block size, currently 4
48      4     luma quantization step
52      4     chroma quantization step
56      4     predictor id, currently 1
60      N     predictive residual coefficient token stream
```

Predictor `1` uses the reconstructed block-left column and block-top row:

- top-left block: predict 128.
- first row: predict from the left block boundary.
- first column: predict from the top block boundary.
- all other blocks: average left boundary and top boundary samples.

### BAS3 Motion-Compensated Tiled Transform Stream

`BAS3` keeps the `BAS2` spatial predictor and adds a small previous-frame motion candidate set. Motion modes are stored in an RLE-compressed substream, while residual coefficients are stored in a separate zero-run/signed-varint substream.

```text
offset  size  name
0       4     magic = "BAS3"
4       2     version = 3
6       2     flags
8       4     base width
12      4     base height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      8     raw YUV420 byte count
36      8     coded byte count
44      4     block size, currently 4
48      4     luma quantization step
52      4     chroma quantization step
56      4     predictor id, currently 2
60      4     motion mode count, currently 6
64      8     RLE-compressed mode stream byte count
72      M     RLE-compressed mode stream
72+M    N     residual coefficient token stream
```

Motion modes:

- `0`: spatial predictor from reconstructed top/left boundaries.
- `1`: previous frame, same pixel position.
- `2`: previous frame, 4 pixels left.
- `3`: previous frame, 4 pixels right.
- `4`: previous frame, 4 pixels up.
- `5`: previous frame, 4 pixels down.

### BAS4 Huffman-Coded Motion Transform Stream

`BAS4` wraps the packed `BAS3` motion/residual payload in a static canonical Huffman byte entropy stream. It is still accepted by the decoder, but current alpha files use packetized `BAS5`.

```text
offset  size  name
0       4     magic = "BAS4"
4       2     version = 4
6       2     flags
8       4     base width
12      4     base height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      8     raw YUV420 byte count
36      8     coded byte count
44      4     block size, currently 4
48      4     luma quantization step
52      4     chroma quantization step
56      4     predictor id, currently 3
60      4     motion mode count, currently 6
64      4     entropy id, currently 1
68      N     HUF0 entropy stream
```

`HUF0` payload:

```text
offset  size  name
0       4     magic = "HUF0"
4       8     decoded byte count
12      256   canonical Huffman code lengths, one byte per symbol
268     N     packed Huffman bits
```

### BAS5 Packetized Huffman Motion Transform Stream

`BAS5` splits the base stream into independently decodable GOP packets. Each packet resets motion prediction at the packet boundary, then stores a BAS4-style `HUF0` entropy stream for that GOP. This lets the web player range-load only the GOP needed for a seek instead of fetching the whole `BASE` chunk.

```text
offset  size  name
0       4     magic = "BAS5"
4       2     version = 5
6       2     flags
8       4     base width
12      4     base height
16      4     fps numerator
20      4     fps denominator
24      4     frame count
28      8     raw YUV420 byte count
36      8     total coded packet byte count
44      4     block size, currently 4
48      4     luma quantization step
52      4     chroma quantization step
56      4     predictor id, currently 4
60      4     motion mode count, currently 6
64      4     entropy id, currently 1
68      4     GOP size
72      4     packet count
76      T     packet table, 24 bytes per packet
76+T    N     concatenated packet payloads
```

Each packet table entry:

```text
offset  size  name
0       4     start frame
4       4     packet frame count
8       8     packet payload offset from start of BAS5 payload
16      8     packet payload byte count
```

## Profiles

### NVC-W1

The web profile targets realtime playback. In alpha, 1080p sources are downscaled to a 540p base stream.

### NVC-XC

The extreme compression profile allows slower encoding and lower base resolutions. In alpha, it uses a quarter-resolution base stream.

## Future v0 Milestones

- Improve `BAS5` motion search with variable vectors and sub-block decisions.
- Replace byte-level Huffman with rANS/range coding if it beats `HUF0` on real clips.
- Make `NVC-TinySR-v0` use full-resolution codec base packets for all playback frames.
- Improve `FET1` from luma tile residuals into richer learned feature residuals.
- Tune `GRN1` from source/base statistics instead of fixed alpha defaults.
- Extend packetized `BASE` GOP range loading into continuous playback buffering.
