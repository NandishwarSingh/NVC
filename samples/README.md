# Samples

Sample media files are ignored by git because videos and `.nvc` outputs can get large.

Create a tiny input clip with:

```bash
ffmpeg -y -f lavfi -i testsrc2=size=1920x1080:rate=30 -t 2 samples/input.mp4
```

Then encode:

```bash
./zig-out/bin/nvc encode samples/input.mp4 samples/output.nvc --profile w1
```
