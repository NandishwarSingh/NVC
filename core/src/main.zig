const std = @import("std");

const allocator = std.heap.page_allocator;

const file_magic = "NVCF";
const file_header_len = 20;
const chunk_header_len = 20;
const base0_header_len = 44;
const base1_header_len = 56;
const base2_header_len = 60;
const base3_header_len = 64;
const base4_header_len = 68;
const base5_header_len = 76;
const base5_packet_entry_len = 24;
const feat1_header_len = 52;
const transform_block_size = 4;
const motion_mode_count = 6;
const range_total = 4096;
const range_top = 1 << 24;
const max_file_bytes = 1024 * 1024 * 1024;

const Profile = enum {
    w1,
    xc,

    fn label(self: Profile) []const u8 {
        return switch (self) {
            .w1 => "NVC-W1",
            .xc => "NVC-XC",
        };
    }
};

const VideoProbe = struct {
    width: u32 = 1920,
    height: u32 = 1080,
    fps_num: u32 = 30,
    fps_den: u32 = 1,
    duration_ms: u64 = 0,
};

const BaseFormat = enum {
    rle,
    transform,
    predictive_transform,
    motion_transform,
    entropy_motion_transform,
    packetized_entropy_motion_transform,
};

const BaseInfo = struct {
    format: BaseFormat,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    raw_size: u64,
    coded_size: u64,
    block_size: u32 = transform_block_size,
    y_quant: u32 = 1,
    uv_quant: u32 = 1,
    predictor: u32 = 0,
    motion_modes: u32 = 0,
    entropy: u32 = 0,
    gop_size: u32 = 0,
    packet_count: u32 = 0,
    coded: []const u8,
};

const BaseBuild = struct {
    payload: []u8,
    coded_size: usize,
    gop_size: u32,
    packet_count: u32,
};

const BasePacketBuild = struct {
    start_frame: u32,
    frame_count: u32,
    coded: []u8,
};

const FeatureInfo = struct {
    width: u32,
    height: u32,
    frame_count: u32,
    tile_size: u32,
    quant_step: u32,
    grid_width: u32,
    grid_height: u32,
    residual_count: u64,
    coded_size: u64,
    coded: []const u8,
};

const Chunk = struct {
    id: [4]u8,
    payload: []const u8,
    flags: u32 = 0,
};

const ParsedChunk = struct {
    id: [4]u8,
    payload: []const u8,
    offset: usize,
    crc: u32,
    flags: u32,
};

const ParsedNvc = struct {
    data: []u8,
    chunks: []ParsedChunk,

    fn deinit(self: ParsedNvc) void {
        allocator.free(self.chunks);
        allocator.free(self.data);
    }

    fn find(self: ParsedNvc, id: []const u8) ?ParsedChunk {
        for (self.chunks) |chunk| {
            if (std.mem.eql(u8, chunk.id[0..], id)) return chunk;
        }
        return null;
    }
};

pub fn main() !void {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printHelp();
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "encode")) {
        try cmdEncode(args[2..]);
    } else if (std.mem.eql(u8, cmd, "decode")) {
        try cmdDecode(args[2..]);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try cmdInfo(args[2..], false);
    } else if (std.mem.eql(u8, cmd, "inspect")) {
        try cmdInfo(args[2..], true);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        try cmdBench(args[0], args[2..]);
    } else if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        printHelp();
    } else {
        std.debug.print("unknown command: {s}\n\n", .{cmd});
        printHelp();
        return error.InvalidCommand;
    }
}

fn printHelp() void {
    std.debug.print(
        \\NVC alpha CLI
        \\
        \\Usage:
        \\  nvc encode <input-video> <output.nvc> --profile w1|xc [--frames N|all] [--model PATH]
        \\  nvc decode <input.nvc> <output.mp4>
        \\  nvc info <input.nvc>
        \\  nvc inspect <input.nvc>
        \\  nvc bench <input-video> --profiles w1,xc [--frames N] [--model PATH] [--out-dir DIR]
        \\
        \\Alpha note:
        \\  This build implements the native .nvc container and a custom tiled transform BASE stream.
        \\  Neural reconstruction chunks include MOD0 model data plus alpha FET1/COL1/GRN1 side data.
        \\
    , .{});
}

fn cmdEncode(args: []const []const u8) !void {
    if (args.len < 2) {
        std.debug.print("encode needs <input-video> <output.nvc>\n", .{});
        return error.InvalidArguments;
    }

    const input = args[0];
    const output = args[1];
    var profile: Profile = .w1;
    var frame_limit: u32 = 0;
    var model_path: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--profile")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            profile = try parseProfile(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--frames")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            frame_limit = try parseFrameLimit(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--model")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            model_path = args[i + 1];
            i += 1;
        } else {
            std.debug.print("unknown encode option: {s}\n", .{args[i]});
            return error.InvalidArguments;
        }
    }

    const probe = probeVideo(input) catch |err| blk: {
        std.debug.print("warning: ffprobe failed ({s}); using 1920x1080/30 defaults\n", .{@errorName(err)});
        break :blk VideoProbe{};
    };

    const base_dims = chooseBaseDimensions(profile, probe.width, probe.height);
    const temp_raw = try std.fmt.allocPrint(allocator, "{s}.nvc-base.tmp.yuv", .{output});
    defer allocator.free(temp_raw);
    defer std.fs.cwd().deleteFile(temp_raw) catch {};

    try extractBaseRaw(input, temp_raw, base_dims.width, base_dims.height, probe.fps_num, probe.fps_den, frame_limit);
    const raw = try std.fs.cwd().readFileAlloc(allocator, temp_raw, max_file_bytes);
    defer allocator.free(raw);

    const frame_size = yuv420FrameSize(base_dims.width, base_dims.height);
    if (frame_size == 0 or raw.len < frame_size) {
        std.debug.print("ffmpeg produced no raw frames for {s}\n", .{input});
        return error.NoFrames;
    }

    const frame_count: u32 = @intCast(raw.len / frame_size);
    const trimmed_raw = raw[0 .. @as(usize, frame_count) * frame_size];
    const base_quant = chooseBaseQuant(profile);
    const base_build = try buildBasePayload(trimmed_raw, base_dims.width, base_dims.height, probe.fps_num, probe.fps_den, frame_count, base_quant.y, base_quant.uv, chooseBaseGopSize(profile));
    defer allocator.free(base_build.payload);
    const decoded_base = try decodeBasePayload(try parseBasePayload(base_build.payload));
    defer allocator.free(decoded_base);

    const head = try buildHeadPayload(profile, probe, base_dims.width, base_dims.height, frame_count);
    defer allocator.free(head);
    const prvw = try buildPreviewPayload(trimmed_raw, base_dims.width, base_dims.height, frame_count, probe.fps_num, probe.fps_den, 160);
    defer allocator.free(prvw);
    const modl = try loadModelPayload(model_path);
    defer allocator.free(modl);
    const motn = "motion/context latent stream placeholder\n";
    const feat = try buildFeaturePayload(trimmed_raw, decoded_base, base_dims.width, base_dims.height, frame_count, profile);
    defer allocator.free(feat);
    const gran = try buildGrainPayload(profile);
    defer allocator.free(gran);
    const colr = try buildColorPayload(trimmed_raw, decoded_base, base_dims.width, base_dims.height, frame_count);
    defer allocator.free(colr);
    const entr = "BASE uses BAS5 packetized GOP payloads; each GOP uses BAS4-style Huffman entropy over BAS3 motion-compensated data\n";
    const seek = try buildSeekPayload(frame_count, base_build.gop_size, base_build.packet_count);
    defer allocator.free(seek);
    const meta = try buildMetaPayload(input, output, raw.len, base_build.coded_size, base_quant.y, base_quant.uv, base_build.gop_size, base_build.packet_count);
    defer allocator.free(meta);
    const audi = "reserved for future audio support\n";

    const chunks = [_]Chunk{
        makeChunk("PRVW", prvw),
        makeChunk("MODL", modl),
        makeChunk("SEEK", seek),
        makeChunk("BASE", base_build.payload),
        makeChunk("MOTN", motn),
        makeChunk("FEAT", feat),
        makeChunk("GRAN", gran),
        makeChunk("COLR", colr),
        makeChunk("ENTR", entr),
        makeChunk("META", meta),
        makeChunk("AUDI", audi),
    };

    try writeNvc(output, head, &chunks);
    std.debug.print("encoded {s} -> {s}\n", .{ input, output });
    std.debug.print("profile={s} source={d}x{d} base={d}x{d} frames={d} raw_base={d} coded_base={d} qY={d} qUV={d}\n", .{
        profile.label(),
        probe.width,
        probe.height,
        base_dims.width,
        base_dims.height,
        frame_count,
        trimmed_raw.len,
        base_build.coded_size,
        base_quant.y,
        base_quant.uv,
    });
}

fn cmdDecode(args: []const []const u8) !void {
    if (args.len < 2) {
        std.debug.print("decode needs <input.nvc> <output.mp4>\n", .{});
        return error.InvalidArguments;
    }

    const input = args[0];
    const output = args[1];
    const parsed = try readNvc(input);
    defer parsed.deinit();

    const base_chunk = parsed.find("BASE") orelse return error.MissingBaseChunk;
    const base = try parseBasePayload(base_chunk.payload);
    const raw = try decodeBasePayload(base);
    defer allocator.free(raw);

    const temp_raw = try std.fmt.allocPrint(allocator, "{s}.decode.tmp.yuv", .{output});
    defer allocator.free(temp_raw);
    defer std.fs.cwd().deleteFile(temp_raw) catch {};

    {
        var file = try std.fs.cwd().createFile(temp_raw, .{ .truncate = true });
        defer file.close();
        try file.writeAll(raw);
    }

    const head = parsed.find("HEAD") orelse return error.MissingHeadChunk;
    const out_width = parseHeadU32(head.payload, "width") orelse base.width;
    const out_height = parseHeadU32(head.payload, "height") orelse base.height;
    try writeOutputVideo(temp_raw, output, base.width, base.height, out_width, out_height, base.fps_num, base.fps_den);
    std.debug.print("decoded {s} -> {s}\n", .{ input, output });
}

fn cmdInfo(args: []const []const u8, inspect: bool) !void {
    if (args.len < 1) {
        std.debug.print("info needs <input.nvc>\n", .{});
        return error.InvalidArguments;
    }

    const parsed = try readNvc(args[0]);
    defer parsed.deinit();

    std.debug.print("file: {s}\n", .{args[0]});
    if (parsed.find("HEAD")) |head| {
        std.debug.print("{s}", .{head.payload});
    }
    if (parsed.find("BASE")) |base_chunk| {
        const base = try parseBasePayload(base_chunk.payload);
        std.debug.print("base_codec={s}\nbase_raw_bytes={d}\nbase_coded_bytes={d}\n", .{
            baseCodecLabel(base.format),
            base.raw_size,
            base.coded_size,
        });
        if (base.format == .transform or base.format == .predictive_transform or base.format == .motion_transform or base.format == .entropy_motion_transform or base.format == .packetized_entropy_motion_transform) {
            std.debug.print("base_block_size={d}\nbase_y_quant={d}\nbase_uv_quant={d}\n", .{
                base.block_size,
                base.y_quant,
                base.uv_quant,
            });
            if (base.format == .predictive_transform or base.format == .motion_transform or base.format == .entropy_motion_transform or base.format == .packetized_entropy_motion_transform) {
                std.debug.print("base_predictor={d}\n", .{base.predictor});
            }
            if (base.format == .motion_transform or base.format == .entropy_motion_transform or base.format == .packetized_entropy_motion_transform) {
                std.debug.print("base_motion_modes={d}\n", .{base.motion_modes});
            }
            if (base.format == .entropy_motion_transform or base.format == .packetized_entropy_motion_transform) {
                std.debug.print("base_entropy={d}\n", .{base.entropy});
            }
            if (base.format == .packetized_entropy_motion_transform) {
                std.debug.print("base_gop_size={d}\nbase_packet_count={d}\n", .{ base.gop_size, base.packet_count });
            }
        }
    }
    if (parsed.find("MODL")) |modl_chunk| {
        try printModlInfo(modl_chunk.payload);
    }
    if (parsed.find("COLR")) |colr_chunk| {
        printColorInfo(colr_chunk.payload);
    }
    if (parsed.find("FEAT")) |feat_chunk| {
        try printFeatureInfo(feat_chunk.payload);
    }
    if (parsed.find("GRAN")) |gran_chunk| {
        printGrainInfo(gran_chunk.payload);
    }
    std.debug.print("chunks: {d}\n", .{parsed.chunks.len});

    if (inspect) {
        for (parsed.chunks) |chunk| {
            std.debug.print("{s} offset={d} bytes={d} crc32=0x{x:0>8} flags={d}\n", .{
                chunk.id,
                chunk.offset,
                chunk.payload.len,
                chunk.crc,
                chunk.flags,
            });
        }
    }
}

fn cmdBench(exe_path: []const u8, args: []const []const u8) !void {
    if (args.len < 1) {
        std.debug.print("bench needs <input-video>\n", .{});
        return error.InvalidArguments;
    }

    const input = args[0];
    var profiles: std.ArrayList(Profile) = .empty;
    defer profiles.deinit(allocator);
    try profiles.append(allocator, .w1);
    var frame_limit: u32 = 60;
    var model_path: ?[]const u8 = null;
    var out_dir: []const u8 = ".nvc-bench";

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--profiles")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            profiles.clearRetainingCapacity();
            try parseProfileList(args[i + 1], &profiles);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--frames")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            frame_limit = try std.fmt.parseInt(u32, args[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--model")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            model_path = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--out-dir")) {
            if (i + 1 >= args.len) return error.InvalidArguments;
            out_dir = args[i + 1];
            i += 1;
        } else {
            std.debug.print("unknown bench option: {s}\n", .{args[i]});
            return error.InvalidArguments;
        }
    }
    if (profiles.items.len == 0) return error.InvalidArguments;

    try std.fs.cwd().makePath(out_dir);
    const probe = probeVideo(input) catch VideoProbe{};
    std.debug.print("NVC alpha bench for {s}\nsource={d}x{d} fps={d}/{d} frames={d} out_dir={s}\n", .{
        input,
        probe.width,
        probe.height,
        probe.fps_num,
        probe.fps_den,
        frame_limit,
        out_dir,
    });
    std.debug.print("profile,nvc_bytes,bitrate_kbps,encode_ms,decode_ms,encode_fps,decode_fps,nvc_path,decoded_path\n", .{});

    for (profiles.items, 0..) |profile, index| {
        const stamp = std.time.milliTimestamp() + @as(i64, @intCast(index));
        const nvc_name = try std.fmt.allocPrint(allocator, "bench-{s}-{d}.nvc", .{ profileFileLabel(profile), stamp });
        defer allocator.free(nvc_name);
        const mp4_name = try std.fmt.allocPrint(allocator, "bench-{s}-{d}.mp4", .{ profileFileLabel(profile), stamp });
        defer allocator.free(mp4_name);
        const nvc_path = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, nvc_name });
        defer allocator.free(nvc_path);
        const mp4_path = try std.fs.path.join(allocator, &[_][]const u8{ out_dir, mp4_name });
        defer allocator.free(mp4_path);
        const frames_text = try std.fmt.allocPrint(allocator, "{d}", .{frame_limit});
        defer allocator.free(frames_text);

        var encode_args: std.ArrayList([]const u8) = .empty;
        defer encode_args.deinit(allocator);
        try encode_args.appendSlice(allocator, &[_][]const u8{ exe_path, "encode", input, nvc_path, "--profile", profileFileLabel(profile), "--frames", frames_text });
        if (model_path) |path| {
            try encode_args.appendSlice(allocator, &[_][]const u8{ "--model", path });
        }

        var timer = try std.time.Timer.start();
        const encode_stdout = try runChecked(encode_args.items, 16 * 1024 * 1024);
        allocator.free(encode_stdout);
        const encode_ns = timer.read();

        var decode_args: std.ArrayList([]const u8) = .empty;
        defer decode_args.deinit(allocator);
        try decode_args.appendSlice(allocator, &[_][]const u8{ exe_path, "decode", nvc_path, mp4_path });
        timer.reset();
        const decode_stdout = try runChecked(decode_args.items, 16 * 1024 * 1024);
        allocator.free(decode_stdout);
        const decode_ns = timer.read();

        const stat = try std.fs.cwd().statFile(nvc_path);
        const parsed = try readNvc(nvc_path);
        defer parsed.deinit();
        const head = parsed.find("HEAD") orelse return error.MissingHeadChunk;
        const frames = parseHeadU32(head.payload, "frames") orelse frame_limit;
        const fps_num = parseHeadU32(head.payload, "fps_num") orelse probe.fps_num;
        const fps_den = parseHeadU32(head.payload, "fps_den") orelse probe.fps_den;
        const duration_seconds = @as(f64, @floatFromInt(frames)) * @as(f64, @floatFromInt(fps_den)) / @as(f64, @floatFromInt(@max(fps_num, 1)));
        const bitrate_kbps = if (duration_seconds > 0) (@as(f64, @floatFromInt(stat.size)) * 8.0 / duration_seconds) / 1000.0 else 0.0;
        const encode_ms = @as(f64, @floatFromInt(encode_ns)) / 1_000_000.0;
        const decode_ms = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0;
        const encode_fps = if (encode_ms > 0) @as(f64, @floatFromInt(frames)) / (encode_ms / 1000.0) else 0.0;
        const decode_fps = if (decode_ms > 0) @as(f64, @floatFromInt(frames)) / (decode_ms / 1000.0) else 0.0;

        std.debug.print("{s},{d},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{s},{s}\n", .{
            profile.label(),
            stat.size,
            bitrate_kbps,
            encode_ms,
            decode_ms,
            encode_fps,
            decode_fps,
            nvc_path,
            mp4_path,
        });
    }
}

fn parseProfileList(text: []const u8, profiles: *std.ArrayList(Profile)) !void {
    var parts = std.mem.tokenizeScalar(u8, text, ',');
    while (parts.next()) |part| {
        try profiles.append(allocator, try parseProfile(part));
    }
}

fn profileFileLabel(profile: Profile) []const u8 {
    return switch (profile) {
        .w1 => "w1",
        .xc => "xc",
    };
}

fn parseProfile(text: []const u8) !Profile {
    if (std.mem.eql(u8, text, "w1") or std.mem.eql(u8, text, "NVC-W1")) return .w1;
    if (std.mem.eql(u8, text, "xc") or std.mem.eql(u8, text, "NVC-XC")) return .xc;
    return error.InvalidProfile;
}

fn parseFrameLimit(text: []const u8) !u32 {
    if (std.mem.eql(u8, text, "all") or std.mem.eql(u8, text, "full") or std.mem.eql(u8, text, "0")) return 0;
    return std.fmt.parseInt(u32, text, 10);
}

fn baseCodecLabel(format: BaseFormat) []const u8 {
    return switch (format) {
        .rle => "BAS0-rle-legacy",
        .transform => "BAS1-tiled-hadamard-zero-run-varint",
        .predictive_transform => "BAS2-predictive-tiled-hadamard-zero-run-varint",
        .motion_transform => "BAS3-motion-tiled-hadamard-zero-run-varint",
        .entropy_motion_transform => "BAS4-huffman-coded-motion-transform",
        .packetized_entropy_motion_transform => "BAS5-packetized-huffman-motion-transform",
    };
}

fn loadModelPayload(model_path: ?[]const u8) ![]u8 {
    if (model_path) |path| {
        return std.fs.cwd().readFileAlloc(allocator, path, 64 * 1024 * 1024);
    }
    return allocator.dupe(u8, "NVC-TinySR-v0 placeholder: run python3 ml/train_tinysr.py --export ml/exports/nvc-tinysr-v0.modl and pass --model.\n");
}

fn printModlInfo(payload: []const u8) !void {
    if (payload.len >= 20 and std.mem.eql(u8, payload[0..4], "MOD0")) {
        const major = readU16(payload[4..6]);
        const minor = readU16(payload[6..8]);
        const metadata_len = readU32(payload[8..12]);
        const weights_len = readU64(payload[12..20]);
        const metadata_start: usize = 20;
        const metadata_end = metadata_start + try castUsize(metadata_len);
        if (metadata_end > payload.len) return error.InvalidModelChunk;
        const metadata = payload[metadata_start..metadata_end];
        const model_id = jsonStringValue(metadata, "model_id") orelse "unknown";
        const architecture = jsonStringValue(metadata, "architecture") orelse "unknown";
        std.debug.print("model_format=MOD0\nmodel_version={d}.{d}\nmodel_id={s}\nmodel_architecture={s}\nmodel_metadata_bytes={d}\nmodel_weights_bytes={d}\n", .{
            major,
            minor,
            model_id,
            architecture,
            metadata_len,
            weights_len,
        });
    } else {
        std.debug.print("model_format=legacy-placeholder\nmodel_bytes={d}\n", .{payload.len});
    }
}

fn printColorInfo(payload: []const u8) void {
    if (payload.len >= 24 and std.mem.eql(u8, payload[0..4], "COL1")) {
        std.debug.print("color_format=COL1\ncolor_luma_scale={d:.4}\ncolor_luma_bias={d:.4}\ncolor_saturation={d:.4}\ncolor_contrast={d:.4}\n", .{
            readF32(payload[8..12]),
            readF32(payload[12..16]),
            readF32(payload[16..20]),
            readF32(payload[20..24]),
        });
    } else {
        std.debug.print("color_format=legacy-placeholder\ncolor_bytes={d}\n", .{payload.len});
    }
}

fn printGrainInfo(payload: []const u8) void {
    if (payload.len >= 20 and std.mem.eql(u8, payload[0..4], "GRN1")) {
        std.debug.print("grain_format=GRN1\ngrain_seed={d}\ngrain_intensity={d:.4}\ngrain_luma_only={d}\n", .{
            readU32(payload[8..12]),
            readF32(payload[12..16]),
            readU32(payload[16..20]),
        });
    } else {
        std.debug.print("grain_format=legacy-placeholder\ngrain_bytes={d}\n", .{payload.len});
    }
}

fn printFeatureInfo(payload: []const u8) !void {
    const feat = parseFeaturePayload(payload) catch {
        std.debug.print("feature_format=legacy-placeholder\nfeature_bytes={d}\n", .{payload.len});
        return;
    };
    std.debug.print("feature_format=FET1\nfeature_width={d}\nfeature_height={d}\nfeature_frames={d}\nfeature_tile_size={d}\nfeature_quant_step={d}\nfeature_grid={d}x{d}\nfeature_residual_bytes={d}\nfeature_coded_bytes={d}\n", .{
        feat.width,
        feat.height,
        feat.frame_count,
        feat.tile_size,
        feat.quant_step,
        feat.grid_width,
        feat.grid_height,
        feat.residual_count,
        feat.coded_size,
    });
}

fn jsonStringValue(json: []const u8, key: []const u8) ?[]const u8 {
    const pattern = std.fmt.allocPrint(allocator, "\"{s}\":\"", .{key}) catch return null;
    defer allocator.free(pattern);
    const start = std.mem.indexOf(u8, json, pattern) orelse return null;
    const value_start = start + pattern.len;
    const value_end_rel = std.mem.indexOfScalar(u8, json[value_start..], '"') orelse return null;
    return json[value_start .. value_start + value_end_rel];
}

fn makeChunk(comptime id: []const u8, payload: []const u8) Chunk {
    return .{ .id = [4]u8{ id[0], id[1], id[2], id[3] }, .payload = payload };
}

fn probeVideo(path: []const u8) !VideoProbe {
    const argv = [_][]const u8{
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:stream_tags=rotate:stream_side_data=rotation:format=duration",
        "-of",
        "default=noprint_wrappers=1",
        path,
    };
    const result = try runChecked(&argv, 64 * 1024);
    defer allocator.free(result);

    var probe = VideoProbe{};
    var rotation: i32 = 0;
    var lines = std.mem.tokenizeScalar(u8, result, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "width=")) {
            probe.width = std.fmt.parseInt(u32, line["width=".len..], 10) catch probe.width;
        } else if (std.mem.startsWith(u8, line, "height=")) {
            probe.height = std.fmt.parseInt(u32, line["height=".len..], 10) catch probe.height;
        } else if (std.mem.startsWith(u8, line, "r_frame_rate=")) {
            parseRate(line["r_frame_rate=".len..], &probe);
        } else if (std.mem.startsWith(u8, line, "TAG:rotate=")) {
            rotation = std.fmt.parseInt(i32, line["TAG:rotate=".len..], 10) catch rotation;
        } else if (std.mem.startsWith(u8, line, "rotation=")) {
            rotation = std.fmt.parseInt(i32, line["rotation=".len..], 10) catch rotation;
        } else if (std.mem.startsWith(u8, line, "duration=")) {
            const seconds = std.fmt.parseFloat(f64, line["duration=".len..]) catch 0;
            if (seconds > 0) probe.duration_ms = @intFromFloat(seconds * 1000.0);
        }
    }
    if (isQuarterTurn(rotation)) {
        const width = probe.width;
        probe.width = probe.height;
        probe.height = width;
    }
    return probe;
}

fn isQuarterTurn(rotation: i32) bool {
    const normalized = @mod(rotation, 360);
    return normalized == 90 or normalized == 270;
}

fn parseRate(text: []const u8, probe: *VideoProbe) void {
    if (std.mem.indexOfScalar(u8, text, '/')) |slash| {
        const num = std.fmt.parseInt(u32, text[0..slash], 10) catch return;
        const den = std.fmt.parseInt(u32, text[slash + 1 ..], 10) catch return;
        if (num > 0 and den > 0) {
            probe.fps_num = num;
            probe.fps_den = den;
        }
    }
}

fn chooseBaseDimensions(profile: Profile, width: u32, height: u32) struct { width: u32, height: u32 } {
    const divisor: u32 = switch (profile) {
        .w1 => 2,
        .xc => 4,
    };
    return .{ .width = evenAtLeast2(width / divisor), .height = evenAtLeast2(height / divisor) };
}

fn chooseBaseQuant(profile: Profile) struct { y: u32, uv: u32 } {
    return switch (profile) {
        .w1 => .{ .y = 8, .uv = 16 },
        .xc => .{ .y = 18, .uv = 32 },
    };
}

fn chooseBaseGopSize(profile: Profile) u32 {
    return switch (profile) {
        .w1 => 15,
        .xc => 30,
    };
}

fn evenAtLeast2(value: u32) u32 {
    var out = if (value < 2) 2 else value;
    if (out % 2 == 1) out -= 1;
    return if (out < 2) 2 else out;
}

fn extractBaseRaw(input: []const u8, temp_raw: []const u8, width: u32, height: u32, fps_num: u32, fps_den: u32, frame_limit: u32) !void {
    const scale = try std.fmt.allocPrint(allocator, "scale={d}:{d}:flags=bicubic,fps={d}/{d}", .{ width, height, fps_num, fps_den });
    defer allocator.free(scale);

    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &[_][]const u8{
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", input, "-an", "-vf", scale,
    });
    var frames: ?[]u8 = null;
    defer if (frames) |value| allocator.free(value);
    if (frame_limit > 0) {
        frames = try std.fmt.allocPrint(allocator, "{d}", .{frame_limit});
        try argv.appendSlice(allocator, &[_][]const u8{ "-frames:v", frames.? });
    }
    try argv.appendSlice(allocator, &[_][]const u8{ "-f", "rawvideo", "-pix_fmt", "yuv420p", temp_raw });
    const out = try runChecked(argv.items, 1024 * 1024);
    allocator.free(out);
}

fn writeOutputVideo(temp_raw: []const u8, output: []const u8, base_width: u32, base_height: u32, out_width: u32, out_height: u32, fps_num: u32, fps_den: u32) !void {
    const size = try std.fmt.allocPrint(allocator, "{d}x{d}", .{ base_width, base_height });
    defer allocator.free(size);
    const rate = try std.fmt.allocPrint(allocator, "{d}/{d}", .{ fps_num, fps_den });
    defer allocator.free(rate);
    const scale = try std.fmt.allocPrint(allocator, "scale={d}:{d}:flags=lanczos", .{ out_width, out_height });
    defer allocator.free(scale);

    const argv = [_][]const u8{
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        size,
        "-r",
        rate,
        "-i",
        temp_raw,
        "-vf",
        scale,
        "-pix_fmt",
        "yuv420p",
        output,
    };
    const out = try runChecked(&argv, 1024 * 1024);
    allocator.free(out);
}

fn runChecked(argv: []const []const u8, max_output: usize) ![]u8 {
    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = argv,
        .max_output_bytes = max_output,
    });
    defer allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("command failed ({d}): {s}\n{s}\n", .{ code, argv[0], result.stderr });
                allocator.free(result.stdout);
                return error.CommandFailed;
            }
        },
        else => {
            std.debug.print("command terminated unexpectedly: {s}\n{s}\n", .{ argv[0], result.stderr });
            allocator.free(result.stdout);
            return error.CommandFailed;
        },
    }

    return result.stdout;
}

fn buildHeadPayload(profile: Profile, probe: VideoProbe, base_width: u32, base_height: u32, frame_count: u32) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\profile={s}
        \\width={d}
        \\height={d}
        \\fps_num={d}
        \\fps_den={d}
        \\duration_ms={d}
        \\base_width={d}
        \\base_height={d}
        \\frames={d}
        \\color=yuv420p
        \\status=alpha-container-base-codec
        \\
    , .{ profile.label(), probe.width, probe.height, probe.fps_num, probe.fps_den, probe.duration_ms, base_width, base_height, frame_count });
}

fn buildMetaPayload(input: []const u8, output: []const u8, raw_len: usize, coded_len: usize, y_quant: u32, uv_quant: u32, gop_size: u32, packet_count: u32) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\created_by=nvc-alpha-zig
        \\input={s}
        \\output={s}
        \\base_raw_bytes={d}
        \\base_coded_bytes={d}
        \\base_codec=BAS5-packetized-huffman-motion-transform
        \\base_y_quant={d}
        \\base_uv_quant={d}
        \\base_gop_size={d}
        \\base_packet_count={d}
        \\note=Neural chunks are alpha MOD0 plus FET1/COL1/GRN1 reconstruction side data.
        \\
    , .{ input, output, raw_len, coded_len, y_quant, uv_quant, gop_size, packet_count });
}

fn buildSeekPayload(frame_count: u32, gop_size: u32, packet_count: u32) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\seek_version=1
        \\gop_size={d}
        \\frames={d}
        \\packets={d}
        \\note=BAS5 stores independently decodable GOP packets inside BASE. Packet byte offsets live in the BAS5 table.
        \\
    , .{ gop_size, frame_count, packet_count });
}

fn buildColorPayload(source_raw: []const u8, decoded_raw: []const u8, width: u32, height: u32, frame_count: u32) ![]u8 {
    const frame_size = yuv420FrameSize(width, height);
    const y_size: usize = @as(usize, width) * @as(usize, height);
    const uv_width = width / 2;
    const uv_height = height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);
    if (frame_size == 0 or source_raw.len < frame_size * @as(usize, frame_count) or decoded_raw.len < frame_size * @as(usize, frame_count)) return error.InvalidColorPayload;

    var source_y_sum: f64 = 0;
    var decoded_y_sum: f64 = 0;
    var source_y_sq_sum: f64 = 0;
    var decoded_y_sq_sum: f64 = 0;
    var source_chroma_sum: f64 = 0;
    var decoded_chroma_sum: f64 = 0;

    var frame: u32 = 0;
    while (frame < frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        const source_y = source_raw[offset .. offset + y_size];
        const source_u = source_raw[offset + y_size .. offset + y_size + uv_size];
        const source_v = source_raw[offset + y_size + uv_size .. offset + y_size + uv_size * 2];
        const decoded_y = decoded_raw[offset .. offset + y_size];
        const decoded_u = decoded_raw[offset + y_size .. offset + y_size + uv_size];
        const decoded_v = decoded_raw[offset + y_size + uv_size .. offset + y_size + uv_size * 2];

        for (source_y, decoded_y) |source, decoded| {
            const source_f: f64 = @floatFromInt(source);
            const decoded_f: f64 = @floatFromInt(decoded);
            source_y_sum += source_f;
            decoded_y_sum += decoded_f;
            source_y_sq_sum += source_f * source_f;
            decoded_y_sq_sum += decoded_f * decoded_f;
        }
        for (source_u, source_v, decoded_u, decoded_v) |su, sv, du, dv| {
            const source_du = @as(f64, @floatFromInt(su)) - 128.0;
            const source_dv = @as(f64, @floatFromInt(sv)) - 128.0;
            const decoded_du = @as(f64, @floatFromInt(du)) - 128.0;
            const decoded_dv = @as(f64, @floatFromInt(dv)) - 128.0;
            source_chroma_sum += @sqrt(source_du * source_du + source_dv * source_dv);
            decoded_chroma_sum += @sqrt(decoded_du * decoded_du + decoded_dv * decoded_dv);
        }
    }

    const y_count: f64 = @floatFromInt(@as(usize, frame_count) * y_size);
    const uv_count: f64 = @floatFromInt(@as(usize, frame_count) * uv_size);
    const source_mean = source_y_sum / y_count;
    const decoded_mean = decoded_y_sum / y_count;
    const source_var = @max(0.0, source_y_sq_sum / y_count - source_mean * source_mean);
    const decoded_var = @max(0.0, decoded_y_sq_sum / y_count - decoded_mean * decoded_mean);
    const source_std = @sqrt(source_var);
    const decoded_std = @sqrt(decoded_var);
    const contrast = clampF64(if (decoded_std > 0.001) source_std / decoded_std else 1.0, 0.85, 1.15);
    const luma_bias = clampF64(source_mean - decoded_mean, -16.0, 16.0);
    const saturation = clampF64(if (decoded_chroma_sum > 0.001) (source_chroma_sum / uv_count) / (decoded_chroma_sum / uv_count) else 1.0, 0.85, 1.20);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "COL1");
    try appendU16(&out, 1);
    try appendU16(&out, 0);
    try appendF32(&out, 1.0); // luma scale
    try appendF32(&out, @floatCast(luma_bias)); // luma bias in byte-space
    try appendF32(&out, @floatCast(saturation)); // saturation scale
    try appendF32(&out, @floatCast(contrast)); // contrast scale
    return out.toOwnedSlice(allocator);
}

fn buildGrainPayload(profile: Profile) ![]u8 {
    const intensity: f32 = switch (profile) {
        .w1 => 0.004,
        .xc => 0.007,
    };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "GRN1");
    try appendU16(&out, 1);
    try appendU16(&out, 0);
    try appendU32(&out, 0x4e564331);
    try appendF32(&out, intensity);
    try appendU32(&out, 1); // luma-only deterministic grain
    return out.toOwnedSlice(allocator);
}

fn buildFeaturePayload(source_raw: []const u8, decoded_raw: []const u8, width: u32, height: u32, frame_count: u32, profile: Profile) ![]u8 {
    const frame_size = yuv420FrameSize(width, height);
    const y_size: usize = @as(usize, width) * @as(usize, height);
    if (frame_size == 0 or source_raw.len < frame_size * @as(usize, frame_count) or decoded_raw.len < frame_size * @as(usize, frame_count)) return error.InvalidFeaturePayload;

    const tile_size: u32 = switch (profile) {
        .w1 => 32,
        .xc => 32,
    };
    const quant_step: u32 = switch (profile) {
        .w1 => 2,
        .xc => 3,
    };
    const grid_width = ceilDivU32(width, tile_size);
    const grid_height = ceilDivU32(height, tile_size);
    const residual_count = @as(usize, frame_count) * @as(usize, grid_width) * @as(usize, grid_height);
    var residuals = try allocator.alloc(u8, residual_count);
    defer allocator.free(residuals);

    var frame: u32 = 0;
    while (frame < frame_count) : (frame += 1) {
        const frame_offset = @as(usize, frame) * frame_size;
        const source_y = source_raw[frame_offset .. frame_offset + y_size];
        const decoded_y = decoded_raw[frame_offset .. frame_offset + y_size];
        var tile_y: u32 = 0;
        while (tile_y < grid_height) : (tile_y += 1) {
            const y0 = tile_y * tile_size;
            const y1 = @min(height, y0 + tile_size);
            var tile_x: u32 = 0;
            while (tile_x < grid_width) : (tile_x += 1) {
                const x0 = tile_x * tile_size;
                const x1 = @min(width, x0 + tile_size);
                var sum: i64 = 0;
                var count: i64 = 0;
                var y = y0;
                while (y < y1) : (y += 1) {
                    var x = x0;
                    while (x < x1) : (x += 1) {
                        const index = @as(usize, y) * @as(usize, width) + @as(usize, x);
                        sum += @as(i64, source_y[index]) - @as(i64, decoded_y[index]);
                        count += 1;
                    }
                }
                const average = divRoundI64(sum, count);
                const quantized = clampI64(divRoundI64(average, @intCast(quant_step)), -127, 127);
                const residual_index = (@as(usize, frame) * @as(usize, grid_height) + @as(usize, tile_y)) * @as(usize, grid_width) + @as(usize, tile_x);
                residuals[residual_index] = @intCast(quantized + 128);
            }
        }
    }

    const coded = try entropyEncode(residuals);
    defer allocator.free(coded);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "FET1");
    try appendU16(&out, 1);
    try appendU16(&out, 0);
    try appendU32(&out, width);
    try appendU32(&out, height);
    try appendU32(&out, frame_count);
    try appendU32(&out, tile_size);
    try appendU32(&out, quant_step);
    try appendU32(&out, grid_width);
    try appendU32(&out, grid_height);
    try appendU64(&out, @intCast(residuals.len));
    try appendU64(&out, @intCast(coded.len));
    try out.appendSlice(allocator, coded);
    return out.toOwnedSlice(allocator);
}

fn parseFeaturePayload(payload: []const u8) !FeatureInfo {
    if (payload.len < feat1_header_len or !std.mem.eql(u8, payload[0..4], "FET1")) return error.InvalidFeaturePayload;
    const coded_size = readU64(payload[44..52]);
    const coded_len = try castUsize(coded_size);
    if (payload.len < feat1_header_len + coded_len) return error.InvalidFeaturePayload;
    return .{
        .width = readU32(payload[8..12]),
        .height = readU32(payload[12..16]),
        .frame_count = readU32(payload[16..20]),
        .tile_size = readU32(payload[20..24]),
        .quant_step = readU32(payload[24..28]),
        .grid_width = readU32(payload[28..32]),
        .grid_height = readU32(payload[32..36]),
        .residual_count = readU64(payload[36..44]),
        .coded_size = coded_size,
        .coded = payload[feat1_header_len .. feat1_header_len + coded_len],
    };
}

fn buildBase4Payload(width: u32, height: u32, fps_num: u32, fps_den: u32, frame_count: u32, raw_size: usize, coded: []const u8, y_quant: u32, uv_quant: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    try out.appendSlice(allocator, "BAS4");
    try appendU16(&out, 4);
    try appendU16(&out, 0);
    try appendU32(&out, width);
    try appendU32(&out, height);
    try appendU32(&out, fps_num);
    try appendU32(&out, fps_den);
    try appendU32(&out, frame_count);
    try appendU64(&out, @intCast(raw_size));
    try appendU64(&out, @intCast(coded.len));
    try appendU32(&out, transform_block_size);
    try appendU32(&out, y_quant);
    try appendU32(&out, uv_quant);
    try appendU32(&out, 3);
    try appendU32(&out, motion_mode_count);
    try appendU32(&out, 1);
    try out.appendSlice(allocator, coded);

    return out.toOwnedSlice(allocator);
}

fn buildBasePayload(raw: []const u8, width: u32, height: u32, fps_num: u32, fps_den: u32, frame_count: u32, y_quant: u32, uv_quant: u32, requested_gop_size: u32) !BaseBuild {
    const frame_size = yuv420FrameSize(width, height);
    if (frame_size == 0 or raw.len < frame_size * @as(usize, frame_count)) return error.InvalidBasePayload;
    const gop_size = @max(1, requested_gop_size);
    const packet_count = (frame_count + gop_size - 1) / gop_size;

    var packets: std.ArrayList(BasePacketBuild) = .empty;
    defer {
        for (packets.items) |packet| allocator.free(packet.coded);
        packets.deinit(allocator);
    }

    var total_coded_size: usize = 0;
    var start_frame: u32 = 0;
    while (start_frame < frame_count) : (start_frame += gop_size) {
        const packet_frames = @min(gop_size, frame_count - start_frame);
        const raw_start = @as(usize, start_frame) * frame_size;
        const raw_end = raw_start + @as(usize, packet_frames) * frame_size;
        const packed_motion = try encodeBaseMotionTransform(raw[raw_start..raw_end], width, height, packet_frames, y_quant, uv_quant);
        defer allocator.free(packed_motion);
        const coded = try entropyEncode(packed_motion);
        errdefer allocator.free(coded);
        try packets.append(allocator, .{
            .start_frame = start_frame,
            .frame_count = packet_frames,
            .coded = coded,
        });
        total_coded_size += coded.len;
    }

    const table_len = @as(usize, packet_count) * base5_packet_entry_len;
    const packet_data_start = base5_header_len + table_len;
    var packet_offset: u64 = @intCast(packet_data_start);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "BAS5");
    try appendU16(&out, 5);
    try appendU16(&out, 0);
    try appendU32(&out, width);
    try appendU32(&out, height);
    try appendU32(&out, fps_num);
    try appendU32(&out, fps_den);
    try appendU32(&out, frame_count);
    try appendU64(&out, @intCast(raw.len));
    try appendU64(&out, @intCast(total_coded_size));
    try appendU32(&out, transform_block_size);
    try appendU32(&out, y_quant);
    try appendU32(&out, uv_quant);
    try appendU32(&out, 4);
    try appendU32(&out, motion_mode_count);
    try appendU32(&out, 1);
    try appendU32(&out, gop_size);
    try appendU32(&out, packet_count);

    for (packets.items) |packet| {
        try appendU32(&out, packet.start_frame);
        try appendU32(&out, packet.frame_count);
        try appendU64(&out, packet_offset);
        try appendU64(&out, @intCast(packet.coded.len));
        packet_offset += @as(u64, @intCast(packet.coded.len));
    }

    for (packets.items) |packet| {
        try out.appendSlice(allocator, packet.coded);
    }

    return .{
        .payload = try out.toOwnedSlice(allocator),
        .coded_size = total_coded_size,
        .gop_size = gop_size,
        .packet_count = packet_count,
    };
}

fn buildPreviewPayload(raw: []const u8, width: u32, height: u32, frame_count: u32, fps_num: u32, fps_den: u32, max_width: u32) ![]u8 {
    const y_size: usize = @as(usize, width) * @as(usize, height);
    const uv_width = width / 2;
    const uv_height = height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);
    const frame_size = y_size + uv_size * 2;
    if (frame_size == 0 or raw.len < frame_size) return error.InvalidBasePayload;

    const preview_width = @max(2, evenU32(@min(width, max_width)));
    const scaled_height = @max(2, @as(u32, @intCast((@as(u64, height) * preview_width + width / 2) / width)));
    const preview_height = evenU32(scaled_height);
    const frame_bytes: usize = @as(usize, preview_width) * @as(usize, preview_height) * 3;
    const frames_available: u32 = @intCast(raw.len / frame_size);
    const preview_frames = @min(frame_count, frames_available);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "PVW1");
    try appendU16(&out, 1);
    try appendU16(&out, 0);
    try appendU32(&out, preview_width);
    try appendU32(&out, preview_height);
    try appendU32(&out, fps_num);
    try appendU32(&out, fps_den);
    try appendU32(&out, preview_frames);
    try appendU32(&out, @intCast(frame_bytes));

    var frame: u32 = 0;
    while (frame < preview_frames) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        const first_frame = raw[offset .. offset + frame_size];
        const y_plane = first_frame[0..y_size];
        const u_plane = first_frame[y_size .. y_size + uv_size];
        const v_plane = first_frame[y_size + uv_size .. y_size + uv_size * 2];
        var py: u32 = 0;
        while (py < preview_height) : (py += 1) {
            const src_y = @min(height - 1, @as(u32, @intCast((@as(u64, py) * height) / preview_height)));
            var px: u32 = 0;
            while (px < preview_width) : (px += 1) {
                const src_x = @min(width - 1, @as(u32, @intCast((@as(u64, px) * width) / preview_width)));
                const y_value: i32 = y_plane[@as(usize, src_y) * @as(usize, width) + src_x];
                const uv_x = @min(uv_width - 1, src_x / 2);
                const uv_y = @min(uv_height - 1, src_y / 2);
                const uv_index = @as(usize, uv_y) * @as(usize, uv_width) + uv_x;
                const u_value: i32 = @as(i32, u_plane[uv_index]) - 128;
                const v_value: i32 = @as(i32, v_plane[uv_index]) - 128;
                try out.append(allocator, clampToU8(y_value + @divTrunc(1436 * v_value, 1024)));
                try out.append(allocator, clampToU8(y_value - @divTrunc(352 * u_value + 731 * v_value, 1024)));
                try out.append(allocator, clampToU8(y_value + @divTrunc(1815 * u_value, 1024)));
            }
        }
    }

    return out.toOwnedSlice(allocator);
}

fn evenU32(value: u32) u32 {
    if (value % 2 == 0) return value;
    return value - 1;
}

fn ceilDivU32(numerator: u32, denominator: u32) u32 {
    if (denominator == 0) return 0;
    return (numerator + denominator - 1) / denominator;
}

fn parseBasePayload(payload: []const u8) !BaseInfo {
    if (payload.len < 4) return error.InvalidBasePayload;

    if (std.mem.eql(u8, payload[0..4], "BAS0")) {
        if (payload.len < base0_header_len) return error.InvalidBasePayload;
        const coded_size = readU64(payload[36..44]);
        const coded_len = try castUsize(coded_size);
        if (payload.len < base0_header_len + coded_len) return error.InvalidBasePayload;
        return .{
            .format = .rle,
            .width = readU32(payload[8..12]),
            .height = readU32(payload[12..16]),
            .fps_num = readU32(payload[16..20]),
            .fps_den = readU32(payload[20..24]),
            .frame_count = readU32(payload[24..28]),
            .raw_size = readU64(payload[28..36]),
            .coded_size = coded_size,
            .coded = payload[base0_header_len .. base0_header_len + coded_len],
        };
    }

    if (std.mem.eql(u8, payload[0..4], "BAS1")) {
        if (payload.len < base1_header_len) return error.InvalidBasePayload;
        const coded_size = readU64(payload[36..44]);
        const coded_len = try castUsize(coded_size);
        if (payload.len < base1_header_len + coded_len) return error.InvalidBasePayload;
        return .{
            .format = .transform,
            .width = readU32(payload[8..12]),
            .height = readU32(payload[12..16]),
            .fps_num = readU32(payload[16..20]),
            .fps_den = readU32(payload[20..24]),
            .frame_count = readU32(payload[24..28]),
            .raw_size = readU64(payload[28..36]),
            .coded_size = coded_size,
            .block_size = readU32(payload[44..48]),
            .y_quant = readU32(payload[48..52]),
            .uv_quant = readU32(payload[52..56]),
            .coded = payload[base1_header_len .. base1_header_len + coded_len],
        };
    }

    if (std.mem.eql(u8, payload[0..4], "BAS2")) {
        if (payload.len < base2_header_len) return error.InvalidBasePayload;
        const coded_size = readU64(payload[36..44]);
        const coded_len = try castUsize(coded_size);
        if (payload.len < base2_header_len + coded_len) return error.InvalidBasePayload;
        return .{
            .format = .predictive_transform,
            .width = readU32(payload[8..12]),
            .height = readU32(payload[12..16]),
            .fps_num = readU32(payload[16..20]),
            .fps_den = readU32(payload[20..24]),
            .frame_count = readU32(payload[24..28]),
            .raw_size = readU64(payload[28..36]),
            .coded_size = coded_size,
            .block_size = readU32(payload[44..48]),
            .y_quant = readU32(payload[48..52]),
            .uv_quant = readU32(payload[52..56]),
            .predictor = readU32(payload[56..60]),
            .coded = payload[base2_header_len .. base2_header_len + coded_len],
        };
    }

    if (std.mem.eql(u8, payload[0..4], "BAS3")) {
        if (payload.len < base3_header_len) return error.InvalidBasePayload;
        const coded_size = readU64(payload[36..44]);
        const coded_len = try castUsize(coded_size);
        if (payload.len < base3_header_len + coded_len) return error.InvalidBasePayload;
        return .{
            .format = .motion_transform,
            .width = readU32(payload[8..12]),
            .height = readU32(payload[12..16]),
            .fps_num = readU32(payload[16..20]),
            .fps_den = readU32(payload[20..24]),
            .frame_count = readU32(payload[24..28]),
            .raw_size = readU64(payload[28..36]),
            .coded_size = coded_size,
            .block_size = readU32(payload[44..48]),
            .y_quant = readU32(payload[48..52]),
            .uv_quant = readU32(payload[52..56]),
            .predictor = readU32(payload[56..60]),
            .motion_modes = readU32(payload[60..64]),
            .coded = payload[base3_header_len .. base3_header_len + coded_len],
        };
    }

    if (std.mem.eql(u8, payload[0..4], "BAS4")) {
        if (payload.len < base4_header_len) return error.InvalidBasePayload;
        const coded_size = readU64(payload[36..44]);
        const coded_len = try castUsize(coded_size);
        if (payload.len < base4_header_len + coded_len) return error.InvalidBasePayload;
        return .{
            .format = .entropy_motion_transform,
            .width = readU32(payload[8..12]),
            .height = readU32(payload[12..16]),
            .fps_num = readU32(payload[16..20]),
            .fps_den = readU32(payload[20..24]),
            .frame_count = readU32(payload[24..28]),
            .raw_size = readU64(payload[28..36]),
            .coded_size = coded_size,
            .block_size = readU32(payload[44..48]),
            .y_quant = readU32(payload[48..52]),
            .uv_quant = readU32(payload[52..56]),
            .predictor = readU32(payload[56..60]),
            .motion_modes = readU32(payload[60..64]),
            .entropy = readU32(payload[64..68]),
            .coded = payload[base4_header_len .. base4_header_len + coded_len],
        };
    }

    if (std.mem.eql(u8, payload[0..4], "BAS5")) {
        if (payload.len < base5_header_len) return error.InvalidBasePayload;
        const coded_size = readU64(payload[36..44]);
        const packet_count = readU32(payload[72..76]);
        const table_len = @as(usize, packet_count) * base5_packet_entry_len;
        if (payload.len < base5_header_len + table_len) return error.InvalidBasePayload;
        return .{
            .format = .packetized_entropy_motion_transform,
            .width = readU32(payload[8..12]),
            .height = readU32(payload[12..16]),
            .fps_num = readU32(payload[16..20]),
            .fps_den = readU32(payload[20..24]),
            .frame_count = readU32(payload[24..28]),
            .raw_size = readU64(payload[28..36]),
            .coded_size = coded_size,
            .block_size = readU32(payload[44..48]),
            .y_quant = readU32(payload[48..52]),
            .uv_quant = readU32(payload[52..56]),
            .predictor = readU32(payload[56..60]),
            .motion_modes = readU32(payload[60..64]),
            .entropy = readU32(payload[64..68]),
            .gop_size = readU32(payload[68..72]),
            .packet_count = packet_count,
            .coded = payload,
        };
    }

    return error.InvalidBasePayload;
}

fn writeNvc(path: []const u8, head: []const u8, chunks_after_toc: []const Chunk) !void {
    const toc_payload = try buildTocPayload(head, chunks_after_toc);
    defer allocator.free(toc_payload);
    const toc_chunk = makeChunk("TOC0", toc_payload);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    try out.appendSlice(allocator, file_magic);
    try appendU16(&out, 0);
    try appendU16(&out, 1);
    try appendU32(&out, file_header_len);
    try appendU64(&out, 0);

    try appendChunk(&out, makeChunk("HEAD", head));
    try appendChunk(&out, toc_chunk);
    for (chunks_after_toc) |chunk| {
        try appendChunk(&out, chunk);
    }

    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(out.items);
}

fn buildTocPayload(head: []const u8, chunks_after_toc: []const Chunk) ![]u8 {
    const toc_payload_len = 8 + chunks_after_toc.len * 28;
    var current_offset: u64 = file_header_len + chunk_header_len + @as(u64, @intCast(head.len)) + chunk_header_len + @as(u64, @intCast(toc_payload_len));

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "TOC0");
    try appendU32(&out, @intCast(chunks_after_toc.len));
    for (chunks_after_toc) |chunk| {
        try out.appendSlice(allocator, chunk.id[0..]);
        try appendU64(&out, current_offset);
        try appendU64(&out, @intCast(chunk.payload.len));
        try appendU32(&out, crc32(chunk.payload));
        try appendU32(&out, chunk.flags);
        current_offset += chunk_header_len + @as(u64, @intCast(chunk.payload.len));
    }
    return out.toOwnedSlice(allocator);
}

fn appendChunk(out: *std.ArrayList(u8), chunk: Chunk) !void {
    try out.appendSlice(allocator, chunk.id[0..]);
    try appendU64(out, @intCast(chunk.payload.len));
    try appendU32(out, crc32(chunk.payload));
    try appendU32(out, chunk.flags);
    try out.appendSlice(allocator, chunk.payload);
}

fn readNvc(path: []const u8) !ParsedNvc {
    const data = try std.fs.cwd().readFileAlloc(allocator, path, max_file_bytes);
    errdefer allocator.free(data);
    if (data.len < file_header_len or !std.mem.eql(u8, data[0..4], file_magic)) return error.InvalidNvcMagic;
    const header_len = readU32(data[8..12]);
    if (header_len < file_header_len or header_len > data.len) return error.InvalidNvcHeader;

    var chunks: std.ArrayList(ParsedChunk) = .empty;
    errdefer chunks.deinit(allocator);

    var offset: usize = header_len;
    while (offset < data.len) {
        if (data.len - offset < chunk_header_len) return error.TruncatedChunkHeader;
        var id: [4]u8 = undefined;
        @memcpy(id[0..], data[offset .. offset + 4]);
        const payload_len = try castUsize(readU64(data[offset + 4 .. offset + 12]));
        const crc = readU32(data[offset + 12 .. offset + 16]);
        const flags = readU32(data[offset + 16 .. offset + 20]);
        const payload_offset = offset + chunk_header_len;
        const end = payload_offset + payload_len;
        if (end > data.len) return error.TruncatedChunkPayload;
        const payload = data[payload_offset..end];
        if (crc32(payload) != crc) return error.CrcMismatch;
        try chunks.append(allocator, .{ .id = id, .payload = payload, .offset = offset, .crc = crc, .flags = flags });
        offset = end;
    }

    return .{ .data = data, .chunks = try chunks.toOwnedSlice(allocator) };
}

fn parseHeadU32(payload: []const u8, key: []const u8) ?u32 {
    var lines = std.mem.tokenizeScalar(u8, payload, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOfScalar(u8, line, '=')) |eq| {
            if (std.mem.eql(u8, line[0..eq], key)) {
                return std.fmt.parseInt(u32, line[eq + 1 ..], 10) catch null;
            }
        }
    }
    return null;
}

fn yuv420FrameSize(width: u32, height: u32) usize {
    const pixels: usize = @as(usize, width) * @as(usize, height);
    return pixels + pixels / 2;
}

fn encodeBaseTransform(raw: []const u8, width: u32, height: u32, frame_count: u32, y_quant: u32, uv_quant: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    const frame_size = yuv420FrameSize(width, height);
    const y_size: usize = @as(usize, width) * @as(usize, height);
    const uv_width = width / 2;
    const uv_height = height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);

    var frame: u32 = 0;
    var zero_run: u64 = 0;
    while (frame < frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        const y = raw[offset .. offset + y_size];
        const u = raw[offset + y_size .. offset + y_size + uv_size];
        const v = raw[offset + y_size + uv_size .. offset + y_size + uv_size + uv_size];
        try encodePlaneTransform(y, width, height, y_quant, &out, &zero_run);
        try encodePlaneTransform(u, uv_width, uv_height, uv_quant, &out, &zero_run);
        try encodePlaneTransform(v, uv_width, uv_height, uv_quant, &out, &zero_run);
    }
    try flushZeroRun(&out, &zero_run);
    return out.toOwnedSlice(allocator);
}

fn encodeBasePredictiveTransform(raw: []const u8, width: u32, height: u32, frame_count: u32, y_quant: u32, uv_quant: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);

    const frame_size = yuv420FrameSize(width, height);
    const y_size: usize = @as(usize, width) * @as(usize, height);
    const uv_width = width / 2;
    const uv_height = height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);

    var frame: u32 = 0;
    var zero_run: u64 = 0;
    while (frame < frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        const y = raw[offset .. offset + y_size];
        const u = raw[offset + y_size .. offset + y_size + uv_size];
        const v = raw[offset + y_size + uv_size .. offset + y_size + uv_size + uv_size];
        try encodePlanePredictiveTransform(y, width, height, y_quant, &out, &zero_run);
        try encodePlanePredictiveTransform(u, uv_width, uv_height, uv_quant, &out, &zero_run);
        try encodePlanePredictiveTransform(v, uv_width, uv_height, uv_quant, &out, &zero_run);
    }
    try flushZeroRun(&out, &zero_run);
    return out.toOwnedSlice(allocator);
}

fn encodeBaseMotionTransform(raw: []const u8, width: u32, height: u32, frame_count: u32, y_quant: u32, uv_quant: u32) ![]u8 {
    var modes: std.ArrayList(u8) = .empty;
    defer modes.deinit(allocator);
    var coeffs: std.ArrayList(u8) = .empty;
    defer coeffs.deinit(allocator);

    const frame_size = yuv420FrameSize(width, height);
    const y_size: usize = @as(usize, width) * @as(usize, height);
    const uv_width = width / 2;
    const uv_height = height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);

    var prev_y: ?[]u8 = null;
    var prev_u: ?[]u8 = null;
    var prev_v: ?[]u8 = null;
    defer if (prev_y) |buf| allocator.free(buf);
    defer if (prev_u) |buf| allocator.free(buf);
    defer if (prev_v) |buf| allocator.free(buf);

    var frame: u32 = 0;
    var zero_run: u64 = 0;
    while (frame < frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        const y = raw[offset .. offset + y_size];
        const u = raw[offset + y_size .. offset + y_size + uv_size];
        const v = raw[offset + y_size + uv_size .. offset + y_size + uv_size + uv_size];

        const cur_y = try encodePlaneMotionTransform(y, prev_y, width, height, y_quant, &modes, &coeffs, &zero_run);
        const cur_u = try encodePlaneMotionTransform(u, prev_u, uv_width, uv_height, uv_quant, &modes, &coeffs, &zero_run);
        const cur_v = try encodePlaneMotionTransform(v, prev_v, uv_width, uv_height, uv_quant, &modes, &coeffs, &zero_run);

        if (prev_y) |buf| allocator.free(buf);
        if (prev_u) |buf| allocator.free(buf);
        if (prev_v) |buf| allocator.free(buf);
        prev_y = cur_y;
        prev_u = cur_u;
        prev_v = cur_v;
    }
    try flushZeroRun(&coeffs, &zero_run);

    const compressed_modes = try rleCompress(modes.items);
    defer allocator.free(compressed_modes);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try appendU64(&out, @intCast(compressed_modes.len));
    try out.appendSlice(allocator, compressed_modes);
    try out.appendSlice(allocator, coeffs.items);
    return out.toOwnedSlice(allocator);
}

fn decodeBasePayload(base: BaseInfo) ![]u8 {
    return switch (base.format) {
        .rle => rleDecompress(base.coded, try castUsize(base.raw_size)),
        .transform => decodeBaseTransform(base),
        .predictive_transform => decodeBasePredictiveTransform(base),
        .motion_transform => decodeBaseMotionTransform(base),
        .entropy_motion_transform => decodeBaseEntropyMotionTransform(base),
        .packetized_entropy_motion_transform => decodeBasePacketizedEntropyMotionTransform(base),
    };
}

fn decodeBaseTransform(base: BaseInfo) ![]u8 {
    if (base.block_size != transform_block_size) return error.UnsupportedBaseBlockSize;
    const expected = try castUsize(base.raw_size);
    var out = try allocator.alloc(u8, expected);
    errdefer allocator.free(out);

    const frame_size = yuv420FrameSize(base.width, base.height);
    const y_size: usize = @as(usize, base.width) * @as(usize, base.height);
    const uv_width = base.width / 2;
    const uv_height = base.height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);

    var reader = CoeffReader{ .data = base.coded };
    var frame: u32 = 0;
    while (frame < base.frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        try decodePlaneTransform(out[offset .. offset + y_size], base.width, base.height, base.y_quant, &reader);
        try decodePlaneTransform(out[offset + y_size .. offset + y_size + uv_size], uv_width, uv_height, base.uv_quant, &reader);
        try decodePlaneTransform(out[offset + y_size + uv_size .. offset + y_size + uv_size + uv_size], uv_width, uv_height, base.uv_quant, &reader);
    }

    return out;
}

fn decodeBasePredictiveTransform(base: BaseInfo) ![]u8 {
    if (base.block_size != transform_block_size or base.predictor != 1) return error.UnsupportedBasePredictor;
    const expected = try castUsize(base.raw_size);
    var out = try allocator.alloc(u8, expected);
    errdefer allocator.free(out);

    const frame_size = yuv420FrameSize(base.width, base.height);
    const y_size: usize = @as(usize, base.width) * @as(usize, base.height);
    const uv_width = base.width / 2;
    const uv_height = base.height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);

    var reader = CoeffReader{ .data = base.coded };
    var frame: u32 = 0;
    while (frame < base.frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        try decodePlanePredictiveTransform(out[offset .. offset + y_size], base.width, base.height, base.y_quant, &reader);
        try decodePlanePredictiveTransform(out[offset + y_size .. offset + y_size + uv_size], uv_width, uv_height, base.uv_quant, &reader);
        try decodePlanePredictiveTransform(out[offset + y_size + uv_size .. offset + y_size + uv_size + uv_size], uv_width, uv_height, base.uv_quant, &reader);
    }

    return out;
}

fn decodeBaseMotionTransform(base: BaseInfo) ![]u8 {
    if (base.block_size != transform_block_size or base.predictor != 2 or base.motion_modes != motion_mode_count) return error.UnsupportedBasePredictor;
    return decodeBaseMotionPacked(base, base.coded);
}

fn decodeBaseEntropyMotionTransform(base: BaseInfo) ![]u8 {
    if (base.block_size != transform_block_size or base.predictor != 3 or base.motion_modes != motion_mode_count or base.entropy != 1) return error.UnsupportedBasePredictor;
    const packed_data = try entropyDecode(base.coded);
    defer allocator.free(packed_data);
    return decodeBaseMotionPacked(base, packed_data);
}

fn decodeBasePacketizedEntropyMotionTransform(base: BaseInfo) ![]u8 {
    if (base.block_size != transform_block_size or base.predictor != 4 or base.motion_modes != motion_mode_count or base.entropy != 1) return error.UnsupportedBasePredictor;
    if (base.coded.len < base5_header_len) return error.InvalidBasePayload;
    const packet_count = base.packet_count;
    const table_len = @as(usize, packet_count) * base5_packet_entry_len;
    if (base.coded.len < base5_header_len + table_len) return error.InvalidBasePayload;

    const expected = try castUsize(base.raw_size);
    var out = try allocator.alloc(u8, expected);
    errdefer allocator.free(out);

    const frame_size = yuv420FrameSize(base.width, base.height);
    var packet_index: u32 = 0;
    while (packet_index < packet_count) : (packet_index += 1) {
        const entry = base5_header_len + @as(usize, packet_index) * base5_packet_entry_len;
        const start_frame = readU32(base.coded[entry .. entry + 4]);
        const packet_frames = readU32(base.coded[entry + 4 .. entry + 8]);
        const payload_offset = try castUsize(readU64(base.coded[entry + 8 .. entry + 16]));
        const payload_size = try castUsize(readU64(base.coded[entry + 16 .. entry + 24]));
        if (packet_frames == 0 or start_frame + packet_frames > base.frame_count) return error.InvalidBasePayload;
        if (base.coded.len < payload_offset + payload_size) return error.InvalidBasePayload;

        const packed_data = try entropyDecode(base.coded[payload_offset .. payload_offset + payload_size]);
        defer allocator.free(packed_data);
        const packet_raw_size = @as(usize, packet_frames) * frame_size;
        const packet_base = BaseInfo{
            .format = .motion_transform,
            .width = base.width,
            .height = base.height,
            .fps_num = base.fps_num,
            .fps_den = base.fps_den,
            .frame_count = packet_frames,
            .raw_size = packet_raw_size,
            .coded_size = payload_size,
            .block_size = base.block_size,
            .y_quant = base.y_quant,
            .uv_quant = base.uv_quant,
            .predictor = 3,
            .motion_modes = base.motion_modes,
            .entropy = base.entropy,
            .coded = packed_data,
        };
        const packet_raw = try decodeBaseMotionPacked(packet_base, packed_data);
        defer allocator.free(packet_raw);

        const out_start = @as(usize, start_frame) * frame_size;
        if (out.len < out_start + packet_raw.len or packet_raw.len != packet_raw_size) return error.InvalidBasePayload;
        @memcpy(out[out_start .. out_start + packet_raw.len], packet_raw);
    }

    return out;
}

fn decodeBaseMotionPacked(base: BaseInfo, packed_data: []const u8) ![]u8 {
    const expected = try castUsize(base.raw_size);
    var out = try allocator.alloc(u8, expected);
    errdefer allocator.free(out);

    const frame_size = yuv420FrameSize(base.width, base.height);
    const y_size: usize = @as(usize, base.width) * @as(usize, base.height);
    const uv_width = base.width / 2;
    const uv_height = base.height / 2;
    const uv_size: usize = @as(usize, uv_width) * @as(usize, uv_height);

    if (packed_data.len < 8) return error.InvalidMotionStream;
    const mode_len = try castUsize(readU64(packed_data[0..8]));
    if (packed_data.len < 8 + mode_len) return error.InvalidMotionStream;
    const expected_modes = try expectedMotionModeCount(base.width, base.height, base.frame_count);
    const mode_data = try rleDecompress(packed_data[8 .. 8 + mode_len], expected_modes);
    defer allocator.free(mode_data);
    var mode_reader = ModeReader{ .data = mode_data };
    var coeff_reader = CoeffReader{ .data = packed_data[8 + mode_len ..] };
    var prev_y: ?[]const u8 = null;
    var prev_u: ?[]const u8 = null;
    var prev_v: ?[]const u8 = null;

    var frame: u32 = 0;
    while (frame < base.frame_count) : (frame += 1) {
        const offset = @as(usize, frame) * frame_size;
        const y = out[offset .. offset + y_size];
        const u = out[offset + y_size .. offset + y_size + uv_size];
        const v = out[offset + y_size + uv_size .. offset + y_size + uv_size + uv_size];

        try decodePlaneMotionTransform(y, prev_y, base.width, base.height, base.y_quant, &mode_reader, &coeff_reader);
        try decodePlaneMotionTransform(u, prev_u, uv_width, uv_height, base.uv_quant, &mode_reader, &coeff_reader);
        try decodePlaneMotionTransform(v, prev_v, uv_width, uv_height, base.uv_quant, &mode_reader, &coeff_reader);

        prev_y = y;
        prev_u = u;
        prev_v = v;
    }

    return out;
}

const zigzag4x4 = [_]usize{ 0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15 };

fn encodePlaneTransform(plane: []const u8, width: u32, height: u32, quant: u32, out: *std.ArrayList(u8), zero_run: *u64) !void {
    var by: u32 = 0;
    while (by < height) : (by += transform_block_size) {
        var bx: u32 = 0;
        while (bx < width) : (bx += transform_block_size) {
            var block: [16]i32 = undefined;
            loadCenteredBlock(plane, width, height, bx, by, &block);
            hadamard4x4(&block);
            for (zigzag4x4) |idx| {
                const q = quantizeCoeff(block[idx], quant);
                try appendCoeffToken(out, zero_run, q);
            }
        }
    }
}

fn decodePlaneTransform(plane: []u8, width: u32, height: u32, quant: u32, reader: *CoeffReader) !void {
    var by: u32 = 0;
    while (by < height) : (by += transform_block_size) {
        var bx: u32 = 0;
        while (bx < width) : (bx += transform_block_size) {
            var block: [16]i32 = .{0} ** 16;
            for (zigzag4x4) |idx| {
                block[idx] = try reader.nextCoeff();
                block[idx] *= @intCast(quant);
            }
            inverseHadamard4x4(&block);
            storeBlock(plane, width, height, bx, by, &block);
        }
    }
}

fn encodePlanePredictiveTransform(plane: []const u8, width: u32, height: u32, quant: u32, out: *std.ArrayList(u8), zero_run: *u64) !void {
    const recon = try allocator.alloc(u8, plane.len);
    defer allocator.free(recon);
    @memset(recon, 128);

    var by: u32 = 0;
    while (by < height) : (by += transform_block_size) {
        var bx: u32 = 0;
        while (bx < width) : (bx += transform_block_size) {
            var residual: [16]i32 = undefined;
            loadPredictedResidualBlock(plane, recon, width, height, bx, by, &residual);
            hadamard4x4(&residual);
            for (zigzag4x4) |idx| {
                const q = quantizeCoeff(residual[idx], quant);
                try appendCoeffToken(out, zero_run, q);
                residual[idx] = q * @as(i32, @intCast(quant));
            }
            inverseHadamard4x4(&residual);
            storePredictedBlock(recon, width, height, bx, by, &residual);
        }
    }
}

fn decodePlanePredictiveTransform(plane: []u8, width: u32, height: u32, quant: u32, reader: *CoeffReader) !void {
    @memset(plane, 128);

    var by: u32 = 0;
    while (by < height) : (by += transform_block_size) {
        var bx: u32 = 0;
        while (bx < width) : (bx += transform_block_size) {
            var residual: [16]i32 = .{0} ** 16;
            for (zigzag4x4) |idx| {
                residual[idx] = try reader.nextCoeff();
                residual[idx] *= @intCast(quant);
            }
            inverseHadamard4x4(&residual);
            storePredictedBlock(plane, width, height, bx, by, &residual);
        }
    }
}

fn encodePlaneMotionTransform(source: []const u8, prev_recon: ?[]const u8, width: u32, height: u32, quant: u32, mode_out: *std.ArrayList(u8), coeff_out: *std.ArrayList(u8), zero_run: *u64) ![]u8 {
    const recon = try allocator.alloc(u8, source.len);
    errdefer allocator.free(recon);
    @memset(recon, 128);

    var by: u32 = 0;
    while (by < height) : (by += transform_block_size) {
        var bx: u32 = 0;
        while (bx < width) : (bx += transform_block_size) {
            const mode = chooseMotionMode(source, recon, prev_recon, width, height, bx, by);
            try appendVarU64(mode_out, mode);

            var residual: [16]i32 = undefined;
            loadMotionResidualBlock(source, recon, prev_recon, width, height, bx, by, mode, &residual);
            hadamard4x4(&residual);
            for (zigzag4x4) |idx| {
                const q = quantizeCoeff(residual[idx], quant);
                try appendCoeffToken(coeff_out, zero_run, q);
                residual[idx] = q * @as(i32, @intCast(quant));
            }
            inverseHadamard4x4(&residual);
            storeMotionBlock(recon, prev_recon, width, height, bx, by, mode, &residual);
        }
    }

    return recon;
}

fn decodePlaneMotionTransform(plane: []u8, prev_recon: ?[]const u8, width: u32, height: u32, quant: u32, mode_reader: *ModeReader, coeff_reader: *CoeffReader) !void {
    @memset(plane, 128);

    var by: u32 = 0;
    while (by < height) : (by += transform_block_size) {
        var bx: u32 = 0;
        while (bx < width) : (bx += transform_block_size) {
            const mode = try mode_reader.next();
            if (mode >= motion_mode_count) return error.InvalidMotionMode;
            if (mode != 0 and prev_recon == null) return error.InvalidMotionMode;

            var residual: [16]i32 = .{0} ** 16;
            for (zigzag4x4) |idx| {
                residual[idx] = try coeff_reader.nextCoeff();
                residual[idx] *= @intCast(quant);
            }
            inverseHadamard4x4(&residual);
            storeMotionBlock(plane, prev_recon, width, height, bx, by, mode, &residual);
        }
    }
}

fn chooseMotionMode(source: []const u8, recon: []const u8, prev_recon: ?[]const u8, width: u32, height: u32, bx: u32, by: u32) u64 {
    var best_mode: u64 = 0;
    var best_score = blockPredictionSad(source, recon, prev_recon, width, height, bx, by, 0);
    if (prev_recon == null) return best_mode;

    var mode: u64 = 1;
    while (mode < motion_mode_count) : (mode += 1) {
        const score = blockPredictionSad(source, recon, prev_recon, width, height, bx, by, mode) + modeTokenPenalty(mode);
        if (score < best_score) {
            best_score = score;
            best_mode = mode;
        }
    }
    return best_mode;
}

fn blockPredictionSad(source: []const u8, recon: []const u8, prev_recon: ?[]const u8, width: u32, height: u32, bx: u32, by: u32, mode: u64) u64 {
    var sad: u64 = 0;
    var y: u32 = 0;
    while (y < transform_block_size and by + y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < transform_block_size and bx + x < width) : (x += 1) {
            const xx = bx + x;
            const yy = by + y;
            const index = @as(usize, yy) * @as(usize, width) + @as(usize, xx);
            const pred = motionPredictSample(recon, prev_recon, width, height, bx, by, x, y, mode);
            sad += absDiff(source[index], pred);
        }
    }
    return sad;
}

fn loadMotionResidualBlock(source: []const u8, recon: []const u8, prev_recon: ?[]const u8, width: u32, height: u32, bx: u32, by: u32, mode: u64, block: *[16]i32) void {
    var y: u32 = 0;
    while (y < transform_block_size) : (y += 1) {
        const yy = @min(by + y, height - 1);
        var x: u32 = 0;
        while (x < transform_block_size) : (x += 1) {
            const xx = @min(bx + x, width - 1);
            const index = @as(usize, yy) * @as(usize, width) + @as(usize, xx);
            const pred = motionPredictSample(recon, prev_recon, width, height, bx, by, x, y, mode);
            block[@as(usize, y) * transform_block_size + @as(usize, x)] = @as(i32, source[index]) - @as(i32, pred);
        }
    }
}

fn storeMotionBlock(plane: []u8, prev_recon: ?[]const u8, width: u32, height: u32, bx: u32, by: u32, mode: u64, residual: *const [16]i32) void {
    var y: u32 = 0;
    while (y < transform_block_size and by + y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < transform_block_size and bx + x < width) : (x += 1) {
            const pred = motionPredictSample(plane, prev_recon, width, height, bx, by, x, y, mode);
            const value = @as(i32, pred) + residual[@as(usize, y) * transform_block_size + @as(usize, x)];
            const index = @as(usize, by + y) * @as(usize, width) + @as(usize, bx + x);
            plane[index] = clampToU8(value);
        }
    }
}

fn motionPredictSample(recon: []const u8, prev_recon: ?[]const u8, width: u32, height: u32, bx: u32, by: u32, x: u32, y: u32, mode: u64) u8 {
    if (mode == 0 or prev_recon == null) return predictBlockSample(recon, width, height, bx, by, x, y);

    const mv = motionVector(mode);
    const src_x = clampI32(@as(i32, @intCast(bx + x)) + mv.dx, 0, @as(i32, @intCast(width - 1)));
    const src_y = clampI32(@as(i32, @intCast(by + y)) + mv.dy, 0, @as(i32, @intCast(height - 1)));
    return prev_recon.?[@as(usize, @intCast(src_y)) * @as(usize, width) + @as(usize, @intCast(src_x))];
}

fn motionVector(mode: u64) struct { dx: i32, dy: i32 } {
    return switch (mode) {
        1 => .{ .dx = 0, .dy = 0 },
        2 => .{ .dx = -@as(i32, transform_block_size), .dy = 0 },
        3 => .{ .dx = @as(i32, transform_block_size), .dy = 0 },
        4 => .{ .dx = 0, .dy = -@as(i32, transform_block_size) },
        5 => .{ .dx = 0, .dy = @as(i32, transform_block_size) },
        else => .{ .dx = 0, .dy = 0 },
    };
}

fn modeTokenPenalty(mode: u64) u64 {
    return if (mode == 1) 4 else 8;
}

fn absDiff(a: u8, b: u8) u64 {
    return if (a >= b) @as(u64, a - b) else @as(u64, b - a);
}

fn clampI32(value: i32, lo: i32, hi: i32) i32 {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

fn clampI64(value: i64, lo: i64, hi: i64) i64 {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

fn clampF64(value: f64, lo: f64, hi: f64) f64 {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

fn loadPredictedResidualBlock(source: []const u8, recon: []const u8, width: u32, height: u32, bx: u32, by: u32, block: *[16]i32) void {
    var y: u32 = 0;
    while (y < transform_block_size) : (y += 1) {
        const yy = @min(by + y, height - 1);
        var x: u32 = 0;
        while (x < transform_block_size) : (x += 1) {
            const xx = @min(bx + x, width - 1);
            const index = @as(usize, yy) * @as(usize, width) + @as(usize, xx);
            const pred = predictBlockSample(recon, width, height, bx, by, x, y);
            block[@as(usize, y) * transform_block_size + @as(usize, x)] = @as(i32, source[index]) - @as(i32, pred);
        }
    }
}

fn storePredictedBlock(plane: []u8, width: u32, height: u32, bx: u32, by: u32, residual: *const [16]i32) void {
    var y: u32 = 0;
    while (y < transform_block_size and by + y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < transform_block_size and bx + x < width) : (x += 1) {
            const xx = bx + x;
            const yy = by + y;
            const pred = predictBlockSample(plane, width, height, bx, by, x, y);
            const value = @as(i32, pred) + residual[@as(usize, y) * transform_block_size + @as(usize, x)];
            const index = @as(usize, yy) * @as(usize, width) + @as(usize, xx);
            plane[index] = clampToU8(value);
        }
    }
}

fn predictBlockSample(recon: []const u8, width: u32, height: u32, bx: u32, by: u32, x: u32, y: u32) u8 {
    const xx = @min(bx + x, width - 1);
    const yy = @min(by + y, height - 1);
    const has_left = bx > 0;
    const has_top = by > 0;
    if (has_left and has_top) {
        const left = recon[@as(usize, yy) * @as(usize, width) + @as(usize, bx - 1)];
        const top = recon[@as(usize, by - 1) * @as(usize, width) + @as(usize, xx)];
        return @intCast((@as(u16, left) + @as(u16, top) + 1) / 2);
    }
    if (has_left) return recon[@as(usize, yy) * @as(usize, width) + @as(usize, bx - 1)];
    if (has_top) return recon[@as(usize, by - 1) * @as(usize, width) + @as(usize, xx)];
    return 128;
}

fn loadCenteredBlock(plane: []const u8, width: u32, height: u32, bx: u32, by: u32, block: *[16]i32) void {
    var y: u32 = 0;
    while (y < transform_block_size) : (y += 1) {
        const yy = @min(by + y, height - 1);
        var x: u32 = 0;
        while (x < transform_block_size) : (x += 1) {
            const xx = @min(bx + x, width - 1);
            const index = @as(usize, yy) * @as(usize, width) + @as(usize, xx);
            block[@as(usize, y) * transform_block_size + @as(usize, x)] = @as(i32, plane[index]) - 128;
        }
    }
}

fn storeBlock(plane: []u8, width: u32, height: u32, bx: u32, by: u32, block: *const [16]i32) void {
    var y: u32 = 0;
    while (y < transform_block_size and by + y < height) : (y += 1) {
        var x: u32 = 0;
        while (x < transform_block_size and bx + x < width) : (x += 1) {
            const value = block[@as(usize, y) * transform_block_size + @as(usize, x)] + 128;
            const index = @as(usize, by + y) * @as(usize, width) + @as(usize, bx + x);
            plane[index] = clampToU8(value);
        }
    }
}

fn hadamard4x4(block: *[16]i32) void {
    var y: usize = 0;
    while (y < 4) : (y += 1) {
        const i = y * 4;
        const a0 = block[i + 0] + block[i + 1];
        const a1 = block[i + 0] - block[i + 1];
        const a2 = block[i + 2] + block[i + 3];
        const a3 = block[i + 2] - block[i + 3];
        block[i + 0] = a0 + a2;
        block[i + 1] = a1 + a3;
        block[i + 2] = a0 - a2;
        block[i + 3] = a1 - a3;
    }

    var x: usize = 0;
    while (x < 4) : (x += 1) {
        const a0 = block[x + 0] + block[x + 4];
        const a1 = block[x + 0] - block[x + 4];
        const a2 = block[x + 8] + block[x + 12];
        const a3 = block[x + 8] - block[x + 12];
        block[x + 0] = a0 + a2;
        block[x + 4] = a1 + a3;
        block[x + 8] = a0 - a2;
        block[x + 12] = a1 - a3;
    }
}

fn inverseHadamard4x4(block: *[16]i32) void {
    hadamard4x4(block);
    for (block) |*value| {
        value.* = divRound(value.*, 16);
    }
}

fn quantizeCoeff(value: i32, quant: u32) i32 {
    return divRound(value, @intCast(@max(quant, 1)));
}

fn divRound(value: i32, denominator: i32) i32 {
    if (value >= 0) return @divTrunc(value + @divTrunc(denominator, 2), denominator);
    return -@divTrunc(-value + @divTrunc(denominator, 2), denominator);
}

fn divRoundI64(value: i64, denominator: i64) i64 {
    if (denominator == 0) return 0;
    if (value >= 0) return @divTrunc(value + @divTrunc(denominator, 2), denominator);
    return -@divTrunc(-value + @divTrunc(denominator, 2), denominator);
}

fn clampToU8(value: i32) u8 {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return @intCast(value);
}

fn entropyEncode(data: []const u8) ![]u8 {
    if (data.len == 0) return error.EmptyEntropyInput;
    const lengths = try huffmanCodeLengths(data);
    const codes = canonicalCodes(lengths);

    var bits = BitWriter{};
    defer bits.deinit();
    for (data) |symbol| {
        try bits.write(codes[symbol], lengths[symbol]);
    }
    const bitstream = try bits.finish();
    defer allocator.free(bitstream);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    try out.appendSlice(allocator, "HUF0");
    try appendU64(&out, @intCast(data.len));
    try out.appendSlice(allocator, lengths[0..]);
    try out.appendSlice(allocator, bitstream);
    return out.toOwnedSlice(allocator);
}

fn entropyDecode(payload: []const u8) ![]u8 {
    if (payload.len < 4 + 8 + 256) return error.InvalidEntropyStream;
    if (!std.mem.eql(u8, payload[0..4], "HUF0")) return error.InvalidEntropyStream;
    const raw_len = try castUsize(readU64(payload[4..12]));

    var lengths: [256]u8 = undefined;
    @memcpy(lengths[0..], payload[12 .. 12 + 256]);
    const codes = canonicalCodes(lengths);
    var tree = try HuffmanDecodeTree.init(lengths, codes);
    defer tree.deinit();

    var reader = BitReader{ .data = payload[12 + 256 ..] };
    const out = try allocator.alloc(u8, raw_len);
    errdefer allocator.free(out);
    for (out) |*byte| {
        byte.* = try tree.nextSymbol(&reader);
    }
    return out;
}

const HuffmanNode = struct {
    freq: u64,
    parent: i16 = -1,
};

fn huffmanCodeLengths(data: []const u8) ![256]u8 {
    var counts = [_]u64{0} ** 256;
    var active_symbols: usize = 0;
    for (data) |byte| {
        if (counts[byte] == 0) active_symbols += 1;
        counts[byte] += 1;
    }

    var lengths = [_]u8{0} ** 256;
    if (active_symbols == 1) {
        for (counts, 0..) |count, i| {
            if (count != 0) {
                lengths[i] = 1;
                return lengths;
            }
        }
    }

    var nodes = [_]HuffmanNode{.{ .freq = 0 }} ** 512;
    var node_count: usize = 0;
    var symbol_node = [_]i16{-1} ** 256;
    for (counts, 0..) |count, i| {
        if (count == 0) continue;
        nodes[node_count] = .{ .freq = count };
        symbol_node[i] = @intCast(node_count);
        node_count += 1;
    }

    while (true) {
        const pair = findHuffmanPair(nodes[0..node_count]) orelse break;
        const parent_index: i16 = @intCast(node_count);
        nodes[pair.a].parent = parent_index;
        nodes[pair.b].parent = parent_index;
        nodes[node_count] = .{ .freq = nodes[pair.a].freq + nodes[pair.b].freq };
        node_count += 1;
    }

    for (symbol_node, 0..) |node_index, symbol| {
        if (node_index < 0) continue;
        var depth: u8 = 0;
        var current = node_index;
        while (nodes[@intCast(current)].parent >= 0) {
            depth += 1;
            current = nodes[@intCast(current)].parent;
        }
        if (depth == 0) depth = 1;
        if (depth > 56) return error.HuffmanCodeTooDeep;
        lengths[symbol] = depth;
    }

    return lengths;
}

fn findHuffmanPair(nodes: []const HuffmanNode) ?struct { a: usize, b: usize } {
    var first: ?usize = null;
    var second: ?usize = null;
    for (nodes, 0..) |node, i| {
        if (node.parent >= 0) continue;
        if (first == null or node.freq < nodes[first.?].freq) {
            second = first;
            first = i;
        } else if (second == null or node.freq < nodes[second.?].freq) {
            second = i;
        }
    }
    if (first == null or second == null) return null;
    return .{ .a = first.?, .b = second.? };
}

fn canonicalCodes(lengths: [256]u8) [256]u64 {
    var codes = [_]u64{0} ** 256;
    var code: u64 = 0;
    var prev_len: u8 = 0;
    var assigned = [_]bool{false} ** 256;
    var assigned_count: usize = 0;
    while (assigned_count < 256) {
        var best: ?usize = null;
        for (lengths, 0..) |len, symbol| {
            if (len == 0 or assigned[symbol]) continue;
            if (best == null or len < lengths[best.?] or (len == lengths[best.?] and symbol < best.?)) {
                best = symbol;
            }
        }
        if (best == null) break;
        const symbol = best.?;
        const len = lengths[symbol];
        code <<= @intCast(len - prev_len);
        codes[symbol] = code;
        code += 1;
        prev_len = len;
        assigned[symbol] = true;
        assigned_count += 1;
    }
    return codes;
}

const BitWriter = struct {
    out: std.ArrayList(u8) = .empty,
    current: u8 = 0,
    used: u4 = 0,

    fn deinit(self: *BitWriter) void {
        self.out.deinit(allocator);
    }

    fn write(self: *BitWriter, code: u64, len: u8) !void {
        var remaining = len;
        while (remaining > 0) {
            remaining -= 1;
            const bit: u8 = @intCast((code >> @intCast(remaining)) & 1);
            self.current |= bit << @intCast(7 - self.used);
            self.used += 1;
            if (self.used == 8) {
                try self.out.append(allocator, self.current);
                self.current = 0;
                self.used = 0;
            }
        }
    }

    fn finish(self: *BitWriter) ![]u8 {
        if (self.used != 0) {
            try self.out.append(allocator, self.current);
            self.current = 0;
            self.used = 0;
        }
        return self.out.toOwnedSlice(allocator);
    }
};

const BitReader = struct {
    data: []const u8,
    offset: usize = 0,
    used: u4 = 0,

    fn next(self: *BitReader) !u1 {
        if (self.offset >= self.data.len) return error.TruncatedEntropyBits;
        const bit: u1 = @intCast((self.data[self.offset] >> @intCast(7 - self.used)) & 1);
        self.used += 1;
        if (self.used == 8) {
            self.used = 0;
            self.offset += 1;
        }
        return bit;
    }
};

const DecodeNode = struct {
    left: i16 = -1,
    right: i16 = -1,
    symbol: i16 = -1,
};

const HuffmanDecodeTree = struct {
    nodes: []DecodeNode,
    len: usize = 1,

    fn init(lengths: [256]u8, codes: [256]u64) !HuffmanDecodeTree {
        var tree = HuffmanDecodeTree{ .nodes = try allocator.alloc(DecodeNode, 512) };
        @memset(tree.nodes, .{});
        for (lengths, 0..) |len, symbol| {
            if (len == 0) continue;
            try tree.insert(codes[symbol], len, @intCast(symbol));
        }
        return tree;
    }

    fn deinit(self: *HuffmanDecodeTree) void {
        allocator.free(self.nodes);
    }

    fn insert(self: *HuffmanDecodeTree, code: u64, len: u8, symbol: u8) !void {
        var node: usize = 0;
        var remaining = len;
        while (remaining > 0) {
            remaining -= 1;
            const bit = (code >> @intCast(remaining)) & 1;
            const next_ptr = if (bit == 0) &self.nodes[node].left else &self.nodes[node].right;
            if (next_ptr.* < 0) {
                if (self.len >= self.nodes.len) return error.HuffmanTreeTooLarge;
                next_ptr.* = @intCast(self.len);
                self.nodes[self.len] = .{};
                self.len += 1;
            }
            node = @intCast(next_ptr.*);
        }
        self.nodes[node].symbol = symbol;
    }

    fn nextSymbol(self: *HuffmanDecodeTree, reader: *BitReader) !u8 {
        var node: usize = 0;
        while (self.nodes[node].symbol < 0) {
            const bit = try reader.next();
            const next = if (bit == 0) self.nodes[node].left else self.nodes[node].right;
            if (next < 0) return error.InvalidEntropyBits;
            node = @intCast(next);
        }
        return @intCast(self.nodes[node].symbol);
    }
};

fn appendCoeffToken(out: *std.ArrayList(u8), zero_run: *u64, coeff: i32) !void {
    if (coeff == 0) {
        zero_run.* += 1;
        return;
    }
    try flushZeroRun(out, zero_run);
    try appendVarU64(out, zigZagI32(coeff) + 1);
}

fn flushZeroRun(out: *std.ArrayList(u8), zero_run: *u64) !void {
    if (zero_run.* == 0) return;
    try appendVarU64(out, 0);
    try appendVarU64(out, zero_run.*);
    zero_run.* = 0;
}

const CoeffReader = struct {
    data: []const u8,
    offset: usize = 0,
    zero_run: u64 = 0,

    fn alignToToken(self: *CoeffReader) !void {
        if (self.zero_run != 0) return error.InvalidCoeffStream;
    }

    fn nextMode(self: *CoeffReader) !u64 {
        try self.alignToToken();
        return readVarU64(self.data, &self.offset);
    }

    fn nextCoeff(self: *CoeffReader) !i32 {
        if (self.zero_run > 0) {
            self.zero_run -= 1;
            return 0;
        }

        const token = try readVarU64(self.data, &self.offset);
        if (token == 0) {
            const count = try readVarU64(self.data, &self.offset);
            if (count == 0) return error.InvalidCoeffStream;
            self.zero_run = count - 1;
            return 0;
        }
        return unZigZagI32(token - 1);
    }
};

const ModeReader = struct {
    data: []const u8,
    offset: usize = 0,

    fn next(self: *ModeReader) !u64 {
        if (self.offset >= self.data.len) return error.TruncatedModeStream;
        const mode = self.data[self.offset];
        self.offset += 1;
        return mode;
    }
};

fn expectedMotionModeCount(width: u32, height: u32, frame_count: u32) !usize {
    const y_blocks = blockCount(width, height);
    const uv_blocks = blockCount(width / 2, height / 2);
    const per_frame = y_blocks + uv_blocks * 2;
    return @intCast(@as(u64, frame_count) * @as(u64, per_frame));
}

fn blockCount(width: u32, height: u32) u64 {
    const bx = (@as(u64, width) + transform_block_size - 1) / transform_block_size;
    const by = (@as(u64, height) + transform_block_size - 1) / transform_block_size;
    return bx * by;
}

fn appendVarU64(out: *std.ArrayList(u8), value_in: u64) !void {
    var value = value_in;
    while (value >= 0x80) {
        try out.append(allocator, @intCast((value & 0x7f) | 0x80));
        value >>= 7;
    }
    try out.append(allocator, @intCast(value));
}

fn readVarU64(data: []const u8, offset: *usize) !u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (offset.* < data.len) {
        const byte = data[offset.*];
        offset.* += 1;
        result |= @as(u64, byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) return result;
        if (shift >= 63) return error.InvalidVarint;
        shift += 7;
    }
    return error.TruncatedVarint;
}

fn zigZagI32(value: i32) u64 {
    const shifted: i64 = (@as(i64, value) << 1) ^ (@as(i64, value) >> 31);
    return @intCast(shifted);
}

fn unZigZagI32(value: u64) i32 {
    const half: i64 = @intCast(value >> 1);
    const sign: i64 = -@as(i64, @intCast(value & 1));
    return @intCast(half ^ sign);
}

fn rleCompress(data: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    var i: usize = 0;
    while (i < data.len) {
        const value = data[i];
        var count: u8 = 1;
        while (i + count < data.len and count < 255 and data[i + count] == value) {
            count += 1;
        }
        try out.append(allocator, count);
        try out.append(allocator, value);
        i += count;
    }
    return out.toOwnedSlice(allocator);
}

fn rleDecompress(data: []const u8, expected: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(allocator);
    var i: usize = 0;
    while (i + 1 < data.len and out.items.len < expected) : (i += 2) {
        const count = data[i];
        const value = data[i + 1];
        var n: u8 = 0;
        while (n < count) : (n += 1) {
            try out.append(allocator, value);
        }
    }
    if (out.items.len != expected) return error.InvalidRleStream;
    return out.toOwnedSlice(allocator);
}

fn crc32(data: []const u8) u32 {
    var crc: u32 = 0xffffffff;
    for (data) |byte| {
        crc ^= byte;
        var i: u8 = 0;
        while (i < 8) : (i += 1) {
            const mask: u32 = if ((crc & 1) == 1) 0xedb88320 else 0;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

fn appendU16(out: *std.ArrayList(u8), value: u16) !void {
    try out.append(allocator, @intCast(value & 0xff));
    try out.append(allocator, @intCast((value >> 8) & 0xff));
}

fn appendU32(out: *std.ArrayList(u8), value: u32) !void {
    try out.append(allocator, @intCast(value & 0xff));
    try out.append(allocator, @intCast((value >> 8) & 0xff));
    try out.append(allocator, @intCast((value >> 16) & 0xff));
    try out.append(allocator, @intCast((value >> 24) & 0xff));
}

fn appendF32(out: *std.ArrayList(u8), value: f32) !void {
    try appendU32(out, @as(u32, @bitCast(value)));
}

fn appendU64(out: *std.ArrayList(u8), value: u64) !void {
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const shift: u6 = @intCast(i * 8);
        try out.append(allocator, @intCast((value >> shift) & 0xff));
    }
}

fn readU32(bytes: []const u8) u32 {
    return @as(u32, bytes[0]) |
        (@as(u32, bytes[1]) << 8) |
        (@as(u32, bytes[2]) << 16) |
        (@as(u32, bytes[3]) << 24);
}

fn readF32(bytes: []const u8) f32 {
    return @as(f32, @bitCast(readU32(bytes)));
}

fn readU16(bytes: []const u8) u16 {
    return @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
}

fn readU64(bytes: []const u8) u64 {
    var value: u64 = 0;
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        value |= @as(u64, bytes[i]) << @intCast(i * 8);
    }
    return value;
}

fn castUsize(value: u64) !usize {
    if (value > std.math.maxInt(usize)) return error.FileTooLarge;
    return @intCast(value);
}

test "rle roundtrip" {
    const input = "aaaabbbbccccccccxyz";
    const compressed = try rleCompress(input);
    defer allocator.free(compressed);
    const decompressed = try rleDecompress(compressed, input.len);
    defer allocator.free(decompressed);
    try std.testing.expectEqualSlices(u8, input, decompressed);
}

test "crc32 known vector" {
    try std.testing.expectEqual(@as(u32, 0xcbf43926), crc32("123456789"));
}

test "frame limit parser supports full encode" {
    try std.testing.expectEqual(@as(u32, 0), try parseFrameLimit("all"));
    try std.testing.expectEqual(@as(u32, 0), try parseFrameLimit("full"));
    try std.testing.expectEqual(@as(u32, 0), try parseFrameLimit("0"));
    try std.testing.expectEqual(@as(u32, 120), try parseFrameLimit("120"));
}

test "quarter-turn rotation detection" {
    try std.testing.expect(isQuarterTurn(90));
    try std.testing.expect(isQuarterTurn(-90));
    try std.testing.expect(isQuarterTurn(270));
    try std.testing.expect(!isQuarterTurn(0));
    try std.testing.expect(!isQuarterTurn(180));
}

test "entropy coder roundtrip" {
    const input = "aaaaaaaabbbbccccccccccccddddxyzxyzxyz0000000000000000";
    const encoded = try entropyEncode(input);
    defer allocator.free(encoded);
    const decoded = try entropyDecode(encoded);
    defer allocator.free(decoded);
    try std.testing.expectEqualSlices(u8, input, decoded);
}

test "container roundtrip" {
    const path = "/private/tmp/nvc-container-test.nvc";
    defer std.fs.cwd().deleteFile(path) catch {};
    const extra = [_]Chunk{
        makeChunk("MODL", "model"),
        makeChunk("BASE", "base"),
    };
    try writeNvc(path, "profile=NVC-W1\nwidth=2\nheight=2\n", &extra);
    const parsed = try readNvc(path);
    defer parsed.deinit();
    try std.testing.expect(parsed.find("HEAD") != null);
    try std.testing.expect(parsed.find("TOC0") != null);
    try std.testing.expect(parsed.find("BASE") != null);
}

test "BAS4 entropy motion transform base roundtrip dimensions" {
    const width: u32 = 8;
    const height: u32 = 8;
    const frame_count: u32 = 1;
    const frame_size = yuv420FrameSize(width, height);
    const raw = try allocator.alloc(u8, frame_size);
    defer allocator.free(raw);

    for (raw, 0..) |*byte, i| {
        byte.* = @intCast((i * 13 + 31) % 256);
    }

    const packed_data = try encodeBaseMotionTransform(raw, width, height, frame_count, 4, 8);
    defer allocator.free(packed_data);
    const coded = try entropyEncode(packed_data);
    defer allocator.free(coded);
    const payload = try buildBase4Payload(width, height, 30, 1, frame_count, raw.len, coded, 4, 8);
    defer allocator.free(payload);
    const base = try parseBasePayload(payload);
    const decoded = try decodeBasePayload(base);
    defer allocator.free(decoded);

    try std.testing.expectEqual(raw.len, decoded.len);
    try std.testing.expectEqual(@as(u32, 8), base.width);
    try std.testing.expectEqual(@as(u32, 8), base.height);
    try std.testing.expectEqual(BaseFormat.entropy_motion_transform, base.format);
    try std.testing.expect(coded.len > 0);
}

test "BAS5 packetized entropy motion transform base roundtrip dimensions" {
    const width: u32 = 8;
    const height: u32 = 8;
    const frame_count: u32 = 3;
    const frame_size = yuv420FrameSize(width, height);
    const raw = try allocator.alloc(u8, frame_size * frame_count);
    defer allocator.free(raw);

    for (raw, 0..) |*byte, i| {
        byte.* = @intCast((i * 17 + 19) % 256);
    }

    const built = try buildBasePayload(raw, width, height, 30, 1, frame_count, 4, 8, 2);
    defer allocator.free(built.payload);
    const base = try parseBasePayload(built.payload);
    const decoded = try decodeBasePayload(base);
    defer allocator.free(decoded);

    try std.testing.expectEqual(raw.len, decoded.len);
    try std.testing.expectEqual(@as(u32, 8), base.width);
    try std.testing.expectEqual(@as(u32, 8), base.height);
    try std.testing.expectEqual(BaseFormat.packetized_entropy_motion_transform, base.format);
    try std.testing.expectEqual(@as(u32, 2), base.gop_size);
    try std.testing.expectEqual(@as(u32, 2), base.packet_count);
    try std.testing.expect(built.coded_size > 0);
}

test "FET1 feature residual payload roundtrip" {
    const width: u32 = 16;
    const height: u32 = 16;
    const frame_count: u32 = 2;
    const frame_size = yuv420FrameSize(width, height);
    const raw = try allocator.alloc(u8, frame_size * frame_count);
    defer allocator.free(raw);
    const decoded = try allocator.alloc(u8, frame_size * frame_count);
    defer allocator.free(decoded);

    for (raw, 0..) |*byte, i| {
        byte.* = @intCast((i * 5 + 93) % 256);
    }
    for (decoded, 0..) |*byte, i| {
        byte.* = @intCast((i * 5 + 89) % 256);
    }

    const payload = try buildFeaturePayload(raw, decoded, width, height, frame_count, .w1);
    defer allocator.free(payload);
    const feat = try parseFeaturePayload(payload);
    const residuals = try entropyDecode(feat.coded);
    defer allocator.free(residuals);

    try std.testing.expectEqual(@as(u32, width), feat.width);
    try std.testing.expectEqual(@as(u32, height), feat.height);
    try std.testing.expectEqual(@as(u32, frame_count), feat.frame_count);
    try std.testing.expectEqual(@as(usize, @intCast(feat.residual_count)), residuals.len);
    try std.testing.expect(feat.coded_size > 0);
}
