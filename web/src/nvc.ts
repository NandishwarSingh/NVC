export type NvcChunk = {
  id: string;
  offset: number;
  length: number;
  crc: number;
  flags: number;
  payload: Uint8Array;
};

export type NvcFile = {
  version: string;
  chunks: NvcChunk[];
  head: Record<string, string>;
  model?: ModlInfo;
  feature?: FeatureInfo;
  color?: ColorInfo;
  grain?: GrainInfo;
  sourceUrl?: string;
  fileBytes?: number;
  toc?: TocEntry[];
  baseIndex?: BaseIndex;
  loadedBasePacketIndex?: number;
  baseFrameCache?: BaseFrameCache;
  baseFrameCaches?: Map<number, BaseFrameCache>;
  baseFrameCacheOrder?: number[];
};

export type NvcLoadStats = {
  loadMode: "range" | "full" | "chunk" | "cache";
  bytesLoaded: number;
  fileBytes?: number;
  chunksLoaded: string[];
};

export type NvcLoadResult = {
  file: NvcFile;
  stats: NvcLoadStats;
};

export type ModlInfo = {
  format: string;
  version?: string;
  modelId?: string;
  architecture?: string;
  metadataBytes?: number;
  weightsBytes?: number;
  metadata?: Mod0Metadata;
  weights?: Float32Array;
};

export type ColorInfo = {
  format: string;
  lumaScale?: number;
  lumaBias?: number;
  saturation?: number;
  contrast?: number;
};

export type GrainInfo = {
  format: string;
  seed?: number;
  intensity?: number;
  lumaOnly?: boolean;
};

export type FeatureInfo = {
  format: string;
  width?: number;
  height?: number;
  frameCount?: number;
  tileSize?: number;
  quantStep?: number;
  gridWidth?: number;
  gridHeight?: number;
  residualCount?: number;
  codedBytes?: number;
  residuals?: Uint8Array;
};

const textDecoder = new TextDecoder();

type Mod0Metadata = {
  model_id?: string;
  architecture?: string;
  layers?: Mod0Layer[];
};

type Mod0Layer = {
  name: string;
  weight_shape: number[];
  bias_shape: number[];
  weight_offset: number;
  bias_offset: number;
};

type BaseRgbFrame = {
  width: number;
  height: number;
  rgb: Uint8ClampedArray;
};

type BaseFrameCache = {
  packetIndex: number;
  startFrame: number;
  frames: BaseRgbFrame[];
};

type Yuv420Frame = {
  width: number;
  height: number;
  y: Uint8Array;
  u: Uint8Array;
  v: Uint8Array;
};

export type NeuralRenderResult = {
  imageData: ImageData;
  backend: "webgpu" | "cpu";
  inputWidth: number;
  inputHeight: number;
  outputWidth: number;
  outputHeight: number;
  fallbackReason?: string;
};

export type PreviewInfo = {
  width: number;
  height: number;
  fpsNum: number;
  fpsDen: number;
  frameCount: number;
};

export type TocEntry = {
  id: string;
  offset: number;
  length: number;
  crc: number;
  flags: number;
};

export type BasePacketEntry = {
  startFrame: number;
  frameCount: number;
  payloadOffset: number;
  payloadSize: number;
  cachedPayload?: Uint8Array;
};

export type BaseIndex = {
  chunkOffset: number;
  chunkLength: number;
  headerAndTable: Uint8Array;
  packets: BasePacketEntry[];
  bytesLoaded: number;
};

export function parseNvc(buffer: ArrayBuffer): NvcFile {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 20 || readText(bytes, 0, 4) !== "NVCF") {
    throw new Error("Not an NVC file");
  }

  const major = view.getUint16(4, true);
  const minor = view.getUint16(6, true);
  const headerLength = view.getUint32(8, true);
  const chunks: NvcChunk[] = [];
  let offset = headerLength;

  while (offset < bytes.length) {
    if (bytes.length - offset < 20) throw new Error("Truncated chunk header");
    const id = readText(bytes, offset, 4);
    const length = Number(view.getBigUint64(offset + 4, true));
    const crc = view.getUint32(offset + 12, true);
    const flags = view.getUint32(offset + 16, true);
    const payloadOffset = offset + 20;
    const end = payloadOffset + length;
    if (end > bytes.length) throw new Error(`Truncated ${id} chunk`);
    chunks.push({
      id,
      offset,
      length,
      crc,
      flags,
      payload: bytes.slice(payloadOffset, end),
    });
    offset = end;
  }

  const headChunk = chunks.find((chunk) => chunk.id === "HEAD");
  const modlChunk = chunks.find((chunk) => chunk.id === "MODL");
  return buildNvcFile(`${major}.${minor}`, chunks, headChunk, modlChunk);
}

export async function loadNvcUrl(url: string): Promise<NvcLoadResult> {
  try {
    const prefix = await fetchByteRange(url, 0, 65535);
    if (prefix.kind === "full") return parseFullLoad(prefix.buffer);

    const index = parseNvcPrefix(prefix.buffer);
    const chunks = [...index.chunks];
    const loadedIds = new Set(chunks.map((chunk) => chunk.id));
    const desiredIds = ["PRVW", "MODL", "SEEK", "FEAT", "COLR", "GRAN"];
    let bytesLoaded = prefix.buffer.byteLength;

    for (const id of desiredIds) {
      if (loadedIds.has(id)) continue;
      const entry = index.toc.find((item) => item.id === id);
      if (!entry) continue;
      const chunk = await fetchChunkRange(url, entry);
      chunks.push(chunk);
      loadedIds.add(chunk.id);
      bytesLoaded += 20 + chunk.payload.byteLength;
    }

    const baseEntry = index.toc.find((item) => item.id === "BASE");
    const baseIndex = baseEntry ? await fetchBaseIndex(url, baseEntry) : undefined;
    if (baseIndex) {
      bytesLoaded += baseIndex.bytesLoaded;
    }

    if (!loadedIds.has("PRVW")) {
      throw new Error("Range preview requires a PRVW chunk");
    }

    chunks.sort((left, right) => left.offset - right.offset);
    return {
      file: buildNvcFile(index.version, chunks, undefined, undefined, { sourceUrl: url, toc: index.toc, fileBytes: prefix.total, baseIndex }),
      stats: {
        loadMode: "range",
        bytesLoaded,
        fileBytes: prefix.total,
        chunksLoaded: [...chunks.map((chunk) => chunk.id), ...(baseIndex ? ["BASE-index"] : [])],
      },
    };
  } catch {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to load ${url}`);
    return parseFullLoad(await response.arrayBuffer(), Number(response.headers.get("content-length") ?? 0) || undefined);
  }
}

export async function ensureNvcChunk(file: NvcFile, id: string): Promise<NvcLoadStats | null> {
  if (file.chunks.some((chunk) => chunk.id === id)) return null;
  if (!file.sourceUrl || !file.toc) throw new Error(`${id} is not loaded and this file has no range index`);
  const entry = file.toc.find((item) => item.id === id);
  if (!entry) throw new Error(`${id} is not listed in TOC0`);
  const chunk = await fetchChunkRange(file.sourceUrl, entry);
  file.chunks.push(chunk);
  file.chunks.sort((left, right) => left.offset - right.offset);
  if (id === "MODL") {
    file.model = parseModl(chunk.payload);
  } else if (id === "COLR") {
    file.color = parseColr(chunk.payload);
  } else if (id === "FEAT") {
    file.feature = parseFeat(chunk.payload);
  } else if (id === "GRAN") {
    file.grain = parseGran(chunk.payload);
  }
  return {
    loadMode: "chunk",
    bytesLoaded: 20 + chunk.payload.byteLength,
    fileBytes: file.fileBytes,
    chunksLoaded: [chunk.id],
  };
}

export async function ensureNvcBasePacket(file: NvcFile, frameIndex: number): Promise<NvcLoadStats | null> {
  if (!file.sourceUrl || !file.baseIndex) return ensureNvcChunk(file, "BASE");
  const packetIndex = file.baseIndex.packets.findIndex((packet) => frameIndex >= packet.startFrame && frameIndex < packet.startFrame + packet.frameCount);
  if (packetIndex < 0) throw new Error(`No BAS5 packet for frame ${frameIndex}`);
  const existingBase = file.chunks.find((chunk) => chunk.id === "BASE");
  if (existingBase && existingBase.flags !== 1) return null;
  if (existingBase && file.loadedBasePacketIndex === packetIndex) return null;
  if (existingBase) {
    file.chunks = file.chunks.filter((chunk) => chunk.id !== "BASE");
  }
  file.baseFrameCache = undefined;
  const packet = file.baseIndex.packets[packetIndex];
  let packetPayload = packet.cachedPayload;
  let loadMode: NvcLoadStats["loadMode"] = "cache";
  if (!packetPayload) {
    const start = file.baseIndex.chunkOffset + 20 + packet.payloadOffset;
    const end = start + packet.payloadSize - 1;
    const response = await fetchByteRange(file.sourceUrl, start, end);
    if (response.kind !== "partial") throw new Error("Server returned a full response for a BAS5 packet range");
    packetPayload = new Uint8Array(response.buffer);
    packet.cachedPayload = packetPayload;
    loadMode = "chunk";
  }
  const headerAndTable = file.baseIndex.headerAndTable.slice();
  writeU64(headerAndTable, 76 + packetIndex * 24 + 8, BigInt(headerAndTable.byteLength));
  const payload = new Uint8Array(headerAndTable.byteLength + packetPayload.byteLength);
  payload.set(headerAndTable, 0);
  payload.set(packetPayload, file.baseIndex.headerAndTable.byteLength);
  file.chunks.push({
    id: "BASE",
    offset: file.baseIndex.chunkOffset,
    length: payload.byteLength,
    crc: 0,
    flags: 1,
    payload,
  });
  file.loadedBasePacketIndex = packetIndex;
  file.chunks.sort((left, right) => left.offset - right.offset);
  return {
    loadMode,
    bytesLoaded: loadMode === "chunk" ? packetPayload.byteLength : 0,
    fileBytes: file.fileBytes,
    chunksLoaded: [`BASE packet ${packetIndex + 1}/${file.baseIndex.packets.length}`],
  };
}

function buildNvcFile(
  version: string,
  chunks: NvcChunk[],
  headChunk?: NvcChunk,
  modlChunk?: NvcChunk,
  options: { sourceUrl?: string; toc?: TocEntry[]; fileBytes?: number; baseIndex?: BaseIndex } = {},
): NvcFile {
  const head = headChunk ?? chunks.find((chunk) => chunk.id === "HEAD");
  const modl = modlChunk ?? chunks.find((chunk) => chunk.id === "MODL");
  const feat = chunks.find((chunk) => chunk.id === "FEAT");
  const colr = chunks.find((chunk) => chunk.id === "COLR");
  const gran = chunks.find((chunk) => chunk.id === "GRAN");
  return {
    version,
    chunks,
    head: head ? parseHead(head.payload) : {},
    model: modl ? parseModl(modl.payload) : undefined,
    feature: feat ? parseFeat(feat.payload) : undefined,
    color: colr ? parseColr(colr.payload) : undefined,
    grain: gran ? parseGran(gran.payload) : undefined,
    sourceUrl: options.sourceUrl,
    toc: options.toc,
    fileBytes: options.fileBytes,
    baseIndex: options.baseIndex,
  };
}

function parseFullLoad(buffer: ArrayBuffer, fileBytes = buffer.byteLength): NvcLoadResult {
  const file = parseNvc(buffer);
  return {
    file,
    stats: {
      loadMode: "full",
      bytesLoaded: buffer.byteLength,
      fileBytes,
      chunksLoaded: file.chunks.map((chunk) => chunk.id),
    },
  };
}

type RangeFetch =
  | { kind: "partial"; buffer: ArrayBuffer; start: number; end: number; total?: number }
  | { kind: "full"; buffer: ArrayBuffer };

async function fetchByteRange(url: string, start: number, end: number): Promise<RangeFetch> {
  const response = await fetch(url, { headers: { range: `bytes=${start}-${end}` } });
  if (response.status === 200) return { kind: "full", buffer: await response.arrayBuffer() };
  if (response.status !== 206) throw new Error(`Range request failed with HTTP ${response.status}`);
  const contentRange = response.headers.get("content-range");
  const parsed = contentRange?.match(/^bytes (\d+)-(\d+)\/(\d+|\*)$/);
  return {
    kind: "partial",
    buffer: await response.arrayBuffer(),
    start: parsed ? Number(parsed[1]) : start,
    end: parsed ? Number(parsed[2]) : end,
    total: parsed && parsed[3] !== "*" ? Number(parsed[3]) : undefined,
  };
}

function parseNvcPrefix(buffer: ArrayBuffer): { version: string; chunks: NvcChunk[]; toc: TocEntry[] } {
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 20 || readText(bytes, 0, 4) !== "NVCF") {
    throw new Error("Not an NVC file");
  }

  const major = view.getUint16(4, true);
  const minor = view.getUint16(6, true);
  const headerLength = view.getUint32(8, true);
  const chunks: NvcChunk[] = [];
  let offset = headerLength;

  while (offset + 20 <= bytes.length) {
    const length = Number(view.getBigUint64(offset + 4, true));
    const end = offset + 20 + length;
    if (end > bytes.length) break;
    chunks.push(parseChunkBytes(bytes.slice(offset, end), offset));
    offset = end;
  }

  const tocChunk = chunks.find((chunk) => chunk.id === "TOC0");
  if (!tocChunk) throw new Error("TOC0 chunk missing from range prefix");
  return {
    version: `${major}.${minor}`,
    chunks,
    toc: parseToc(tocChunk.payload),
  };
}

function parseToc(payload: Uint8Array): TocEntry[] {
  if (payload.length < 8 || readText(payload, 0, 4) !== "TOC0") throw new Error("Invalid TOC0 payload");
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const count = view.getUint32(4, true);
  if (payload.length < 8 + count * 28) throw new Error("Truncated TOC0 payload");
  const entries: TocEntry[] = [];
  for (let i = 0; i < count; i += 1) {
    const offset = 8 + i * 28;
    entries.push({
      id: readText(payload, offset, 4),
      offset: Number(view.getBigUint64(offset + 4, true)),
      length: Number(view.getBigUint64(offset + 12, true)),
      crc: view.getUint32(offset + 20, true),
      flags: view.getUint32(offset + 24, true),
    });
  }
  return entries;
}

async function fetchChunkRange(url: string, entry: TocEntry): Promise<NvcChunk> {
  const end = entry.offset + 20 + entry.length - 1;
  const response = await fetchByteRange(url, entry.offset, end);
  if (response.kind !== "partial") throw new Error("Server returned a full response for a chunk range");
  const chunk = parseChunkBytes(new Uint8Array(response.buffer), entry.offset);
  if (chunk.id !== entry.id || chunk.length !== entry.length) throw new Error(`Unexpected ${entry.id} chunk range`);
  return chunk;
}

async function fetchBaseIndex(url: string, entry: TocEntry): Promise<BaseIndex | undefined> {
  const header = await fetchByteRange(url, entry.offset, entry.offset + 20 + 75);
  if (header.kind !== "partial") return undefined;
  const bytes = new Uint8Array(header.buffer);
  if (bytes.length < 96 || readText(bytes, 0, 4) !== "BASE" || readText(bytes, 20, 4) !== "BAS5") return undefined;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const packetCount = view.getUint32(20 + 72, true);
  if (packetCount < 1) return undefined;
  const tableBytes = 76 + packetCount * 24;
  const tableEnd = entry.offset + 20 + tableBytes - 1;
  const tableFetch = await fetchByteRange(url, entry.offset, tableEnd);
  if (tableFetch.kind !== "partial") return undefined;
  const full = new Uint8Array(tableFetch.buffer);
  if (full.length < 20 + tableBytes) throw new Error("Truncated BAS5 index");
  const payload = full.slice(20, 20 + tableBytes);
  return {
    chunkOffset: entry.offset,
    chunkLength: entry.length,
    headerAndTable: payload,
    packets: parseBas5Packets(payload),
    bytesLoaded: bytes.byteLength + full.byteLength,
  };
}

function parseBas5Packets(payload: Uint8Array): BasePacketEntry[] {
  if (payload.length < 76 || readText(payload, 0, 4) !== "BAS5") throw new Error("Invalid BAS5 index");
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const packetCount = view.getUint32(72, true);
  if (payload.length < 76 + packetCount * 24) throw new Error("Truncated BAS5 packet table");
  const packets: BasePacketEntry[] = [];
  for (let i = 0; i < packetCount; i += 1) {
    const offset = 76 + i * 24;
    packets.push({
      startFrame: view.getUint32(offset, true),
      frameCount: view.getUint32(offset + 4, true),
      payloadOffset: Number(view.getBigUint64(offset + 8, true)),
      payloadSize: Number(view.getBigUint64(offset + 16, true)),
    });
  }
  return packets;
}

function parseChunkBytes(bytes: Uint8Array, absoluteOffset: number): NvcChunk {
  if (bytes.length < 20) throw new Error("Truncated chunk header");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const id = readText(bytes, 0, 4);
  const length = Number(view.getBigUint64(4, true));
  const end = 20 + length;
  if (end > bytes.length) throw new Error(`Truncated ${id} chunk`);
  return {
    id,
    offset: absoluteOffset,
    length,
    crc: view.getUint32(12, true),
    flags: view.getUint32(16, true),
    payload: bytes.slice(20, end),
  };
}

function writeU64(bytes: Uint8Array, offset: number, value: bigint): void {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  view.setBigUint64(offset, value, true);
}

function parseModl(payload: Uint8Array): ModlInfo {
  if (payload.length < 20 || readText(payload, 0, 4) !== "MOD0") {
    return { format: "legacy-placeholder", weightsBytes: payload.length };
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const major = view.getUint16(4, true);
  const minor = view.getUint16(6, true);
  const metadataBytes = view.getUint32(8, true);
  const weightsBytes = Number(view.getBigUint64(12, true));
  const metadataStart = 20;
  const metadataEnd = metadataStart + metadataBytes;
  if (metadataEnd > payload.length) throw new Error("Invalid MOD0 metadata length");
  const metadata = JSON.parse(textDecoder.decode(payload.slice(metadataStart, metadataEnd)));
  const weightsStart = metadataEnd;
  const weightsEnd = weightsStart + weightsBytes;
  if (weightsEnd > payload.length) throw new Error("Invalid MOD0 weights length");
  const weights = new Float32Array(weightsBytes / 4);
  const weightView = new DataView(payload.buffer, payload.byteOffset + weightsStart, weightsBytes);
  for (let i = 0; i < weights.length; i += 1) {
    weights[i] = weightView.getFloat32(i * 4, true);
  }
  return {
    format: "MOD0",
    version: `${major}.${minor}`,
    modelId: metadata.model_id,
    architecture: metadata.architecture,
    metadataBytes,
    weightsBytes,
    metadata,
    weights,
  };
}

function parseColr(payload: Uint8Array): ColorInfo {
  if (payload.length < 24 || readText(payload, 0, 4) !== "COL1") {
    return { format: "legacy-placeholder" };
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  return {
    format: "COL1",
    lumaScale: view.getFloat32(8, true),
    lumaBias: view.getFloat32(12, true),
    saturation: view.getFloat32(16, true),
    contrast: view.getFloat32(20, true),
  };
}

function parseGran(payload: Uint8Array): GrainInfo {
  if (payload.length < 20 || readText(payload, 0, 4) !== "GRN1") {
    return { format: "legacy-placeholder" };
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  return {
    format: "GRN1",
    seed: view.getUint32(8, true),
    intensity: view.getFloat32(12, true),
    lumaOnly: view.getUint32(16, true) !== 0,
  };
}

function parseFeat(payload: Uint8Array): FeatureInfo {
  if (payload.length < 52 || readText(payload, 0, 4) !== "FET1") {
    return { format: "legacy-placeholder" };
  }
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const residualCount = Number(view.getBigUint64(36, true));
  const codedBytes = Number(view.getBigUint64(44, true));
  const codedStart = 52;
  const codedEnd = codedStart + codedBytes;
  if (codedEnd > payload.length) throw new Error("Invalid FET1 coded length");
  const residuals = entropyDecode(payload.slice(codedStart, codedEnd));
  if (residuals.length !== residualCount) throw new Error("Invalid FET1 residual length");
  return {
    format: "FET1",
    width: view.getUint32(8, true),
    height: view.getUint32(12, true),
    frameCount: view.getUint32(16, true),
    tileSize: view.getUint32(20, true),
    quantStep: view.getUint32(24, true),
    gridWidth: view.getUint32(28, true),
    gridHeight: view.getUint32(32, true),
    residualCount,
    codedBytes,
    residuals,
  };
}

export function decodeFirstBaseFrame(file: NvcFile): ImageData {
  return decodeBaseFrame(file, 0);
}

export function decodeBaseFrame(file: NvcFile, frameIndex = 0, options: { preferPreview?: boolean } = {}): ImageData {
  const frame = decodeFirstBaseRgb(file, frameIndex, options.preferPreview ?? true);
  return rgbToImageData(frame.rgb, frame.width, frame.height);
}

export function decodeCodecBaseFrame(file: NvcFile, frameIndex = 0): ImageData {
  const frame = decodeFirstBaseRgb(file, frameIndex, false);
  return rgbToImageData(frame.rgb, frame.width, frame.height);
}

export async function decodeFirstNeuralFrame(
  file: NvcFile,
  options: { maxInputWidth?: number; cpuMaxInputWidth?: number; preferWebGPU?: boolean; webGpuTimeoutMs?: number; preferCodecInput?: boolean } = {},
): Promise<NeuralRenderResult> {
  return decodeNeuralFrame(file, 0, options);
}

export async function decodeNeuralFrame(
  file: NvcFile,
  frameIndex = 0,
  options: { maxInputWidth?: number; cpuMaxInputWidth?: number; preferWebGPU?: boolean; webGpuTimeoutMs?: number; preferCodecInput?: boolean } = {},
): Promise<NeuralRenderResult> {
  const model = requireTinySrModel(file.model);
  const base = decodeFirstBaseRgb(file, frameIndex, !(options.preferCodecInput ?? false));
  const maxInputWidth = options.maxInputWidth ?? 320;
  if (options.preferWebGPU) {
    const { input, inputWidth, inputHeight } = prepareTinySrInput(base, maxInputWidth);
    try {
      const output = await runTinySrWebGpu(model, input, inputWidth, inputHeight, options.webGpuTimeoutMs ?? 6000);
      return {
        imageData: applyReconstructionTuning(file, planarRgbToImageData(output.data, output.width, output.height), frameIndex, base),
        backend: "webgpu",
        inputWidth,
        inputHeight,
        outputWidth: output.width,
        outputHeight: output.height,
      };
    } catch (error) {
      const fallback = prepareTinySrInput(base, Math.min(maxInputWidth, options.cpuMaxInputWidth ?? 160));
      const output = runTinySrCpu(model, fallback.input, fallback.inputWidth, fallback.inputHeight);
      return {
        imageData: applyReconstructionTuning(file, planarRgbToImageData(output.data, output.width, output.height), frameIndex, base),
        backend: "cpu",
        inputWidth: fallback.inputWidth,
        inputHeight: fallback.inputHeight,
        outputWidth: output.width,
        outputHeight: output.height,
        fallbackReason: error instanceof Error ? error.message : String(error),
      };
    }
  }
  const { input, inputWidth, inputHeight } = prepareTinySrInput(base, Math.min(maxInputWidth, options.cpuMaxInputWidth ?? 160));
  const output = runTinySrCpu(model, input, inputWidth, inputHeight);
  return {
    imageData: applyReconstructionTuning(file, planarRgbToImageData(output.data, output.width, output.height), frameIndex, base),
    backend: "cpu",
    inputWidth,
    inputHeight,
    outputWidth: output.width,
    outputHeight: output.height,
  };
}

function applyReconstructionTuning(file: NvcFile, imageData: ImageData, frameIndex: number, base: BaseRgbFrame): ImageData {
  applyCodecBaseGuide(imageData, base);
  applyFeatureResidual(imageData, file.feature, frameIndex);
  applyColorCorrection(imageData, file.color);
  applyGrain(imageData, file.grain, frameIndex);
  return imageData;
}

function applyCodecBaseGuide(imageData: ImageData, base: BaseRgbFrame): void {
  const data = imageData.data;
  const strength = 0.72;
  for (let y = 0; y < imageData.height; y += 1) {
    const baseY = Math.min(base.height - 1, Math.floor((y * base.height) / imageData.height));
    for (let x = 0; x < imageData.width; x += 1) {
      const baseX = Math.min(base.width - 1, Math.floor((x * base.width) / imageData.width));
      const baseIndex = (baseY * base.width + baseX) * 3;
      const outIndex = (y * imageData.width + x) * 4;
      const baseLuma = 0.2126 * base.rgb[baseIndex] + 0.7152 * base.rgb[baseIndex + 1] + 0.0722 * base.rgb[baseIndex + 2];
      const outLuma = 0.2126 * data[outIndex] + 0.7152 * data[outIndex + 1] + 0.0722 * data[outIndex + 2];
      const delta = (baseLuma - outLuma) * strength;
      data[outIndex] = clampToByte(data[outIndex] + delta);
      data[outIndex + 1] = clampToByte(data[outIndex + 1] + delta);
      data[outIndex + 2] = clampToByte(data[outIndex + 2] + delta);
    }
  }
}

function applyFeatureResidual(imageData: ImageData, feature: FeatureInfo | undefined, frameIndex: number): void {
  if (
    !feature ||
    feature.format !== "FET1" ||
    !feature.residuals ||
    !feature.width ||
    !feature.height ||
    !feature.frameCount ||
    !feature.tileSize ||
    !feature.quantStep ||
    !feature.gridWidth ||
    !feature.gridHeight
  ) {
    return;
  }
  const tilesPerFrame = feature.gridWidth * feature.gridHeight;
  const targetFrame = Math.max(0, Math.min(feature.frameCount - 1, Math.trunc(frameIndex)));
  const frameOffset = targetFrame * tilesPerFrame;
  if (frameOffset + tilesPerFrame > feature.residuals.length) return;

  const data = imageData.data;
  for (let y = 0; y < imageData.height; y += 1) {
    const baseY = Math.min(feature.height - 1, Math.floor((y * feature.height) / imageData.height));
    const tileY = Math.min(feature.gridHeight - 1, Math.floor(baseY / feature.tileSize));
    for (let x = 0; x < imageData.width; x += 1) {
      const baseX = Math.min(feature.width - 1, Math.floor((x * feature.width) / imageData.width));
      const tileX = Math.min(feature.gridWidth - 1, Math.floor(baseX / feature.tileSize));
      const residual = feature.residuals[frameOffset + tileY * feature.gridWidth + tileX] - 128;
      if (residual === 0) continue;
      const delta = residual * feature.quantStep;
      const index = (y * imageData.width + x) * 4;
      data[index] = clampToByte(data[index] + delta);
      data[index + 1] = clampToByte(data[index + 1] + delta);
      data[index + 2] = clampToByte(data[index + 2] + delta);
    }
  }
}

function applyColorCorrection(imageData: ImageData, color?: ColorInfo): void {
  if (!color || color.format !== "COL1") return;
  const lumaScale = color.lumaScale ?? 1;
  const lumaBias = color.lumaBias ?? 0;
  const saturation = color.saturation ?? 1;
  const contrast = color.contrast ?? 1;
  const data = imageData.data;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const lumaDelta = luma * lumaScale + lumaBias - luma;
    data[i] = clampToByte(128 + (luma + (r - luma) * saturation + lumaDelta - 128) * contrast);
    data[i + 1] = clampToByte(128 + (luma + (g - luma) * saturation + lumaDelta - 128) * contrast);
    data[i + 2] = clampToByte(128 + (luma + (b - luma) * saturation + lumaDelta - 128) * contrast);
  }
}

function applyGrain(imageData: ImageData, grain?: GrainInfo, frameIndex = 0): void {
  if (!grain || grain.format !== "GRN1" || !grain.intensity || grain.intensity <= 0) return;
  const data = imageData.data;
  const amp = grain.intensity * 255;
  const seed = grain.seed ?? 0x4e564331;
  for (let y = 0; y < imageData.height; y += 1) {
    for (let x = 0; x < imageData.width; x += 1) {
      const index = (y * imageData.width + x) * 4;
      const value = hash32(seed ^ Math.imul(frameIndex + 1, 0x9e3779b1) ^ Math.imul(x + 17, 0x85ebca6b) ^ Math.imul(y + 31, 0xc2b2ae35));
      const delta = (((value & 1023) / 1023) * 2 - 1) * amp;
      data[index] = clampToByte(data[index] + delta);
      data[index + 1] = clampToByte(data[index + 1] + delta);
      data[index + 2] = clampToByte(data[index + 2] + delta);
    }
  }
}

function hash32(value: number): number {
  value ^= value >>> 16;
  value = Math.imul(value, 0x7feb352d);
  value ^= value >>> 15;
  value = Math.imul(value, 0x846ca68b);
  value ^= value >>> 16;
  return value >>> 0;
}

function prepareTinySrInput(base: BaseRgbFrame, maxInputWidth: number): { input: Float32Array; inputWidth: number; inputHeight: number } {
  const inputWidth = Math.max(16, Math.min(base.width, even(Math.floor(maxInputWidth))));
  const inputHeight = Math.max(16, even(Math.round((base.height * inputWidth) / base.width)));
  return {
    input: resizeRgbToPlanarFloat(base.rgb, base.width, base.height, inputWidth, inputHeight),
    inputWidth,
    inputHeight,
  };
}

function decodeFirstBaseRgb(file: NvcFile, frameIndex = 0, preferPreview = true): BaseRgbFrame {
  if (preferPreview) {
    const preview = decodePreviewRgb(file, frameIndex);
    if (preview) return preview;
  }
  const base = file.chunks.find((chunk) => chunk.id === "BASE");
  if (!base) throw new Error("BASE chunk missing");
  const payload = base.payload;
  const magic = readText(payload, 0, 4);
  if (magic === "BAS5") return decodeBas5FrameRgb(file, payload, frameIndex);
  const frame =
    magic === "BAS0"
      ? decodeBas0FirstYuv(payload)
      : magic === "BAS1"
        ? decodeBas1FirstYuv(payload)
        : magic === "BAS2"
          ? decodeBas2FirstYuv(payload)
          : magic === "BAS3"
            ? decodeBas3FirstYuv(payload)
            : magic === "BAS4"
              ? decodeBas4FirstYuv(payload)
        : unsupportedBase(magic);
  return yuv420ToRgb(frame);
}

export function getPreviewInfo(file: NvcFile): PreviewInfo | null {
  const preview = file.chunks.find((chunk) => chunk.id === "PRVW");
  if (!preview) return null;
  const payload = preview.payload;
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const magic = readText(payload, 0, 4);
  if (magic === "PVW0") {
    if (payload.length < 20) throw new Error("Invalid PVW0 payload");
    const width = view.getUint32(8, true);
    const height = view.getUint32(12, true);
    return { width, height, fpsNum: 1, fpsDen: 1, frameCount: 1 };
  }
  if (magic === "PVW1") {
    if (payload.length < 32) throw new Error("Invalid PVW1 payload");
    return {
      width: view.getUint32(8, true),
      height: view.getUint32(12, true),
      fpsNum: view.getUint32(16, true),
      fpsDen: view.getUint32(20, true),
      frameCount: view.getUint32(24, true),
    };
  }
  return null;
}

function decodePreviewRgb(file: NvcFile, frameIndex = 0): BaseRgbFrame | null {
  const preview = file.chunks.find((chunk) => chunk.id === "PRVW");
  if (!preview) return null;
  const payload = preview.payload;
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const magic = readText(payload, 0, 4);
  if (magic === "PVW0") {
    if (payload.length < 20) throw new Error("Invalid PVW0 payload");
    const width = view.getUint32(8, true);
    const height = view.getUint32(12, true);
    const rgbSize = view.getUint32(16, true);
    if (payload.length < 20 + rgbSize || rgbSize !== width * height * 3) {
      throw new Error("Invalid PVW0 payload");
    }
    return {
      width,
      height,
      rgb: new Uint8ClampedArray(payload.slice(20, 20 + rgbSize)),
    };
  }
  if (magic !== "PVW1") return null;
  if (payload.length < 32) throw new Error("Invalid PVW1 payload");
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const frameCount = view.getUint32(24, true);
  const frameBytes = view.getUint32(28, true);
  if (frameBytes !== width * height * 3) throw new Error("Invalid PVW1 frame size");
  const clampedIndex = Math.max(0, Math.min(frameCount - 1, Math.trunc(frameIndex)));
  const start = 32 + clampedIndex * frameBytes;
  if (payload.length < start + frameBytes) throw new Error("Truncated PVW1 frame data");
  return {
    width,
    height,
    rgb: new Uint8ClampedArray(payload.slice(start, start + frameBytes)),
  };
}

function rgbToImageData(rgb: Uint8ClampedArray, width: number, height: number): ImageData {
  const framePixels = width * height;
  const rgba = new Uint8ClampedArray(framePixels * 4);

  for (let i = 0; i < framePixels; i += 1) {
    const out = i * 4;
    const src = i * 3;
    rgba[out] = rgb[src];
    rgba[out + 1] = rgb[src + 1];
    rgba[out + 2] = rgb[src + 2];
    rgba[out + 3] = 255;
  }

  return new ImageData(rgba, width, height);
}

function decodeBas0FirstYuv(payload: Uint8Array): Yuv420Frame {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const rawSize = Number(view.getBigUint64(28, true));
  const rleSize = Number(view.getBigUint64(36, true));
  const rle = payload.slice(44, 44 + rleSize);
  const raw = decodeRle(rle, rawSize);
  return firstYuv420FromRaw(raw, width, height);
}

function decodeBas1FirstYuv(payload: Uint8Array): Yuv420Frame {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const codedSize = Number(view.getBigUint64(36, true));
  const blockSize = view.getUint32(44, true);
  const yQuant = view.getUint32(48, true);
  const uvQuant = view.getUint32(52, true);
  if (blockSize !== 4) throw new Error(`Unsupported BAS1 block size ${blockSize}`);
  const coded = payload.slice(56, 56 + codedSize);
  const reader = new CoeffReader(coded);
  return decodeTransformFirstYuv(width, height, yQuant, uvQuant, reader, decodeTransformPlaneWithReader);
}

function decodeBas2FirstYuv(payload: Uint8Array): Yuv420Frame {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const codedSize = Number(view.getBigUint64(36, true));
  const blockSize = view.getUint32(44, true);
  const yQuant = view.getUint32(48, true);
  const uvQuant = view.getUint32(52, true);
  const predictor = view.getUint32(56, true);
  if (blockSize !== 4) throw new Error(`Unsupported BAS2 block size ${blockSize}`);
  if (predictor !== 1) throw new Error(`Unsupported BAS2 predictor ${predictor}`);
  const coded = payload.slice(60, 60 + codedSize);
  const reader = new CoeffReader(coded);
  return decodeTransformFirstYuv(width, height, yQuant, uvQuant, reader, decodePredictiveTransformPlaneWithReader);
}

function decodeBas3FirstYuv(payload: Uint8Array): Yuv420Frame {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const frameCount = view.getUint32(24, true);
  const codedSize = Number(view.getBigUint64(36, true));
  const blockSize = view.getUint32(44, true);
  const yQuant = view.getUint32(48, true);
  const uvQuant = view.getUint32(52, true);
  const predictor = view.getUint32(56, true);
  const motionModes = view.getUint32(60, true);
  if (blockSize !== 4) throw new Error(`Unsupported BAS3 block size ${blockSize}`);
  if (predictor !== 2) throw new Error(`Unsupported BAS3 predictor ${predictor}`);
  if (motionModes !== 6) throw new Error(`Unsupported BAS3 motion mode count ${motionModes}`);
  const coded = payload.slice(64, 64 + codedSize);
  return decodeMotionPackedFirstYuv(coded, width, height, frameCount, yQuant, uvQuant, "BAS3");
}

function decodeBas4FirstYuv(payload: Uint8Array): Yuv420Frame {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const frameCount = view.getUint32(24, true);
  const codedSize = Number(view.getBigUint64(36, true));
  const blockSize = view.getUint32(44, true);
  const yQuant = view.getUint32(48, true);
  const uvQuant = view.getUint32(52, true);
  const predictor = view.getUint32(56, true);
  const motionModes = view.getUint32(60, true);
  const entropy = view.getUint32(64, true);
  if (blockSize !== 4) throw new Error(`Unsupported BAS4 block size ${blockSize}`);
  if (predictor !== 3) throw new Error(`Unsupported BAS4 predictor ${predictor}`);
  if (motionModes !== 6) throw new Error(`Unsupported BAS4 motion mode count ${motionModes}`);
  if (entropy !== 1) throw new Error(`Unsupported BAS4 entropy ${entropy}`);
  const coded = payload.slice(68, 68 + codedSize);
  const packed = entropyDecode(coded);
  return decodeMotionPackedFirstYuv(packed, width, height, frameCount, yQuant, uvQuant, "BAS4");
}

function decodeBas5FrameRgb(file: NvcFile, payload: Uint8Array, frameIndex: number): BaseRgbFrame {
  const packet = findBas5Packet(payload, frameIndex);
  if (file.baseFrameCache?.packetIndex === packet.packetIndex) {
    return file.baseFrameCache.frames[packet.targetFrame - file.baseFrameCache.startFrame];
  }
  const packetCaches = file.baseFrameCaches ?? new Map<number, BaseFrameCache>();
  file.baseFrameCaches = packetCaches;
  const cached = packetCaches.get(packet.packetIndex);
  if (cached) {
    file.baseFrameCache = cached;
    rememberBaseFrameCacheUse(file, packet.packetIndex);
    return cached.frames[packet.targetFrame - cached.startFrame];
  }
  const frames = decodeBas5PacketRgbFrames(payload, packet);
  file.baseFrameCache = { packetIndex: packet.packetIndex, startFrame: packet.startFrame, frames };
  packetCaches.set(packet.packetIndex, file.baseFrameCache);
  rememberBaseFrameCacheUse(file, packet.packetIndex);
  evictOldBaseFrameCaches(file, 2);
  return frames[packet.targetFrame - packet.startFrame];
}

function rememberBaseFrameCacheUse(file: NvcFile, packetIndex: number): void {
  const order = file.baseFrameCacheOrder ?? [];
  file.baseFrameCacheOrder = order;
  const existing = order.indexOf(packetIndex);
  if (existing >= 0) order.splice(existing, 1);
  order.push(packetIndex);
}

function evictOldBaseFrameCaches(file: NvcFile, maxPackets: number): void {
  const caches = file.baseFrameCaches;
  const order = file.baseFrameCacheOrder;
  if (!caches || !order) return;
  while (order.length > maxPackets) {
    const oldest = order.shift();
    if (oldest === undefined || oldest === file.baseFrameCache?.packetIndex) continue;
    caches.delete(oldest);
  }
}

type Bas5PacketLookup = {
  width: number;
  height: number;
  yQuant: number;
  uvQuant: number;
  packetIndex: number;
  packetEntry: number;
  startFrame: number;
  frameCount: number;
  targetFrame: number;
  packetOffset: number;
  packetSize: number;
};

function findBas5Packet(payload: Uint8Array, frameIndex: number): Bas5PacketLookup {
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const width = view.getUint32(8, true);
  const height = view.getUint32(12, true);
  const totalFrames = view.getUint32(24, true);
  const blockSize = view.getUint32(44, true);
  const yQuant = view.getUint32(48, true);
  const uvQuant = view.getUint32(52, true);
  const predictor = view.getUint32(56, true);
  const motionModes = view.getUint32(60, true);
  const entropy = view.getUint32(64, true);
  const packetCount = view.getUint32(72, true);
  if (payload.length < 76 || packetCount < 1) throw new Error("Invalid BAS5 packet table");
  if (blockSize !== 4) throw new Error(`Unsupported BAS5 block size ${blockSize}`);
  if (predictor !== 4) throw new Error(`Unsupported BAS5 predictor ${predictor}`);
  if (motionModes !== 6) throw new Error(`Unsupported BAS5 motion mode count ${motionModes}`);
  if (entropy !== 1) throw new Error(`Unsupported BAS5 entropy ${entropy}`);
  const targetFrame = Math.max(0, Math.min(totalFrames - 1, Math.trunc(frameIndex)));
  let packetEntry = -1;
  let packetIndex = -1;
  let packetStartFrame = 0;
  let packetFrameCount = 0;
  for (let i = 0; i < packetCount; i += 1) {
    const entry = 76 + i * 24;
    const startFrame = view.getUint32(entry, true);
    const frameCount = view.getUint32(entry + 4, true);
    if (targetFrame >= startFrame && targetFrame < startFrame + frameCount) {
      packetEntry = entry;
      packetIndex = i;
      packetStartFrame = startFrame;
      packetFrameCount = frameCount;
      break;
    }
  }
  if (packetEntry < 0 || packetFrameCount < 1) throw new Error(`No BAS5 packet for frame ${targetFrame}`);
  const packetOffset = Number(view.getBigUint64(packetEntry + 8, true));
  const packetSize = Number(view.getBigUint64(packetEntry + 16, true));
  if (payload.length < packetOffset + packetSize) throw new Error("Truncated BAS5 packet");
  return {
    width,
    height,
    yQuant,
    uvQuant,
    packetIndex,
    packetEntry,
    startFrame: packetStartFrame,
    frameCount: packetFrameCount,
    targetFrame,
    packetOffset,
    packetSize,
  };
}

function decodeBas5PacketRgbFrames(payload: Uint8Array, packet: Bas5PacketLookup): BaseRgbFrame[] {
  const packed = entropyDecode(payload.slice(packet.packetOffset, packet.packetOffset + packet.packetSize));
  return decodeMotionPackedFramesYuv(packed, packet.width, packet.height, packet.frameCount, packet.yQuant, packet.uvQuant, "BAS5").map(yuv420ToRgb);
}

function decodeTransformFirstYuv(
  width: number,
  height: number,
  yQuant: number,
  uvQuant: number,
  reader: CoeffReader,
  decoder: (reader: CoeffReader, width: number, height: number, quant: number) => Uint8Array,
): Yuv420Frame {
  const uvWidth = Math.trunc(width / 2);
  const uvHeight = Math.trunc(height / 2);
  return {
    width,
    height,
    y: decoder(reader, width, height, yQuant),
    u: decoder(reader, uvWidth, uvHeight, uvQuant),
    v: decoder(reader, uvWidth, uvHeight, uvQuant),
  };
}

function decodeMotionPackedFirstYuv(
  coded: Uint8Array,
  width: number,
  height: number,
  frameCount: number,
  yQuant: number,
  uvQuant: number,
  label: string,
): Yuv420Frame {
  return decodeMotionPackedFrameYuv(coded, width, height, frameCount, 0, yQuant, uvQuant, label);
}

function decodeMotionPackedFrameYuv(
  coded: Uint8Array,
  width: number,
  height: number,
  frameCount: number,
  frameOffset: number,
  yQuant: number,
  uvQuant: number,
  label: string,
): Yuv420Frame {
  return decodeMotionPackedFramesYuv(coded, width, height, frameCount, yQuant, uvQuant, label)[Math.max(0, Math.min(frameCount - 1, Math.trunc(frameOffset)))];
}

function decodeMotionPackedFramesYuv(
  coded: Uint8Array,
  width: number,
  height: number,
  frameCount: number,
  yQuant: number,
  uvQuant: number,
  label: string,
): Yuv420Frame[] {
  if (coded.length < 8) throw new Error("Invalid BAS3 motion stream");
  const codedView = new DataView(coded.buffer, coded.byteOffset, coded.byteLength);
  const modeLength = Number(codedView.getBigUint64(0, true));
  const modeStream = decodeRle(coded.slice(8, 8 + modeLength), expectedMotionModes(width, height, frameCount));
  const coeffStream = coded.slice(8 + modeLength);
  if (label.length === 0) throw new Error("Invalid motion label");
  const modeReader = new ModeReader(modeStream);
  const reader = new CoeffReader(coeffStream);
  const uvWidth = Math.trunc(width / 2);
  const uvHeight = Math.trunc(height / 2);
  const frames: Yuv420Frame[] = [];
  let prevY: Uint8Array | null = null;
  let prevU: Uint8Array | null = null;
  let prevV: Uint8Array | null = null;
  for (let frame = 0; frame < frameCount; frame += 1) {
    const y = decodeMotionTransformPlaneWithReaders(modeReader, reader, width, height, yQuant, prevY);
    const u = decodeMotionTransformPlaneWithReaders(modeReader, reader, uvWidth, uvHeight, uvQuant, prevU);
    const v = decodeMotionTransformPlaneWithReaders(modeReader, reader, uvWidth, uvHeight, uvQuant, prevV);
    frames.push({ width, height, y, u, v });
    prevY = y;
    prevU = u;
    prevV = v;
  }
  return frames;
}

function unsupportedBase(magic: string): never {
  throw new Error(`Unsupported BASE payload ${magic}`);
}

function decodeRle(rle: Uint8Array, expected: number): Uint8Array {
  const out = new Uint8Array(expected);
  let outIndex = 0;
  for (let i = 0; i + 1 < rle.length && outIndex < expected; i += 2) {
    const count = rle[i];
    const value = rle[i + 1];
    out.fill(value, outIndex, Math.min(outIndex + count, expected));
    outIndex += count;
  }
  if (outIndex !== expected) throw new Error("Invalid RLE stream");
  return out;
}

function entropyDecode(payload: Uint8Array): Uint8Array {
  if (payload.length < 4 + 8 + 256 || readText(payload, 0, 4) !== "HUF0") {
    throw new Error("Invalid BAS4 entropy stream");
  }

  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  const rawLength = Number(view.getBigUint64(4, true));
  const lengths = payload.slice(12, 12 + 256);
  const codes = canonicalCodes(lengths);
  const tree = buildHuffmanTree(lengths, codes);
  const reader = new BitReader(payload.slice(12 + 256));
  const out = new Uint8Array(rawLength);

  for (let i = 0; i < out.length; i += 1) {
    out[i] = tree.nextSymbol(reader);
  }

  return out;
}

function canonicalCodes(lengths: Uint8Array): bigint[] {
  const codes = Array<bigint>(256).fill(0n);
  const assigned = Array<boolean>(256).fill(false);
  let code = 0n;
  let previousLength = 0;

  while (true) {
    let best = -1;
    for (let symbol = 0; symbol < 256; symbol += 1) {
      const length = lengths[symbol];
      if (length === 0 || assigned[symbol]) continue;
      if (best < 0 || length < lengths[best] || (length === lengths[best] && symbol < best)) {
        best = symbol;
      }
    }
    if (best < 0) break;
    const length = lengths[best];
    code <<= BigInt(length - previousLength);
    codes[best] = code;
    code += 1n;
    previousLength = length;
    assigned[best] = true;
  }

  return codes;
}

class BitReader {
  private offset = 0;
  private used = 0;

  constructor(private readonly data: Uint8Array) {}

  next(): number {
    if (this.offset >= this.data.length) throw new Error("Truncated entropy bits");
    const bit = (this.data[this.offset] >> (7 - this.used)) & 1;
    this.used += 1;
    if (this.used === 8) {
      this.used = 0;
      this.offset += 1;
    }
    return bit;
  }
}

type DecodeNode = {
  left: number;
  right: number;
  symbol: number;
};

class HuffmanTree {
  constructor(private readonly nodes: DecodeNode[]) {}

  nextSymbol(reader: BitReader): number {
    let node = 0;
    while (this.nodes[node].symbol < 0) {
      const bit = reader.next();
      node = bit === 0 ? this.nodes[node].left : this.nodes[node].right;
      if (node < 0) throw new Error("Invalid entropy bits");
    }
    return this.nodes[node].symbol;
  }
}

function buildHuffmanTree(lengths: Uint8Array, codes: bigint[]): HuffmanTree {
  const nodes: DecodeNode[] = [{ left: -1, right: -1, symbol: -1 }];
  for (let symbol = 0; symbol < 256; symbol += 1) {
    const length = lengths[symbol];
    if (length === 0) continue;
    let node = 0;
    for (let remaining = length - 1; remaining >= 0; remaining -= 1) {
      const bit = Number((codes[symbol] >> BigInt(remaining)) & 1n);
      const nextKey = bit === 0 ? "left" : "right";
      if (nodes[node][nextKey] < 0) {
        nodes[node][nextKey] = nodes.length;
        nodes.push({ left: -1, right: -1, symbol: -1 });
      }
      node = nodes[node][nextKey];
    }
    nodes[node].symbol = symbol;
  }
  return new HuffmanTree(nodes);
}

const zigzag4x4 = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

function decodeTransformPlaneWithReader(reader: CoeffReader, width: number, height: number, quant: number): Uint8Array {
  const out = new Uint8Array(width * height);

  for (let by = 0; by < height; by += 4) {
    for (let bx = 0; bx < width; bx += 4) {
      const block = new Int32Array(16);
      for (const index of zigzag4x4) {
        block[index] = reader.nextCoeff() * quant;
      }
      inverseHadamard4x4(block);
      storeBlock(out, width, height, bx, by, block);
    }
  }

  return out;
}

function decodePredictiveTransformPlaneWithReader(reader: CoeffReader, width: number, height: number, quant: number): Uint8Array {
  const out = new Uint8Array(width * height);
  out.fill(128);

  for (let by = 0; by < height; by += 4) {
    for (let bx = 0; bx < width; bx += 4) {
      const residual = new Int32Array(16);
      for (const index of zigzag4x4) {
        residual[index] = reader.nextCoeff() * quant;
      }
      inverseHadamard4x4(residual);
      storePredictedBlock(out, width, height, bx, by, residual);
    }
  }

  return out;
}

function decodeMotionTransformPlaneWithReaders(
  modeReader: ModeReader,
  reader: CoeffReader,
  width: number,
  height: number,
  quant: number,
  previous: Uint8Array | null,
): Uint8Array {
  const out = new Uint8Array(width * height);
  out.fill(128);

  for (let by = 0; by < height; by += 4) {
    for (let bx = 0; bx < width; bx += 4) {
      const mode = modeReader.next();
      if (mode >= 6) throw new Error(`Invalid BAS3 motion mode ${mode}`);
      if (mode !== 0 && previous === null) throw new Error("BAS3 first frame cannot use temporal prediction");
      const residual = new Int32Array(16);
      for (const index of zigzag4x4) {
        residual[index] = reader.nextCoeff() * quant;
      }
      inverseHadamard4x4(residual);
      storeMotionBlock(out, previous, width, height, bx, by, mode, residual);
    }
  }

  return out;
}

class ModeReader {
  private offset = 0;

  constructor(private readonly data: Uint8Array) {}

  next(): number {
    if (this.offset >= this.data.length) throw new Error("Truncated BAS3 mode stream");
    return this.data[this.offset++];
  }
}

class CoeffReader {
  private offset = 0;
  private zeroRun = 0;

  constructor(private readonly data: Uint8Array) {}

  alignToToken(): void {
    if (this.zeroRun !== 0) throw new Error("Invalid coefficient stream alignment");
  }

  nextUint(): number {
    this.alignToToken();
    return this.readVarint();
  }

  nextCoeff(): number {
    if (this.zeroRun > 0) {
      this.zeroRun -= 1;
      return 0;
    }

    const token = this.readVarint();
    if (token === 0) {
      const count = this.readVarint();
      if (count <= 0) throw new Error("Invalid BAS1 zero run");
      this.zeroRun = count - 1;
      return 0;
    }
    return unZigZag(token - 1);
  }

  private readVarint(): number {
    let result = 0;
    let shift = 0;
    while (this.offset < this.data.length) {
      const byte = this.data[this.offset++];
      result += (byte & 0x7f) * 2 ** shift;
      if ((byte & 0x80) === 0) return result;
      shift += 7;
      if (shift > 53) throw new Error("BAS1 varint is too large for JS preview");
    }
    throw new Error("Truncated BAS1 varint");
  }
}

function unZigZag(value: number): number {
  return value % 2 === 0 ? value / 2 : -((value + 1) / 2);
}

function inverseHadamard4x4(block: Int32Array): void {
  hadamard4x4(block);
  for (let i = 0; i < 16; i += 1) {
    block[i] = divRound(block[i], 16);
  }
}

function hadamard4x4(block: Int32Array): void {
  for (let y = 0; y < 4; y += 1) {
    const i = y * 4;
    const a0 = block[i] + block[i + 1];
    const a1 = block[i] - block[i + 1];
    const a2 = block[i + 2] + block[i + 3];
    const a3 = block[i + 2] - block[i + 3];
    block[i] = a0 + a2;
    block[i + 1] = a1 + a3;
    block[i + 2] = a0 - a2;
    block[i + 3] = a1 - a3;
  }

  for (let x = 0; x < 4; x += 1) {
    const a0 = block[x] + block[x + 4];
    const a1 = block[x] - block[x + 4];
    const a2 = block[x + 8] + block[x + 12];
    const a3 = block[x + 8] - block[x + 12];
    block[x] = a0 + a2;
    block[x + 4] = a1 + a3;
    block[x + 8] = a0 - a2;
    block[x + 12] = a1 - a3;
  }
}

function storeBlock(out: Uint8Array, width: number, height: number, bx: number, by: number, block: Int32Array): void {
  for (let y = 0; y < 4 && by + y < height; y += 1) {
    for (let x = 0; x < 4 && bx + x < width; x += 1) {
      const index = (by + y) * width + bx + x;
      out[index] = clampToByte(block[y * 4 + x] + 128);
    }
  }
}

function storePredictedBlock(out: Uint8Array, width: number, height: number, bx: number, by: number, residual: Int32Array): void {
  for (let y = 0; y < 4 && by + y < height; y += 1) {
    for (let x = 0; x < 4 && bx + x < width; x += 1) {
      const index = (by + y) * width + bx + x;
      const pred = predictBlockSample(out, width, height, bx, by, x, y);
      out[index] = clampToByte(pred + residual[y * 4 + x]);
    }
  }
}

function storeMotionBlock(
  out: Uint8Array,
  previous: Uint8Array | null,
  width: number,
  height: number,
  bx: number,
  by: number,
  mode: number,
  residual: Int32Array,
): void {
  for (let y = 0; y < 4 && by + y < height; y += 1) {
    for (let x = 0; x < 4 && bx + x < width; x += 1) {
      const index = (by + y) * width + bx + x;
      const pred = motionPredictSample(out, previous, width, height, bx, by, x, y, mode);
      out[index] = clampToByte(pred + residual[y * 4 + x]);
    }
  }
}

function motionPredictSample(
  recon: Uint8Array,
  previous: Uint8Array | null,
  width: number,
  height: number,
  bx: number,
  by: number,
  x: number,
  y: number,
  mode: number,
): number {
  if (mode === 0 || previous === null) return predictBlockSample(recon, width, height, bx, by, x, y);
  const mv = motionVector(mode);
  const srcX = clamp(Math.trunc(bx + x + mv.dx), 0, width - 1);
  const srcY = clamp(Math.trunc(by + y + mv.dy), 0, height - 1);
  return previous[srcY * width + srcX];
}

function motionVector(mode: number): { dx: number; dy: number } {
  switch (mode) {
    case 1:
      return { dx: 0, dy: 0 };
    case 2:
      return { dx: -4, dy: 0 };
    case 3:
      return { dx: 4, dy: 0 };
    case 4:
      return { dx: 0, dy: -4 };
    case 5:
      return { dx: 0, dy: 4 };
    default:
      return { dx: 0, dy: 0 };
  }
}

function predictBlockSample(recon: Uint8Array, width: number, height: number, bx: number, by: number, x: number, y: number): number {
  const xx = Math.min(bx + x, width - 1);
  const yy = Math.min(by + y, height - 1);
  const hasLeft = bx > 0;
  const hasTop = by > 0;
  if (hasLeft && hasTop) {
    const left = recon[yy * width + bx - 1];
    const top = recon[(by - 1) * width + xx];
    return Math.trunc((left + top + 1) / 2);
  }
  if (hasLeft) return recon[yy * width + bx - 1];
  if (hasTop) return recon[(by - 1) * width + xx];
  return 128;
}

function divRound(value: number, denominator: number): number {
  if (value >= 0) return Math.trunc((value + Math.trunc(denominator / 2)) / denominator);
  return -Math.trunc((-value + Math.trunc(denominator / 2)) / denominator);
}

function clampToByte(value: number): number {
  if (value < 0) return 0;
  if (value > 255) return 255;
  return value;
}

function clamp(value: number, low: number, high: number): number {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

function firstYuv420FromRaw(raw: Uint8Array, width: number, height: number): Yuv420Frame {
  const ySize = width * height;
  const uvWidth = Math.trunc(width / 2);
  const uvHeight = Math.trunc(height / 2);
  const uvSize = uvWidth * uvHeight;
  if (raw.length < ySize + uvSize * 2) throw new Error("Invalid YUV420 frame");
  return {
    width,
    height,
    y: raw.slice(0, ySize),
    u: raw.slice(ySize, ySize + uvSize),
    v: raw.slice(ySize + uvSize, ySize + uvSize * 2),
  };
}

function yuv420ToRgb(frame: Yuv420Frame): BaseRgbFrame {
  const rgb = new Uint8ClampedArray(frame.width * frame.height * 3);
  const uvWidth = Math.trunc(frame.width / 2);

  for (let y = 0; y < frame.height; y += 1) {
    for (let x = 0; x < frame.width; x += 1) {
      const yValue = frame.y[y * frame.width + x];
      const uvIndex = Math.trunc(y / 2) * uvWidth + Math.trunc(x / 2);
      const u = frame.u[uvIndex] - 128;
      const v = frame.v[uvIndex] - 128;
      const out = (y * frame.width + x) * 3;
      rgb[out] = clampToByte(Math.round(yValue + 1.402 * v));
      rgb[out + 1] = clampToByte(Math.round(yValue - 0.344136 * u - 0.714136 * v));
      rgb[out + 2] = clampToByte(Math.round(yValue + 1.772 * u));
    }
  }

  return { width: frame.width, height: frame.height, rgb };
}

function requireTinySrModel(model: ModlInfo | undefined): { metadata: Mod0Metadata; weights: Float32Array } {
  if (!model || model.format !== "MOD0" || !model.metadata || !model.weights) {
    throw new Error("A MOD0 TinySR model is required for neural reconstruction");
  }
  if (model.metadata.architecture !== "tiny_cnn_pixel_shuffle_x2") {
    throw new Error(`Unsupported MOD0 architecture ${model.metadata.architecture ?? "unknown"}`);
  }
  return { metadata: model.metadata, weights: model.weights };
}

function resizeRgbToPlanarFloat(
  rgb: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Float32Array {
  const out = new Float32Array(targetWidth * targetHeight * 3);
  const xScale = sourceWidth / targetWidth;
  const yScale = sourceHeight / targetHeight;

  for (let y = 0; y < targetHeight; y += 1) {
    const srcY = Math.min(sourceHeight - 1, Math.floor((y + 0.5) * yScale));
    for (let x = 0; x < targetWidth; x += 1) {
      const srcX = Math.min(sourceWidth - 1, Math.floor((x + 0.5) * xScale));
      const src = (srcY * sourceWidth + srcX) * 3;
      const pixel = y * targetWidth + x;
      out[pixel] = rgb[src] / 255;
      out[targetWidth * targetHeight + pixel] = rgb[src + 1] / 255;
      out[targetWidth * targetHeight * 2 + pixel] = rgb[src + 2] / 255;
    }
  }

  return out;
}

function runTinySrCpu(
  model: { metadata: Mod0Metadata; weights: Float32Array },
  input: Float32Array,
  width: number,
  height: number,
): { width: number; height: number; data: Float32Array } {
  const conv0 = requireLayer(model.metadata, "conv0");
  const conv1 = requireLayer(model.metadata, "conv1");
  const conv2 = requireLayer(model.metadata, "conv2");
  const hidden0 = conv2d(input, width, height, 3, 16, conv0, model.weights, true);
  const hidden1 = conv2d(hidden0, width, height, 16, 16, conv1, model.weights, true);
  const shuffled = conv2d(hidden1, width, height, 16, 12, conv2, model.weights, false);
  return pixelShuffle2x(shuffled, width, height, 3);
}

async function runTinySrWebGpu(
  model: { metadata: Mod0Metadata; weights: Float32Array },
  input: Float32Array,
  width: number,
  height: number,
  timeoutMs: number,
): Promise<{ width: number; height: number; data: Float32Array }> {
  const conv0 = requireLayer(model.metadata, "conv0");
  const conv1 = requireLayer(model.metadata, "conv1");
  const conv2 = requireLayer(model.metadata, "conv2");
  return runTinySrWebGpuWorker(input, width, height, model.weights, [conv0, conv1, conv2], timeoutMs);
}

function createStorageBuffer(device: GPUDevice, data: Float32Array, extraUsage = 0): GPUBuffer {
  const buffer = device.createBuffer({
    size: alignedBufferSize(data.byteLength),
    usage: GPUBufferUsage.STORAGE | extraUsage,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function runTinySrWebGpuWorker(
  input: Float32Array,
  width: number,
  height: number,
  weights: Float32Array,
  layers: Mod0Layer[],
  timeoutMs: number,
): Promise<{ width: number; height: number; data: Float32Array }> {
  return new Promise((resolve, reject) => {
    const workerUrl = URL.createObjectURL(new Blob([webGpuWorkerSource()], { type: "text/javascript" }));
    const worker = new Worker(workerUrl, { type: "module" });
    const timeout = window.setTimeout(() => {
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      reject(new Error(`WebGPU TinySR worker timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    worker.onmessage = (event: MessageEvent) => {
      window.clearTimeout(timeout);
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      const message = event.data as { ok: boolean; error?: string; width?: number; height?: number; data?: ArrayBuffer };
      if (!message.ok || !message.data || !message.width || !message.height) {
        reject(new Error(message.error ?? "WebGPU TinySR worker failed"));
        return;
      }
      resolve({
        width: message.width,
        height: message.height,
        data: new Float32Array(message.data),
      });
    };

    worker.onerror = (event) => {
      window.clearTimeout(timeout);
      worker.terminate();
      URL.revokeObjectURL(workerUrl);
      reject(new Error(event.message));
    };

    const inputCopy = input.slice();
    const weightsCopy = weights.slice();
    worker.postMessage(
      {
        input: inputCopy.buffer,
        weights: weightsCopy.buffer,
        width,
        height,
        layers,
        conv2dShader,
        pixelShuffleShader,
      },
      [inputCopy.buffer, weightsCopy.buffer],
    );
  });
}

async function runConv2dWebGpu(
  device: GPUDevice,
  input: GPUBuffer,
  weights: GPUBuffer,
  width: number,
  height: number,
  inChannels: number,
  outChannels: number,
  layer: Mod0Layer,
  relu: boolean,
): Promise<GPUBuffer> {
  const [shapeOut, shapeIn, kernelH, kernelW] = layer.weight_shape;
  if (shapeOut !== outChannels || shapeIn !== inChannels || kernelH !== 3 || kernelW !== 3) {
    throw new Error(`Unsupported MOD0 layer shape for ${layer.name}`);
  }

  const outputByteLength = width * height * outChannels * 4;
  const output = device.createBuffer({
    size: alignedBufferSize(outputByteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const params = new Uint32Array([
    width,
    height,
    inChannels,
    outChannels,
    Math.trunc(layer.weight_offset / 4),
    Math.trunc(layer.bias_offset / 4),
    relu ? 1 : 0,
    0,
  ]);
  const paramsBuffer = device.createBuffer({
    size: alignedBufferSize(params.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: conv2dShader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weights } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((width * height * outChannels) / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  paramsBuffer.destroy();
  return output;
}

async function runPixelShuffleWebGpu(
  device: GPUDevice,
  input: GPUBuffer,
  width: number,
  height: number,
  channels: number,
): Promise<{ width: number; height: number; data: Float32Array }> {
  const outWidth = width * 2;
  const outHeight = height * 2;
  const outputByteLength = outWidth * outHeight * channels * 4;
  const output = device.createBuffer({
    size: alignedBufferSize(outputByteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: alignedBufferSize(outputByteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const params = new Uint32Array([width, height, channels, 0]);
  const paramsBuffer = device.createBuffer({
    size: alignedBufferSize(params.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);

  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: pixelShuffleShader }),
      entryPoint: "main",
    },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((outWidth * outHeight * channels) / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  output.destroy();
  readback.destroy();
  paramsBuffer.destroy();
  return { width: outWidth, height: outHeight, data };
}

function requireLayer(metadata: Mod0Metadata, name: string): Mod0Layer {
  const layer = metadata.layers?.find((item) => item.name === name);
  if (!layer) throw new Error(`MOD0 layer ${name} missing`);
  return layer;
}

function conv2d(
  input: Float32Array,
  width: number,
  height: number,
  inChannels: number,
  outChannels: number,
  layer: Mod0Layer,
  weights: Float32Array,
  relu: boolean,
): Float32Array {
  const [shapeOut, shapeIn, kernelH, kernelW] = layer.weight_shape;
  if (shapeOut !== outChannels || shapeIn !== inChannels || kernelH !== 3 || kernelW !== 3) {
    throw new Error(`Unsupported MOD0 layer shape for ${layer.name}`);
  }

  const out = new Float32Array(width * height * outChannels);
  const planeSize = width * height;
  const weightBase = Math.trunc(layer.weight_offset / 4);
  const biasBase = Math.trunc(layer.bias_offset / 4);

  for (let oc = 0; oc < outChannels; oc += 1) {
    const outPlane = oc * planeSize;
    const bias = weights[biasBase + oc] ?? 0;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let acc = bias;
        for (let ic = 0; ic < inChannels; ic += 1) {
          const inPlane = ic * planeSize;
          for (let ky = 0; ky < 3; ky += 1) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= height) continue;
            for (let kx = 0; kx < 3; kx += 1) {
              const ix = x + kx - 1;
              if (ix < 0 || ix >= width) continue;
              const weightIndex = weightBase + (((oc * inChannels + ic) * 3 + ky) * 3 + kx);
              acc += input[inPlane + iy * width + ix] * weights[weightIndex];
            }
          }
        }
        out[outPlane + y * width + x] = relu && acc < 0 ? 0 : acc;
      }
    }
  }

  return out;
}

function pixelShuffle2x(input: Float32Array, width: number, height: number, channels: number): { width: number; height: number; data: Float32Array } {
  const outWidth = width * 2;
  const outHeight = height * 2;
  const inPlane = width * height;
  const outPlane = outWidth * outHeight;
  const out = new Float32Array(outPlane * channels);

  for (let channel = 0; channel < channels; channel += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const subpixel = dy * 2 + dx;
            const sourceChannel = channel * 4 + subpixel;
            const value = input[sourceChannel * inPlane + y * width + x];
            const outY = y * 2 + dy;
            const outX = x * 2 + dx;
            out[channel * outPlane + outY * outWidth + outX] = clamp(value, 0, 1);
          }
        }
      }
    }
  }

  return { width: outWidth, height: outHeight, data: out };
}

function planarRgbToImageData(data: Float32Array, width: number, height: number): ImageData {
  const pixels = width * height;
  const rgba = new Uint8ClampedArray(pixels * 4);
  for (let i = 0; i < pixels; i += 1) {
    const out = i * 4;
    rgba[out] = clampToByte(Math.round(data[i] * 255));
    rgba[out + 1] = clampToByte(Math.round(data[pixels + i] * 255));
    rgba[out + 2] = clampToByte(Math.round(data[pixels * 2 + i] * 255));
    rgba[out + 3] = 255;
  }
  return new ImageData(rgba, width, height);
}

function even(value: number): number {
  return value % 2 === 0 ? value : value - 1;
}

function alignedBufferSize(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const id = window.setTimeout(() => reject(new Error(`WebGPU TinySR timed out after ${timeoutMs}ms`)), timeoutMs);
    promise.then(
      (value) => {
        window.clearTimeout(id);
        resolve(value);
      },
      (error) => {
        window.clearTimeout(id);
        reject(error);
      },
    );
  });
}

function webGpuWorkerSource(): string {
  return `
self.onmessage = async (event) => {
  try {
    const { input, weights, width, height, layers, conv2dShader, pixelShuffleShader } = event.data;
    const result = await runTinySr(new Float32Array(input), width, height, new Float32Array(weights), layers, conv2dShader, pixelShuffleShader);
    self.postMessage({ ok: true, width: result.width, height: result.height, data: result.data.buffer }, [result.data.buffer]);
  } catch (error) {
    self.postMessage({ ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};

async function runTinySr(input, width, height, weights, layers, conv2dShader, pixelShuffleShader) {
  const gpu = self.navigator && self.navigator.gpu;
  if (!gpu) throw new Error("WebGPU is unavailable in worker");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU adapter is unavailable");
  const device = await adapter.requestDevice();
  const conv0 = findLayer(layers, "conv0");
  const conv1 = findLayer(layers, "conv1");
  const conv2 = findLayer(layers, "conv2");
  const weightBuffer = createStorageBuffer(device, weights, GPUBufferUsage.COPY_DST);
  const inputBuffer = createStorageBuffer(device, input, GPUBufferUsage.COPY_DST);

  try {
    const hidden0 = await runConv2d(device, inputBuffer, weightBuffer, width, height, 3, 16, conv0, true, conv2dShader);
    try {
      const hidden1 = await runConv2d(device, hidden0, weightBuffer, width, height, 16, 16, conv1, true, conv2dShader);
      try {
        const shuffledInput = await runConv2d(device, hidden1, weightBuffer, width, height, 16, 12, conv2, false, conv2dShader);
        try {
          return await runPixelShuffle(device, shuffledInput, width, height, 3, pixelShuffleShader);
        } finally {
          shuffledInput.destroy();
        }
      } finally {
        hidden1.destroy();
      }
    } finally {
      hidden0.destroy();
    }
  } finally {
    inputBuffer.destroy();
    weightBuffer.destroy();
    device.destroy();
  }
}

function findLayer(layers, name) {
  const layer = layers.find((item) => item.name === name);
  if (!layer) throw new Error("MOD0 layer " + name + " missing");
  return layer;
}

function createStorageBuffer(device, data, extraUsage) {
  const buffer = device.createBuffer({
    size: alignedBufferSize(data.byteLength),
    usage: GPUBufferUsage.STORAGE | extraUsage,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

async function runConv2d(device, input, weights, width, height, inChannels, outChannels, layer, relu, shader) {
  const outputByteLength = width * height * outChannels * 4;
  const output = device.createBuffer({
    size: alignedBufferSize(outputByteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const params = new Uint32Array([
    width,
    height,
    inChannels,
    outChannels,
    Math.trunc(layer.weight_offset / 4),
    Math.trunc(layer.bias_offset / 4),
    relu ? 1 : 0,
    0,
  ]);
  const paramsBuffer = device.createBuffer({
    size: alignedBufferSize(params.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: weights } },
      { binding: 2, resource: { buffer: output } },
      { binding: 3, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((width * height * outChannels) / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  paramsBuffer.destroy();
  return output;
}

async function runPixelShuffle(device, input, width, height, channels, shader) {
  const outWidth = width * 2;
  const outHeight = height * 2;
  const outputByteLength = outWidth * outHeight * channels * 4;
  const output = device.createBuffer({
    size: alignedBufferSize(outputByteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: alignedBufferSize(outputByteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const params = new Uint32Array([width, height, channels, 0]);
  const paramsBuffer = device.createBuffer({
    size: alignedBufferSize(params.byteLength),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, params);
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: input } },
      { binding: 1, resource: { buffer: output } },
      { binding: 2, resource: { buffer: paramsBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil((outWidth * outHeight * channels) / 64));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputByteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  output.destroy();
  readback.destroy();
  paramsBuffer.destroy();
  return { width: outWidth, height: outHeight, data };
}

function alignedBufferSize(byteLength) {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}
`;
}

const conv2dShader = `
struct Params {
  width: u32,
  height: u32,
  in_channels: u32,
  out_channels: u32,
  weight_base: u32,
  bias_base: u32,
  relu: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let plane = params.width * params.height;
  let total = plane * params.out_channels;
  let index = id.x;
  if (index >= total) {
    return;
  }

  let oc = index / plane;
  let pixel = index % plane;
  let x = pixel % params.width;
  let y = pixel / params.width;
  var acc = weights[params.bias_base + oc];

  for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
    for (var ky: u32 = 0u; ky < 3u; ky = ky + 1u) {
      let iy = i32(y) + i32(ky) - 1;
      if (iy < 0 || iy >= i32(params.height)) {
        continue;
      }
      for (var kx: u32 = 0u; kx < 3u; kx = kx + 1u) {
        let ix = i32(x) + i32(kx) - 1;
        if (ix < 0 || ix >= i32(params.width)) {
          continue;
        }
        let weight_index = params.weight_base + (((oc * params.in_channels + ic) * 3u + ky) * 3u + kx);
        let input_index = ic * plane + u32(iy) * params.width + u32(ix);
        acc = acc + input_data[input_index] * weights[weight_index];
      }
    }
  }

  if (params.relu == 1u && acc < 0.0) {
    output_data[index] = 0.0;
  } else {
    output_data[index] = acc;
  }
}
`;

const pixelShuffleShader = `
struct Params {
  width: u32,
  height: u32,
  channels: u32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_data: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let out_width = params.width * 2u;
  let out_height = params.height * 2u;
  let in_plane = params.width * params.height;
  let out_plane = out_width * out_height;
  let total = out_plane * params.channels;
  let index = id.x;
  if (index >= total) {
    return;
  }

  let channel = index / out_plane;
  let pixel = index % out_plane;
  let out_x = pixel % out_width;
  let out_y = pixel / out_width;
  let subpixel = (out_y % 2u) * 2u + (out_x % 2u);
  let source_channel = channel * 4u + subpixel;
  let source_x = out_x / 2u;
  let source_y = out_y / 2u;
  let value = input_data[source_channel * in_plane + source_y * params.width + source_x];
  output_data[index] = clamp(value, 0.0, 1.0);
}
`;

function expectedMotionModes(width: number, height: number, frameCount: number): number {
  const yBlocks = blockCount(width, height);
  const uvBlocks = blockCount(Math.trunc(width / 2), Math.trunc(height / 2));
  return frameCount * (yBlocks + uvBlocks * 2);
}

function blockCount(width: number, height: number): number {
  return Math.ceil(width / 4) * Math.ceil(height / 4);
}

function parseHead(payload: Uint8Array): Record<string, string> {
  const text = textDecoder.decode(payload);
  const out: Record<string, string> = {};
  for (const line of text.split(/\r?\n/)) {
    const eq = line.indexOf("=");
    if (eq > 0) out[line.slice(0, eq)] = line.slice(eq + 1);
  }
  return out;
}

function readText(bytes: Uint8Array, offset: number, length: number): string {
  return textDecoder.decode(bytes.slice(offset, offset + length));
}
