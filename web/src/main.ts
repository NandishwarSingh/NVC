import { decodeBaseFrame, decodeCodecBaseFrame, decodeNeuralFrame, decodeNeuralFrameOnnx, ensureNvcBasePacket, ensureNvcChunk, getPreviewInfo, loadNvcUrl, parseNvc, prepareBaseDecode, type NvcFile, type NvcLoadStats, type PreviewInfo } from "./nvc";

type PlayerOptions = {
  canvas: HTMLCanvasElement;
  preferWebGPU?: boolean;
};

type RenderMode = "preview" | "codec" | "neural";

export class NVCPlayer extends EventTarget {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private parsed: NvcFile | null = null;
  // Default mode is Neural — matches the UI's default-active segmented button and
  // gives the highest-quality first frame. While the user presses Play we transparently
  // render in Codec mode (BAS6 cache lookup, ~1ms) so the video animates at native fps;
  // returning to paused state re-renders in renderMode (Neural) for the still-frame view.
  private renderMode: RenderMode = "neural";
  // The mode used while `playing` is true. Always "codec" because per-frame Neural SR
  // (ORT-Web run on WebGPU + composite) costs ~100ms+ which can't sustain 30 fps.
  private readonly playbackMode: RenderMode = "codec";
  private preferWebGPU: boolean;
  private preview: PreviewInfo | null = null;
  private frameIndex = 0;
  private playing = false;
  private timer: number | null = null;
  private frameDisplaySize: { width: number; height: number } | null = null;
  private basePrep: Promise<void> | null = null;
  private neuralFrameCache: Map<number, ImageData> = new Map();
  private codecFrameCache: Map<number, ImageData> = new Map();
  // True when the loaded file is XC profile. We keep parser + decoders intact (they still
  // run for the Decode tool's server-side path), but we skip in-browser playback because
  // TinySR x2 can't recover a 4× ratio cleanly. UI shows an info panel instead.
  private isXc = false;
  private fileName = "";
  private readonly handleResize = () => this.resizeCanvasDisplay();

  constructor(options: PlayerOptions) {
    super();
    this.canvas = options.canvas;
    this.preferWebGPU = options.preferWebGPU ?? true;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D is unavailable");
    this.ctx = ctx;
    window.addEventListener("resize", this.handleResize);
  }

  async loadFile(file: File): Promise<void> {
    this.resetCaches();
    const buffer = await file.arrayBuffer();
    this.parsed = parseNvc(buffer);
    this.preview = getPreviewInfo(this.parsed);
    this.frameIndex = 0;
    this.fileName = file.name || "input.nvc";
    this.detectProfile(file.size);
    this.emitStats({
      loadMode: "full",
      bytesLoaded: buffer.byteLength,
      fileBytes: file.size,
      chunksLoaded: this.parsed.chunks.map((chunk) => chunk.id),
    });
    this.dispatchEvent(new CustomEvent("ready", { detail: this.parsed }));
    if (this.isXc) return; // skip canvas; UI shows info panel instead
    await this.warmBaseCache();
    await this.renderFrame(0);
  }

  async loadUrl(url: string): Promise<void> {
    this.resetCaches();
    const result = await loadNvcUrl(url);
    this.emitStats(result.stats);
    this.parsed = result.file;
    this.preview = getPreviewInfo(this.parsed);
    this.frameIndex = 0;
    const slash = url.lastIndexOf("/");
    this.fileName = slash >= 0 ? url.slice(slash + 1) : url;
    this.detectProfile(result.stats.fileBytes ?? 0);
    this.dispatchEvent(new CustomEvent("ready", { detail: this.parsed }));
    if (this.isXc) return;
    await this.warmBaseCache();
    await this.renderFrame(0);
  }

  private resetCaches(): void {
    this.basePrep = null;
    this.neuralFrameCache.clear();
    this.codecFrameCache.clear();
  }

  // Warm the BAS6/VP9 frame cache before the first render so Codec/Neural mode never
  // hits "cache not ready" the moment the user clicks Play. Stores the in-flight promise
  // so concurrent renderFrame calls all await the same warmup.
  private async warmBaseCache(): Promise<void> {
    if (!this.parsed) return;
    if (!this.basePrep) {
      this.dispatchEvent(new CustomEvent("buffering", { detail: { mode: "codec" } }));
      this.basePrep = prepareBaseDecode(this.parsed).catch(() => undefined);
    }
    return this.basePrep;
  }

  isXcProfile(): boolean {
    return this.isXc;
  }

  getFileName(): string {
    return this.fileName;
  }

  private detectProfile(fileBytes: number): void {
    if (!this.parsed) return;
    const profile = (this.parsed.head.profile || "").toUpperCase();
    this.isXc = profile.includes("XC");
    const head = this.parsed.head;
    this.dispatchEvent(
      new CustomEvent("profile", {
        detail: {
          profile,
          isXc: this.isXc,
          fileName: this.fileName,
          fileBytes,
          frames: Number(head.frames || 0),
          baseWidth: Number(head.base_width || 0),
          baseHeight: Number(head.base_height || 0),
          sourceWidth: Number(head.width || 0),
          sourceHeight: Number(head.height || 0),
          fpsNum: Number(head.fps_num || 30),
          fpsDen: Number(head.fps_den || 1),
        },
      }),
    );
  }

  async setRenderMode(mode: RenderMode): Promise<void> {
    if (this.isXc && (mode === "codec" || mode === "neural")) {
      // Hard-block: in-browser Codec/Neural for XC produces ¼-resolution+CSS-stretched output
      // that misrepresents the codec. Tell the user where the real quality is.
      this.dispatchEvent(
        new CustomEvent("xc-blocked", {
          detail: { mode, suggestion: `nvc decode ${this.fileName} out.mp4 --enhancer realesrgan --interpolate-rife` },
        }),
      );
      return;
    }
    this.renderMode = mode;
    if (mode !== "preview") this.pause();
    if (this.parsed && !this.isXc) await this.renderFrame(this.frameIndex);
  }

  async seek(seconds: number): Promise<void> {
    if (!this.preview || this.isXc) return;
    const fps = previewFps(this.preview);
    const index = Math.max(0, Math.min(this.preview.frameCount - 1, Math.round(seconds * fps)));
    await this.renderFrame(index);
  }

  private async renderFrame(index: number): Promise<void> {
    if (!this.parsed) return;
    const maxIndex = this.preview ? this.preview.frameCount - 1 : 0;
    this.frameIndex = Math.max(0, Math.min(maxIndex, index));

    // Effective mode: while playing, force Codec (fast cache lookup, sustains native fps);
    // while paused/seeking, honour the user's chosen renderMode (Neural by default).
    const effective: RenderMode = this.playing ? this.playbackMode : this.renderMode;

    this.dispatchEvent(new CustomEvent("buffering", { detail: { mode: effective } }));
    if (effective === "codec" || effective === "neural") await this.warmBaseCache();

    let firstFrame: ImageData;
    let backend = "preview";
    let fallbackReason: string | undefined;

    if (effective === "neural") {
      const cached = this.neuralFrameCache.get(this.frameIndex);
      if (cached) {
        firstFrame = cached;
        backend = "neural-cache";
      } else {
        const neural = await this.runNeural();
        firstFrame = neural.imageData;
        backend = neural.backend;
        fallbackReason = neural.fallbackReason;
        // Cache only the still-frame Neural output (saves the 100ms+ ORT round-trip on re-seek).
        // Cap is conservative because each frame is now at source resolution (e.g. 1080p ≈ 8 MB).
        if (this.neuralFrameCache.size < 60) this.neuralFrameCache.set(this.frameIndex, firstFrame);
      }
    } else if (effective === "codec") {
      const cached = this.codecFrameCache.get(this.frameIndex);
      if (cached) {
        firstFrame = cached;
        backend = "codec-cache";
      } else {
        firstFrame = decodeCodecBaseFrame(this.parsed, this.frameIndex);
        backend = "native-base";
        // Codec frames are decoded in O(1) from the BAS6 cache, but the RGB→ImageData
        // conversion still costs a few ms; caching makes Play smooth.
        if (this.codecFrameCache.size < 600) this.codecFrameCache.set(this.frameIndex, firstFrame);
      }
    } else {
      firstFrame = decodeBaseFrame(this.parsed, this.frameIndex);
    }

    this.canvas.width = firstFrame.width;
    this.canvas.height = firstFrame.height;
    this.frameDisplaySize = { width: firstFrame.width, height: firstFrame.height };
    this.resizeCanvasDisplay();
    this.ctx.putImageData(firstFrame, 0, 0);
    this.dispatchEvent(
      new CustomEvent("frame", {
        detail: {
          index: this.frameIndex,
          mode: effective,
          width: firstFrame.width,
          height: firstFrame.height,
          backend,
          fallbackReason,
          seconds: this.preview ? this.frameIndex / previewFps(this.preview) : 0,
          frameCount: this.preview?.frameCount ?? 1,
          fps: this.preview ? previewFps(this.preview) : 1,
        },
      }),
    );
  }

  play(): void {
    if (this.isXc) return; // no in-browser playback for XC
    if (!this.preview || this.preview.frameCount <= 1 || this.playing) return;
    this.playing = true;
    this.dispatchEvent(new Event("play"));
    // Make sure the codec cache is warmed before the first scheduled tick fires —
    // otherwise the user sees Play do "nothing" for a few seconds while VP9 decodes.
    this.warmBaseCache().then(() => {
      // Re-render the current frame in playbackMode so the user sees the codec image
      // immediately, then start ticking.
      this.renderFrame(this.frameIndex).then(() => this.scheduleNextFrame());
    });
  }

  pause(): void {
    this.playing = false;
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.dispatchEvent(new Event("pause"));
  }

  destroy(): void {
    this.pause();
    window.removeEventListener("resize", this.handleResize);
  }

  private resizeCanvasDisplay(): void {
    const size = this.frameDisplaySize ?? { width: this.canvas.width, height: this.canvas.height };
    if (size.width <= 0 || size.height <= 0) return;
    const wrap = this.canvas.parentElement;
    const wrapStyle = wrap ? window.getComputedStyle(wrap) : null;
    const paddingX = wrapStyle ? parseFloat(wrapStyle.paddingLeft) + parseFloat(wrapStyle.paddingRight) : 0;
    const availableWidth = Math.max(180, (wrap?.clientWidth ?? 1120) - paddingX);
    const availableHeight = Math.max(240, Math.min(760, Math.floor(window.innerHeight * 0.64)));
    const scale = Math.min(availableWidth / size.width, availableHeight / size.height);
    const displayWidth = Math.max(1, Math.round(size.width * scale));
    const displayHeight = Math.max(1, Math.round(size.height * scale));
    this.canvas.style.width = `${displayWidth}px`;
    this.canvas.style.height = `${displayHeight}px`;
    this.canvas.style.aspectRatio = `${size.width} / ${size.height}`;
  }

  private scheduleNextFrame(): void {
    if (!this.playing || !this.preview) return;
    const delay = Math.max(16, 1000 / previewFps(this.preview));
    this.timer = window.setTimeout(async () => {
      if (!this.playing || !this.preview) return;
      const next = (this.frameIndex + 1) % this.preview.frameCount;
      try {
        await this.renderFrame(next);
      } catch (error) {
        this.pause();
        this.dispatchEvent(new CustomEvent("error", { detail: error }));
        return;
      }
      this.scheduleNextFrame();
    }, delay);
  }

  private async ensureChunk(id: string): Promise<void> {
    if (!this.parsed) return;
    const stats = await ensureNvcChunk(this.parsed, id);
    if (stats) this.emitStats(stats);
  }

  // Neural mode tries the bigger ORT-Web realesr-animevideov3 model first (~2.5 MB,
  // matches the CLI's --enhancer realesrgan output), falls back to the bundled MOD0
  // TinySR (~5K params) if ORT fails to load (no WebGPU + no WASM, model fetch error,
  // etc.). Either way the user sees a Neural-mode frame.
  private async runNeural() {
    if (!this.parsed) throw new Error("no parsed file");
    try {
      return await decodeNeuralFrameOnnx(this.parsed, this.frameIndex, { preferCodecInput: true });
    } catch (error) {
      this.dispatchEvent(
        new CustomEvent("neural-fallback", {
          detail: { reason: error instanceof Error ? error.message : String(error) },
        }),
      );
      return decodeNeuralFrame(this.parsed, this.frameIndex, {
        maxInputWidth: 480,
        cpuMaxInputWidth: 160,
        preferWebGPU: this.preferWebGPU,
        webGpuTimeoutMs: 3000,
        preferCodecInput: true,
      });
    }
  }

  // Legacy ensureBasePacket kept for BAS5 fallback files; BAS6 path uses warmBaseCache.
  private async ensureBasePacket(frameIndex: number): Promise<void> {
    if (!this.parsed) return;
    const stats = await ensureNvcBasePacket(this.parsed, frameIndex);
    if (stats) this.emitStats(stats);
    // For BAS6 files, the BASE chunk is a single VP9 IVF bitstream; pre-decode all frames
    // through WebCodecs into a sync ImageData cache before Codec/Neural decode can run.
    await prepareBaseDecode(this.parsed);
  }

  private emitStats(stats: NvcLoadStats): void {
    this.dispatchEvent(new CustomEvent("stats", { detail: stats }));
  }
}

function previewFps(preview: PreviewInfo): number {
  return preview.fpsDen === 0 ? 1 : preview.fpsNum / preview.fpsDen;
}

const canvas = document.querySelector<HTMLCanvasElement>("#preview");
const dropzone = document.querySelector<HTMLElement>("#dropzone");
const metadata = document.querySelector<HTMLElement>("#metadata");
const log = document.querySelector<HTMLElement>("#log");
const webgpu = document.querySelector<HTMLElement>("#webgpu");
const loadSample = document.querySelector<HTMLButtonElement>("#loadSample");
const playFile = document.querySelector<HTMLInputElement>("#playFile");
const playPause = document.querySelector<HTMLButtonElement>("#playPause");
const seek = document.querySelector<HTMLInputElement>("#seek");
const timecode = document.querySelector<HTMLElement>("#timecode");
const modeButtons = document.querySelectorAll<HTMLButtonElement>("[data-mode]");
const encodeForm = document.querySelector<HTMLFormElement>("#encodeForm");
const encodeFile = document.querySelector<HTMLInputElement>("#encodeFile");
const encodeFileName = document.querySelector<HTMLElement>("#encodeFileName");
const encodeProfile = document.querySelector<HTMLSelectElement>("#encodeProfile");
const encodeFrames = document.querySelector<HTMLInputElement>("#encodeFrames");
const profileNote = document.querySelector<HTMLElement>("#profileNote");
const encodeSubmit = document.querySelector<HTMLButtonElement>("#encodeSubmit");
const encodeStatus = document.querySelector<HTMLElement>("#encodeStatus");
const decodeForm = document.querySelector<HTMLFormElement>("#decodeForm");
const decodeFile = document.querySelector<HTMLInputElement>("#decodeFile");
const decodeFileName = document.querySelector<HTMLElement>("#decodeFileName");
const decodeSubmit = document.querySelector<HTMLButtonElement>("#decodeSubmit");
const decodeStatus = document.querySelector<HTMLElement>("#decodeStatus");
const decodedVideo = document.querySelector<HTMLVideoElement>("#decodedVideo");
const xcPanel = document.querySelector<HTMLElement>("#xcPanel");
const xcStats = document.querySelector<HTMLElement>("#xcStats");
const xcFileNameEl = document.querySelector<HTMLElement>("#xcFileName");
const canvasWrap = document.querySelector<HTMLElement>(".canvas-wrap");
const segmentedBar = document.querySelector<HTMLElement>(".segmented");

if (
  !canvas ||
  !dropzone ||
  !metadata ||
  !log ||
  !webgpu ||
  !loadSample ||
  !playFile ||
  !playPause ||
  !seek ||
  !timecode ||
  !encodeForm ||
  !encodeFile ||
  !encodeFileName ||
  !encodeProfile ||
  !encodeFrames ||
  !profileNote ||
  !encodeSubmit ||
  !encodeStatus ||
  !decodeForm ||
  !decodeFile ||
  !decodeFileName ||
  !decodeSubmit ||
  !decodeStatus ||
  !decodedVideo
) {
  throw new Error("Missing required DOM nodes");
}

const player = new NVCPlayer({ canvas, preferWebGPU: true });
let lastLoadStats: NvcLoadStats | null = null;
let decodedVideoUrl: string | null = null;

webgpu.textContent =
  "gpu" in navigator
    ? "WebGPU API detected. Neural mode reconstructs from codec base packets first, then CPU fallback."
    : "WebGPU API not detected. CPU MOD0 neural reconstruction is active.";
updateProfileNote();

player.addEventListener("profile", (event) => {
  const d = (event as CustomEvent<{
    profile: string;
    isXc: boolean;
    fileName: string;
    fileBytes: number;
    frames: number;
    baseWidth: number;
    baseHeight: number;
    sourceWidth: number;
    sourceHeight: number;
    fpsNum: number;
    fpsDen: number;
  }>).detail;

  if (xcPanel && xcStats && xcFileNameEl && canvasWrap && segmentedBar) {
    if (d.isXc) {
      const fps = d.fpsDen > 0 ? d.fpsNum / d.fpsDen : 30;
      const durationSec = fps > 0 ? d.frames / fps : 0;
      const sizeMB = d.fileBytes / (1024 * 1024);
      const ratio = d.frames && d.fileBytes ? estimateSourceMB(d.sourceWidth, d.sourceHeight, durationSec) / sizeMB : 0;
      xcStats.innerHTML = "";
      pushStat(xcStats, sizeMB.toFixed(2) + " MB", `this .nvc on disk (${d.fileName})`);
      pushStat(xcStats, `${d.sourceWidth}×${d.sourceHeight}`, `source dims, ${d.frames} frames @ ${fps.toFixed(0)} fps`);
      pushStat(xcStats, `${d.baseWidth}×${d.baseHeight}`, `1/4-res base, VP9 IVF inside .nvc`);
      if (ratio > 0) pushStat(xcStats, `~${ratio.toFixed(1)}×`, "smaller than the source MP4 (estimated)");
      xcFileNameEl.textContent = d.fileName;
      xcPanel.hidden = false;
      canvasWrap.classList.add("xc-mode");
      segmentedBar.classList.add("xc-mode");
    } else {
      xcPanel.hidden = true;
      canvasWrap.classList.remove("xc-mode");
      segmentedBar.classList.remove("xc-mode");
    }
  }
});

function pushStat(host: HTMLElement, big: string, sub: string): void {
  const li = document.createElement("li");
  const strong = document.createElement("strong");
  strong.textContent = big;
  const span = document.createElement("span");
  span.textContent = sub;
  li.append(strong, span);
  host.append(li);
}

function estimateSourceMB(w: number, h: number, sec: number): number {
  // Rough heuristic: assume the source MP4 was ~4 Mbps for typical phone video.
  // Used only to display an "approx ratio". Honest with itself; flagged "estimated".
  const bitsPerSec = 4_000_000;
  return (bitsPerSec * sec) / 8 / (1024 * 1024);
}

player.addEventListener("xc-blocked", (event) => {
  const d = (event as CustomEvent<{ mode: RenderMode; suggestion: string }>).detail;
  if (log) {
    log.textContent = `XC files don't play in-browser cleanly (${d.mode} mode would only run a 5K-param SR model on a 1/4-res base, then CSS-stretch).\nFor full quality run the CLI:\n  ${d.suggestion}`;
  }
});

player.addEventListener("ready", (event) => {
  const parsed = (event as CustomEvent<NvcFile>).detail;
  const preview = getPreviewInfo(parsed);
  seek.value = "0";
  seek.max = String(Math.max(0, (preview?.frameCount ?? 1) - 1));
  timecode.textContent = formatTime(0, preview);
  playPause.textContent = "Play";
  metadata.innerHTML = "";
  for (const [key, value] of Object.entries(parsed.head)) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    metadata.append(dt, dd);
  }
  if (preview) {
    appendMetadataObject("preview", {
      width: preview.width,
      height: preview.height,
      fps: previewFps(preview).toFixed(3),
      source_frames: preview.frameCount,
      stored_frames: preview.storedFrameCount ?? preview.frameCount,
      frame_stride: preview.frameStride ?? 1,
    });
  }
  if (parsed.model) {
    for (const [key, value] of Object.entries(parsed.model)) {
      if (value === undefined) continue;
      if (key === "metadata" || key === "weights") continue;
      const dt = document.createElement("dt");
      dt.textContent = `model_${key}`;
      const dd = document.createElement("dd");
      dd.textContent = String(value);
      metadata.append(dt, dd);
    }
  }
  if (parsed.feature) appendMetadataObject("feature", parsed.feature);
  if (parsed.color) appendMetadataObject("color", parsed.color);
  if (parsed.grain) appendMetadataObject("grain", parsed.grain);
  log.textContent = parsed.chunks
    .map((chunk: { id: string; offset: number; length: number }) => `${chunk.id} offset=${chunk.offset} bytes=${chunk.length}`)
    .join("\n");
});

function appendMetadataObject(prefix: string, values: Record<string, unknown>): void {
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) continue;
    if (value instanceof Uint8Array) continue;
    const dt = document.createElement("dt");
    dt.textContent = `${prefix}_${key}`;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    metadata.append(dt, dd);
  }
}

player.addEventListener("stats", (event) => {
  const detail = (event as CustomEvent<NvcLoadStats>).detail;
  lastLoadStats = detail;
  log.textContent = formatLoadStats(detail);
});

player.addEventListener("buffering", (event) => {
  const mode = (event as CustomEvent<{ mode: RenderMode }>).detail.mode;
  log.textContent =
    mode === "neural"
      ? "Running MOD0 TinySR neural reconstruction..."
      : mode === "codec"
        ? "Decoding base stream (VP9 via WebCodecs / BAS5 fallback)..."
        : "Decoding PRVW preview...";
});

player.addEventListener("frame", (event) => {
  const detail = (event as CustomEvent<{
    index: number;
    mode: RenderMode;
    width: number;
    height: number;
    backend: string;
    fallbackReason?: string;
    seconds: number;
    frameCount: number;
    fps: number;
  }>).detail;
  const fallback = detail.fallbackReason ? ` (fallback: ${detail.fallbackReason})` : "";
  seek.value = String(detail.index);
  timecode.textContent = formatSeconds(detail.seconds, detail.frameCount / detail.fps);
  const loadStats = lastLoadStats ? `${formatLoadStats(lastLoadStats)}\n` : "";
  log.textContent = `${loadStats}Rendered ${detail.mode} frame ${detail.index + 1}/${detail.frameCount}: ${detail.width}x${detail.height} via ${detail.backend}${fallback}`;
});

player.addEventListener("error", (event) => {
  const custom = (event as CustomEvent).detail;
  log.textContent = String(custom ?? (event as ErrorEvent).error ?? "Unknown player error");
});

player.addEventListener("play", () => {
  playPause.textContent = "Pause";
});

player.addEventListener("pause", () => {
  playPause.textContent = "Play";
});

player.addEventListener("modechange", (event) => {
  const mode = (event as CustomEvent<{ mode: RenderMode }>).detail.mode;
  for (const item of modeButtons) item.classList.toggle("active", item.dataset.mode === mode);
});

for (const button of modeButtons) {
  button.addEventListener("click", async () => {
    const mode = button.dataset.mode as RenderMode;
    for (const item of modeButtons) item.classList.toggle("active", item === button);
    try {
      await player.setRenderMode(mode);
    } catch (error) {
      log.textContent = error instanceof Error ? error.message : String(error);
    }
  });
}

playPause.addEventListener("click", () => {
  if (playPause.textContent === "Pause") player.pause();
  else player.play();
});

seek.addEventListener("input", async () => {
  const frame = Number(seek.value);
  const previewFrames = Number(seek.max) + 1;
  const fps = previewFrames > 1 ? Number(document.body.dataset.previewFps ?? "30") : 1;
  try {
    await player.seek(frame / fps);
  } catch (error) {
    log.textContent = error instanceof Error ? error.message : String(error);
  }
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, () => {
    dropzone.classList.remove("drag");
  });
}

dropzone.addEventListener("drop", async (event) => {
  event.preventDefault();
  const file = event.dataTransfer?.files[0];
  if (!file) return;
  try {
    log.textContent = `Loading ${file.name}...`;
    await player.loadFile(file);
  } catch (error) {
    log.textContent = error instanceof Error ? error.message : String(error);
  }
});

loadSample.addEventListener("click", async () => {
  try {
    log.textContent = "Loading sample output.nvc...";
    await player.loadUrl("/samples/output.nvc");
  } catch (error) {
    log.textContent = error instanceof Error ? error.message : String(error);
  }
});

playFile.addEventListener("change", async () => {
  const file = playFile.files?.[0];
  if (!file) return;
  try {
    log.textContent = `Loading ${file.name}...`;
    await player.loadFile(file);
  } catch (error) {
    log.textContent = error instanceof Error ? error.message : String(error);
  }
});

encodeFile.addEventListener("change", () => {
  const file = encodeFile.files?.[0];
  encodeFileName.textContent = file ? file.name : "Choose source video";
  setStatus(encodeStatus, file ? `${formatBytes(file.size)} selected` : "Idle");
});

encodeProfile.addEventListener("change", updateProfileNote);

decodeFile.addEventListener("change", () => {
  const file = decodeFile.files?.[0];
  decodeFileName.textContent = file ? file.name : "Choose NVC file";
  setStatus(decodeStatus, file ? `${formatBytes(file.size)} selected` : "Idle");
});

encodeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = encodeFile.files?.[0];
  if (!file) {
    setStatus(encodeStatus, "Choose a source video first.", true);
    return;
  }
  encodeSubmit.disabled = true;
  setStatus(encodeStatus, `Encoding ${file.name}...`);
  try {
    const form = new FormData();
    form.append("source", file);
    form.append("profile", encodeProfile.value);
    form.append("frames", encodeFrames.value.trim());
    const result = await postFileJob("/api/encode", form);
    downloadBlob(result.blob, result.filename);
    setStatus(encodeStatus, `Created ${result.filename} (${formatBytes(result.blob.size)})`);
    const nvcFile = new File([result.blob], result.filename, { type: "application/octet-stream" });
    await player.loadFile(nvcFile);
  } catch (error) {
    setStatus(encodeStatus, error instanceof Error ? error.message : String(error), true);
  } finally {
    encodeSubmit.disabled = false;
  }
});

decodeForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = decodeFile.files?.[0];
  if (!file) {
    setStatus(decodeStatus, "Choose an NVC file first.", true);
    return;
  }
  decodeSubmit.disabled = true;
  setStatus(decodeStatus, `Decoding ${file.name}...`);
  try {
    const form = new FormData();
    form.append("source", file);
    const result = await postFileJob("/api/decode", form);
    downloadBlob(result.blob, result.filename);
    if (decodedVideoUrl) URL.revokeObjectURL(decodedVideoUrl);
    decodedVideoUrl = URL.createObjectURL(result.blob);
    decodedVideo.src = decodedVideoUrl;
    decodedVideo.hidden = false;
    setStatus(decodeStatus, `Created ${result.filename} (${formatBytes(result.blob.size)})`);
    await player.loadFile(file);
  } catch (error) {
    setStatus(decodeStatus, error instanceof Error ? error.message : String(error), true);
  } finally {
    decodeSubmit.disabled = false;
  }
});

async function postFileJob(endpoint: string, form: FormData): Promise<{ blob: Blob; filename: string }> {
  const response = await fetch(endpoint, { method: "POST", body: form });
  if (!response.ok) {
    const message = await response
      .json()
      .then((payload) => String(payload.error ?? response.statusText))
      .catch(() => response.statusText);
    throw new Error(message);
  }
  const blob = await response.blob();
  return {
    blob,
    filename: filenameFromDisposition(response.headers.get("content-disposition")) ?? (endpoint.includes("encode") ? "output.nvc" : "decoded.mp4"),
  };
}

function filenameFromDisposition(value: string | null): string | null {
  const match = value?.match(/filename="([^"]+)"/);
  return match?.[1] ?? null;
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function setStatus(node: HTMLElement, message: string, error = false): void {
  node.textContent = message;
  node.classList.toggle("error", error);
}

function updateProfileNote(): void {
  const isXc = encodeProfile.value === "xc";
  profileNote.innerHTML = isXc
    ? "<strong>NVC-XC</strong><span>Maximum compression. Uses a quarter-resolution base stream, 12 fps cap, heavier quantization, and more neural reconstruction.</span>"
    : "<strong>NVC-W1</strong><span>Realtime web playback target. Keeps up to 30 fps, uses a larger base stream, and usually makes bigger files.</span>";
}

function formatTime(seconds: number, preview: PreviewInfo | null): string {
  const duration = preview ? preview.frameCount / previewFps(preview) : 0;
  if (preview) document.body.dataset.previewFps = String(previewFps(preview));
  return formatSeconds(seconds, duration);
}

function formatSeconds(seconds: number, duration: number): string {
  return `${clock(seconds)} / ${clock(duration)}`;
}

function clock(seconds: number): string {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MiB`;
}

function formatLoadStats(detail: NvcLoadStats): string {
  if (detail.loadMode === "cache") return `Loaded from cache: ${detail.chunksLoaded.join(", ")}`;
  const total = detail.fileBytes ? ` of ${formatBytes(detail.fileBytes)}` : "";
  return `Loaded ${formatBytes(detail.bytesLoaded)}${total} by ${detail.loadMode} fetch: ${detail.chunksLoaded.join(", ")}`;
}
