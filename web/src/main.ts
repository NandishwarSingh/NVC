import { decodeBaseFrame, decodeCodecBaseFrame, decodeNeuralFrame, ensureNvcBasePacket, ensureNvcChunk, getPreviewInfo, loadNvcUrl, parseNvc, type NvcFile, type NvcLoadStats, type PreviewInfo } from "./nvc";

type PlayerOptions = {
  canvas: HTMLCanvasElement;
  preferWebGPU?: boolean;
};

type RenderMode = "preview" | "codec" | "neural";

export class NVCPlayer extends EventTarget {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private parsed: NvcFile | null = null;
  private renderMode: RenderMode = "preview";
  private preferWebGPU: boolean;
  private preview: PreviewInfo | null = null;
  private frameIndex = 0;
  private playing = false;
  private timer: number | null = null;

  constructor(options: PlayerOptions) {
    super();
    this.canvas = options.canvas;
    this.preferWebGPU = options.preferWebGPU ?? true;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D is unavailable");
    this.ctx = ctx;
  }

  async loadFile(file: File): Promise<void> {
    const buffer = await file.arrayBuffer();
    this.parsed = parseNvc(buffer);
    this.preview = getPreviewInfo(this.parsed);
    this.frameIndex = 0;
    this.emitStats({
      loadMode: "full",
      bytesLoaded: buffer.byteLength,
      fileBytes: file.size,
      chunksLoaded: this.parsed.chunks.map((chunk) => chunk.id),
    });
    this.dispatchEvent(new CustomEvent("ready", { detail: this.parsed }));
    await this.renderFrame(0);
  }

  async loadUrl(url: string): Promise<void> {
    const result = await loadNvcUrl(url);
    this.emitStats(result.stats);
    this.parsed = result.file;
    this.preview = getPreviewInfo(this.parsed);
    this.frameIndex = 0;
    this.dispatchEvent(new CustomEvent("ready", { detail: this.parsed }));
    await this.renderFrame(0);
  }

  async setRenderMode(mode: RenderMode): Promise<void> {
    this.renderMode = mode;
    if (mode !== "preview") this.pause();
    if (this.parsed) await this.renderFrame(this.frameIndex);
  }

  async seek(seconds: number): Promise<void> {
    if (!this.preview) return;
    const fps = previewFps(this.preview);
    const index = Math.max(0, Math.min(this.preview.frameCount - 1, Math.round(seconds * fps)));
    await this.renderFrame(index);
  }

  private async renderFrame(index: number): Promise<void> {
    if (!this.parsed) return;
    const maxIndex = this.preview ? this.preview.frameCount - 1 : 0;
    this.frameIndex = Math.max(0, Math.min(maxIndex, index));
    this.dispatchEvent(new CustomEvent("buffering", { detail: { mode: this.renderMode } }));
    if (this.renderMode === "codec" || this.renderMode === "neural") await this.ensureBasePacket(this.frameIndex);
    const neural =
      this.renderMode === "neural"
        ? await decodeNeuralFrame(this.parsed, this.frameIndex, { maxInputWidth: 480, cpuMaxInputWidth: 160, preferWebGPU: this.preferWebGPU, webGpuTimeoutMs: 3000, preferCodecInput: true })
        : null;
    const firstFrame = neural ? neural.imageData : this.renderMode === "codec" ? decodeCodecBaseFrame(this.parsed, this.frameIndex) : decodeBaseFrame(this.parsed, this.frameIndex);
    this.canvas.width = firstFrame.width;
    this.canvas.height = firstFrame.height;
    this.ctx.putImageData(firstFrame, 0, 0);
    this.dispatchEvent(
      new CustomEvent("frame", {
        detail: {
          index: this.frameIndex,
          mode: this.renderMode,
          width: firstFrame.width,
          height: firstFrame.height,
          backend: neural?.backend ?? (this.renderMode === "codec" ? "native-base" : "preview"),
          fallbackReason: neural?.fallbackReason,
          seconds: this.preview ? this.frameIndex / previewFps(this.preview) : 0,
          frameCount: this.preview?.frameCount ?? 1,
          fps: this.preview ? previewFps(this.preview) : 1,
        },
      }),
    );
  }

  play(): void {
    if (!this.preview || this.preview.frameCount <= 1 || this.playing) return;
    this.playing = true;
    this.dispatchEvent(new Event("play"));
    this.scheduleNextFrame();
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

  private async ensureBasePacket(frameIndex: number): Promise<void> {
    if (!this.parsed) return;
    const stats = await ensureNvcBasePacket(this.parsed, frameIndex);
    if (stats) this.emitStats(stats);
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
const playPause = document.querySelector<HTMLButtonElement>("#playPause");
const seek = document.querySelector<HTMLInputElement>("#seek");
const timecode = document.querySelector<HTMLElement>("#timecode");
const modeButtons = document.querySelectorAll<HTMLButtonElement>("[data-mode]");

if (!canvas || !dropzone || !metadata || !log || !webgpu || !loadSample || !playPause || !seek || !timecode) {
  throw new Error("Missing required DOM nodes");
}

const player = new NVCPlayer({ canvas, preferWebGPU: true });
let lastLoadStats: NvcLoadStats | null = null;

webgpu.textContent =
  "gpu" in navigator
    ? "WebGPU API detected. Neural mode reconstructs from codec base packets first, then CPU fallback."
    : "WebGPU API not detected. CPU MOD0 neural reconstruction is active.";

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
        ? "Decoding native BAS5 packet..."
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
