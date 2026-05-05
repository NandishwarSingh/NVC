import { mkdir, rm } from "node:fs/promises";
import { extname, join } from "node:path";
import { tmpdir } from "node:os";

const root = new URL(".", import.meta.url);
const projectRoot = new URL("..", root);
const port = Number(Bun.env.PORT ?? "5173");
const nvcBin = new URL("../zig-out/bin/nvc", root).pathname;
const defaultModel = new URL("../ml/exports/nvc-tinysr-v0-xiph-3gb.modl", root).pathname;

function file(path: string) {
  return Bun.file(new URL(path, root));
}

function parseRange(range: string, size: number): { start: number; end: number } | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(range);
  if (!match) return null;

  const startText = match[1];
  const endText = match[2];
  if (!startText && !endText) return null;

  if (!startText) {
    const suffixLength = Number(endText);
    if (!Number.isSafeInteger(suffixLength) || suffixLength <= 0) return null;
    const start = Math.max(0, size - suffixLength);
    return { start, end: size - 1 };
  }

  const start = Number(startText);
  const end = endText ? Number(endText) : size - 1;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end)) return null;
  if (start < 0 || end < start || start >= size) return null;
  return { start, end: Math.min(end, size - 1) };
}

function rangeError(size: number, contentType: string): Response {
  return new Response("Requested range not satisfiable", {
    status: 416,
    headers: {
      "accept-ranges": "bytes",
      "content-range": `bytes */${size}`,
      "content-type": contentType,
    },
  });
}

function serveFile(req: Request, blob: Blob, contentType: string): Response {
  const headers = {
    "accept-ranges": "bytes",
    "content-type": contentType,
  };
  const range = req.headers.get("range");
  if (!range) {
    return new Response(blob, {
      headers: {
        ...headers,
        "content-length": String(blob.size),
      },
    });
  }

  const parsed = parseRange(range, blob.size);
  if (!parsed) return rangeError(blob.size, contentType);

  const body = blob.slice(parsed.start, parsed.end + 1);
  return new Response(body, {
    status: 206,
    headers: {
      ...headers,
      "content-length": String(body.size),
      "content-range": `bytes ${parsed.start}-${parsed.end}/${blob.size}`,
    },
  });
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function safeStem(name: string, fallback: string): string {
  const base = name.replace(/\.[^.]*$/, "").replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return base || fallback;
}

function safeExt(name: string, fallback: string): string {
  const ext = extname(name).toLowerCase();
  return ext && ext.length <= 8 ? ext : fallback;
}

async function runNvc(args: string[]): Promise<string> {
  const proc = Bun.spawn(args, {
    cwd: projectRoot.pathname,
    stdout: "pipe",
    stderr: "pipe",
  });
  const [stdout, stderr, code] = await Promise.all([
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
    proc.exited,
  ]);
  if (code !== 0) {
    throw new Error((stderr || stdout || `nvc exited with code ${code}`).trim());
  }
  return `${stdout}${stderr}`.trim();
}

async function commandFileResponse(path: string, filename: string, contentType: string): Promise<Response> {
  const bytes = await Bun.file(path).arrayBuffer();
  return new Response(bytes, {
    headers: {
      "content-type": contentType,
      "content-length": String(bytes.byteLength),
      "content-disposition": `attachment; filename="${filename}"`,
    },
  });
}

async function handleEncode(req: Request): Promise<Response> {
  const form = await req.formData();
  const source = form.get("source");
  if (!(source instanceof File)) return json({ error: "Upload a source video." }, 400);
  const profileValue = String(form.get("profile") ?? "w1");
  const profile = profileValue === "xc" ? "xc" : "w1";
  const frameText = String(form.get("frames") ?? "60");
  const frames = Math.max(1, Math.min(900, Number.parseInt(frameText, 10) || 60));
  const workDir = join(tmpdir(), `nvc-web-encode-${crypto.randomUUID()}`);
  await mkdir(workDir, { recursive: true });
  try {
    const stem = safeStem(source.name, "source");
    const inputPath = join(workDir, `${stem}${safeExt(source.name, ".mp4")}`);
    const outputName = `${stem}-${profile}.nvc`;
    const outputPath = join(workDir, outputName);
    await Bun.write(inputPath, source);
    await runNvc([nvcBin, "encode", inputPath, outputPath, "--profile", profile, "--frames", String(frames), "--model", defaultModel]);
    return await commandFileResponse(outputPath, outputName, "application/octet-stream");
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
}

async function handleDecode(req: Request): Promise<Response> {
  const form = await req.formData();
  const source = form.get("source");
  if (!(source instanceof File)) return json({ error: "Upload an NVC file." }, 400);
  const workDir = join(tmpdir(), `nvc-web-decode-${crypto.randomUUID()}`);
  await mkdir(workDir, { recursive: true });
  try {
    const stem = safeStem(source.name, "decoded");
    const inputPath = join(workDir, `${stem}.nvc`);
    const outputName = `${stem}.mp4`;
    const outputPath = join(workDir, outputName);
    await Bun.write(inputPath, source);
    await runNvc([nvcBin, "decode", inputPath, outputPath]);
    return await commandFileResponse(outputPath, outputName, "video/mp4");
  } finally {
    await rm(workDir, { recursive: true, force: true });
  }
}

const server = Bun.serve({
  port,
  async fetch(req) {
    const url = new URL(req.url);
    if (req.method === "POST" && url.pathname === "/api/encode") {
      try {
        return await handleEncode(req);
      } catch (error) {
        return json({ error: error instanceof Error ? error.message : String(error) }, 500);
      }
    }
    if (req.method === "POST" && url.pathname === "/api/decode") {
      try {
        return await handleDecode(req);
      } catch (error) {
        return json({ error: error instanceof Error ? error.message : String(error) }, 500);
      }
    }
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(file("index.html"));
    }
    if (url.pathname === "/bundle.js") {
      const result = await Bun.build({
        entrypoints: [new URL("src/main.ts", root).pathname],
        target: "browser",
        sourcemap: "inline",
      });
      if (!result.success) {
        return new Response(result.logs.map((log) => log.message).join("\n"), { status: 500 });
      }
      return new Response(await result.outputs[0].text(), {
        headers: { "content-type": "text/javascript" },
      });
    }
    if (url.pathname === "/style.css") {
      return new Response(file("src/style.css"), {
        headers: { "content-type": "text/css" },
      });
    }
    if (url.pathname === "/samples/output.nvc") {
      return serveFile(req, Bun.file(new URL("../samples/output.nvc", root)), "application/octet-stream");
    }
    return new Response("Not found", { status: 404 });
  },
});

console.log(`NVC web player: http://localhost:${server.port}`);
