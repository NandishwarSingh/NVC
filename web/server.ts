const root = new URL(".", import.meta.url);
const port = Number(Bun.env.PORT ?? "5173");

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

const server = Bun.serve({
  port,
  async fetch(req) {
    const url = new URL(req.url);
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
