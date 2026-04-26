const http = require("http");
const fs = require("fs");
const fsp = require("fs").promises;
const path = require("path");

const root = __dirname;
const preferredPort = Number(process.env.PORT || 4173);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml"
};

function send(res, status, body, type = "text/plain; charset=utf-8") {
  res.writeHead(status, {
    "Content-Type": type,
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function resolveRequestPath(reqUrl) {
  const url = new URL(reqUrl, "http://localhost");
  let pathname = decodeURIComponent(url.pathname);

  if (pathname === "/") {
    pathname = "/app/index.html";
  }

  const target = path.normalize(path.join(root, pathname));
  const rel = path.relative(root, target);
  if (rel.startsWith("..") || path.isAbsolute(rel)) {
    return null;
  }
  return target;
}

async function serve(req, res) {
  const target = resolveRequestPath(req.url);
  if (!target) {
    send(res, 403, "Forbidden");
    return;
  }

  try {
    let resolved = target;
    const stat = await fsp.stat(resolved);
    if (stat.isDirectory()) {
      resolved = path.join(resolved, "index.html");
    }
    const data = await fsp.readFile(resolved);
    const ext = path.extname(resolved).toLowerCase();
    send(res, 200, data, mimeTypes[ext] || "application/octet-stream");
  } catch (error) {
    send(res, error.code === "ENOENT" ? 404 : 500, error.code === "ENOENT" ? "Not found" : "Server error");
  }
}

const server = http.createServer((req, res) => {
  serve(req, res).catch(() => send(res, 500, "Server error"));
});

function listen(port) {
  server.once("error", (error) => {
    if (error.code === "EADDRINUSE" && port < preferredPort + 20) {
      listen(port + 1);
      return;
    }
    throw error;
  });

  server.listen(port, () => {
    console.log(`Image Embedding Visualizer is running at http://localhost:${port}/app/`);
  });
}

listen(preferredPort);
