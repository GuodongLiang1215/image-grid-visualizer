const http = require("http");
const fs = require("fs");
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

  let target = path.normalize(path.join(root, pathname));
  if (!target.startsWith(root)) {
    return null;
  }

  if (fs.existsSync(target) && fs.statSync(target).isDirectory()) {
    target = path.join(target, "index.html");
  }

  return target;
}

const server = http.createServer((req, res) => {
  const target = resolveRequestPath(req.url);
  if (!target) {
    send(res, 403, "Forbidden");
    return;
  }

  fs.readFile(target, (error, data) => {
    if (error) {
      send(res, error.code === "ENOENT" ? 404 : 500, error.code === "ENOENT" ? "Not found" : "Server error");
      return;
    }

    const ext = path.extname(target).toLowerCase();
    send(res, 200, data, mimeTypes[ext] || "application/octet-stream");
  });
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
