from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from scene_store import SceneStore


WEB_DIR = Path(__file__).resolve().parent / "web"
STORE = SceneStore()


class GraphworldHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/scenes":
            self._json({"scenes": STORE.list_scenes()})
            return
        if path.startswith("/api/scene/"):
            scene_id = unquote(path.removeprefix("/api/scene/")).strip("/")
            payload = STORE.get_scene(scene_id)
            if payload is None:
                self._json({"error": f"Scene not found: {scene_id}"}, HTTPStatus.NOT_FOUND)
                return
            self._json(payload)
            return
        if path in {"", "/"}:
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format: str, *args) -> None:
        print(f"[Graphworld] {self.address_string()} - {format % args}")

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), GraphworldHandler)
    print(f"Graphworld running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Graphworld server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
