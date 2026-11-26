"""Server HTTP lokal ringan (Range-capable) untuk streaming video overlay dashboard."""

from __future__ import annotations

import hashlib
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CHUNK_SIZE = 256 * 1024
RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")


class LocalVideoServer:
    """Sajikan file MP4 lokal dengan dukungan HTTP 206 Partial Content untuk seeking mulus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paths_by_token: dict[str, Path] = {}
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._build_handler())
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="dashboard-local-video-server",
            daemon=True,
        )
        self._thread.start()

    def register(self, video_path: Path) -> str:
        resolved = video_path.resolve()
        token = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:24]
        with self._lock:
            self._paths_by_token[token] = resolved
        return f"http://127.0.0.1:{self.port}/video/{token}.mp4"

    def resolve(self, token: str) -> Path | None:
        with self._lock:
            return self._paths_by_token.get(token)

    def _build_handler(self):
        parent = self

        class VideoRangeHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

            def _resolve_requested_file(self) -> Path | None:
                clean_path = self.path.split("?", 1)[0]
                if not clean_path.startswith("/video/") or not clean_path.endswith(".mp4"):
                    return None
                token = clean_path[len("/video/") : -len(".mp4")]
                target = parent.resolve(token)
                if target is None or not target.is_file():
                    return None
                return target

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Range")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_HEAD(self) -> None:  # noqa: N802
                self._serve(send_body=False)

            def do_GET(self) -> None:  # noqa: N802
                self._serve(send_body=True)

            def _serve(self, *, send_body: bool) -> None:
                target = self._resolve_requested_file()
                if target is None:
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                file_size = target.stat().st_size
                range_header = self.headers.get("Range")
                start = 0
                end = file_size - 1
                status = HTTPStatus.OK

                if range_header:
                    match = RANGE_PATTERN.match(range_header.strip())
                    if match is None:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    start_text, end_text = match.groups()
                    if start_text == "" and end_text == "":
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if start_text == "":
                        suffix_length = int(end_text)
                        start = max(file_size - suffix_length, 0)
                        end = file_size - 1
                    else:
                        start = int(start_text)
                        end = int(end_text) if end_text else file_size - 1
                    if start >= file_size or end < start:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{file_size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    end = min(end, file_size - 1)
                    status = HTTPStatus.PARTIAL_CONTENT

                content_length = end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                if status == HTTPStatus.PARTIAL_CONTENT:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Content-Length", str(content_length))
                self.end_headers()

                if not send_body:
                    return
                try:
                    with target.open("rb") as video_file:
                        video_file.seek(start)
                        remaining = content_length
                        while remaining > 0:
                            chunk = video_file.read(min(CHUNK_SIZE, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

        return VideoRangeHandler
