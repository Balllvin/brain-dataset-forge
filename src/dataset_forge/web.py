from __future__ import annotations

import functools
import http.server
import socketserver
from pathlib import Path


class QuietReportHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


class ReusableTcpServer(socketserver.TCPServer):
    allow_reuse_address = True


def serve_report(run_dir: Path, *, host: str, port: int) -> None:
    report_dir = run_dir / "report"
    index = report_dir / "index.html"
    if not index.exists():
        raise FileNotFoundError(f"Report not found: {index}")
    handler = functools.partial(QuietReportHandler, directory=str(report_dir))
    with ReusableTcpServer((host, port), handler) as server:
        print(f"Serving {index} at http://{host}:{port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Server stopped")
