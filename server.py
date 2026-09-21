"""Local web server for STEMS.

Requires the Demucs source-separation model: `python -m pip install -r requirements.txt`.
"""
from __future__ import annotations

from email import policy
from email.parser import BytesParser
import json
import subprocess
import sys
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).parent.resolve()
UPLOADS = ROOT / ".stems-uploads"
OUTPUTS = ROOT / "outputs"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


class StemsHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self) -> None:
        if self.path != "/api/separate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.separate()

    def separate(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        if not content_length:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Ingen ljudfil togs emot."})
            return
        if content_length > MAX_UPLOAD_BYTES:
            self.respond_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Filen måste vara mindre än 500 MB."})
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Förväntade en multipart-uppladdning."})
            return
        raw_request = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + self.rfile.read(content_length)
        message = BytesParser(policy=policy.default).parsebytes(raw_request)
        uploaded = next((part for part in message.iter_attachments() if part.get_param("name", header="content-disposition") == "file"), None)
        if uploaded is None or not uploaded.get_filename():
            self.respond_json(HTTPStatus.BAD_REQUEST, {"error": "Ingen ljudfil togs emot."})
            return
        original_name = Path(uploaded.get_filename()).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            self.respond_json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Formatet stöds inte. Välj MP3, WAV, FLAC, M4A eller OGG."})
            return
        job_id = uuid.uuid4().hex
        upload_dir = UPLOADS / job_id
        output_dir = OUTPUTS / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        input_path = upload_dir / f"input{suffix}"
        with input_path.open("wb") as destination:
            destination.write(uploaded.get_payload(decode=True))
        command = [sys.executable, "-m", "demucs.separate", "--name", "htdemucs", "--out", str(output_dir), str(input_path)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
        except FileNotFoundError:
            self.respond_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Demucs är inte installerat. Kör: python -m pip install -r requirements.txt"})
            return
        except subprocess.TimeoutExpired:
            self.respond_json(HTTPStatus.GATEWAY_TIMEOUT, {"error": "Bearbetningen tog för lång tid."})
            return
        except subprocess.CalledProcessError as error:
            self.respond_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Demucs kunde inte separera filen.", "details": error.stderr[-500:]})
            return
        stem_dir = output_dir / "htdemucs" / input_path.stem
        stems = []
        for name in ("vocals", "drums", "bass", "other"):
            match = next(iter(stem_dir.glob(f"{name}.*")), None)
            if match:
                stems.append({"name": name, "url": f"/outputs/{job_id}/htdemucs/{input_path.stem}/{match.name}"})
        if len(stems) != 4:
            self.respond_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Demucs skapade inte alla fyra spår."})
            return
        self.respond_json(HTTPStatus.OK, {"stems": stems})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/outputs/"):
            requested = (ROOT / unquote(parsed.path).lstrip("/")).resolve()
            if OUTPUTS not in requested.parents or not requested.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.path = "/" + str(requested.relative_to(ROOT))
        super().do_GET()

    def respond_json(self, status: HTTPStatus, content: dict) -> None:
        payload = json.dumps(content).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    UPLOADS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 4173), StemsHandler)
    print("STEMS körs på http://127.0.0.1:4173")
    server.serve_forever()


if __name__ == "__main__":
    main()
