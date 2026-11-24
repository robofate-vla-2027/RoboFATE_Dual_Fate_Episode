"""Original streaming byte transport, used only by the model-only Unix listener."""
import http.client
import sys
from http.server import BaseHTTPRequestHandler
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "content-length"}

def log(message):
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


class Proxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream = None  # urlsplit result
    allowed_prefixes = ()
    key_file = None
    timeout_s = 900

    def log_message(self, *_):
        pass

    def read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline().strip():
                        pass
                    return body
                body += self.rfile.read(size)
                self.rfile.readline()
        length = int(self.headers.get("Content-Length", 0) or 0)
        return self.rfile.read(length) if length else b""

    def refuse(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def forward(self):
        if not self.path.startswith(self.allowed_prefixes):
            log(f"DENY {self.command} {self.path.split('?')[0]}")
            return self.refuse(403, "path not allowed\n")
        body = self.read_body()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        if self.key_file:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("authorization", "api-key", "x-api-key")}
            with open(self.key_file) as handle:
                headers["Authorization"] = "Bearer " + handle.read().strip()
        headers["Host"] = self.upstream.netloc
        connection = http.client.HTTPConnection(self.upstream.hostname, self.upstream.port or 80, timeout=self.timeout_s)
        try:
            connection.request(self.command, self.path, body=body or None, headers=headers)
            response = connection.getresponse()
        except OSError as err:
            log(f"FAIL {self.command} {self.path.split('?')[0]}: {err}")
            return self.refuse(502, f"upstream error: {err}\n")
        self.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            if key.lower() not in HOP_BY_HOP:
                self.send_header(key, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        sent = 0
        try:
            while True:
                chunk = response.read1(65536)
                if not chunk:
                    break
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                sent += len(chunk)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError as err:
            log(f"ABORT {self.path.split('?')[0]}: {err}")
            self.close_connection = True
        finally:
            connection.close()
        log(f"{response.status} {self.command} {self.path.split('?')[0]} in={len(body)} out={sent}")

    do_GET = do_POST = do_PUT = do_DELETE = forward
