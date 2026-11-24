"""Private-netns byte transport, then ORIGINAL Codex command. Parent launch only."""
import json
import selectors
import socket
import socketserver
import subprocess
import sys
import threading


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(self.server.unix_path)
            with selectors.DefaultSelector() as select:
                select.register(self.request, selectors.EVENT_READ, upstream)
                select.register(upstream, selectors.EVENT_READ, self.request)
                while True:
                    events = select.select(timeout=1000)
                    if not events:
                        return
                    for key, _ in events:
                        block = key.fileobj.recv(65536)
                        if not block:
                            return
                        key.data.sendall(block)
        finally:
            upstream.close()


def main():
    servers = []
    try:
        for port, name in ((8080, "model.sock"),):
            server = socketserver.ThreadingTCPServer(("127.0.0.1", port), Relay)
            server.daemon_threads = True
            server.unix_path = "/gateway/" + name
            threading.Thread(target=server.serve_forever, daemon=True).start()
            servers.append(server)
        with open("/command.json") as handle:
            command = json.load(handle)
        return subprocess.call(command, stdin=subprocess.DEVNULL)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    sys.exit(main())
