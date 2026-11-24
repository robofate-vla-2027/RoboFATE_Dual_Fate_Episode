"""Opt-in metadata timing for a fresh task-private egress module; NOT deployed.

No timeout, route, request, retry, sandbox or status changes. No content logging.
"""
from contextlib import contextmanager
import json
import threading
import time
from types import SimpleNamespace


class JsonlSink:
    def __init__(self, path):
        self.stream = open(path, 'x', buffering=1)
        self.lock = threading.Lock()

    def __call__(self, event):
        with self.lock:
            self.stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
            self.stream.flush()

    def close(self):
        with self.lock:
            self.stream.close()


class Timing:
    def __init__(self, sink, clock=time.monotonic_ns, wall_clock=time.time_ns):
        self.sink, self.clock, self.wall_clock = sink, clock, wall_clock
        self.local = threading.local()
        self.lock = threading.Lock()
        self.next_id = 0

    def emit(self, kind, **fields):
        self.sink({'schema': 'refate.private_proxy_timing.v1', 'event': kind,
                   'monotonic_ns': self.clock(), **fields})

    @contextmanager
    def request(self, channel):
        if getattr(self.local, 'current', None) is not None:
            raise RuntimeError('nested handler instrumentation is unsupported')
        if channel not in ('model', 'robo'):
            raise ValueError('explicit model/robo channel required')
        with self.lock:
            request_id = self.next_id
            self.next_id += 1
        record = dict(request_id=request_id, channel=channel, phase_ns={}, phase_calls={},
                      response_bytes_read=0, response_status=None, first_body_seen=False,
                      error_type=None, upstream_eof=False, returned_normally=False,
                      phase_errors={}, downstream_wire_bytes=0)
        self.local.current = record
        try:
            self.emit('request_begin', request_id=request_id, channel=channel,
                      unix_ns=self.wall_clock())
            try:
                yield record
            except BaseException as primary:
                record['error_type'] = type(primary).__name__
                try:
                    self.emit('request_end', **record)
                except Exception as failure:
                    if hasattr(primary, 'add_note'):
                        primary.add_note('Timing receipt failed: ' + type(failure).__name__)
                raise
            else:
                record['returned_normally'] = True
                self.emit('request_end', **record)
        finally:
            self.local.current = None

    def timed(self, phase, call, *args, **kwargs):
        record = getattr(self.local, 'current', None)
        if record is None:
            return call(*args, **kwargs)
        start = self.clock()
        try:
            return call(*args, **kwargs)
        except BaseException as failure:
            record['phase_errors'][phase] = type(failure).__name__
            raise
        finally:
            elapsed = self.clock() - start
            record['phase_ns'][phase] = record['phase_ns'].get(phase, 0) + elapsed
            record['phase_calls'][phase] = record['phase_calls'].get(phase, 0) + 1


def prepare_private_module(module, sink, clock=time.monotonic_ns, wall_clock=time.time_ns):
    """Call once on a freshly imported module before it serves any requests."""
    if getattr(module, '_refate_timing_prepared', False):
        raise ValueError('instrument once, before serving any requests')
    original_connection = module.http.client.HTTPConnection
    original_proxy = module.Proxy
    timing = Timing(sink, clock, wall_clock)

    class ResponseView:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def read1(self, size):
            chunk = timing.timed('upstream_body_read_wait', self.original.read1, size)
            record = timing.local.current
            record['response_bytes_read'] += len(chunk)
            if chunk and not record['first_body_seen']:
                record['first_body_seen'] = True
                timing.emit('first_body', request_id=record['request_id'], channel=record['channel'])
            if not chunk:
                record['upstream_eof'] = True
            return chunk

    class Connection(original_connection):
        def connect(self):
            return timing.timed('connect_including_name_resolution', super().connect)

        def request(self, *args, **kwargs):
            return timing.timed('request_send_including_connect', super().request, *args, **kwargs)

        def getresponse(self):
            response = timing.timed('response_header_wait', super().getresponse)
            record = timing.local.current
            record['response_status'] = response.status
            timing.emit('response_headers', request_id=record['request_id'], channel=record['channel'],
                        status=response.status)
            return ResponseView(response)

    class WriterView:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            return getattr(self.original, name)

        def write(self, data):
            result = timing.timed('downstream_write', self.original.write, data)
            timing.local.current['downstream_wire_bytes'] += len(data)
            return result

        def flush(self):
            return timing.timed('downstream_flush', self.original.flush)

    # Replace only the private module binding, never process-global http.client.
    module.http = SimpleNamespace(client=SimpleNamespace(HTTPConnection=Connection))
    module._refate_timing_prepared = True

    def proxy_for(channel):
        if channel not in ('model', 'robo'):
            raise ValueError('explicit model/robo channel required')

        class TimedProxy(original_proxy):
            def read_body(self):
                return timing.timed('client_body_read', super().read_body)

            def forward(self):
                with timing.request(channel):
                    writer = self.wfile
                    self.wfile = WriterView(writer)
                    try:
                        return super().forward()
                    finally:
                        self.wfile = writer

            do_GET = do_POST = do_PUT = do_DELETE = forward

        return TimedProxy

    return proxy_for, timing
