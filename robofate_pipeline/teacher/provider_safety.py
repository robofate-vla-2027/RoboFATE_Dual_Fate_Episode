"""Stop on structured provider safety errors, without changing the original response."""
import json
import threading
import time
from ..evidence import exclusive as save

CODES={'content_filter','content_policy_violation','cyber_policy','cyberPolicy','safety_refusal'}

class ProviderSafetyStop(RuntimeError):
    pass


def refusal(value):
    if not isinstance(value,dict):return None
    event=value.get('type')
    response=value.get('response') if isinstance(value.get('response'),dict) else value
    details=response.get('incomplete_details')
    if event in ('response.incomplete',None) and isinstance(details,dict) and details.get('reason') in CODES:
        return dict(event_type=event,code=details['reason'],incomplete_details=details)
    error=response.get('error')
    if isinstance(error,dict):
        code=error.get('code') or error.get('type')
        if code in CODES:return dict(event_type=event,code=code,error=error)
        inner=error.get('innererror') or error.get('inner_error')
        if isinstance(inner,dict) and inner.get('code') in CODES:
            return dict(event_type=event,code=inner['code'],error=error)
    return None


class Gate:
    def __init__(self,path):
        self.path=path;self.stopped=threading.Event();self.lock=threading.Lock();self.reason=None

    def observe(self,value):
        reason=refusal(value)
        if reason is None:return False
        with self.lock:
            if not self.stopped.is_set():
                self.reason=reason
                self.stopped.set()
                save(self.path,dict(schema='refate.provider_safety_latch.v1',monotonic_ns=time.monotonic_ns(),
                    unix_ns=time.time_ns(),provider_refusal=reason,original_response_not_rewritten=True,
                    subsequent_upstream_requests_forbidden=True,automatic_replay=False))
        return True

    def require_open(self):
        if self.stopped.is_set():raise ProviderSafetyStop('Original provider safety refusal preserved; no further upstream request')


class Parser:
    """Bounded SSE/JSON inspection; retain only structured refusal fields."""
    def __init__(self,gate):self.gate=gate;self.buffer=b'';self.overflow=False
    def feed(self,chunk):
        if self.overflow:return
        self.buffer+=chunk
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1)
            self.line(line)
        if len(self.buffer)>1024*1024:
            self.overflow=True;self.buffer=b''
        if not chunk and self.buffer:
            self.line(self.buffer);self.buffer=b''
    def line(self,line):
        if line.startswith(b'data:'):line=line[5:].strip()
        try:value=json.loads(line)
        except (ValueError,UnicodeError):return
        self.gate.observe(value)


def install(proxy,handlers,timing,gate):
    connection=proxy.http.client.HTTPConnection
    class Response:
        def __init__(self,original):self.original=original;self.parser=Parser(gate)
        def __getattr__(self,name):return getattr(self.original,name)
        def read1(self,size):
            chunk=self.original.read1(size);self.parser.feed(chunk);return chunk
    class Connection(connection):
        def request(self,*args,**kwargs):
            current=getattr(timing.local,'current',None)
            if current and current['channel']=='model':gate.require_open()
            return super().request(*args,**kwargs)
        def getresponse(self):
            response=super().getresponse();current=getattr(timing.local,'current',None)
            return Response(response) if current and current['channel']=='model' else response
    proxy.http.client.HTTPConnection=Connection
    def factory(channel):
        original=handlers(channel)
        if channel!='model':return original
        class Guarded(original):
            def forward(self):
                if gate.stopped.is_set():
                    self.send_error(409,'Local task stopped after provider safety refusal; no upstream request')
                    return
                return super().forward()
            do_GET=do_POST=do_PUT=do_DELETE=forward
        return Guarded
    return factory

