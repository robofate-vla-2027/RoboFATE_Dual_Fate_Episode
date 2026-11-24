"""Single-writer fence; callers own tasks, never peer processes."""
import contextlib
import copy
import os
from pathlib import Path
import threading
from .common import need


class SoleOwner:
    def __init__(self,task,recorder,identity):
        self.task,self.recorder,self.identity=task,recorder,copy.deepcopy(identity)
        self.lock=threading.Lock();self.thread=threading.get_ident();self.held=False;self.inflight=0;self.failed=False
        self.original=task.control_step;self.wrapper=self.control_step;task.control_step=self.wrapper
    def check(self):
        need(not self.failed and threading.get_ident()==self.thread and self.task.control_step is self.wrapper,'Control owner changed/failed')
        self.recorder.writer.check()
    @contextlib.contextmanager
    def guard(self):
        self.check();need(not self.held and self.lock.acquire(blocking=False),'Competing Native control owner')
        self.held=True
        try:yield;self.check();need(self.inflight==0,'Control still inflight')
        except BaseException:self.failed=True;raise
        finally:self.held=False;self.lock.release()
    def control_step(self,action):
        self.check();need(self.held and self.inflight==0,'Control outside sole-owner section')
        self.inflight=1
        try:result=self.original(action)
        except BaseException:self.failed=True;raise
        else:self.inflight=0;return result
    def snapshot(self):
        self.check();need(self.held and self.inflight==0,'Snapshot outside idle owner lock')
        raw=Path('/proc/self/stat').read_text();ticks=int(raw.rsplit(')',1)[1].split()[19])
        return dict(self.identity,dm_submissions_disabled=True,dm_inflight=0,dm_queue_discarded=True,
                    pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),start_ticks=ticks)
