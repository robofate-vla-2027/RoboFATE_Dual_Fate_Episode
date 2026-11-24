"""Original finite owned runsc supervisor; runtime supplied by the entrypoint."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

def write_once(path,value):
    with Path(path).open('x') as f:
        f.write(value);f.flush();os.fsync(f.fileno())


class ParentStopped(Exception):
    def __init__(self,signum):self.signum=signum


def supervise(run,expected_sha,parent_pid):
    run=Path(run)
    gs.host_guard()
    if run.resolve(strict=True)!=run or run.parent!=gs.sessions:
        raise ValueError('canonical owned run on exact host required')
    config=run/'oci/config.json'
    if hashlib.sha256(config.read_bytes()).hexdigest()!=expected_sha:
        raise ValueError('OCI config changed after staging')
    state=(run/'oci/state').stat()
    if any((run/'oci/state').iterdir()):raise ValueError('fresh own state directory required')
    def stopped(signum,frame):raise ParentStopped(signum)
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG')
    if os.getppid()!=parent_pid:raise ValueError('original launcher no longer parent')
    child=None;primary=None;failure=None;cleanup=None;listing=None
    start=time.monotonic()
    try:
        child=subprocess.Popen(gs.prefix(run)+['run','--bundle='+str(run/'oci'),gs.ID],stdin=subprocess.DEVNULL)
        write_once(run/'oci/OWNED_RUNSC.json',json.dumps(dict(pid=child.pid,proc_stat=Path(f'/proc/{child.pid}/stat').read_text(),
            supervisor_pid=os.getpid(),supervisor_stat=Path('/proc/self/stat').read_text(),parent_pid=parent_pid,
            state_device=state.st_dev,state_inode=state.st_ino,container_id=gs.ID,config_sha256=expected_sha,qualification=False)))
        primary=child.wait()
    except ParentStopped as e:
        failure={'type':'parent_or_owned_stop','signal':e.signum}
    except Exception as e:
        failure={'type':type(e).__name__,'message':str(e)}
    finally:
        # Repeated parent signals cannot interrupt the bounded owned cleanup.
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            child.terminate()
            try:primary=child.wait(timeout=4)
            except subprocess.TimeoutExpired:child.kill();primary=child.wait(timeout=4)
        if child is not None:
            try:
                write_once(run/'oci/runsc.exit',str(primary)+'\n')
            except Exception as e:
                failure=failure or {'type':type(e).__name__,'message':str(e),'stage':'raw_exit_receipt'}
        if ((run/'oci/state').stat().st_dev,(run/'oci/state').stat().st_ino)!=(state.st_dev,state.st_ino):raise RuntimeError('state identity changed; refusing cleanup')
        try:
            cleanup=subprocess.run(gs.prefix(run)+['delete','--force',gs.ID],capture_output=True,text=True,timeout=8)
            listing=subprocess.run(gs.prefix(run)+['list','--format=json'],capture_output=True,text=True,timeout=5)
            clean=cleanup.returncode==listing.returncode==0 and json.loads(listing.stdout) in (None,[])
        except Exception as e:
            clean=False;failure=failure or {'type':type(e).__name__,'message':str(e)}
        result=dict(schema='refate.teacher.gvisor_lifecycle.v1',qualification=False,run_root=str(run),config_sha256=expected_sha,supervisor_pid=os.getpid(),primary_raw_exit=primary,error=failure,
            cleanup_raw_exit=None if cleanup is None else cleanup.returncode,
            cleanup_stderr=None if cleanup is None else cleanup.stderr,
            list_raw_exit=None if listing is None else listing.returncode,
            remaining=None if listing is None else listing.stdout,owned_cleanup_complete=clean,
            elapsed_s=time.monotonic()-start)
        write_once(run/'oci/LIFECYCLE.json',json.dumps(result,indent=2)+'\n')
    if not clean:return 72
    if failure:return 70
    if primary is None:return 70
    return primary if 0<=primary<=255 else 1
