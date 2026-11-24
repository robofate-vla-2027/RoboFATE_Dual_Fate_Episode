"""Original Linux process identity, independent of deployment bindings."""
from pathlib import Path

def identity(pid):
    raw=Path(f'/proc/{pid}/stat').read_text();tail=raw.rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(tail[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'process_group':int(tail[2])}
