"""Original runsc flags with external executable binding."""
from pathlib import Path

def prefix(run):
    return [str(RUNSC),'--root='+str(Path(run)/'oci/state'),'--ignore-cgroups','--network=none',
            '--platform=systrap','--directfs=false','--host-settings=ignore','--file-access=shared',
            '--overlay2=none','--host-uds=open']
