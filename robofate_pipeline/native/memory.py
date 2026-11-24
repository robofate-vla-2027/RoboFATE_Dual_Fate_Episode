"""Original host/cgroup memory admission; no reclamation or peer management."""
from pathlib import Path
from ..evidence import need

def memory_headroom(minimum_gib):
    need(minimum_gib == 120, 'Original120GiB admission required')
    mem = dict((line.split(':', 1)[0], int(line.split()[1])*1024)
               for line in Path('/proc/meminfo').read_text().splitlines()
               if line.startswith(('MemAvailable:', 'MemTotal:')))
    limits = [mem['MemAvailable']]
    rows = []
    cgroups = Path('/proc/self/cgroup').read_text().splitlines()
    for line in cgroups:
        _, controllers, rel = line.split(':', 2)
        if not controllers:
            base, usage_name, limit_name = Path('/sys/fs/cgroup'), 'memory.current', 'memory.max'
        elif 'memory' in controllers.split(','):
            base, usage_name, limit_name = Path('/sys/fs/cgroup/memory'), 'memory.usage_in_bytes', 'memory.limit_in_bytes'
        else:
            continue
        current = base/rel.lstrip('/')
        if not (current/usage_name).exists():
            current = base
        while current == base or current.is_relative_to(base):
            if (current/usage_name).is_file() and (current/limit_name).is_file():
                used = int((current/usage_name).read_text())
                raw = (current/limit_name).read_text().strip()
                limit = None if raw == 'max' else int(raw)
                rows.append({'path': str(current), 'usage_bytes': used, 'limit_bytes': limit})
                if limit is not None:
                    limits.append(limit-used)
            if current == base:
                break
            current = current.parent
    need(rows, 'No readable cgroup memory accounting')
    available = min(limits)
    need(available >= minimum_gib*1024**3, 'Insufficient host/cgroup memory headroom')
    return {'available_bytes': available, 'minimum_free_gib': minimum_gib,
            'host': mem, 'cgroups': rows, 'inherited_wait_clocks_reset': False}
