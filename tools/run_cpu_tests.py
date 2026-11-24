"""Only synthetic CPU/file fixtures; prohibit real peers and process launches."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the empty string")


def offline(event, args):
    if event.startswith(("socket.connect", "socket.bind", "subprocess.")) or event in {
            "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.kill", "os.killpg"}:
        raise RuntimeError("Offline synthetic tests forbid: " + event)


sys.addaudithook(offline)
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
suite.addTests(unittest.defaultTestLoader.loadTestsFromName("test_dataset_reader"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
