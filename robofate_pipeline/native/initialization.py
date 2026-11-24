"""Original primary-exception diagnostics; no retry or qualification."""
import ast
import json
import os
from pathlib import Path
import sys
import time
import traceback

def wrap_class(cls, run, *, evidence_kind="actual_initialization_exception"):
    original = cls._create_native
    path = Path(run) / "initialization_failure.json"

    def create(self):
        try:
            return original(self)
        except BaseException as error:
            value = {
                "data_kind": evidence_kind, "wall_ns": time.time_ns(),
                "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "pid": os.getpid(),
                "scientific_qualification": False, "original_exception_reraised": True,
            }
            try:
                with path.open("x") as stream:
                    json.dump(value, stream, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException as recording_error:
                print("Initialization exception evidence write failed: "
                      + repr(recording_error), file=sys.stderr, flush=True)
            raise

    cls._create_native = create
    return cls
