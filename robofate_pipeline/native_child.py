"""Explicit parent-bound Native entry. Never start this as a standalone campaign."""
import argparse
from pathlib import Path

from .execution import run_teacher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--lease-sha256", required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args(argv)
    run_teacher({"path": str(args.context), "sha256": args.sha256},
                lease_ref={"path": str(args.lease), "sha256": args.lease_sha256}, lock_fd=args.lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
