"""Parent-lease-only entry for baseline Native or frozen-DM policy."""
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("native", "policy"))
    for name in ("context", "sha256", "lease", "lease-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args(argv)
    from . import runtime, service
    function = runtime.run if args.role == "native" else service.run
    function({"path": args.context, "sha256": args.sha256},
             lease_ref={"path": args.lease, "sha256": args.lease_sha256}, lock_fd=args.lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
