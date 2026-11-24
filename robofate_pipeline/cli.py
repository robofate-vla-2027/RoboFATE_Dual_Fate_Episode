"""Offline checks and explicit-plan data tools; no implicit collection launch."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Check every distributed path and byte")
    audit.add_argument("--repo", type=Path, default=Path.cwd())
    audit.add_argument("--denylist", type=Path, help="Private literal list kept outside the project")
    check = commands.add_parser("doctor", help="Read-only deployment checks; never grants launch")
    check.add_argument("--config", required=True, type=Path)
    settle = commands.add_parser("settle", help="One-shot saved-data readback; no simulator or model launch")
    settle.add_argument("--context", required=True, type=Path)
    settle.add_argument("--context-sha256", required=True)
    settle.add_argument("--policy", required=True, type=Path)
    settle.add_argument("--policy-sha256", required=True)
    campaign = commands.add_parser("campaign", help="Explicit campaign metadata or foreground execution")
    campaign.add_argument("action", choices=("initialize", "status", "run", "reconcile", "confirm-cleanup"))
    campaign.add_argument("--policy", required=True, type=Path)
    campaign.add_argument("--policy-sha256", required=True)
    campaign.add_argument("--plan", type=Path)
    campaign.add_argument("--plan-sha256")
    campaign.add_argument("--operation-id")
    campaign.add_argument("--context", type=Path)
    campaign.add_argument("--context-sha256")
    for name in ("inventory", "export"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "audit":
            from .privacy import audit_tree
            deny = json.loads(args.denylist.read_text()) if args.denylist else []
            result = audit_tree(args.repo, denylist=deny)
            status = 0 if result["passed"] else 1
        elif args.command == "doctor":
            from .deployment import inspect
            result = inspect(json.loads(args.config.read_text()))
            status = 0 if result["static_checks_passed"] else 2
        elif args.command == "campaign":
            from .campaign.runner import Campaign
            from .evidence import need
            instance = Campaign({"path": str(args.policy), "sha256": args.policy_sha256})
            if args.action == "initialize":
                result = instance.ledger.initialize()
            elif args.action == "status":
                value, ref = instance.ledger.store.snapshot()
                result = dict(revision=value["revision"], state_ref=ref,
                    positive=value["qualified_count"], negative=value["negative_count"],
                    nodes=value["nodes"])
            elif args.action == "reconcile":
                need(args.operation_id is not None, "Name the exact uncertain metadata operation")
                result = instance.ledger.store.reconcile(args.operation_id)
            elif args.action == "confirm-cleanup":
                need(args.context is not None and args.context_sha256 is not None, "Pinned settled context required")
                result = instance.confirm_cleanup({"path": str(args.context), "sha256": args.context_sha256})
            else:
                need(args.plan is not None and args.plan_sha256 is not None, "Pinned finite execution plan required")
                result = instance.run_plan({"path": str(args.plan), "sha256": args.plan_sha256})
            status = 0
        elif args.command == "inventory":
            from inventory_episodes import main as inventory
            result = inventory(json.loads(args.plan.read_text()))
            status = 1 if result["errors"] else 0
        elif args.command == "settle":
            from .readback import settle as readback
            result = readback({"path": str(args.context), "sha256": args.context_sha256},
                              {"path": str(args.policy), "sha256": args.policy_sha256})
            status = 0 if result["qualified_positive"] else 1
        else:
            from export_training import main as export
            export(json.loads(args.plan.read_text()))
            return 0
        print(json.dumps(result, indent=2))
        return status
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
