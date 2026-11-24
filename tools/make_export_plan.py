"""Build a portable exporter plan from a successful recording inventory."""
import argparse
import json
from pathlib import Path


def make_plan(inventory, output):
    if inventory.get("errors") != 0 or not inventory.get("episodes"):
        raise ValueError("A successful, nonempty inventory is required")
    if not Path(output).is_absolute():
        raise ValueError("Export directory must be absolute")
    required = {"episode_key", "source_id", "task", "t1_native", "native_total_budget",
                "source_manifest", "source_manifest_sha256", "inventory_path", "branches"}
    for episode in inventory["episodes"]:
        if required - set(episode) or episode.get("errors"):
            raise ValueError("Inventory episode is incomplete")
        if set(episode["branches"]) != {"dm_failure", "teacher_success"}:
            raise ValueError("Both original baseline and teacher recording paths are required")
    return {"expected_hostname": inventory["host"], "output": str(output),
            "episodes": inventory["episodes"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = make_plan(json.loads(args.inventory.read_text()), args.output_directory)
    with args.plan.open("x") as output:
        json.dump(plan, output, indent=2)
        output.write("\n")
