"""Build only audited package sources in an external disposable directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robofate_pipeline.privacy import audit_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Wheel output must be outside the source checkout")
    if not audit_tree(ROOT)["passed"]:
        raise ValueError("Source audit failed; build refused")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robofate-build-") as temporary:
        stage = Path(temporary)
        for name in ("pyproject.toml", "dataset_reader.py", "inventory_episodes.py", "export_training.py"):
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / "robofate_pipeline", stage / "robofate_pipeline")
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                        "--no-build-isolation", "--no-index", "--wheel-dir", str(output),
                        str(stage)], check=True)


if __name__ == "__main__":
    main()
