"""
Run full post-labeling pipeline: preprocess → train → CV → evaluate → active learning export.

Usage:
    python scripts/run_post_labeling_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent


def run(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(SCRIPTS / script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n>>> Running {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    run("preprocess_trainval.py")
    run("train_models.py")
    run("train_human_gold_cv.py")
    run("evaluation.py")
    run("validate_improvements.py")
    run("select_labeling_candidates.py", ["--task", "theme", "--top-n", "50"])
    run("select_labeling_candidates.py", ["--task", "sentiment", "--top-n", "50"])
    print("\nPost-labeling pipeline finished successfully.")


if __name__ == "__main__":
    main()
