"""
Run full post-labeling pipeline: preprocess → train → evaluate.

Usage:
    python scripts/run_post_labeling_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent


def run(script: str) -> None:
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT, check=True)


def main() -> None:
    run("preprocess_trainval.py")
    run("train_models.py")
    run("evaluation.py")
    print("\nPost-labeling pipeline finished successfully.")


if __name__ == "__main__":
    main()
