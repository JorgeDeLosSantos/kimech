import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = [
    "four_bar.py",
    "four_bar_analysis.py",
    "slider_crank.py",
    "slider_crank_analysis.py",
    "slider_crank_analysis_comparison.py",
]


@pytest.mark.parametrize("example_name", _EXAMPLES)
def test_public_example_runs(example_name):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(_REPOSITORY_ROOT / "examples" / example_name)],
        cwd=_REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"{example_name} failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
