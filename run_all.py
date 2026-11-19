"""Run the full pipeline: generate panel -> analyse."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for script in ["generate_panel.py", "analysis.py"]:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True, cwd=ROOT / "src")
