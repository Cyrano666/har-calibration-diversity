"""Convenient entry point for the unchanged locked acquisition replay."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
if not (root/'revision6/results/predictions/uschad_0_0_MR.npz').exists():
    raise SystemExit('Run python get_artifacts.py first to install the frozen numerical artifacts.')
raise SystemExit(subprocess.call([sys.executable,str(root/'revision6/reproduce.py'),*sys.argv[1:]],cwd=root))
