import sys
from pathlib import Path

# Ensure tests import the local project package, not similarly named site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
