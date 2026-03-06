import sys
from pathlib import Path

Basepath = Path()

# find base path.
if getattr(sys, "frozen", False):
    Basepath = Path(sys.executable).parent
else:
    Basepath = Path(__file__).parent.parent.parent
