import os
import sys
from pathlib import Path

# Run tests from the project folder so `sample_data/` and the package resolve.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
