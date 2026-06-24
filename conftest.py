import os
import sys

# Ensure the repo root is importable so `from src...` works under pytest,
# matching how scripts are run from the project root.
sys.path.insert(0, os.path.dirname(__file__))
