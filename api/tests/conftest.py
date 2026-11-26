"""Make `app` importable and point the API to the model in the repository."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "api"))
os.environ.setdefault("MODEL_PATH", str(ROOT / "model" / "artifacts" / "model-1.0.0.joblib"))
