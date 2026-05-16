import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "hand_landmarks_data.csv"
sys.path.insert(0, str(ROOT / "src"))
