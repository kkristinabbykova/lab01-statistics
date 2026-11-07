import sys
from pathlib import Path

# чтобы pytest находил пакет labstat из корня репозитория
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
