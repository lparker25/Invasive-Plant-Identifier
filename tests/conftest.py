import sys
from pathlib import Path

# Add parent directory to path so tests can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))
