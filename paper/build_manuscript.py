"""
Build manuscript entry point.
Invokes build_ieee_access_paper.py to generate Sunflower_PDI_IEEE_Access_Manuscript.docx
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
script = ROOT / "build_ieee_access_paper.py"
ret = subprocess.run([sys.executable, str(script)])
sys.exit(ret.returncode)
