import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = ROOT / "paper" / "template" / "ACCESS_latex_template_20260513"
ZIP_OUT = ROOT / "paper" / "Sunflower_PDI_IEEE_Access_Overleaf_Package.zip"

print(f"Building Overleaf package from {TEMPLATE_DIR}...")

# Make sure main.tex is synchronized with access.tex
tex_src = TEMPLATE_DIR / "access.tex"
tex_main = TEMPLATE_DIR / "main.tex"
shutil.copy2(tex_src, tex_main)

# Files to exclude (auxiliary, logs, backup pdfs, equations not needed)
EXCLUDE_EXTS = {".aux", ".log", ".out", ".synctex.gz", ".DS_Store"}
EXCLUDE_FILES = {"access.pdf", "access.aux", "access.log"}

if ZIP_OUT.exists():
    ZIP_OUT.unlink()

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in TEMPLATE_DIR.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix in EXCLUDE_EXTS or path.name in EXCLUDE_FILES:
            continue
        # rel path inside template dir
        rel_path = path.relative_to(TEMPLATE_DIR)
        zf.write(path, arcname=str(rel_path))

print(f"Successfully generated Overleaf zip archive: {ZIP_OUT}")
print(f"Archive size: {ZIP_OUT.stat().st_size / (1024 * 1024):.2f} MB")
