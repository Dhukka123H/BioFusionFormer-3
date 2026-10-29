import datetime
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
models_dir = ROOT / "models" / "final"
out_dir = ROOT.parent

zip_path = out_dir / "models_final.zip"
sha_path = out_dir / "models_final_sha256.txt"
root_sha_path = ROOT / "models_final_sha256.txt"

if zip_path.exists():
    zip_path.unlink()
    print(f"Removed previous {zip_path}")

print(f"Building {zip_path} from {models_dir}...")

manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
expected_hashes = manifest["file_hashes_sha256"]

sha_lines = []
sha_lines.append("# BioFusionFormer E1 Final Models SHA-256 Hashes")
sha_lines.append(f"# Generated: {datetime.date.today().isoformat()}")
sha_lines.append("")

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for fpath in sorted(models_dir.iterdir()):
        if fpath.is_file():
            actual_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            sha_lines.append(f"{actual_sha}  models/final/{fpath.name}")
            arcname = f"models/final/{fpath.name}"
            zf.write(fpath, arcname=arcname)
            print(f"  Added {arcname} ({fpath.stat().st_size:,} bytes) [SHA256: {actual_sha[:12]}...]")

sha_content = "\n".join(sha_lines) + "\n"
sha_path.write_text(sha_content, encoding="utf-8")
root_sha_path.write_text(sha_content, encoding="utf-8")
print(f"Saved SHA-256 hash list to {sha_path} and {root_sha_path}")

with zipfile.ZipFile(zip_path, "a") as zf:
    zf.writestr("models_final_sha256.txt", sha_content)

print(f"Successfully created {zip_path} ({zip_path.stat().st_size:,} bytes)")

