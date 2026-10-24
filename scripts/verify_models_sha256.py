import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
models_dir = ROOT / "models" / "final"
manifest_path = models_dir / "manifest.json"

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_hashes = manifest["file_hashes_sha256"]

print("\n### Model Weights SHA-256 Verification Table\n")
print("| File | Actual SHA-256 | Manifest SHA-256 | Status |")
print("| :--- | :--- | :--- | :---: |")

all_matched = True
for fname, exp_hash in sorted(expected_hashes.items()):
    fpath = models_dir / fname
    if not fpath.exists():
        print(f"| `{fname}` | *MISSING* | `{exp_hash}` | **MISSING** |")
        all_matched = False
        continue
    actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
    status = "MATCH" if actual_hash == exp_hash else "MISMATCH"
    if status != "MATCH":
        all_matched = False
    print(f"| `{fname}` | `{actual_hash}` | `{exp_hash}` | **{status}** |")

print(f"\n**Overall Status:** {'ALL 9 FILES MATCH IDENTICALLY' if all_matched else 'VERIFICATION FAILED'}\n")
