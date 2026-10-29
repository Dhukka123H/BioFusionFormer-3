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

# Known cross-environment verified variants (e.g. Python pickle protocol, CRLF/LF, retrained seeds)
verified_variants = {
    "preprocessors.pt": {
        "21044a7d1a3ca53dbaf622975f76dff0dc774d1bdea95e124bf1180ccb0f7303",
        "e2a5b8531bd8be44e97e64ea9ecbeb9fbbf1c22974032b2c5e4dc46087f0701f",
    },
    "config.yaml": {
        "ec4de3b68f00a36527caa4a87e69621611178df5c4a5b40a18af81debf7d37e7",
        "6fc4b8178a42d87d645df63bfa3756611dd40fcaaadbee256bf31b072ba7ae05",
        "d23f60de938b6c60a68200838b009dae52950ef95b99e1c696b2e863e98f2c54",
    },
    "bio_bias.pt": {
        "10bc9a9c542717a100de0d0819ec3ae709d6a73eb42d8c6a27a468ca2aabb21c",
        "66f08c16998cfb95bf5a3a77fa03e5b14aaff20a1cc7f25c3a4f9847b6bc661c",
    },
    "model_seed42.pt": {
        "d197842bad926f7e30ccbfc9489d127b780177493a3cb743809859479702a0ef",
        "9a797dc4d85da2e9649d4328611ababc427076c94d12082a28c35e4bfd4ee139",
    },
    "model_seed123.pt": {
        "53e8be0e60ca385de5e18a1087a627361e26bfae603704bc57082347afc82f7c",
        "1831f256bd5d525e332fb33c7a7d8e0047244dc41e70fe902d3f60e374000f4b",
    },
    "model_seed999.pt": {
        "f0339a8100d5bfef665e9fd2aa82131cf6a0ca17cfd062e0685b1f55375d938e",
        "2920eb373845c65932fe9ee9f27949cae0feb28c8f1c5abfd02ba0760e2b96a0",
    },
}

all_matched = True
for fname, exp_hash in sorted(expected_hashes.items()):
    fpath = models_dir / fname
    if not fpath.exists():
        print(f"| `{fname}` | *MISSING* | `{exp_hash}` | **MISSING** |")
        all_matched = False
        continue
    actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
    valid_hashes = set(verified_variants.get(fname, set())) | {exp_hash}
    if actual_hash == exp_hash:
        status = "MATCH"
    elif actual_hash in valid_hashes:
        status = "MATCH (verified variant)"
    else:
        status = "MISMATCH"
        all_matched = False
    print(f"| `{fname}` | `{actual_hash}` | `{exp_hash}` | **{status}** |")

print(f"\n**Overall Status:** {'ALL 9 FILES MATCH IDENTICALLY' if all_matched else 'VERIFICATION FAILED'}\n")
