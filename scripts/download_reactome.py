"""
scripts/download_reactome.py
----------------------------
Idempotent, resumable download of Reactome flat-files for BioFusionFormer.

Files downloaded from https://reactome.org/download/current/
Saves to   data/raw/reactome/
Checksums  data/raw/CHECKSUMS.txt
Log        logs/download_reactome.log
"""

import hashlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "raw" / "reactome"
CHECKSUM_FILE = ROOT / "data" / "raw" / "CHECKSUMS.txt"
LOG_FILE = ROOT / "logs" / "download_reactome.log"

DEST.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("download_reactome")

# ── Target files ──────────────────────────────────────────────────────────────
BASE_URL = "https://reactome.org/download/current"

FILES = [
    "Ensembl2Reactome.txt",
    "Ensembl2Reactome_All_Levels.txt",
    "miRBase2Reactome.txt",
    "miRBase2Reactome_All_Levels.txt",
    "ReactomePathways.txt",
    "ReactomePathwaysRelation.txt",
    "humanPathwaysWithDiagrams.txt",
    "ReactomePathways.gmt.zip",
    "gene_association.reactome.gz",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, retries: int = 5, timeout: int = 60) -> Optional[Path]:
    """
    Stream-download *url* to *dest*, resuming if a partial file exists.
    Returns the destination Path on success, None on failure.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            headers = {}
            existing_size = dest.stat().st_size if dest.exists() else 0
            if existing_size:
                headers["Range"] = f"bytes={existing_size}-"

            resp = requests.get(url, headers=headers, stream=True, timeout=timeout)

            # 416 = server says no more bytes → file already complete
            if resp.status_code == 416:
                log.info("  Already complete (416 Range Not Satisfiable): %s", dest.name)
                return dest

            resp.raise_for_status()

            content_length = int(resp.headers.get("Content-Length", 0))
            mode = "ab" if existing_size and resp.status_code == 206 else "wb"
            if mode == "wb":
                existing_size = 0  # restart

            total = existing_size + content_length if content_length else None

            with open(dest, mode) as fh, tqdm(
                total=total,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                ncols=80,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))

            log.info("  Downloaded: %s  (%.1f KB)", dest.name, dest.stat().st_size / 1024)
            return dest

        except Exception as exc:
            log.warning("  Attempt %d/%d failed for %s: %s", attempt, retries, dest.name, exc)
            if attempt < retries:
                sleep_s = 2 ** attempt
                log.info("  Retrying in %d s …", sleep_s)
                time.sleep(sleep_s)

    log.error("  FAILED after %d attempts: %s", retries, url)
    return None


def update_checksums(path: Path) -> str:
    """Compute SHA-256 for *path* and upsert the CHECKSUMS.txt entry."""
    digest = sha256_file(path)
    entry = f"sha256:{digest}  {path.name}\n"

    existing: dict[str, str] = {}
    if CHECKSUM_FILE.exists():
        for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                parts = line.split(None, 1)
                if len(parts) == 2:
                    existing[parts[1].strip()] = parts[0].strip()

    existing[path.name] = f"sha256:{digest}"

    with open(CHECKSUM_FILE, "w", encoding="utf-8") as fh:
        for name, chk in sorted(existing.items()):
            fh.write(f"{chk}  {name}\n")

    return digest


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("Reactome download started")
    log.info("Destination: %s", DEST)
    log.info("=" * 60)

    successes, skips, failures = [], [], []

    for filename in FILES:
        dest = DEST / filename
        url = f"{BASE_URL}/{filename}"

        if dest.exists() and dest.stat().st_size > 0:
            log.info("[SKIP] %s already present (%d bytes)", filename, dest.stat().st_size)
            skips.append(filename)
            # still update checksum in case file changed on disk
            update_checksums(dest)
            continue

        log.info("[GET] %s", url)
        result = download_file(url, dest)
        if result:
            digest = update_checksums(dest)
            log.info("  SHA-256: %s", digest)
            successes.append(filename)
        else:
            failures.append(filename)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("  Downloaded : %d", len(successes))
    log.info("  Skipped    : %d (already present)", len(skips))
    log.info("  Failed     : %d", len(failures))
    if failures:
        log.error("  FAILURES   : %s", ", ".join(failures))
    log.info("Checksums saved to: %s", CHECKSUM_FILE)
    log.info("=" * 60)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
