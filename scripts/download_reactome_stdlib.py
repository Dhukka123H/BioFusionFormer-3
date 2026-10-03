"""
scripts/download_reactome_stdlib.py
------------------------------------
Stdlib-only version of download_reactome.py (no third-party dependencies).
Used when the venv is still being set up.
Delegates to download_reactome.py once requests is available.
"""

import hashlib
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

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
log = logging.getLogger("download_reactome_stdlib")

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def update_checksums(path: Path) -> str:
    digest = sha256_file(path)
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


def download_file(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> bool:
    """Download url to dest with resume support, using urllib only."""
    for attempt in range(1, retries + 1):
        try:
            existing_size = dest.stat().st_size if dest.exists() else 0
            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if status == 416:
                    log.info("  Already complete: %s", dest.name)
                    return True

                content_length = int(resp.headers.get("Content-Length", 0))
                mode = "ab" if (existing_size > 0 and status == 206) else "wb"
                if mode == "wb":
                    existing_size = 0

                downloaded = existing_size
                total = existing_size + content_length if content_length else "?"
                log.info("  Saving %s (total ~%s bytes) …", dest.name, total)

                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        fh.write(chunk)
                        downloaded += len(chunk)

                        # Simple progress print every ~10 MB
                        if downloaded % (10 * 1024 * 1024) < 65536:
                            log.info("    ... %.1f MB", downloaded / 1024 / 1024)

            log.info("  Done: %s  (%.1f KB)", dest.name, dest.stat().st_size / 1024)
            return True

        except HTTPError as exc:
            if exc.code == 416:
                log.info("  Already complete (416): %s", dest.name)
                return True
            log.warning("  Attempt %d/%d HTTP %s for %s", attempt, retries, exc.code, dest.name)
        except URLError as exc:
            log.warning("  Attempt %d/%d URL error for %s: %s", attempt, retries, dest.name, exc)
        except Exception as exc:
            log.warning("  Attempt %d/%d error for %s: %s", attempt, retries, dest.name, exc)

        if attempt < retries:
            sleep_s = 2 ** attempt
            log.info("  Retrying in %d s …", sleep_s)
            time.sleep(sleep_s)

    log.error("  FAILED after %d attempts: %s", retries, url)
    return False


def main() -> None:
    log.info("=" * 60)
    log.info("Reactome download (stdlib) started")
    log.info("Destination: %s", DEST)
    log.info("=" * 60)

    successes, skips, failures = [], [], []

    for filename in FILES:
        dest = DEST / filename
        url = f"{BASE_URL}/{filename}"

        if dest.exists() and dest.stat().st_size > 0:
            log.info("[SKIP] %s already present (%d bytes)", filename, dest.stat().st_size)
            skips.append(filename)
            update_checksums(dest)
            continue

        log.info("[GET] %s", url)
        ok = download_file(url, dest)
        if ok:
            digest = update_checksums(dest)
            log.info("  SHA-256: %s", digest)
            successes.append(filename)
        else:
            failures.append(filename)

    log.info("")
    log.info("=" * 60)
    log.info("DOWNLOAD SUMMARY")
    log.info("  Downloaded : %d", len(successes))
    log.info("  Skipped    : %d  (already present)", len(skips))
    log.info("  Failed     : %d", len(failures))
    if failures:
        log.error("  FAILURES   : %s", ", ".join(failures))
    log.info("Checksums saved to: %s", CHECKSUM_FILE)
    log.info("=" * 60)

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
