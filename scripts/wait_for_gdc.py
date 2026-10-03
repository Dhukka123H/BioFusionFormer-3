"""
Phase B1 - GDC Status Poller
==============================
Polls https://api.gdc.cancer.gov/status every 10 minutes (max 48 h).
Logs every poll to logs/gdc_poll.log.
Exits 0 if status is OK, 1 if 48 h timeout reached.
"""
import sys
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import subprocess

ROOT = Path(__file__).resolve().parent.parent
LOG  = ROOT / "logs" / "gdc_poll.log"
ROOT.joinpath("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("gdc_poll")

GDC_STATUS_URL = "https://api.gdc.cancer.gov/status"
POLL_INTERVAL  = 600   # 10 minutes in seconds
MAX_POLLS      = 288   # 48 hours / 10 min

UA = "Mozilla/5.0 (compatible; BioFusionFormer/1.0; academic)"


def check_status() -> dict | None:
    try:
        r = requests.get(GDC_STATUS_URL, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("  Request failed: %s", exc)
        return None


def write_outage_report(poll_history: list[dict]) -> None:
    report = ROOT / "logs" / "gdc_outage_report.md"
    lines = [
        "# GDC Outage Report",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Poll History",
        "| Timestamp | Status | Response |",
        "|-----------|--------|----------|",
    ]
    for h in poll_history:
        lines.append(f"| {h['time']} | {h['status']} | {h['detail']} |")
    lines += [
        "",
        "## Recommended Action",
        "If the GDC remains unavailable, send the following email:",
        "",
        "**To:** support@nci-gdc.datacommons.io",
        "**Subject:** GDC API Sustained Outage - BioFusionFormer Research Project",
        "",
        "Dear GDC Support Team,",
        "",
        "We have been unable to access the GDC API (https://api.gdc.cancer.gov/status)",
        f"for the past {len(poll_history) * 10} minutes (starting approximately",
        f"{poll_history[0]['time']}).",
        "",
        "Project context: Academic final-year project accessing open-access",
        "TARGET-WT data for multi-modal cancer analysis research.",
        "",
        "Could you please advise on the expected restoration timeline?",
        "",
        "Thank you.",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    log.info("Outage report written to: %s", report)


def trigger_phase_b() -> int:
    pb_script = ROOT / "scripts" / "phase_b_gdc_metadata.py"
    if pb_script.exists():
        log.info("GDC is UP! Launching Phase B: %s ...", pb_script)
        print("\n" + "=" * 60)
        print("GDC RESTORED! Starting Phase B (Cohort Discovery) automatically...")
        print("=" * 60)
        res = subprocess.run([sys.executable, str(pb_script)])
        log.info("Phase B script exited with code %d", res.returncode)
        return res.returncode
    else:
        log.error("Phase B script not found at %s", pb_script)
        return 1


def run_single_check() -> bool:
    """Run a single status check. Returns True if GDC is OK."""
    ts = datetime.now(timezone.utc).isoformat()
    data = check_status()
    if data is not None:
        status  = data.get("status", "unknown")
        tag     = data.get("tag", "")
        version = data.get("version", "")
        detail  = f"status={status} tag={tag} version={version}"
        log.info("Single poll: %s", detail)
        if status == "OK":
            log.info("GDC is UP. Triggering Phase B.")
            trigger_phase_b()
            return True
        else:
            log.warning("GDC status not OK: %s", status)
            return False
    else:
        log.warning("Single poll: connection_failed / 503")
        return False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Poll GDC API status")
    parser.add_argument("--once", action="store_true",
                        help="Run a single check and exit immediately")
    args = parser.parse_args()

    if args.once:
        is_ok = run_single_check()
        sys.exit(0 if is_ok else 2)

    log.info("GDC Status Poller started (max %d polls, interval %d s)",
             MAX_POLLS, POLL_INTERVAL)
    poll_history = []

    for poll_num in range(1, MAX_POLLS + 1):
        ts = datetime.now(timezone.utc).isoformat()
        data = check_status()

        if data is not None:
            status  = data.get("status", "unknown")
            tag     = data.get("tag", "")
            version = data.get("version", "")
            detail  = f"status={status} tag={tag} version={version}"
            log.info("Poll %d/%d: %s", poll_num, MAX_POLLS, detail)
            poll_history.append({"time": ts, "status": status, "detail": detail})

            if status == "OK":
                log.info("GDC is UP. Proceeding to Phase B.")
                print(f"\nGDC STATUS OK — tag={tag}, version={version}")
                code = trigger_phase_b()
                sys.exit(code)
        else:
            detail = "connection_failed"
            log.warning("Poll %d/%d: %s", poll_num, MAX_POLLS, detail)
            poll_history.append({"time": ts, "status": "ERROR", "detail": detail})

        if poll_num < MAX_POLLS:
            log.info("  Next poll in %d minutes ...", POLL_INTERVAL // 60)
            time.sleep(POLL_INTERVAL)

    # 48 h elapsed without OK
    log.error("48-hour polling limit reached without GDC OK.")
    write_outage_report(poll_history)
    print("\nGDC OUTAGE: 48 hours elapsed. See logs/gdc_outage_report.md.")
    sys.exit(1)


if __name__ == "__main__":
    main()
