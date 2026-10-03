import subprocess
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
bat_path = root / "scripts" / "run_gdc_poll.bat"
py_path = root / ".venv" / "Scripts" / "python.exe"
script_path = root / "scripts" / "wait_for_gdc.py"
log_path = root / "logs" / "gdc_poll.log"

bat_content = f"""@echo off
cd /d "{root}"
"{py_path}" "{script_path}" --once >> "{log_path}" 2>&1
"""

bat_path.write_text(bat_content, encoding="ascii")
print(f"Created batch wrapper: {bat_path}")

# Register scheduled task every 10 minutes
cmd = [
    "schtasks.exe", "/create",
    "/tn", "BioFusionFormer_GDC_Poller",
    "/tr", f'"{bat_path}"',
    "/sc", "minute",
    "/mo", "10",
    "/f"
]

res = subprocess.run(cmd, capture_output=True, text=True)
print("schtasks returncode:", res.returncode)
print("schtasks stdout:", res.stdout.strip())
if res.stderr:
    print("schtasks stderr:", res.stderr.strip())
