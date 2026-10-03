@echo off
cd /d "C:\Users\LIB-01\Documents\CSE - IOTCSBCT\BioFusionFormer"
"C:\Users\LIB-01\Documents\CSE - IOTCSBCT\BioFusionFormer\.venv\Scripts\python.exe" "C:\Users\LIB-01\Documents\CSE - IOTCSBCT\BioFusionFormer\scripts\wait_for_gdc.py" --once >> "C:\Users\LIB-01\Documents\CSE - IOTCSBCT\BioFusionFormer\logs\gdc_poll.log" 2>&1
