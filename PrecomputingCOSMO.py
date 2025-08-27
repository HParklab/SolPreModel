import os
import csv
import subprocess
import numpy as np
import shutil
from tqdm import tqdm
import itertools

# ──────────────────────────────────────────────────────────
XYZ_DIR  = "/Users/cxxoseo/Desktop/SolPreModel/Data/SMILEStoXYZ.curated/"
POT_DIR  = "/Users/cxxoseo/Desktop/SolPreModel/COSMO.new/"
LOG_PATH = os.path.join(POT_DIR, "xtb_run_log_re4.csv")
RE_LOG_PATH = os.path.join(POT_DIR, "xtb_run_log_re5.csv")

os.makedirs(POT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────
XTB_EXE = shutil.which("xtb")
if XTB_EXE is None:
    XTB_EXE = "/opt/anaconda3/bin/xtb"

BASE_ENV = os.environ.copy()
BASE_ENV["OMP_NUM_THREADS"] = "1"
# ──────────────────────────────────────────────────────────
def log_result(mol_id, status, message):
    with open(RE_LOG_PATH, 'a', newline='') as logf:
        writer=csv.writer(logf)
        writer.writerow([mol_id, status, message])
# ──────────────────────────────────────────────────────────
def compute_cosmo(xyz_path, new_name, mol_id):

    try:
        proc = subprocess.run(
            [XTB_EXE, xyz_path, "--esp", "--alpb", "WATER", "--uhf", "0"],
            cwd=POT_DIR,
            env=BASE_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        esp_cosmo = os.path.join(POT_DIR, "xtb_esp.cosmo")

        if os.path.isfile(esp_cosmo):
            os.rename(esp_cosmo,new_name)
            log_result(mol_id, "SUCCESS", f"xtb returncode={proc.returncode} (forced success)")
            return
        
        
        log_result(mol_id, "FAIL", f"returncode={proc.returncode}, stdout = {proc.stdout}")

    except Exception as e:
        log_result(mol_id, "ERROR", str(e))

completed_ids = set()
failed_ids = set()

if os.path.isfile(LOG_PATH):
    with open(LOG_PATH, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                mol_id, status = row[0], row[1]
                if status == "SUCCESS":
                    completed_ids.add(row[0])
                elif status in {"FAIL", "ERROR"}:
                    failed_ids.add(mol_id)

xyz_files = sorted([f for f in os.listdir(XYZ_DIR) if f.endswith(".xyz")])
all_ids = [os.path.splitext(f)[0] for f in xyz_files]
priority_ids = list(failed_ids) #+ [mol_id for mol_id in all_ids if mol_id not in completed_ids and mol_id not in failed_ids]


print(f"총 {len(priority_ids)}개의 분자를 계산합니다.")

for mol_id in tqdm(priority_ids, desc="COSMO 계산 진행중", ncols=100):
    xyz_path = os.path.join(XYZ_DIR, f"{mol_id}.xyz")
    out_name = os.path.join(POT_DIR, f"{mol_id}.cosmo")
    
    if not os.path.exists(xyz_path):
        log_result(mol_id, "SKIPPED", "xyz 파일 없음")
        continue

    compute_cosmo(xyz_path, out_name, mol_id)