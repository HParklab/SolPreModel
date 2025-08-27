# --- Splitter for two experiments (AqSolDBc / OChemUnseen) ---
import os, json, random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.ML.Cluster import Butina

random.seed(42)

# -----------------------------
# 0) 데이터 로딩
# -----------------------------
def load_aqsoldbc(aqsol_path, id_col='ID', smi_col='SmilesCurated'):
    df = pd.read_csv(aqsol_path)
    df = df[[id_col, smi_col]].dropna()
    id2smi = dict(zip(df[id_col].astype(str), df[smi_col].astype(str)))
    return id2smi

def load_ochemunseen(ochem_path, id_col = 'index', smi_col='SMILES'):
    df = pd.read_csv(ochem_path)
    df = df[[id_col, smi_col]].dropna()
    id2smi = dict(zip(df[id_col].astype(str), df[smi_col].astype(str)))
    return id2smi

# -----------------------------
# 1) Fingerprint 생성 (ECFP4, 2048bits)
# -----------------------------
def smiles_to_mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    Chem.SanitizeMol(m)
    return m

def make_fps(id2smi, radius=2, nBits=2048):
    ids, fps = [], []
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    for i, smi in id2smi.items():
        m = smiles_to_mol(smi)
        if m is None: 
            continue
        fp = fpg.GetFingerprint(m)
        ids.append(i); fps.append(fp)
    return ids, fps  # ids와 fps의 인덱스가 대응

# -----------------------------
# 2) Butina 클러스터링 (Tanimoto)
# -----------------------------
def butina_cluster_ids(ids, fps, cutoff=0.35):
    # RDKit Butina는 1D distance list 필요 → 1 - similarity
    dists = []
    n = len(fps)
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=1 - cutoff, isDistData=True)
    # clusters는 인덱스 튜플 묶음 → id 리스트로 변환
    cluster_id_lists = [[ids[idx] for idx in c] for c in clusters]
    # 큰 클러스터부터 정렬(배정 안정성↑)
    cluster_id_lists.sort(key=len, reverse=True)
    return cluster_id_lists

# -----------------------------
# 3) 클러스터 단위 분배
#    ratios: {'train':0.8, 'val':0.1, 'test':0.1}
# -----------------------------
def assign_clusters(cluster_id_lists, ratios, total_count):
    target = {k: int(round(v*total_count)) for k, v in ratios.items()}
    # 합 보정
    diff = total_count - sum(target.values())
    if diff != 0:
        # 가장 큰 비중 split에 보정
        main = max(ratios, key=ratios.get)
        target[main] += diff

    bins = {k: [] for k in ratios}
    counts = {k: 0  for k in ratios}

    # 그리디: 매 클러스터를 현재/목표 gap이 가장 큰 split에 배정
    for cl in cluster_id_lists:
        gaps = {k: target[k] - counts[k] for k in ratios}
        # 남은 여유가 가장 큰 split
        pick = max(gaps, key=gaps.get)
        bins[pick].extend(cl)
        counts[pick] += len(cl)

    # 혹시 미세 오차 있으면 가장 작은 split에서 큰 split로 이동(옵션)
    return bins

# -----------------------------
# 4) 버전1: AqSolDBc → 8:1:1
# -----------------------------
def make_split_version1(aq_csv, cutoff=0.35):
    aq = load_aqsoldbc(aq_csv)
    ids, fps = make_fps(aq)
    clusters = butina_cluster_ids(ids, fps, cutoff=cutoff)
    ratios = {'train':0.8,'val':0.1,'test':0.1}
    bins = assign_clusters(clusters, ratios, total_count=len(ids))
    return bins  # dict: {'train':[ids...], 'val':[...], 'test':[...]}

# -----------------------------
# 5) 버전2: AqSolDBc → 8:2,  Test=OChemUnseen 전체
# -----------------------------
def make_split_version2(aq_csv, oc_csv, cutoff=0.35):
    aq = load_aqsoldbc(aq_csv)
    ids, fps = make_fps(aq)
    clusters = butina_cluster_ids(ids, fps, cutoff=cutoff)
    ratios = {'train':0.8,'val':0.2}
    bins_tv = assign_clusters(clusters, ratios, total_count=len(ids))

    oc = load_ochemunseen(oc_csv)
    test_ids = list(oc.keys())  # OChemUnseen 전부 테스트
    return {'train': bins_tv['train'], 'val': bins_tv['val'], 'test': test_ids}

# -----------------------------
# 6) 저장/로드 헬퍼
# -----------------------------
def save_split(bins, out_json):
    with open(out_json, 'w') as f:
        json.dump(bins, f, indent=2, ensure_ascii=False)
    print(f"Saved split → {out_json}")

def load_split(path_json):
    with open(path_json, 'r') as f:
        return json.load(f)

# -----------------------------
# 7) 실행
# -----------------------------
if __name__ == "__main__":
    aq_csv = "/Users/cxxoseo/Desktop/SolPreModel/AqSolDBc/AqSolDBc.filtered.csv"
    oc_csv = "/Users/cxxoseo/Desktop/SolPreModel/OChemUnseen/OChemUnseen.filtered.csv"

    # 버전1
    v1 = make_split_version1(aq_csv, cutoff=0.35)
    save_split(v1, "/Users/cxxoseo/Desktop/SolPreModel/DataSplit/split.v1.json")

    # 버전2
    v2 = make_split_version2(aq_csv, oc_csv, cutoff=0.35)
    save_split(v2, "/Users/cxxoseo/Desktop/SolPreModel/DataSplit/split.v2.json")
