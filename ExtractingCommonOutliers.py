import pandas as pd
import numpy as np

# ==== 파일 경로 수정하세요 ====
CSV_ECFP  = "./REALEND/ecfp_test.csv"
CSV_GLEM  = "./REALEND/glem_test.csv"
CSV_FUSED = "./REALEND/fused_test.csv"   # 파일명에 맞게 바꿔주세요

TOP_N = 10

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 컬럼 이름 정리(BOM/공백 제거)
    df = df.copy()
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    return df

def _ensure_rank(df: pd.DataFrame) -> pd.DataFrame:
    # rank_by_abs_error 없으면 생성 (abs_residual 큰 값이 rank 1)
    if "rank_by_abs_error" not in df.columns:
        if "abs_residual" not in df.columns:
            raise KeyError("abs_residual column not found")
        # 중복 id가 있으면 id별로 abs_residual 최대행만 유지
        df = df.sort_values("abs_residual", ascending=False)
        df = df.groupby("id", as_index=False).first()
        ranks = (-df["abs_residual"]).rank(method="min").astype(int)
        df["rank_by_abs_error"] = ranks
    else:
        # 혹시 중복 id가 있다면 rank가 가장 작은(=오차 큰) 행 한 개만 남김
        df = df.sort_values("rank_by_abs_error", ascending=True)
        df = df.groupby("id", as_index=False).first()
    return df

def load_one(path: str, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _clean_cols(df)
    # 필수 컬럼 확인
    for col in ["id", "true", "pred", "abs_residual"]:
        if col not in df.columns:
            raise KeyError(f"{path}: missing column '{col}'")
    df = _ensure_rank(df)
    # 필요한 컬럼만, 접두사 붙여서 반환
    keep = ["id", "true", "pred", "abs_residual", "rank_by_abs_error"]
    df = df[keep].rename(columns={
        "true": f"true_{tag}",
        "pred": f"pred_{tag}",
        "abs_residual": f"abs_{tag}",
        "rank_by_abs_error": f"rank_{tag}",
    })
    return df

# 로드
ecfp  = load_one(CSV_ECFP,  "ecfp")
glem  = load_one(CSV_GLEM,  "glem")
fused = load_one(CSV_FUSED, "fused")

# 공통 ID로 inner merge
m = ecfp.merge(glem, on="id", how="inner").merge(fused, on="id", how="inner")

# 집계 지표: 평균/최대 절대오차, 평균 랭크(작을수록 좋음)
m["mean_abs_residual"] = m[["abs_ecfp", "abs_glem", "abs_fused"]].mean(axis=1)
m["max_abs_residual"]  = m[["abs_ecfp", "abs_glem", "abs_fused"]].max(axis=1)
m["avg_rank"]          = m[["rank_ecfp", "rank_glem", "rank_fused"]].mean(axis=1)

# 최종 선택: 평균 랭크 기준 상위 TOP_N (동점 시 mean_abs_residual 큰 순)
top = m.sort_values(["avg_rank", "mean_abs_residual"], ascending=[True, False]).head(TOP_N)

# 보기 좋게 컬럼 정리
cols = [
    "id",
    "avg_rank", "mean_abs_residual", "max_abs_residual",
    "rank_ecfp", "rank_glem", "rank_fused",
    "abs_ecfp", "abs_glem", "abs_fused",
    "true_ecfp", "pred_ecfp",
    "true_glem", "pred_glem",
    "true_fused", "pred_fused",
]
top = top[cols]

# 저장 & 출력
OUT_CSV = "./REALEND/common_high_error_top10.csv"
top.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
print(top.to_string(index=False))
