import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


CSV_PATH = "shanghai_final_score_88.05_12.16.csv"


DATASET_FORMAT = "sht"  

smoothing_window = 12
ema_alpha = 0.33
λ = 0.8
μ = 1.0


df = pd.read_csv(CSV_PATH)


required = ["image_path", "label", "memloss", "score", "top1_recon_losses"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# （可选）top1_sentences 不一定有，后面保存时会补空
# =================================================


# ===================== 解析函数（新增UBn支持） =====================
def extract_frame_id(p: str) -> int:
    """
    兼容：
    - SHT: .../089.jpg
    - UBn: .../frame_002.jpg
    """
    base = os.path.basename(str(p)).split("?")[0]
    nums = re.findall(r"\d+", base)
    if not nums:
        raise ValueError(f"Cannot parse frame id from image_path: {p}")
    return int(nums[-1])


def parse_video_scene_ids(image_path: str, dataset_format: str):
    """
    (video_id, scene_id)
    - SHT: scene_id=01, video_id=01_0014 
    - UBn: scene_id=Scene1, video_id=abnormal_scene_1_scenario_2 
    """
    p = str(image_path).replace("\\", "/")
    fmt = dataset_format.lower()

    if fmt in ("sht", "shtech", "shanghai"):
        m_vid = re.search(r"/(\d{2}_\d{4})/", p)
        m_sid = re.search(r"/(\d{2})_\d{4}/", p)
        video_id = m_vid.group(1) if m_vid else None
        scene_id = m_sid.group(1) if m_sid else None
        return video_id, scene_id

    if fmt in ("ubn", "ubnormal", "ub"):
        # .../Scene1/abnormal_scene_1_scenario_2/frame_002.jpg
        parts = p.strip("/").split("/")
        if len(parts) < 3:
            return None, None
        video_id = parts[-2]  # abnormal_scene_1_scenario_2
        scene_id = parts[-3]  # Scene1
        return video_id, scene_id

    raise ValueError(f"Unknown DATASET_FORMAT: {dataset_format}")


# ===================== video_id / scene_id / frame_id =====================
vs = df["image_path"].astype(str).apply(lambda x: parse_video_scene_ids(x, DATASET_FORMAT))
df["video_id"] = vs.apply(lambda t: t[0])
df["scene_id"] = vs.apply(lambda t: t[1])
df["frame_id"] = df["image_path"].astype(str).apply(extract_frame_id)


df = df.dropna(subset=["video_id", "scene_id", "frame_id"]).copy()


def anomaly_diffusion_v2(values, window_size, gama=0.6):
    values = np.array(values, dtype=float)
    values_new = values.copy()
    for i in range(window_size, len(values) - window_size):
        window = values[i - window_size:i + window_size + 1]
        non_zero_count = np.count_nonzero(window != 0)
        threshold = int(len(window) * gama)
        if non_zero_count >= threshold:
            values_new[i] = max(window)
    return values_new

df["memloss_smooth"] = np.nan
df["score_smooth"] = np.nan
df["top1_smooth"] = np.nan

for vid, group in df.groupby("video_id"):
    group = group.sort_values("frame_id")
    idx = group.index

    memloss = group["memloss"].astype(float).values
    scores  = group["score"].astype(float).values
    top1    = group["top1_recon_losses"].astype(float).values

    memloss = anomaly_diffusion_v2(memloss, smoothing_window)
    scores  = anomaly_diffusion_v2(scores,  smoothing_window)
    top1    = anomaly_diffusion_v2(top1,    smoothing_window)

    memloss = pd.Series(memloss).ewm(alpha=ema_alpha, adjust=True).mean().values
    scores  = pd.Series(scores ).ewm(alpha=ema_alpha, adjust=True).mean().values
    top1    = pd.Series(top1   ).ewm(alpha=ema_alpha, adjust=True).mean().values

    df.loc[idx, "memloss_smooth"] = memloss
    df.loc[idx, "score_smooth"]   = scores
    df.loc[idx, "top1_smooth"]    = top1

# =====================  min-max （memloss & top1） =====================
df["memloss_norm"] = np.nan
df["top1_norm"] = np.nan

for sid, group in df.groupby("scene_id"):
    idx = group.index

    mvals = group["memloss_smooth"].astype(float).values
    mmax, mmin = np.nanmax(mvals), np.nanmin(mvals)
    memloss_norm = (mvals - mmin) / (mmax - mmin) if np.isfinite(mmax) and np.isfinite(mmin) and mmax > mmin else np.zeros_like(mvals)

    tvals = group["top1_smooth"].astype(float).values
    tmax, tmin = np.nanmax(tvals), np.nanmin(tvals)
    top1_norm = (tvals - tmin) / (tmax - tmin) if np.isfinite(tmax) and np.isfinite(tmin) and tmax > tmin else np.zeros_like(tvals)

    df.loc[idx, "memloss_norm"] = memloss_norm
    df.loc[idx, "top1_norm"] = top1_norm

# ===================== AUC =====================
labels = df["label"].astype(int).values
if len(np.unique(labels)) < 2:
    raise RuntimeError("Labels contain only one class, cannot compute ROC-AUC.")

auc_mem = roc_auc_score(labels, df["memloss_norm"].astype(float).values)
print(f"Overall AUC (Memloss scene-level norm): {auc_mem:.6f}")

auc_score = roc_auc_score(labels, df["score_smooth"].astype(float).values)
print(f"Overall AUC (Scores smooth):            {auc_score:.6f}")

auc_top1 = roc_auc_score(labels, df["top1_norm"].astype(float).values)
print(f"Overall AUC (Top1Recon scene norm):     {auc_top1:.6f}")

df["combined_new"] = df["memloss_norm"].astype(float) + λ * df["top1_norm"].astype(float) + 0.1 * μ * df["score_smooth"].astype(float)
auc_new = roc_auc_score(labels, df["combined_new"].values.astype(float))
print(f"Overall AUC (Combined + top1):          {auc_new:.6f}")


