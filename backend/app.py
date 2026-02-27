import os
import re
import sys
import json
import math
import types
import uuid
import tempfile
import traceback
import subprocess

import vlm_code
import google.generativeai as genai
import time


from dotenv import load_dotenv

import subprocess # 맨 위쪽에 추가 안 되어 있다면 추가할 것

def crop_and_resize_video(input_path, output_path):
    """영상의 가로/세로 중 짧은 쪽 기준으로 중앙 정사각형 크롭 후 224x224 리사이즈"""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            # 🌟 [수정] 무조건 1080이 아니라, 동적으로 중앙 정사각형을 잡음
            '-vf', "crop='min(iw,ih)':'min(iw,ih)':'(iw-min(iw,ih))/2':'(ih-min(iw,ih))/2',scale=224:224",
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast',
            '-c:a', 'copy',
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ [FFmpeg] 비디오 전처리 실패: {e}")
        return False

# .env 파일을 찾아서 환경변수로 강제 로드
load_dotenv('/home/ubuntu/ai-muncheol/backend/.env')
# ══════════════════════════════════════════════════════════
# 🔧 mmaction2 drn 모듈 버그 패치
# ══════════════════════════════════════════════════════════
def patch_mmaction_drn():
    try:
        drn_pkg = types.ModuleType("mmaction.models.localizers.drn")
        drn_drn = types.ModuleType("mmaction.models.localizers.drn.drn")
        class DRN: pass
        drn_drn.DRN = DRN
        drn_pkg.drn = drn_drn
        sys.modules["mmaction.models.localizers.drn"] = drn_pkg
        sys.modules["mmaction.models.localizers.drn.drn"] = drn_drn
        print("✅ mmaction drn 모듈 패치 완료")
    except Exception as e:
        print(f"⚠️ drn 패치 실패: {e}")

patch_mmaction_drn()

import torch
import torch.nn as nn                          
import numpy as np                             
import cv2                                     
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.config import Config

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════
# 📂 경로 및 모델 설정
# ══════════════════════════════════════════════════════════
BASE_DIR = "/home/ubuntu/ai-muncheol/backend"

MODEL_META = {
    1: {"k": 5,  "out_key": "accident_place",              "prob_key": "probability", "map_key": "model1", "db_map": "place", "label": "장소"},
    2: {"k": 10, "out_key": "accident_place_feature_code", "prob_key": "probability", "map_key": "model2", "db_map": "type",  "label": "사고유형"},
    3: {"k": 10, "out_key": "vehicle_a_code",              "prob_key": "prob",        "map_key": "model3", "db_map": "action", "label": "차량A"},
    4: {"k": 10, "out_key": "vehicle_b_code",              "prob_key": "prob",        "map_key": "model4", "db_map": "action", "label": "차량B"},
}

GROUPS = {
    "은석": "es",
    "형선": "hs"
}

MODELS_CONFIG = {}
for name_kr, prefix in GROUPS.items():
    for i in range(1, 5):
        key = f"{prefix}_model{i}"
        meta = MODEL_META[i]
        
        MODELS_CONFIG[key] = {
            "config": os.path.join(BASE_DIR, "configs", f"{key}_config.py"),
            "checkpoint": os.path.join(BASE_DIR, "weights", f"{key}.pth"),
            "meta": meta,
            "group": name_kr
        }

# ══════════════════════════════════════════════════════════
# 🆕 C3D 모델 설정
# ══════════════════════════════════════════════════════════
C3D_CHECKPOINT = os.path.join(BASE_DIR, "weights", "best_c3d.pt")
C3D_RESIZE = 224       # 학습 시 사용한 해상도 (노트북 Cell 35 확인: RESIZE=224)
C3D_T = 16             # 클립 프레임 수
C3D_NUM_CLASSES = 117  # 학습 시 클래스 수


# ══════════════════════════════════════════════════════════
# 🆕 C3D 모델 클래스 (v9: AdaptivePool, 224×224 입력)
#    - 노트북 Cell 35 추론 코드에서 그대로 가져옴
# ══════════════════════════════════════════════════════════
class C3D(nn.Module):
    """
    jfzhang95 C3D 구조 (Sports-1M pretrained 호환)
    학습 입력: (B, 3, 16, 224, 224) + AdaptiveAvgPool3d
    """
    def __init__(self, num_classes=117):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,1,1))
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x)); x = self.pool1(x)
        x = self.relu(self.conv2(x)); x = self.pool2(x)
        x = self.relu(self.conv3a(x)); x = self.relu(self.conv3b(x)); x = self.pool3(x)
        x = self.relu(self.conv4a(x)); x = self.relu(self.conv4b(x)); x = self.pool4(x)
        x = self.relu(self.conv5a(x)); x = self.relu(self.conv5b(x)); x = self.pool5(x)
        x = self.adaptive_pool(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x)); x = self.dropout(x)
        x = self.relu(self.fc7(x)); x = self.dropout(x)
        return self.fc8(x)


# ══════════════════════════════════════════════════════════
# 🆕 C3D 전처리 함수
#    - 노트북 Cell 35의 read_frames + sample_multi_clips(val) 재현
#    - 224×224 리사이즈 → 16프레임 중앙 클립 → BGR→RGB → /255 → tensor
# ══════════════════════════════════════════════════════════
def preprocess_video_for_c3d(video_path, T=16, resize=224):
    """
    C3D 추론용 영상 전처리 (학습 노트북 val 경로 그대로)

    Returns: (1, 3, T, resize, resize) float32 tensor
    """
    # 1) 프레임 읽기 + 224×224 리사이즈
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (resize, resize))
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames in {video_path}")

    # 2) loop pad (프레임 < T일 때 반복)
    if len(frames) < T:
        repeat = (T + len(frames) - 1) // len(frames)
        frames = (frames * repeat)

    # 3) 중앙 클립 샘플링 (val: num_clips=1, center)
    L = len(frames)
    start = max(0, (L - T) // 2)
    clip = frames[start:start + T]
    if len(clip) < T:
        clip = clip + [clip[-1]] * (T - len(clip))

    # 4) numpy → tensor 변환
    clip = np.stack(clip, axis=0)              # (T, 224, 224, 3) BGR
    clip = clip[..., ::-1].copy()               # BGR → RGB
    clip = clip.astype(np.float32) / 255.0
    clip = np.transpose(clip, (3, 0, 1, 2))    # (3, T, 224, 224)

    # 5) 배치 차원 추가
    tensor = torch.from_numpy(clip).unsqueeze(0)  # (1, 3, T, 224, 224)
    return tensor


def run_c3d_inference(model, video_path, device, idx_to_class, k=10):
    """
    C3D 추론 실행 → top-K 예측 반환

    Returns: list of {class_label: int, prob: float, model_idx: int}
    """
    tensor = preprocess_video_for_c3d(video_path, T=C3D_T, resize=C3D_RESIZE)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                          # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    topk_vals, topk_inds = torch.topk(probs, min(k, len(probs)))

    results = []
    for idx, prob in zip(topk_inds.tolist(), topk_vals.tolist()):
        class_label = idx_to_class.get(idx, idx)   # 모델 인덱스 → 원래 클래스 라벨
        results.append({
            "class_label": int(class_label),
            "model_idx": idx,
            "prob": float(prob),
        })

    print(f"  📊 [C3D] Top-5 예측:")
    for r in results[:5]:
        print(f"      클래스={r['class_label']}, prob={r['prob']:.4f}")

    return results


# ══════════════════════════════════════════════════════════
# 🗺️ 모델 인덱스 → DB ID 매핑 (기존 그대로)
# ══════════════════════════════════════════════════════════
MAP_MODEL1 = {i: v for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 13])}
MAP_MODEL2 = {i: v for i, v in enumerate([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 18, 20, 21, 22, 23, 24,
    37, 38, 39, 40, 41, 45, 48, 49, 50, 59, 60
])}
MAP_MODEL3 = {i: v for i, v in enumerate([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 88, 89,
    90, 91, 133, 134, 135, 138, 139, 140, 144, 147, 148, 154, 169, 170, 171,
    172, 173, 174, 175, 176, 177, 178, 179
])}
MAP_MODEL4 = {i: v for i, v in enumerate([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 45, 46, 47, 50,
    52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 74,
    87, 88, 89, 90, 91, 92, 139, 140, 142, 143, 146, 147, 150, 151, 165, 166,
    167, 168, 169, 170, 171, 172, 173
])}

MODEL_MAPS = {
    "model1": MAP_MODEL1,
    "model2": MAP_MODEL2,
    "model3": MAP_MODEL3,
    "model4": MAP_MODEL4,
}

# ══════════════════════════════════════════════════════════
# 📊 라벨 맵
# ══════════════════════════════════════════════════════════
LABEL_MAP_PLACE = {
    0: "직선 도로", 1: "신호 없는 교차로", 2: "신호 있는 교차로",
    3: "t자형 도로", 4: "기타 도로", 5: "주차장",
    6: "회전 교차로", 13: "고속도로"
}

LABEL_MAP_TYPE = {}
LABEL_MAP_ACTION = {}
CRASH_DF = pd.DataFrame()

def load_csv_labels():
    global CRASH_DF, LABEL_MAP_TYPE, LABEL_MAP_ACTION

    csv_candidates = [
        os.path.join(BASE_DIR, "data", "matching.csv"),
    ]

    df = pd.DataFrame()
    final_path = None

    for p in csv_candidates:
        if not os.path.exists(p):
            continue
        for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                temp = pd.read_csv(p, encoding=enc)
                temp.columns = temp.columns.str.strip()
                if "과실비율A" in temp.columns and "사고장소특징_ID" in temp.columns:
                    df = temp
                    final_path = p
                    break
            except Exception:
                continue
        if not df.empty:
            break

    if df.empty:
        print("⚠️ '과실비율A' 컬럼이 포함된 유효한 CSV 파일을 찾을 수 없습니다.")
        return

    for col in ["사고장소특징_ID", "A진행방향_ID", "B진행방향_ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

    CRASH_DF = df

    if "사고장소특징_ID" in df.columns and "사고장소특징" in df.columns:
        LABEL_MAP_TYPE = df.groupby("사고장소특징_ID")["사고장소특징"].first().to_dict()
        LABEL_MAP_TYPE[45] = LABEL_MAP_TYPE.get(9, "기타 사고(48번 대체)")

    if "A진행방향_ID" in df.columns:
        map_a = df[["A진행방향_ID", "A진행방향"]].dropna().drop_duplicates()
        map_b = df[["B진행방향_ID", "B진행방향"]].dropna().drop_duplicates()
        map_a.columns = ["ID", "Label"]
        map_b.columns = ["ID", "Label"]
        combined = pd.concat([map_a, map_b]).drop_duplicates(subset="ID")
        LABEL_MAP_ACTION = combined.set_index("ID")["Label"].to_dict()

    print(f"✅ CSV 로드 완료 ({os.path.basename(final_path)}): {len(df)}행, 사고유형 {len(LABEL_MAP_TYPE)}개, 진행방향 {len(LABEL_MAP_ACTION)}개")

LABEL_MAPS = {
    "place": LABEL_MAP_PLACE,
    "type": LABEL_MAP_TYPE,
    "action": LABEL_MAP_ACTION,
}

# ══════════════════════════════════════════════════════════
# 🔧 Config 로드
# ══════════════════════════════════════════════════════════
def safe_load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = re.sub(r"custom_imports\s*=\s*dict\(.*?\)\s*\n", "", text, flags=re.DOTALL)
    
    if "LDAMLossCustom" in text:
        print(f" 🛠️ [Config 패치] {os.path.basename(config_path)}: LDAMLossCustom 제거 중...")
        text = text.replace("'LDAMLossCustom'", "'CrossEntropyLoss'")
        text = text.replace('"LDAMLossCustom"', '"CrossEntropyLoss"')
        text = re.sub(r"cls_num_list\s*=\s*\[.*?\]\s*,?", "", text, flags=re.DOTALL)
        text = re.sub(r"\bmax_m\s*=\s*[\d\.]+\s*,?", "", text)
        text = re.sub(r"\bs\s*=\s*[\d\.]+\s*,?", "", text)
        
    text = re.sub(
        r"loss_cls=dict\(\s*alpha=[\s\S]*?type='mmdet\.FocalLoss'[\s\S]*?\)",
        "loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)",
        text,
    )
    
    text = re.sub(r"load_from\s*=\s*'[^']*'", "load_from = None", text)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    
    try:
        cfg = Config.fromfile(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return cfg


# ══════════════════════════════════════════════════════════
# 🎬 영상 코덱 확인 / 변환 (ffmpeg)
# ══════════════════════════════════════════════════════════
def get_video_codec(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_video_duration(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def convert_to_h264(input_path, output_path):
    try:
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-vcodec', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '23',
            '-acodec', 'aac', '-strict', '-2',
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        return True
    except Exception as e:
        print(f"  ⚠️ H.264 변환 실패: {e}")
        return False


# ══════════════════════════════════════════════════════════
# 🧠 Top-K 추출 (mmaction2 1.2.0 호환)
# ══════════════════════════════════════════════════════════
def extract_top_k(res, model_name="", k=3):
    if isinstance(res, (list, tuple)):
        res = res[0]

    scores = None
    attrs = [a for a in dir(res) if not a.startswith('_')]

    if hasattr(res, 'pred_score') and scores is None:
        val = getattr(res, 'pred_score')
        if torch.is_tensor(val):
            scores = val

    if hasattr(res, 'pred_scores') and scores is None:
        pred_scores = getattr(res, 'pred_scores')
        if torch.is_tensor(pred_scores):
            scores = pred_scores
        else:
            if hasattr(pred_scores, 'keys'):
                try:
                    for key in pred_scores.keys():
                        val = pred_scores[key]
                        if torch.is_tensor(val):
                            scores = val
                            break
                except Exception:
                    pass
            if scores is None and hasattr(pred_scores, 'values'):
                try:
                    for val in pred_scores.values():
                        if torch.is_tensor(val):
                            scores = val
                            break
                except Exception:
                    pass
            for attr in ['data', 'score', 'scores', 'label']:
                if scores is not None:
                    break
                if hasattr(pred_scores, attr):
                    val = getattr(pred_scores, attr)
                    if torch.is_tensor(val):
                        scores = val

    if scores is None:
        for attr_name in attrs:
            if 'score' in attr_name.lower():
                val = getattr(res, attr_name, None)
                if torch.is_tensor(val) and val.dim() >= 1:
                    scores = val
                    break

    if scores is None:
        raise ValueError(f"[{model_name}] scores 추출 실패!")

    if scores.dim() > 1:
        scores = scores.squeeze()
    scores = scores.cpu().to(torch.float64)

    print(f"  📊 [{model_name}] scores shape: {scores.shape}")
    top5 = scores.topk(min(5, len(scores)))
    print(f"  📊 [{model_name}] 상위5 값: {[f'{v:.4f}' for v in top5.values.tolist()]}")
    print(f"  📊 [{model_name}] 상위5 idx: {top5.indices.tolist()}")

    if scores.min() >= 0 and scores.max() <= 1 and scores.sum() > 0.5:
        probs = scores / scores.sum()
    else:
        probs = torch.nn.functional.softmax(scores, dim=0)

    topk_vals, topk_inds = torch.topk(probs, min(k, len(probs)))
    return topk_inds.tolist(), topk_vals.tolist()


# ══════════════════════════════════════════════════════════
# ⚖️ 매칭 알고리즘 (은석/형선용 - 기존 그대로)
# ══════════════════════════════════════════════════════════
def calculate_fault_scores(group_data, crash_df):
    """
    group_data: final_output["은석"] 또는 final_output["형선"] 리스트
    """
    if crash_df.empty or len(group_data) < 4:
        return None, []

    cand_type = group_data[1] if group_data[1] else []
    cand_a = group_data[2] if group_data[2] else []
    cand_b = group_data[3] if group_data[3] else []

    eps = 1e-12
    combinations = []

    for t in cand_type:
        for a in cand_a:
            for b in cand_b:
                t_code = t.get("accident_place_feature_code")
                a_code = a.get("vehicle_a_code")
                b_code = b.get("vehicle_b_code", b.get("vehicle_b_info_code"))
                
                t_prob = t.get("probability", t.get("prob", 0))
                a_prob = a.get("probability", a.get("prob", 0))
                b_prob = b.get("probability", b.get("prob", 0))

                if t_code is None or a_code is None or b_code is None:
                    continue

                log_score = (
                    math.log(max(float(t_prob), eps))
                    + math.log(max(float(a_prob), eps))
                    + math.log(max(float(b_prob), eps))
                )
                combinations.append({
                    "type": t_code, "a": a_code, "b": b_code,
                    "log_score": log_score,
                })

    if not combinations:
        return None, []

    log_scores_tensor = torch.tensor([c["log_score"] for c in combinations], dtype=torch.float64)
    norm_confs = torch.nn.functional.softmax(log_scores_tensor, dim=0).tolist()

    for c, p in zip(combinations, norm_confs):
        c["norm_conf"] = p

    combinations.sort(key=lambda x: x["norm_conf"], reverse=True)

    fault_result = None
    alt_faults = []

    for combo in combinations:
        match_rows = crash_df[
            (crash_df["사고장소특징_ID"] == combo["type"])
            & (crash_df["A진행방향_ID"] == combo["a"])
            & (crash_df["B진행방향_ID"] == combo["b"])
        ]

        if not match_rows.empty:
            row = match_rows.iloc[0]
            fa = int(row["과실비율A"])
            fb = int(row["과실비율B"])

            entry = {
                "fa": fa,
                "fb": fb,
                "role_a": "가해자" if fa > fb else ("피해자" if fa < fb else "쌍방"),
                "role_b": "피해자" if fa > fb else ("가해자" if fa < fb else "쌍방"),
                "confidence": round(combo["norm_conf"] * 100, 2),
                "accident_place": str(row.get("사고장소", "")),
                "accident_feature": str(row.get("사고장소특징", "")),
                "codes": f"T{combo['type']}-A{combo['a']}-B{combo['b']}"
            }

            if fault_result is None:
                fault_result = entry
            elif len(alt_faults) < 3:
                alt_faults.append(entry)

            if len(alt_faults) >= 3 and fault_result is not None:
                break

    return fault_result, alt_faults


# ══════════════════════════════════════════════════════════
# 🆕 C3D 과실비율 매칭 함수
# ══════════════════════════════════════════════════════════
def calculate_c3d_fault(c3d_predictions, crash_df):
    """
    C3D 예측 결과로 과실비율 매칭.
    
    C3D는 traffic_accident_type(사고유형 ID)을 직접 예측하므로
    crash_df에서 해당 사고유형의 대표 행을 찾아 과실비율을 반환.
    
    매칭 전략 (순서대로 시도):
      1순위: class_label → crash_df 행 인덱스로 직접 조회
      2순위: class_label → 사고장소특징_ID 컬럼에서 검색
    """
    if crash_df.empty or not c3d_predictions:
        return None, []

    fault_result = None
    alt_faults = []

    for pred in c3d_predictions:
        label = pred["class_label"]
        prob = pred["prob"]
        row = None

        # 전략 1: class_label을 crash_df 행 인덱스로 시도
        if 0 <= label < len(crash_df):
            row = crash_df.iloc[label]

        # 전략 2: 사고장소특징_ID 컬럼에서 검색
        if row is None:
            match = crash_df[crash_df["사고장소특징_ID"] == label]
            if not match.empty:
                row = match.iloc[0]

        if row is not None:
            fa = int(row["과실비율A"])
            fb = int(row["과실비율B"])
            entry = {
                "fa": fa,
                "fb": fb,
                "role_a": "가해자" if fa > fb else ("피해자" if fa < fb else "쌍방"),
                "role_b": "피해자" if fa > fb else ("가해자" if fa < fb else "쌍방"),
                "confidence": round(prob * 100, 2),
                "accident_place": str(row.get("사고장소", "")),
                "accident_feature": str(row.get("사고장소특징", "")),
                "codes": f"C3D-class{label}"
            }

            if fault_result is None:
                fault_result = entry
            elif len(alt_faults) < 3:
                alt_faults.append(entry)

            if len(alt_faults) >= 3:
                break

    return fault_result, alt_faults


# ══════════════════════════════════════════════════════════
# 🆕 C3D 예측 → 프론트엔드 4-모델 형식 변환
# ══════════════════════════════════════════════════════════
def build_c3d_data(c3d_predictions, crash_df):
    """
    C3D 예측 결과를 프론트엔드가 기대하는 4-모델 배열 형식으로 변환.
    
    프론트엔드 기대 형식:
      c3d_data = [
        [{accident_place: code, probability: p}, ...],            # model1: 장소
        [{accident_place_feature_code: code, probability: p}, ...], # model2: 사고유형
        [{vehicle_a_code: code, prob: p}, ...],                   # model3: 차량A
        [{vehicle_b_code: code, prob: p}, ...],                   # model4: 차량B
      ]
    
    C3D는 사고유형을 통째로 예측하므로, crash_df 행에서 개별 코드를 추출.
    """
    slot_place = []    # model1
    slot_type = []     # model2
    slot_a = []        # model3
    slot_b = []        # model4

    for pred in c3d_predictions[:10]:
        label = pred["class_label"]
        prob = pred["prob"]
        row = None

        # crash_df에서 매칭
        if 0 <= label < len(crash_df):
            row = crash_df.iloc[label]
        if row is None:
            match = crash_df[crash_df["사고장소특징_ID"] == label]
            if not match.empty:
                row = match.iloc[0]

        if row is not None:
            # 매칭 성공 → crash_df 행에서 개별 코드 추출
            if "사고장소_ID" in row.index:
                place_id = row["사고장소_ID"]
                if pd.notna(place_id):
                    slot_place.append({"accident_place": int(place_id), "probability": prob})

            type_id = row.get("사고장소특징_ID", -1)
            if pd.notna(type_id) and int(type_id) >= 0:
                slot_type.append({"accident_place_feature_code": int(type_id), "probability": prob})

            a_id = row.get("A진행방향_ID", -1)
            if pd.notna(a_id) and int(a_id) >= 0:
                slot_a.append({"vehicle_a_code": int(a_id), "prob": prob})

            b_id = row.get("B진행방향_ID", -1)
            if pd.notna(b_id) and int(b_id) >= 0:
                slot_b.append({"vehicle_b_code": int(b_id), "prob": prob})
        else:
            # 매칭 실패 → class_label을 사고유형 코드로 직접 사용
            slot_type.append({"accident_place_feature_code": label, "probability": prob})

    return [slot_place, slot_type, slot_a, slot_b]


# ══════════════════════════════════════════════════════════
# 🚀 모델 로드
# ══════════════════════════════════════════════════════════
loaded_models = {}
c3d_model = None                # 🆕
c3d_idx_to_class = {}           # 🆕
c3d_class_to_idx = {}           # 🆕
VLM_SESSIONS = {}               # 🆕 세션별 Gemini video_file + pred_codes 저장


# ══════════════════════════════════════════════════════════
# 🌐 API 엔드포인트
# ══════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(loaded_models.keys()),
        "c3d_loaded": c3d_model is not None,            # 🆕
        "c3d_classes": len(c3d_idx_to_class),            # 🆕
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "label_map_type_count": len(LABEL_MAPS.get("type", {})),
        "label_map_action_count": len(LABEL_MAPS.get("action", {})),
        "csv_rows": len(CRASH_DF),
    })


@app.route("/api/convert", methods=["POST"])
def convert_preview():
    """브라우저 미리보기용 H.264 변환"""
    if "video" not in request.files:
        return jsonify({"error": "영상 파일이 필요합니다"}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()
    input_path = tmp.name

    codec = get_video_codec(input_path)
    print(f"  🎬 [변환 요청] 코덱: {codec}")

    if codec == "h264":
        from flask import send_file
        return send_file(input_path, mimetype="video/mp4", download_name="preview.mp4")

    output_path = input_path + "_h264.mp4"
    if convert_to_h264(input_path, output_path):
        os.remove(input_path)
        from flask import send_file
        resp = send_file(output_path, mimetype="video/mp4", download_name="preview.mp4")

        @resp.call_on_close
        def cleanup():
            try:
                os.remove(output_path)
            except Exception:
                pass

        return resp
    else:
        os.remove(input_path)
        return jsonify({"error": "변환 실패"}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """8개 mmaction 모델 + C3D 실행 → 그룹별 결과 + 과실비율"""  # ✏️
    if "video" not in request.files:
        return jsonify({"error": "영상 파일이 필요합니다"}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()
    video_path = tmp.name

    actual_video = video_path

    def generate():
        try:
            # ─── 1. 결과 그릇 초기화 ───
            final_output = {
                "은석": [[], [], [], []],
                "형선": [[], [], [], []]
            }

            # ✏️ 진행률: 8개 mmaction + 1개 C3D = 총 9단계
            total_models = len(MODELS_CONFIG)
            total_steps = total_models + (1 if c3d_model else 0)
            current_idx = 0

            # ─── 2. mmaction 모델 8개 순회 (기존 그대로) ───
            sorted_keys = sorted(MODELS_CONFIG.keys())

            for key in sorted_keys:
                cfg = MODELS_CONFIG[key]
                group_name = cfg.get("group", "은석")

                model_num = int(key[-1])
                idx_in_group = model_num - 1

                meta = cfg.get("meta", cfg)
                k_val = meta.get("k", 10)
                out_key = meta.get("out_key", "code")
                prob_key = meta.get("prob_key", "prob")
                label_name = meta.get("label", f"모델{model_num}")
                map_key = meta.get("map_key", f"model{model_num}")

                model = loaded_models.get(key)

                msg_text = f"{group_name} {label_name} 분석 중..."
                yield f"data: {json.dumps({'type': 'progress', 'message': msg_text, 'percent': int(current_idx / total_steps * 90)}, ensure_ascii=False)}\n\n"

                if not model:
                    print(f"❌ {key} 모델 미로드")
                    current_idx += 1
                    continue

                res = inference_recognizer(model, actual_video)
                inds, probs = extract_top_k(res, model_name=key, k=k_val)

                mapping = MODEL_MAPS.get(map_key, {})

                model_result_list = []
                for idx, prob in zip(inds, probs):
                    code = mapping.get(idx, idx)
                    item = {
                        out_key: int(code),
                        prob_key: float(prob)
                    }
                    model_result_list.append(item)

                final_output[group_name][idx_in_group] = model_result_list
                current_idx += 1

            # ══════════════════════════════════════════════
            # 🆕 3. C3D 모델 추론
            # ══════════════════════════════════════════════
            c3d_data = None
            c3d_predictions = []

            if c3d_model is not None:
                yield f"data: {json.dumps({'type': 'progress', 'message': '수민 3D CNN 분석 중...', 'percent': int(current_idx / total_steps * 90)}, ensure_ascii=False)}\n\n"

                try:
                    device = next(c3d_model.parameters()).device
                    c3d_predictions = run_c3d_inference(
                        c3d_model, actual_video, device,
                        c3d_idx_to_class, k=10
                    )
                    c3d_data = build_c3d_data(c3d_predictions, CRASH_DF)
                    print(f"✅ [C3D] 추론 완료: top1={c3d_predictions[0]['class_label']} "
                          f"(prob={c3d_predictions[0]['prob']:.4f})")
                except Exception as e:
                    print(f"❌ [C3D] 추론 실패: {e}")
                    traceback.print_exc()

                current_idx += 1

            # ─── 4. 과실비율 매칭 ───
            fault_es, alt_es = calculate_fault_scores(final_output["은석"], CRASH_DF)
            fault_hs, alt_hs = calculate_fault_scores(final_output["형선"], CRASH_DF)
            fault_c3d, alt_c3d = calculate_c3d_fault(c3d_predictions, CRASH_DF)  # 🆕

            if fault_es:
                print(f"⚖️ [은석] 과실비율: A={fault_es['fa']}% / B={fault_es['fb']}%")
            else:
                print("⚠️ [은석] 과실비율 매칭 실패")

            if fault_hs:
                print(f"⚖️ [형선] 과실비율: A={fault_hs['fa']}% / B={fault_hs['fb']}%")
            else:
                print("⚠️ [형선] 과실비율 매칭 실패")

            if fault_c3d:                                                         # 🆕
                print(f"⚖️ [C3D] 과실비율: A={fault_c3d['fa']}% / B={fault_c3d['fb']}%")
            else:
                print("⚠️ [C3D] 과실비율 매칭 실패")

            
            # ... (앞부분: 은석/형선/C3D 과실비율 print 출력 완료) ...

            sumin_result = {"accident_type": None}
            if c3d_predictions and len(c3d_predictions) > 0:
                try:
                    top_class = int(c3d_predictions[0].get('class_label', -1))
                    sumin_result = {"accident_type": top_class}
                except Exception:
                    sumin_result = {"accident_type": c3d_predictions[0].get('class_label')}
            final_output["수민"] = sumin_result


            # 🌟 1차 전송 (partial_complete): 모델 분석 3개가 끝났으니 먼저 6페이지로 넘김!
            partial_evt = {
                "type": "partial_complete",
                "input_data": final_output,
                "c3d_data": c3d_data, # 기존 호환성 유지
                "fault_results": {
                    "은석": {"best": fault_es, "alts": alt_es},
                    "형선": {"best": fault_hs, "alts": alt_hs},
                    "c3d":  {"best": fault_c3d, "alts": alt_c3d},
                },
                "fault": fault_es,
                "alt_faults": alt_es,
                "label_maps": {
                    "place":  {str(k): v for k, v in LABEL_MAPS["place"].items()},
                    "type":   {str(k): v for k, v in LABEL_MAPS["type"].items()},
                    "action": {str(k): v for k, v in LABEL_MAPS["action"].items()},
                }
            }
            yield f"data: {json.dumps(partial_evt, ensure_ascii=False)}\n\n"


            # ─── 4.5 VLM 스코어링 (리포트는 개별 요청 시 생성) ───
            print(f"\n🚀 [VLM] 비디오 전처리 (Crop & Resize) 시작...", flush=True)
            
            cropped_video_path = video_path.replace('.mp4', '_cropped.mp4')
            preprocess_success = crop_and_resize_video(actual_video, cropped_video_path)
            vlm_upload_path = cropped_video_path if preprocess_success else actual_video
            
            video_stem = "test_video"
            api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            
            print(f"  🚀 [VLM] Gemini API 비디오 업로드 시작...", flush=True)
            video_file = genai.upload_file(path=vlm_upload_path) 
            
            print("  ⏳ [VLM] 구글 서버에서 영상 처리 대기 중...", flush=True)
            while True:
                video_file = genai.get_file(video_file.name)
                state = video_file.state.name
                if state == "PROCESSING":
                    print(".", end="", flush=True)
                    time.sleep(2)
                elif state == "ACTIVE":
                    print("\n  ✅ [VLM] 비디오 처리 완료! (ACTIVE 확인)", flush=True)
                    break
                else:
                    print(f"\n  ❌ [VLM] 비디오 처리 실패! 상태: {state}", flush=True)
                    break
            
            if video_file.state.name == "ACTIVE":
                success, best_pred_code, _, model_results = vlm_code.run_score_test(video_stem, 0, video_file, final_output)

                if success:
                    es_pred, hs_pred, total_pred1, total_pred2, sm_pred, vlm_scores, vlm_sources = model_results

                    # 🆕 개별 모델(은석/형선/수민) 중에서만 최적 모델 선정 (통합 표시 X)
                    def _code_match_count(code_a, code_b):
                        """두 예측코드의 4자리 중 일치하는 개수 반환 (높을수록 유사)"""
                        if not code_a or not code_b: return -1
                        nums_a = re.findall(r'\d+', str(code_a))
                        nums_b = re.findall(r'\d+', str(code_b))
                        if len(nums_a) < 4 or len(nums_b) < 4: return -1
                        return sum(1 for a, b in zip(nums_a[:4], nums_b[:4]) if a == b)

                    best_model_name = None
                    # 1순위: 정확히 일치하는 모델
                    if best_pred_code == es_pred: best_model_name = "민다정"
                    elif best_pred_code == hs_pred: best_model_name = "엄도식"
                    elif best_pred_code == sm_pred: best_model_name = "윤 슬"
                    else:
                        # 2순위: 가장 유사한(코드 일치 수 많은) 모델 선택
                        candidates = [
                            ("은석", es_pred, _code_match_count(best_pred_code, es_pred)),
                            ("형선", hs_pred, _code_match_count(best_pred_code, hs_pred)),
                            ("수민", sm_pred, _code_match_count(best_pred_code, sm_pred)),
                        ]
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        best_model_name = candidates[0][0]
                        print(f"  ℹ️ [VLM] 정확히 일치하는 모델 없음 → 가장 유사한 '{best_model_name}' 선택 "
                              f"(일치 {candidates[0][2]}/4)", flush=True)

                    print(f"  🏆 [VLM] 1등 모델 선정: {best_model_name}", flush=True)

                    # 🆕 세션에 Gemini video_file + 예측코드 저장 (개별 리포트 생성용)
                    session_id = str(uuid.uuid4())[:8]
                    pred_codes = {}
                    if es_pred and es_pred != "(-1, -1, -1, -1)":
                        pred_codes["은석"] = es_pred
                    if hs_pred and hs_pred != "(-1, -1, -1, -1)":
                        pred_codes["형선"] = hs_pred
                    if sm_pred and sm_pred != "(-1, -1, -1, -1)":
                        pred_codes["수민"] = sm_pred

                    VLM_SESSIONS[session_id] = {
                        "video_file": video_file,
                        "pred_codes": pred_codes,
                        "video_stem": video_stem,
                        "created_at": time.time(),
                    }
                    print(f"  💾 [VLM] 세션 저장: {session_id}, 예측코드: {list(pred_codes.keys())}", flush=True)

                    # 🌟 vlm_ready 이벤트: 프론트에서 개별 리포트 요청 가능해짐
                    vlm_ready_evt = {
                        "type": "vlm_ready",
                        "session_id": session_id,
                        "best_model": best_model_name,
                        "best_code": best_pred_code,
                        "pred_codes": pred_codes,
                    }
                    yield f"data: {json.dumps(vlm_ready_evt, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'vlm_ready', 'session_id': None, 'error': 'VLM 스코어링 실패'}, ensure_ascii=False)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'vlm_ready', 'session_id': None, 'error': f'Gemini 영상 처리 실패: {video_file.state.name}'}, ensure_ascii=False)}\n\n"

            # ─── 임시 파일 청소 (자른 영상 지움) ───
            if os.path.exists(cropped_video_path):
                os.remove(cropped_video_path)

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

    return Response(generate(), mimetype="text/event-stream")


# ══════════════════════════════════════════════════════════
# 🆕 개별 VLM 리포트 생성 엔드포인트
# ══════════════════════════════════════════════════════════
@app.route("/api/vlm_report", methods=["POST"])
def vlm_report():
    """세션 ID + 모델 이름으로 개별 VLM 리포트 생성"""
    data = request.get_json()
    session_id = data.get("session_id")
    model_name = data.get("model_name")  # "형선" / "은석" / "수민"

    if not session_id or not model_name:
        return jsonify({"error": "session_id와 model_name이 필요합니다"}), 400

    session = VLM_SESSIONS.get(session_id)
    if not session:
        return jsonify({"error": "세션이 만료되었거나 존재하지 않습니다"}), 404

    pred_code = session["pred_codes"].get(model_name)
    if not pred_code:
        return jsonify({"error": f"{model_name} 모델의 예측코드가 없습니다"}), 404

    video_file = session["video_file"]
    video_stem = session["video_stem"]

    try:
        print(f"  📝 [VLM] {model_name} 리포트 작성 요청 (session={session_id})...", flush=True)
        report_text = vlm_code.run_explan_test(video_stem, model_name, video_file, pred_code, "")

        if report_text:
            print(f"  ✅ [VLM] {model_name} 리포트 작성 완료!", flush=True)
            return jsonify({"status": "success", "report": report_text, "pred_code": pred_code})
        else:
            return jsonify({"status": "error", "message": f"{model_name} 리포트 생성에 실패했습니다."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


# ══════════════════════════════════════════════════════════
# 🆕 VLM 세션 정리 (오래된 세션 삭제)
# ══════════════════════════════════════════════════════════
@app.route("/api/vlm_cleanup", methods=["POST"])
def vlm_cleanup():
    """30분 이상 된 세션 자동 정리"""
    now = time.time()
    expired = [sid for sid, s in VLM_SESSIONS.items() if now - s["created_at"] > 1800]
    for sid in expired:
        try:
            VLM_SESSIONS[sid]["video_file"].delete()
        except Exception:
            pass
        del VLM_SESSIONS[sid]
    return jsonify({"cleaned": len(expired), "remaining": len(VLM_SESSIONS)})


# ══════════════════════════════════════════════════════════
# 🚀 모델 로드 함수
# ══════════════════════════════════════════════════════════
def load_all_models():
    global loaded_models, c3d_model, c3d_idx_to_class, c3d_class_to_idx
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  디바이스: {device}")

    # ── mmaction 모델 8개 로드 (기존 그대로) ──
    sorted_keys = sorted(MODELS_CONFIG.keys())

    for key in sorted_keys:
        info = MODELS_CONFIG[key]
        config_path = info["config"]
        ckpt_path = info["checkpoint"]
        meta = info["meta"]

        if not os.path.exists(config_path):
            print(f"❌ {key}: config 없음 → {config_path}")
            continue
        if not os.path.exists(ckpt_path):
            print(f"❌ {key}: checkpoint 없음 → {ckpt_path}")
            continue

        try:
            print(f"📦 {key} ({meta['label']}) 로딩 중...")
            cfg = safe_load_config(config_path)

            if not hasattr(cfg, "test_pipeline") or cfg.test_pipeline is None:
                if hasattr(cfg, "val_pipeline"):
                    cfg.test_pipeline = cfg.val_pipeline

            model = init_recognizer(cfg, ckpt_path, device=device)
            loaded_models[key] = model
            print(f"✅ {key} 로드 완료")
        except Exception as e:
            print(f"❌ {key} 로드 실패: {e}")

    print(f"\n🎉 총 {len(loaded_models)}/{len(MODELS_CONFIG)} mmaction 모델 로드 완료")

    # ══════════════════════════════════════════════════════
    # 🆕 C3D 모델 로드
    # ══════════════════════════════════════════════════════
    if os.path.exists(C3D_CHECKPOINT):
        try:
            print(f"\n📦 C3D 모델 로딩 중... ({C3D_CHECKPOINT})")
            ckpt = torch.load(C3D_CHECKPOINT, map_location=device, weights_only=False)

            # 체크포인트에서 클래스 매핑 복원
            c3d_class_to_idx = ckpt.get("class_to_idx", {})
            c3d_idx_to_class = ckpt.get("idx_to_class", {})
            num_classes = len(c3d_class_to_idx) if c3d_class_to_idx else C3D_NUM_CLASSES

            # 모델 생성 + 가중치 로드
            c3d_model = C3D(num_classes=num_classes).to(device)
            c3d_model.load_state_dict(ckpt["model_state"])
            c3d_model.eval()

            epoch = ckpt.get("epoch", "?")
            val_acc = ckpt.get("best_val_acc", 0)
            print(f"✅ C3D 로드 완료: {num_classes}개 클래스, "
                  f"epoch={epoch}, val_acc={val_acc * 100:.2f}%, "
                  f"입력=({C3D_T}×{C3D_RESIZE}×{C3D_RESIZE})")
        except Exception as e:
            print(f"❌ C3D 로드 실패: {e}")
            traceback.print_exc()
            c3d_model = None
    else:
        print(f"\n⚠️ C3D 체크포인트 없음: {C3D_CHECKPOINT}")
        print("   → C3D 없이 은석/형선 모델만으로 실행됩니다")


# ══════════════════════════════════════════════════════════
# 🏁 서버 시작
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI 문철 백엔드 서버 v5 (SSE + C3D 통합)")       # ✏️ v4→v5
    print("=" * 60)
    load_csv_labels()
    LABEL_MAPS["type"] = LABEL_MAP_TYPE
    LABEL_MAPS["action"] = LABEL_MAP_ACTION
    load_all_models()
    print("\n" + "=" * 60)
    print("🌐 서버 실행: http://localhost:5002")
    if c3d_model:                                               # 🆕
        print(f"🧬 C3D 모델: 활성 ({len(c3d_idx_to_class)}개 클래스)")
    else:
        print("🧬 C3D 모델: 비활성 (체크포인트 없음)")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5002, debug=False)