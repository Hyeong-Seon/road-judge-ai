import os
import re
import sys
import json
import math
import types
import tempfile
import traceback
import subprocess

import vlm_code
import google.generativeai as genai
import time

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
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

from mmaction.apis import init_recognizer, inference_recognizer
from mmengine.config import Config

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════
# 📂 경로 설정
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# 📂 경로 및 모델 설정 (수정됨)
# ══════════════════════════════════════════════════════════
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
# 🗺️ 모델 인덱스 → DB ID 매핑
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
    
    # 1. 커스텀 임포트 구문 제거
    text = re.sub(r"custom_imports\s*=\s*dict\(.*?\)\s*\n", "", text, flags=re.DOTALL)
    
    # 2. 🚨 핵심 수정: LDAMLossCustom -> CrossEntropyLoss 로 강제 변환
    if "LDAMLossCustom" in text:
        print(f" 🛠️ [Config 패치] {os.path.basename(config_path)}: LDAMLossCustom 제거 중...")
        # 타입 변경
        text = text.replace("'LDAMLossCustom'", "'CrossEntropyLoss'")
        text = text.replace('"LDAMLossCustom"', '"CrossEntropyLoss"')
        
        # LDAM 전용 파라미터(리스트) 제거
        text = re.sub(r"cls_num_list\s*=\s*\[.*?\]\s*,?", "", text, flags=re.DOTALL)
        
        # 🔥 [수정됨] LDAM 기타 파라미터 제거 (max_m, s)
        # \b (단어 경계)를 추가하여 'eps' 같은 다른 변수가 망가지지 않도록 보호함
        text = re.sub(r"\bmax_m\s*=\s*[\d\.]+\s*,?", "", text)
        text = re.sub(r"\bs\s*=\s*[\d\.]+\s*,?", "", text)
        
    # 3. 기존 FocalLoss 처리 (유지)
    text = re.sub(
        r"loss_cls=dict\(\s*alpha=[\s\S]*?type='mmdet\.FocalLoss'[\s\S]*?\)",
        "loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0)",
        text,
    )
    
    # 4. load_from 경로 제거
    text = re.sub(r"load_from\s*=\s*'[^']*'", "load_from = None", text)

    # 임시 파일 생성 및 로드
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
    """영상 코덱 확인"""
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
    """영상 길이(초) 반환"""
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
    """H.264로 변환"""
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

    # 방법 1: pred_score
    if hasattr(res, 'pred_score') and scores is None:
        val = getattr(res, 'pred_score')
        if torch.is_tensor(val):
            scores = val

    # 방법 2: pred_scores → LabelData
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

    # 방법 3: fallback
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
# 🚀 모델 로드
# ══════════════════════════════════════════════════════════
loaded_models = {}

# ══════════════════════════════════════════════════════════
# ⚖️ 매칭 알고리즘 (새로운 JSON 구조에 맞게 수정됨)
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# ⚖️ 매칭 알고리즘 (키 이름 호환성 강화)
# ══════════════════════════════════════════════════════════
def calculate_fault_scores(group_data, crash_df):
    """
    group_data: final_output["은석"] 또는 final_output["형선"] 리스트
    """
    if crash_df.empty or len(group_data) < 4:
        return None, []

    # 모델 2, 3, 4 결과 매핑 (인덱스: 1, 2, 3)
    cand_type = group_data[1] if group_data[1] else []
    cand_a = group_data[2] if group_data[2] else []
    cand_b = group_data[3] if group_data[3] else []

    eps = 1e-12
    combinations = []

    for t in cand_type:
        for a in cand_a:
            for b in cand_b:
                # 1. 사고유형 코드 추출
                t_code = t.get("accident_place_feature_code")
                
                # 2. 차량 A 코드 추출
                a_code = a.get("vehicle_a_code")
                
                # 3. 차량 B 코드 추출 (은석: vehicle_b_code, 형선: vehicle_b_info_code 호환)
                b_code = b.get("vehicle_b_code", b.get("vehicle_b_info_code"))
                
                # 확률 추출
                t_prob = t.get("probability", t.get("prob", 0))
                a_prob = a.get("probability", a.get("prob", 0))
                b_prob = b.get("probability", b.get("prob", 0))

                # 필수 코드가 없으면 스킵
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

    # 점수 정렬 및 상위 후보 추출
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
                # 디버깅용 정보
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
# 🌐 API
# ══════════════════════════════════════════════════════════
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(loaded_models.keys()),
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
        # 이미 H.264면 그대로 반환
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
    """8개 모델 실행 후 그룹별 JSON 포맷 반환 및 과실비율 계산"""
    if "video" not in request.files:
        return jsonify({"error": "영상 파일이 필요합니다"}), 400

    video_file = request.files["video"]
    suffix = os.path.splitext(video_file.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.save(tmp.name)
    tmp.close()
    video_path = tmp.name

    # ... (코덱 변환 로직은 기존과 동일하게 유지) ...
    # 편의상 코덱 변환 후 실제 사용할 영상 경로를 actual_video라고 가정합니다.
    actual_video = video_path 
    # (필요시 위 코드의 변환 로직 그대로 사용)

    def generate():
        try:
            # 1. 결과 담을 그릇 초기화
            final_output = {
                "은석": [[], [], [], []], # model 1, 2, 3, 4 순서대로 저장
                "형선": [[], [], [], []]
            }
            
            # 진행률 계산용
            total_models = len(MODELS_CONFIG)
            current_idx = 0

            # 2. 모델 순회 (es_model1 -> es_model4 -> hs_model1 -> ...)
            # 순서를 보장하기 위해 정렬 (es가 먼저 오도록)
            sorted_keys = sorted(MODELS_CONFIG.keys()) 

            for key in sorted_keys:
                cfg = MODELS_CONFIG[key]
                group_name = cfg.get("group", "은석")
                
                # 1. 키 이름의 맨 끝 숫자(1~4)를 추출해 자동으로 배열 인덱스(0~3)로 변환
                model_num = int(key[-1])
                idx_in_group = model_num - 1
                
                # 2. 기존 meta 방식과 직접 하드코딩 방식 모두 호환되도록 안전하게 값 가져오기
                meta = cfg.get("meta", cfg)
                k_val = meta.get("k", 10)
                out_key = meta.get("out_key", "code")
                prob_key = meta.get("prob_key", "prob")
                label_name = meta.get("label", f"모델{model_num}")
                map_key = meta.get("map_key", f"model{model_num}")
                
                model = loaded_models.get(key)
                
                # 진행 상황 전송
                msg_text = f"{group_name} {label_name} 분석 중..."
                yield f"data: {json.dumps({'type': 'progress', 'message': msg_text, 'percent': int(current_idx/total_models*90)}, ensure_ascii=False)}\n\n"

                if not model:
                    print(f"❌ {key} 모델 미로드")
                    current_idx += 1
                    continue

                # 3. 추론 실행
                res = inference_recognizer(model, actual_video)
                
                # 4. Top-K 추출 (동적 K 적용)
                inds, probs = extract_top_k(res, model_name=key, k=k_val)
                
                # 매핑 테이블 가져오기
                mapping = MODEL_MAPS.get(map_key, {})

                # 5. 결과 리스트 생성 (각각 다른 키 이름 적용)
                model_result_list = []
                for idx, prob in zip(inds, probs):
                    code = mapping.get(idx, idx)
                    
                    item = {
                        out_key: int(code),
                        prob_key: float(prob)
                    }
                    model_result_list.append(item)

                # 6. 결과 저장
                final_output[group_name][idx_in_group] = model_result_list
                current_idx += 1

            # -------------------------------------------------------------
            # [추가된 부분] 8개 모델 추론 완료 후 과실비율 매칭 (은석 기준)
            # -------------------------------------------------------------
            # ... (앞부분 for문 생략) ...

            # -------------------------------------------------------------
            # [수정됨] 과실비율 매칭: 은석 / 형선 각각 수행
            # -------------------------------------------------------------
            
            # 1. 은석 모델 기준 과실비율
            fault_es, alt_es = calculate_fault_scores(final_output["은석"], CRASH_DF)
            
            # 2. 형선 모델 기준 과실비율
            fault_hs, alt_hs = calculate_fault_scores(final_output["형선"], CRASH_DF)

            # 로그 출력
            if fault_es:
                print(f"⚖️ [은석] 과실비율: A={fault_es['fa']}% / B={fault_es['fb']}%")
            else:
                print("⚠️ [은석] 과실비율 매칭 실패")

            if fault_hs:
                print(f"⚖️ [형선] 과실비율: A={fault_hs['fa']}% / B={fault_hs['fb']}%")
            else:
                print("⚠️ [형선] 과실비율 매칭 실패")

            # 3. 최종 결과 전송 (구조 변경)
            # 프론트엔드에서 fault_es, fault_hs를 각각 써야 합니다.
            final_evt = {
                "type": "complete",
                "input_data": final_output,
                
                # 각각의 결과 객체를 담아 보냅니다
                "fault_results": {
                    "은석": {"best": fault_es, "alts": alt_es},
                    "형선": {"best": fault_hs, "alts": alt_hs}
                },
                
                # (하위 호환성 유지용) 기존 fault 키에는 은석 결과를 넣어둠
                "fault": fault_es, 
                "alt_faults": alt_es,
                
                "vlm_report": "VLM 분석은 현재 비활성화 상태입니다." 
            }
            yield f"data: {json.dumps(final_evt, ensure_ascii=False)}\n\n"

        except Exception as e:
# ... (뒷부분 동일)
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
            # 변환된 파일 삭제 로직 등 추가

    return Response(generate(), mimetype="text/event-stream")

# ══════════════════════════════════════════════════════════
# 🚀 모델 로드 함수 (이게 없어서 에러가 난 겁니다)
# ══════════════════════════════════════════════════════════
loaded_models = {}

def load_all_models():
    global loaded_models
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  디바이스: {device}")
    
    # MODELS_CONFIG 키 정렬 (로그 보기 좋게)
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
            
            # 파이프라인 설정 안전장치
            if not hasattr(cfg, "test_pipeline") or cfg.test_pipeline is None:
                if hasattr(cfg, "val_pipeline"):
                    cfg.test_pipeline = cfg.val_pipeline
            
            model = init_recognizer(cfg, ckpt_path, device=device)
            loaded_models[key] = model
            print(f"✅ {key} 로드 완료")
        except Exception as e:
            print(f"❌ {key} 로드 실패: {e}")
            # traceback.print_exc() # 필요시 주석 해제

    print(f"\n🎉 총 {len(loaded_models)}/{len(MODELS_CONFIG)} 모델 로드 완료")
# ══════════════════════════════════════════════════════════
# 🏁 서버 시작
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI 문철 백엔드 서버 v4 (SSE + ffmpeg 변환)")
    print("=" * 60)
    load_csv_labels()
    LABEL_MAPS["type"] = LABEL_MAP_TYPE
    LABEL_MAPS["action"] = LABEL_MAP_ACTION
    load_all_models()
    print("\n" + "=" * 60)
    print("🌐 서버 실행: http://localhost:5002")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5002, debug=False)