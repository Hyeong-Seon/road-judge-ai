import smtplib
from email.mime.text import MIMEText
import http.client
import json, re, os, random, csv, io, time
import pandas as pd
import google.generativeai as genai
from pathlib import Path
from google.generativeai import caching
import datetime
import itertools
import math
import ast

import os
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # 이렇게 바꿔주세요
#pdf_path = "/content/drive/MyDrive/cv_final/데이터 관련/1-56_교통사고 영상 데이터_과실비율 내용 정리.pdf"
#csv_gt_path = "/content/drive/MyDrive/cv_final/데이터 관련/블랙박스_AB여부.csv" # 업로드하신 정답지 경로
mapping_path = "/home/ubuntu/ai-muncheol/backend/data/matching.csv"
csv_type_path = "/home/ubuntu/ai-muncheol/backend/data/accident_type.csv"

#base_video_root = "/content/cache/val/"
#base_label_root = "/content/cache/pred/"
#base_csv_root = "/content/cache/ana_log_final/"
#drive_root = "/content/drive/MyDrive/260220_ai"

place_hierarchy_instruction = """
[Place 계층 판단 규칙]
- place 코드를 바로 고르지 말고, 먼저 도로 토폴로지(대분류)를 판단하십시오.
  1) 직선 도로 계열: code 0
  2) 사거리 교차로 계열: code 1 또는 2
  3) T자형 교차로: code 3
  4) 차도/비차도 경계 또는 비도로 계열: code 4 또는 5
  5) 회전교차로: code 6
  6) 고속도로/자동차전용도로 계열: code 13

- 이후에만 세부 place code를 판단하십시오.
  * 사거리 계열(1 vs 2): 신호등 유무로 구분
    - 1: 사거리교차로(신호등 없음)
    - 2: 사거리교차로(신호등 있음)
  * 비도로 계열(4 vs 5):
    - 5: 주차면/주차동선/주차구획 단서가 있으면 우선
    - 4: 차도↔비차도 경계/도로 가장자리/비도로 진입 계열이면 우선
  * 직선 vs 고속도로(0 vs 13):
    - 13: 중앙분리대/다차로 고속주행/램프/방음벽 등 고속도로형 시설 단서가 있을 때 우선

- 대분류가 명백히 맞지 않는 가설은 place_score를 낮게 주고, 필요 시 hard_contradiction=true로 표시하십시오.
"""

system_instruction_score_only = """
너는 교통사고 블랙박스 영상의 후보 가설들을 '최종 선택'하지 않고, 오직 시각적 일치도만 채점하는 평가기다.

[목표]
- 각 후보 가설에 대해 place / feature / maneuver / role 4개 축의 시각 일치도를 0~4점으로 채점한다.
- 최종 1위 선택, 우승 후보 결정, 결론 서사 작성은 하지 않는다.
- source_tag(Eunseok/Hyeongseon/Integrated 등)와 section_type은 점수에 사용하지 않는다. (후처리에서만 사용)

[핵심 규칙]
1) 최종 선택 금지
- 어떤 후보가 정답인지 고르지 마라.
- winner, top1, 최종추천, 최종판단, best hypothesis 같은 표현을 쓰지 마라.
- final_decision_logic, why_not_runner_up 같은 서사형 판단을 만들지 마라.
- axis_comparison은 축별 비교 메모일 뿐 최종 승자를 의미하지 않는다.

2) 동점 허용
- 두 후보가 같은 축에서 모두 높은 점수(예: 둘 다 4점)일 수 있다.
- 억지로 차이를 만들지 마라.
- 변별이 어려운 축은 동점/불확실로 남겨라.

3) 반증 우선
- 각 후보에 대해 score_reasons뿐 아니라 counter_evidence도 반드시 기록한다.
- 반증이 명확하면 높은 점수를 주지 마라.
- 점수는 '후보를 변호'하지 말고, 보이는 단서와 반증을 함께 반영해 채점하라.

4) hard_contradiction 사용 기준 (매우 보수적)
- 영상에서 직접 확인 가능한 명백한 모순일 때만 hard_contradiction=true로 둔다.
- 추정/애매함/가려짐/프레임 부족은 hard_contradiction=true 사유가 아니다.
- hard_contradiction=true 이면 contradiction_axes에 해당 축명(place, feature, maneuver, role)을 넣어라.
- hard_contradiction=false 이면 contradiction_axes는 빈 배열로 둘 수 있다.

5) source_tag / section_type 비사용
- source_tag, section_type은 메타데이터로만 기록한다.
- 'Eunseok이라 더 높게', 'Hyeongseon이라 더 낮게' 같은 prior 판단 금지.

6) 출력 형식
- 한국어로 작성한다.
- 코드펜스 없이 JSON 객체 1개만 출력한다.
- JSON 외 텍스트를 출력하지 마라.
- output_format에 정의된 키 이름을 그대로 사용하라.
- hypothesis_scoring에는 입력으로 제공된 모든 후보(H1, H2, H3)를 빠짐없이 포함하라.

[점수 기준: 모든 축 공통, 0~4 정수]
- 0: 명확한 불일치/모순 (영상과 직접 충돌)
- 1: 약한 일치 또는 단서 부족 (근거가 약함)
- 2: 부분 일치 (맞는 부분과 불확실/충돌 요소가 혼재)
- 3: 강한 일치 (주요 단서들이 대부분 일치)
- 4: 매우 강한 일치 (직접 시각 단서가 명확하게 뒷받침)

[축별 해석 가이드]
- place:
  교차로/직선/도로 구조/차선 흐름/정지선/신호 위치 등 '장소/형태' 일치도
- feature:
  장소 세부 특징(합류, 분기, 횡단보도, 중앙선 형태, 차로 수, 차로 배치 등) 일치도
- maneuver:
  차량들의 진행/회전/진입/정지/차선변경/상대 접근 방향/충돌 직전 동작 일치도
- role:
  블랙박스 차량(ego)과 상대차량(other)의 역할/방향/관계, 그리고 A/B 매핑 일치도

[가시성/근거강도 표기 규칙]
- visibility 값은 다음 중 하나만 사용:
  clear | partial | occluded | unknown
- basis 값은 다음 중 하나만 사용:
  direct_visual | partial_visual | weak_inference | unknown

[관찰 작성 가이드]
- visual_observation의 ego_maneuver_guess / other_vehicle_maneuver_guess에는 A/B 용어를 쓰지 말고 ego/other 기준으로 작성하라.
- role_identification에서만 A/B 매핑을 다룬다.
- score_reasons는 각 축의 점수 부여 이유를 짧고 구체적으로 작성하라.
- counter_evidence는 해당 후보에 불리한 단서를 기록하라. 없으면 None 형태를 사용해도 된다.
- 과도한 장문 설명 대신, 프레임 단서 중심으로 간결하게 작성하라.

[axis_comparison 작성 규칙]
- axis_comparison은 place/feature/maneuver/role 각 축의 비교 메모다.
- equal_groups에는 동률/유사한 후보 묶음을 기록한다. (예: [["H1","H2"]])
- better_supported에는 해당 축에서 근거가 상대적으로 더 선명한 후보를 기록할 수 있다.
- 축별 메모(notes)는 가능하면 짧게 작성한다.
- axis_comparison으로 전체 우승 후보를 만들지 마라.
"""

output_format_score_only = """
[output_format]
아래 형식의 JSON 객체 1개만 출력하라.
- 코드펜스 금지
- JSON 외 텍스트 금지

{
  "meta": {
    "video_id": "입력으로 받은 video_id 문자열 그대로",
    "section_type": "입력으로 받은 section_type 문자열 그대로"
  },

  "pov_observation": {
    "camera_view": "전방|후방|측면|불명",
    "confidence": "low|med|high",
    "evidence": [
      {
        "time": "초반|중반|충돌직전|Xs",
        "detail": "블랙박스 시점(전방/후방/측면/불명) 판단 근거를 짧게"
      }
    ]
  },

  "visual_observation": {
    "road_topology_guess": "직선도로계열|사거리계열|T자형|비도로계열|회전교차로|고속도로계열|불명",
    "ego_maneuver_guess": "블박차(ego)의 물리적 움직임 요약 (A/B 용어 금지, ego/other 기준)",
    "other_vehicle_maneuver_guess": "상대차(other)의 물리적 움직임 요약",
    "collision_geometry": "충돌 유형/각도/부위 요약 (예: 측면 접촉, 정면 추돌 가능성 등)",
    "observation_confidence": "low|med|high",
    "environment_cues": [
      {
        "time": "초반|중반|충돌직전|Xs",
        "detail": "도로형상/신호/차선/정지선/중앙선/횡단보도/교통흐름 단서"
      }
    ]
  },

  "role_identification": {
    "blackbox_is": "A|B|unknown",
    "confidence": "low|med|high",
    "mapping_reason": "ego/other 기동 관찰을 바탕으로 A/B에 매핑한 근거를 1~2문장으로",
    "evidence": [
      {
        "time": "초반|중반|충돌직전|Xs|None",
        "detail": "A/B 매핑 근거 단서 (없으면 None)"
      }
    ]
  },

  "hypothesis_scoring": [
    {
      "hypothesis_id": "H1",
      "target_code_combination": {
        "place": 0,
        "feature": 0,
        "vehicle_a": 0,
        "vehicle_b": 0
      },
      "target": "(P,F,A,B) 조합의 사람이 읽는 설명 텍스트",
      "source_tag": "Agreement_Rank_1|Agreement_Rank_2|Agreement_Rank_3|Eunseok|Hyeongseon|Integrated|기타입력값",

      "hard_contradiction": false,
      "contradiction_axes": ["place"],

      "scores": {
        "place_score": 0,
        "feature_score": 0,
        "maneuver_score": 0,
        "role_score": 0
      },

      "visibility": {
        "place": "clear|partial|occluded|unknown",
        "feature": "clear|partial|occluded|unknown",
        "maneuver": "clear|partial|occluded|unknown",
        "role": "clear|partial|occluded|unknown"
      },

      "basis": {
        "place": "direct_visual|partial_visual|weak_inference|unknown",
        "feature": "direct_visual|partial_visual|weak_inference|unknown",
        "maneuver": "direct_visual|partial_visual|weak_inference|unknown",
        "role": "direct_visual|partial_visual|weak_inference|unknown"
      },

      "score_reasons": {
        "place_reason": "장소/도로형상 관찰 기준 점수 부여 이유",
        "feature_reason": "세부 특징 단서 기준 점수 부여 이유",
        "maneuver_reason": "진행/회전/충돌 동작 기준 점수 부여 이유",
        "role_reason": "A/B 역할 매핑 기준 점수 부여 이유"
      },

      "counter_evidence": [
        {
          "time": "초반|중반|충돌직전|Xs|None",
          "type": "place|feature|maneuver|role|None",
          "detail": "이 후보에 불리한 반증 단서 (없으면 None)"
        }
      ]
    },

    {
      "hypothesis_id": "H2",
      "target_code_combination": {
        "place": 0,
        "feature": 0,
        "vehicle_a": 0,
        "vehicle_b": 0
      },
      "target": "(P,F,A,B) 조합의 사람이 읽는 설명 텍스트",
      "source_tag": "Agreement_Rank_1|Agreement_Rank_2|Agreement_Rank_3|Eunseok|Hyeongseon|Integrated|기타입력값",
      "hard_contradiction": false,
      "contradiction_axes": [],
      "scores": {
        "place_score": 0,
        "feature_score": 0,
        "maneuver_score": 0,
        "role_score": 0
      },
      "visibility": {
        "place": "clear|partial|occluded|unknown",
        "feature": "clear|partial|occluded|unknown",
        "maneuver": "clear|partial|occluded|unknown",
        "role": "clear|partial|occluded|unknown"
      },
      "basis": {
        "place": "direct_visual|partial_visual|weak_inference|unknown",
        "feature": "direct_visual|partial_visual|weak_inference|unknown",
        "maneuver": "direct_visual|partial_visual|weak_inference|unknown",
        "role": "direct_visual|partial_visual|weak_inference|unknown"
      },
      "score_reasons": {
        "place_reason": "장소/도로형상 관찰 기준 점수 부여 이유",
        "feature_reason": "세부 특징 단서 기준 점수 부여 이유",
        "maneuver_reason": "진행/회전/충돌 동작 기준 점수 부여 이유",
        "role_reason": "A/B 역할 매핑 기준 점수 부여 이유"
      },
      "counter_evidence": [
        {
          "time": "초반|중반|충돌직전|Xs|None",
          "type": "place|feature|maneuver|role|None",
          "detail": "이 후보에 불리한 반증 단서 (없으면 None)"
        }
      ]
    },

    {
      "hypothesis_id": "H3",
      "target_code_combination": {
        "place": 0,
        "feature": 0,
        "vehicle_a": 0,
        "vehicle_b": 0
      },
      "target": "(P,F,A,B) 조합의 사람이 읽는 설명 텍스트",
      "source_tag": "Agreement_Rank_1|Agreement_Rank_2|Agreement_Rank_3|Eunseok|Hyeongseon|Integrated|기타입력값",
      "hard_contradiction": false,
      "contradiction_axes": [],
      "scores": {
        "place_score": 0,
        "feature_score": 0,
        "maneuver_score": 0,
        "role_score": 0
      },
      "visibility": {
        "place": "clear|partial|occluded|unknown",
        "feature": "clear|partial|occluded|unknown",
        "maneuver": "clear|partial|occluded|unknown",
        "role": "clear|partial|occluded|unknown"
      },
      "basis": {
        "place": "direct_visual|partial_visual|weak_inference|unknown",
        "feature": "direct_visual|partial_visual|weak_inference|unknown",
        "maneuver": "direct_visual|partial_visual|weak_inference|unknown",
        "role": "direct_visual|partial_visual|weak_inference|unknown"
      },
      "score_reasons": {
        "place_reason": "장소/도로형상 관찰 기준 점수 부여 이유",
        "feature_reason": "세부 특징 단서 기준 점수 부여 이유",
        "maneuver_reason": "진행/회전/충돌 동작 기준 점수 부여 이유",
        "role_reason": "A/B 역할 매핑 기준 점수 부여 이유"
      },
      "counter_evidence": [
        {
          "time": "초반|중반|충돌직전|Xs|None",
          "type": "place|feature|maneuver|role|None",
          "detail": "이 후보에 불리한 반증 단서 (없으면 None)"
        }
      ]
    }
  ],

  "axis_comparison": {
    "place": {
      "equal_groups": [["H1","H2"]],
      "better_supported": ["H3"],
      "notes": ["장소 축 비교 메모 (최종 승자 의미 아님)"]
    },
    "feature": {
      "equal_groups": [],
      "better_supported": [],
      "notes": ["세부특징 축 비교 메모 (없으면 빈 배열 가능)"]
    },
    "maneuver": {
      "equal_groups": [],
      "better_supported": [],
      "notes": ["기동 축 비교 메모 (없으면 빈 배열 가능)"]
    },
    "role": {
      "equal_groups": [],
      "better_supported": [],
      "notes": ["역할 축 비교 메모 (없으면 빈 배열 가능)"]
    }
  }
}
"""

system_instruction_explanation_direct = """
너는 교통사고 블랙박스 영상에 대해 '이미 확정된 사고유형'을 설명하는 작성자다.

역할:
- 입력으로 주어진 확정 유형(장소/특징/A 차량 기동/B 차량 기동/과실비율)을 바탕으로,
  영상을 참고하여 사용자에게 읽기 쉬운 설명을 작성한다.
- 최종 유형을 다시 고르거나 바꾸지 않는다.
- 점수 재계산, 후보 비교, 재선택을 하지 않는다.

중요 규칙:
1) 선택 변경 금지
- 입력으로 주어진 확정 유형이 최종 결과다.
- 다른 유형이 더 맞아 보인다는 식의 재판단 금지.
- 후보 비교/우승 후보/재랭크 금지.

2) A/B 중립 서술 유지 (매우 중요)
- '내 차량', '블박 차량', '상대 차량', '가해차량', '피해차량' 같은 표현 금지.
- 반드시 'A 차량', 'B 차량'으로만 서술한다.
- 영상에서 카메라 시점으로 A/B를 새로 추정하려고 하지 마라.
- A/B의 의미는 입력된 확정 유형 정의를 그대로 따른다.

3) 영상 관찰은 '보강 설명'으로만 사용
- 영상은 확정 유형 설명을 더 구체적으로 만드는 용도로만 사용한다.
- 영상에서 확실하지 않은 내용은 단정하지 말고 '확인 어려움'으로 쓴다.
- 화질/가림/야간/원거리 등으로 불명확하면 불확실성을 명시한다.

4) 서술 톤
- 한국어로 작성한다.
- 설명은 자연스럽고 간결하게 작성한다.
- 법률 자문처럼 단정하지 말고, '입력된 유형 기준/비율 기준'에 따른 설명임을 유지한다.

5) 출력 형식
- 코드펜스 없이 JSON 객체 1개만 출력한다.
- output_format에 정의된 키 이름을 그대로 사용한다.
"""

output_format_explanation_direct = """
{
  "video_observation": {
    "scene_condition": {
      "time_of_day": "주간|야간|불명",
      "weather": "맑음|우천|흐림|불명",
      "visibility_note": "화질/거리/가림/역광 등 관찰 품질 메모 (없으면 '없음')"
    },
    "road_context": {
      "intersection_type_observed": "사거리|T자형|직선도로|기타|불명",
      "signal_observed": "신호등 있음|신호등 없음|확인 어려움",
      "road_scale_hint": "대로/소로 단서 있음|단서 약함|확인 어려움",
      "lane_or_stopline_hint": "차선/정지선/횡단보도 등 보이는 단서 요약 (없으면 '확인 어려움')"
    },
    "movement_observation": {
      "a_vehicle_observation": "입력된 A 차량 기동과 충돌하지 않도록, 영상에서 보이는 움직임 단서를 A 차량 기준으로 요약",
      "b_vehicle_observation": "입력된 B 차량 기동과 충돌하지 않도록, 영상에서 보이는 움직임 단서를 B 차량 기준으로 요약",
      "collision_moment": "충돌 시점/위치/각도/접촉 양상 요약 (확인 어려우면 그렇게 명시)"
    },
    "uncertainties": [
      "불확실한 점 1 (없으면 '없음')"
    ]
  },
  "explanation_text": "사용자에게 보여줄 최종 설명 문단. 반드시 'A 차량', 'B 차량' 표현만 사용하고, 입력된 과실비율(A/B)을 포함해 자연스럽게 설명.",
}
"""

## VLM용 함수 모음

def process_single_json_to_csv(json_path, output_path, csv_type_path='/content/drive/MyDrive/cv_final/데이터 관련/accident_type.csv'):
    """
    JSON 파일 1개를 입력받아 통합 분석 점수를 계산하고 CSV 파일 1개를 생성합니다.
    """
    # --- 설정 값 ---
    SCORE_MODE = 'log'
    EUNSEOK_WEIGHT, HYUNGSUN_WEIGHT = 1.0, 1.0
    MODEL_WEIGHTS = [1.2, 1.0, 1.0, 0.8]
    EPSILON = 1e-6
    TARGET_ATTRIBUTES = ['accident_place', 'accident_place_feature', 'vehicle_a_progress_info', 'vehicle_b_progress_info']
    MODEL_MAP = {
        'model1_place': {'attr': 'accident_place', 'e_id': 'accident_place', 'h_id': 'accident_place'},
        'model2_feature': {'attr': 'accident_place_feature', 'e_id': 'accident_place_feature_code', 'h_id': 'accident_place_feature_code'},
        'model3_vehicle_a': {'attr': 'vehicle_a_progress_info', 'e_id': 'vehicle_a_code', 'h_id': 'vehicle_a_code'},
        'model4_vehicle_b': {'attr': 'vehicle_b_progress_info', 'e_id': 'vehicle_b_code', 'h_id': 'vehicle_b_info_code'}
    }

    # 1. 유효 조합 로드
    valid_combinations = None
    if os.path.exists(csv_type_path):
        df_valid = pd.read_csv(csv_type_path)
        valid_combinations = set(zip(df_valid[TARGET_ATTRIBUTES[0]], df_valid[TARGET_ATTRIBUTES[1]],
                                     df_valid[TARGET_ATTRIBUTES[2]], df_valid[TARGET_ATTRIBUTES[3]]))

    # 2. JSON 데이터 로드
    #with open(json_path, 'r', encoding='utf-8') as f:
    #    data = json.load(f)

    #video = data.get('video', {})
    #gt = tuple(video.get(a) for a in TARGET_ATTRIBUTES)
    #preds = data.get('predictions', {})

    # 3. 모델별 확률 데이터 구조화
    model_data = {attr: {'probs': {'은석': {}, '형선': {}}} for attr in TARGET_ATTRIBUTES}
    for m_key, m_info in MODEL_MAP.items():
        m_val = preds.get(m_key, {})
        attr = m_info['attr']
        # 은석/형선 데이터 파싱 (e_prob/h_prob 키 이름 차이 대응)
        for expert, k_id, k_prob in [('은석', m_info['e_id'], 'probability' if 'model4' not in m_key else 'prob'),
                                     ('형선', m_info['h_id'], 'probability' if 'model3' not in m_key else 'prob')]:
            # 원본 코드의 미묘한 키 명칭 차이 통합 (필요시 m_info에 명시적으로 추가 가능)
            p_key = 'prob' if (expert == '은석' and 'model3' in m_key) or (expert == '형선' and 'model3' in m_key) else 'probability'
            for item in m_val.get(f'{expert}_pred', []):
                model_data[attr]['probs'][expert][item.get(k_id)] = item.get(p_key, 0)

    # 4. 분석 DF 생성 함수
    def get_df(mode):
        c_lists = [list(set(model_data[a]['probs']['은석'].keys()) | set(model_data[a]['probs']['형선'].keys())) for a in TARGET_ATTRIBUTES]
        res = []
        for comb in itertools.product(*c_lists):
            if valid_combinations and comb not in valid_combinations: continue

            # Score 계산 (Log 모드 기준)
            raw_e = sum(MODEL_WEIGHTS[i] * math.log(model_data[TARGET_ATTRIBUTES[i]]['probs']['은석'].get(comb[i], 0) + EPSILON) for i in range(4))
            raw_h = sum(MODEL_WEIGHTS[i] * math.log(model_data[TARGET_ATTRIBUTES[i]]['probs']['형선'].get(comb[i], 0) + EPSILON) for i in range(4))

            weighted_e = raw_e * EUNSEOK_WEIGHT
            weighted_h = raw_h * HYUNGSUN_WEIGHT

            res.append({
                '순위': 0, '정답여부': 'O' if comb == gt else 'X',
                **{attr: val for attr, val in zip(TARGET_ATTRIBUTES, comb)},
                '은석_score': round(weighted_e, 4), '형선_score': round(weighted_h, 4),
                '통합_score': round(weighted_e + weighted_h, 8)
            })

        df = pd.DataFrame(res)
        if df.empty: return df
        sort_col = {'eunseok': '은석_score', 'hyungsun': '형선_score', 'integrated': '통합_score'}[mode]
        df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        df['순위'] = df.index + 1
        return df

    # 5. CSV 저장
    df1, df2, df3 = get_df('eunseok'), get_df('hyungsun'), get_df('integrated')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["[실제 정답 정보]", f"Mode: {SCORE_MODE}", f"Weights: {EUNSEOK_WEIGHT}:{HYUNGSUN_WEIGHT}"])
        writer.writerow(TARGET_ATTRIBUTES)
        writer.writerow(gt)
        writer.writerow([])
        for label, df in [("은석_pred", df1), ("형선_pred", df2), ("통합_분석", df3)]:
            writer.writerow([f"### {label} 결과 ###"])
            if not df.empty: df.to_csv(f, index=False)
            else: writer.writerow(["결과 없음"])
            writer.writerow([])

    return output_path

# --- 사용 예시 ---
# process_single_json_to_csv("input.json", "output.csv")

# 출력 생성용 헬퍼 함수들
# ==============================
# Short analysis row helpers
# ==============================
CONF_MAP = {"low": 1, "med": 2, "high": 3}
CAM_VIEW_MAP = {"전방": 1, "후방": 2, "측면": 3, "불명": 0}
ROAD_TOPO_MAP = {
    "불명": 0,
    "직선도로계열": 1,
    "사거리계열": 2,
    "T자형": 3,
    "비도로계열": 4,
    "회전교차로": 5,
    "고속도로계열": 6,
}
VIS_MAP = {"unknown": 0, "occluded": 1, "partial": 2, "clear": 3}
BASIS_MAP = {"unknown": 0, "weak_inference": 1, "partial_visual": 2, "direct_visual": 3}

# source_tag를 숫자로 인코딩 (필요 시 추가)
SRC_MAP = {
    "": 0,
    "Eunseok": 1,
    "Hyeongseon": 2,
    "Integrated": 3,
    "Agreement_Rank_1": 11,
    "Agreement_Rank_2": 12,
    "Agreement_Rank_3": 13,
}

def _safe_int(v, default=-1):
    try:
        return int(float(v))
    except:
        return default

def _safe_score_04(v):
    x = _safe_int(v, default=-1)
    if x < 0:
        return -1
    if x > 4:
        return 4
    return x

def _bool01(v):
    return 1 if bool(v) else 0

def _yesno01(v):
    s = str(v).strip().lower()
    return 1 if s in ["1", "true", "yes", "y", "pass", "ok"] else 0

def _enc_conf(v):
    return CONF_MAP.get(str(v).strip().lower(), 0)

def _enc_cam(v):
    return CAM_VIEW_MAP.get(str(v).strip(), 0)

def _enc_road(v):
    return ROAD_TOPO_MAP.get(str(v).strip(), 0)

def _enc_vis(v):
    return VIS_MAP.get(str(v).strip().lower(), 0)

def _enc_basis(v):
    return BASIS_MAP.get(str(v).strip().lower(), 0)

def _enc_ab(v):
    s = str(v).strip().upper()
    if s == "A":
        return 1
    if s == "B":
        return 2
    return 0

def _enc_src(v):
    s = str(v).strip()
    if s in SRC_MAP:
        return SRC_MAP[s]
    # fallback (부분 문자열 대응)
    sl = s.lower()
    if "eunseok" in sl:
        return 1
    if "hyeong" in sl:
        return 2
    if "integrated" in sl:
        return 3
    if "agreement_rank_1" in sl:
        return 11
    if "agreement_rank_2" in sl:
        return 12
    if "agreement_rank_3" in sl:
        return 13
    return -1

def _enc_section(v):
    """
    section code 예시:
      1 = section1/agreement-both
      2 = section2/eunseok 우세
      3 = section3/hyeongseon 우세
      4 = section4/third-answer
      0 = unknown
    """
    s = str(v).strip().lower()
    if ("section" in s and "1" in s) or ("agreement" in s and "1" in s):
        return 1
    if ("section" in s and "2" in s) or ("eunseok" in s) or ("은석" in s):
        return 2
    if ("section" in s and "3" in s) or ("hyeong" in s) or ("형선" in s):
        return 3
    if ("section" in s and "4" in s):
        return 4
    return 0

def _parse_code_any(v):
    """
    target_code_combination이 dict 또는 문자열일 수 있으므로 둘 다 처리
    반환: (p, f, a, b)
    """
    if isinstance(v, dict):
        return (
            _safe_int(v.get("place"), None),
            _safe_int(v.get("feature"), None),
            _safe_int(v.get("vehicle_a"), None),
            _safe_int(v.get("vehicle_b"), None),
        )
    # 문자열 "(1, 11, 31, 34)" 같은 형태 대응
    nums = re.findall(r"\d+", str(v))
    if len(nums) >= 4:
        return tuple(int(x) for x in nums[:4])
    return (None, None, None, None)

def _contra_bits(axes):
    """
    place=1, feature=2, maneuver=4, role=8
    """
    if not isinstance(axes, list):
        axes = []
    bits = 0
    for a in axes:
        t = str(a).strip().lower()
        if t == "place":
            bits |= 1
        elif t == "feature":
            bits |= 2
        elif t == "maneuver":
            bits |= 4
        elif t == "role":
            bits |= 8
    return bits

def _count_vis_basis(vis_dict, basis_dict):
    # visibility counts
    vis_dict = vis_dict if isinstance(vis_dict, dict) else {}
    basis_dict = basis_dict if isinstance(basis_dict, dict) else {}

    vis_vals = [str(vis_dict.get(k, "")).strip().lower() for k in ["place", "feature", "maneuver", "role"]]
    basis_vals = [str(basis_dict.get(k, "")).strip().lower() for k in ["place", "feature", "maneuver", "role"]]

    vis_clear_cnt = sum(1 for x in vis_vals if x == "clear")
    vis_partial_cnt = sum(1 for x in vis_vals if x == "partial")
    vis_occ_cnt = sum(1 for x in vis_vals if x == "occluded")
    vis_unknown_cnt = sum(1 for x in vis_vals if x == "unknown" or x == "")

    basis_direct_cnt = sum(1 for x in basis_vals if x == "direct_visual")
    basis_partial_cnt = sum(1 for x in basis_vals if x == "partial_visual")
    basis_weak_cnt = sum(1 for x in basis_vals if x == "weak_inference")
    basis_unknown_cnt = sum(1 for x in basis_vals if x == "unknown" or x == "")

    return {
        "vis_clear_cnt": vis_clear_cnt,
        "vis_partial_cnt": vis_partial_cnt,
        "vis_occ_cnt": vis_occ_cnt,
        "vis_unknown_cnt": vis_unknown_cnt,
        "basis_direct_cnt": basis_direct_cnt,
        "basis_partial_cnt": basis_partial_cnt,
        "basis_weak_cnt": basis_weak_cnt,
        "basis_unknown_cnt": basis_unknown_cnt,
    }

def _counter_counts(counter_evidence):
    """
    counter_evidence 타입별 개수 (place/feature/maneuver/role/None)
    """
    ce = counter_evidence if isinstance(counter_evidence, list) else []
    out = {"ctr_cnt": 0, "ctr_place_cnt": 0, "ctr_feature_cnt": 0, "ctr_maneuver_cnt": 0, "ctr_role_cnt": 0, "ctr_none_cnt": 0}
    for item in ce:
        if not isinstance(item, dict):
            continue
        out["ctr_cnt"] += 1
        t = str(item.get("type", "")).strip().lower()
        if t == "place":
            out["ctr_place_cnt"] += 1
        elif t == "feature":
            out["ctr_feature_cnt"] += 1
        elif t == "maneuver":
            out["ctr_maneuver_cnt"] += 1
        elif t == "role":
            out["ctr_role_cnt"] += 1
        else:
            out["ctr_none_cnt"] += 1
    return out

def _argmax_hid_by_sum(h_rows, n_cands):
    """
    h_rows: {1:{...}, 2:{...}, 3:{...}}
    반환:
      top1_idx, top2_idx, top1_sum, top2_sum
    tie-break: sum desc -> hard asc -> idx asc
    """
    items = []
    for i in range(1, n_cands + 1):
        r = h_rows.get(i, {})
        s = _safe_int(r.get("sum16"), -1)
        hard = _safe_int(r.get("hard"), 0)
        items.append((i, s, hard))

    if not items:
        return (0, 0, -1, -1)

    items_sorted = sorted(items, key=lambda x: (-x[1], x[2], x[0]))
    top1 = items_sorted[0]
    top2 = items_sorted[1] if len(items_sorted) >= 2 else (0, -1, 0)
    return (top1[0], top2[0], top1[1], top2[1])

def _pack_h(i, pruned_df, h_score_map, g_p, g_f, g_a, g_b):
    prefix = f"h{i}_"

    # 후보가 없는 경우 (2개 후보 구간 대비)
    if len(pruned_df) < i:
        flat = {
            prefix + "valid": 0, prefix + "src": 0,
            prefix + "code_p": -1, prefix + "code_f": -1, prefix + "code_a": -1, prefix + "code_b": -1,
            prefix + "p": -1, prefix + "f": -1, prefix + "m": -1, prefix + "r": -1, prefix + "sum16": -1,
            prefix + "hard": 0,
            prefix + "contra_bits": 0, prefix + "contra_p": 0, prefix + "contra_f": 0, prefix + "contra_m": 0, prefix + "contra_r": 0,
            prefix + "vis_p": 0, prefix + "vis_f": 0, prefix + "vis_m": 0, prefix + "vis_r": 0,
            prefix + "basis_p": 0, prefix + "basis_f": 0, prefix + "basis_m": 0, prefix + "basis_r": 0,
            prefix + "clear_cnt": 0, prefix + "partial_cnt": 0, prefix + "occ_cnt": 0, prefix + "vis_unk_cnt": 0,
            prefix + "direct_cnt": 0, prefix + "basis_partial_cnt": 0, prefix + "weak_cnt": 0, prefix + "basis_unk_cnt": 0,
            prefix + "ctr_cnt": 0, prefix + "ctr_p_cnt": 0, prefix + "ctr_f_cnt": 0, prefix + "ctr_m_cnt": 0, prefix + "ctr_r_cnt": 0,
            prefix + "p_match": 0, prefix + "f_match": 0, prefix + "a_match": 0, prefix + "b_match": 0,
            prefix + "ab_match": 0, prefix + "exact": 0,
        }
        meta = {"sum16": -1, "hard": 0, "p": -1, "f": -1, "m": -1, "r": -1, "clear_cnt": 0, "direct_cnt": 0, "weak_cnt": 0, "exact": 0}
        return flat, meta

    row = pruned_df.iloc[i-1]
    expected_hid = str(row.get("hypothesis_id", f"H{i}"))
    expected_code = row.get("target_code_combination", row.get("code_combination", ""))
    expected_source = str(row.get("source_tag", ""))

    h = h_score_map.get(expected_hid, {}) if isinstance(h_score_map.get(expected_hid, {}), dict) else {}

    # scores
    s = h.get("scores", {}) if isinstance(h.get("scores", {}), dict) else {}
    p = _safe_score_04(s.get("place_score", -1))
    f = _safe_score_04(s.get("feature_score", -1))
    m = _safe_score_04(s.get("maneuver_score", -1))
    r = _safe_score_04(s.get("role_score", -1))
    sum16 = sum([x for x in [p, f, m, r] if x >= 0]) if any(x >= 0 for x in [p, f, m, r]) else -1

    # code parse
    code_src = expected_code if str(expected_code).strip() != "" else h.get("target_code_combination", "")
    hp, hf, ha, hb = _parse_code_any(code_src)

    # exact / matches
    p_match = 1 if (hp is not None and g_p is not None and hp == g_p) else 0
    f_match = 1 if (hf is not None and g_f is not None and hf == g_f) else 0
    a_match = 1 if (ha is not None and g_a is not None and ha == g_a) else 0
    b_match = 1 if (hb is not None and g_b is not None and hb == g_b) else 0
    ab_match = 1 if (a_match and b_match) else 0
    exact = 1 if (p_match and f_match and a_match and b_match) else 0

    # contradiction
    hard = 1 if bool(h.get("hard_contradiction", False)) else 0
    contra_bits = _contra_bits(h.get("contradiction_axes", []))

    # visibility / basis
    vis = h.get("visibility", {}) if isinstance(h.get("visibility", {}), dict) else {}
    basis = h.get("basis", {}) if isinstance(h.get("basis", {}), dict) else {}
    vb_cnts = _count_vis_basis(vis, basis)

    # counter evidence
    ce_cnts = _counter_counts(h.get("counter_evidence", []))

    # source
    src_val = str(h.get("source_tag", "")).strip() or expected_source
    src_code = _enc_src(src_val)

    flat = {
        prefix + "valid": 1, prefix + "src": src_code,
        prefix + "code_p": hp if hp is not None else -1,
        prefix + "code_f": hf if hf is not None else -1,
        prefix + "code_a": ha if ha is not None else -1,
        prefix + "code_b": hb if hb is not None else -1,

        prefix + "p": p, prefix + "f": f, prefix + "m": m, prefix + "r": r, prefix + "sum16": sum16,

        prefix + "hard": hard,
        prefix + "contra_bits": contra_bits,
        prefix + "contra_p": 1 if (contra_bits & 1) else 0,
        prefix + "contra_f": 1 if (contra_bits & 2) else 0,
        prefix + "contra_m": 1 if (contra_bits & 4) else 0,
        prefix + "contra_r": 1 if (contra_bits & 8) else 0,

        prefix + "vis_p": _enc_vis(vis.get("place", "")),
        prefix + "vis_f": _enc_vis(vis.get("feature", "")),
        prefix + "vis_m": _enc_vis(vis.get("maneuver", "")),
        prefix + "vis_r": _enc_vis(vis.get("role", "")),

        prefix + "basis_p": _enc_basis(basis.get("place", "")),
        prefix + "basis_f": _enc_basis(basis.get("feature", "")),
        prefix + "basis_m": _enc_basis(basis.get("maneuver", "")),
        prefix + "basis_r": _enc_basis(basis.get("role", "")),

        prefix + "clear_cnt": vb_cnts["vis_clear_cnt"],
        prefix + "partial_cnt": vb_cnts["vis_partial_cnt"],
        prefix + "occ_cnt": vb_cnts["vis_occ_cnt"],
        prefix + "vis_unk_cnt": vb_cnts["vis_unknown_cnt"],

        prefix + "direct_cnt": vb_cnts["basis_direct_cnt"],
        prefix + "basis_partial_cnt": vb_cnts["basis_partial_cnt"],
        prefix + "weak_cnt": vb_cnts["basis_weak_cnt"],
        prefix + "basis_unk_cnt": vb_cnts["basis_unknown_cnt"],

        prefix + "ctr_cnt": ce_cnts["ctr_cnt"],
        prefix + "ctr_p_cnt": ce_cnts["ctr_place_cnt"],
        prefix + "ctr_f_cnt": ce_cnts["ctr_feature_cnt"],
        prefix + "ctr_m_cnt": ce_cnts["ctr_maneuver_cnt"],
        prefix + "ctr_r_cnt": ce_cnts["ctr_role_cnt"],

        prefix + "p_match": p_match, prefix + "f_match": f_match,
        prefix + "a_match": a_match, prefix + "b_match": b_match,
        prefix + "ab_match": ab_match, prefix + "exact": exact,
    }

    meta = {
        "sum16": sum16, "hard": hard, "p": p, "f": f, "m": m, "r": r,
        "clear_cnt": vb_cnts["vis_clear_cnt"], "direct_cnt": vb_cnts["basis_direct_cnt"], "weak_cnt": vb_cnts["basis_weak_cnt"],
        "exact": exact
    }
    return flat, meta

# 결과 파일에 한 줄씩 쓰기 위한 헬퍼 함수
def save_result_to_csv(result_dict, file_path):
    file_exists = os.path.exists(file_path)
    # 딕셔너리의 키를 컬럼명으로 사용
    fieldnames = list(result_dict.keys())

    with open(file_path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 파일이 처음 생성되는 경우에만 헤더 작성
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)

def get_processed_videos(file_path):
    processed = set()
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if '파일명' in df.columns:
                processed = set(df['파일명'].astype(str).tolist())
        except Exception as e:
            print(f"⚠️ 기존 파일 로드 중 오류(무시하고 진행): {e}")
    return processed

def get_top_10_from_csv(file_path):
    """
    CSV의 은석/형선 개별 섹션을 파싱하여 독립적인 합치 여부를 판별하고,
    통합 분석 결과 섹션에서 Top-10 후보 리스트와 (P, F, A, B) 형식의 GT를 생성합니다.
    """
    # 인코딩 대응 (BOM 및 CP949)
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    except:
        with open(file_path, 'r', encoding='cp949') as f:
            lines = f.readlines()

    # 1. 각 섹션의 시작 위치 파악
    eun_start, hye_start, total_start = -1, -1, -1
    for i, line in enumerate(lines):
        if "### 은석_pred 결과 ###" in line: eun_start = i + 1
        elif "### 형선_pred 결과 ###" in line: hye_start = i + 1
        elif "### 통합_분석 결과 ###" in line: total_start = i + 1

    def get_top1_scenario_id(start_idx):
        """해당 섹션의 최상단(1순위) 시나리오 조합 ID를 문자열로 반환"""
        if start_idx == -1: return None
        section_lines = []
        for line in lines[start_idx:]:
            if line.startswith("###") or not line.strip(): break
            section_lines.append(line)

        if not section_lines: return None

        tmp_df = pd.read_csv(io.StringIO("".join(section_lines)))
        if tmp_df.empty: return None

        top = tmp_df.iloc[0]
        return f"{int(top['accident_place'])}_{int(top['accident_place_feature'])}_{int(top['vehicle_a_progress_info'])}_{int(top['vehicle_b_progress_info'])}"

    # 2. 전문가별 독립적 Top-1 추출 및 합치 여부 판별
    eun_top1_id = get_top1_scenario_id(eun_start)
    hye_top1_id = get_top1_scenario_id(hye_start)
    total_top1_id = get_top1_scenario_id(total_start)
    if eun_top1_id == hye_top1_id == total_top1_id and total_top1_id is not None:
        # 양쪽 전문가와 통합 결과가 모두 일치 (1구간)
        is_agreement = "Agreement"
    elif eun_top1_id == total_top1_id and total_top1_id is not None:
        # 은석 모델의 1위가 통합 1위인 경우 (2구간)
        is_agreement = "Eunseok"
    elif hye_top1_id == total_top1_id and total_top1_id is not None:
        # 형선 모델의 1위가 통합 1위인 경우 (3구간)
        is_agreement = "Hyeongseon"
    else:
        # 전문가 간의 의견이 완전히 갈리거나 통합 결과가 제3의 안인 경우 (4구간)
        is_agreement = "Disagreement"

    # 3. 통합 분석 결과 테이블 파싱
    if total_start != -1:
        total_lines = [line for line in lines[total_start:] if not line.strip().startswith("###")]
        df = pd.read_csv(io.StringIO("".join(total_lines)))
        df.columns = df.columns.str.strip()
    else:
        print(f"⚠️ 통합 분석 섹션을 찾을 수 없음: {file_path}")
        return "Unknown", "[]", "Unknown"

    # 데이터 타입 정수형 통일
    res_id_keys = ['accident_place', 'accident_place_feature', 'vehicle_a_progress_info', 'vehicle_b_progress_info']
    for k in res_id_keys:
        df[k] = pd.to_numeric(df[k], errors='coerce').fillna(-1).astype(int)

    # [수정] gt_rank 대신 (P, F, A, B) 형식의 gt_str 추출
    gt_str = "Unknown"
    # lines[1]은 컬럼명(accident_place...), lines[2]는 실제 값
    if len(lines) > 2 and "accident_place" in lines[1]:
        gt_header = lines[1].strip()
        gt_values = lines[2].strip()
        gt_df = pd.read_csv(io.StringIO(f"{gt_header}\n{gt_values}"))
        if not gt_df.empty:
            row = gt_df.iloc[0]
            gt_str = f"({int(row['accident_place'])}, {int(row['accident_place_feature'])}, {int(row['vehicle_a_progress_info'])}, {int(row['vehicle_b_progress_info'])})"

    # 통합 테이블 내에서의 모델별 최댓값 위치 찾기 (추천 태그용)
    eunseok_best_idx = df['은석_score'].idxmax() if '은석_score' in df.columns else -1
    hyeongseon_best_idx = df['형선_score'].idxmax() if '형선_score' in df.columns else -1

    # 4. 마스터 가이드 명칭 결합 (Top-10로 제한)
    top_10 = df.head(10).copy()
    if 'master_df' in globals() and master_df is not None:
        top_10 = pd.merge(top_10, master_df,
                          left_on=res_id_keys,
                          right_on=['사고장소_ID', '사고장소특징_ID', 'A진행방향_ID', 'B진행방향_ID'],
                          how='left')

    # 추천 태그 생성
    def get_tag(idx):
        tags = []
        if idx == eunseok_best_idx: tags.append("Eunseok Top-1")
        if idx == hyeongseon_best_idx: tags.append("Hyeongseon Top-1")
        return ", ".join(tags)

    top_10['recommendation'] = [get_tag(i) for i in top_10.index]
    top_10 = top_10.fillna("")

    # VLM 전달용 최종 컬럼 구성 (P, F, A, B 코드 정보 포함)
    top_10['code_combination'] = top_10.apply(lambda r: f"({int(r['accident_place'])}, {int(r['accident_place_feature'])}, {int(r['vehicle_a_progress_info'])}, {int(r['vehicle_b_progress_info'])})", axis=1)

    res_df = top_10.rename(columns={
        '순위': 'Rank', '사고장소': 'place', '사고장소특징': 'feature',
        'A진행방향': 'veh_a', 'B진행방향': 'veh_b'
    })

    final_cols = ['Rank', 'code_combination', 'place', 'feature', 'veh_a', 'veh_b', 'recommendation']

    return gt_str, res_df[final_cols], is_agreement
'''
def find_file_paths(video_stem):
    """파일명으로 실제 물리 경로(Video, Label, CSV)를 찾아 반환"""
    for label_env, video_env in label_env_name_mapping.items():
        v_path = os.path.join(base_video_root, video_env, f"{video_stem}.mp4")
        if os.path.exists(v_path):
            l_path = os.path.join(base_label_root, label_env, f"{video_stem}.json")
            c_path = os.path.join(base_csv_root, label_env, f"{video_stem}.csv")
            return v_path, l_path, c_path
    return None, None, None
'''
def make_json(pred_str, mapping_path):
    """
    pred_str (예: "(1, 11, 31, 34)")을 입력받아
    CSV 파일의 ID 컬럼들과 매칭되는 행을 찾아 JSON 딕셔너리로 반환합니다.
    """
    # 1. pred_str 문자열 파싱 (예: "(1, 11, 31, 34)" -> 1, 11, 31, 34)
    try:
        # ast.literal_eval을 사용하면 괄호와 쉼표가 포함된 문자열을 튜플로 안전하게 변환합니다.
        p_id, f_id, a_id, b_id = ast.literal_eval(pred_str)
    except Exception as e:
        print(f"Error parsing pred_str: {e}")
        return None

    # 2. CSV 파일 로드 (인코딩은 상황에 맞게 조정 가능)
    try:
        df = pd.read_csv(mapping_path, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(mapping_path, encoding='utf-8')

    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()

    # 3. ID 조건에 맞는 행 필터링
    condition = (
        (df['사고장소_ID'] == p_id) &
        (df['사고장소특징_ID'] == f_id) &
        (df['A진행방향_ID'] == a_id) &
        (df['B진행방향_ID'] == b_id)
    )

    match = df[condition]

    if match.empty:
        print(f"해당 조합({pred_str})에 일치하는 데이터를 찾을 수 없습니다.")
        return None

    # 매칭된 첫 번째 행 데이터 추출
    row = match.iloc[0]

    # 각 항목의 명칭 (문자열 앞뒤 공백 제거)
    place = str(row['사고장소']).strip()
    feature = str(row['사고장소특징']).strip()
    a_action = str(row['A진행방향']).strip()
    b_action = str(row['B진행방향']).strip()

    # 4. JSON 형태의 딕셔너리 구성
    selected_explanation_case_json = {
        "accident_type_name": f"{place}, {feature}, {a_action}, {b_action}",
        "target_code_combination": pred_str,
        "place_name": place,
        "feature_name": feature,
        "a_vehicle_action": a_action,
        "b_vehicle_action": b_action,
        "negligence_ratio_a": int(row['과실비율A']),
        "negligence_ratio_b": int(row['과실비율B'])
    }

    return selected_explanation_case_json


genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
#current_model_name = "gemini-3.1-pro-preview"
current_model_name = "gemini-3-flash-preview"
model_scorer = genai.GenerativeModel(model_name=current_model_name, system_instruction=system_instruction_score_only)
model_analyzer = genai.GenerativeModel(model_name=current_model_name, system_instruction=system_instruction_explanation_direct)

#매핑 정의
label_env_name_mapping = {
    "roundabout_label": "VS_차대차_영상_회전교차로",
    "4way_signal_label": "VS_차대차_영상_사거리교차로(신호등있음)",
    "road_and_other_label": "VS_차대차_영상_차도와차도가아닌장소",
    "4way_no_signal_label": "VS_차대차_영상_사거리교차로(신호등없음)",
    "highway_label": "VS_차대차_영상_고속도로(자동차전용도로)포함",
    "parking_lot_label": "VS_차대차_영상_주차장(또는차도가아닌장소)",
    "t_junction_label": "VS_차대차_영상_T자형교차로",
    "straight_road_label": "VS_차대차_영상_직선도로",
}

def run_explan_test(video_stem, idx, video_file, pred_str, gt_str=""):
    selected_explanation_case_json = make_json(pred_str, mapping_path)

    prompt_explanation_direct = f"""
    아래는 교통사고 블랙박스 영상에 대해 파이썬 후처리로 이미 확정된 사고유형 정보입니다.
    당신의 역할은 이 확정된 유형을 바탕으로, 영상을 참고해 사용자에게 읽기 쉬운 설명을 작성하는 것입니다.

    [중요]
    - 최종 유형은 이미 확정되었습니다. 변경할 수 없습니다.
    - 후보 비교, 재채점, 재선택을 하지 마십시오.
    - 반드시 'A 차량', 'B 차량'으로만 서술하십시오.
    - '내 차량', '블박 차량', '상대 차량', '가해차량', '피해차량' 표현은 사용하지 마십시오.
    - 영상에서 불확실한 내용은 단정하지 말고 '확인 어려움'으로 작성하십시오.

    [확정된 유형 입력(JSON)]
    {selected_explanation_case_json}

    [출력]
    {output_format_explanation_direct}
    """
    print(f"\n🚀 [설명 시작] {video_stem}")
    try:
        max_retries = 3
        attempt = 0
        response = None
        while attempt < max_retries:
            try:
                print("hi")
                response = model_analyzer.generate_content([prompt_explanation_direct, video_file])
                print("bye")
                break # 성공 시 루프 탈출

            except (http.client.RemoteDisconnected, Exception) as e:
                if ("429" in str(e) or "Quota" in str(e)):
                    print(f"🚨 할당량 초과! {current_model_name}의 할당량을 초과했습니다.")
                    continue

                attempt += 1
                print(f"⚠️ {attempt}차 시도 중 오류 발생: {str(e)}")
                time.sleep(10)

                if attempt >= max_retries:
                    print(f"❌ 최종 실패: {video_stem}")
                    raise e

        vlm_text = response.text

        # 1. 마크다운 기호 제거 및 JSON 파싱 (LLM이 ```json ... ``` 형태로 출력할 경우 대비)
        clean_json_str = re.sub(r"```json\s*", "", vlm_text)
        clean_json_str = re.sub(r"```\s*$", "", clean_json_str).strip()

        vlm_json = json.loads(clean_json_str)

        # [추가된 로직] 만약 결과가 리스트 형식이면 첫 번째 요소(딕셔너리)를 선택
        if isinstance(vlm_json, list):
            if len(vlm_json) > 0:
                vlm_json = vlm_json[0]
            else:
                raise ValueError("Empty JSON list received")

        # 중첩된 JSON 구조에서 필요한 값들을 추출하여 평탄화(Flatten)합니다.
        obs = vlm_json.get("video_observation", {})
        scene = obs.get("scene_condition", {})
        road = obs.get("road_context", {})
        movement = obs.get("movement_observation", {})

        # 불확실성 리스트를 문자열로 변환
        uncertainties = ", ".join(vlm_json.get("uncertainties", []))

        # 2. 결과 리스트에 append (딕셔너리 형태)
        # selected_explanation_case_json에 있는 정보와 VLM이 생성한 상세 분석을 통합합니다.
        results_exp_list=[]
        results_exp_list.append({
            "video_stem": video_stem,  # 파일명 등 식별자

            # Mapping 데이터 (이전 단계에서 만든 json 데이터 활용)
            #"gt_code_combination": gt_str,
            "target_code_combination": selected_explanation_case_json.get("target_code_combination"),
            "accident_type_name": selected_explanation_case_json.get("accident_type_name"),
            "negligence_ratio_a": selected_explanation_case_json.get("negligence_ratio_a"),
            "negligence_ratio_b": selected_explanation_case_json.get("negligence_ratio_b"),

            # VLM 상세 관찰 데이터 (JSON 파싱 결과)
            "time_of_day": scene.get("time_of_day"),
            "weather": scene.get("weather"),
            "visibility_note": scene.get("visibility_note"),
            "intersection_type": road.get("intersection_type_observed"),
            "signal_observed": road.get("signal_observed"),
            "road_scale_hint": road.get("road_scale_hint"),
            "a_vehicle_observation": movement.get("a_vehicle_observation"),
            "b_vehicle_observation": movement.get("b_vehicle_observation"),
            "collision_moment": movement.get("collision_moment"),
            "uncertainties": uncertainties,

            # 최종 설명 문구
            "explanation_text": vlm_json.get("explanation_text")
        })
    except Exception as e:
        print(str(e))
        print("Why")
        return False
    return vlm_json.get("explanation_text")

def run_score_test(video_stem, idx, video_file, c_path):
    #global model_scorer, current_model_name

    # 데이터 준비
    gt_str, candidates_df, is_agreement = get_top_10_from_csv(c_path)
    gt_str = ""
    if is_agreement == "Agreement":
        return False, candidates_df['code_combination'].iloc[0], gt_str
        pruned_df = candidates_df[candidates_df['Rank'].isin([1, 2, 3])]
    elif is_agreement == "Eunseok":
        pruned_df = candidates_df[candidates_df['recommendation'].str.contains("Top-1", na=False)]
    elif is_agreement == "Hyeongseon":
        #current_section_instruction = common_instruction + "\n" + section_3_instruction_6_2
        pruned_df = candidates_df[candidates_df['recommendation'].str.contains("Top-1", na=False)]
    elif is_agreement == "Disagreement":
        pruned_df = candidates_df[(candidates_df['recommendation'].str.contains("Top-1", na=False)) | (candidates_df['Rank'] == 1)]
        # 동일한 코드가 중복 선택되는 것을 방지
        pruned_df = pruned_df.drop_duplicates(subset=['code_combination'])

    print(f"\n🚀 [분석 시작] {video_stem}")

    # --- [입력 JSON 강화 로직 추가] ---
    pruned_df = pruned_df.reset_index(drop=True)

    # 1. hypothesis_id 부여 (H1, H2, H3)
    pruned_df['hypothesis_id'] = [f"H{i+1}" for i in range(len(pruned_df))]


    def extract_code_tuple(s):
        nums = re.findall(r'\d+', str(s))
        return tuple(map(int, nums[:4])) if len(nums) >= 4 else None

    #gt_tuple = extract_code_tuple(gt_str)
    #is_gt_in_candidates = any(extract_code_tuple(c) == gt_tuple for c in pruned_df['code_combination'])

    # 2. source_tag 생성 (추천 정보를 기반으로)
    def get_source_tag(row):
        rec = str(row.get('recommendation', ''))
        rank = row.get('Rank', None)

        if is_agreement == "Agreement":
            if rank == 1:
                return "Agreement_Rank_1"
            elif rank == 2:
                return "Agreement_Rank_2"
            elif rank == 3:
                return "Agreement_Rank_3"
            return "Agreement_Rank_3"
        if "Eunseok" in rec: return "Eunseok"
        if "Hyeongseon" in rec: return "Hyeongseon"
        return "Integrated" # Rank 1 등
    pruned_df['source_tag'] = pruned_df.apply(get_source_tag, axis=1)

    # 3. 필드명 매핑 및 규격화 (target 생성)
    pruned_df['target_code_combination'] = pruned_df['code_combination']
    pruned_df['target'] = pruned_df.apply(
        lambda r: f"{r['code_combination']}: ({r['place']}, {r['feature']}, {r['veh_a']}, {r['veh_b']})", axis=1
    )

    # 4. VLM에 전달할 컬럼만 추출하여 JSON 변환
    vlm_input_cols = ['hypothesis_id', 'source_tag', 'target_code_combination', 'target', 'place', 'feature', 'veh_a', 'veh_b']
    selected_candidates_json = pruned_df[vlm_input_cols].to_json(orient='records', force_ascii=False)

    # 5. 후처리를 위한 ID 매핑 테이블 미리 생성
    id_to_numeric_map = dict(zip(pruned_df['hypothesis_id'], pruned_df['target_code_combination']))

    prompt_score_only = f"""
    아래는 교통사고 블랙박스 영상에 대한 후보 가설 목록입니다.
    이번 작업은 최종 선택이 아니라, 각 후보의 시각적 일치도 채점(score-only)입니다.

    [입력값 유지 규칙]
    - hypothesis_id, target_code_combination, target, source_tag는 입력값을 그대로 유지하십시오. 임의 수정 금지.
    - hypothesis_scoring에는 입력된 모든 후보를 빠짐없이 포함하십시오.

    [실행 지시]
    - 모든 후보를 같은 기준으로 채점하십시오.
    - 각 후보마다 counter_evidence를 최소 1개 작성하십시오.
    (반증이 없으면 {{"time":"None","type":"None","detail":"None"}} 사용)
    - evidence / environment_cues / counter_evidence / axis_comparison.notes 배열은 각각 최대 3개까지만 작성하십시오.

    [추가 관찰 기준: Place 계층 판단]
    - 아래 지침은 place/feature 관찰을 정리하기 위한 참고 기준입니다.
    - hard-rule로 강제하지 말고, 영상에서 실제로 보이는 단서를 우선하십시오.
    {place_hierarchy_instruction}

    [후보 가설(JSON)]
    {selected_candidates_json}

    [출력]
    {output_format_score_only}
    """

    if selected_candidates_json=="[]" or selected_candidates_json == "{}":
        print("csv 파싱 오류")
        return False, "(-1,-1,-1,-1)", gt_str
    try:
        max_retries = 3
        attempt = 0
        response = None
        while attempt < max_retries:
            try:
                response = model_scorer.generate_content([prompt_score_only, video_file])
                break # 성공 시 루프 탈출

            except (http.client.RemoteDisconnected, Exception) as e:
                if ("429" in str(e) or "Quota" in str(e)):
                    print(f"🚨 할당량 초과! {current_model_name}의 할당량을 초과했습니다.")
                    continue

                attempt += 1
                print(f"⚠️ {attempt}차 시도 중 오류 발생: {str(e)}")
                time.sleep(10)

                if attempt >= max_retries:
                    print(f"❌ 최종 실패: {video_stem}")
                    raise e

        vlm_text = response.text

        # 1. 마크다운 기호 제거 및 JSON 파싱 (LLM이 ```json ... ``` 형태로 출력할 경우 대비)
        clean_json_str = re.sub(r"```json\s*", "", vlm_text)
        clean_json_str = re.sub(r"```\s*$", "", clean_json_str).strip()

        vlm_json = json.loads(clean_json_str)

        # [추가된 로직] 만약 결과가 리스트 형식이면 첫 번째 요소(딕셔너리)를 선택
        if isinstance(vlm_json, list):
            if len(vlm_json) > 0:
                vlm_json = vlm_json[0]
            else:
                raise ValueError("Empty JSON list received")

        # score-only에서는 hypothesis_scoring만 사용
        h_scores = vlm_json.get("hypothesis_scoring", [])
        if not isinstance(h_scores, list):
            h_scores = []

        # hypothesis_id 기준으로 빠르게 찾기 위한 dict
        h_score_map = {}
        for h in h_scores:
            h_id = str(h.get("hypothesis_id", "")).strip()
            if h_id:
                h_score_map[h_id] = h

        # 후보/GT 파싱용 함수
        def parse_code(code_str):
            nums = re.findall(r'\d+', str(code_str))
            return [int(n) for n in nums] if len(nums) >= 4 else [None, None, None, None]

        # GT 파싱
        #g_p, g_f, g_a, g_b = parse_code(gt_str)

        # score-only 단계에서는 최종 선택 없음
        # 대신 후보별 점수만 저장하고, 나중에 파이썬에서 후처리 가능하도록 충분히 기록
        h_data = {}

        # pruned_df 기준으로 기대 후보(H1/H2/H3) 순서대로 저장 (VLM 순서 꼬여도 안전)
        for i in range(1, 4):
            if len(pruned_df) >= i:
                row = pruned_df.iloc[i-1]
                expected_hid = str(row.get("hypothesis_id", f"H{i}"))
                expected_code = str(row.get("target_code_combination", row.get("code_combination", "")))
                expected_target = str(row.get("target", ""))
                expected_source = str(row.get("source_tag", ""))

                h = h_score_map.get(expected_hid, {})

                s = h.get("scores", {}) if isinstance(h.get("scores", {}), dict) else {}
                sr = h.get("score_reasons", {}) if isinstance(h.get("score_reasons", {}), dict) else {}

                vis = h.get("visibility", {}) if isinstance(h.get("visibility", {}), dict) else {}
                basis = h.get("basis", {}) if isinstance(h.get("basis", {}), dict) else {}

                contr_axes = h.get("contradiction_axes", [])
                if not isinstance(contr_axes, list):
                    contr_axes = []
                contr_axes_str = ",".join([str(x) for x in contr_axes])

                def _safe_score(v):
                    try:
                        return int(float(v))
                    except:
                        return -1

                p_score = _safe_score(s.get("place_score", -1))
                f_score = _safe_score(s.get("feature_score", -1))
                m_score = _safe_score(s.get("maneuver_score", -1))
                r_score = _safe_score(s.get("role_score", -1))

                # 반증 문자열 직렬화
                ce_list = h.get("counter_evidence", [])
                if not isinstance(ce_list, list):
                    ce_list = []
                ce_str = " | ".join([
                    f"[{ce.get('time','')}/{ce.get('type','')}] {ce.get('detail','')}"
                    for ce in ce_list if isinstance(ce, dict)
                ]) if ce_list else "[None/None] None"

                # score-only 단계용 합계 (후처리 전 임시 분석용)
                raw_sum = sum([x for x in [p_score, f_score, m_score, r_score] if isinstance(x, int) and x >= 0])

                h_data[f"가설{i}_ID"] = expected_hid
                h_data[f"가설{i}_입력코드"] = expected_code
                h_data[f"가설{i}_입력타겟"] = expected_target
                h_data[f"가설{i}_출처"] = expected_source

                # VLM이 target을 그대로 안 돌려줘도 입력 기준으로 보존
                h_data[f"가설{i}_VLM타겟"] = h.get("target", "")
                h_data[f"가설{i}_하드모순"] = h.get("hard_contradiction", False)

                h_data[f"가설{i}_점수_P"] = p_score
                h_data[f"가설{i}_점수_F"] = f_score
                h_data[f"가설{i}_점수_M"] = m_score
                h_data[f"가설{i}_점수_R"] = r_score
                h_data[f"가설{i}_점수합(0~16)"] = raw_sum

                h_data[f"가설{i}_근거_P"] = sr.get("place_reason", "")
                h_data[f"가설{i}_근거_F"] = sr.get("feature_reason", "")
                h_data[f"가설{i}_근거_M"] = sr.get("maneuver_reason", "")
                h_data[f"가설{i}_근거_R"] = sr.get("role_reason", "")

                h_data[f"가설{i}_반증"] = ce_str
                h_data[f"가설{i}_모순축"] = contr_axes_str

                h_data[f"가설{i}_가시성_P"] = vis.get("place", "")
                h_data[f"가설{i}_가시성_F"] = vis.get("feature", "")
                h_data[f"가설{i}_가시성_M"] = vis.get("maneuver", "")
                h_data[f"가설{i}_가시성_R"] = vis.get("role", "")

                h_data[f"가설{i}_근거강도_P"] = basis.get("place", "")
                h_data[f"가설{i}_근거강도_F"] = basis.get("feature", "")
                h_data[f"가설{i}_근거강도_M"] = basis.get("maneuver", "")
                h_data[f"가설{i}_근거강도_R"] = basis.get("role", "")
            else:
                # 후보가 2개인 구간 대비 빈칸 채우기
                h_data[f"가설{i}_ID"] = ""
                h_data[f"가설{i}_입력코드"] = ""
                h_data[f"가설{i}_입력타겟"] = ""
                h_data[f"가설{i}_출처"] = ""
                h_data[f"가설{i}_VLM타겟"] = ""
                h_data[f"가설{i}_하드모순"] = ""
                h_data[f"가설{i}_점수_P"] = ""
                h_data[f"가설{i}_점수_F"] = ""
                h_data[f"가설{i}_점수_M"] = ""
                h_data[f"가설{i}_점수_R"] = ""
                h_data[f"가설{i}_점수합(0~16)"] = ""
                h_data[f"가설{i}_근거_P"] = ""
                h_data[f"가설{i}_근거_F"] = ""
                h_data[f"가설{i}_근거_M"] = ""
                h_data[f"가설{i}_근거_R"] = ""
                h_data[f"가설{i}_반증"] = ""
                h_data[f"가설{i}_모순축"] = ""
                h_data[f"가설{i}_가시성_P"] = ""
                h_data[f"가설{i}_가시성_F"] = ""
                h_data[f"가설{i}_가시성_M"] = ""
                h_data[f"가설{i}_가시성_R"] = ""
                h_data[f"가설{i}_근거강도_P"] = ""
                h_data[f"가설{i}_근거강도_F"] = ""
                h_data[f"가설{i}_근거강도_M"] = ""
                h_data[f"가설{i}_근거강도_R"] = ""

        # score-only에서도 관찰 필드는 있을 수 있으니 있으면 저장 (없으면 빈값)
        visual_obs = vlm_json.get("visual_observation", {}) if isinstance(vlm_json.get("visual_observation", {}), dict) else {}
        role_id = vlm_json.get("role_identification", {}) if isinstance(vlm_json.get("role_identification", {}), dict) else {}
        pov_obs = vlm_json.get("pov_observation", {}) if isinstance(vlm_json.get("pov_observation", {}), dict) else {}

        ego_is = role_id.get("blackbox_is", "unknown")
        #gt_other = gt_ab_dict.get(video_stem, "Unknown").upper()
        #gt_ego_true = 'B' if 'A' in gt_other else ('A' if 'B' in gt_other else 'Unknown')
        ego_is_clean = 'A' if 'A' in str(ego_is).upper() else ('B' if 'B' in str(ego_is).upper() else 'Unknown')
        #is_ego_correct = "Pass" if (gt_ego_true == ego_is_clean and gt_ego_true != "Unknown") else "Fail"

        # 최종 선택/정답 적중은 아직 계산 안함 (후처리 전)
        results_list= []
        results_list.append({
            "파일명": video_stem,
            "구간유형": is_agreement,
            #"GT_코드": gt_str,

            # score-only 단계 상태
            "후보개수": len(pruned_df),
            #"후보내정답존재": "Yes" if is_gt_in_candidates else "No",

            # 관찰/보조 정보
            "카메라시점추정": pov_obs.get("camera_view", ""),
            "도로토폴로지추정": visual_obs.get("road_topology_guess", ""),
            "기동관찰_Ego": visual_obs.get("ego_maneuver_guess", ""),
            "기동관찰_Other": visual_obs.get("other_vehicle_maneuver_guess", ""),
            "충돌기하": visual_obs.get("collision_geometry", ""),
            #"GT_블박차량": gt_ego_true,
            "VLM_블박차량": ego_is_clean,
            #"블박식별_성공여부": is_ego_correct,

            # 추가 필드
            "카메라시점추정_신뢰도": pov_obs.get("confidence", ""),
            "관찰신뢰도": visual_obs.get("observation_confidence", ""),
            "역할매핑근거": role_id.get("mapping_reason", ""),
            "역할식별신뢰도": role_id.get("confidence", ""),

            # 디버깅용 원문/파싱 상태
            "파싱상태": "OK",
            "모델": current_model_name,

            # 가설별 상세
            **h_data
        })

        # --------------------------------------------------------
        # build compact candidates (H1/H2/H3)
        # --------------------------------------------------------
        h1_flat, h1m = _pack_h(1, pruned_df, h_score_map, g_p, g_f, g_a, g_b)
        h2_flat, h2m = _pack_h(2, pruned_df, h_score_map, g_p, g_f, g_a, g_b)
        h3_flat, h3m = _pack_h(3, pruned_df, h_score_map, g_p, g_f, g_a, g_b)

        # rank by sum (tie: hard 작은 후보 우선)
        _hrows_rank = {1: h1m, 2: h2m, 3: h3m}
        top1_idx, top2_idx, top1_sum, top2_sum = _argmax_hid_by_sum(_hrows_rank, int(len(pruned_df)))

        # small helper for delta
        def _delta(a, b, invalid=-99):
            return (a - b) if (a is not None and b is not None and a >= 0 and b >= 0) else invalid

        # --------------------------------------------------------
        # short analysis row append (main block stays compact)
        # --------------------------------------------------------
        results_short_list = []
        results_short_list.append({
            # meta
            "schema_ver": 2,
            "row_id": f"{video_stem}__{_enc_section(is_agreement)}",
            "video_id": video_stem,
            "section_code": _enc_section(is_agreement),   # 1/2/3/4/0
            "n_cands": int(len(pruned_df)),
            #"gt_in_cands": 1 if is_gt_in_candidates else 0,
            "parse_ok": 1,

            # GT split
            #"gt_p": g_p if g_p is not None else -1,
            #"gt_f": g_f if g_f is not None else -1,
            #"gt_a": g_a if g_a is not None else -1,
            #"gt_b": g_b if g_b is not None else -1,

            # observation / role (encoded)
            "cam_view": _enc_cam(pov_obs.get("camera_view", "불명")),
            "cam_conf": _enc_conf(pov_obs.get("confidence", "")),
            "road_topo": _enc_road(visual_obs.get("road_topology_guess", "불명")),
            "obs_conf": _enc_conf(visual_obs.get("observation_confidence", "")),
            "role_conf": _enc_conf(role_id.get("confidence", "")),
            #"ego_gt": _enc_ab(gt_ego_true),
            "ego_pred": _enc_ab(ego_is_clean),
            #"ego_pass": 1 if str(is_ego_correct).lower() == "pass" else 0,

            # flattened candidates
            **h1_flat, **h2_flat, **h3_flat,

            # baseline (H1)
            "base_idx": 1,
            "base_exact": h1m["exact"],

            # rank summary (pure score sum)
            "top1_idx": top1_idx,
            "top2_idx": top2_idx,
            "top1_sum16": top1_sum,
            "top2_sum16": top2_sum,
            "top12_margin_sum": (top1_sum - top2_sum) if (top1_sum >= 0 and top2_sum >= 0) else -99,
            "top1_exact": (h1m["exact"] if top1_idx == 1 else h2m["exact"] if top1_idx == 2 else h3m["exact"] if top1_idx == 3 else 0),
            "top2_exact": (h1m["exact"] if top2_idx == 1 else h2m["exact"] if top2_idx == 2 else h3m["exact"] if top2_idx == 3 else 0),

            # commonly used H2 vs H1 deltas (threshold search용)
            "h2m1_p": _delta(h2m["p"], h1m["p"]),
            "h2m1_f": _delta(h2m["f"], h1m["f"]),
            "h2m1_m": _delta(h2m["m"], h1m["m"]),
            "h2m1_r": _delta(h2m["r"], h1m["r"]),
            "h2m1_sum16": _delta(h2m["sum16"], h1m["sum16"]),
            "h2m1_clear_cnt": h2m["clear_cnt"] - h1m["clear_cnt"],
            "h2m1_direct_cnt": h2m["direct_cnt"] - h1m["direct_cnt"],
            "h2m1_weak_cnt": h2m["weak_cnt"] - h1m["weak_cnt"],

            # optional: H3 vs H1 / H3 vs H2도 자주 보면 추가
            "h3m1_sum16": _delta(h3m["sum16"], h1m["sum16"]),
            "h3m2_sum16": _delta(h3m["sum16"], h2m["sum16"]),

            "model": current_model_name,
        })

        #print(f"{idx}: [{video_stem}] 처리 완료 - GT: {gt_str}")#/ VLM: {vlm_codes_str} ({exact_match})")

    except KeyboardInterrupt:
        print(f"\n🛑 사용자에 의해 중단됨: {video_stem}")
        raise  # 루프 전체를 멈추려면 다시 raise

    except Exception as e:
        print(f"❌ 오류 ({video_stem}): {e}")
        return False, "(-1,-1,-1,-1)", gt_str

    final_pred_code = h_data.get(f"가설{top1_idx}_입력코드", "(-1,-1,-1,-1)")
    return True, final_pred_code, gt_str



'''
sec1_samples = ['bb_1_160910_vehicle_241_26225', 'bb_1_160116_vehicle_222_29116', 'bb_1_180222_vehicle_148_238', 'bb_1_160614_vehicle_112_113', 'bb_1_120318_vehicle_113_157', 'bb_1_220827_vehicle_256_50486', 'bb_1_170120_vehicle_233_22160', 'bb_1_150517_vehicle_37_150', 'bb_1_210210_vehicle_212_45839', 'bb_1_181104_vehicle_195_096']

target_samples = sec1_samples

output_csv_path = "/content/drive/MyDrive/260224_ai/4th_experiment_results_score.csv"
output_short_csv_path = "/content/drive/MyDrive/260224_ai/4th_experiment_results_score_short.csv"
output_exp_csv_path = "/content/drive/MyDrive/260224_ai/4th_experiment_results_exp.csv"
processed_in_this_env = get_processed_videos(output_exp_csv_path)
video_files = [v for v in target_samples if v not in processed_in_this_env] # 미처리 파일만 추출


input_c_path = "/content/input.csv"

print(f"{len(processed_in_this_env)}, {len(video_files)}")

for i, sample in enumerate(video_files):
    try:
        v_path, l_path, _ = find_file_paths(sample)

        process_single_json_to_csv(l_path, input_c_path, csv_type_path)

        if not input_c_path:
            print(f"❌ 파일을 찾을 수 없음: {sample}")
        else:
            video_file = genai.upload_file(path=v_path)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

        run, pred_str, gt_str = run_score_test(sample, i, video_file, input_c_path)

        if results_list and False:
            last_result = results_list[-1]
            save_result_to_csv(last_result, output_csv_path)
            last_result_short = results_short_list[-1]
            save_result_to_csv(last_result_short, output_short_csv_path)

        # API 호출 제한 방지를 위한 대기
        time.sleep(3)

        #pred_str = "(-1,-1,-1,-1)"
        run2 = run_explan_test(sample, i, video_file, pred_str, gt_str)

        if results_exp_list and run2:
            last_exp_result = results_exp_list[-1]
            save_result_to_csv(last_exp_result, output_exp_csv_path)

        # API 호출 제한 방지를 위한 대기
        
    except Exception as e:
        print(f"❌ {sample} 처리 중 예외 발생(건너뜀): {e}")
        continue

    finally:
        if 'video_file' in locals():
            try:
                video_file.delete()
            except Exception as e:
                print("비디오 삭제 실패")
'''

