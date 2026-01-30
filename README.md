# 🚗 Road Judge AI

### :oncoming_automobile: AI 기반 교통사고 과실 비율 분석 및 운전 습관 케어 서비스

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?logo=yolo&logoColor=black)
![MMAction2](https://img.shields.io/badge/MMAction2-blueviolet)
![Status](https://img.shields.io/badge/Status-In%20Progress-green)

<div align="center">
  <img src="https://via.placeholder.com/600x200?text=Road+Judge+AI+Logo" alt="Logo" width="100%">
  <br><br>
  <p>
    <b>"변호사보다 빠르게, 블랙박스보다 똑똑하게"</b><br>
    딥러닝 모델을 활용하여 블랙박스 영상을 분석하고, 객체 탐지 및 사고 상황 분류를 통해<br>
    <b>과실비율을 빠르고 객관적으로 평가</b>하는 AI 서비스입니다.
  </p>
</div>

<br>

## 📝 프로젝트 개요 (Project Overview)

**Road Judge AI**는 교통사고 발생 시 복잡하고 오래 걸리는 과실 비율 산정 과정을 AI를 통해 자동화하여 개인에게 1차적인 판단 근거를 제공합니다. 더불어 비사고 주행 영상을 분석하여 운전자의 습관을 점수화하고 개선점을 제안하는 2차 목표를 가지고 있습니다.

### 📅 개발 배경

- **분쟁의 증가**: 과실비율 분쟁 심의 청구 건수 매년 증가 (2019년 10.2만 건 → 2023년 13.6만 건)
- **느린 처리 속도**: 민사 소송 시 최소 4개월에서 최대 3년 이상 소요
- **비용 부담**: 변호사 선임 및 소송 비용 발생

---

## 🎯 목표 및 기대효과 (Goals & Effects)

|            구분             | 주요 내용                                                                                                                                                                      |
| :-------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1차 목표**<br>(과실 분석) | • 블랙박스 영상 기반 사고 유형 자동 분류<br>• 공식 '과실비율 인정기준'에 따른 비율 산출<br>• 사고 핵심 프레임 및 행동 근거 제시                                                |
| **2차 목표**<br>(습관 케어) | • 비사고 영상 입력을 통한 운전 습관 점수 산출<br>• 위험 이벤트(급차로 변경, 신호위반 등) 타임라인 제공<br>• 개인 맞춤형 운전 습관 피드백                                       |
|        **기대 효과**        | • **시간 단축**: 수개월의 심의 기간을 **1분 내외**로 단축<br>• **비용 절감**: 불필요한 법적 분쟁 및 상담 비용 절약<br>• **객관성 확보**: 데이터 기반의 정량적 판결 가이드 제공 |

---

## 💾 데이터셋 구조 (Dataset Structure)

본 프로젝트는 **AI-Hub 교통사고 영상 데이터**를 활용하여 학습을 진행합니다.

```bash
095.교통사고 영상 데이터
│
└── 01.데이터
    ├── 1.Training
    │   ├── 라벨링데이터
    │   │   ├── TL_차대차_영상_T자형교차로 (.json)
    │   │   └── ...
    │   └── 원천데이터
    │       ├── TL_차대차_영상_T자형교차로 (.mp4)
    │       └── ...
    │
    └── 2.Validation
        ├── 라벨링데이터
        └── 원천데이터
```
