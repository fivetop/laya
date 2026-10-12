---
language:
  - ko
library_name: laya
base_model: answerdotai/ModernBERT-large
tags:
  - trading
  - stock
  - decision-making
  - calibration
  - korean
  - modernbert
license: other
---

# laya-stock-model

[Laya](https://github.com/consai/laya) 결정 모델을 **한국 주식 10분봉** 데이터로
파인튜닝한 buy / hold / sell 의사결정 모델입니다. 30분(3봉) 후 수익률을 기준으로
분류된 라벨을 학습했습니다.

## 모델 개요

| 항목 | 값 |
|---|---|
| 베이스 인코더 | `answerdotai/ModernBERT-large` (hidden 1024, 28층, 16 heads) |
| 헤드 | Laya rl-agent (`head_layers=2`) |
| vocab / 최대길이 | 50,368 / `max_len` 512, `head_max_len` 192 |
| 캘리브레이션 | temperature 1.1709 (choice) |
| 가중치 | `model.safetensors` (약 803MB, fp32) |

## 학습 데이터

- **소스**: 한국 주식 실시간 틱 → 10분봉 집계. 일별 거래대금/거래량 **상위 50위 종목만** 저장된 모집단.
- **기간**: 2025-01-06 ~ 2025-07-23 (약 100 거래일)
- **표본**: train 85,473 / eval 27,725 (시간순 80:20 분할, 시계열 누수 없음)
- **입력**: state 피처 25개(가격·거래량·RSI·MACD·이동평균·당일 순위 등), 질문 `action` 1개, gold 3-class 확률
- **라벨**: 미래 3봉(30분) 수익률 → `buy`(≥ +0.5%) / `hold` / `sell`(≤ −0.5%)

## 학습 설정

- Kaggle T4 ×2, 4 epochs, `micro_batch` 8, `grad_accum` 8
- loss `rlcd`, `head_lr` 1e-4, `encoder_lr` 2.5e-5, `rl_samples` 4, `sigma` 0.4 → 0.1

## 평가 (held-out 27,725)

| 지표 | before | after |
|---|---|---|
| Accuracy | 0.3189 | **0.4648** |
| Loss | 1.1264 | 1.0832 |
| ECE | 0.0872 | **0.0646** |
| Brier | 0.1513 | 0.1213 |

## 사용법

```python
import laya

agent = laya.load("romee/laya-stock-model")            # HF Hub
# agent = laya.load(r"C:\0GitHub\라야\stock-model")     # 로컬 경로
res = agent.predict(state=state, questions=questions)  # buy/hold/sell + 확률
```

## 한계 및 주의

- **모집단 일치 필수**: 학습 모집단이 "상위 50위 종목"이므로, 다른 모집단에 적용하면
  보정된 확률(confidence)이 무의미해집니다.
- **상한가 체결 제약**: 현재가가 상한가면 신규 매수 체결이 불가하므로 buy 신호를
  무시하는 규칙이 필요합니다(보유 매도는 정상).
- **표본 한계**: 20거래일 백테스트에서 buy&hold 대비 초과수익(+)을 보였으나 기간이 짧습니다.

## 라이선스

저작자 지정 예정(`other`). 베이스 모델 `answerdotai/ModernBERT-large`는 Apache-2.0입니다.
