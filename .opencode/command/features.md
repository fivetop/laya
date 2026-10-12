---
description: 10분봉 → 피처 + 라벨 + laya 학습용 JSONL 생성
---

# 피처/라벨/JSONL 생성

`data/bars_10m.parquet`를 읽어 laya 파인튜닝용 JSONL을 만듭니다.

## 작업 순서

1. `pipeline/build_features.py` 생성:
   - 10분봉 per 종목 → RSI(14), MACD(12,26,9), SMA(5,20,60), 등락률, 거래량비(현재/20봉평균), 연속상승횟수
   - 과거 데이터만 사용 (state에는 현재 시점 이전 정보만)
2. `pipeline/build_labels.py` 생성:
   - 미래 LABEL_HORIZON_BARS(3봉) 수익률 기준: ≥+0.5% → buy, ≤-0.5% → sell, 그 외 hold
   - 마지막 3봉은 라벨 불가 → drop (lookahead 아님, 라벨 미생성)
   - soft label: 수익률 선형 매핑으로 gold 확률분포 생성 (예: +0.3% → buy 0.6/hold 0.4) — RLCD 손실이 보정된 확률을 학습하도록
3. `pipeline/make_jsonl.py` 생성:
   - 스키마: `{"state": {...}, "questions": {"action": {"type": "choice", "instructions": config.INSTRUCTIONS, "criteria": config.CRITERIA}}, "gold": {"action": {"probabilities": {"buy": p, "hold": q, "sell": r}}}}`
   - state는 수치 JSON + stockb의 PER/시총/업종 등 정적 정보 join (가능하면)
   - **시간순 분할**: 전체 기간의 앞 80% 날짜 → train.jsonl, 뒤 20% → eval.jsonl (config.TRAIN_RATIO)
   - random split 절대 금지
4. 검증: `python -m laya.train` dry_run 또는 laya.train.read_jsonl로 각 파일 로드 가능 확인, class 분포(buy/hold/sell 비율) 출력

## MUST NOT
- 랜덤 셔플 후 분할
- 종목 간 정보 교차 (다른 종목 피처가 state에 섞이지 않게)
- 미래 봉 정보가 state에 포함되는任何形式의 누수

작업 완료 후: train/eval 행수, 클래스 분포, 피처 목록 보고.
