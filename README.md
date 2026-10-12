# 라야 (Laya) 주식 트레이딩 파이프라인

Laya 결정 모델(consaiinnovations/laya)을 10분봉 주식 데이터로 파인튜닝하여
매수/매도/홀드 의사결정을 내리는 파이프라인.

## 아키텍처

```
MongoDB stock2020 (C# m1.exe가 기록한 실시간 틱)
        │
        ▼
[1] extract_bars.py   틱 → 10분봉 집계
        │
        ▼
[2] build_features.py 봉 → 지표 state 피처 (RSI/MACD/MA/거래량...)
        │
        ▼
[3] build_labels.py   미래 1~3봉 수익률 → buy/hold/sell gold 확률
        │
        ▼
[4] make_jsonl.py     {state, questions, gold} JSONL (시간순 80:20 분할)
        │
        ▼
[5] finetune_kaggle.py  Kaggle T4에서 laya.train.finetune()
        │
        ▼
[6] serve/laya_api.py   추론 서버 → C# m1.exe가 HTTP 호출
        │
        ▼
[7] backtest.py         10분봉 백테스트 (수수료·슬리피지 반영)
```

## 사용법

각 단계는 opencode 커맨드로 실행:

| 커맨드 | 작업 |
|---|---|
| `/extract` | 틱 → 10분봉 추출 |
| `/features` | 피처 + 라벨 + JSONL 생성 |
| `/train` | Kaggle 학습 안내/스크립트 준비 |
| `/serve` | 추론 서버 실행 |
| `/backtest` | 백테스트 실행 |

## 모델 (가중치)

파인튜닝된 가중치는 HuggingFace Hub에 배포됩니다:
[`romee/laya-stock-model`](https://huggingface.co/romee/laya-stock-model)

803MB 가중치는 이 저장소에 포함되지 않습니다(`.gitignore`).
`config.MODEL_DIR`(=`stock-model/`)에 내려받거나 `laya.load("romee/laya-stock-model")`로 로드하세요.

## 데이터 출처

- **MongoDB**: 로컬 `stock2020` (MongoDB 3.4, `C:\Program Files\MongoDB\Server\3.4`)
- **일자별 틱 컬렉션**: `2025-01-06` 형식, 하루 10~27만건
- **보조**: `stockdt`(일봉), `stockb`(기본정보), `rssc`(뉴스 테마)

## ⚠️ 데이터 모집단 규칙 (가장 중요)

**틱 데이터는 "그 거래일의 거래대금/거래량 상위 50위 종목"만 저장됨**
(전종목 저장은 데이터량 과다로 불가 — 5년간 운용해온 m1.exe의 설계)

따라서:

1. **학습 모집단 = 추론 모집단 일치 (필수)**
   모델은 "상위 50권 종목의 상태"만 보고 학습됨.
   실시간 추론 시에도 **현재 상위 50권에 든 종목만** Laya에 질문해야 함.
   모집단이 다르면 확률(보정된 confidence)이 무의미해짐.
2. **전략 맥락**: 특성상 종목이 상위 50권에 들어온 뒤 **보통 하루~이틀 내에 처리(순위 이탈)** 됨 → 라벨 지평선 설계 시 참고 (현재: 미래 3봉=30분, 후보: 1~2일 스윙 라벨 추가)
3. **모집단 관련 피처**: `거래금누적`, 당일 거래순위 등을 state에 포함하면 "지금 상위권인가" 맥락이 모델에 전달됨
4. 하루 distinct 종목 ≈ 105개 (상위 50 + 장중 순위 교체분)이 정상
5. **전략 성격**: 상위 50 필터 = 그날 강하게 움직이는 급등/모멘텀 종목. 모델이 학습하는 것은
   "이미 핫한 종목의 다음 움직임(상한가 근접 매수세 vs 되돌림)" — 상한가 근처 종목은
   양극단 분포라 분류(래야)에 적합
6. **상한가 체결 제약**: 현재가가 상한가면 신규 매수 체결 불가 → 추론/백테스트에서
   buy 시그널을 무시하는 규칙 필요 (보유분 매도는 정상 가능)

## 핵심 규칙

1. **시간순 분할만** — 랜덤 분할은 시계열 누수 (eval은 반드시 마지막 기간)
2. **피처는 과거만, 라벨만 미래** — lookahead 금지
3. **학습은 Kaggle T4** — 이 노트북에는 CUDA 없음, 추론만 로컬 CPU
4. **실전 전 백테스트 필수** — 거래세·슬리피지 반영
5. **모집단 일치** — 상위 50권 종목만 학습/추론 (위 섹션 참조)
