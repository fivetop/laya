# 라야(Laya) 주식 트레이딩 — 결과 리포트

**작성일**: 2026-10-10 (토)
**대상**: Laya 결정모델(ModernBERT-large 기반)을 10분봉 한국 주식 데이터로 파인튜닝한 결과와 백테스트 성적

---

## 1. 요약 (Executive Summary)

| 항목 | 결과 |
|---|---|
| 학습 | Kaggle T4에서 파인튜닝 성공 (4 epochs, ~10h) |
| held-out 정확도 | **31.9% → 46.5%** (+14.6%p), 보정오차(ECE) 0.087 → 0.065 |
| 로컬 서빙 | `serve/laya_api.py` (포트 8001) 정상 — 로드 12초, 추론 1.53초/건(CPU) |
| 백테스트(모델) | **+4.70%** (20일, 341종목 평균) vs Buy&Hold **+2.31%**, 승률 53.7% |
| 백테스트(완벽예지 상한) | +32.3%, 승률 90.2% → 모델은 상한의 약 15%만 포착 |

**핵심 판단**: 모델은 Buy&Hold를 상회하는 **약한 양(+)의 엣지**를 보였으나, 수익이 소수 종목에 집중되고(중앙값 0%), "buy"를 고신뢰로 거의 내지 못하는 구조적 한계가 확인됨. 실전 투입 전 추가 개선이 필요.

---

## 2. 시스템 구성

```
MongoDB stock2020 (C# m1.exe가 기록한 실시간 틱, 상위 50위 종목만)
   │  extract_bars.py   (틱 → 10분봉, 496초/164,460행)
   ▼
features.parquet  (18 피처: OHLCV·지표 + 당일거래 rank/거래금액 4개)
   │  build_labels.py   (미래 3봉 = 30분 수익률 ±0.5% → buy/hold/sell)
   ▼
labeled.parquet (161,309행)
   │  make_jsonl.py     (시간순 80:20, state 25키 + gold 확률)
   ▼
train.jsonl(85,473) / eval.jsonl(27,725)
   │  finetune_kaggle.py → Kaggle T4 (GPU)
   ▼
stock-model (ModernBERT-large 1024d·28층, 842MB)
   │  serve/laya_api.py (:8001)
   ▼
C# m1.exe (주문/체결) ← HTTP
```

- **라벨**: 30분 스캘핑, ±0.5% (buy ≥ +0.5% / sell ≤ -0.5% / 그 외 hold)
- **분할**: 시간순 80:20 (last train 2025-06-24 < first eval 2025-06-25, 날짜 격리 no leakage)
- **모집단**: 학습·추론 모두 "그날 상위 50위 종목" (모집단 일치 필수)
- **상한가 규칙**: 체결 bar(t+1) open ≥ 전일종가×1.30 → 신규 매수 차단

---

## 3. 학습 결과 (Kaggle T4)

`train_report.json` (held-out 27,725건, 독립 평가):

| 지표 | Base | Fine-tuned | 개선 |
|---|---|---|---|
| Accuracy | 0.3189 | **0.4648** | +0.1459 |
| Loss | 1.1264 | 1.0832 | -0.0432 |
| ECE (보정오차) | 0.0872 | **0.0646** | -0.0226 |
| Brier | 0.1513 | 0.1213 | -0.0300 |
| mean confidence | 0.4061 | 0.4002 | — |

- 학습 설정: loss=`rlcd`, rl_samples=4, epochs=4, micro_batch=8, grad_accum=8,
  encoder_lr=2.5e-5, head_lr=1e-4, φ(σ) 0.4→0.1, label_smoothing=0
- calibration temperature: 1.171 (choice)
- 부트스트랩: `convaiinnovations/laya` → `rl-agent` 헤드
- 참고: 학습 로그에 "predictions collapsed to the class prior" 경고가 있었으나
  최종 held-out 성적은 분리된 클래스 우선확률보다 개선됨

---

## 4. 로컬 검증

| 검증 | 결과 |
|---|---|
| 모델 로드 (`laya.load`) | 성공, 12–14초 (CPU, fp32) |
| 단건 추론 (`agent.predict`) | 성공, 1.53초/건 |
| 배치 추론 (`predict_batch`) | 0.74–0.83 행/초 (4물리코어·8논리코어 한계) |
| 서버 `/health` | 200 `{"model_loaded": true}` |
| 서버 `/api/laya-predict` | 200 buy/hold/sell + probabilities |
| 서버 `/api/laya-batch` | 200 다건 일괄 |
| Swagger UI `/docs` | 200 (브라우저 테스트 가능) |

- **주의**: PC는 물리 4코어 → 전체 eval 27,725행 추론에 약 10.4시간 소요. 실시간
  분당 다종목 추론에는 부족 (아래 8. 한계 참조).

---

## 5. 백테스트 결과

**설정**: t 시그널 → t+1 open 체결(no lookahead), 비용 0.015%/방향(왕복 0.03%),
종목별 초기자금 1,000만원, 롱온리, 상한가 매수차단 반영, eval 20일(2025-06-25~07-23, 341종목)

| 전략 | 거래수 | 승률 | 평균수익률 | MDD | 차단매수 |
|---|---|---|---|---|---|
| Buy & Hold (기준) | — | — | +2.31% | — | — |
| **모델 예측 (pred)** | 1,519 | 53.65% | **+4.70%** | 3.96% | 14 |
| 진입게이트 conf≥0.40 | 445 | 58.65% | +3.17% | 2.57% | 6 |
| 진입게이트 conf≥0.45 | 4 | 75.00% | +0.03% | 0.05% | 0 |
| 완벽예지 상한 (label) | 2,204 | 90.15% | +32.27% | 1.49% | 8 |

**종목별 분포 (pred)**: 341종목 중 161종목 수익(47.2%), 중앙값 0.0%, 최고 +300%, 최저 -26%.
→ 수익이 소수 대박 종목에 집중.

---

## 6. 핵심 발견 — 신뢰도(confidence)의 정보력

`pred_conf = max(softmax 확률)` 구간별 실제 정확도:

| confidence 구간 | 건수 | 정확도 |
|---|---|---|
| < 0.34 | 475 | 33.7% |
| 0.34–0.40 | 16,512 | 43.4% |
| 0.40–0.45 | 7,270 | 54.9% |
| 0.45–0.50 | 1,877 | **69.3%** |
| 0.50–0.60 | 1,403 | **92.0%** |
| ≥ 0.60 | 188 | 85.6% |

→ **신뢰도는 정확도와 단조 증가** — 보정(calibration)이 실제로 작동함. 확률을 신뢰할 수 있음.

**그러나** 고신뢰(≥0.45) 신호의 방향 분포:

| 방향 | 건수 |
|---|---|
| sell | 685 |
| hold | 2,779 |
| **buy** | **4** |

→ 모델은 **"buy"를 고신뢰로 사실상 내지 못함**. 고신뢰 신호는 대부분 sell/hold.
→ 롱온리(long-only) 전략에서 쓸 수 있는 고신뢰 buy가 없어, 게이트를 걸면 거래가 소멸.

**클래스별 정확도**: hold 58.6% / sell 45.4% / buy 41.3%
→ 모델의 강점은 "보유/회피(hold·sell) 판단"이고, "매수 타이밍"은 약함.

---

## 7. 해석

긍정적:
1. held-out에서 정확도·보정 모두 학습으로 명확히 개선 → 데이터에 학습 가능한 신호가 존재.
2. 모델 확률이 잘 보정되어 있어(고신뢰=고정확) 리스크 관리(게이트)에 활용 가능.
3. Buy&Hold를 상회하는 (+) 기대수익 (+4.70% vs +2.31%), 승률 50% 초과.

한계:
1. **수익 집중**: 종목 중앙값 0% → 대부분 거래는 무의미, 소수 종목이 성과 견인.
2. **buy 신호 부재**: 고신뢰 매수 신호가 4건뿐 → 롱온리 전략의 핵심 동력 부족.
3. **상한 대비 낮은 포착률**: 완벽예지 +32%의 약 15%만 달성.
4. **롱온리 구조**: 모델의 강점인 sell(회피) 신호를 수익화할 수단이 없음.

---

## 8. 다음 단계 (권장)

1. **롱온리 탈피 검토** — sell 신호를 활용하는 전략(현금비중 확대·선물/공매도·인버스 ETF)
   또는 "sell = 청산 + 현금 보유" 최적화로 모델 강점 활용.
2. **라벨 재설계** — buy 클래스 불균형 완화(임계값·가중·스윙 라벨 혼합)로 고신뢰 buy 유도.
3. **데이터/학습 확대** — 전체 101일 → 더 긴 기간, 클래스 가중, early-stopping.
4. **추론 가속** — CPU 실시간 한계 극복: ONNX Runtime / 양자화 / 경량화, 또는 추론 서버 상시화.
5. **실전 드라이런** — C# m1.exe ↔ :8001 연동, 소액 모의투자로 3번 항목 검증 후 실투입.
6. **거버넌스** — Kaggle 토큰 재발급(채팅 노출됨), 임시 산출물 정리.

---

## 9. 산출물 (파일)

| 파일 | 내용 |
|---|---|
| `stock-model/` | 파인튜닝 모델 (model.safetensors, encoder, tokenizer, rl_agent_config.json) |
| `model/stock-model.zip` | Kaggle 원본 ZIP (1.45GB, 백업) |
| `data/eval_pred.parquet` | eval 27,725행 예측(pred, pred_conf) |
| `data/eval_pred_gated.parquet` | 신뢰도 게이트 변형 포함 |
| `data/backtest_pred.csv` | 모델 백테스트 종목별 성적 |
| `data/backtest_label.csv` | 완벽예지 상한 종목별 성적 |
| `backtest/generate_predictions.py` | eval 예측 신호 생성기 (증분 저장) |
| `serve/laya_api.py` | 추론 서버 (:8001) |

---

## 부록: 재현 명령

```powershell
# 1) eval 예측 생성 (약 10시간)
python backtest/generate_predictions.py --batch-size 32 --partial-every 10

# 2) 백테스트 (모델 / 상한)
python backtest/backtest.py --input data/eval_pred.parquet --signal-col pred  --csv data/backtest_pred.csv
python backtest/backtest.py --input data/eval_pred.parquet --signal-col label --csv data/backtest_label.csv

# 3) 추론 서버
python serve/laya_api.py     # http://127.0.0.1:8001/docs
```
