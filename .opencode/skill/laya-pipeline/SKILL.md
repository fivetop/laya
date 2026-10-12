---
name: laya-pipeline
description: 라야(Laya) 주식 트레이딩 파이프라인 — MongoDB 틱 → 10분봉 → 피처/라벨 → laya 파인튜닝 → 추론 서버. 이 프로젝트의 데이터 구조, 규칙, 명령어를 알고 있을 때 사용.
---

# Laya 주식 파이프라인 스킬

## 프로젝트 구조
```
C:\0GitHub\라야\
├── config.py              # 모든 공용 설정 (경로, 임계값, 체크포인트)
├── pipeline/              # extract → features → labels → jsonl
├── train/                 # Kaggle 파인튜닝 스크립트
├── serve/                 # 추론 FastAPI (포트 8001)
├── backtest/              # 백테스트
├── data/                  # bars_10m.parquet, *.jsonl (생성물)
└── stock-model/           # laya.load() 대상 (Kaggle에서 내려받음)
```

## 데이터 소스 (MongoDB)
- 로컬 MongoDB **3.4** (`C:\Program Files\MongoDB\Server\3.4\bin\mongo.exe`), DB `stock2020`
- ** pymongo 4.x는 서버 3.4와 호환 안 됨 → pymongo==3.12.3 필수**
- 일자별 틱 컬렉션 `YYYY-MM-DD` (2025-01-06 ~ 2025-07-23, 결측 구간 있음: 4/23~5/27, 6/3, 6/5~6/8)
- 틱 문서 주요 필드: `str종목코드`, `str종목명`, `dt1`(UTC), `int현재가`, `int거래량`, `f등락률`, `int체결강도`, `int사인`, `강세`, `조건1~5`, `세력`, `급등` (파생지표 다수 보유)
- 보조: `stockdt`(일봉 355종목), `stockb`(기본정보 PER/PBR/시총), `stockb2021`(기업개요), `rssc`(뉴스테마 588건)
- 하루 distinct 종목 ≈ 105개 (프로그램이 추적한 종목만)
- mongo 셸에서 한글 필드/PowerShell `$` 출혈 주의 → 스크립트 파일로 실행

## 핵심 규칙
1. **시간순 분할만** — train/eval 랜덤 분할 절대 금지 (eval = 기간 뒤 20%)
2. **피처 = 과거만, 라벨 = 미래만** — lookahead 누수 없음
3. **봉 주기 10분** (초단타/분봉 아님) — 하루 39봉 × ~105종목 ≈ 4,100샘플/일
4. **학습 = Kaggle 무료 T4**, 추론 = 로컬 CPU (이 노트북 CUDA 없음)
5. **Laya JSONL 스키마**: `{state, questions: {action: {type: "choice", instructions, criteria}}, gold: {action: {probabilities: {buy, hold, sell}}}}`
6. 파인튜닝: `laya.train.finetune(data, model_dir, output_dir, TrainConfig(eval_data=...))`
7. 실전 투입 전 백테스트 필수 (수수료+슬리피지 반영, buy&hold 대비 우위 확인)

## 커맨드
- `/extract` — 틱 → 10분봉 (data/bars_10m.parquet)
- `/features` — 피처+라벨+JSONL 생성
- `/train` — Kaggle 학습 준비
- `/serve` — 추론 서버(8001) / 백테스트

## 기존 시스템 (연동 대상)
- C# 키움호텔 자동매매 `C:\romee\주식\m0\Kiwoom0414` → m1.exe
- 기존 Python 서버 `C:\romee\주식\m0\ai_server` (FastAPI :8000, RSI/MACD 규칙신호)
- 신규 laya 서버는 **8001**에서 돌려 병행 비교 후 교체
