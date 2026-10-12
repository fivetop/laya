# 진행 이력 (날짜별)

프로젝트 진행 기록. 각 날짜 항목에 그날 완료한 작업과 결과를 기록한다.

---

## 2026-10-08 (목)

### 완료
- [x] **프로젝트 스캐폴드** — `C:\0GitHub\라야` 생성
  - README.md, requirements.txt, config.py
  - `.opencode/command/`: `/extract`, `/features`, `/train`, `/serve`
  - `.opencode/skill/laya-pipeline/SKILL.md` (DB 스키마·핵심 규칙 지식베이스)
- [x] **환경 확인**
  - MongoDB 3.4 로컬 실행 확인 (`mongod.exe`, DB `stock2020`)
  - pymongo **3.12.3** 설치 (MongoDB 3.4 호환 — 4.x는 서버 3.6+ 필요)
  - laya 0.4.0 설치 확인
  - CUDA GPU 없음 확인 → 학습은 Kaggle T4, 추론은 로컬 CPU로 확정
- [x] **핵심 설계 확정**
  - 봉 주기: 10분봉 (초단타/분봉은 노트북 환경에서 부적합으로 판단)
  - 라벨: 미래 3봉(30분) 수익률 ±0.5% → buy/hold/sell (soft label)
  - 분할: 시간순 80:20 (랜덤 분할 금지 — 시계열 누수)
  - 서버: laya 추론은 포트 8001 (기존 ai_server 8000과 병행)

### 데이터 현황 (MongoDB stock2020)
- 일자별 틱 컬렉션: 2025-01-06 ~ 2025-07-23 (약 96일, 결측 구간 존재)
- 하루 10~27만건, distinct 종목 ≈ 105개
- 보조: stockdt(일봉 355종목), stockb(기본정보), rssc(뉴스테마 588건)

### ⚠️ 모집단 규칙 확정 (사용자 전달 — 중요)
- **틱은 "그 거래일 상위 50위 종목"만 저장** (전종목은 데이터량 과다로 저장하지 않음)
- 종목 특성: 상위 50권 진입 후 **보통 하루~이틀 내 처리(순위 이탈)**
- 파생 규칙: 학습 모집단 = 추론 모집단 (실시간 추론도 현재 상위 50권 종목만 질문),
  라벨 지평선에 1~2일 스윙 라벨 후보 추가 검토, 당일 거래순위를 state 피처에 포함
- README "데이터 모집단 규칙" 섹션에 기록 완료
- 추가 전달: **상위 50 = 그날 강하게 움직이는(상한가 근접) 급등종목** — 모델은 "핫 종목의
  다음 움직임(매수세 vs 되돌림)"을 학습. 상한가 체결 불가 규칙 → 추론·백테스트에 반영 필요로 README 기록

### 완료 (컴포넌트)
- [x] train/finetune_kaggle.py + train/README_kaggle.md — `--help` 실측 검증 통과
- [x] backtest/backtest.py — 합성데이터(2종목×20일)로 실행 검증 통과 (t+1 open 체결, 비용 0.015%/방향, buy&hold 비교)
- [x] **pipeline/extract_bars.py** — 실제 DB 추출 완료 (496.6초)
  - 틱 14,697,996건 → `data/bars_10m.parquet` **164,460행**
  - distinct 종목 1,058 / 일자 101일 (2025-01-06 ~ 2025-07-23, 결측 구간 정상 반영)
  - 이슈 해결: MongoDB 3.4 `find().sort()`는 32MB 초과 시 디스크 스파일 없음 →
    1일 단위 클라이언트 사이드 정렬로 전환
  - 검증: OHLC 불변식, 중복키 0, 봉시각 09:00~15:20 10분 간격, 샘플 교차검증
    (2025-07-16/006920 09:00봉 open=3485 = 원본 첫 틱가 일치)
  - 비고: 대부분의 날 마지막 봉이 15:10~15:19까지 (15:20봉 1개뿐) — C# 수집 시점의 실제 패턴
- [x] **pipeline/build_features.py** — 당일 거래순위 피처 4개 추가 (`_add_market_context()`)
  - `cum_val`(당일 누적거래금액), `rank_val`/`rank_vol`(시점별 순위, method='first'),
    `day_tot_val_share`(거래금액 점유율) — 모집단(상위 50) 맥락 전달용
  - 합성데이터 검증 **71/71 PASS** (기존 46개 + 새 피처 13개 등)
  - 주의: rank는 (date, bar_time) 단위 cross-sectional — 과거만 사용, lookahead 없음
- [x] **backtest/backtest.py** — 상한가 체결 불가 규칙 반영
  - 전일종가 +30% (보수적 근사), 체결 bar(t+1) open ≥ 상한가 → 매수 skip
  - `blocked_buys` 통계 추가, sell 경로 불변, 기존 검증 케이스 전부 유지 확인
- [x] pipeline/build_labels.py, pipeline/make_jsonl.py — 합성데이터 e2e + laya read_jsonl 라운드트립 통과 (이전 완료)

### 설계 확정 (사용자 결정)
- **라벨 기준: 30분 스콜핑, ±0.5% 유지** (1~2일 스윙 3% 후보는 기각)

### 완료 (실제 데이터 통합 실행)
- [x] **features → labels → make_jsonl 전체 성공** (bars_10m.parquet 164,460행 기반)
  - features: 18개 피처 (14 + rank 4개), 60봉 warmup 이후 NaN 0
  - labels: 161,309행 (tail 3,151 제외) — buy 27.5% / hold 39.1% / sell 33.4%
  - JSONL: train **85,473행** (2025-01-07~06-24) / eval **27,725행** (2025-06-25~07-23)
  - 시간순 분할 검증: 경계 날짜 중복 없음 (last train 06-24 < first eval 06-25)
  - laya `read_jsonl` 실로드 검증: train/eval 정상, state 25개 키 전부 수치형, gold 확률합 1.0
  - state에 rank 피처 포함 확인 (예: rank_val=7, rank_vol=9, share=0.049)
  - 파일: train.jsonl 76MB / eval.jsonl 25MB

### 진행 중
- [ ] serve/laya_api.py — 추론 서버 (포트 8001, fail-safe: 응답 없으면 관망 기본값)

### 완료 (Kaggle GPU 학습 준비·디버깅 — 장애 해결 전 과정)
- [x] **Kaggle 계정/인증** — CLI 2.2.4 설치, `~/.kaggle/access_token` 인증, quota GPU 30h 확인
- [x] **데이터셋 업로드** — https://www.kaggle.com/datasets/romyoh/laya-stock-data (private, train 76MB + eval 25MB)
- [x] **학습 커널 생성/푸시** — https://www.kaggle.com/code/romyoh/laya-stock-finetune
  (dataset_sources 연동, enable_internet, script 커널)
- [x] **장애1: API로는 GPU 미할당** — `machine_shape=NvidiaTeslaT4`가 save_kernel /
  create_kernel_session 모두에서 서버에 의해 무시 (쿼터 예약 0, `/dev/nvidia*` 없음)
  → 브라우저 경로로 전환
- [x] **장애2: 전화 인증 미이행** — Kaggle GPU 이용에 **phone verification 필수**
  (quota는 30h 표시되지만 장치 미할당). 인증 후 GPU 할당 확인 (쿼터 소모 시작, `has_ever_run: True`)
- [x] **장애3: `__file__` 미정의** — 노트북 셀 환경에서 NameError → `_SCRIPT_DIR` 가드 +
  argparse `parse_known_args`(Jupyter 커널 인자 무시) + `sys.exit(0)` 가드
- [x] **장애4: 데이터셋 마운트 경로** — 세션이 `/kaggle/input/datasets/romyoh/laya-stock-data/`
  (레거스 레이아웃)에 마운트 → rglob 자동 탐색으로 수정
- [x] **장애5: 세션 Internet 꺼짐** — Session options 토글 켜기로 pip 통과
- [x] **장애6: 구형 docker image 핀** — `f4b6dd72`(py3.10, torch < 2.5) 핀이 transformers 5.19와
  불호환 (`torch.accelerator` 없음) → `docker_image_pinning_type="latest"`로 핀 해제
- [x] **장애7: laya 0.4.0 디바이스 버그** — `finetune()`이 eval_data 존재 시 모델(CPU) 상태에서
  `calibration_records(_forward)`를 먼저 호출하고 `model.to(device)`는 `train_model` 안에서
  뒤에 발생 → GPU에서 "Expected all tensors to be on the same device" 크래시.
  → `_forward` 호출 시 `model.to(device)` 하는 monkey-patch를 finetune_kaggle.py에 적용
- [x] **학습 실행 성공** — T4에서 `device cuda` / train 85,073 / calib 400 / eval 27,725 로드,
  base 평가 통과, `epoch 1/4 step 100 loss 0.6851` 확인 (4에포크 진행 중)
- 비고: 학습 중 로그 API는 완료 시점에만 반환됨 → 진행 확인은 노트북 콘솔에서
- 비고: 학습 완료 후 Kaggle 토큰 재발급 권장 (채팅에 노출된 적 있음)

### 다음 단계
- [ ] Kaggle 학습 완료 대기 → `stock-model.zip` 다운로드 (`kaggle kernels output -k romyoh/laya-stock-finetune`)
- [ ] 학습된 모델 로컬 추출 → 서버 연동 → m1.exe HTTP 연결
- [ ] 백테스트 (상한가 blocked_buys 리포트 포함)

---

## 2026-10-10 (토)

### 완료 — Kaggle 학습 종료 및 산출물 회수
- [x] **이전 "멈춘 것 같다"는 재실행이었고, 실제로는 4에포크 완주** (T4, 약 10h)
  - 재실행분은 사용자가 중지 → GPU 쿼터 낭비 방지
- [x] **`stock-model.zip` 다운로드** (`kaggle kernels output -k romyoh/laya-stock-finetune`)
  - 원본 1.45GB → `model/stock-model.zip`, 압축 해제 후 `stock-model/`로 이동(config.MODEL_DIR 일치)
- [x] **최종 학습 리포트 확보** (`train_report.json`, held-out 27,725건)
  - Accuracy 0.3189 → **0.4648** (+14.59%p), Loss 1.1264→1.0832, ECE 0.0872→0.0646
  - 모델: ModernBERT-large(1024d·28층), rl-agent 헤드, temperature 1.171

### 완료 — 로컬 검증
- [x] **모델 로컬 로드·추론 성공** — `laya.load` 12–14초, 단건 `agent.predict` 1.53초
  - 반환 구조 확정: `answers[action] = {type, choice, probabilities{buy,hold,sell}, confidence, ...}`
- [x] **`serve/laya_api.py` 실 HTTP 검증** — `/health`(model_loaded=true), `/api/laya-predict`,
  `/api/laya-batch`, `/docs`(Swagger UI) 전부 200
  - 발견한 것: Agent에는 `predict` 메서드가 *없는 게 아니라* `predict = system_one` 별칭으로 존재
  - 이전 세션의 고아 서버(PID 8524, 모델 없이 8001 점유) 종료 후 정상 기동
- [x] **서버 상시 기동** — 백그라운드 (Start-Process, 로그 `serve/server.log`)

### 완료 — 백테스트 (실제 모델)
- [x] **`backtest/generate_predictions.py` 신규 작성** — eval 구간 재구성 + 배치 추론 + 증분 저장
  - 기존 `backtest.py`는 `--signal-col label`(정답)로만 검증돼 있었음 → 실제 모델 예측 경로 공백이었음
- [x] **eval 27,725행 예측 완료** (배치 0.74행/초, 총 37,710초 ≈ 10.4h)
  - pred 분포: hold 12,334 / sell 11,548 / buy 3,843, pred==label 정확도 0.5073
- [x] **`backtest.py` 버그 수정** — 실제 데이터 `bar_time`이 full datetime인데 `"%H:%M:%S"`만
  가정하던 문제 (합성데이터에서만 통과했던 latent bug) → robust 파싱으로 수정
- [x] **백테스트 결과** (eval 20일, 341종목, 비용 0.015%/방향):
  - Buy&Hold +2.31% | **모델 pred +4.70% (승률 53.65%, 거래 1,519)** | 상한(label) +32.27%
  - 종목별: 161/341 수익, 중앙값 0.0% → 수익이 소수 종목에 집중
- [x] **신뢰도 분석** — confidence 구간별 정확도 단조 증가 (0.45–0.50: 69%, 0.50–0.60: 92%)
  - 단, 고신뢰(≥0.45) 신호 중 buy는 **4건뿐** (sell 685 / hold 2,779) → 롱온리 전략의 근본 한계

### 완료 — 리포트
- [x] **`REPORT.md` 작성** — 시스템 구성·학습·로컬검증·백테스트·발견·한계·다음단계

### 다음 단계
- [ ] 롱온리 탈피 검토 (sell 신호 활용: 현금비중/인버스 등)
- [ ] 라벨 재설계로 고신뢰 buy 유도 (클래스 가중·임계 조정)
- [ ] 추론 가속 (ONNX/양자화) — CPU 4코어 실시간 한계 극복
- [ ] C# m1.exe ↔ :8001 드라이런
- [ ] Kaggle 토큰 재발급 (채팅 노출됨) + 임시파일 정리

---

## 2026-10-12 (월)

### 완료 — 공개 배포
- [x] **GitHub 공개 저장소 업로드** — `github.com/fivetop/laya`
  - 코드+문서+데이터 41파일 137MB (초기커밋 `2c9da8a`, 모델카드 커밋 `447f92d`)
  - `.gitignore`·`.gitattributes` 신설, 모델 가중치(1.6GB) 제외
- [x] **HF Hub 모델 배포** — [`romee/laya-stock-model`](https://huggingface.co/romee/laya-stock-model)
  - `model.safetensors`(803MB)+설정+토크나이저+모델카드 8파일, public, 커밋 `f3786f1`
  - HF 계정 확인 = `romee` (기존 가정 `romyoh` 아님)
  - `checkpoint_latest/`(중복 803MB)는 제외하고 업로드

### 이슈 — Git 인증
- 첫 push는 성공했으나 이후 push가 무한 대기 (Git Credential Manager 프롬프트 버그)
- 원인: ① GCM이 비대화형에서는 대기만 함 ② GitHub git 엔드포인트는 Bearer 인증 거부
- 우회: `git credential fill`로 저장 토큰 추출 → `http.extraheader=Authorization: Basic x-access-token:` 을 한 번만 적용
- 추후 해결 권장: `gh auth login` + `gh auth setup-git`

### 다음 단계 (갱신)
- [ ] **롱온리 탈피: 스마트 롱 포트폴리오 백테스트** (고신뢰 buy 집중·신뢰도 비중 — 방향 승인됨, 아직 미착수)
- [ ] 모델 카드 라이선스 명시 (`other` → MIT/Apache-2.0 결정 필요)
- [ ] 라벨 재설계로 고신뢰 buy 유도 | 추론 가속(ONNX) | C# m1.exe ↔ :8001 드라이런
- [ ] Kaggle 토큰 재발급 (채팅 노출됨)
