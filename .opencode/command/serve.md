---
description: 추론 서버 실행 및 백테스트 실행
---

# 서버 실행 / 백테스트

## serve 모드 (`/serve serve`)
1. `serve/laya_api.py` 생성:
   - FastAPI, 포트 8001 (기존 ai_server가 8000 사용)
   - startup에서 `laya.load(config.MODEL_DIR)` 로드
   - `POST /api/laya-predict`: 요청 `{stock_code, stock_name, state}` → `{"signal": "buy|hold|sell", "probabilities": {...}, "confidence": float}` — 기존 ai_server 응답 포맷과 호환
   - 모델 미로드 시 503 + 명확한 에러 메시지
2. 실행: `python serve/laya_api.py`
3. 검증: `/health` 200, 샘플 state로 predict 응답 확인

## backtest 모드 (`/serve backtest`)
1. `backtest/backtest.py` 생성:
   - labeled 데이터로 시간순 walk-forward: 각 봉에서 signal별 포지션 반영
   - 비용: 양쪽 0.015% (수수료+세금+슬리피지 추정) per 거래
   - 지표: 누적수익률, max drawdown, 승률, 거래 횟수, buy&hold 비교
2. 실행 및 결과 표로 보고

## MUST NOT
- 포트 8000 사용 금지 (기존 서버)
- 백테스트에서 미래 정보 사용 (각 시점의 signal만으로 결정)
