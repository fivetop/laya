"""Laya 주식 의사결정 추론 서버 (port 8001).

사용법 (프로젝트 루트에서):
    python serve/laya_api.py

- GET  /health              -> 200 {"status":"healthy", ..., "model_loaded": bool}
- POST /api/laya-predict    -> 모델 없으면 503, 있으면 signal/confidence
- POST /api/laya-batch      -> 여러 종목 일괄 예측

모델 경로는 config.MODEL_DIR (stock-model). 학습 전에는 디렉터리가 없으므로
서버는 기동하되 /api/laya-predict는 503을 반환한다 (train/README_kaggle.md 참고).
"""
import sys
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Windows 콘솔(cp949)에서도 한글/특수문자 출력이 죽지 않도록
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 프로젝트 루트를 import 경로에 추가 (python serve/laya_api.py 로 실행할 때)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402

PORT = 8001
LABELS = ("buy", "hold", "sell")

# 서버 전역 상태. 모델은 import 시점이 아닌 startup에서만 로드한다.
agent: Optional[Any] = None
model_loaded: bool = False


class PredictRequest(BaseModel):
    stock_code: str
    stock_name: str = ""
    state: Dict[str, Any]


class BatchRequest(BaseModel):
    requests: List[PredictRequest]


def _load_model() -> Optional[Any]:
    """config.MODEL_DIR 존재 시 laya.load, 없으면 None + 경고."""
    model_dir = Path(config.MODEL_DIR)
    if not model_dir.exists():
        print(f"[warn] 모델 없음 — {model_dir} 존재하지 않음. "
              "/api/laya-predict는 503 반환 (train/README_kaggle.md 참고)")
        return None
    try:
        import laya
        loaded = laya.load(str(model_dir))
        print(f"[info] 모델 로드 완료: {model_dir}")
        return loaded
    except Exception:
        print(f"[error] 모델 로드 실패:\n{traceback.format_exc()}")
        return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global agent, model_loaded
    agent = _load_model()
    model_loaded = agent is not None
    yield
    agent = None
    model_loaded = False


app = FastAPI(title="Laya Stock Decision Server", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "Laya Stock Decision Server",
        "model_loaded": model_loaded,
    }


def _extract_probabilities(result: Any) -> Dict[str, float]:
    """laya predict 결과에서 buy/hold/sell 확률을 견고하게 추출.

    지원 형태:
      - result["answers"]["action"]["choice"] + ["probs"|"probabilities"|...]
      - result["answers"]["action"] 가 직접 라벨->확률 dict
      - DecisionResult 유사 객체 (속성 접근)
    """
    answers: Any = None
    if isinstance(result, dict):
        answers = result.get("answers", result)
    else:
        answers = getattr(result, "answers", result)

    action: Any = None
    if isinstance(answers, dict):
        action = answers.get(config.QUESTION_ID)
        if action is None and len(answers) == 1:
            action = next(iter(answers.values()))
    else:
        action = getattr(answers, config.QUESTION_ID, answers)

    # 1) {label: prob} 직접 매핑
    if isinstance(action, dict):
        for key in ("probs", "probabilities", "prob", "distribution", "scores"):
            inner = action.get(key)
            if isinstance(inner, dict) and any(k in inner for k in LABELS):
                return {k: float(inner.get(k, 0.0)) for k in LABELS}
        if any(k in action for k in LABELS):
            return {k: float(action.get(k, 0.0)) for k in LABELS}
        # 라벨 외 키가 섞인 경우: choice 값의 확률만 있는 등 방어 코드
        choice = action.get("choice")
        if isinstance(choice, str):
            return {k: (1.0 if k == choice else 0.0) for k in LABELS}

    # 2) 객체 속성 형태
    if action is not None:
        for key in ("probs", "probabilities", "prob"):
            inner = getattr(action, key, None)
            if isinstance(inner, dict) and any(k in inner for k in LABELS):
                return {k: float(inner.get(k, 0.0)) for k in LABELS}

    raise ValueError(f"확률을 추출할 수 없는 응답 형태: {result!r}")


def _predict_one(req: PredictRequest) -> Dict[str, Any]:
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="stock-model 없음: train/README_kaggle.md 참고 "
                   "(model missing: extract the stock-model first)",
        )

    questions = {
        config.QUESTION_ID: {
            "type": "choice",
            "instructions": config.INSTRUCTIONS,
            "criteria": config.CRITERIA,
        }
    }

    try:
        result = agent.predict(req.state, questions)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"predict 실패: {exc}",
        ) from exc

    probs = _extract_probabilities(result)
    signal = max(probs, key=probs.get)
    confidence = probs[signal]
    return {
        "status": "success",
        "stock_code": req.stock_code,
        "stock_name": req.stock_name,
        "signal": signal,
        "probabilities": probs,
        "confidence": confidence,
    }


@app.post("/api/laya-predict")
def laya_predict(req: PredictRequest) -> Dict[str, Any]:
    return _predict_one(req)


@app.post("/api/laya-batch")
def laya_batch(body: BatchRequest) -> Dict[str, Any]:
    # 모델이 없으면 요청 전체를 503으로 명확히 거부
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="stock-model 없음: train/README_kaggle.md 참고 "
                   "(model missing: extract the stock-model first)",
        )
    results = [_predict_one(r) for r in body.requests]
    return {"status": "success", "count": len(results), "results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
