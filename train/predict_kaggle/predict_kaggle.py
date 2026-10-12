"""Kaggle T4 GPU: 전 기간(101일, 113,198행) 예측 생성 → full_pred.parquet

로컬 backtest/generate_predictions.py 와 동일한 state 구성·cleaning 로직을
**자체 포함**(마운트되는 것은 labeled.parquet 하나뿐 — 프로젝트 패키지 없음).

모델: HF Hub 공개 저장소 `romee/laya-stock-model` 에서 laya.load 로 다운로드.
      (public repo → 인증 불필요; finetune 커널과 동일하게 enable_internet 필요)

출력: /kaggle/working/full_pred.parquet
      컬럼: date, bar_time, code, name, open, high, low, close, fwd_ret,
            label, pred, pred_conf  (로컬 eval_pred.parquet 와 동일 스키마)

검증: eval 구간(마지막 20일) 분포·정확도를 로컬 결과와 비교 인쇄
      (로컬: dist hold=12334 / sell=11548 / buy=3843, acc≈0.5073)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------- #
# make_jsonl.py 와 동일한 상수/로직 (자체 포함 — Kaggle에는 패키지가 없음)
# --------------------------------------------------------------------------- #
QUESTION_ID = "action"
INSTRUCTIONS = "Should we buy, hold, or sell this stock for the next 30 minutes?"
CRITERIA = {
    "buy": "expected return above +0.5% in the next 3 bars",
    "hold": "return between -0.5% and +0.5%",
    "sell": "expected return below -0.5% in the next 3 bars",
}

# state에서 제외할 컬럼 (미래 파생 + 식별/달력/시각)
STATE_EXCLUDE = {
    "date", "bar_time", "name",
    "fwd_ret", "label",
    "buy_prob", "hold_prob", "sell_prob",
}
STATE_STRING_COLS = ["code"]
FLOAT_DECIMALS = 6

KEEP_COLS = ["date", "bar_time", "code", "name", "open", "high", "low", "close",
             "fwd_ret", "label", "pred", "pred_conf"]


def select_state_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in STATE_EXCLUDE:
            continue
        if c in STATE_STRING_COLS:
            cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _clean_value(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    f = float(v)
    if not np.isfinite(f):
        raise ValueError(f"non-finite state value leaked: {v!r}")
    return round(f, FLOAT_DECIMALS)


def find_labeled() -> Path:
    """/kaggle/input 아래에서 labeled.parquet 탐색 (레이아웃 차이 흡수)."""
    roots = [Path("/kaggle/input")]
    hits = []
    for root in roots:
        if root.exists():
            hits += list(root.rglob("labeled.parquet"))
    if not hits:
        raise SystemExit("labeled.parquet not found under /kaggle/input")
    return hits[0]


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    """make_jsonl 과 동일한 cleaning + 시간순 정렬 (split 없이 전체 기간)."""
    state_cols = select_state_columns(df)
    numeric_cols = [c for c in state_cols if c != "code"]
    arr = df[numeric_cols].to_numpy(dtype="float64", na_value=np.nan)
    bad = df[numeric_cols].isna().any(axis=1) | ~np.isfinite(arr).all(axis=1)
    bad = bad | df["code"].isna()
    n_dropped = int(bad.sum())
    df = df[~bad].copy()
    df = df.sort_values(["date", "bar_time", "code"], kind="mergesort").reset_index(drop=True)
    return df, n_dropped


def states_from_frame(df: pd.DataFrame) -> list[dict]:
    state_cols = select_state_columns(df)
    records = df[state_cols].to_dict("records")
    return [{k: _clean_value(v) for k, v in rec.items()} for rec in records]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="romee/laya-stock-model")
    ap.add_argument("--out", type=str, default="/kaggle/working/full_pred.parquet")
    ap.add_argument("--out-partial", type=str, default="/kaggle/working/full_pred.partial.parquet")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--partial-every", type=int, default=50)   # N배치마다 partial 저장
    ap.add_argument("--max-rows", type=int, default=0)          # 0=전체 (디버그용)
    args, _ = ap.parse_known_args()  # Jupyter 커널 인자 무시

    # laya는 커널 이미지에 없음 → pip 설치 (finetune 커널과 동일 방식, 로컬과 동일한 0.4.0 고정)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.0"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"device={device}", flush=True)

    labeled = find_labeled()
    print(f"[in ] {labeled} (from /kaggle/input)", flush=True)
    df = pd.read_parquet(labeled)
    df, n_dropped = build_frame(df)
    print(f"[prep] rows={len(df):,} (dropped {n_dropped:,}) dates={df['date'].nunique()} "
          f"({df['date'].min()}..{df['date'].max()})", flush=True)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    df = df.reset_index(drop=True)

    states = states_from_frame(df)
    questions = {
        QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}
    }

    print(f"[load] {args.model} -> device={device}", flush=True)
    t0 = time.perf_counter()
    import laya  # noqa: E402
    agent = laya.load(args.model, device=device)
    print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- device sanity: 1 배치로 로드/추론 확인 ----
    sanity = states[:8]
    try:
        r0 = agent.predict_batch(sanity, questions, batch_size=8)
        ans = r0[0]["answers"][QUESTION_ID]
        print(f"[sanity] pred={ans['choice']} probs={ans['probabilities']}", flush=True)
        assert "buy" in ans["probabilities"], "probabilities keys unexpected"
    except Exception as exc:
        print(f"[sanity] FAILED: {exc!r}", flush=True)
        raise

    # ---- 전체 배치 추론 ----
    bs = args.batch_size
    preds: list[str] = []
    confs: list[float] = []

    def save(path: str, upto: int) -> None:
        sub = df.iloc[:upto].copy()
        sub["pred"] = preds
        sub["pred_conf"] = confs
        sub[[c for c in KEEP_COLS if c in sub.columns]].to_parquet(path, index=False)

    t0 = time.perf_counter()
    n_batches = (len(states) + bs - 1) // bs
    for b_idx, start in enumerate(range(0, len(states), bs)):
        part = states[start:start + bs]
        try:
            results = agent.predict_batch(part, questions, batch_size=bs)
            for r in results:
                ans = r["answers"][QUESTION_ID]
                probs = ans["probabilities"]
                top = max(probs, key=probs.get)
                preds.append(top)
                confs.append(float(probs[top]))
        except Exception as exc:
            print(f"[warn] batch {b_idx} 실패: {exc}", flush=True)
            preds.extend(["hold"] * len(part))
            confs.extend([0.0] * len(part))
        done = min(start + bs, len(states))
        if args.partial_every and b_idx % args.partial_every == 0:
            save(args.out_partial, done)
        if b_idx % 25 == 0 or done == len(states):
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0
            eta = (len(states) - done) / rate if rate > 0 else 0
            print(f"[pred] {done:,}/{len(states):,}  {el:.0f}s  "
                  f"{rate:.1f} rows/s  ETA {eta/60:.1f}min", flush=True)

    save(args.out, len(states))
    out = pd.read_parquet(args.out)
    print(f"[out] {args.out} rows={len(out):,}", flush=True)
    print("[dist] pred:", out["pred"].value_counts().to_dict(), flush=True)
    print("[dist] label:", out["label"].value_counts().to_dict(), flush=True)
    print(f"[acc ] pred==label : {(out['pred'] == out['label']).mean():.4f}", flush=True)

    # ---- eval 구간(마지막 20일) 로컬 결과와 대조 ----
    edates = sorted(out["date"].unique())[-20:]
    ev = out[out["date"].isin(edates)]
    local = {"hold": 12334, "sell": 11548, "buy": 3843}   # 로컬 eval_pred.parquet
    got = ev["pred"].value_counts().to_dict()
    print(f"[xref] eval-slice dist {got}", flush=True)
    print(f"[xref] local          {local}", flush=True)
    print(f"[xref] pred==label (eval-slice): {(ev['pred'] == ev['label']).mean():.4f} "
          f"(local 0.5073)", flush=True)
    print("PREDICT_DONE", flush=True)
    sys.exit(0)  # kernel 셸 종료 가드


if __name__ == "__main__":
    main()