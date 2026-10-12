"""단계 [4]: labeled.parquet → laya 학습용 JSONL (train/eval, 시간순 80:20 분할).

읽기: data/labeled.parquet
쓰기: data/train.jsonl, data/eval.jsonl

JSONL 1행 스키마 (laya 0.4.0):
  {"state": {...},
   "questions": {"action": {"type": "choice",
                             "instructions": config.INSTRUCTIONS,
                             "criteria": config.CRITERIA}},
   "gold": {"action": {"probabilities": {"buy": p, "hold": q, "sell": r}}}}

state 규칙:
  - code(문자열) + 모든 수치 피처(OHLCV, n_ticks, 지표 전체)
  - 제외: date, bar_time, name (달력/시각 기억 방지 — day_progress가 장진행도 대신 표현),
          fwd_ret, label, buy/hold/sell_prob (미래 파생 = 누수 금지)
  - state에 NaN/None/inf 가 있으면 그 행 자체를 제거 (warmup 구간 등), 건수 출력
  - 실수는 소수점 6자리 반올림

분할 규칙:
  - 고유 날짜 정렬 후 앞 config.TRAIN_RATIO(0.8) → train, 나머지 → eval
  - 날짜 단위로만 갈라짐(한 날짜가 양쪽에 걸치지 않음), 격리 assert 수행
  - 랜덤 분할 금지, 셔플 금지 — date, bar_time, code 오름차순(시간순) 기록

실행: python pipeline/make_jsonl.py [--labeled PATH] [--train PATH] [--eval PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (project-root config)

# state에서 제외할 컬럼 (미래 파생 + 식별/달류 정보)
STATE_EXCLUDE = {
    "date", "bar_time", "name",          # 달류/시각 → 날짜 기억 방지
    "fwd_ret", "label",                  # 미래 파생
    "buy_prob", "hold_prob", "sell_prob",  # 라벨 확률 (gold로만)
}
STATE_STRING_COLS = ["code"]
FLOAT_DECIMALS = 6


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
    """JSON용 값 변환: str 유지, int는 int, 그 외 float 6자리 반올림."""
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


def make_jsonl(labeled_path: Path, train_path: Path, eval_path: Path) -> None:
    df = pd.read_parquet(labeled_path)
    for col in ("code", "date", "bar_time", "label"):
        if col not in df.columns:
            raise SystemExit(f"labeled parquet missing column: {col}")

    state_cols = select_state_columns(df)
    print("=== make_jsonl ===")
    print(f"in        : {labeled_path} ({len(df):,} rows)")
    print(f"state cols: {state_cols}")

    # --- 결측/무한 state 행 제거 ---
    numeric_cols = [c for c in state_cols if c not in STATE_STRING_COLS]
    bad = df[numeric_cols].isna().any(axis=1)
    arr = df[numeric_cols].to_numpy(dtype="float64", na_value=np.nan)
    bad = bad | ~np.isfinite(arr).all(axis=1)
    bad = bad | df["code"].isna()
    n_dropped = int(bad.sum())
    df = df[~bad].copy()
    print(f"dropped   : {n_dropped:,} rows with NaN/inf state values (warmup etc.)")

    # --- 시간순 정렬 (셔플 금지) ---
    df = df.sort_values(["date", "bar_time", "code"], kind="mergesort").reset_index(drop=True)

    # --- 날짜 기준 80:20 분할 ---
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        raise SystemExit(f"need >= 2 distinct dates to split, got {len(dates)}")
    split = int(len(dates) * config.TRAIN_RATIO)
    split = max(1, min(split, len(dates) - 1))
    train_dates, eval_dates = dates[:split], dates[split:]
    assert set(train_dates).isdisjoint(eval_dates), "date leakage between splits"
    boundary = f"last train={train_dates[-1]}  first eval={eval_dates[0]}"
    print(f"dates     : {len(dates)} total | train {len(train_dates)} "
          f"({train_dates[0]}..{train_dates[-1]}) | eval {len(eval_dates)} "
          f"({eval_dates[0]}..{eval_dates[-1]})")
    print(f"boundary  : {boundary}")

    def rows_for(sub: pd.DataFrame) -> list[str]:
        records = sub[state_cols].to_dict("records")
        probs = sub[["buy_prob", "hold_prob", "sell_prob"]].to_numpy(dtype="float64")
        lines = []
        for rec, (p_buy, p_hold, p_sell) in zip(records, probs):
            state = {k: _clean_value(v) for k, v in rec.items()}
            line = {
                "state": state,
                "questions": {
                    config.QUESTION_ID: {
                        "type": "choice",
                        "instructions": config.INSTRUCTIONS,
                        "criteria": config.CRITERIA,
                    }
                },
                "gold": {
                    config.QUESTION_ID: {
                        "probabilities": {
                            "buy": round(float(p_buy), FLOAT_DECIMALS),
                            "hold": round(float(p_hold), FLOAT_DECIMALS),
                            "sell": round(float(p_sell), FLOAT_DECIMALS),
                        }
                    }
                },
            }
            lines.append(json.dumps(line, ensure_ascii=False))
        return lines

    def hard_dist(sub: pd.DataFrame) -> dict[str, int]:
        probs = sub[["buy_prob", "hold_prob", "sell_prob"]].to_numpy(dtype="float64")
        idx = probs.argmax(axis=1)  # 0=buy,1=hold,2=sell
        names = ["buy", "hold", "sell"]
        return {names[i]: int(c) for i, c in
                zip(*np.unique(idx, return_counts=True))}

    train_df = df[df["date"].isin(train_dates)]
    eval_df = df[df["date"].isin(eval_dates)]
    assert len(train_df) + len(eval_df) == len(df)

    train_lines = rows_for(train_df)
    eval_lines = rows_for(eval_df)

    for path, lines in ((train_path, train_lines), (eval_path, eval_lines)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")

    print(f"train rows: {len(train_lines):,}  "
          f"dist(hard argmax)={hard_dist(train_df)}")
    print(f"eval rows : {len(eval_lines):,}  "
          f"dist(hard argmax)={hard_dist(eval_df)}")
    for path in (train_path, eval_path):
        print(f"file      : {path}  {path.stat().st_size:,} bytes")
    # 경계 날짜 재확인 (보고용)
    print(f"split ok  : {boundary} (no shared dates)")


def main() -> None:
    ap = argparse.ArgumentParser(description="labeled.parquet → train/eval JSONL")
    ap.add_argument("--labeled", type=Path, default=config.LABELED_PATH, help="input labeled parquet")
    ap.add_argument("--train", type=Path, default=config.TRAIN_JSONL, help="output train jsonl")
    ap.add_argument("--eval", type=Path, default=config.EVAL_JSONL, help="output eval jsonl")
    args = ap.parse_args()
    make_jsonl(args.labeled, args.train, args.eval)


if __name__ == "__main__":
    main()
