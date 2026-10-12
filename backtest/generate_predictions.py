"""eval 구간 예측 신호 생성 → 백테스트용 parquet.

make_jsonl.py 의 state 구성/split 로직을 그대로 재사용해 eval 구간을 재구성하고,
laya 모델로 배치 추론하여 pred(buy/hold/sell) 컬럼을 붙인다.

실행:
  python backtest/generate_predictions.py [--batch-size 32] [--max-dates N] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(min(8, (__import__("os").cpu_count() or 8)))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from pipeline.make_jsonl import select_state_columns, _clean_value  # noqa: E402


def build_frame(labeled_path: Path, all_dates: bool = False) -> pd.DataFrame:
    """make_jsonl 과 동일한 drop + 날짜기준 80:20 split.

    all_dates=True 면 split 없이 전체 기간을 반환 (101일 전체 재검증용).
    """
    df = pd.read_parquet(labeled_path)
    state_cols = select_state_columns(df)
    numeric_cols = [c for c in state_cols if c != "code"]
    bad = df[numeric_cols].isna().any(axis=1)
    arr = df[numeric_cols].to_numpy(dtype="float64", na_value=np.nan)
    bad = bad | ~np.isfinite(arr).all(axis=1)
    bad = bad | df["code"].isna()
    df = df[~bad].copy()
    df = df.sort_values(["date", "bar_time", "code"], kind="mergesort").reset_index(drop=True)
    if all_dates:
        return df
    dates = sorted(df["date"].unique())
    split = int(len(dates) * config.TRAIN_RATIO)
    split = max(1, min(split, len(dates) - 1))
    eval_dates = dates[split:]
    return df[df["date"].isin(eval_dates)].copy()


def states_from_frame(df: pd.DataFrame) -> list[dict]:
    state_cols = select_state_columns(df)
    records = df[state_cols].to_dict("records")
    out = []
    for rec in records:
        out.append({k: _clean_value(v) for k, v in rec.items()})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", type=Path, default=config.LABELED_PATH)
    ap.add_argument("--model", type=str, default=str(config.MODEL_DIR))
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "eval_pred.parquet")
    ap.add_argument("--out-partial", type=Path, default=config.DATA_DIR / "eval_pred.partial.parquet")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--partial-every", type=int, default=10, help="N배치마다 partial 저장(0=끔)")
    ap.add_argument("--max-dates", type=int, default=0, help="0=전체 eval 날짜, N=마지막 N일만(속도 테스트)")
    ap.add_argument("--all-dates", action="store_true",
                    help="80:20 split 없이 전체 기간(101일) 예측 — 전 기간 재검증용")
    args = ap.parse_args()

    df = build_frame(args.labeled, all_dates=args.all_dates)
    if args.max_dates > 0:
        edates = sorted(df["date"].unique())[-args.max_dates:]
        df = df[df["date"].isin(edates)].copy()
    df = df.reset_index(drop=True)
    print(f"[{('all' if args.all_dates else 'eval')}] rows={len(df):,} dates={df['date'].nunique()} "
          f"({df['date'].min()}..{df['date'].max()})", flush=True)

    states = states_from_frame(df)
    questions = {
        config.QUESTION_ID: {
            "type": "choice",
            "instructions": config.INSTRUCTIONS,
            "criteria": config.CRITERIA,
        }
    }

    import laya  # noqa: E402
    t0 = time.perf_counter()
    agent = laya.load(args.model)
    print(f"[load] {time.perf_counter()-t0:.1f}s", flush=True)

    bs = args.batch_size
    preds: list[str] = []
    confs: list[float] = []
    keep = ["date", "bar_time", "code", "name", "open", "high", "low", "close",
            "fwd_ret", "label", "pred", "pred_conf"]

    def save(path: Path, upto: int) -> None:
        sub = df.iloc[:upto].copy()
        sub["pred"] = preds
        sub["pred_conf"] = confs
        sub[[c for c in keep if c in sub.columns]].to_parquet(path, index=False)

    t0 = time.perf_counter()
    n_batches = (len(states) + bs - 1) // bs
    for b_idx, start in enumerate(range(0, len(states), bs)):
        part = states[start:start + bs]
        try:
            results = agent.predict_batch(part, questions, batch_size=bs)
            for r in results:
                ans = r["answers"][config.QUESTION_ID]
                probs = ans["probabilities"]
                top = max(probs, key=probs.get)
                preds.append(top)
                confs.append(float(probs[top]))
        except Exception as exc:  # 한 배치 실패가 전체를 죽이지 않게
            print(f"[warn] batch {b_idx} 실패: {exc}", flush=True)
            preds.extend(["hold"] * len(part))
            confs.extend([0.0] * len(part))
        done = min(start + bs, len(states))
        if args.partial_every and b_idx % args.partial_every == 0:
            save(args.out_partial, done)
        if b_idx % 20 == 0 or done == len(states):
            el = time.perf_counter() - t0
            rate = done / el if el > 0 else 0
            eta = (len(states) - done) / rate if rate > 0 else 0
            ndates = df["date"].iloc[:done].nunique()
            print(f"[pred] {done:,}/{len(states):,} ({ndates}일)  {el:.0f}s  "
                  f"{rate:.2f} rows/s  ETA {eta/60:.1f}min", flush=True)

    save(args.out, len(states))
    out = pd.read_parquet(args.out)
    print(f"[out] {args.out}  rows={len(out):,}")
    print("[dist] pred:", out["pred"].value_counts().to_dict())
    print("[dist] label:", out["label"].value_counts().to_dict())
    acc = (out["pred"] == out["label"]).mean()
    print(f"[acc ] pred==label accuracy: {acc:.4f}")
    print("PREDICT_DONE")


if __name__ == "__main__":
    main()
