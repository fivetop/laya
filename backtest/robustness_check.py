"""스마트 롱 최적 전략(0.42/CW/SELLSIG)의 통계적 견고성 검증.

20일 eval 윈도우는 표본이 작아 단일 백테스트 수치만으로는 우연일 수 있다.
다음 세 가지로 "0.42 엣지가 실재하는가 / 그냥 운인가"를 정량화한다:

  1) 일별 삭제 진단 (leave-one-day-out)     — 수익이 특정 1일(s)에 좌우되는지
  2) 무작위 16일 부분표본 × N (=leave-5-out) — 0.42/CW/SELLSIG 수익률 분포, CI,
     B&H 초과 확률, 0.42가 게이트 0.40·0.45를 이기는 빈도 (knife-edge 판정)
  3) 주간 워크포워드 (4×5일)               — 시간이 지나도 우위가 유지되는지

엔진(backtest_portfolio.py)을 모듈로 import 하여 prepare()/run_portfolio()를
재사용한다 (파일 I/O 없이 인프로세스 실행).

실행:
  python backtest/robustness_check.py [--n 250] [--seed 42] [--out CSV]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# backtest/ 는 패키지가 아니므로 파일 경로로 모듈 로드
_spec = importlib.util.spec_from_file_location(
    "backtest_portfolio", ROOT / "backtest" / "backtest_portfolio.py")
_bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bp)  # type: ignore[union-attr]
prepare = _bp.prepare
run_portfolio = _bp.run_portfolio
run_buy_hold = _bp.run_buy_hold

CASH, COST, MP = 10_000_000.0, 0.00015, 8

# 추적 전략: 베이스라인 3종 + 0.42 주변 3종 + 0.45 (희소성 확인용)
CONFIGS = [
    (None, "EW", "SELLSIG", "none_ew"),
    (0.40, "EW", "SELLSIG", "g40_ew"),
    (0.40, "CW", "SELLSIG", "g40_cw"),
    (0.42, "EW", "SELLSIG", "g42_ew"),
    (0.42, "CW", "SELLSIG", "g42_cw"),   # 권장 실전 파라미터
    (0.42, "CW", "FIXED3", "g42_cw_f3"),
    (0.45, "EW", "SELLSIG", "g45_ew"),
    (0.45, "CW", "SELLSIG", "g45_cw"),
]


def run_matrix(df: pd.DataFrame) -> dict[str, float]:
    """모든 CONFIGS의 total_return_pct와 B&H를 계산."""
    P = prepare(df)
    out = {"bh": run_buy_hold(P, CASH, COST)["total_return_pct"]}
    for gate, sz, ex, tag in CONFIGS:
        r = run_portfolio(P, gate, sz, ex, CASH, COST, MP)
        out[tag] = r["total_return_pct"]
        out[tag + "_n"] = r["n_trades"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "eval_pred.parquet")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "robustness_leave5out.csv")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    df["date"] = df["date"].astype(str)
    dates = sorted(df["date"].unique())
    print(f"rows={len(df):,}  days={len(dates)} ({dates[0]}..{dates[-1]})", flush=True)

    rng = np.random.default_rng(args.seed)
    n_drop = max(1, len(dates) // 5)  # ~16일 부분표본 (=leave-5-out)

    # ---------------------------------------------------------------- baseline
    print("\n== full window (all days) ==", flush=True)
    full = run_matrix(df)
    for tag in [t for _, _, _, t in CONFIGS] + ["bh"]:
        print(f"  {tag:<10} {full[tag]:+7.2f}%  (trades={full.get(tag+'_n','-')})", flush=True)

    # -------------------------------------------------------- leave-one-day-out
    print(f"\n== leave-one-day-out (drop exactly one date, {len(dates)} runs) ==", flush=True)
    loo = []
    for d in dates:
        sub = df[df["date"] != d].copy()
        r = run_matrix(sub)
        r["dropped"] = d
        loo.append(r)
    loo_df = pd.DataFrame(loo)
    g42 = loo_df["g42_cw"]
    print(f"  g42_cw over drop-one-day runs: min={g42.min():+.2f}%  max={g42.max():+.2f}%  "
          f"mean={g42.mean():+.2f}%  always>0: {(g42 > 0).all()}  always>BH: {(loo_df['g42_cw'] > loo_df['bh']).all()}",
          flush=True)
    print("  worst single-day drop (가장 민감한 날):")
    worst = loo_df.reindex(loo_df["g42_cw"].argsort())  # ascending
    for _, row in worst.head(3).iterrows():
        print(f"    drop {row['dropped']}: g42_cw={row['g42_cw']:+.2f}%  bh={row['bh']:+.2f}%", flush=True)

    # ------------------------------------------------------------- leave-5-out
    print(f"\n== leave-{n_drop}-out bootstrap (N={args.n}, seed={args.seed}) ==", flush=True)
    rows = []
    for i in range(args.n):
        keep = sorted(rng.choice(dates, size=len(dates) - n_drop, replace=False))
        sub = df[df["date"].isin(keep)].copy()
        r = run_matrix(sub)
        r["sample"] = i
        r["n_days"] = len(keep)
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n} done", flush=True)

    b = pd.DataFrame(rows)
    # 0.42/CW/SELLSIG 우위 정량화
    g42 = b["g42_cw"]
    better_bh = (g42 > b["bh"]).mean() * 100.0
    better_zero = (g42 > 0).mean() * 100.0
    better_g40 = (g42 > b["g40_cw"]).mean() * 100.0
    better_g42ew = (g42 > b["g42_ew"]).mean() * 100.0
    best_tag = b[[t for _, _, _, t in CONFIGS]].idxmax(axis=1).value_counts(normalize=True) * 100.0
    print(f"  g42_cw return  : mean={g42.mean():+.2f}%  median={g42.median():+.2f}%  "
          f"p5={g42.quantile(.05):+.2f}%  p95={g42.quantile(.95):+.2f}%", flush=True)
    print(f"  g42_cw>B&H    : {better_bh:.1f}%  |  g42_cw>0: {better_zero:.1f}%", flush=True)
    print(f"  g42_cw wins   : vs 0.40/CW {better_g40:.1f}%  |  vs 0.42/EW {better_g42ew:.1f}%", flush=True)
    print("  'best strategy' 빈도 (8개 전략 중):", flush=True)
    for tag, frac in best_tag.items():
        print(f"    {tag:<12} {frac:5.1f}%", flush=True)
    print(f"  g42_cw median trades: {int(b['g42_cw_n'].median())}  (p5={int(b['g42_cw_n'].quantile(.05))}, "
          f"p95={int(b['g42_cw_n'].quantile(.95))})", flush=True)

    # --------------------------------------------------------- weekly walk-fwd
    print("\n== weekly walk-forward (4 consecutive 5-day windows) ==", flush=True)
    for k in range(0, len(dates), 5):
        w = dates[k:k + 5]
        sub = df[df["date"].isin(w)].copy()
        r = run_matrix(sub)
        best = max([(r[t], t) for t, *_ in []]) if False else None
        best_tag_w = max(((r[t], t) for _, _, _, t in CONFIGS))[1]
        print(f"  wk {w[0]}..{w[-1]}: g42_cw={r['g42_cw']:+6.2f}%  bh={r['bh']:+6.2f}%  "
              f"best={best_tag_w}({r[best_tag_w]:+.2f}%)", flush=True)

    # --------------------------------------------------------------- persist
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        b[[c for c in b.columns if not c.endswith("_n")] + ["n_days", "sample"]].to_csv(
            args.out, index=False, encoding="utf-8-sig")
        print(f"\n[out] {args.out}  ({len(b)} rows)", flush=True)


if __name__ == "__main__":
    main()