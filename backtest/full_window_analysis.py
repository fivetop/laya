"""전체 101일(full_pred.parquet) 결과의 해석 검증.

1) 평가 구간 동치: GPU 예측에서 eval 20일만 떼어 엔진 실행 → 원래 결과(+8.05%) 재현 확인
2) 월별 분해: 0.42/CW/SELLSIG 의 일별 수익을 월별 합산 → in-sample(01~06) vs
   out-of-sample(06-25~07-23) 구간에서 수익이 어디서 났는지 판정

실행: python backtest/full_window_analysis.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CASH, COST, MP = 10_000_000.0, 0.00015, 8

_spec = importlib.util.spec_from_file_location(
    "backtest_portfolio", ROOT / "backtest" / "backtest_portfolio.py")
_bp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bp)  # type: ignore[union-attr]
prepare, run_portfolio = _bp.prepare, _bp.run_portfolio


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "full_pred.parquet")
    df["date"] = df["date"].astype(str)
    print(f"rows={len(df):,} dates={df['date'].nunique()} "
          f"({df['date'].min()}..{df['date'].max()})", flush=True)

    # ---------------------------------------------------------------- 1) equiv
    edates = sorted(df["date"].unique())[-20:]
    ev = df[df["date"].isin(edates)].copy()
    P_ev = prepare(ev)
    r_ev = run_portfolio(P_ev, 0.42, "CW", "SELLSIG", CASH, COST, MP)
    print(f"[equiv] eval-slice(20d) 0.42/CW/SELLSIG = {r_ev['total_return_pct']:+.2f}%  "
          f"(기존 로컬 예측 결과 +8.05%)", flush=True)

    # ------------------------------------------------------------- 2) monthly
    P = prepare(df)
    r7 = run_portfolio(P, 0.42, "CW", "SELLSIG", CASH, COST, MP, collect_trades=True)
    print(f"[row7 ] full-window 0.42/CW/SELLSIG = {r7['total_return_pct']:+.2f}%  "
          f"(trades={r7['n_trades']}, MDD={r7['max_drawdown_pct']:.2f}%)", flush=True)

    s = pd.Series(r7["daily_return_pct"], index=pd.to_datetime(r7["daily_dates"]))
    monthly = s.groupby(s.index.to_period("M")).agg(["sum", "count"])
    monthly.columns = ["ret_pct", "n_days"]
    print("\nmonthly (0.42/CW/SELLSIG daily returns, % ):")
    for m, row in monthly.iterrows():
        tag = "  <- OOS(eval)" if str(m) in ("2025-06", "2025-07") else ""
        print(f"  {m}  ret={row['ret_pct']:+7.2f}%  days={int(row['n_days'])}{tag}", flush=True)

    trades = pd.DataFrame(r7["trades"])
    trades["em"] = pd.to_datetime(trades["entry_date"]).dt.to_period("M")
    tm = trades.groupby("em").size()
    print("\nentry trades by month:", tm.to_dict(), flush=True)

    # eval(마지막 20일)에 해당하는 일별 수익 합
    ev_offset = (s.index >= pd.Timestamp("2025-06-25")).sum()
    oos = s[s.index >= pd.Timestamp("2025-06-25")]
    ins = s[s.index < pd.Timestamp("2025-06-25")]
    print(f"\n[OOS] eval-slice({len(oos)}일) row7 수익 합: {oos.sum():+.2f}%", flush=True)
    print(f"[INS] train-window({len(ins)}일) row7 수익 합: {ins.sum():+.2f}%", flush=True)


if __name__ == "__main__":
    main()