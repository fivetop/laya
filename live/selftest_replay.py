#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live/selftest_replay.py — 라이브 증분 트레이더 골든 재현 테스트.

목적
----
live/trader.py 의 증분(incremental) 포트폴리오 트레이더가, 확정된 백테스트
(backtest/backtest_portfolio.py::run_portfolio, gate=0.42 / CW / SELLSIG,
initial_cash 10,000,000 / cost 0.00015 / max_positions 8) 결과를
data/portfolio_trades_golden_row7.csv 와 trade-by-trade 로 100% 재현하는지 검증한다.
"실전 투입 전 골든 테스트".

검증 항목
---------
1) 산출 트레이드 == 골든 CSV (16컬럼, 조인키 9개 + 수치 7개 within 1e-6) → 1,020/1,020
2) 지표가 표시정밀도에서 정확:
     ret% = +10.44 | MDD(bar)% = 7.93 | n_trades = 1,020 (EOD 598 + SELLSIG 422)
     win% = 40.0 | blocked = 48 | avg_pos = 0.67 | exposed% = 30.8
3) (앵커) 참조 엔진 run_portfolio 산출도 골든 CSV 와 일치(골든의 출처 확인)

실행 (프로젝트 루트, UTF-8):
    python -X utf8 live/selftest_replay.py

종료코드: 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from live.strategy import BarsSignalSource          # noqa: E402
from live.trader import (IncrementalTrader, TraderConfig,  # noqa: E402
                         compare_trades, trades_to_frame)

PRED_PATH = ROOT / "data" / "full_pred.parquet"
GOLDEN_PATH = ROOT / "data" / "portfolio_trades_golden_row7.csv"

# 골든 전체창(full_pred) — ts 인덱스가 골든과 동일하도록 전체 범위로 실행
START_DATE = "2025-01-06"
END_DATE = "2025-07-23"

# 기대 지표(확정 백테스트 row7)
EXP = {
    "total_return_pct": 10.44,
    "max_drawdown_pct": 7.93,
    "n_trades": 1020,
    "n_eod": 598,
    "n_sellsig": 422,
    "win_rate_pct": 40.0,
    "blocked_buys": 48,
    "avg_concurrent_positions": 0.67,
    "pct_bars_exposed": 30.8,
}


def load_reference():
    spec = importlib.util.spec_from_file_location(
        "backtest_portfolio", ROOT / "backtest" / "backtest_portfolio.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def main() -> int:
    checks = []  # (name, ok, detail)

    golden = pd.read_csv(GOLDEN_PATH)
    print("=== selftest_replay (live incremental trader == golden backtest) ===")
    print(f"pred   : {PRED_PATH.name}")
    print(f"golden : {GOLDEN_PATH.name}  ({len(golden)} rows)")

    # ---------------------------------------------------------------- 트레이더
    src = BarsSignalSource(str(PRED_PATH), START_DATE, END_DATE)
    cfg = TraderConfig(gate=0.42, sizing="CW", exit_mode="SELLSIG",
                       initial_cash=10_000_000.0, cost_per_side=0.00015,
                       max_positions=8)
    res = IncrementalTrader(cfg).run(src.events())
    mine = trades_to_frame(res["trades"])
    n_eod = int((mine["exit_reason"] == "EOD").sum())
    n_sig = int((mine["exit_reason"] == "SELLSIG").sum())

    print(f"trades : {len(mine)}  (EOD {n_eod} + SELLSIG {n_sig})")
    print(f"ret%   : {res['total_return_pct']:+.2f}  |  MDD(bar)%: {res['max_drawdown_pct']:.2f}  "
          f"|  win%: {res['win_rate_pct']:.1f}  |  blocked: {res['blocked_buys']}")
    print(f"avg_pos: {res['avg_concurrent_positions']:.2f}  |  exposed%: {res['pct_bars_exposed']:.1f}  "
          f"|  equity_final: {res['equity_final']:,.2f}")

    # ---------------------------------------------------- 1) 골든 트레이드 일치
    cmp = compare_trades(mine, golden, tol=1e-6)
    checks.append(("trades == golden (16 cols, tol 1e-6)", cmp["ok"], cmp["message"]))

    # ---------------------------------------------------- 2) 지표 표시정밀도 일치
    checks.append(("total_return_pct", round(res["total_return_pct"], 2) == EXP["total_return_pct"],
                   f"{res['total_return_pct']:.4f} vs {EXP['total_return_pct']}"))
    checks.append(("max_drawdown_pct(bar)", round(res["max_drawdown_pct"], 2) == EXP["max_drawdown_pct"],
                   f"{res['max_drawdown_pct']:.4f} vs {EXP['max_drawdown_pct']}"))
    checks.append(("n_trades", len(mine) == EXP["n_trades"],
                   f"{len(mine)} vs {EXP['n_trades']}"))
    checks.append(("n_eod", n_eod == EXP["n_eod"], f"{n_eod} vs {EXP['n_eod']}"))
    checks.append(("n_sellsig", n_sig == EXP["n_sellsig"], f"{n_sig} vs {EXP['n_sellsig']}"))
    checks.append(("win_rate_pct", round(res["win_rate_pct"], 1) == EXP["win_rate_pct"],
                   f"{res['win_rate_pct']:.4f} vs {EXP['win_rate_pct']}"))
    checks.append(("blocked_buys", res["blocked_buys"] == EXP["blocked_buys"],
                   f"{res['blocked_buys']} vs {EXP['blocked_buys']}"))
    checks.append(("avg_concurrent_positions",
                   round(res["avg_concurrent_positions"], 2) == EXP["avg_concurrent_positions"],
                   f"{res['avg_concurrent_positions']:.4f} vs {EXP['avg_concurrent_positions']}"))
    checks.append(("pct_bars_exposed", round(res["pct_bars_exposed"], 1) == EXP["pct_bars_exposed"],
                   f"{res['pct_bars_exposed']:.4f} vs {EXP['pct_bars_exposed']}"))

    # ---------------------------------------------------- 3) 앵커: 참조엔진==골든
    ref = load_reference()
    P = ref.prepare(src.df)
    ref_res = ref.run_portfolio(P, 0.42, "CW", "SELLSIG", 10_000_000.0, 0.00015, 8,
                                collect_trades=True)
    ref_cmp = compare_trades(trades_to_frame(ref_res["trades"]), golden, tol=1e-6)
    checks.append(("anchor: reference run_portfolio == golden", ref_cmp["ok"], ref_cmp["message"]))

    # ---------------------------------------------------- 4) 증분==참조(지표)
    mt = (abs(res["total_return_pct"] - ref_res["total_return_pct"]) < 1e-9
          and abs(res["max_drawdown_pct"] - ref_res["max_drawdown_pct"]) < 1e-9
          and abs(res["equity_final"] - ref_res["equity_final"]) < 1e-6)
    checks.append(("incremental metrics == reference metrics", mt,
                   f"ret {res['total_return_pct']:.10f} vs {ref_res['total_return_pct']:.10f}"))

    # ---------------------------------------------------- verdict
    print("--- verdict ---")
    all_ok = True
    for name, ok, detail in checks:
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if all_ok:
        print(f"\nPASS: live trader reproduces golden backtest EXACTLY "
              f"({cmp['n_matched']}/{len(golden)} trades, max|diff|={cmp['max_abs_diff']:.2e})")
        return 0
    print("\nFAIL: live trader does NOT reproduce the golden backtest.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
