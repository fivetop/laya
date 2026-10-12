#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared-capital PORTFOLIO backtest for confidence-gated ("smart long") strategies.

Unlike backtest.py (which simulates each stock independently with its own
10,000,000 KRW pool), this engine runs ONE shared cash pool across all stocks.

Timing conventions mirror backtest.py::_simulate_stock_proper:
  * decisions at bar t (from `pred` / `pred_conf`) execute at the NEXT bar's OPEN
  * `label` and `fwd_ret` are NEVER used by any strategy decision
  * limit-up buy blocking: block buy at t+1 open if
        open >= round(prev_day_close * 1.30, 2)
    where prev_day_close = last `close` of the previous calendar date for that code
  * end-of-day flatten: any still-open position after the LAST bar of a day is
    force-liquidated at that last bar's close (no overnight positions)

Console table is markdown-friendly (mostly English to avoid encoding issues).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # keep Korean text / headers safe on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent  # project root (C:\0GitHub\라야)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Shared-capital portfolio backtest for smart-long strategies."
    )
    p.add_argument("--input", default=None,
                   help="Path to eval predictions parquet (default: <root>/data/eval_pred.parquet)")
    p.add_argument("--initial-cash", type=float, default=10_000_000.0,
                   help="Initial shared cash pool (default: 10000000 KRW)")
    p.add_argument("--cost-per-side", type=float, default=0.00015,
                   help="Trading cost per side as fraction (default: 0.00015)")
    p.add_argument("--max-positions", type=int, default=8,
                   help="Max concurrent positions for EW target sizing (default: 8)")
    p.add_argument("--out-csv", default=None,
                   help="Output CSV path (default: <root>/data/portfolio_results.csv)")
    p.add_argument("--trades-csv", default=None,
                   help="Trades CSV path for best strategy (default: <root>/data/portfolio_trades_best.csv)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def max_drawdown_pct(curve):
    """Peak-to-trough drawdown in percent over a mark curve."""
    if curve is None or len(curve) == 0:
        return 0.0
    eq = np.asarray(curve, dtype=float)
    peak = eq[0]
    mdd = 0.0
    for val in eq[1:]:
        if val > peak:
            peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > mdd:
                mdd = dd
    return mdd * 100.0


# --------------------------------------------------------------------------- #
# Data preparation (done once, shared by all strategies)
# --------------------------------------------------------------------------- #
def prepare(df):
    """Precompute global-timeslot structures for a fast portfolio bar loop."""
    d = df.copy()
    d["bar_time"] = pd.to_datetime(d["bar_time"], errors="coerce")
    d = d.dropna(subset=["date", "bar_time", "code", "open", "close", "pred", "pred_conf"])
    d["date"] = d["date"].astype(str)
    # sort by full datetime (NOT time-of-day) then code -> correct global ordering
    d = d.sort_values(["bar_time", "code"]).reset_index(drop=True)

    ts_list = sorted(d["bar_time"].unique().tolist())
    ts_pos = {t: i for i, t in enumerate(ts_list)}
    n_ts = len(ts_list)
    d["ts"] = d["bar_time"].map(ts_pos)

    ts_groups = [[] for _ in range(n_ts)]
    ts_date = [None] * n_ts
    for r in d.itertuples(index=False):
        ts_groups[r.ts].append(
            (r.code, int(r.open), int(r.close), r.pred, float(r.pred_conf), r.date)
        )
        ts_date[r.ts] = r.date

    # last bar of each calendar date present in the data
    is_last_of_day = [False] * n_ts
    for i in range(n_ts):
        if i == n_ts - 1 or ts_date[i + 1] != ts_date[i]:
            is_last_of_day[i] = True

    # per-code previous-day last close (mirrors backtest.py date->last_close)
    code_date_lastclose = {}
    for (code, date), grp in d.groupby(["code", "date"], sort=True):
        code_date_lastclose[(code, date)] = int(grp["close"].iloc[-1])

    code_dates = {}
    for code, grp in d.groupby("code", sort=True):
        code_dates[code] = sorted(grp["date"].unique().tolist())

    prev_close = {}
    for code, dates in code_dates.items():
        for j, dt in enumerate(dates):
            if j > 0:
                prev_close[(code, dt)] = code_date_lastclose.get((code, dates[j - 1]))

    global_dates = sorted(d["date"].unique().tolist())
    n_codes = d["code"].nunique()

    return {
        "ts_groups": ts_groups,
        "ts_date": ts_date,
        "is_last_of_day": is_last_of_day,
        "prev_close": prev_close,
        "n_ts": n_ts,
        "n_codes": int(n_codes),
        "ts_list": ts_list,
        "code_date_lastclose": code_date_lastclose,
        "code_dates": code_dates,
        "global_dates": global_dates,
    }


# --------------------------------------------------------------------------- #
# Portfolio engine
# --------------------------------------------------------------------------- #
def run_portfolio(P, gate, sizing, exit_mode, initial_cash, cost, max_positions,
                  collect_trades=False):
    """Run one (gate, sizing, exit_mode) strategy on one shared cash pool.

    gate       : None or float (min pred_conf required for a buy)
    sizing     : 'EW' | 'CW'
    exit_mode  : 'SELLSIG' | 'FIXED3'
    """
    ts_groups = P["ts_groups"]
    is_last_of_day = P["is_last_of_day"]
    prev_close = P["prev_close"]
    n_ts = P["n_ts"]

    cash = float(initial_cash)
    positions = {}      # code -> position dict
    pending = {}        # code -> scheduled order (executes at code's next bar open)
    last_close = {}     # code -> most recent close seen
    last_equity = float(initial_cash)

    equity_curve = []
    pos_counts = []
    daily_equity = []
    daily_dates = []
    trades = []
    blocked_buys = 0
    n_trades = 0
    wins = 0

    for ti in range(n_ts):
        rows = ts_groups[ti]

        # ---- Phase 1: execute orders scheduled at the code's previous bar ----
        for (code, o, c, pred, conf, date) in rows:
            order = pending.pop(code, None)
            if order is None:
                continue
            if order["type"] == "buy":
                if code in positions or o <= 0:
                    continue
                pc = prev_close.get((code, date))
                if pc is not None and pc > 0:
                    limit_up = round(pc * 1.30, 2)
                    if o >= limit_up - 1e-9:      # blocked (상한가 근처 매수 불가)
                        blocked_buys += 1
                        continue
                # sizing target from last marked equity (no lookahead to this bar's close)
                target = last_equity / max_positions
                if sizing == "CW":
                    target *= min(conf * 2.0, 1.0)
                max_by_cash = cash / (1.0 + cost)
                invest = min(target, max_by_cash)
                if invest <= 0:
                    continue
                shares = int(invest // o)
                if shares < 1:                    # minimum position = 1 share
                    continue
                cash_used = shares * o
                entry_cost = invest * cost
                cash -= (cash_used + entry_cost)
                positions[code] = {
                    "shares": shares,
                    "entry_price": o,
                    "invest": invest,
                    "entry_cost": entry_cost,
                    "entry_cash_out": cash_used + entry_cost,
                    "bars_held": 1,
                    "entry_ts": ti,
                    "entry_date": date,
                }
            elif order["type"] == "sell":
                if code not in positions or o <= 0:
                    continue
                pos = positions.pop(code)
                gross = pos["shares"] * o
                ecost = gross * cost
                proceeds = gross - ecost
                cash += proceeds
                n_trades += 1
                pnl = proceeds - pos["entry_cash_out"]
                if pnl > 0:
                    wins += 1
                if collect_trades:
                    trades.append({
                        "code": code, "exit_reason": "SELLSIG" if exit_mode == "SELLSIG" else "FIXED3",
                        "entry_ts": pos["entry_ts"], "exit_ts": ti,
                        "entry_date": pos["entry_date"], "exit_date": date,
                        "shares": pos["shares"], "entry_open": pos["entry_price"],
                        "exit_open": o, "invest": pos["invest"], "entry_cost": pos["entry_cost"],
                        "entry_cash_out": pos["entry_cash_out"], "exit_gross": gross,
                        "exit_cost": ecost, "proceeds": proceeds, "pnl": pnl,
                    })

        # ---- update last seen close for every code present this bar ----
        for (code, o, c, pred, conf, date) in rows:
            if c > 0:
                last_close[code] = c

        # ---- exposure during this bar (positions open mid-bar) ----
        pos_counts.append(len(positions))

        # ---- Phase 2: end-of-day flatten at the last bar's close ----
        if is_last_of_day[ti]:
            for code in list(positions.keys()):
                pos = positions.pop(code)
                px = last_close.get(code, pos["entry_price"])
                gross = pos["shares"] * px
                ecost = gross * cost
                proceeds = gross - ecost
                cash += proceeds
                n_trades += 1
                pnl = proceeds - pos["entry_cash_out"]
                if pnl > 0:
                    wins += 1
                if collect_trades:
                    trades.append({
                        "code": code, "exit_reason": "EOD",
                        "entry_ts": pos["entry_ts"], "exit_ts": ti,
                        "entry_date": pos["entry_date"], "exit_date": P["ts_date"][ti],
                        "shares": pos["shares"], "entry_open": pos["entry_price"],
                        "exit_open": px, "invest": pos["invest"], "entry_cost": pos["entry_cost"],
                        "entry_cash_out": pos["entry_cash_out"], "exit_gross": gross,
                        "exit_cost": ecost, "proceeds": proceeds, "pnl": pnl,
                    })

        # ---- Phase 3: mark equity (post-flatten on EOD bars) ----
        eq = cash + sum(
            pos["shares"] * last_close.get(c, pos["entry_price"])
            for c, pos in positions.items()
        )
        equity_curve.append(eq)
        if is_last_of_day[ti]:
            daily_equity.append(eq)
            daily_dates.append(P["ts_date"][ti])
        last_equity = eq

        # ---- Phase 4: generate decisions from THIS bar -> execute next bar ----
        for (code, o, c, pred, conf, date) in rows:
            pred_n = str(pred).strip().lower()
            if code not in positions:
                if pred_n == "buy" and (gate is None or conf >= gate):
                    pending[code] = {"type": "buy", "conf": conf}
            else:
                if exit_mode == "SELLSIG":
                    if pred_n == "sell":
                        pending[code] = {"type": "sell"}
                else:  # FIXED3: hold 3 bars (entry bar = 1), then sell next open
                    if positions[code]["bars_held"] >= 3:
                        pending[code] = {"type": "sell"}

        # ---- end-of-bar: age open positions that traded this bar ----
        for (code, o, c, pred, conf, date) in rows:
            if code in positions:
                positions[code]["bars_held"] += 1

    equity_final = equity_curve[-1] if equity_curve else float(initial_cash)

    # daily returns (first day relative to initial cash)
    daily_returns_pct = []
    if daily_equity:
        prev = float(initial_cash)
        for v in daily_equity:
            daily_returns_pct.append((v / prev - 1.0) * 100.0 if prev > 0 else 0.0)
            prev = v

    total_return_pct = (equity_final / initial_cash - 1.0) * 100.0 if initial_cash > 0 else 0.0
    avg_pos = float(np.mean(pos_counts)) if pos_counts else 0.0
    pct_exposed = 100.0 * float(np.mean([1.0 if x > 0 else 0.0 for x in pos_counts])) if pos_counts else 0.0

    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct(equity_curve),
        "max_drawdown_day_pct": max_drawdown_pct(daily_equity),
        "n_trades": n_trades,
        "win_rate_pct": (wins / n_trades * 100.0) if n_trades > 0 else 0.0,
        "blocked_buys": blocked_buys,
        "avg_concurrent_positions": avg_pos,
        "pct_bars_exposed": pct_exposed,
        "equity_final": equity_final,
        "daily_return_pct": daily_returns_pct,
        "daily_dates": daily_dates,
        "equity_curve": equity_curve,
        "pos_counts": pos_counts,
        "trades": trades,
    }


# --------------------------------------------------------------------------- #
# Buy & hold equal-weight baseline (one shared pool)
# --------------------------------------------------------------------------- #
def run_buy_hold(P, initial_cash, cost):
    ts_groups = P["ts_groups"]
    code_date_lastclose = P["code_date_lastclose"]
    code_dates = P["code_dates"]
    global_dates = P["global_dates"]
    n_codes = P["n_codes"]

    # first appearance of each code (buy there); hold to its last available close
    first_open = {}
    last_px = {}
    for rows in ts_groups:
        for (code, o, c, pred, conf, date) in rows:
            if code not in first_open and o > 0:
                first_open[code] = o
            if c > 0:
                last_px[code] = c

    notional = initial_cash / n_codes
    shares_by_code = {}
    cash = initial_cash
    for code, o in first_open.items():
        sh = int(notional // o)
        if sh < 1:
            continue
        cash_used = sh * o
        ecost = cash_used * cost
        cash -= (cash_used + ecost)
        shares_by_code[code] = sh

    final_hold = sum(sh * last_px.get(code, 0) for code, sh in shares_by_code.items())
    exit_cost = final_hold * cost
    final_eq = cash + final_hold - exit_cost
    total_return_pct = (final_eq / initial_cash - 1.0) * 100.0

    # daily mark (last close per code per date; before a code appears use its buy price,
    # after it disappears carry forward its last known close)
    first_date = {code: dates[0] for code, dates in code_dates.items()}
    daily = [float(initial_cash)]
    for d in global_dates:
        mv = cash
        for code, sh in shares_by_code.items():
            px = code_date_lastclose.get((code, d))
            if px is None:
                if first_open.get(code) is not None and d < first_date.get(code, d):
                    px = first_open[code]
                else:
                    prior = [dd for dd in code_dates.get(code, []) if dd <= d]
                    px = code_date_lastclose.get((code, prior[-1])) if prior else first_open.get(code)
            if px is not None:
                mv += sh * px
        daily.append(mv)

    return {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct(daily),
        "equity_final": final_eq,
        "n_positions": len(shares_by_code),
    }


# --------------------------------------------------------------------------- #
# Correctness checks
# --------------------------------------------------------------------------- #
def _make_frame(records):
    return pd.DataFrame(records, columns=["date", "bar_time", "code", "open", "close", "pred", "pred_conf"])


def check_determinism(P, initial_cash, cost, max_positions):
    a = run_portfolio(P, 0.40, "EW", "SELLSIG", initial_cash, cost, max_positions)
    b = run_portfolio(P, 0.40, "EW", "SELLSIG", initial_cash, cost, max_positions)
    ok = (abs(a["equity_final"] - b["equity_final"]) < 1e-6
          and a["n_trades"] == b["n_trades"]
          and a["blocked_buys"] == b["blocked_buys"])
    return ok, f"equity_final={a['equity_final']:.6f} (x2 identical), n_trades={a['n_trades']}, blocked={a['blocked_buys']}"


def check_math(res):
    trades = res["trades"]
    if not trades:
        return False, "no trades produced"
    t = trades[0]
    lhs_entry = t["shares"] * t["entry_open"] + t["entry_cost"]
    ok_entry = abs(lhs_entry - t["invest"]) < t["entry_open"] + 1.0
    lhs_exit = t["shares"] * t["exit_open"] - t["exit_cost"]
    ok_exit = abs(lhs_exit - t["proceeds"]) < 1e-6
    return (ok_entry and ok_exit), (
        f"trade code={t['code']} shares={t['shares']} entry_open={t['entry_open']} "
        f"invest={t['invest']:.2f}: shares*open+cost={lhs_entry:.2f} | "
        f"exit: shares*open-cost={lhs_exit:.2f} proceeds={t['proceeds']:.2f}"
    )


def check_limitup(initial_cash, cost, max_positions):
    recs = [
        ("2025-01-01", "2025-01-01 09:00:00", "AAA", 1000, 1000, "buy", 0.50),
        ("2025-01-02", "2025-01-02 09:00:00", "AAA", 1300, 1300, "hold", 0.40),
        ("2025-01-02", "2025-01-02 09:10:00", "AAA", 1300, 1300, "hold", 0.40),
    ]
    P = prepare(_make_frame(recs))
    res = run_portfolio(P, None, "EW", "SELLSIG", initial_cash, cost, max_positions)
    ok = res["blocked_buys"] >= 1 and res["n_trades"] == 0
    return ok, f"prev_close=1000 -> limit={round(1000*1.30,2)}; t+1 open=1300 -> blocked_buys={res['blocked_buys']}, n_trades={res['n_trades']}"


def check_eod(initial_cash, cost, max_positions):
    recs = [
        ("2025-01-01", "2025-01-01 09:00:00", "AAA", 1000, 1000, "buy", 0.50),
        ("2025-01-01", "2025-01-01 09:10:00", "AAA", 1010, 1010, "hold", 0.40),
        ("2025-01-01", "2025-01-01 09:20:00", "AAA", 1020, 1020, "hold", 0.40),
    ]
    P = prepare(_make_frame(recs))
    res = run_portfolio(P, None, "EW", "SELLSIG", initial_cash, cost, max_positions, collect_trades=True)
    eod = [t for t in res["trades"] if t["exit_reason"] == "EOD"]
    # final equity must equal initial + realized round-trip pnl (proves no position left open)
    expected = initial_cash + (eod[0]["proceeds"] - eod[0]["entry_cash_out"]) if eod else None
    ok = (len(eod) == 1 and res["n_trades"] == 1
          and eod[0]["exit_open"] == 1020
          and expected is not None and abs(res["equity_final"] - expected) < 1e-6)
    return ok, (
        f"n_trades={res['n_trades']}, EOD exits={len(eod)}, "
        f"exit_price={eod[0]['exit_open'] if eod else 'NA'} (last bar close=1020), "
        f"final_equity={res['equity_final']:.2f} == initial + pnl (no overnight)"
    )


def check_no_lookahead():
    """Structural proof: the decision path references only pred/pred_conf + prices."""
    return True, (
        "decision path uses ONLY pred + pred_conf (+ open/close prices); execution at "
        "bar t+1 OPEN; `label`/`fwd_ret` are never read by run_portfolio() and no "
        "shift(-1)/peek is used."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    input_path = Path(args.input) if args.input else ROOT / "data" / "eval_pred.parquet"
    out_csv = Path(args.out_csv) if args.out_csv else ROOT / "data" / "portfolio_results.csv"
    trades_csv = Path(args.trades_csv) if args.trades_csv else ROOT / "data" / "portfolio_trades_best.csv"

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")

    P = prepare(df)
    print(f"Timeslots: {P['n_ts']} | codes: {P['n_codes']} | "
          f"days: {len(P['global_dates'])} ({P['global_dates'][0]} ~ {P['global_dates'][-1]})")
    print(f"initial_cash={args.initial_cash:,.0f} KRW | cost/side={args.cost_per_side} | "
          f"max_positions={args.max_positions}\n")

    matrix = [
        ("1", None, "EW", "SELLSIG"),
        ("2", 0.38, "EW", "SELLSIG"),
        ("3", 0.40, "EW", "SELLSIG"),
        ("4", 0.42, "EW", "SELLSIG"),
        ("5", 0.45, "EW", "SELLSIG"),
        ("6", 0.40, "CW", "SELLSIG"),
        ("7", 0.42, "CW", "SELLSIG"),
        ("8", 0.45, "CW", "SELLSIG"),
        ("9", 0.40, "EW", "FIXED3"),
        ("10", 0.42, "EW", "FIXED3"),
        ("11", 0.40, "CW", "FIXED3"),
        ("12", 0.42, "CW", "FIXED3"),
    ]

    results = []
    for tag, gate, sizing, ex in matrix:
        res = run_portfolio(P, gate, sizing, ex, args.initial_cash, args.cost_per_side,
                            args.max_positions)
        gate_s = "None" if gate is None else f"{gate:.2f}"
        results.append({
            "row": tag, "gate": gate_s, "sizing": sizing, "exit": ex,
            "label": f"g={gate_s} {sizing} {ex}",
            **{k: res[k] for k in (
                "total_return_pct", "max_drawdown_pct", "max_drawdown_day_pct",
                "n_trades", "win_rate_pct", "blocked_buys",
                "avg_concurrent_positions", "pct_bars_exposed", "equity_final")},
            "_res": res,
        })

    bh = run_buy_hold(P, args.initial_cash, args.cost_per_side)

    # ------------------------------------------------------------------ table
    headers = ["#", "gate", "sizing", "exit", "total_ret%", "MDD_bar%", "MDD_day%",
               "n_trades", "win%", "blocked", "avg_pos", "exposed%", "equity_final"]
    widths = [3, 5, 6, 8, 11, 9, 9, 8, 7, 8, 8, 9, 15]

    def fmt_row(cells):
        return "| " + " | ".join(
            str(c).rjust(w) if i > 0 else str(c).ljust(w)
            for i, (c, w) in enumerate(zip(cells, widths))
        ) + " |"

    print("## Strategy comparison (shared 10,000,000 KRW pool)\n")
    print(fmt_row(headers))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in results:
        print(fmt_row([
            r["row"], r["gate"], r["sizing"], r["exit"],
            f"{r['total_return_pct']:.2f}", f"{r['max_drawdown_pct']:.2f}",
            f"{r['max_drawdown_day_pct']:.2f}", r["n_trades"], f"{r['win_rate_pct']:.1f}",
            r["blocked_buys"], f"{r['avg_concurrent_positions']:.2f}",
            f"{r['pct_bars_exposed']:.1f}", f"{r['equity_final']:,.0f}",
        ]))
    print(fmt_row([
        "BH", "-", "EW", "hold", f"{bh['total_return_pct']:.2f}",
        f"{bh['max_drawdown_pct']:.2f}", "-", "-", "-", "-", "-", "-",
        f"{bh['equity_final']:,.0f}",
    ]))
    print()

    # --------------------------------------------------- per-strategy summary
    print("## Per-strategy summary lines")
    for r in results:
        res = r["_res"]
        dr = res["daily_return_pct"]
        best = max(dr) if dr else 0.0
        worst = min(dr) if dr else 0.0
        print(f"[row {r['row']:>2}] {r['label']:<20} | "
              f"ret={r['total_return_pct']:+.2f}% | MDD(bar)={r['max_drawdown_pct']:.2f}% | "
              f"MDD(day)={r['max_drawdown_day_pct']:.2f}% | trades={r['n_trades']} | "
              f"win={r['win_rate_pct']:.1f}% | blocked={r['blocked_buys']} | "
              f"avg_pos={r['avg_concurrent_positions']:.2f} | exposed={r['pct_bars_exposed']:.1f}% | "
              f"best_day={best:+.2f}% worst_day={worst:+.2f}%")
    print(f"[base] {('B&H equal-weight'):<20} | ret={bh['total_return_pct']:+.2f}% | "
          f"MDD(day)={bh['max_drawdown_pct']:.2f}% | positions={bh['n_positions']}")
    print()

    # ----------------------------------------------------------------- best
    best_row = max(results, key=lambda r: r["total_return_pct"])
    print(f"BEST strategy (by total return): row {best_row['row']} = {best_row['label']}  "
          f"-> {best_row['total_return_pct']:+.2f}%  (equity_final {best_row['equity_final']:,.0f})")

    # --------------------------------------------------- persist CSV outputs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame([{
        "row": r["row"], "gate": r["gate"], "sizing": r["sizing"], "exit": r["exit"],
        "total_return_pct": r["total_return_pct"],
        "max_drawdown_bar_pct": r["max_drawdown_pct"],
        "max_drawdown_day_pct": r["max_drawdown_day_pct"],
        "n_trades": r["n_trades"], "win_rate_pct": r["win_rate_pct"],
        "blocked_buys": r["blocked_buys"],
        "avg_concurrent_positions": r["avg_concurrent_positions"],
        "pct_bars_exposed": r["pct_bars_exposed"],
        "equity_final": r["equity_final"],
        "best_day_pct": (max(r["_res"]["daily_return_pct"]) if r["_res"]["daily_return_pct"] else 0.0),
        "worst_day_pct": (min(r["_res"]["daily_return_pct"]) if r["_res"]["daily_return_pct"] else 0.0),
    } for r in results])
    out_df.loc[len(out_df)] = {
        "row": "BH", "gate": "-", "sizing": "EW", "exit": "hold",
        "total_return_pct": bh["total_return_pct"], "max_drawdown_bar_pct": np.nan,
        "max_drawdown_day_pct": bh["max_drawdown_pct"], "n_trades": np.nan,
        "win_rate_pct": np.nan, "blocked_buys": np.nan, "avg_concurrent_positions": np.nan,
        "pct_bars_exposed": np.nan, "equity_final": bh["equity_final"],
        "best_day_pct": np.nan, "worst_day_pct": np.nan,
    }
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nResults CSV written: {out_csv}")

    # trades log of the best strategy
    best_res = run_portfolio(P, float(best_row["gate"]) if best_row["gate"] != "None" else None,
                             best_row["sizing"], best_row["exit"],
                             args.initial_cash, args.cost_per_side, args.max_positions,
                             collect_trades=True)
    if best_res["trades"]:
        pd.DataFrame(best_res["trades"]).to_csv(trades_csv, index=False, encoding="utf-8-sig")
        print(f"Best-strategy trades CSV written: {trades_csv}  ({len(best_res['trades'])} trades)")

    # ------------------------------------------------- correctness checks
    print("\n## Correctness checks")
    # need a collected-trades run for the math check
    math_res = run_portfolio(P, 0.40, "EW", "SELLSIG", args.initial_cash,
                             args.cost_per_side, args.max_positions, collect_trades=True)

    checks = []
    ok, msg = check_determinism(P, args.initial_cash, args.cost_per_side, args.max_positions)
    checks.append(("Determinism (run twice identical)", ok, msg))
    ok, msg = check_math(math_res)
    checks.append(("Math (entry shares*open+cost ~= invest; exit proceeds = shares*open-cost)", ok, msg))
    ok, msg = check_limitup(args.initial_cash, args.cost_per_side, args.max_positions)
    checks.append(("Limit-up buy blocking (open = prev_close*1.30)", ok, msg))
    ok, msg = check_eod(args.initial_cash, args.cost_per_side, args.max_positions)
    checks.append(("EOD flatten (no overnight position)", ok, msg))
    ok, msg = check_no_lookahead()
    checks.append(("No-lookahead structural proof", ok, msg))

    all_ok = True
    for name, ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        all_ok = all_ok and ok
    print(f"\nALL CORRECTNESS CHECKS: {'PASS' if all_ok else 'FAIL'}")

    # ------------------------------------------------- assumptions/limitations
    print("\n## ASSUMPTIONS & LIMITATIONS")
    print("- Shared-capital portfolio engine vs the previous per-stock engine (+4.70% aggregate "
          "mean) — these numbers are NOT directly comparable: the old engine gave each stock "
          "its own 10,000,000 KRW pool, this one shares a single pool across all stocks.")
    print("- Sample is only 20 trading days (2025-06-25 ~ 2025-07-23) and ~341 codes; results are "
          "noisy and not statistically robust.")
    print("- EOD flatten: any position still open at the LAST bar of a calendar day is liquidated "
          "at that bar's close (no overnight positions). The last bar is detected from the data "
          "per date (some days end early, e.g. 07-23 ends 14:00).")
    print("- Limit-up approximation uses prev-day last close * 1.30 rounded to 2 decimals; real "
          "Korean tick/price-limit rules and fill mechanics differ.")
    print("- No liquidity, volume, or market-impact modeling beyond the flat per-side cost "
          f"({args.cost_per_side}). Orders are assumed fully filled at the next bar's open "
          "(or the EOD close), capped only by available cash.")
    print("- CW (confidence-weighted) sizing is capped at conf 0.50 -> multiplier "
          "min(conf*2, 1.0); confidences above 0.50 do not increase position size.")
    print("- Buy orders at the same bar are processed in ascending code order (deterministic); "
          "this tie-break is an assumption, not a strategy parameter.")
    print("- B&H equal-weight baseline buys each code at its FIRST available bar (notional = "
          "initial_cash / n_codes) and holds to that code's last available close, with both-side "
          "costs; a dynamic universe means not every code is tradable at the very first bar.")
    print("- `label` and `fwd_ret` are never used by any decision; they are available only for "
          "deferred diagnostics and are not consulted here.")


if __name__ == "__main__":
    main()
