#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest engine for 10-minute bar signals with trading costs (long-only)."
    )
    parser.add_argument(
        "--input",
        default="data/labeled.parquet",
        help="Path to labeled parquet (default: data/labeled.parquet)",
    )
    parser.add_argument(
        "--cost-per-side",
        type=float,
        default=0.00015,
        help="Trading cost per side as fraction (default: 0.00015 = 0.015%%)",
    )
    parser.add_argument(
        "--signal-col",
        default="label",
        help="Signal column name (default: label)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=10_000_000.0,
        help="Initial cash per stock simulation (default: 10_000_000 KRW)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path",
    )
    return parser.parse_args()


def generate_synthetic_data(output_path: Path, n_stocks=2, n_days=20, bars_per_day=39):
    """Generate small synthetic labeled parquet for verification."""
    rng = np.random.default_rng(42)
    codes = [f"STK{i:03d}" for i in range(n_stocks)]
    # 10-min bar times roughly 09:00-15:30 KST (39 bars)
    bar_times = pd.date_range("09:00", periods=bars_per_day, freq="10min").time

    rows = []
    for code in codes:
        # random walk prices
        price = 100.0
        for day_idx in range(n_days):
            date_str = f"2026-01-{day_idx + 1:02d}"
            for b, bt in enumerate(bar_times):
                # small noise
                price = price * (1.0 + rng.normal(0, 0.001))
                open_p = price
                close_p = price * (1.0 + rng.normal(0, 0.0005))
                high_p = max(open_p, close_p) * (1.0 + abs(rng.normal(0, 0.0003)))
                low_p = min(open_p, close_p) * (1.0 + abs(rng.normal(0, 0.0003)))
                volume = int(rng.integers(1000, 10000))
                n_ticks = int(rng.integers(10, 100))
                # fwd_ret not used for decisions; include for realism
                fwd_ret = rng.normal(0, 0.002)
                # label consistent-ish with fwd_ret (simple heuristic)
                if fwd_ret > 0.001:
                    label = "buy"
                elif fwd_ret < -0.001:
                    label = "sell"
                else:
                    label = "hold"
                rows.append(
                    {
                        "date": date_str,
                        "bar_time": bt,
                        "code": code,
                        "name": f"{code}_name",
                        "open": int(round(open_p * 100)) / 100.0,
                        "high": int(round(high_p * 100)) / 100.0,
                        "low": int(round(low_p * 100)) / 100.0,
                        "close": int(round(close_p * 100)) / 100.0,
                        "volume": volume,
                        "n_ticks": n_ticks,
                        "fwd_ret": fwd_ret,
                        "label": label,
                    }
                )
                price = close_p

    df = pd.DataFrame(rows)
    # ensure datetime-like bar_time
    df["bar_time"] = pd.to_datetime(df["bar_time"].astype(str), format="%H:%M:%S").dt.time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df


def max_drawdown(equity_curve):
    if len(equity_curve) == 0:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    peak = eq[0] if eq.size > 0 else 0.0
    mdd = 0.0
    for val in eq[1:]:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd * 100.0  # percent


def simulate_stock(df_stock, cost_per_side, signal_col, initial_cash):
    """Simulate long-only per stock. Decisions use labels at t -> execute at t+1 open (no lookahead)."""
    df = df_stock.copy()
    # sort chronologically
    df = df.sort_values(["date", "bar_time"]).reset_index(drop=True)

    if len(df) < 2:
        return {
            "total_return_pct": 0.0,
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "buy_hold_return_pct": 0.0,
            "equity_final": initial_cash,
            "equity_curve": [initial_cash],
        }

    cash = initial_cash
    position = 0.0  # shares
    in_position = False
    trades = 0
    wins = 0
    equity_curve = [cash]

    # buy&hold: buy at first open, sell at last close (costs on both sides)
    first_open = df.iloc[0]["open"]
    last_close = df.iloc[-1]["close"]
    bh_shares = initial_cash / first_open if first_open > 0 else 0.0
    bh_notional_entry = bh_shares * first_open
    bh_notional_exit = bh_shares * last_close
    bh_costs = (bh_notional_entry + bh_notional_exit) * cost_per_side * 2.0  # entry+exit
    bh_final = bh_notional_exit - bh_costs
    buy_hold_return_pct = ((bh_final / initial_cash) - 1.0) * 100.0 if initial_cash > 0 else 0.0

    for i in range(len(df) - 1):
        # execute signal from current bar at next bar open (t -> t+1 open)
        sig = df.iloc[i][signal_col]
        if pd.isna(sig):
            sig = "hold"
        sig = str(sig).strip().lower()

        next_open = df.iloc[i + 1]["open"]
        if next_open <= 0:
            equity_curve.append(cash + position * df.iloc[i]["close"])
            continue

        # entry
        if not in_position and sig == "buy":
            # go long all-in at next open
            notional = cash
            cost_entry = notional * cost_per_side
            shares = (notional - cost_entry) / next_open
            if shares > 0:
                position = shares
                cash = 0.0
                in_position = True
                trades += 1
                entry_price = next_open
                entry_cash = notional  # for win/loss tracking of this round-trip
                # store entry info for this trade
                df.at[i + 1, "_entry_price"] = entry_price
                df.at[i + 1, "_entry_cash"] = entry_cash
                df.at[i + 1, "_trade_open"] = True
        # exit
        elif in_position and sig == "sell":
            # flatten at next open
            notional_exit = position * next_open
            cost_exit = notional_exit * cost_per_side
            proceeds = notional_exit - cost_exit
            # determine if this round-trip was win
            # find last entry cash/price
            entry_cash = df[df["_trade_open"] == True]["_entry_cash"].iloc[-1] if "_trade_open" in df.columns else initial_cash
            # simpler: track last entry
            if "_last_entry_cash" not in locals() and "_last_entry_cash" not in globals():
                pass
            # easier tracking with variables
            pass
            # recompute win using stored last entry
            last_entry_cash = getattr(simulate_stock, "_last_entry_cash", initial_cash)
            # but easier to use local by closure; store on df by scanning? simpler: track
            cash_new = proceeds
            # find last open trade entry cash
            open_entries = df[(df["_trade_open"] == True)]
            if len(open_entries) > 0:
                last_entry_cash = open_entries.iloc[-1]["_entry_cash"]
            else:
                last_entry_cash = initial_cash
            # close trade
            trades_counted = 1  # this is closing trade action
            if proceeds > last_entry_cash:
                wins += 1
            # mark trades closed
            df.loc[df["_trade_open"] == True, "_trade_open"] = False
            cash = cash_new
            position = 0.0
            in_position = False
            trades += 0  # trades already incremented on entry? wait - entry increments trades; closing is part of round-trip
            # better: increment trades only on round-trip completion? spec says "n_trades" - count trades (round-trips) typically
            # but easier: count entry as trade start; count round-trip on exit - increment trades on exit? or count both sides.
            # "n_trades, win_rate (% of round-trips with net profit)" - round-trips => each completed round-trip counts as 1 trade
            pass
        # hold - no change

        # update equity
        equity_curve.append(cash + position * df.iloc[i]["close"])

    # if ended in position, equity is cash+position*last close (no forced close counted as trade)
    equity_curve.append(cash + position * df.iloc[-1]["close"])
    equity_final = equity_curve[-1] if equity_curve else initial_cash

    # recompute trades/wins properly by simulating with proper tracking
    return _simulate_stock_proper(df_stock, cost_per_side, signal_col, initial_cash)


def _simulate_stock_proper(df_stock, cost_per_side, signal_col, initial_cash):
    """Simulate long-only per stock with limit-up buy blocking.

    Limit-up rule (Korean stocks, conservative approximation):
    - 상한가 가격 = 전일 종가(prev close) * 1.30 (코스피/코스닥 공통 가정)
    - 가격 반올림은 간단한 round() 사용. 실전 키움 체결가 계식과 미세차이 가능, 보수적 근사
    - 매수 시점이 t+1 open 체결이므로 핵심: t 시점 buy 시그널 → t+1의 open이 상한가(open >= limit-up price)이면 체결 불가
    - 또한 시그널 bar 자체가 상한가 상황(close >= limit-up 등)인 경우도 차단하되, 체결 bar 기준(open >= limit-up)이 핵심
    - sell은 체결 가능(상한가에서 보유분 매도는 가능) — sell 경로는 변경하지 않음
    """
    df = df_stock.copy()
    df = df.sort_values(["date", "bar_time"]).reset_index(drop=True)

    if len(df) < 2:
        return {
            "total_return_pct": 0.0,
            "n_trades": 0,
            "win_rate_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "buy_hold_return_pct": 0.0,
            "equity_final": initial_cash,
            "equity_curve": [initial_cash],
            "blocked_buys": 0,
        }

    cash = initial_cash
    position = 0.0
    in_position = False
    n_trades = 0  # round-trips
    wins = 0
    entry_cash = 0.0
    blocked_buys = 0
    equity_curve = []

    # buy&hold
    first_open = df.iloc[0]["open"]
    last_close = df.iloc[-1]["close"]
    bh_shares = initial_cash / first_open if first_open > 0 else 0.0
    bh_notional_entry = bh_shares * first_open
    bh_notional_exit = bh_shares * last_close
    bh_costs = (bh_notional_entry + bh_notional_exit) * cost_per_side * 2.0
    bh_final = bh_notional_exit - bh_costs
    buy_hold_return_pct = ((bh_final / initial_cash) - 1.0) * 100.0 if initial_cash > 0 else 0.0

    # build date -> last close mapping for prev day close lookup
    date_groups = df.groupby("date")
    ordered_dates = list(date_groups.groups.keys())
    date_last_close = {d: date_groups.get_group(d).iloc[-1]["close"] for d in ordered_dates}
    date_to_idx = {d: i for i, d in enumerate(ordered_dates)}

    for i in range(len(df) - 1):
        sig = df.iloc[i][signal_col]
        if pd.isna(sig):
            sig = "hold"
        sig = str(sig).strip().lower()
        next_open = df.iloc[i + 1]["open"]
        if next_open <= 0:
            equity_curve.append(cash + position * df.iloc[i]["close"])
            continue

        if not in_position and sig == "buy":
            # determine prev day close for limit-up check
            # execution is at t+1 open; limit-up is based on previous trading day's closing price
            # For Korean stocks, 상한가 = 전일 종가 * 1.30
            exec_date = df.iloc[i + 1]["date"]
            exec_date_idx = date_to_idx.get(exec_date)
            prev_day_close = None
            if exec_date_idx is not None and exec_date_idx > 0:
                prev_day = ordered_dates[exec_date_idx - 1]
                prev_day_close = date_last_close.get(prev_day)
            # Also consider signal bar's context? But execution bar (t+1) is key
            blocked = False
            if prev_day_close is not None and prev_day_close > 0:
                limit_up_price = round(prev_day_close * 1.30, 2)  # conservative approx
                if next_open >= limit_up_price - 1e-9:
                    blocked = True
            if blocked:
                blocked_buys += 1
            else:
                notional = cash
                cost_entry = notional * cost_per_side
                shares = (notional - cost_entry) / next_open
                if shares > 0:
                    position = shares
                    cash = 0.0
                    in_position = True
                    entry_cash = notional
        elif in_position and sig == "sell":
            notional_exit = position * next_open
            cost_exit = notional_exit * cost_per_side
            proceeds = notional_exit - cost_exit
            n_trades += 1
            if proceeds > entry_cash:
                wins += 1
            cash = proceeds
            position = 0.0
            in_position = False
            entry_cash = 0.0
        # hold: no-op

        equity_curve.append(cash + position * df.iloc[i]["close"])

    equity_curve.append(cash + position * df.iloc[-1]["close"])
    equity_final = equity_curve[-1] if equity_curve else initial_cash
    total_return_pct = ((equity_final / initial_cash) - 1.0) * 100.0 if initial_cash > 0 else 0.0
    win_rate_pct = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
    mdd_pct = max_drawdown(equity_curve) if equity_curve else 0.0

    return {
        "total_return_pct": total_return_pct,
        "n_trades": n_trades,
        "win_rate_pct": win_rate_pct,
        "max_drawdown_pct": mdd_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "equity_final": equity_final,
        "equity_curve": equity_curve,
        "blocked_buys": blocked_buys,
    }


def main():
    args = parse_args()

    # If input doesn't exist, maybe generate synthetic in temp and run? But spec says verify with synthetic.
    input_path = Path(args.input)
    if not input_path.exists():
        # try to generate synthetic at temp location as fallback? but follow spec: verify explicitly
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        print("Generate synthetic data first or provide valid --input.", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"ERROR: Failed to read parquet: {e}", file=sys.stderr)
        sys.exit(1)

    if df.empty:
        print("Empty input data.")
        sys.exit(0)

    required_cols = ["date", "bar_time", "code", "open", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.signal_col not in df.columns:
        print(f"ERROR: Missing signal column '{args.signal_col}'", file=sys.stderr)
        sys.exit(1)

    # drop rows with NaN in critical fields? skip NaN rows as per spec
    df = df.dropna(subset=["date", "bar_time", "code", "open", "close", args.signal_col])

    if df.empty:
        print("No valid rows after dropping NaN.")
        sys.exit(0)

    # ensure types (bar_time may be "HH:MM:SS" 문자열, datetime.time, 또는 full datetime)
    try:
        df["bar_time"] = pd.to_datetime(df["bar_time"].astype(str), format="%H:%M:%S")
    except (ValueError, TypeError):
        df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce")
    df = df.dropna(subset=["bar_time"])

    results = []
    for code, g in df.groupby("code"):
        res = _simulate_stock_proper(g, args.cost_per_side, args.signal_col, args.initial_cash)
        results.append(
            {
                "code": code,
                "total_return_pct": res["total_return_pct"],
                "n_trades": res["n_trades"],
                "win_rate_pct": res["win_rate_pct"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "buy_hold_return_pct": res["buy_hold_return_pct"],
                "equity_final": res["equity_final"],
            }
        )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No results.")
        sys.exit(0)

    # aggregate
    agg = {
        "code": "AGGREGATE",
        "total_return_pct": res_df["total_return_pct"].mean(),
        "n_trades": int(res_df["n_trades"].sum()),
        "win_rate_pct": (res_df["wins"].sum() / res_df["n_trades"].sum() * 100.0) if "wins" in res_df else (res_df["win_rate_pct"].mean()),
        "max_drawdown_pct": res_df["max_drawdown_pct"].mean(),
        "buy_hold_return_pct": res_df["buy_hold_return_pct"].mean(),
        "equity_final": res_df["equity_final"].mean(),
    }
    # recompute aggregate win rate properly
    total_trades = res_df["n_trades"].sum()
    # need wins per stock - easier to recompute by storing wins; simpler: recompute by running again is wasteful
    # but we have win_rate_pct and n_trades - approximate aggregate win rate as weighted? spec says "win_rate (% of round-trips with net profit)" - aggregate across all round-trips => total wins/total trades
    wins_total = 0
    # recompute wins by re-simulating quickly? easier to track wins in results
    pass

    # better to store wins
    res_df2 = []
    for code, g in df.groupby("code"):
        res = _simulate_stock_proper(g, args.cost_per_side, args.signal_col, args.initial_cash)
        res_df2.append(
            {
                "code": code,
                "total_return_pct": res["total_return_pct"],
                "n_trades": res["n_trades"],
                "wins": (res["win_rate_pct"] * res["n_trades"] / 100.0) if res["n_trades"] > 0 else 0,
                "win_rate_pct": res["win_rate_pct"],
                "max_drawdown_pct": res["max_drawdown_pct"],
                "buy_hold_return_pct": res["buy_hold_return_pct"],
                "equity_final": res["equity_final"],
                "blocked_buys": res.get("blocked_buys", 0),
            }
        )
    res_df2 = pd.DataFrame(res_df2)
    wins_total = int(round(res_df2["wins"].sum())) if "wins" in res_df2 else 0
    total_trades = res_df2["n_trades"].sum()
    agg_win_rate = (wins_total / total_trades * 100.0) if total_trades > 0 else 0.0
    agg_row = {
        "code": "AGGREGATE",
        "total_return_pct": res_df2["total_return_pct"].mean(),
        "n_trades": int(total_trades),
        "wins": wins_total,
        "win_rate_pct": agg_win_rate,
        "max_drawdown_pct": res_df2["max_drawdown_pct"].mean(),
        "buy_hold_return_pct": res_df2["buy_hold_return_pct"].mean(),
        "equity_final": res_df2["equity_final"].mean(),
        "blocked_buys": int(res_df2["blocked_buys"].sum()) if "blocked_buys" in res_df2 else 0,
    }
    out_df = pd.concat([res_df2, pd.DataFrame([agg_row])], ignore_index=True)
    # format for display
    display_cols = ["code", "total_return_pct", "n_trades", "win_rate_pct", "max_drawdown_pct", "buy_hold_return_pct", "equity_final", "blocked_buys"]
    print(out_df[display_cols].to_string(index=False, float_format="%.2f"))
    print("\nSummary:")
    print(f"  total_trades (aggregate): {int(total_trades)}")
    print(f"  win_rate (aggregate): {agg_win_rate:.2f}%")
    print(f"  avg_total_return_pct: {res_df2['total_return_pct'].mean():.2f}%")
    print(f"  avg_buy_hold_return_pct: {res_df2['buy_hold_return_pct'].mean():.2f}%")
    print(f"  avg_max_drawdown_pct: {res_df2['max_drawdown_pct'].mean():.2f}%")
    print(f"  blocked_buys (aggregate): {int(res_df2['blocked_buys'].sum()) if 'blocked_buys' in res_df2 else 0}")

    if args.csv:
        out_df.to_csv(args.csv, index=False)
        print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()
