#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live/replay.py — live 트레이더 리플레이 하네스 (CLI).

두 가지 소스로 라이브 증분 트레이더(live/trader.py)를 돌리고, 참조 백테스트
(backtest/backtest_portfolio.py::run_portfolio)와 trade-by-trade 로 비교한다.

- bars  : data/full_pred.parquet 자체를 이벤트 유니버스/시그널로 사용.
          (골든 재현 1차 경로. 전체 구간 실행 시 data/portfolio_trades_golden_row7.csv
           와 100% 일치해야 한다.)
- ticks : MongoDB 일자별 틱 → live/bars.BarStream(10분봉) → full_pred 시그널 join.
          (실전 경로 모사. 기본 20 거래일 창 권장. MongoDB 불가 시 SKIP, exit 0.)

실행 예 (프로젝트 루트, UTF-8):
    # 골든 전체창 (1,020 거래 + 골든 CSV 완전일치)
    python -X utf8 live/replay.py --source bars --start-date 2025-01-06 --end-date 2025-07-23

    # 부분 구간 (참조 대조만)
    python -X utf8 live/replay.py --source bars --start-date 2025-06-25 --end-date 2025-07-23

    # 실전(틱) 경로 — 20 거래일
    python -X utf8 live/replay.py --source ticks --start-date 2025-06-25 --end-date 2025-07-23

    # warmup 1일 포함(prev_day_close 확보) 후 2025-06-25 부터 거래
    python -X utf8 live/replay.py --source ticks --start-date 2025-06-24 --trade-start 2025-06-25 --end-date 2025-07-23
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

# 프로젝트 루트를 import 경로에 추가
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows cp949 콘솔 대비
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from live.strategy import (BarsSignalSource, FrameSignalSource,  # noqa: E402
                           aggregate_ticks, join_bars_signals, load_signal_frame)
from live.trader import (IncrementalTrader, TraderConfig,  # noqa: E402
                         compare_trades, trades_to_frame)

DEFAULT_PRED = ROOT / "data" / "full_pred.parquet"
DEFAULT_GOLDEN = ROOT / "data" / "portfolio_trades_golden_row7.csv"

_REF_MOD = None


def load_reference():
    """backtest_portfolio 모듈을 동적 로드(참조 엔진). 최초 1회 캐시."""
    global _REF_MOD
    if _REF_MOD is None:
        spec = importlib.util.spec_from_file_location(
            "backtest_portfolio", ROOT / "backtest" / "backtest_portfolio.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _REF_MOD = mod
    return _REF_MOD


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description="Live incremental-portfolio replay (bars-mode golden + ticks-mode).")
    ap.add_argument("--source", choices=["bars", "ticks"], default="bars")
    ap.add_argument("--start-date", default=None,
                    help="포함 시작일 YYYY-MM-DD (기본: bars=full_pred 최소일, ticks=2025-06-25)")
    ap.add_argument("--end-date", default="2025-07-23", help="포함 종료일 YYYY-MM-DD")
    ap.add_argument("--trade-start", default=None,
                    help="(live 전용) 이 날짜 이전은 warmup(가격상태만) → 이 날짜부터 거래")
    ap.add_argument("--pred-path", default=str(DEFAULT_PRED))
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--out-trades", default=None, help="산출 트레이드 CSV 저장 경로")
    ap.add_argument("--compare-golden", action="store_true",
                    help="창과 무관하게 골든 CSV 와 강제 비교")
    ap.add_argument("--no-compare", action="store_true", help="골든 CSV 비교 생략")
    # 전략 파라미터(참조 run_portfolio 와 동일 기본값)
    ap.add_argument("--initial-cash", type=float, default=10_000_000.0)
    ap.add_argument("--cost-per-side", type=float, default=0.00015)
    ap.add_argument("--max-positions", type=int, default=8)
    ap.add_argument("--gate", type=float, default=0.42)
    ap.add_argument("--sizing", default="CW")
    ap.add_argument("--exit", default="SELLSIG")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# 참조 엔진 실행 (동일 event frame)
# --------------------------------------------------------------------------- #
def run_reference(frame: pd.DataFrame, cfg: TraderConfig) -> dict:
    ref = load_reference()
    P = ref.prepare(frame)
    return ref.run_portfolio(P, cfg.gate, cfg.sizing, cfg.exit_mode,
                             cfg.initial_cash, cfg.cost_per_side, cfg.max_positions,
                             collect_trades=True)


# --------------------------------------------------------------------------- #
# 출력 유틸
# --------------------------------------------------------------------------- #
def _fmt_daily(daily) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in daily) + "]"


def print_verdict(label: str, window: str, res: dict, cmp_info=None,
                  ref_info=None) -> None:
    trades = res["trades"]
    n_eod = sum(1 for t in trades if t["exit_reason"] == "EOD")
    n_sig = sum(1 for t in trades if t["exit_reason"] == "SELLSIG")
    print(f"\n=== replay verdict ({label}) ===")
    print(f"window        : {window}")
    print(f"ret%          : {res['total_return_pct']:+.2f}")
    print(f"MDD(bar)%     : {res['max_drawdown_pct']:.2f}")
    print(f"MDD(day)%     : {res['max_drawdown_day_pct']:.2f}")
    print(f"n_trades      : {res['n_trades']}  (EOD {n_eod} + SELLSIG {n_sig})")
    print(f"win%          : {res['win_rate_pct']:.1f}")
    print(f"blocked_buys  : {res['blocked_buys']}")
    print(f"avg_pos       : {res['avg_concurrent_positions']:.2f}")
    print(f"exposed%      : {res['pct_bars_exposed']:.1f}")
    print(f"equity_final  : {res['equity_final']:,.2f}")
    dr = res["daily_return_pct"]
    print(f"daily_returns : n={len(dr)} sum={sum(dr):+.2f}%")
    print(f"                {_fmt_daily(dr)}")
    if cmp_info is not None:
        status = "PASS" if cmp_info["ok"] else "FAIL"
        print(f"trade-compare : {status} (golden) {cmp_info['message']}")
    if ref_info is not None:
        status = "PASS" if ref_info["ok"] else "FAIL"
        print(f"ref-compare   : {status} (vs run_portfolio on same frame) {ref_info['message']}")


# --------------------------------------------------------------------------- #
# bars-mode
# --------------------------------------------------------------------------- #
def run_bars(args, cfg: TraderConfig):
    src = BarsSignalSource(args.pred_path, args.start_date, args.end_date)
    print(f"[bars] event frame rows={len(src)}  "
          f"ts={len(src.ts_index())}  window={args.start_date}..{args.end_date}")
    trader = IncrementalTrader(cfg)
    res = trader.run(src.events())

    window = f"{args.start_date}..{args.end_date}"

    # 참조 대조(동일 frame)
    ref_res = run_reference(src.df, cfg)
    ref_info = compare_trades(res["trades"], trades_to_frame(ref_res["trades"]))

    # 골든 비교(전체창이거나 명시 요청 시)
    full = _is_full_window(args.pred_path, args.start_date, args.end_date)
    cmp_info = None
    if not args.no_compare and (full or args.compare_golden):
        golden = pd.read_csv(args.golden)
        cmp_info = compare_trades(res["trades"], golden)
    else:
        print("[bars] golden 비교 생략(부분 구간). --compare-golden 로 강제 가능.")

    print_verdict("bars", window, res, cmp_info, ref_info)
    return res, cmp_info, ref_info


def _is_full_window(pred_path, start_date, end_date) -> bool:
    df = pd.read_parquet(pred_path, columns=["date"])
    lo, hi = str(df["date"].min()), str(df["date"].max())
    ok_lo = (start_date is None) or (str(start_date) <= lo)
    ok_hi = (end_date is None) or (str(end_date) >= hi)
    return ok_lo and ok_hi


# --------------------------------------------------------------------------- #
# ticks-mode
# --------------------------------------------------------------------------- #
def run_ticks(args, cfg: TraderConfig):
    try:
        from live.feed import MongoFeed
    except Exception as exc:  # pragma: no cover
        print(f"SKIP: live.feed import 실패 ({type(exc).__name__}: {exc})")
        return None, None, None

    try:
        feed = MongoFeed()
        feed.ping()
        all_days = feed.list_days()
    except Exception as exc:
        print(f"SKIP: MongoDB 연결 실패 ({type(exc).__name__}: {exc}) — ticks-mode 건너뜀")
        return None, None, None

    start = args.start_date or "2025-06-25"
    end = args.end_date or "2025-07-23"
    try:
        days = [d for d in all_days if start <= d <= end]
        if not days:
            print(f"SKIP: MongoDB 에 {start}..{end} 컬렉션이 없습니다.")
            return None, None, None

        print(f"[ticks] aggregating {len(days)} days from MongoDB: {days[0]}..{days[-1]}")
        bars_rows = aggregate_ticks(feed, days)
        print(f"[ticks] closed bars={len(bars_rows):,}")
    finally:
        feed.close()

    sig = load_signal_frame(args.pred_path, start, end)
    joined = join_bars_signals(bars_rows, sig)
    print(f"[ticks] joined (bars ∩ signals) rows={len(joined):,}")
    if joined.empty:
        print("SKIP: join 결과가 비어 있습니다.")
        return None, None, None

    src = FrameSignalSource(joined)
    trader = IncrementalTrader(cfg)
    res = trader.run(src.events())

    window = f"{start}..{end}" + (f" (trade_start={args.trade_start})" if args.trade_start else "")

    # 참조 대조: tick 기반 동일 frame 에서 run_portfolio 와 비교
    ref_info = None
    if args.trade_start is None:
        ref_frame = joined[["date", "bar_time", "code", "open", "close",
                            "pred", "pred_conf"]].copy()
        ref_res = run_reference(ref_frame, cfg)
        ref_info = compare_trades(res["trades"], trades_to_frame(ref_res["trades"]))
    else:
        print("[ticks] trade_start 사용 중 → 참조 대조 생략(warmup 의미론은 참조 미지원)")

    # 골든 비교는 전체창일 때만
    cmp_info = None
    full = _is_full_window(args.pred_path, start, end)
    if (not args.no_compare) and full:
        golden = pd.read_csv(args.golden)
        cmp_info = compare_trades(res["trades"], golden)

    print_verdict("ticks", window, res, cmp_info, ref_info)
    return res, cmp_info, ref_info


# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    cfg = TraderConfig(
        initial_cash=args.initial_cash,
        cost_per_side=args.cost_per_side,
        max_positions=args.max_positions,
        gate=args.gate,
        sizing=args.sizing,
        exit_mode=args.exit,
        trade_start=args.trade_start,
    )

    if args.source == "bars":
        if args.start_date is None:
            args.start_date = str(pd.read_parquet(args.pred_path,
                                                  columns=["date"])["date"].min())
        res, cmp_info, ref_info = run_bars(args, cfg)
    else:
        if args.start_date is None:
            args.start_date = "2025-06-25"
        res, cmp_info, ref_info = run_ticks(args, cfg)
        if res is None:
            return 0  # SKIP 은 실패 아님

    if args.out_trades and res is not None:
        out = Path(args.out_trades)
        out.parent.mkdir(parents=True, exist_ok=True)
        trades_to_frame(res["trades"]).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[out] trades CSV: {out}  ({len(res['trades'])} rows)")

    # 종료코드: 골든/참조 비교가 수행됐고 실패면 1
    failed = False
    if cmp_info is not None and not cmp_info["ok"]:
        failed = True
    if ref_info is not None and not ref_info["ok"]:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
