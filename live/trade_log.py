#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live/trade_log.py — 매매일지(체결 내역) 출력기.

게이트 0.42 / CW / SELLSIG 스마트-롱 전략의 매수·매도 체결을 사람이 읽기 좋게
표로 보여준다. 기본 소스는 골든 체결 CSV(data/portfolio_trades_golden_row7.csv,
1,020건)로, 백테스트와 100% 동일한 실제 체결 기록이다.

필터:
  - 특정 일자만:   위치인자  2025-01-10   또는  --date 2025-01-10
  - 기간:          --start 2025-01-06 --end 2025-01-10
  - 종목:          --code 089790
  - 사유:          --reason EOD | SELLSIG
  - 소스 재실행:   --source bars  (full_pred 로 트레이더를 다시 돌려 뽑음)

실행 예 (프로젝트 루트, UTF-8):
    python -X utf8 live/trade_log.py                      # 전체 + 일자별 요약
    python -X utf8 live/trade_log.py 2025-01-10           # 특정일만
    python -X utf8 live/trade_log.py --start 2025-01-06 --end 2025-01-10
    python -X utf8 live/trade_log.py --date 2025-01-10 --code 089790
    python -X utf8 live/trade_log.py --reason SELLSIG --limit 30
    python -X utf8 live/trade_log.py --source bars --start 2025-06-25 --end 2025-07-23
    python -X utf8 live/trade_log.py --out 매매일지.csv   # 필터 결과 저장
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows cp949 콘솔 대비
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_GOLDEN = ROOT / "data" / "portfolio_trades_golden_row7.csv"
DEFAULT_PRED = ROOT / "data" / "full_pred.parquet"

# (컬럼, 표시헤더, 너비, 오른쪽정렬)
COLS = [
    ("entry_date", "날짜", 10, False),
    ("code", "종목", 7, False),
    ("shares", "수량", 8, True),
    ("entry_open", "매수가", 9, True),
    ("exit_open", "매도가", 9, True),
    ("exit_reason", "사유", 8, False),
    ("pnl", "손익", 12, True),
]


# --------------------------------------------------------------------------- #
# 표 정렬 유틸 (한글은 2칸 폭으로 계산)
# --------------------------------------------------------------------------- #
def _dw(s: str) -> int:
    return sum(2 if ord(c) > 0x2E80 else 1 for c in str(s))


def _pad(s, width: int, right: bool = False) -> str:
    s = str(s)
    gap = " " * max(0, width - _dw(s))
    return (gap + s) if right else (s + gap)


def _won(v) -> str:
    return f"{v:+,.0f}"


# --------------------------------------------------------------------------- #
# 로딩 / 필터
# --------------------------------------------------------------------------- #
def load_trades(args) -> tuple[pd.DataFrame, str]:
    """(df, 출처 라벨) 반환."""
    if args.source == "bars":
        from live.strategy import BarsSignalSource
        from live.trader import IncrementalTrader, TraderConfig, trades_to_frame

        start = args.start or args.date
        end = args.end or args.date or "2025-07-23"
        src = BarsSignalSource(str(args.pred_path), start, end)
        trader = IncrementalTrader(TraderConfig(
            gate=args.gate, sizing=args.sizing, exit_mode=args.exit))
        res = trader.run(src.events())
        label = f"bars 재실행 ({start}..{end})"
        return trades_to_frame(res["trades"]), label

    df = pd.read_csv(args.golden, dtype={"code": str})
    return df, f"골든 CSV ({Path(args.golden).name})"


def filter_trades(df: pd.DataFrame, args) -> pd.DataFrame:
    if "entry_date" not in df.columns:
        return df
    if args.date:
        d = str(args.date)
        df = df[(df["entry_date"].astype(str) == d) | (df["exit_date"].astype(str) == d)]
    if args.start:
        df = df[df["exit_date"].astype(str) >= str(args.start)]
    if args.end:
        df = df[df["exit_date"].astype(str) <= str(args.end)]
    if args.code:
        df = df[df["code"].astype(str).str.zfill(6) == str(args.code).zfill(6)]
    if args.reason:
        df = df[df["exit_reason"] == args.reason]
    return df


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #
def print_table(df: pd.DataFrame, limit: int, tail: bool) -> None:
    line_w = sum(c[2] for c in COLS) + 2 * len(COLS)
    print("-" * line_w)
    print("  ".join(_pad(h, w, r) for _, h, w, r in COLS))
    print("-" * line_w)
    rows = df
    if limit and len(df) > limit:
        rows = df.tail(limit) if tail else df.head(limit)
    for _, r in rows.iterrows():
        vals = [
            _pad(str(r["entry_date"]), 10),
            _pad(str(r["code"]).zfill(6), 7),
            _pad(f'{int(r["shares"]):,}', 8, True),
            _pad(f'{float(r["entry_open"]):,.0f}', 9, True),
            _pad(f'{float(r["exit_open"]):,.0f}', 9, True),
            _pad(str(r["exit_reason"]), 8),
            _pad(_won(float(r["pnl"])), 12, True),
        ]
        print("  ".join(vals))
    print("-" * line_w)
    if limit and len(df) > limit:
        which = "마지막" if tail else "처음"
        print(f"... ({len(df):,}건 중 {which} {len(rows):,}건만 표시 — 전부 보려면 --limit 0)")


def summary_line(df: pd.DataFrame, title: str) -> None:
    n = len(df)
    if n == 0:
        print(f"{title}: 거래 없음")
        return
    win = int((df["pnl"] > 0).sum())
    lose = n - win
    tot = float(df["pnl"].sum())
    n_eod = int((df["exit_reason"] == "EOD").sum())
    n_sig = int((df["exit_reason"] == "SELLSIG").sum())
    print(f"{title}: {n:,}건  (EOD {n_eod} + SELLSIG {n_sig})  "
          f"승 {win:,} / 패 {lose:,}  승률 {win / n * 100:.1f}%  "
          f"실현손익 {tot:+,.0f}  평균 {tot / n:+,.0f}")


def print_daily(df: pd.DataFrame) -> None:
    print("\n--- 일자별 요약 ---")
    hdr = (_pad("날짜", 10) + "  " + _pad("거래", 6, True) + "  "
           + _pad("승", 5, True) + "  " + _pad("패", 5, True) + "  "
           + _pad("승률", 8, True) + "  " + _pad("실현손익", 14, True))
    print(hdr)
    print("-" * (_dw(hdr)))
    for d, sub in df.groupby("exit_date"):
        n = len(sub)
        w = int((sub["pnl"] > 0).sum())
        line = (_pad(str(d), 10) + "  " + _pad(f"{n}", 6, True) + "  "
                + _pad(f"{w}", 5, True) + "  " + _pad(f"{n - w}", 5, True) + "  "
                + _pad(f"{w / n * 100:.1f}%", 8, True) + "  "
                + _pad(f'{float(sub["pnl"].sum()):+,.0f}', 14, True))
        print(line)


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description="매매일지 출력 (스마트-롱 gate 0.42 / CW / SELLSIG)")
    ap.add_argument("date", nargs="?", default=None,
                    help="특정 일자 YYYY-MM-DD (예: 2025-01-10)")
    ap.add_argument("--date", dest="date_opt", default=None,
                    help="특정 일자 YYYY-MM-DD (위치인자와 동일)")
    ap.add_argument("--start", default=None, help="시작일(포함) YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="종료일(포함) YYYY-MM-DD")
    ap.add_argument("--code", default=None, help="종목코드 6자리 (예: 089790)")
    ap.add_argument("--reason", choices=["EOD", "SELLSIG"], default=None)
    ap.add_argument("--limit", type=int, default=40, help="표시 최대 행수 (0=무제한)")
    ap.add_argument("--tail", action="store_true", help="뒤에서 limit개")
    ap.add_argument("--no-daily", action="store_true", help="일자별 요약 생략")
    ap.add_argument("--no-prompt", action="store_true", help="대화형 날짜 입력 안 함")
    ap.add_argument("--out", default=None, help="필터 결과 CSV 저장 경로")
    ap.add_argument("--source", choices=["golden", "bars"], default="golden",
                    help="golden=골든 CSV(기본), bars=트레이더 재실행")
    ap.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    ap.add_argument("--pred-path", default=str(DEFAULT_PRED))
    ap.add_argument("--gate", type=float, default=0.42)
    ap.add_argument("--sizing", default="CW")
    ap.add_argument("--exit", default="SELLSIG")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.date = args.date or args.date_opt

    # 대화형(더블클릭 등): 아무 필터도 없고 콘솔이면 날짜만 물어봄
    if (args.date is None and args.start is None and args.end is None
            and args.code is None and args.reason is None
            and not args.no_prompt and sys.stdin.isatty()):
        try:
            v = input("날짜 입력 (예: 2025-01-10, 전체는 그냥 엔터): ").strip()
        except EOFError:
            v = ""
        if v:
            args.date = v

    df_all, label = load_trades(args)
    df = filter_trades(df_all, args)

    sort_cols = [c for c in ["entry_date", "entry_ts", "code"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    if args.date:
        title = f"=== 매매일지 {args.date} ==="
    elif args.start or args.end:
        title = f"=== 매매일지 {args.start or df_all['exit_date'].min()}..{args.end or df_all['exit_date'].max()} ==="
    else:
        title = "=== 매매일지 (전체) ==="

    print(title)
    print(f"소스: {label}  |  전체 {len(df_all):,}건 중 조건 {len(df):,}건")
    print()

    if df.empty:
        print("조건에 맞는 거래가 없습니다.")
        return 0

    print_table(df, args.limit, args.tail)
    print()
    summary_line(df, "합계")
    if df["exit_date"].nunique() > 1 and not args.no_daily:
        print_daily(df)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n[저장] {out}  ({len(df)}건)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
