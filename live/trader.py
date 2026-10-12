#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live/trader.py — 증분(incremental) 포트폴리오 트레이더.

backtest/backtest_portfolio.py::run_portfolio(gate, sizing, exit_mode) 의 의미를
틱-단위(봉-단위) 증분 루프로 100% 재현한다. 이 모듈은 live/strategy.py 의
BarEvent 스트림만 소비하며, 참조 엔진과 동일한 순서/연산으로 trade-by-trade,
equity-curve 까지 일치해야 한다.

참조 엔진과의 대응
------------------
참조 prepare() 는 전역 [bar_time, code] 정렬 + 타임슬롯 그룹핑을 미리 하지만,
본 트레이더는 이벤트를 흘려보내며 "한 타임슬롯 룩어헤드 버퍼"로 is_last_of_day
를 판정한다(다음 타임슬롯의 date 가 다르거나 스트림 끝이면 그 날의 마지막 봉).

타임슬롯 처리 순서(참조와 동일, 엄수):
  Phase 0  코드별 date 전환 감지 → prev_day_close(=직전 등장일의 마지막 close) 확정
  Phase 1  대기주문 실행(코드 오름차순):
             buy  : positions 존재/o<=0 → skip; 상한가 블록; sizing; shares=int(invest//o)
             sell : gross=shares*o; proceeds=gross-gross*cost; pnl; win
  Phase 2  last_close[code]=close (존재 코드 전부)
  Phase 3  pos_counts.append(len(positions))   (EOD flatten 이전!)
  Phase 4  is_last_of_day → 보유 전량 EOD 청산(positions 삽입순, last_close 기준)
  Phase 5  equity = cash + Σ shares*last_close  → equity_curve; EOD면 daily_equity
  Phase 6  결정 생성(이 봉 → 다음 봉): buy/sell pending
  Phase 7  bars_held += 1

주의
----
- 시그널 소스에 없는 봉은 이벤트 유니버스에 없다(live/strategy.py 규칙). 따라서
  EOD 판정/equity/last_close 모두 이벤트 유니버스 기준으로만 이루어진다.
- 대기주문(pending)은 "그 코드의 다음 봉"에서 실행된다(전역 다음 타임슬롯이 아님).
- 아무 시그널 소스/설정 없이도 단독 동작한다(외부 의존성: numpy/pandas 만).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from live.strategy import BarEvent

# Windows cp949 콘솔 대비
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 골든 트레이드 로그의 16개 컬럼(순서 고정)
TRADE_COLUMNS = [
    "code", "exit_reason", "entry_ts", "exit_ts", "entry_date", "exit_date",
    "shares", "entry_open", "exit_open", "invest", "entry_cost", "entry_cash_out",
    "exit_gross", "exit_cost", "proceeds", "pnl",
]

# 비교 시 문자열로 취급할 컬럼
_TRADE_STR_COLS = ["exit_reason", "entry_date", "exit_date"]
_TRADE_KEY_COLS = ["code", "entry_ts", "exit_ts", "entry_date", "exit_date",
                   "shares", "entry_open", "exit_open", "exit_reason"]


def max_drawdown_pct(curve) -> float:
    """peak-to-trough drawdown (%) — 참조 backtest_portfolio.max_drawdown_pct 와 동일."""
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


@dataclass
class TraderConfig:
    """참조 run_portfolio 설정 + live 전용 옵션.

    gate/sizing/exit_mode/initial_cash/cost_per_side/max_positions 는 참조와 동일 의미.
    trade_start: (live 전용) 이 날짜 이전 타임슬롯은 가격상태(prev_day_close/last_close)
                 적재만 하고 거래하지 않는다. 실전 warmup 표현용. None = 전 구간 거래.
    """

    initial_cash: float = 10_000_000.0
    cost_per_side: float = 0.00015
    max_positions: int = 8
    gate: Optional[float] = 0.42
    sizing: str = "CW"
    exit_mode: str = "SELLSIG"
    trade_start: Optional[str] = None


class IncrementalTrader:
    """BarEvent 스트림을 소비하는 증분 포트폴리오 트레이더."""

    def __init__(self, config: Optional[TraderConfig] = None) -> None:
        self.cfg = config or TraderConfig()

    # ------------------------------------------------------------------ #
    def run(self, events: Iterator[BarEvent]) -> Dict[str, Any]:
        """이벤트(정렬된 BarEvent generator)를 소비해 결과 dict 반환."""
        cfg = self.cfg
        # --- 상태 초기화 ---
        self._cash = float(cfg.initial_cash)
        self._positions: Dict[str, dict] = {}
        self._pending: Dict[str, dict] = {}
        self._last_close: Dict[str, int] = {}          # code -> 최근 close(전역)
        self._day_last_close: Dict[str, int] = {}      # code -> 현재 date 의 마지막 close
        self._prev_day_close: Dict[str, Optional[int]] = {}  # code -> 직전 등장일 마지막 close
        self._code_date: Dict[str, str] = {}           # code -> 마지막 처리 date
        self._last_equity = float(cfg.initial_cash)

        self._equity_curve: List[float] = []
        self._pos_counts: List[int] = []
        self._daily_equity: List[float] = []
        self._daily_dates: List[str] = []
        self._trades: List[dict] = []
        self._blocked_buys = 0
        self._n_trades = 0
        self._wins = 0

        # --- 한 타임슬롯 룩어헤드 버퍼 ---
        cur_bt = None
        cur_date = None
        cur_rows: List[BarEvent] = []
        for ev in events:
            if cur_bt is None:
                cur_bt, cur_date, cur_rows = ev.bar_time, ev.date, [ev]
                continue
            if ev.bar_time != cur_bt:
                self._process_timeslot(cur_bt, cur_date, cur_rows,
                                       is_last=(ev.date != cur_date))
                cur_bt, cur_date, cur_rows = ev.bar_time, ev.date, [ev]
            else:
                cur_rows.append(ev)
        if cur_bt is not None:
            self._process_timeslot(cur_bt, cur_date, cur_rows, is_last=True)

        return self._finalize()

    # ------------------------------------------------------------------ #
    def _process_timeslot(self, bt, date: str, rows: List[BarEvent],
                          is_last: bool) -> None:
        cfg = self.cfg
        trading = (cfg.trade_start is None) or (str(date) >= str(cfg.trade_start))
        ts = rows[0].ts  # 같은 타임슬롯의 모든 행은 동일 ts 를 갖는다

        # ---- Phase 0: 코드별 date 전환 감지 → prev_day_close 확정 ----
        for ev in rows:
            code = ev.code
            if self._code_date.get(code) != date:
                self._prev_day_close[code] = self._day_last_close.get(code)
                self._code_date[code] = date

        # ---- Phase 1: 대기주문 실행 (코드 오름차순 = rows 순서) ----
        if trading:
            for ev in rows:
                code = ev.code
                o = ev.open
                order = self._pending.pop(code, None)
                if order is None:
                    continue
                if order["type"] == "buy":
                    if code in self._positions or o <= 0:
                        continue
                    pc = self._prev_day_close.get(code)
                    if pc is not None and pc > 0:
                        limit_up = round(pc * 1.30, 2)
                        if o >= limit_up - 1e-9:
                            self._blocked_buys += 1
                            continue
                    target = self._last_equity / cfg.max_positions
                    if cfg.sizing == "CW":
                        # 참조 run_portfolio 와 동일: CW 배수는 "실행 봉의" pred_conf 를
                        # 쓴다(결정 봉 conf 가 아님). order["conf"] 는 미사용(참조도 저장만).
                        target *= min(ev.pred_conf * 2.0, 1.0)
                    max_by_cash = self._cash / (1.0 + cfg.cost_per_side)
                    invest = min(target, max_by_cash)
                    if invest <= 0:
                        continue
                    shares = int(invest // o)
                    if shares < 1:
                        continue
                    cash_used = shares * o
                    entry_cost = invest * cfg.cost_per_side
                    self._cash -= (cash_used + entry_cost)
                    self._positions[code] = {
                        "shares": shares,
                        "entry_price": o,
                        "invest": invest,
                        "entry_cost": entry_cost,
                        "entry_cash_out": cash_used + entry_cost,
                        "bars_held": 1,
                        "entry_ts": ts,
                        "entry_date": str(date),
                    }
                elif order["type"] == "sell":
                    if code not in self._positions or o <= 0:
                        continue
                    pos = self._positions.pop(code)
                    gross = pos["shares"] * o
                    ecost = gross * cfg.cost_per_side
                    proceeds = gross - ecost
                    self._cash += proceeds
                    self._n_trades += 1
                    pnl = proceeds - pos["entry_cash_out"]
                    if pnl > 0:
                        self._wins += 1
                    self._trades.append(self._trade_record(
                        code, "SELLSIG" if cfg.exit_mode == "SELLSIG" else "FIXED3",
                        pos, ts, str(date), o, gross, ecost, proceeds, pnl))

        # ---- Phase 2: last_close 갱신(존재 코드 전부) ----
        for ev in rows:
            if ev.close > 0:
                self._last_close[ev.code] = ev.close
                self._day_last_close[ev.code] = ev.close

        if not trading:
            # warmup 구간: 가격상태만 적재하고 지표/EOD/결정은 하지 않는다.
            return

        # ---- Phase 3: 이 봉 중 보유 포지션 수(EOD flatten 이전) ----
        self._pos_counts.append(len(self._positions))

        # ---- Phase 4: EOD flatten (그 날 마지막 봉, positions 삽입순) ----
        if is_last:
            for code in list(self._positions.keys()):
                pos = self._positions.pop(code)
                px = self._last_close.get(code, pos["entry_price"])
                gross = pos["shares"] * px
                ecost = gross * cfg.cost_per_side
                proceeds = gross - ecost
                self._cash += proceeds
                self._n_trades += 1
                pnl = proceeds - pos["entry_cash_out"]
                if pnl > 0:
                    self._wins += 1
                self._trades.append(self._trade_record(
                    code, "EOD", pos, ts, str(date), px, gross, ecost, proceeds, pnl))

        # ---- Phase 5: equity 마킹(EOD 봉은 flatten 이후) ----
        eq = self._cash + sum(
            pos["shares"] * self._last_close.get(c, pos["entry_price"])
            for c, pos in self._positions.items()
        )
        self._equity_curve.append(eq)
        if is_last:
            self._daily_equity.append(eq)
            self._daily_dates.append(str(date))
        self._last_equity = eq

        # ---- Phase 6: 이 봉 결정 → 다음 봉 대기주문 생성 ----
        for ev in rows:
            pred_n = str(ev.pred).strip().lower()
            code = ev.code
            if code not in self._positions:
                if pred_n == "buy" and (cfg.gate is None or ev.pred_conf >= cfg.gate):
                    self._pending[code] = {"type": "buy", "conf": ev.pred_conf}
            else:
                if cfg.exit_mode == "SELLSIG":
                    if pred_n == "sell":
                        self._pending[code] = {"type": "sell"}
                else:  # FIXED3
                    if self._positions[code]["bars_held"] >= 3:
                        self._pending[code] = {"type": "sell"}

        # ---- Phase 7: 이 봉에서 거래된 보유 포지션 aging ----
        for ev in rows:
            if ev.code in self._positions:
                self._positions[ev.code]["bars_held"] += 1

    # ------------------------------------------------------------------ #
    @staticmethod
    def _trade_record(code, exit_reason, pos, exit_ts, exit_date, exit_open,
                      gross, ecost, proceeds, pnl) -> dict:
        return {
            "code": code,
            "exit_reason": exit_reason,
            "entry_ts": pos["entry_ts"],
            "exit_ts": exit_ts,
            "entry_date": pos["entry_date"],
            "exit_date": exit_date,
            "shares": pos["shares"],
            "entry_open": pos["entry_price"],
            "exit_open": exit_open,
            "invest": pos["invest"],
            "entry_cost": pos["entry_cost"],
            "entry_cash_out": pos["entry_cash_out"],
            "exit_gross": gross,
            "exit_cost": ecost,
            "proceeds": proceeds,
            "pnl": pnl,
        }

    # ------------------------------------------------------------------ #
    def _finalize(self) -> Dict[str, Any]:
        cfg = self.cfg
        equity_final = self._equity_curve[-1] if self._equity_curve else float(cfg.initial_cash)

        daily_return_pct: List[float] = []
        if self._daily_equity:
            prev = float(cfg.initial_cash)
            for v in self._daily_equity:
                daily_return_pct.append((v / prev - 1.0) * 100.0 if prev > 0 else 0.0)
                prev = v

        total_return_pct = ((equity_final / cfg.initial_cash - 1.0) * 100.0
                            if cfg.initial_cash > 0 else 0.0)
        avg_pos = float(np.mean(self._pos_counts)) if self._pos_counts else 0.0
        pct_exposed = (100.0 * float(np.mean([1.0 if x > 0 else 0.0
                                              for x in self._pos_counts]))
                       if self._pos_counts else 0.0)

        return {
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct(self._equity_curve),
            "max_drawdown_day_pct": max_drawdown_pct(self._daily_equity),
            "n_trades": self._n_trades,
            "win_rate_pct": (self._wins / self._n_trades * 100.0) if self._n_trades > 0 else 0.0,
            "blocked_buys": self._blocked_buys,
            "avg_concurrent_positions": avg_pos,
            "pct_bars_exposed": pct_exposed,
            "equity_final": equity_final,
            "daily_return_pct": daily_return_pct,
            "daily_dates": self._daily_dates,
            "equity_curve": self._equity_curve,
            "pos_counts": self._pos_counts,
            "trades": self._trades,
            "n_ts": len(self._equity_curve),
            "n_wins": self._wins,
        }


# --------------------------------------------------------------------------- #
# 트레이드 로그 -> DataFrame / 비교
# --------------------------------------------------------------------------- #
def trades_to_frame(trades: List[dict]) -> "pd.DataFrame":
    """trades(dict 리스트)를 16컬럼 DataFrame 으로 변환(타입 정규화 포함)."""
    import pandas as pd

    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    df["code"] = df["code"].astype(str)
    for c in ("entry_ts", "exit_ts", "shares", "entry_open", "exit_open", "exit_gross"):
        df[c] = df[c].astype("int64")
    for c in ("invest", "entry_cost", "entry_cash_out", "exit_cost", "proceeds", "pnl"):
        df[c] = df[c].astype("float64")
    df["exit_reason"] = df["exit_reason"].astype(str)
    df["entry_date"] = df["entry_date"].astype(str)
    df["exit_date"] = df["exit_date"].astype(str)
    return df


def _norm_code(x) -> str:
    """종목코드 정규화: '018500'(str) / 18500(int/float) 모두 '18500' 으로."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return str(x)


def _normalize_trade_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    d = df.copy()
    d["code"] = d["code"].map(_norm_code)
    for c in ("entry_ts", "exit_ts", "shares", "entry_open", "exit_open", "exit_gross"):
        d[c] = d[c].astype("int64")
    for c in ("invest", "entry_cost", "entry_cash_out", "exit_cost", "proceeds", "pnl"):
        d[c] = d[c].astype("float64")
    for c in _TRADE_STR_COLS:
        d[c] = d[c].astype(str)
    return d


def compare_trades(produced, golden, tol: float = 1e-6) -> Dict[str, Any]:
    """산출 트레이드 vs 골든 트레이드 16컬럼 완전 일치 검사.

    조인 키 = code+entry_ts+exit_ts+entry_date+exit_date+shares+entry_open+exit_open+
             exit_reason. 조인된 행에 대해 나머지 수치 컬럼을 tol 이내로, 문자열 컬럼을
             정확 일치로 비교한다. 골든/산출 양쪽 중복 키도 실패로 간주.

    반환 dict: ok, n_prod, n_golden, n_matched, n_only_prod, n_only_golden,
              n_numeric_bad, max_abs_diff, dup_prod, dup_golden, message.
    """
    import pandas as pd

    p = _normalize_trade_frame(trades_to_frame(produced) if isinstance(produced, list) else produced)
    g = _normalize_trade_frame(golden)

    dup_p = int(p.duplicated(_TRADE_KEY_COLS).sum())
    dup_g = int(g.duplicated(_TRADE_KEY_COLS).sum())

    m = p.merge(g, on=_TRADE_KEY_COLS, how="outer", suffixes=("_p", "_g"),
                indicator=True)
    n_both = int((m["_merge"] == "both").sum())
    n_left = int((m["_merge"] == "left_only").sum())
    n_right = int((m["_merge"] == "right_only").sum())

    # 조인 키(9개: code+entry_ts+exit_ts+entry_date+exit_date+shares+entry_open+
    # exit_open+exit_reason)는 조인으로 동일성이 보장된다. 나머지 컬럼(수치 7개)을
    # tol 이내로 비교한다. (키 + 비키 = 16컬럼 전체가 결과적으로 일치해야 한다.)
    numeric_nonkey = ["invest", "entry_cost", "entry_cash_out", "exit_gross",
                      "exit_cost", "proceeds", "pnl"]
    matched = m[m["_merge"] == "both"]
    max_diff = 0.0
    n_numeric_bad = 0
    if len(matched):
        for c in numeric_nonkey:
            diff = (matched[c + "_p"].astype("float64")
                    - matched[c + "_g"].astype("float64")).abs()
            if len(diff):
                max_diff = max(max_diff, float(diff.max()))
            n_numeric_bad += int((diff > tol).sum())

    ok = (n_left == 0 and n_right == 0 and n_numeric_bad == 0
          and dup_p == 0 and dup_g == 0 and len(p) == len(g))

    msg = (f"matched {n_both}/{len(g)} | only_prod={n_left} only_golden={n_right} | "
           f"numeric_bad={n_numeric_bad} max|diff|={max_diff:.3e} | "
           f"dup(prod={dup_p},golden={dup_g})")
    return {
        "ok": ok, "n_prod": len(p), "n_golden": len(g), "n_matched": n_both,
        "n_only_prod": n_left, "n_only_golden": n_right,
        "n_numeric_bad": n_numeric_bad, "max_abs_diff": max_diff,
        "dup_prod": dup_p, "dup_golden": dup_g, "message": msg,
    }
