"""live/state.py — 실시간(증분) model state 빌더.

배치 파이프라인의 피처 계산(pipeline/build_features.py)과 state 정리
(pipeline/make_jsonl.py) 규칙을 그대로 재현하되, "닫힌 10분봉이 도착할 때마다
그 시점의 state dict"를 스냅샷 쿼리(state_for)로 계산하는 증분 구현이다.

핵심 규칙
---------
- lookahead 금지: state_for(code, T)는 T 이하의 "이미 닫힌 봉"만 사용한다.
- 반환 dict 키 25개 = code + OHLCV/n_ticks(6) + 피처 18개:
    per-code : ret1, rsi14, macd, macd_signal, macd_hist, sma5, sma20, sma60,
               vol_ratio, range_pct, bar_ret, dist_sma20, consec_up, day_progress
    market   : cum_val, rank_val, rank_vol, day_tot_val_share
  (키 순서 = make_jsonl.select_state_columns(labeled) 와 동일)
- state 정리(make_jsonl): 실수는 소수점 6자리 round, str 은 그대로, bool→int,
  비유한(NaN/inf/None) 값이 하나라도 섞이면 그 스냅샷은 "미완성"으로 보고
  None 을 반환한다(warmup 구간 등).

HistoryStore
------------
live/bars.py 의 HistoryStore 를 사용한다(동료 작성):
    bars_for(code) -> list[row]  (row = COLUMNS 순서 튜플, bar_time 오름차순)
    codes() / all_codes()        -> 보관 중인 종목코드
    add_closed_bar(...) / add_row(row)
bars.py 가 아직 없는 환경에서도 state.py 가 독립 동작하도록 최소 fallback 저장소를
내장한다(아래 _FallbackHistoryStore).

실행/사용:
    from live.bars import HistoryStore
    from live.state import LiveState
    store = HistoryStore()
    # ... 닫힌 봉을 add_closed_bar/add_row 로 적재 ...
    builder = LiveState(store)
    state = builder.state_for("005930", some_bar_time)   # dict(25키) 또는 None
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (project-root config)

# --- HistoryStore: live/bars.py 를 우선 사용, 없으면 내부 fallback ---
try:  # 패키지로 import (권장)
    from live.bars import HistoryStore  # noqa: F401
except ImportError:  # pragma: no cover - bars.py 미완성 시
    try:  # live/ 를 sys.path 에 넣고 단독 실행하는 경우
        from bars import HistoryStore  # type: ignore  # noqa: F401
    except ImportError:
        HistoryStore = None  # 아래에서 fallback 정의


# extract_bars.py / live/bars.py 와 동일한 봉 컬럼 순서 (row 튜플 레이아웃)
COLUMNS = ["date", "bar_time", "code", "name",
           "open", "high", "low", "close", "volume", "n_ticks"]
_COL = {name: i for i, name in enumerate(COLUMNS)}

# make_jsonl 과 동일한 정리 상수
FLOAT_DECIMALS = 6

# 반환 state 키 (25개) — make_jsonl.select_state_columns(labeled) 와 동일 순서
STATE_KEYS = [
    "code",
    "open", "high", "low", "close", "volume", "n_ticks",
    "ret1", "rsi14", "macd", "macd_signal", "macd_hist",
    "sma5", "sma20", "sma60", "vol_ratio", "range_pct",
    "bar_ret", "dist_sma20", "consec_up", "day_progress",
    "cum_val", "rank_val", "rank_vol", "day_tot_val_share",
]

# 피처 상수 (build_features.py 와 동일)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOL_MA_PERIOD = 20


class _NonFiniteState(Exception):
    """state 값에 유한하지 않은 수치가 섞였을 때(warmup 등)."""


class _FallbackHistoryStore:
    """live/bars.py 미존재 시 최소 인터페이스(add_closed_bar/add_row/bars_for/codes).

    row = (date, bar_time, code, name, open, high, low, close, volume, n_ticks)
    """

    def __init__(self) -> None:
        self._by_code: dict[str, list] = {}

    def add_row(self, row) -> tuple:
        row = tuple(row)
        code = str(row[_COL["code"]])
        self._by_code.setdefault(code, []).append(row)
        return row

    def add_closed_bar(self, date, bar_time, code, name, open, high, low,
                       close, volume, n_ticks) -> tuple:
        return self.add_row((date, bar_time, code, name, open, high, low,
                             close, volume, n_ticks))

    def bars_for(self, code) -> list:
        return sorted(self._by_code.get(str(code), ()),
                      key=lambda r: r[_COL["bar_time"]])

    def codes(self) -> list:
        return list(self._by_code.keys())

    all_codes = codes

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_code.values())


if HistoryStore is None:  # pragma: no cover
    HistoryStore = _FallbackHistoryStore


def _to_timedelta_hhmm(s: str) -> pd.Timedelta:
    hh, mm = s.split(":")
    return pd.Timedelta(hours=int(hh), minutes=int(mm))


def _all_codes(store) -> list:
    """HistoryStore 에서 종목코드 목록을 얻는다 (all_codes/codes 둘 다 지원)."""
    if hasattr(store, "all_codes"):
        return list(store.all_codes())
    if hasattr(store, "codes"):
        return list(store.codes())
    raise AttributeError("HistoryStore 에 all_codes()/codes() 가 없습니다")


def _rows_to_df(rows) -> pd.DataFrame:
    """row 튜플 리스트 → 컬럼 DataFrame (bar_time 은 datetime64)."""
    data = {name: [r[idx] for r in rows] for name, idx in _COL.items()}
    df = pd.DataFrame(data)
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df


def _clean_value(v):
    """make_jsonl._clean_value 와 동일: str 유지, int/bool→int, 실수 6자리 round."""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    f = float(v)
    if not np.isfinite(f):
        raise _NonFiniteState(repr(v))
    return round(f, FLOAT_DECIMALS)


def _stock_features(g: pd.DataFrame) -> pd.DataFrame:
    """한 종목의 (오름차순) 봉 시계열에 per-code 피처를 계산.

    pipeline/build_features._add_stock_features 와 규칙이 100% 동일하다.
    (전 시계열에 대해 계산하되, 값 자체는 인과적이므로 각 행은 과거만 반영)
    """
    g = g.sort_values("bar_time", kind="mergesort")

    close = g["close"].astype("float64")
    open_ = g["open"].astype("float64")
    high = g["high"].astype("float64")
    low = g["low"].astype("float64")
    volume = g["volume"].astype("float64")

    out = g.copy()

    # --- 수익률 ---
    out["ret1"] = close.pct_change()

    # --- RSI14: 단순 14봉 평균 (Wilder 아님) ---
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    out["rsi14"] = rsi

    # --- MACD(12,26,9), adjust=False ---
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # --- 이동평균 ---
    out["sma5"] = close.rolling(5, min_periods=5).mean()
    out["sma20"] = close.rolling(20, min_periods=20).mean()
    out["sma60"] = close.rolling(60, min_periods=60).mean()

    # --- 거래량/변동성/막봉 ---
    out["vol_ratio"] = volume / volume.rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean()
    out["range_pct"] = (high - low) / close
    out["bar_ret"] = close / open_ - 1.0
    out["dist_sma20"] = close / out["sma20"] - 1.0

    # --- 연속 상승봉 ---
    up = (out["ret1"] > 0.0).fillna(False)
    block = (~up).cumsum()
    out["consec_up"] = up.groupby(block).cumsum().astype("int64")

    # --- 장 진행률 0..1 ---
    start = _to_timedelta_hhmm(config.TRADING_START)
    end = _to_timedelta_hhmm(config.TRADING_END)
    tod = out["bar_time"] - out["bar_time"].dt.normalize()
    out["day_progress"] = ((tod - start) / (end - start)).astype("float64").clip(0.0, 1.0)

    return out


def _market_context(date_df: pd.DataFrame, code: str, T: pd.Timestamp) -> dict | None:
    """(date, bar_time<=T) 스냅샷의 모집단 맥락 피처 (cross-section).

    build_features._add_market_context 와 동일 규칙:
      cum_val          : (date, code) cumsum(close*volume) — T까지 그 종목 누적 거래금액
      rank_val/rank_vol: (date, bar_time=T) 스냅샷 내 rank(1=최대, method='first')
                         — tie-break 는 code 오름차순(정렬 [bar_time, code] 기준)
      day_tot_val_share: cum_val / (T까지 당일 전체 누적 거래금액)
    """
    df = date_df[date_df["bar_time"] <= T]
    if df.empty:
        return None
    df = df.sort_values(["bar_time", "code"], kind="mergesort").reset_index(drop=True)

    volume = df["volume"].astype("float64")
    close = df["close"].astype("float64")
    bar_val = close * volume

    cum_val = bar_val.groupby(df["code"], sort=False).cumsum()
    cum_vol = volume.groupby(df["code"], sort=False).cumsum()

    # 당일 전체 누적 거래금액: bar_time 스냅샷 합을 date(단일) 내 누적
    ts_sum = bar_val.groupby(df["bar_time"], sort=False).transform("sum")
    first_of_ts = ~df.duplicated(subset=["bar_time"], keep="first")
    day_cum = ts_sum[first_of_ts].cumsum()
    denom = day_cum.reindex(df.index).ffill()

    df = df.assign(cum_val=cum_val.values, cum_vol=cum_vol.values, denom=denom.values)

    snap = df[df["bar_time"] == T]
    if snap.empty:
        return None
    snap = snap.reset_index(drop=True)
    rank_val = snap["cum_val"].rank(method="first", ascending=False)
    rank_vol = snap["cum_vol"].rank(method="first", ascending=False)

    mask = (snap["code"] == code).to_numpy()
    if not mask.any():
        return None
    i = int(np.flatnonzero(mask)[0])
    my_cum_val = float(snap["cum_val"].iloc[i])
    my_denom = float(snap["denom"].iloc[i])
    share = my_cum_val / my_denom if my_denom > 0.0 else float("nan")
    return {
        "cum_val": my_cum_val,
        "rank_val": float(rank_val.iloc[i]),
        "rank_vol": float(rank_vol.iloc[i]),
        "day_tot_val_share": share,
    }


class LiveState:
    """HistoryStore 위에서 동작하는 증분(스냅샷) state 빌더.

    state_for(code, bar_time) 는 그 시점에 이미 닫힌 봉만 사용해 25키 state 를
    계산한다. 미완성(warmup/비유한) 이면 None 을 반환한다.
    """

    def __init__(self, history) -> None:
        self.history = history
        self._date_cache: dict[str, pd.DataFrame] = {}
        self._cache_size: int = -1

    # --- 공개 API ---
    def state_for(self, code, bar_time) -> dict | None:
        """(code, bar_time) 스냅샷의 model state dict(25키) 또는 None(미완성)."""
        code = str(code)
        T = pd.Timestamp(bar_time)

        # 1) per-code 피처: 해당 종목의 T 이하 닫힌 봉만 사용
        rows = [r for r in self.history.bars_for(code)
                if pd.Timestamp(r[_COL["bar_time"]]) <= T]
        if not rows:
            return None
        g = _rows_to_df(rows)
        if g["bar_time"].max() != T:
            return None  # T 봉이 아직 닫히지 않음

        feat = _stock_features(g)
        last = feat.iloc[-1]

        # 2) 모집단(cross-section) 맥락: 같은 date, T 이하 봉 전체
        date_str = T.strftime("%Y-%m-%d")
        date_df = self._bars_on_date(date_str)
        if date_df is None or date_df.empty:
            return None
        mkt = _market_context(date_df, code, T)
        if mkt is None:
            return None

        # 3) state 조립 (키 순서 = STATE_KEYS)
        state = {
            "code": code,
            "open": last["open"], "high": last["high"], "low": last["low"],
            "close": last["close"], "volume": last["volume"], "n_ticks": last["n_ticks"],
            "ret1": last["ret1"], "rsi14": last["rsi14"],
            "macd": last["macd"], "macd_signal": last["macd_signal"],
            "macd_hist": last["macd_hist"],
            "sma5": last["sma5"], "sma20": last["sma20"], "sma60": last["sma60"],
            "vol_ratio": last["vol_ratio"], "range_pct": last["range_pct"],
            "bar_ret": last["bar_ret"], "dist_sma20": last["dist_sma20"],
            "consec_up": last["consec_up"], "day_progress": last["day_progress"],
            "cum_val": mkt["cum_val"], "rank_val": mkt["rank_val"],
            "rank_vol": mkt["rank_vol"], "day_tot_val_share": mkt["day_tot_val_share"],
        }

        # 4) 정리: 비유한 값이 하나라도 있으면 미완성(None)
        try:
            return {k: _clean_value(state[k]) for k in STATE_KEYS}
        except _NonFiniteState:
            return None

    def snapshot(self, bar_time) -> dict[str, dict]:
        """한 bar_time 에서 state 가 완성되는 모든 종목의 {code: state} 를 반환."""
        T = pd.Timestamp(bar_time)
        date_str = T.strftime("%Y-%m-%d")
        date_df = self._bars_on_date(date_str)
        out: dict[str, dict] = {}
        if date_df is None or date_df.empty:
            return out
        codes = date_df.loc[date_df["bar_time"] == T, "code"].unique().tolist()
        for c in codes:
            s = self.state_for(c, T)
            if s is not None:
                out[str(c)] = s
        return out

    # --- 내부 ---
    def _bars_on_date(self, date_str: str) -> pd.DataFrame | None:
        self._refresh_cache()
        if date_str not in self._date_cache:
            rows = []
            for c in _all_codes(self.history):
                for r in self.history.bars_for(c):
                    if str(r[_COL["date"]]) == date_str:
                        rows.append(r)
            self._date_cache[date_str] = _rows_to_df(rows) if rows else pd.DataFrame()
        return self._date_cache[date_str]

    def _refresh_cache(self) -> None:
        try:
            size = len(self.history)
        except TypeError:  # __len__ 미지원 저장소 방어
            size = sum(len(self.history.bars_for(c)) for c in _all_codes(self.history))
        if size != self._cache_size:
            self._cache_size = size
            self._date_cache = {}
