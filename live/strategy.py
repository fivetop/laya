#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live/strategy.py — signal-source abstraction (bar feed -> decision signals).

역할
----
live 파이프라인의 "시그널 소스"를 추상화한다. 트레이더(live/trader.py)는 이 모듈이
방출하는 BarEvent 스트림만 소비하며, 각 이벤트는 (code, bar_time, open, close,
pred, pred_conf) 를 담는다. 즉 "이 봉에서 이 종목의 모델 결정은 무엇인가"를
제공하는 것이 이 모듈의 책임이다.

두 가지 소스를 제공한다 (백테스트 참조: backtest/backtest_portfolio.py::prepare):
- BarsSignalSource  : data/full_pred.parquet 를 그대로 이벤트 유니버스로 사용.
                      (bars-mode 골든 재현용. prepare()와 동일한 dropna/정렬 규칙)
- FrameSignalSource : 이미 만들어진 event frame(예: ticks-mode에서 MongoFeed →
                      live/bars.BarStream 로 만든 봉 ∩ full_pred 시그널)을 소비.

핵심 규칙 (참조 엔진과 100% 일치해야 함)
---------------------------------------
* 이벤트 유니버스 = 시그널 소스에 존재하는 (code, bar_time) 행만. 시그널이 없는 봉은
  last_close/EOD/equity 에도 전혀 반영되지 않는다(prepare()가 pred/pred_conf 결측
  행을 drop 하는 것과 동일).
* 정렬은 [bar_time, code] (bar_time 은 full datetime, code 는 문자열 오름차순).
* ts 인덱스 = 해당 이벤트 프레임의 정렬된 unique bar_time 에서의 0-기반 위치.
  (참조 엔진의 ti 와 동일. 트레이드 로그의 entry_ts/exit_ts 로 기록됨.)

실행/사용
---------
    from live.strategy import BarsSignalSource
    src = BarsSignalSource(start_date="2025-01-06", end_date="2025-07-23")
    for ev in src.events():  # bar_time,code 오름차순
        ...                  # ev.pred, ev.pred_conf, ev.ts, ...
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

# 프로젝트 루트를 import 경로에 추가(어디서 실행해도 config/live import 가능)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows cp949 콘솔에서도 한글/기호가 깨지지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# full_pred 스키마 중 이벤트에 필요한 컬럼 (순서 = _iter_events 언팩 순서)
SIGNAL_COLUMNS = ["date", "bar_time", "code", "open", "close", "pred", "pred_conf"]

DEFAULT_PRED_PATH = ROOT / "data" / "full_pred.parquet"


@dataclass(frozen=True)
class BarEvent:
    """시그널 소스가 방출하는 트레이딩 가능 봉 1개.

    ts 는 이벤트 프레임의 정렬된 unique bar_time 기준 0-기반 전역 타임슬롯 인덱스로,
    참조 엔진(run_portfolio)의 ``ti`` 와 동일하다.
    """

    bar_time: pd.Timestamp
    date: str
    code: str
    open: int
    close: int
    pred: str
    pred_conf: float
    ts: int = -1


# --------------------------------------------------------------------------- #
# 프레임 정규화 / 이벤트 생성
# --------------------------------------------------------------------------- #
def load_signal_frame(path: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """full_pred parquet 를 prepare()와 동일한 규칙으로 정규화한 event frame 반환.

    - bar_time → datetime, 필수 컬럼 결측 행 drop
    - date/code → str
    - date 창(start_date <= date <= end_date) 필터 (포함 경계)
    - [bar_time, code] 안정정렬 (kind='mergesort')
    """
    p = Path(path) if path is not None else DEFAULT_PRED_PATH
    df = pd.read_parquet(p).copy()
    df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce")
    df = df.dropna(subset=["date", "bar_time", "code", "open", "close",
                           "pred", "pred_conf"])
    df["date"] = df["date"].astype(str)
    df["code"] = df["code"].astype(str)
    if start_date is not None:
        df = df[df["date"] >= str(start_date)]
    if end_date is not None:
        df = df[df["date"] <= str(end_date)]
    df = df.sort_values(["bar_time", "code"], kind="mergesort").reset_index(drop=True)
    return df


def _ts_index(df: pd.DataFrame, offset: int = 0) -> dict:
    """정렬된 unique bar_time -> 0-기반(또는 offset-기반) 전역 인덱스."""
    ts_list = sorted(df["bar_time"].unique().tolist())
    return {t: i + offset for i, t in enumerate(ts_list)}


def _iter_events(df: pd.DataFrame, ts_map: dict) -> Iterator[BarEvent]:
    """정규화된 frame 을 BarEvent 스트림으로 변환(frame 은 이미 [bar_time, code] 정렬)."""
    cols = SIGNAL_COLUMNS
    for date, bar_time, code, o, c, pred, conf in df[cols].itertuples(
            index=False, name=None):
        yield BarEvent(
            bar_time=bar_time,
            date=str(date),
            code=str(code),
            open=int(o),
            close=int(c),
            pred=str(pred),
            pred_conf=float(conf),
            ts=ts_map[bar_time],
        )


# --------------------------------------------------------------------------- #
# SignalSource 추상
# --------------------------------------------------------------------------- #
class SignalSource:
    """시그널 소스 공통 인터페이스.

    events() 는 BarEvent 를 [bar_time, code] 오름차순으로 yield 한다.
    ts_index() 는 bar_time -> ts 인덱스 매핑을 반환한다.
    """

    def events(self) -> Iterator[BarEvent]:  # pragma: no cover - 인터페이스
        raise NotImplementedError

    def ts_index(self) -> dict:
        return {}

    def __len__(self) -> int:
        return 0


class BarsSignalSource(SignalSource):
    """bars-mode: full_pred parquet 를 그대로 이벤트 유니버스로 사용."""

    def __init__(self, path: Optional[str] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None) -> None:
        self.df = load_signal_frame(path, start_date, end_date)
        self._ts = _ts_index(self.df)

    def events(self) -> Iterator[BarEvent]:
        return _iter_events(self.df, self._ts)

    def ts_index(self) -> dict:
        return dict(self._ts)

    def __len__(self) -> int:
        return len(self.df)


class FrameSignalSource(SignalSource):
    """주어진 event frame(정규화 전)을 소비하는 소스.

    ticks-mode 에서 MongoFeed→BarStream 봉과 full_pred 시그널을 join 한 frame 을
    그대로 넘긴다.
    """

    def __init__(self, df: pd.DataFrame, ts_offset: int = 0) -> None:
        d = df.copy()
        d["bar_time"] = pd.to_datetime(d["bar_time"], errors="coerce")
        d = d.dropna(subset=["date", "bar_time", "code", "open", "close",
                             "pred", "pred_conf"])
        d["date"] = d["date"].astype(str)
        d["code"] = d["code"].astype(str)
        d = d.sort_values(["bar_time", "code"], kind="mergesort").reset_index(drop=True)
        self.df = d
        self._ts = _ts_index(d, ts_offset)

    def events(self) -> Iterator[BarEvent]:
        return _iter_events(self.df, self._ts)

    def ts_index(self) -> dict:
        return dict(self._ts)

    def __len__(self) -> int:
        return len(self.df)


# --------------------------------------------------------------------------- #
# ticks-mode 헬퍼: 틱 -> 10분봉 -> full_pred 시그널 join
# --------------------------------------------------------------------------- #
def aggregate_ticks(feed, dates: List[str]) -> list:
    """MongoFeed(또는 TickSource)의 일자별 틱을 live/bars.BarStream 으로 10분봉 집계.

    반드시 하루 단위(feed_day: feed + close_day)로 소비해 일자 간 상태 누수를 막는다.
    반환: COLUMNS 순서의 row 튜플 리스트(방출 순서).
    """
    from live.bars import BarStream

    stream = BarStream()
    rows: list = []
    for day in dates:
        rows.extend(stream.feed_day(feed.iter_day(day), date=str(day)))
    return rows


def join_bars_signals(bars_rows: list, signal_df: pd.DataFrame) -> pd.DataFrame:
    """집계된 봉(open/close)과 full_pred 시그널(pred/pred_conf)을 (code, bar_time)로 join.

    이벤트 유니버스 = 두 소스의 교집합(inner join). 반환 frame 은
    [date, bar_time, code, open, close, pred, pred_conf] 이며 date 는 시그널 프레임의
    값을 사용한다. 반환 전 [bar_time, code] 오름차순 정렬.
    """
    from live.bars import COLUMNS

    out_cols = SIGNAL_COLUMNS
    if not bars_rows:
        return pd.DataFrame(columns=out_cols)

    b = pd.DataFrame(list(bars_rows), columns=COLUMNS)
    b["bar_time"] = pd.to_datetime(b["bar_time"])
    b["code"] = b["code"].astype(str)

    s = signal_df[["date", "bar_time", "code", "pred", "pred_conf"]].copy()
    s["bar_time"] = pd.to_datetime(s["bar_time"])
    s["code"] = s["code"].astype(str)

    m = b.merge(s, on=["code", "bar_time"], how="inner", suffixes=("_bar", "_sig"))
    if m.empty:
        return pd.DataFrame(columns=out_cols)
    m["date"] = m["date_sig"].astype(str)
    out = m[out_cols].copy()
    out = out.sort_values(["bar_time", "code"], kind="mergesort").reset_index(drop=True)
    return out
