"""live/bars.py — 틱 → 10분봉 실시간 증분 집계 (BarStream) + 종목별 봉 히스토리 (HistoryStore).

역할
----
실시간 트레이딩 파이프라인의 두 번째 모듈.

- BarStream   : TickSource가 흘려보내는 틱을 하나씩 소비하면서 종목별 상태를 보존하고,
                10분 버킷이 바뀔 때마다 "완성된 봉"을 즉시 방출(반환/on_close 콜백).
                close_day()로 하루를 마감(잔여 봉 확정)하고 상태를 리셋한다.
- HistoryStore: 확정된 (date, code) 봉을 호출순서대로 보관하고,
                bars_for(code)는 bar_time 오름차순(날짜 포함)으로 반환한다.

집계 규칙은 pipeline/extract_bars.py와 100% 동일한 "증분 버전"이다:
 (a) 버킷 = KST(+9h) 기준 minute//BAR_MINUTES*BAR_MINUTES, second=microsecond=0
 (b) 장중 [TRADING_START, TRADING_END) 밖 틱은 봉에서 제외하되,
     종목별 첫 틱의 누적 거래량은 기준점(day_first_cum)으로 기록
 (c) 봉 거래량 = 봉 마지막 누적 − 직전 봉 마지막 누적(첫 봉은 장 첫 틱 누적 기준),
     음수면 0으로 클램프
 (d) open=첫가 high=max low=min close=마지막가, n_ticks=틱 수
 (e) close_day()에서 남은 pending 봉을 확정

실행/사용
---------
    from feed import MongoFeed
    from bars import BarStream, HistoryStore

    store = HistoryStore()
    stream = BarStream(on_close=store.add_row)
    with MongoFeed() as feed:
        for day in feed.list_days():
            rows = stream.feed_day(feed.iter_day(day), date=day)  # 완성 봉 리스트
            # feed_day = feed() 소비 + close_day() 잔여 확정

주의
----
- 반드시 하루 단위로 feed() 후 close_day()를 호출해야 한다(일자 간 상태 누수 방지).
- date 인자는 extract_bars의 "date" 컬럼(컬렉션명)과 일치시켜야 배치와 결과가 동일하다.
"""
from __future__ import annotations

import sys
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional

# 프로젝트 루트를 sys.path에 넣어 어디서 실행해도 config import 가능
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BAR_MINUTES, TRADING_END, TRADING_START

# extract_bars와 동일한 봉 컬럼 순서
COLUMNS = ["date", "bar_time", "code", "name",
           "open", "high", "low", "close", "volume", "n_ticks"]

KST = timedelta(hours=9)  # 한국 표준시 오프셋(UTC+9)
T_START = datetime.strptime(TRADING_START, "%H:%M").time()
T_END = datetime.strptime(TRADING_END, "%H:%M").time()

# 한 행(row) 튜플: (date, bar_time, code, name, open, high, low, close, volume, n_ticks)
Row = tuple


def _new_pending(bucket: datetime, price: int, cum: int, name: str) -> dict:
    """새 버킷(진행 중 봉) 상태 생성. extract_bars._new_pending과 동일."""
    return {"bucket": bucket, "open": price, "high": price, "low": price,
            "close": price, "cum_end": cum, "n": 1, "name": name}


class BarStream:
    """틱을 10분봉으로 실시간 증분 집계하는 상태 보존형 스트림.

    내부 상태는 종목별 dict(states)이며 extract_bars의 states 구조를 유지한다:
        states[code] = {"day_first_cum": int, "prev_cum_end": int|None, "pending": dict|None}

    feed(ticks, date)는 틱 하나씩 소비하며 버킷 전환 시 완성 봉을 yield 하고,
    (on_close가 지정되면) 같은 봉을 콜백으로도 전달한다.
    close_day()는 잔여 pending 봉을 확정해 반환하고 상태를 리셋한다.
    """

    def __init__(self, on_close: Optional[Callable[[Row], None]] = None,
                 bar_minutes: int = BAR_MINUTES,
                 t_start: dtime = T_START, t_end: dtime = T_END) -> None:
        self._on_close = on_close
        self._bar_minutes = bar_minutes
        self._t_start = t_start
        self._t_end = t_end
        self._states: dict = {}     # 종목코드 -> 상태 dict (삽입순 보존)
        self._date: Optional[str] = None   # 현재 처리 중인 일자 라벨(컬렉션명)

    # --- 공개 API ---
    def feed(self, ticks: Iterable[tuple], date: Optional[str] = None) -> Iterator[Row]:
        """틱을 하나씩 소비하며 버킷 전환 시 완성 봉을 yield.

        date: extract_bars의 'date' 컬럼 값(컬렉션명). None이면 버킷 날짜에서 유도.
        """
        if date is not None:
            self._date = str(date)
        for tick in ticks:
            row = self._consume(tick)
            if row is not None:
                if self._on_close is not None:
                    self._on_close(row)
                yield row

    def close_day(self) -> List[Row]:
        """하루 마감: 남은 pending 봉을 모두 확정해 반환하고 상태를 리셋.

        extract_bars.process_collection의 컬렉션 종료 처리와 동일
        (states 삽입순으로 확정 → 배치와 방출 순서 일치).
        """
        rows: List[Row] = []
        for code, st in self._states.items():
            if st["pending"] is not None:
                row = self._finalize(code, st)
                rows.append(row)
                if self._on_close is not None:
                    self._on_close(row)
        self._states = {}
        self._date = None
        return rows

    def feed_day(self, ticks: Iterable[tuple], date: Optional[str] = None) -> List[Row]:
        """편의 메서드: 하루치 틱 소비 + 잔여 봉 확정까지 수행한 완성 봉 리스트."""
        rows = list(self.feed(ticks, date=date))
        rows.extend(self.close_day())
        return rows

    # --- 내부: 틱 1개 처리 ---
    def _consume(self, tick: tuple) -> Optional[Row]:
        dt, code, price, cum, name = tick

        # dt1이 aware면 UTC naive로 정규화(피드에서 이미 정규화했더라도 방어적 처리)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        kst = dt + KST  # KST 기준 시각

        st = self._states.get(code)
        if st is None:  # 종목의 장 첫 틱 = 누적 거래량 기준점
            st = self._states[code] = {"day_first_cum": cum,
                                       "prev_cum_end": None, "pending": None}

        # 장시간 밖 틱: 봉 만들지 않음 (기준점 설정은 위에서 이미 수행)
        if not (self._t_start <= kst.time() < self._t_end):
            return None

        # 10분 버킷으로 바닥 정렬
        bucket = kst.replace(minute=(kst.minute // self._bar_minutes) * self._bar_minutes,
                             second=0, microsecond=0)

        p = st["pending"]
        if p is None:
            st["pending"] = _new_pending(bucket, price, cum, name)
            return None
        if p["bucket"] != bucket:      # 새 버킷 진입 → 이전 봉 확정 후 새 봉 시작
            row = self._finalize(code, st)
            st["pending"] = _new_pending(bucket, price, cum, name)
            return row
        # 같은 버킷 갱신
        if price > p["high"]:
            p["high"] = price
        if price < p["low"]:
            p["low"] = price
        p["close"] = price
        p["cum_end"] = cum
        p["n"] += 1
        p["name"] = name
        return None

    # --- 내부: pending 봉 확정 ---
    def _finalize(self, code: str, st: dict) -> Row:
        """open 상태인 봉을 확정하고 결과 행(row)을 반환.

        extract_bars._finalize와 동일:
          첫 봉은 장 첫 틱 누적(day_first_cum) / 이후 봉은 직전 봉 마지막 누적 기준.
        """
        p = st["pending"]
        baseline = st["prev_cum_end"] if st["prev_cum_end"] is not None else st["day_first_cum"]
        vol = p["cum_end"] - baseline
        if vol < 0:                    # 데이터 이상 시 음수 클램프
            vol = 0
        date_label = self._date if self._date is not None else p["bucket"].strftime("%Y-%m-%d")
        row = (date_label, p["bucket"], code, p["name"],
               p["open"], p["high"], p["low"], p["close"],
               int(vol), p["n"])
        st["prev_cum_end"] = p["cum_end"]
        st["pending"] = None
        return row


class HistoryStore:
    """확정 봉을 종목별로 보관하는 히스토리 저장소.

    - add_closed_bar(...) : 호출순서대로 (date, code) 봉을 append.
    - bars_for(code)      : 해당 종목 봉을 bar_time 오름차순(날짜 포함)으로 반환.
    """

    def __init__(self) -> None:
        self._by_code: dict = {}   # code -> list[row]
        self._order: list = []     # (date, code) 호출순 기록

    def add_closed_bar(self, date: str, bar_time: datetime, code: str, name: str,
                       open: int, high: int, low: int, close: int,
                       volume: int, n_ticks: int) -> Row:
        """확정 봉 1개를 호출순서대로 보관하고 row를 반환."""
        row = (str(date), bar_time, str(code), str(name),
               int(open), int(high), int(low), int(close), int(volume), int(n_ticks))
        self._by_code.setdefault(str(code), []).append(row)
        self._order.append((str(date), str(code)))
        return row

    def add_row(self, row: tuple) -> Row:
        """COLUMNS 순서의 row 튜플을 add_closed_bar로 위임 (BarStream on_close용)."""
        return self.add_closed_bar(*row)

    def bars_for(self, code: str) -> List[Row]:
        """종목 봉을 bar_time 오름차순(날짜 포함)으로 반환."""
        return sorted(self._by_code.get(str(code), ()), key=lambda r: r[1])

    def days_for(self, code: str) -> List[str]:
        """해당 종목이 등장한 일자 목록(중복 제거, 등장 순서)."""
        seen: dict = {}
        for row in self._by_code.get(str(code), ()):
            seen.setdefault(row[0], None)
        return list(seen.keys())

    def codes(self) -> List[str]:
        """보관 중인 종목코드(삽입순)."""
        return list(self._by_code.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_code.values())
