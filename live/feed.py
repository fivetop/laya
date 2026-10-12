"""live/feed.py — 실시간/재생 틱 소스 (TickSource).

역할
----
실시간 트레이딩 파이프라인의 첫 모듈. MongoDB 일자별 틱 컬렉션을 순서대로
흘려보내는 소스를 제공한다. 모든 소스는 동일한 시그니처(TickSource)를 따른다.

- TickSource : 틱을 (dt_utc_naive, code, price, cum_vol, name) 튜플로 하루씩 yield.
- MongoFeed  : MongoDB 일자별 컬렉션(YYYY-MM-DD)을 읽기 전용으로 읽어
               하루치를 dt1(=UTC) 오름차순으로 yield. (라이브용)
- ReplayFeed : 사전 로드된 틱 리스트를 메모리에서 재생(반복 가능). (백테스트/검증용)

실행/사용
---------
    from feed import MongoFeed, ReplayFeed

    with MongoFeed() as feed:
        print(feed.list_days())                 # 존재하는 일자 컬렉션
        for tick in feed.iter_day("2025-06-25"):
            dt, code, price, cum, name = tick

    # 메모리 재생(반복 생성 가능)
    replay = ReplayFeed({"2025-06-25": ticks_a, "2025-06-26": ticks_b})
    for _pass in range(2):
        for tick in replay.iter_day("2025-06-25"):
            ...

설계 원칙 (pipeline/extract_bars.py 이식)
----------------------------------------
- QUERY / PROJECTION / 클라이언트 재검사 / dt1 오름차순 정렬을 extract_bars와 동일하게 유지.
- MongoDB 3.4는 find().sort()가 32MB 초과 시 실패하므로 서버 sort를 쓰지 않고,
  커서를 스트림으로 수신한 뒤 하루치(투영된 소형 튜플)만 메모리에서 정렬한다.
- dt1이 timezone-aware면 UTC naive로 정규화한 뒤 소비자(BarStream)가 KST로 집계한다.
- 읽기 전용: 어떤 쓰기/생성/수정도 하지 않는다.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, Tuple

# 프로젝트 루트를 sys.path에 넣어 어디서 실행해도 config import 가능
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymongo import MongoClient

from config import MONGO_DB, MONGO_URI

# --- 대상 컬렉션: 일자별 틱만 (rssc/stock/stockb/... 제외) ---
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 필요한 필드만 투영 (dt1=UTC, str종목코드, str종목명, int현재가, int거래량=누적)
PROJECTION = {
    "_id": 0,
    "dt1": 1,
    "str종목코드": 1,
    "str종목명": 1,
    "int현재가": 1,
    "int거래량": 1,
}
# 서버측 1차 필터 (클라이언트에서도 방어적 재검사)
QUERY = {
    "dt1": {"$ne": None},
    "int현재가": {"$gt": 0},
    "str종목코드": {"$exists": True},
}

# 틱 튜플 타입: (dt_utc_naive, code, price, cum_vol, name)
Tick = Tuple[datetime, str, int, int, str]


def to_naive_utc(dt: datetime) -> datetime:
    """timezone-aware datetime이면 UTC naive로 변환. naive면 그대로 반환."""
    if dt is not None and dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def normalize_tick(dt: datetime, code: str, price: int, cum_vol: int, name: str) -> Tick:
    """원시 값을 표준 틱 튜플로 정규화(dt는 UTC naive, 숫자는 int, 이름은 str)."""
    return (to_naive_utc(dt), str(code), int(price), int(cum_vol or 0), name or "")


class TickSource:
    """틱 소스 공통 인터페이스.

    iter_day(date_str)는 (dt_utc_naive, code, price, cum_vol, name) 튜플을
    dt1 오름차순으로 yield 한다.
    """

    def iter_day(self, date_str: str) -> Iterator[Tick]:  # pragma: no cover - 인터페이스
        raise NotImplementedError

    def list_days(self) -> list:
        """제공 가능한 일자 목록(YYYY-MM-DD). 기본은 빈 리스트."""
        return []


class MongoFeed(TickSource):
    """MongoDB 일자별 틱 컬렉션을 읽기 전용으로 흘려보내는 라이브 소스."""

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB,
                 timeout_ms: int = 5000) -> None:
        self._client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        self._db = self._client[db_name]

    # --- 연결/메타 ---
    @property
    def db(self):
        """pymongo Database 핸들(읽기 전용 용도). 배치 검증 등에서 재사용."""
        return self._db

    def ping(self) -> None:
        """연결 조기 실패 확인용(서버 미기동이면 예외)."""
        self._db.command("ping")

    def list_days(self) -> list:
        """존재하는 일자 컬렉션만 정렬해 반환."""
        return sorted(n for n in self._db.list_collection_names() if DATE_RE.match(n))

    def has_day(self, date_str: str) -> bool:
        return date_str in set(self._db.list_collection_names())

    # --- 핵심: 하루치 틱 스트림 ---
    def iter_day(self, date_str: str) -> Iterator[Tick]:
        """하루 컬렉션을 dt1 오름차순으로 yield.

        extract_bars.process_collection과 동일하게:
          1) 서버 sort 없이 커서를 스트림 수신
          2) 클라이언트 재검사(가격>0, 코드 존재)
          3) dt1 aware -> UTC naive
          4) 하루치만 메모리 정렬(stable, 동일 dt는 커서 순서 유지)
        """
        cursor = self._db[date_str].find(QUERY, PROJECTION)
        ticks: list = []
        for doc in cursor:
            dt = doc.get("dt1")
            if dt is None:
                continue
            price = doc.get("int현재가")
            if price is None or price <= 0:
                continue
            code = doc.get("str종목코드")
            if not code:
                continue
            ticks.append(normalize_tick(
                dt, code, price, doc.get("int거래량"), doc.get("str종목명")))

        # dt1 오름차순(하루 단위, 메모리 안전). 동일 dt는 수신 순서 유지.
        ticks.sort(key=lambda t: t[0])
        yield from ticks

    # --- 정리 ---
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MongoFeed":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ReplayFeed(TickSource):
    """사전 로드된 틱 튜플 리스트를 메모리에서 재생하는 소스.

    반복 생성(replay)이 가능하도록 생성 시점에 정규화하여 보관한다.
    ticks_by_day: {date_str: [(dt, code, price, cum_vol, name), ...]}
    """

    def __init__(self, ticks_by_day: Mapping[str, Sequence[tuple]]) -> None:
        # 정규화 후 튜플로 고정(재생 중 원본 변형 방지)
        self._days: dict = {
            str(d): [normalize_tick(*t) for t in ticks]
            for d, ticks in ticks_by_day.items()
        }

    def list_days(self) -> list:
        return sorted(self._days.keys())

    def iter_day(self, date_str: str) -> Iterator[Tick]:
        """해당 일자의 틱을 저장된 순서대로 yield. 여러 번 호출 가능."""
        for tick in self._days.get(str(date_str), ()):
            yield tick

    @classmethod
    def from_ticks(cls, ticks: Iterable[tuple], date_str: str) -> "ReplayFeed":
        """단일 일자 틱 리스트로 편의 생성.

        입력 순서를 신뢰하지 않으므로 dt1 오름차순으로 정렬해 보관한다
        (MongoFeed/process_collection과 동일한 소비 순서 보장).
        """
        norm = [normalize_tick(*t) for t in ticks]
        norm.sort(key=lambda t: t[0])
        return cls({str(date_str): norm})
