"""extract_bars.py — MongoDB 일자별 틱 → 10분 OHLCV 봉 집계.

실행 (프로젝트 루트에서):  python pipeline/extract_bars.py
출력: data/bars_10m.parquet

- pymongo 3.12.3 (MongoDB 3.4 서버) — 읽기 전용, 컬렉션 생성/수정 없음
- 일자별 컬렉션(YYYY-MM-DD)만 대상. 하루씩 스트림 처리(전체 적재 없음).
- int거래량 = 장중 누적 거래량 → 봉 거래량 = (봉 마지막 누적) - (직전 봉 마지막 누적),
  첫 봉은 장 첫 틱의 누적값 기준. 음수면 0으로 클램프.
"""
from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 프로젝트 루트를 sys.path에 넣어 어디서 실행해도 config import 가능
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from pymongo import MongoClient

from config import BARS_PATH, BAR_MINUTES, MONGO_DB, MONGO_URI, TRADING_END, TRADING_START

# --- 대상 컬렉션: 일자별 틱만 (rssc/stock/stockb/stockb2021/stockdt 등 제외) ---
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

T_START = datetime.strptime(TRADING_START, "%H:%M").time()
T_END = datetime.strptime(TRADING_END, "%H:%M").time()
KST = timedelta(hours=9)

COLUMNS = ["date", "bar_time", "code", "name",
           "open", "high", "low", "close", "volume", "n_ticks"]


def _new_pending(bucket: datetime, price: int, cum: int, name: str) -> dict:
    return {"bucket": bucket, "open": price, "high": price, "low": price,
            "close": price, "cum_end": cum, "n": 1, "name": name}


def _finalize(rows: list, date_str: str, code: str, st: dict) -> None:
    """open 상태인 봉을 확정하고 결과 행을 append."""
    p = st["pending"]
    # 첫 봉: 장 첫 틱의 누적량 기준 / 이후 봉: 직전 봉 마지막 누적량 기준
    baseline = st["prev_cum_end"] if st["prev_cum_end"] is not None else st["day_first_cum"]
    vol = p["cum_end"] - baseline
    if vol < 0:                      # 데이터 이상 시 음수 클램프
        vol = 0
    rows.append((date_str, p["bucket"], code, p["name"],
                 p["open"], p["high"], p["low"], p["close"],
                 int(vol), p["n"]))
    st["prev_cum_end"] = p["cum_end"]
    st["pending"] = None


def process_collection(db, date_str: str, rows: list) -> tuple[int, int]:
    """하루 컬렉션 1개를 dt1 오름차순으로 집계. (틱 수, 종목 수) 반환.

    MongoDB 3.4는 find().sort()가 32MB 초과 시 실패(디스크 스파일 없음)하므로
    서버 sort 없이 스트림으로 받은 뒤 하루치만(투영된 소형 튜플) 메모리에서 정렬.
    """
    cursor = db[date_str].find(QUERY, PROJECTION)

    # --- 스트림 수신: 필요한 5개 필드만 소형 튜플로 저장 ---
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
        if dt.tzinfo is not None:  # pymongo 기본은 naive UTC datetime
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        ticks.append((dt, code, int(price),
                      int(doc.get("int거래량") or 0),
                      doc.get("str종목명") or ""))
    n_docs = len(ticks)

    # --- dt1 오름차순 정렬(하루 단위, 메모리 안전) 후 집계 ---
    ticks.sort(key=lambda t: t[0])
    states: dict[str, dict] = {}

    for dt, code, price, cum, name in ticks:
        kst = dt + KST

        st = states.get(code)
        if st is None:  # 종목의 장 첫 틱 = 누적 거래량 기준점
            st = states[code] = {"day_first_cum": cum, "prev_cum_end": None, "pending": None}

        # 장시간 밖 틱: 봉 만들지 않음 (첫 틱 기준점 설정은 위에서 이미 수행)
        if not (T_START <= kst.time() < T_END):
            continue

        # 10분 버킷으로 바닥 정렬
        bucket = kst.replace(minute=(kst.minute // BAR_MINUTES) * BAR_MINUTES,
                             second=0, microsecond=0)

        p = st["pending"]
        if p is None:
            st["pending"] = _new_pending(bucket, price, cum, name)
        elif p["bucket"] != bucket:  # 새 버킷 진입 → 이전 봉 확정
            _finalize(rows, date_str, code, st)
            st["pending"] = _new_pending(bucket, price, cum, name)
        else:  # 같은 버킷 갱신
            if price > p["high"]:
                p["high"] = price
            if price < p["low"]:
                p["low"] = price
            p["close"] = price
            p["cum_end"] = cum
            p["n"] += 1
            p["name"] = name

    # 컬렉션 종료: 남은 open 봉 모두 확정
    for code, st in states.items():
        if st["pending"] is not None:
            _finalize(rows, date_str, code, st)

    return n_docs, len(states)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.perf_counter()
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGO_DB]
    db.command("ping")  # 연결 조기 실패

    collections = sorted(n for n in db.list_collection_names() if DATE_RE.match(n))
    print(f"daily tick collections: {len(collections)}", flush=True)

    all_rows: list = []
    total_docs = 0
    for i, date_str in enumerate(collections, 1):
        t1 = time.perf_counter()
        before = len(all_rows)
        n_docs, n_codes = process_collection(db, date_str, all_rows)
        total_docs += n_docs
        print(f"[{i:>3}/{len(collections)}] {date_str}: {n_docs:>8} ticks, "
              f"{n_codes:>4} codes -> {len(all_rows) - before:>5} bars "
              f"({time.perf_counter() - t1:.1f}s)", flush=True)
    client.close()

    # --- DataFrame 작성 (멱등: 매 실행 덮어쓰기) ---
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["date"] = df["date"].astype(str)
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    df["code"] = df["code"].astype(str)
    df["name"] = df["name"].astype(str)
    for c in ("open", "high", "low", "close", "volume", "n_ticks"):
        df[c] = df[c].astype("int64")
    df = df.sort_values(["date", "bar_time", "code"]).reset_index(drop=True)

    BARS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BARS_PATH, index=False)  # 기존 파일 덮어쓰기

    # --- 검증 요약 ---
    per_stock_day = df.groupby(["date", "code"], sort=False).size()
    dup = int(df.duplicated(["date", "code", "bar_time"]).sum())
    print("=" * 64)
    print(f"total rows          : {len(df):,}")
    print(f"distinct codes      : {df['code'].nunique()}")
    print(f"date range          : {df['date'].min()} .. {df['date'].max()}")
    print(f"distinct dates      : {df['date'].nunique()}")
    print(f"input ticks         : {total_docs:,}")
    print(f"bar_time range      : {df['bar_time'].min()} .. {df['bar_time'].max()}")
    print(f"duplicate keys      : {dup}  (date, code, bar_time)")
    print("bars-per-stock-per-day distribution (bars: stock-days):")
    print(per_stock_day.value_counts().sort_index().to_string())
    print(f"max bars/stock/day  : {per_stock_day.max()}  (expect <= 39)")
    print(f"written             : {BARS_PATH}")
    print(f"elapsed             : {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
