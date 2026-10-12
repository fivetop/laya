"""live/selftest_state.py — LiveState.state_for 가 배치 state(labeled.parquet)와 일치하는지 검증.

실행 (프로젝트 루트, UTF-8 필수):
    python -X utf8 live/selftest_state.py
    python -X utf8 live/selftest_state.py --source parquet   # 빠른 검증(data/bars_10m.parquet)
    python -X utf8 live/selftest_state.py --source mongo     # MongoDB 일자별 컬렉션에서 직접 집계

절차
----
1) 2025-01-06 .. 2025-06-25 (평가일 2025-06-25 직전 예열 포함) 봉을 시간순으로
   live/bars.HistoryStore 에 적재한다.
     - source=mongo  : live/feed.MongoFeed(틱) → live/bars.BarStream(10분봉) → HistoryStore
     - source=parquet: data/bars_10m.parquet (extract_bars 가 MongoDB 일자별 컬렉션에서
                       추출한 동일한 10분봉) → HistoryStore
     - source=auto   : MongoDB 연결 가능하면 mongo, 아니면 parquet (기본값)
2) 2025-06-25 스냅샷(labeled 의 date=='2025-06-25') 중 state 컬럼이 전부 유한한 행에 대해
   state_for(code, bar_time) 결과를 6자리 반올림으로 비교한다.
   비교 state 컬럼 = labeled 전체 컬럼에서 make_jsonl.STATE_EXCLUDE 를 뺀 것
   (= code + 수치 24개, 총 25키).
3) 일치율(셀 기준/행 기준)과 불일치 샘플(최대 5개)을 출력한다.
   일치율 >= 97% 이면 PASS.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- 경로: 프로젝트 루트(+pipeline) 를 import 경로에 추가 ---
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows 콘솔에서 한글/기호 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
from pipeline.make_jsonl import select_state_columns  # noqa: E402  (읽기 전용 import)
from live.bars import HistoryStore, COLUMNS  # noqa: E402
from live.state import LiveState, STATE_KEYS  # noqa: E402

EVAL_DATE = "2025-06-25"
LOAD_FROM = "2025-01-06"      # 평가일 직전 예열 포함(가능한 가장 이른 일자)
LOAD_TO = EVAL_DATE
PASS_THRESHOLD = 0.97


def load_store_parquet(store: HistoryStore) -> int:
    """data/bars_10m.parquet(MongoDB 추출 봉)을 시간순으로 HistoryStore 에 적재."""
    bars = pd.read_parquet(config.BARS_PATH)
    bars["bar_time"] = pd.to_datetime(bars["bar_time"])
    bars = bars[(bars["date"] >= LOAD_FROM) & (bars["date"] <= LOAD_TO)]
    bars = bars.sort_values(["bar_time", "code"], kind="mergesort").reset_index(drop=True)
    for row in bars[COLUMNS].itertuples(index=False, name=None):
        store.add_row(row)
    return len(bars)


def load_store_mongo(store: HistoryStore) -> int:
    """MongoDB 일자별 컬렉션 → BarStream → HistoryStore (실시간 경로 그대로)."""
    from live.feed import MongoFeed
    from live.bars import BarStream

    n = 0
    with MongoFeed() as feed:
        feed.ping()
        stream = BarStream(on_close=store.add_row)
        days = [d for d in feed.list_days() if LOAD_FROM <= d <= LOAD_TO]
        for i, day in enumerate(days, 1):
            rows = stream.feed_day(feed.iter_day(day), date=day)
            n += len(rows)
            print(f"  [{i:>3}/{len(days)}] {day}: {len(rows):>5} closed bars", flush=True)
    return n


def build_store(source: str) -> tuple[HistoryStore, str]:
    """지정 소스로 HistoryStore 를 채운다. (store, 실제사용소스) 반환."""
    store = HistoryStore()
    if source == "parquet":
        n = load_store_parquet(store)
        return store, f"parquet({config.BARS_PATH.name})"
    if source == "mongo":
        n = load_store_mongo(store)
        return store, "mongo(MongoFeed→BarStream)"
    # auto: Mongo 연결되면 mongo, 아니면 parquet
    try:
        from live.feed import MongoFeed
        f = MongoFeed()
        f.ping()
        days = set(f.list_days())
        f.close()
        needed = [d for d in days if LOAD_FROM <= d <= LOAD_TO]
        if not needed:
            raise RuntimeError("MongoDB 에 대상 일자 컬렉션이 없음")
        n = load_store_mongo(store)
        return store, "mongo(MongoFeed→BarStream)"
    except Exception as exc:  # noqa: BLE001 - 폴백 사유 출력
        print(f"  (MongoDB 사용 불가: {type(exc).__name__}: {exc} → parquet 로 대체)")
        store = HistoryStore()  # 부분 적재분 혼입 방지: 새 저장소로 재적재
        load_store_parquet(store)
        return store, f"parquet({config.BARS_PATH.name})"


def main() -> int:
    ap = argparse.ArgumentParser(description="LiveState vs labeled.parquet 일치 검증")
    ap.add_argument("--source", choices=["auto", "mongo", "parquet"], default="auto",
                    help="봉 적재 소스 (auto: mongo 시도 후 parquet 폴백)")
    args = ap.parse_args()

    print("=== selftest_state (live state builder) ===")
    print(f"기간      : {LOAD_FROM} .. {LOAD_TO}  (평가일 {EVAL_DATE} 예열 포함)")

    store, used = build_store(args.source)
    print(f"봉 적재   : {len(store):,} 봉 / {len(store.codes()):,} 종목  (source={used})")

    builder = LiveState(store)

    # --- 배치 정답(labeled.parquet) ---
    labeled = pd.read_parquet(config.LABELED_PATH)
    state_cols = select_state_columns(labeled)   # make_jsonl 규칙과 동일
    if set(state_cols) != set(STATE_KEYS):
        print(f"WARN: state 컬럼 불일치\n  labeled: {state_cols}\n  state  : {STATE_KEYS}")
    if set(state_cols) != set(STATE_KEYS) or len(state_cols) != 25:
        print(f"FAIL: state 키 개수/집합 불일치 (labeled={len(state_cols)}, state={len(STATE_KEYS)})")
        return 1
    print(f"state 키  : {len(state_cols)}개 (code + 수치 {len(state_cols) - 1})")

    # --- 평가 대상 행: 2025-06-25, state 전부 유한 ---
    ev = labeled[labeled["date"] == EVAL_DATE].copy()
    num_cols = [c for c in state_cols if c != "code"]
    arr = ev[num_cols].to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(arr).all(axis=1)
    n_all = len(ev)
    ev = ev[finite].reset_index(drop=True)
    print(f"평가 행   : {EVAL_DATE} {len(ev):,}건 (state 유한) / 전체 {n_all:,}건")

    # --- 비교 ---
    total_cells = 0
    match_cells = 0
    row_ok = 0
    mismatches: list[tuple] = []

    for r in ev.itertuples(index=False):
        code = str(getattr(r, "code"))
        T = pd.Timestamp(getattr(r, "bar_time"))
        actual = builder.state_for(code, T)
        if actual is None:
            total_cells += len(state_cols)
            row_match = False
            if len(mismatches) < 5:
                mismatches.append((T, code, "<state>", "dict(25)", "None(미완성)"))
        else:
            row_match = True
            for c in state_cols:
                total_cells += 1
                exp = getattr(r, c)
                act = actual.get(c)
                if c == "code":
                    ok = str(exp) == str(act)
                else:
                    try:
                        ok = round(float(exp), 6) == round(float(act), 6)
                    except (TypeError, ValueError):
                        ok = False
                if ok:
                    match_cells += 1
                else:
                    row_match = False
                    if len(mismatches) < 5:
                        mismatches.append((T, code, c, exp, act))
        if row_match:
            row_ok += 1

    cell_rate = match_cells / total_cells if total_cells else 1.0
    row_rate = row_ok / len(ev) if len(ev) else 1.0

    print("--- verdict ---")
    print(f"일치율(셀) : {cell_rate * 100:.4f}%  ({match_cells:,}/{total_cells:,})")
    print(f"일치율(행) : {row_rate * 100:.4f}%  ({row_ok:,}/{len(ev):,})")

    if mismatches:
        print("불일치 샘플 (최대 5):")
        for T, code, col, exp, act in mismatches:
            print(f"  bar_time={T}  code={code}  col={col}  expected={exp!r}  actual={act!r}")

    passed = cell_rate >= PASS_THRESHOLD and row_rate >= PASS_THRESHOLD
    if passed:
        print(f"PASS: state_for == labeled (셀 {cell_rate * 100:.4f}%, 행 {row_rate * 100:.4f}% "
              f">= {PASS_THRESHOLD * 100:.0f}%)  [source={used}]")
        return 0
    print(f"FAIL: 일치율이 {PASS_THRESHOLD * 100:.0f}% 미만입니다 "
          f"(셀 {cell_rate * 100:.4f}%, 행 {row_rate * 100:.4f}%)  [source={used}]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
