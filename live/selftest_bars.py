"""live/selftest_bars.py — live 증분 집계가 배치(extract_bars)와 동일한지 검증하는 셀프테스트.

실행 (프로젝트 루트에서):
    python -X utf8 live/selftest_bars.py

검증 항목
---------
1) [핵심] MongoDB의 2025-06-25, 2025-06-26 두 컬렉션을 MongoFeed로 읽어
   BarStream에 흘려보낸 하루치 closed bars 가, 같은 컬렉션을
   extract_bars.process_collection으로 집계한 결과와 DataFrame.equals로 동일.
   (컬럼: date, bar_time, code, name, open, high, low, close, volume, n_ticks)
2) HistoryStore.add_closed_bar() 가 호출순서대로 보관하고,
   bars_for(code) 가 bar_time 오름차순(날짜 포함)으로 반환.
3) ReplayFeed(메모리 재생)가 MongoFeed와 동일한 결과를 반복 생성.

건너뜀(실패 아님)
-----------------
MongoDB에 연결할 수 없거나 대상 컬렉션이 없으면 명확한 SKIP 메시지를 출력하고
종료코드 0으로 끝난다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# --- 경로 설정: 프로젝트 루트 / pipeline / live 를 import 경로에 추가 ---
ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for _p in (str(ROOT), str(ROOT / "pipeline"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Windows cp949 콘솔에서도 한글/기호가 깨지지 않도록 UTF-8 강제
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

import extract_bars                                   # pipeline/extract_bars.py (배치 기준)
from feed import MongoFeed, ReplayFeed                # live/feed.py
from bars import BarStream, HistoryStore, COLUMNS     # live/bars.py

TEST_DATES = ["2025-06-25", "2025-06-26"]


def rows_to_df(rows: list) -> pd.DataFrame:
    """row 리스트를 extract_bars.main과 동일한 스키마/타입/정렬의 DataFrame으로 변환."""
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    df["date"] = df["date"].astype(str)
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    df["code"] = df["code"].astype(str)
    df["name"] = df["name"].astype(str)
    for c in ("open", "high", "low", "close", "volume", "n_ticks"):
        df[c] = df[c].astype("int64")
    return df.sort_values(["date", "bar_time", "code"]).reset_index(drop=True)


def skip(msg: str) -> int:
    print(f"SKIP: {msg}")
    print("SKIP: MongoDB(2025-06-25/26) 준비 후 다시 실행하세요. (실패 아님)")
    return 0


def test_history_store() -> None:
    """HistoryStore: 호출순서 보관 + bars_for bar_time 오름차순(날짜 포함) 검증."""
    store = HistoryStore()
    # 일부러 시간 역순/날짜 뒤섞어 호출 → bars_for가 정렬해 주는지 확인
    store.add_closed_bar("2025-06-26", datetime(2025, 6, 26, 9, 0), "AAA", "에이",
                         100, 110, 90, 105, 5, 3)
    store.add_closed_bar("2025-06-25", datetime(2025, 6, 25, 9, 30), "AAA", "에이",
                         101, 111, 91, 106, 6, 4)
    store.add_closed_bar("2025-06-25", datetime(2025, 6, 25, 9, 10), "AAA", "에이",
                         99, 109, 89, 104, 4, 2)
    store.add_closed_bar("2025-06-25", datetime(2025, 6, 25, 9, 20), "BBB", "비",
                         50, 55, 45, 52, 3, 1)

    bars = store.bars_for("AAA")
    times = [r[1] for r in bars]
    assert times == sorted(times), f"bars_for 정렬 실패: {times}"
    assert len(bars) == 3, f"AAA 봉 개수 오류: {len(bars)}"
    # 날짜를 넘나드는 오름차순(6/25 09:10, 09:30, 6/26 09:00)
    assert times[0].date() == datetime(2025, 6, 25).date()
    assert times[-1].date() == datetime(2025, 6, 26).date()
    # 호출순서 보관 확인
    assert store._order == [("2025-06-26", "AAA"), ("2025-06-25", "AAA"),
                            ("2025-06-25", "AAA"), ("2025-06-25", "BBB")]
    assert store.bars_for("BBB")[0][2] == "BBB"
    assert len(store) == 4
    print("  [ok] HistoryStore: 호출순서 보관 + bars_for bar_time 오름차순")


def main() -> int:
    # --- 1) MongoDB 연결/컬렉션 확인 (없으면 SKIP) ---
    try:
        feed = MongoFeed()
        feed.ping()
    except Exception as exc:  # 서버 미기동 등
        return skip(f"MongoDB 연결 실패 ({type(exc).__name__}: {exc})")

    with feed:
        try:
            days = set(feed.list_days())
        except Exception as exc:
            return skip(f"컬렉션 조회 실패 ({type(exc).__name__}: {exc})")

        missing = [d for d in TEST_DATES if d not in days]
        if missing:
            return skip(f"대상 컬렉션 없음: {missing} (존재 일자 {min(days) if days else '없음'}.."
                        f"{max(days) if days else '없음'}, 총 {len(days)}개)")

        db = feed.db  # 배치 경로(process_collection)에서 재사용
        print(f"테스트 대상: {TEST_DATES}")

        # --- 2) 일자별 증분 vs 배치 비교 ---
        all_batch, all_inc = [], []
        ok = True
        for day in TEST_DATES:
            # (a) 배치: extract_bars.process_collection
            batch_rows: list = []
            n_docs, n_codes = extract_bars.process_collection(db, day, batch_rows)

            # (b) 증분: MongoFeed -> BarStream(하루 단위 feed + close_day)
            stream = BarStream()
            inc_rows = stream.feed_day(feed.iter_day(day), date=day)

            df_batch = rows_to_df(batch_rows)
            df_inc = rows_to_df(inc_rows)
            same = df_batch.equals(df_inc)
            ok = ok and same
            print(f"  [{'ok' if same else 'FAIL'}] {day}: "
                  f"ticks={n_docs:,} codes={n_codes} "
                  f"batch_bars={len(df_batch)} incr_bars={len(df_inc)}")

            if not same:
                # 불일치 진단(개수/키 차이 요약)
                bkeys = set(map(tuple, df_batch[["date", "code", "bar_time"]].to_numpy()))
                ikeys = set(map(tuple, df_inc[["date", "code", "bar_time"]].to_numpy()))
                print(f"        only_batch={len(bkeys - ikeys)} only_incr={len(ikeys - bkeys)}")
                print("        rows_equal=", df_batch.shape == df_inc.shape)

            all_batch.extend(batch_rows)
            all_inc.extend(inc_rows)

        # --- 3) 두 날짜 합산 비교 ---
        df_all_b = rows_to_df(all_batch)
        df_all_i = rows_to_df(all_inc)
        all_same = df_all_b.equals(df_all_i)
        ok = ok and all_same
        print(f"  [{'ok' if all_same else 'FAIL'}] 2일 합산: "
              f"batch_bars={len(df_all_b)} incr_bars={len(df_all_i)}")

        # --- 4) HistoryStore 단위 검증 ---
        test_history_store()

        # --- 5) HistoryStore 통합: on_close로 실데이터 수신 ---
        store = HistoryStore()
        stream2 = BarStream(on_close=store.add_row)
        stream2_rows = []
        for day in TEST_DATES:
            stream2_rows.extend(stream2.feed_day(feed.iter_day(day), date=day))
        assert len(store) == len(stream2_rows), "HistoryStore 개수 불일치"
        ascall = all(
            [r[1] for r in store.bars_for(c)] == sorted(r[1] for r in store.bars_for(c))
            for c in store.codes()
        )
        assert ascall, "HistoryStore bars_for 정렬 위반"
        # HistoryStore 결과도 배치와 동일해야 한다
        assert rows_to_df([r for c in store.codes() for r in store.bars_for(c)]).equals(df_all_b), \
            "HistoryStore 통합 결과가 배치와 불일치"
        print(f"  [ok] HistoryStore 통합: on_close 수신 {len(store)}봉, 종목 {len(store.codes())}개")

        # --- 6) ReplayFeed(메모리 재생)가 MongoFeed와 동일 & 반복 가능 ---
        cached = {day: list(feed.iter_day(day)) for day in TEST_DATES}
        replay = ReplayFeed(cached)
        for pass_no in (1, 2):
            stream3 = BarStream()
            rep_rows = []
            for day in TEST_DATES:
                rep_rows.extend(stream3.feed_day(replay.iter_day(day), date=day))
            assert rows_to_df(rep_rows).equals(df_all_b), f"ReplayFeed pass{pass_no} 불일치"
        print("  [ok] ReplayFeed: 2회 재생 모두 배치와 동일(반복 생성)")

    if ok:
        print("PASS: live 증분 집계 == extract_bars 배치 집계 (DataFrame.equals)")
        print("PASS: HistoryStore / ReplayFeed 검증 완료")
        return 0
    print("FAIL: 증분 집계가 배치와 다릅니다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
