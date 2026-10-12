---
description: MongoDB 틱 데이터를 10분봉으로 추출 (data/bars_10m.parquet)
---

# 틱 → 10분봉 추출

MongoDB `stock2020`의 일자별 틱 컬렉션에서 데이터를 추출해 10분봉으로 집계합니다.

## 작업 순서

1. `pipeline/extract_bars.py`가 없으면 생성:
   - pymongo 3.12.3 사용 (MongoDB 3.4 호환, 4.x 불가)
   - pymongo 연결 실패 시 `C:\Program Files\MongoDB\Server\3.4\bin\mongo.exe` fallback
   - 일자별 컬렉션 목록: 이름이 `YYYY-MM-DD` 정규식에 매칭되는 것만 (rssc, stock, stockb, stockdt, stockb2021 제외)
   - 필요한 필드만 projection: `str종목코드`, `str종목명`, `dt1`, `int현재가`, `int거래량`, `f등락률`
   - dt1(UTC 기준)을 KST 9:00~15:30 범위로 필터링 후 10분 구간으로 group by
   - 구간별: 시가(first 현재가), 고가(max), 저가(min), 종가(last 현재가), 거래량(sum)
   - 결측 일자(4/23~5/27, 6/3 등)는 건너뛰고 로그만 남김
2. 출력: `data/bars_10m.parquet` (config.BARS_PATH)
3. 검증: 행 수, distinct 종목 수, 날짜 범위, 구간당 봉 수(하루 39봉/종목 기대) 출력

## MUST NOT
- 모든 필드를 그대로 읽지 말 것 (메모리 폭발 방지, projection 필수)
- dt1이 null인 문서 혼입
- 8/17 이전(데이터 없음)·주말 포함 날짜 강제 생성

작업 완료 후 결과 요약(행수/종목수/기간)을 보고.
