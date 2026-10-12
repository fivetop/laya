"""단계 [2]: 10분봉 → 기술지표 피처 (per 종목).

읽기: data/bars_10m.parquet  (date, bar_time, code, name, open, high, low, close, volume, n_ticks)
쓰기: data/features.parquet  (원본 컬럼 유지 + 아래 피처 추가)

종목별(code 그룹, bar_time 정렬)로 계산하는 피처:
  ret1        = close.pct_change()                    1봉 전가격 대비 수익률
  rsi14       = 단순 14봉 평균 RSI (avg gain/avg loss, Wilder 아님 — ai_server 스타일)
                avg_loss==0 이고 avg_gain>0 → 100, 둘 다 0 → 50
  macd        = EMA(12) - EMA(26), adjust=False
  macd_signal = EMA(9) of macd
  macd_hist   = macd - macd_signal
  sma5/20/60  = close rolling mean (sma60은 처음 59봉 NaN 유지)
  vol_ratio   = volume / volume.rolling(20).mean()
  range_pct   = (high - low) / close
  bar_ret     = close / open - 1
  dist_sma20  = close / sma20 - 1
  consec_up   = 연속 상승봉 수 (ret1>0 유지, 아님 0으로 리셋)
  day_progress= (bar_time - 09:00) / (15:30 - 09:00)  0..1 (config.TRADING_START/END)

모집단 맥락 피처 (date 내 cross-section — DB에는 그날 거래 상위 50종목만 저장되므로
"저장된 종목 안의 rank"가 곧 시장 내 상위권 강도 순위의 부분집합):
  cum_val          = 현재 bar까지의 당일 누적 거래금액 cumsum(close * volume), (date, code) 그룹
  rank_val         = cum_val의 (date, bar_time) 스냅샷 내 rank — 1 = 최대, method='first'로 결정적
  rank_vol         = 당일 누적 거래량 cumsum(volume)의 동일 rank (같은 방식)
  day_tot_val_share= cum_val / 해당 시점까지의 당일 전체 누적 거래금액 (모든 code 합) — 0..1
  rank는 date 그룹 안에서만 계산 (cross-day 금지). volume은 bar 거래량(daily cumvolume의 diff).

과거만 사용 — 미래 정보 없음. 워밍업 구간 NaN은 허용(다음 단계 make_jsonl에서 제거).

실행: python pipeline/build_features.py [--bars PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import config  # noqa: E402  (project-root config)

FEATURE_COLUMNS = [
    "ret1", "rsi14", "macd", "macd_signal", "macd_hist",
    "sma5", "sma20", "sma60", "vol_ratio", "range_pct",
    "bar_ret", "dist_sma20", "consec_up", "day_progress",
    # --- 모집단 맥락 (당일 누적 거래금액/거래량 rank, date 내 cross-section) ---
    "cum_val", "rank_val", "rank_vol", "day_tot_val_share",
]

RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOL_MA_PERIOD = 20


def _to_timedelta_hhmm(s: str) -> pd.Timedelta:
    hh, mm = s.split(":")
    return pd.Timedelta(hours=int(hh), minutes=int(mm))


def _add_stock_features(g: pd.DataFrame) -> pd.DataFrame:
    """한 종목(오름차순 정렬 보장)에 피처 컬럼 추가."""
    g = g.sort_values("bar_time")

    close = g["close"].astype("float64")
    open_ = g["open"].astype("float64")
    high = g["high"].astype("float64")
    low = g["low"].astype("float64")
    volume = g["volume"].astype("float64")

    out = g.copy()

    # --- 수익률 ---
    out["ret1"] = close.pct_change()

    # --- RSI14: 단순 14봉 평균 (Wilder 평균이 아닌 simple rolling mean) ---
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)  # 0 나눗셈 회피
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    rsi = rsi.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    out["rsi14"] = rsi

    # --- MACD(12,26,9) ---
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    out["macd"] = ema_fast - ema_slow
    out["macd_signal"] = out["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # --- 이동평균 ---
    out["sma5"] = close.rolling(5, min_periods=5).mean()
    out["sma20"] = close.rolling(20, min_periods=20).mean()
    out["sma60"] = close.rolling(60, min_periods=60).mean()  # 첫 59봉 NaN 유지

    # --- 거래량/변동성/막봉 ---
    out["vol_ratio"] = volume / volume.rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean()
    out["range_pct"] = (high - low) / close
    out["bar_ret"] = close / open_ - 1.0
    out["dist_sma20"] = close / out["sma20"] - 1.0

    # --- 연속 상승봉: (ret1>0)가 False일 때 블록 번호를 올리고 블록 내 누적합 ---
    up = (out["ret1"] > 0.0).fillna(False)
    block = (~up).cumsum()
    out["consec_up"] = up.groupby(block).cumsum().astype("int64")

    # --- 장 진행률 0..1 ---
    start = _to_timedelta_hhmm(config.TRADING_START)
    end = _to_timedelta_hhmm(config.TRADING_END)
    tod = out["bar_time"] - out["bar_time"].dt.normalize()
    out["day_progress"] = ((tod - start) / (end - start)).astype("float64").clip(0.0, 1.0)

    return out


def _add_market_context(out: pd.DataFrame) -> pd.DataFrame:
    """date 내 cross-section 피처 (모집단 맥락) 추가 — 코드별 피처가 계산된 뒤 호출.

    out: ["code", "date", "bar_time"] 오름차순 정렬, index 고유 보장.
      cum_val          : (date, code) 그룹 cumsum(close * volume) — 현재 bar까지의 당일 누적 거래금액
      rank_val         : (date, bar_time) 스냅샷 내 cum_val rank, 1 = 최대,
                         method='first' (동률은 정렬된 code 순서로 결정적)
      rank_vol         : (date, bar_time) 스냅샷 내 cum(volume) rank — 동일 방식
      day_tot_val_share: cum_val / (해당 시점까지의 당일 전체 누적 거래금액, 모든 code 합) 0..1
    rank는 (date, bar_time) 그룹 = date 안에서만 계산 — cross-day rank 없음.
    """
    # 시점별 스냅샷 계산용: 같은 bar_time은 code 오름차순 → 동률 tie-break 결정적
    t = out.sort_values(["date", "bar_time", "code"], kind="mergesort")
    volume = t["volume"].astype("float64")
    bar_val = t["close"].astype("float64") * volume  # bar 거래금액 근사

    # --- 당일 누적 거래금액/거래량 (date별 code 그룹, 과거만) ---
    cum_val = bar_val.groupby([t["date"], t["code"]], sort=False).cumsum()
    cum_vol = volume.groupby([t["date"], t["code"]], sort=False).cumsum()

    # --- 스냅샷 내 rank (1 = 최대, method='first'로 결정적) ---
    rank_val = cum_val.groupby([t["date"], t["bar_time"]], sort=False).rank(
        method="first", ascending=False)
    rank_vol = cum_vol.groupby([t["date"], t["bar_time"]], sort=False).rank(
        method="first", ascending=False)

    # --- 당일 전체 누적 거래금액: 같은 bar_time의 전 code 합(스냅샷)을 date별로 누적 ---
    # 분모 = 해당 시점까지 거래된 모든 code의 bar 거래금액 총합 = Σ_code cum_val(그 시점)
    ts_sum = bar_val.groupby([t["date"], t["bar_time"]], sort=False).transform("sum")
    first_of_ts = ~t.duplicated(subset=["date", "bar_time"], keep="first")
    day_cum = ts_sum[first_of_ts].groupby(t.loc[first_of_ts, "date"], sort=False).cumsum()
    denom = day_cum.reindex(t.index).ffill()  # 각 행 → 자신의 bar_time 스냅샷 총합
    day_tot_val_share = cum_val / denom.where(denom > 0.0)  # Series: index 정렬 유지

    out["cum_val"] = cum_val
    out["rank_val"] = rank_val
    out["rank_vol"] = rank_vol
    out["day_tot_val_share"] = day_tot_val_share  # Series → index 정렬로 대입 (순서 보존)
    return out


def build_features(bars_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(bars_path)
    required = {"date", "bar_time", "code", "name", "open", "high", "low",
                "close", "volume", "n_ticks"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"bars parquet missing columns: {sorted(missing)}")

    df = df.copy()
    df["bar_time"] = pd.to_datetime(df["bar_time"])

    parts = [_add_stock_features(g) for _, g in df.groupby("code", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["code", "date", "bar_time"], kind="mergesort").reset_index(drop=True)
    out = _add_market_context(out)  # 모집단 맥락 피처 (date cross-section) — 맨 뒤 append

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    # --- 요약 ---
    print("=== build_features ===")
    print(f"in       : {bars_path}")
    print(f"rows     : {len(out):,}  stocks={out['code'].nunique()}  dates={out['date'].nunique()}")
    print(f"new cols : {', '.join(FEATURE_COLUMNS)}")
    nan = out[FEATURE_COLUMNS].isna().sum()
    print("NaN counts (warmup expected):")
    for col in FEATURE_COLUMNS:
        print(f"  {col:<13} {int(nan[col]):>7,}")
    tail = out[out.groupby('code').cumcount() >= 60]
    print(f"rows past 60-bar warmup with any NaN in new cols: "
          f"{int(tail[FEATURE_COLUMNS].isna().any(axis=1).sum()):,} (expect 0)")
    print(f"out      : {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="bars_10m.parquet → features.parquet")
    ap.add_argument("--bars", type=Path, default=config.BARS_PATH, help="input bars parquet")
    ap.add_argument("--out", type=Path, default=config.FEATURES_PATH, help="output features parquet")
    args = ap.parse_args()
    build_features(args.bars, args.out)


if __name__ == "__main__":
    main()
