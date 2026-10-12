"""단계 [3]: 피처 → 미래수익률 라벨 (hard + soft gold).

읽기: data/features.parquet
쓰기: data/labeled.parquet  (원본/피처 유지 + fwd_ret, label, buy/hold/sell_prob)

라벨 규칙 (config 상수 사용):
  fwd_ret  = close.shift(-LABEL_HORIZON_BARS) / close - 1   (종목별, bar_time 정렬)
  마지막 LABEL_HORIZON_BARS 봉은 미래가 없으므로 label=NaN → drop.
    (미래를 몰라서 라벨을 거부하는 것 — lookahead 누수 아님)
  hard: fwd_ret >= BUY_THRESHOLD(0.005)  → "buy"
        fwd_ret <= SELL_THRESHOLD(-0.005) → "sell"
        그 외                            → "hold"

soft gold 확률 (RLCD용, 선형 멤버십 — 정확히 이 공식):
    t   = clip(fwd_ret / BUY_THRESHOLD, -1, 1)
    buy = max(0,  t) * 0.5 + 0.125
    sell= max(0, -t) * 0.5 + 0.125
    hold= 1 - buy - sell
    정규화: 세 값을 합으로 나눠 sum=1 보장 (합은 원래 1.0, 방어적 실행)
  참고 t=0 → (0.125, 0.75, 0.125), t=1 → (0.625, 0.25, 0.125),
       t=-1 → (0.125, 0.25, 0.625). 세 값 항상 >= 0.125/0.25 이상 → 음수 없음.

실행: python pipeline/build_labels.py [--features PATH] [--out PATH]
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

PROB_COLUMNS = ["buy_prob", "hold_prob", "sell_prob"]


def soft_probs(fwd_ret: pd.Series) -> pd.DataFrame:
    """fwd_ret → {buy,hold,sell} 선형 soft 확률 (docstring 상단 공식)."""
    t = (fwd_ret / config.BUY_THRESHOLD).clip(-1.0, 1.0)
    buy = t.clip(lower=0.0) * 0.5 + 0.125
    sell = (-t).clip(lower=0.0) * 0.5 + 0.125
    hold = 1.0 - buy - sell
    buy, hold, sell = buy.clip(lower=0.0), hold.clip(lower=0.0), sell.clip(lower=0.0)
    total = buy + hold + sell
    return pd.DataFrame({
        "buy_prob": buy / total,
        "hold_prob": hold / total,
        "sell_prob": sell / total,
    }, index=fwd_ret.index)


def build_labels(features_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(features_path)
    if "close" not in df.columns or "code" not in df.columns:
        raise SystemExit("features parquet must contain 'close' and 'code'")

    n_in = len(df)

    # 종목별 미래 수익률 (shift(-H): 미래를 못 본다 = 마지막 H봉 NaN)
    df["fwd_ret"] = (
        df.groupby("code", sort=False)["close"]
        .transform(lambda s: s.shift(-config.LABEL_HORIZON_BARS) / s - 1.0)
    )

    # 미래 봉이 부족한 마지막 H봉/종목 → 라벨 불가 → 제거
    n_tail_dropped = int(df["fwd_ret"].isna().sum())
    df = df[df["fwd_ret"].notna()].copy()

    # hard label
    df["label"] = np.where(
        df["fwd_ret"] >= config.BUY_THRESHOLD, "buy",
        np.where(df["fwd_ret"] <= config.SELL_THRESHOLD, "sell", "hold"),
    )

    # soft gold 확률
    df = pd.concat([df, soft_probs(df["fwd_ret"])], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # --- 요약 ---
    print("=== build_labels ===")
    print(f"in        : {features_path} ({n_in:,} rows)")
    print(f"horizon   : {config.LABEL_HORIZON_BARS} bars  "
          f"buy>={config.BUY_THRESHOLD}  sell<={config.SELL_THRESHOLD}")
    print(f"dropped   : {n_tail_dropped:,} tail rows (future unavailable, "
          f"{config.LABEL_HORIZON_BARS} per stock)")
    print(f"out rows  : {len(df):,}")
    dist = df["label"].value_counts()
    for lab in ("buy", "hold", "sell"):
        c = int(dist.get(lab, 0))
        print(f"label {lab:<5}: {c:>7,}  ({c / len(df) * 100:5.1f}%)")
    probs = df[PROB_COLUMNS]
    print(f"prob check: min={probs.to_numpy().min():.6f}  "
          f"max={probs.to_numpy().max():.6f}  "
          f"max|sum-1|={np.abs(probs.sum(axis=1) - 1.0).max():.2e}")
    print(f"fwd_ret   : min={df['fwd_ret'].min():+.6f}  max={df['fwd_ret'].max():+.6f}")
    print(f"out       : {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="features.parquet → labeled.parquet")
    ap.add_argument("--features", type=Path, default=config.FEATURES_PATH, help="input features parquet")
    ap.add_argument("--out", type=Path, default=config.LABELED_PATH, help="output labeled parquet")
    args = ap.parse_args()
    build_labels(args.features, args.out)


if __name__ == "__main__":
    main()
