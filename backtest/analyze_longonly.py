"""롱온리 탈피 분석: pred 클래스별/신뢰도별 미래수익률(fwd_ret) 분포.

핵심 질문: sell 신호가 '절대적 하락'(fwd_ret<0)을 예측하는가? → 숏/인버스 수익화 가능성.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\0GitHub\라야")
df = pd.read_parquet(ROOT / "data" / "eval_pred.parquet")
df = df.dropna(subset=["fwd_ret"]).copy()

def summ(g):
    return pd.Series({
        "n": len(g),
        "mean_ret_%": round(g["fwd_ret"].mean() * 100, 4),
        "median_ret_%": round(g["fwd_ret"].median() * 100, 4),
        "p(ret>0)": round((g["fwd_ret"] > 0).mean(), 4),
        "p(ret<0)": round((g["fwd_ret"] < 0).mean(), 4),
        "mean_ret(하위10%)_%": round(g["fwd_ret"].quantile(0.10) * 100, 4),
    })

print("=== [1] pred 클래스별 fwd_ret (30분 미래수익) ===")
print(df.groupby("pred").apply(summ).to_string())
print()
print("=== [2] label(정답) 클래스별 fwd_ret (참고) ===")
print(df.groupby("label").apply(summ).to_string())
print()

print("=== [3] sell 예측의 신뢰도 구간별 fwd_ret ===")
edges = [0.0, 0.38, 0.42, 0.45, 1.01]
lbl = ["<0.38", "0.38-0.42", "0.42-0.45", ">=0.45"]
sell = df[df["pred"] == "sell"].copy()
sell["cb"] = pd.cut(sell["pred_conf"], edges, labels=lbl, right=False)
print(sell.groupby("cb", observed=True).apply(summ).to_string())
print()

print("=== [4] buy 예측의 신뢰도 구간별 fwd_ret ===")
buy = df[df["pred"] == "buy"].copy()
buy["cb"] = pd.cut(buy["pred_conf"], edges, labels=lbl, right=False)
print(buy.groupby("cb", observed=True).apply(summ).to_string())
print()

# 전체 상관: pred_conf가 방향성과 관련?
print("=== [5] 전체: pred_conf vs |fwd_ret| 상관 ===")
print("corr(pred_conf, fwd_ret):", round(df["pred_conf"].corr(df["fwd_ret"]), 4))
print("corr(pred_conf, abs(fwd_ret)):", round(df["pred_conf"].corr(df["fwd_ret"].abs()), 4))
