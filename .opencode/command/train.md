---
description: Kaggle T4에서 laya 파인튜닝 (로컬 GPU 없음)
---

# 파인튜닝 (Kaggle)

이 노트북에는 CUDA가 없어 학습은 Kaggle(무료 T4)에서 합니다.

## 작업 순서

1. `train/finetune_kaggle.py` 생성:
   ```python
   from laya.train import TrainConfig, finetune
   finetune(
       data="train.jsonl", model_dir="convaiinnovations/laya",
       output_dir="stock-model",
       config=TrainConfig(epochs=4, eval_data="eval.jsonl",
                          micro_batch=8, grad_accum=8),  # T4 1장 기준
       device="cuda",
   )
   ```
   - 한국어 지표/텍스트가 state에 많으면 model_dir을 `convaiinnovations/laya-multilingual`로 변경 (config.BASE_CHECKPOINT 참조)
2. Kaggle 업로드 체크리스트 안내:
   - train.jsonl, eval.jsonl 업로드
   - `pip install laya` 셀
   - 출력(stock-model 디렉터리)을 zip 다운로드 → 로컬 `C:\0GitHub\라야\stock-model\`에 압축 풀기
3. 학습 완료 판단 기준: eval accuracy, ECE(보정오차)가 보고됨 — 기존 zero-shot 대비 개선 확인

## MUST NOT
- 로컬 CPU로 전체 학습 시도 금지 (수일 소요)
- eval_data 누락 금지 (시간순 held-out 필수)

Kaggle 노트북 셀 순서가 담긴 마크다운도 train/README_kaggle.md로 생성할 것.
