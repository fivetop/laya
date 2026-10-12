# Kaggle T4 파인튜닝 가이드 (Laya 주식 모델)

Kaggle 노트북(GPU T4)에서 `laya`를 사용해 주식 트레이딩 모델을 파인튜닝하고
결과물을 ZIP으로 내려받는 전체 절차입니다.

## 1. 데이터셋 업로드

1. `data/train.jsonl`, `data/eval.jsonl` 준비 (파이프라인 [4] make_jsonl.py 출력).
2. Kaggle 우측 상단 **Code → New Dataset** (또는 www.kaggle.com → Datasets → New Dataset).
3. 파일 두 개를 드래그해 업로드. 데이터셋 이름은 예: `laya-stock-data`.
4. 업로드 후 데이터셋 경로가 다음과 같은지 확인:
   ```
   /kaggle/input/laya-stock-data/train.jsonl
   /kaggle/input/laya-stock-data/eval.jsonl
   ```

## 2. 노트북 생성 + GPU 활성화

1. Kaggle에서 **Code → New Notebook** 생성.
2. 우측 상단 **Session options (Settings) → Accelerator → GPU T4 x2 (단일 T4면 GPU T4 x1)** 선택.
   - 반드시 **GPU**가 선택되어야 합니다 (CPU로는 학습 불가).
3. 저장된 데이터셋을 노트북에 첨부: 오른쪽 패널 **Input → + Add Input → search `laya-stock-data`**.

## 3. 노트버 셀 구성

### 셀 1 — laya 설치

```python
!pip install -q laya
```

### 셀 2 — 스크립트 붙여넣기

`train/finetune_kaggle.py` 전체 내용을 새 셀에 붙여넣어
**`/kaggle/working/finetune_kaggle.py`** 로 저장합니다.

```python
%%writefile /kaggle/working/finetune_kaggle.py
# (finetune_kaggle.py 내용 전체를 여기에 붙여넣기)
```

> GitHub에서 받은 경우 `!curl -o /kaggle/working/finetune_kaggle.py <raw URL>` 로 대체 가능.
> 단, `config.py`가 없는 Kaggle 환경이므로 스크립트 내 `BASE_CHECKPOINT` 기본값만 사용합니다.

### 셀 3 — 파인튜닝 실행

```python
!python /kaggle/working/finetune_kaggle.py \
    --train /kaggle/input/laya-stock-data/train.jsonl \
    --eval  /kaggle/input/laya-stock-data/eval.jsonl \
    --output ./stock-model
```

- 기본값: `--model convaiinnovations/laya`, `--epochs 4`,
  `--micro-batch 8`, `--grad-accum 8`, `--device cuda`.
- **예상 소요 시간: 단일 T4 기준 약 1~2시간** (데이터 크기·epochs 4 기준).
- 학습이 끝나면 `=== 학습 완료 요약 ===` dict 출력과
  `ZIP 생성: /kaggle/working/stock-model.zip` 메시지가 표시됩니다.

## 4. ZIP 다운로드 및 로컬 압축 해제

1. 노트북 우측 패널 **Output** 탭 → `/kaggle/working/stock-model.zip` 확인.
2. 다운로드 (체크박스 선택 → Download).
3. 로컬에서 압축 해제:
   - Windows (PowerShell):
     ```powershell
     Expand-Archive -Path stock-model.zip -DestinationPath C:\0GitHub\라야\
     ```
   - 결과물: `C:\0GitHub\라야\stock-model\` (laya.load / serve 경로와 일치).

## 5. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `CUDA out of memory` / OOM | `--micro-batch 4` 로 낮춤 (grad-accum은 8 유지 또는 16으로 상향). 예: `--micro-batch 4 --grad-accum 16` |
| 학습 시작 후 멈춤(hang) | 셀 맨 앞에 `import os; os.environ["USE_TF"] = "0"` 설정 후 재실행 |
| `Kaggle에서 GPU 런타임 활성화 필요` 오류 | Runtime > Change runtime type > **T4 GPU** 선택 후 커널 재시작 |
| 한국어 state 텍스트 사용 시 | 한국어 문장이 state에 포함되면 `--model convaiinnovations/laya-multilingual` 지정. **본 프로젝트 state는 숫자+코드(티커/지표)로 구성되므로 기본 영어 체크포인트(`convaiinnovations/laya`) 사용 가능** |
