"""Kaggle(GPU T4)용 Laya 파인튜닝 스크립트.

사용 예:
    python finetune_kaggle.py \
        --train /kaggle/input/laya-stock-data/train.jsonl \
        --eval  /kaggle/input/laya-stock-data/eval.jsonl \
        --output ./stock-model
"""
import argparse
import shutil
import sys
from pathlib import Path

# 노트북 셀 환경(Jupyter)에서는 __file__이 정의되지 않음
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# config.py가 있으면 프로젝트 기본값 사용, Kaggle 등 외부 환경에서는 fallback
try:
    if "__file__" in globals():
        sys.path.insert(0, str(_SCRIPT_DIR.parent))
    from config import BASE_CHECKPOINT, QUESTION_ID  # noqa: E402
except ImportError:
    BASE_CHECKPOINT = "convaiinnovations/laya"
    QUESTION_ID = "action"


def _default_data(name: str) -> str:
    """Kaggle 데이터셋 마운트를 실제 레이아웃과 무관하게 탐색."""
    candidates = [
        Path(f"/kaggle/input/laya-stock-data/{name}"),
        Path(f"/kaggle/input/datasets/romyoh/laya-stock-data/{name}"),
        Path(f"/kaggle/input/datasets/laya-stock-data/{name}"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    root = Path("/kaggle/input")
    if root.exists():
        hits = sorted(root.rglob(name))
        if hits:
            return str(hits[0])
    return str(_SCRIPT_DIR.parent / "data" / name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Kaggle T4에서 laya.train.finetune() 실행 스크립트"
    )
    p.add_argument("--train", default=_default_data("train.jsonl"),
                   help="학습 JSONL 경로 (기본: Kaggle 데이터셋 또는 로컬 data/)")
    p.add_argument("--eval", default=_default_data("eval.jsonl"),
                   help="평가 JSONL 경로 (기본: Kaggle 데이터셋 또는 로컬 data/)")
    p.add_argument("--output", default="./stock-model", help="모델 출력 디렉터리")
    p.add_argument("--model", default=BASE_CHECKPOINT,
                   help=f"기본 체크포인트: {BASE_CHECKPOINT}")
    p.add_argument("--epochs", type=int, default=4, help="학습 에폭 (기본 4)")
    p.add_argument("--device", default="cuda", help="학습 디바이스 (기본 cuda)")
    p.add_argument("--micro-batch", type=int, default=8, help="마이크로 배치 크기 (기본 8)")
    p.add_argument("--grad-accum", type=int, default=8, help="그래디언트 누적 스텝 (기본 8)")
    p.add_argument("--freeze-encoder", action="store_true", help="인코더 가중치 고정")
    # Jupyter 커널로 실행되면 sys.argv에 커널 json 경로가 들어 있어 argparse가 죽음
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[알림] 무시된 인자: {unknown}")
    return args


def main() -> int:
    args = parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            print(
                "Kaggle에서 GPU 런타임 활성화 필요: "
                "Runtime > Change runtime type > T4",
                file=sys.stderr,
            )
            return 1

    # Kaggle 등 laya 미설치 환경 대비 (이미 설치돼 있으면 즉시 통과)
    try:
        import laya  # noqa: F401
    except ImportError:
        import subprocess

        print("laya 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "laya"])

    from laya.train import TrainConfig, finetune
    import laya.train as _laya_train

    # laya 0.4.0 버그 workaround:
    # finetune()은 eval_data가 있으면 모델이 아직 CPU 상태(load_checkpoint 직후)에서
    # calibration_records(_forward)를 먼저 호출하고, model.to(device)는 train_model 안에서
    # 이후에 발생 → GPU 학습 시 "Expected all tensors to be on the same device" 크래시.
    # _forward 호출 시점에 모델을 해당 디바이스로 옮기도록 패치한다 (이미 같은 디바이스면 no-op).
    _orig_forward = _laya_train._forward

    def _forward_dev(model, batch, device, *a, **k):
        model.to(device)
        return _orig_forward(model, batch, device, *a, **k)

    _laya_train._forward = _forward_dev

    train_path = Path(args.train)
    eval_path = Path(args.eval)
    for path, name in ((train_path, "--train"), (eval_path, "--eval")):
        if not path.exists():
            print(f"[오류] {name} 파일을 찾을 수 없습니다: {path}", file=sys.stderr)
            return 1

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        eval_data=str(eval_path),
        freeze_encoder=args.freeze_encoder,
        question_id=QUESTION_ID,
    )

    print(f"모델: {args.model}")
    print(f"학습: {train_path} / 평가: {eval_path}")
    print(f"epochs={args.epochs}, micro_batch={args.micro_batch}, "
          f"grad_accum={args.grad_accum}, freeze_encoder={args.freeze_encoder}")

    try:
        summary = finetune(
            data=str(train_path),
            model_dir=args.model,
            output_dir=args.output,
            config=config,
            device=args.device,
        )
        print("=== 학습 완료 요약 ===")
        print(summary)

        # 출력 디렉터리를 zip으로 묶어 다운로드 용이하게
        out_dir = Path(args.output).resolve()
        if out_dir.is_dir():
            zip_base = shutil.make_archive(
                str(out_dir), "zip", root_dir=out_dir.parent, base_dir=out_dir.name
            )
            print(f"ZIP 생성: {zip_base}")
        else:
            print(f"[경고] 출력 디렉터리가 없습니다: {out_dir}", file=sys.stderr)
            return 1
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("GPU 메모리 초기화 완료 (torch.cuda.empty_cache())")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    _code = main()
    if _code:
        sys.exit(_code)
