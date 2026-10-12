"""프로젝트 공용 설정. 모든 파이프라인이 이 파일을 import 한다."""
from pathlib import Path

# --- 경로 ---
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
BARS_PATH = DATA_DIR / "bars_10m.parquet"        # [1] 출력
FEATURES_PATH = DATA_DIR / "features.parquet"     # [2] 출력
LABELED_PATH = DATA_DIR / "labeled.parquet"       # [3] 출력
TRAIN_JSONL = DATA_DIR / "train.jsonl"            # [4] 출력
EVAL_JSONL = DATA_DIR / "eval.jsonl"              # [4] 출력
MODEL_DIR = ROOT / "stock-model"                  # [5] 출력 (laya.load 경로)

for d in (DATA_DIR,):
    d.mkdir(parents=True, exist_ok=True)

# --- MongoDB ---
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "stock2020"
MONGO_HOST = r"C:\Program Files\MongoDB\Server\3.4\bin\mongo.exe"  # pymongo 불가 시 fallback

# --- 봉 집계 ---
BAR_MINUTES = 10                       # 10분봉
TRADING_START = "09:00"                # 장 시작
TRADING_END = "15:30"                  # 장 종료

# --- 라벨 ---
LABEL_HORIZON_BARS = 3                 # 미래 3봉(=30분) 수익률로 라벨
BUY_THRESHOLD = 0.005                  # +0.5% 이상 → buy
SELL_THRESHOLD = -0.005                # -0.5% 이하면 → sell
# 그 외 → hold

# --- 시간순 분할 ---
TRAIN_RATIO = 0.8                      # 앞 80% 시점 train / 뒤 20% eval

# --- Laya 학습 ---
BASE_CHECKPOINT = "convaiinnovations/laya"   # 영어 기본. 한국어 state 텍스트 포함 시 laya-multilingual
QUESTION_ID = "action"
INSTRUCTIONS = "Should we buy, hold, or sell this stock for the next 30 minutes?"
CRITERIA = {
    "buy": "expected return above +0.5% in the next 3 bars",
    "hold": "return between -0.5% and +0.5%",
    "sell": "expected return below -0.5% in the next 3 bars",
}
