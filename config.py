import os
from pathlib import Path

BASE_DIR = Path(__file__).parent


def load_dotenv(path):
    env = {}

    if not path.exists():
        return env

    for raw_line in path.read_text(encoding="utf8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value

    return env


dotenv = load_dotenv(BASE_DIR / ".env")
if not dotenv:
    dotenv = load_dotenv(BASE_DIR / ".env.example")

# API
API_KEY = os.getenv("DASHSCOPE_API_KEY", dotenv.get("DASHSCOPE_API_KEY", ""))

MODEL_NAME = os.getenv("MODEL_NAME", dotenv.get("MODEL_NAME", "qwen-plus"))
_model_list_raw = os.getenv("MODEL_NAMES", dotenv.get("MODEL_NAMES", ""))
MODEL_NAMES = [m.strip() for m in _model_list_raw.split(",") if m.strip()]
if not MODEL_NAMES:
    MODEL_NAMES = [MODEL_NAME]

# 目录
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# RAG
TOP_K = 5
USE_RAG = True

# 向量库
VECTOR_DIR = BASE_DIR / "vector_store"
VECTOR_DIR.mkdir(exist_ok=True)

# 翻译记忆
MEMORY_FILE = BASE_DIR / "translation_memory.jsonl"

# 术语库
TERMBASE_FILE = BASE_DIR / "termbase.json"

# chunk
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
