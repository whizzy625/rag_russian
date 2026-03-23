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


def _env(key, default=""):
    return os.getenv(key, dotenv.get(key, default))


def _path_env(key, default: Path) -> Path:
    raw = _env(key, str(default))
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = BASE_DIR / p
    return p

# API
API_KEY = _env("DASHSCOPE_API_KEY", "")

MODEL_NAME = _env("MODEL_NAME", "qwen-plus")
_model_list_raw = _env("MODEL_NAMES", "")
MODEL_NAMES = [m.strip() for m in _model_list_raw.split(",") if m.strip()]
if not MODEL_NAMES:
    MODEL_NAMES = [MODEL_NAME]

# 目录
DATA_DIR = _path_env("DATA_DIR", BASE_DIR / "data")
OUTPUT_DIR = _path_env("OUTPUT_DIR", BASE_DIR / "output")
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 系统文件目录（可配置，默认都在项目内）
LIBRARY_SOURCE_DIR = _path_env("LIBRARY_SOURCE_DIR", DATA_DIR / "my_library" / "source")
LIBRARY_OUTPUT_DIR = _path_env("LIBRARY_OUTPUT_DIR", OUTPUT_DIR / "my_library")
TASK_DB_PATH = _path_env("TASK_DB_PATH", DATA_DIR / "tasks.db")
DEFAULT_INPUT_DIR = _path_env("DEFAULT_INPUT_DIR", LIBRARY_SOURCE_DIR)
LIBRARY_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TASK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Web 服务
WEB_HOST = _env("WEB_HOST", "0.0.0.0")
WEB_PORT = int(_env("WEB_PORT", "8000"))
WEB_RELOAD = _env("WEB_RELOAD", "true").lower() in {"1", "true", "yes", "on"}

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
