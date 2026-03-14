import json
import os
import warnings
from pathlib import Path
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS
from config import TERMBASE_FILE, MEMORY_FILE, VECTOR_DIR

embedding = None
VECTOR_BASE = Path(VECTOR_DIR)
TERM_INDEX_DIR = VECTOR_BASE / "terms_faiss"
MEMORY_INDEX_DIR = VECTOR_BASE / "memory_faiss"
META_FILE = VECTOR_BASE / "index_meta.json"


def get_embedding():
    global embedding

    if embedding is not None:
        return embedding

    try:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError:
                from langchain.embeddings import HuggingFaceEmbeddings

        embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        return embedding
    except Exception as exc:
        warnings.warn(
            f"Embedding 初始化失败，RAG 将自动降级关闭: {exc}",
            RuntimeWarning,
        )
        return None


def _load_meta():
    if not META_FILE.exists():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf8"))
    except Exception:
        return {}


def _save_meta(meta):
    VECTOR_BASE.mkdir(parents=True, exist_ok=True)
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")


def _file_signature(file_path):
    path = Path(file_path)
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_store(index_dir):
    emb = get_embedding()
    if emb is None:
        return None
    if not Path(index_dir).exists():
        return None
    try:
        return FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)
    except TypeError:
        return FAISS.load_local(str(index_dir), emb)
    except Exception as exc:
        warnings.warn(f"加载向量库失败，将自动重建: {exc}", RuntimeWarning)
        return None


def _save_store(store, index_dir):
    if store is None:
        return
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))


def load_termbase():

    if not os.path.exists(TERMBASE_FILE):
        return []

    with open(TERMBASE_FILE, "r", encoding="utf8") as f:
        data = json.load(f)

    return [f"{i['ru']} -> {i['zh']}" for i in data]


def load_memory():

    if not os.path.exists(MEMORY_FILE):
        return []

    docs = []

    with open(MEMORY_FILE, "r", encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            docs.append(item["ru"] + "\n" + item["zh"])

    return docs


def build_store(docs):

    if not docs:
        return None

    emb = get_embedding()
    if emb is None:
        return None

    return FAISS.from_texts(docs, emb)


def build_term_store():
    meta = _load_meta()
    source_sig = _file_signature(TERMBASE_FILE)
    if meta.get("termbase_sig") == source_sig:
        cached = _load_store(TERM_INDEX_DIR)
        if cached is not None:
            return cached

    store = build_store(load_termbase())
    _save_store(store, TERM_INDEX_DIR)
    meta["termbase_sig"] = source_sig
    _save_meta(meta)
    return store


def build_memory_store():
    meta = _load_meta()
    source_sig = _file_signature(MEMORY_FILE)
    if meta.get("memory_sig") == source_sig:
        cached = _load_store(MEMORY_INDEX_DIR)
        if cached is not None:
            return cached

    store = build_store(load_memory())
    _save_store(store, MEMORY_INDEX_DIR)
    meta["memory_sig"] = source_sig
    _save_meta(meta)
    return store


def build_retrievers():

    return {
        "terms": build_term_store(),
        "memory": build_memory_store(),
    }


def add_memory(memory_store, ru, zh):

    text = f"{ru}\n{zh}"

    if memory_store is None:
        memory_store = build_store([text])
    else:
        memory_store.add_texts([text])

    _save_store(memory_store, MEMORY_INDEX_DIR)
    meta = _load_meta()
    meta["memory_sig"] = _file_signature(MEMORY_FILE)
    _save_meta(meta)
    return memory_store


def build_vector_store():

    docs = load_termbase() + load_memory()
    return build_store(docs)


def search(store, query, k=5):

    if store is None:
        return []

    docs = store.similarity_search(query, k=k)

    return [d.page_content for d in docs]
