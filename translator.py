import json
from pathlib import Path
from prompts import translation_prompt
from llm_client import llm_chat
from rag_store import add_memory, search
from config import MEMORY_FILE, TOP_K, USE_RAG


def save_memory(ru, zh):

    item = {"ru": ru, "zh": zh}

    with open(MEMORY_FILE, "a", encoding="utf8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_checkpoint(checkpoint_path, total):
    path = Path(checkpoint_path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf8"))
        if data.get("total_chunks") != total:
            return []
        translated = data.get("translated_chunks", [])
        if not isinstance(translated, list):
            return []
        return translated
    except Exception:
        return []


def save_checkpoint(checkpoint_path, translated_chunks, total):
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_chunks": total,
        "translated_chunks": translated_chunks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf8")


def _is_data_inspection_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    signals = [
        "data_inspection_failed",
        "datainspectionfailed",
        "inappropriate-content",
        "inappropriate content",
        "input data may contain inappropriate content",
        "error-code#inappropriate-content",
    ]
    return any(flag in msg for flag in signals)


def _safe_fallback_translate(text: str) -> str:
    prompt = (
        "请将以下俄文翻译成中文。\n"
        "要求：\n"
        "1 只输出译文，不要解释\n"
        "2 使用中性、学术、克制表达\n"
        "3 不扩写，不添加原文没有的信息\n\n"
        f"原文：\n{text}"
    )
    return llm_chat(prompt, temperature=0.1)


def translate_chunks(chunks, stores, file_label="", checkpoint_path=None, progress_callback=None):

    term_store = stores.get("terms") if stores else None
    memory_store = stores.get("memory") if stores else None

    total = len(chunks)
    result = load_checkpoint(checkpoint_path, total) if checkpoint_path else []
    start = len(result)

    if start > 0:
        if file_label:
            print(f"[{file_label}] resume from chunk {start + 1}/{total}")
        else:
            print(f"resume from chunk {start + 1}/{total}")

    for i, chunk in enumerate(chunks[start:], start=start + 1):

        if USE_RAG:
            terms = search(term_store, chunk, TOP_K)
            memory = search(memory_store, chunk, min(3, TOP_K))
        else:
            terms = []
            memory = []

        context = "\n".join(result[-2:])

        prompt = translation_prompt.format(
            terms="\n".join(terms),
            memory="\n".join(memory),
            context=context,
            text=chunk,
        )

        allow_memory = True
        try:
            zh = llm_chat(prompt)
        except Exception as exc:
            if not _is_data_inspection_error(exc):
                raise

            if file_label:
                print(f"[{file_label}] chunk {i}/{total} 触发内容审查，使用保守模式重试...")
            else:
                print(f"chunk {i}/{total} 触发内容审查，使用保守模式重试...")

            try:
                zh = _safe_fallback_translate(chunk)
            except Exception as retry_exc:
                if not _is_data_inspection_error(retry_exc):
                    raise
                zh = "【该段触发平台内容审查，已跳过自动翻译，请人工处理】"
                allow_memory = False
                if file_label:
                    print(f"[{file_label}] chunk {i}/{total} 再次被拦截，已跳过并继续后续段落。")
                else:
                    print(f"chunk {i}/{total} 再次被拦截，已跳过并继续后续段落。")

        if allow_memory:
            save_memory(chunk, zh)
            memory_store = add_memory(memory_store, chunk, zh)

        result.append(zh)
        if checkpoint_path:
            save_checkpoint(checkpoint_path, result, total)
        if progress_callback:
            progress_callback(
                {
                    "file": file_label or "",
                    "done_chunks": i,
                    "total_chunks": total,
                }
            )

        if file_label:
            print(f"[{file_label}] translated chunk {i}/{total}")
        else:
            print(f"translated chunk {i}/{total}")

    return "\n".join(result)
