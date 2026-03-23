import hashlib
from pathlib import Path

from config import OUTPUT_DIR


def _resolve_files(input_path: Path, include_files=None):
    if not input_path.exists():
        raise FileNotFoundError(
            f"输入路径不存在: {input_path}\n请检查路径，或运行: python main.py --input 你的文件路径"
        )
    if input_path.is_file():
        return [input_path]

    if include_files:
        selected = []
        missing = []
        for name in include_files:
            p = input_path / name
            if p.exists() and p.is_file() and p.suffix.lower() in {".pdf", ".txt"}:
                selected.append(p)
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(f"以下文件不存在或不可翻译: {', '.join(missing)}")
        if not selected:
            raise FileNotFoundError("未匹配到可翻译文件")
        return selected

    files = sorted(
        f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}
    )
    if not files:
        raise FileNotFoundError(f"目录中未找到可翻译文件: {input_path}")
    return files


def translate_path(
    input_path,
    output_path=None,
    force=False,
    progress_callback=None,
    include_files=None,
):
    try:
        from pdf_utils import extract_pdf_text, split_text
        from pdf_writer import save_pdf
        from rag_store import build_retrievers
        from translator import translate_chunks
    except ImportError as exc:
        raise RuntimeError("运行依赖缺失，请先执行: pip install -r requirements.txt") from exc

    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else OUTPUT_DIR
    files = _resolve_files(input_path, include_files=include_files)
    batch_mode = len(files) > 1 or input_path.is_dir()

    if batch_mode:
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_path.parent if output_path.suffix.lower() == ".pdf" else output_path
        output_dir.mkdir(parents=True, exist_ok=True)

    def emit(event, **payload):
        if progress_callback:
            progress_callback({"event": event, **payload})

    summary = []
    emit("task_start", total_files=len(files))
    for idx, file_path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] reading input: {file_path}")
        emit("file_start", file=file_path.name, file_index=idx, total_files=len(files))

        try:
            if not batch_mode and output_path.suffix.lower() == ".pdf":
                final_output = output_path
            else:
                final_output = output_dir / f"{file_path.stem}_translated.pdf"

            file_hash = hashlib.md5(str(file_path.resolve()).encode("utf8")).hexdigest()[:10]
            checkpoint_dir = output_dir / ".checkpoints"
            checkpoint_path = checkpoint_dir / f"{file_path.stem}_{file_hash}.json"

            if final_output.exists() and not force:
                print(f"skip (already translated): {final_output.name}")
                summary.append({"file": file_path.name, "status": "already_translated"})
                emit(
                    "file_already_translated",
                    file=file_path.name,
                    file_index=idx,
                    total_files=len(files),
                )
                continue

            if force and checkpoint_path.exists():
                checkpoint_path.unlink()

            text = extract_pdf_text(file_path)
            print("splitting")
            chunks = split_text(text)
            emit(
                "file_chunk_ready",
                file=file_path.name,
                file_index=idx,
                total_files=len(files),
                total_chunks=len(chunks),
            )

            print("building rag")
            stores = build_retrievers()

            print(f"translating: {file_path.name}")
            zh = translate_chunks(
                chunks,
                stores,
                file_label=file_path.name,
                checkpoint_path=checkpoint_path,
                progress_callback=lambda p: emit(
                    "chunk_progress",
                    file=file_path.name,
                    file_index=idx,
                    total_files=len(files),
                    done_chunks=p.get("done_chunks", 0),
                    total_chunks=p.get("total_chunks", len(chunks)),
                ),
            )

            save_pdf(zh, final_output)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            print("done", final_output)
            summary.append(
                {"file": file_path.name, "status": "translated", "output": str(final_output)}
            )
            emit(
                "file_done",
                file=file_path.name,
                file_index=idx,
                total_files=len(files),
                output=str(final_output),
            )
        except Exception as exc:
            print(f"[{file_path.name}] failed: {exc}")
            print("continue with next file...")
            summary.append({"file": file_path.name, "status": "failed", "error": str(exc)})
            emit(
                "file_failed",
                file=file_path.name,
                file_index=idx,
                total_files=len(files),
                error=str(exc),
            )
            continue

    emit("task_done", total_files=len(files))
    return summary
