import argparse
import hashlib
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="俄语文档翻译（Qwen + RAG）")
    parser.add_argument(
        "--input",
        default=str(Path.home() / "Documents" / "俄语"),
        help="输入文件或目录路径（支持 .pdf 或 .txt）",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="输出文件或目录路径",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重译（忽略已输出文件与断点）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from pdf_utils import extract_pdf_text, split_text
        from pdf_writer import save_pdf
        from rag_store import build_retrievers
        from translator import translate_chunks
    except ImportError as exc:
        raise RuntimeError(
            "运行依赖缺失，请先执行: pip install -r requirements.txt"
        ) from exc

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"输入路径不存在: {input_path}\n请检查路径，或运行: python main.py --input 你的文件路径"
        )

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in {".pdf", ".txt"}
        )
        if not files:
            raise FileNotFoundError(f"目录中未找到可翻译文件: {input_path}")

    batch_mode = len(files) > 1 or input_path.is_dir()

    if batch_mode:
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if output_path.suffix.lower() == ".pdf":
            output_dir = output_path.parent
        else:
            output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)

    for idx, file_path in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] reading input: {file_path}")

        if not batch_mode and output_path.suffix.lower() == ".pdf":
            final_output = output_path
        else:
            final_output = output_dir / f"{file_path.stem}_translated.pdf"

        file_hash = hashlib.md5(str(file_path.resolve()).encode("utf8")).hexdigest()[:10]
        checkpoint_dir = output_dir / ".checkpoints"
        checkpoint_path = checkpoint_dir / f"{file_path.stem}_{file_hash}.json"

        if final_output.exists() and not args.force:
            print(f"skip (already translated): {final_output.name}")
            continue

        if args.force and checkpoint_path.exists():
            checkpoint_path.unlink()

        text = extract_pdf_text(file_path)

        print("splitting")
        chunks = split_text(text)

        print("building rag")
        stores = build_retrievers()

        print(f"translating: {file_path.name}")
        zh = translate_chunks(
            chunks,
            stores,
            file_label=file_path.name,
            checkpoint_path=checkpoint_path,
        )

        save_pdf(zh, final_output)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        print("done", final_output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc)
