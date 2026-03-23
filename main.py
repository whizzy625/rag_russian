import argparse
from pathlib import Path

from config import DEFAULT_INPUT_DIR, OUTPUT_DIR
from translation_service import translate_path


def parse_args():
    parser = argparse.ArgumentParser(description="俄语文档翻译（Qwen + RAG）")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR),
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
    translate_path(
        input_path=Path(args.input),
        output_path=Path(args.output),
        force=args.force,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc)
