from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from .core.config import get_settings
from .core.pipeline import TranslationPipeline


console = Console()


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def cmd_translate(args: argparse.Namespace) -> None:
    settings = get_settings()

    if args.model:
        settings.translation_model = args.model
    if args.chunk_size:
        settings.chunk_size = args.chunk_size

    pipeline = TranslationPipeline(settings=settings)

    if args.glossary:
        console.print(f"[blue]加载术语表: {args.glossary}[/blue]")
        count = pipeline.load_glossary(args.glossary)
        console.print(f"[green]已加载 {count} 条术语[/green]")

    input_path = Path(args.input)
    files: list[Path] = []

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir() if f.suffix.lower() in {".pdf", ".docx", ".doc", ".txt"}
        )
    else:
        console.print(f"[red]路径不存在: {input_path}[/red]")
        sys.exit(1)

    if not files:
        console.print("[red]未找到可翻译的文件[/red]")
        sys.exit(1)

    console.print(f"[bold]共 {len(files)} 个文件待翻译[/bold]\n")

    for file_idx, file_path in enumerate(files):
        console.print(f"\n[bold cyan]{'=' * 50}[/bold cyan]")
        console.print(f"[bold]文件 [{file_idx + 1}/{len(files)}]: {file_path.name}[/bold]")

        output = Path(args.output) if args.output else None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = None

            def on_progress(current: int, total: int, _chunk):
                nonlocal task
                if task is None:
                    task = progress.add_task("翻译中...", total=total)
                progress.update(task, completed=current)

            result = pipeline.translate_file(
                file_path,
                output_path=output,
                auto_extract_terms=not args.no_extract_terms,
                progress_callback=on_progress,
            )

        console.print("\n[green]✓ 翻译完成[/green]")
        console.print(f"  段落数: {result.chunk_count}")
        console.print(f"  耗时: {result.elapsed_seconds:.1f}s")
        console.print(f"  译文长度: {len(result.translated_text)} 字符")

        report = result.quality_report
        if report.issues:
            console.print(f"  [yellow]⚠ 发现 {len(report.issues)} 个问题:[/yellow]")
            for issue in report.issues:
                console.print(f"    - {issue}")
        else:
            console.print("  [green]✓ 无质量问题[/green]")


def cmd_build_glossary(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = TranslationPipeline(settings=settings)

    glossary_path = Path(args.input)
    if not glossary_path.exists():
        console.print(f"[red]术语表文件不存在: {glossary_path}[/red]")
        sys.exit(1)

    count = pipeline.load_glossary(glossary_path)
    console.print(f"[green]术语知识库构建完成: {count} 条术语[/green]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ru-translate",
        description="RAG + LLM 俄语学术文献翻译系统",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_translate = subparsers.add_parser("translate", help="翻译文档")
    p_translate.add_argument("input", help="输入文件或目录路径")
    p_translate.add_argument("-o", "--output", help="输出文件路径")
    p_translate.add_argument("-g", "--glossary", help="术语表 JSON 文件路径")
    p_translate.add_argument("-m", "--model", help="翻译模型名称")
    p_translate.add_argument("--chunk-size", type=int, help="分段大小 (tokens)")
    p_translate.add_argument("--no-extract-terms", action="store_true", help="禁用自动术语提取")
    p_translate.set_defaults(func=cmd_translate)

    p_glossary = subparsers.add_parser("build-glossary", help="构建术语知识库")
    p_glossary.add_argument("input", help="术语表 JSON 文件路径")
    p_glossary.set_defaults(func=cmd_build_glossary)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
