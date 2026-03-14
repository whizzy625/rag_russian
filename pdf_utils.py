from pathlib import Path
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_pdf_text(path):

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf8")

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "缺少 PyMuPDF 依赖。请先执行: pip install -r requirements.txt"
        ) from exc

    doc = fitz.open(path)

    text = []
    for page in doc:
        text.append(page.get_text())

    return "\n".join(text)


def split_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_text(text)
