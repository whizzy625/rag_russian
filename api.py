import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from config import LIBRARY_OUTPUT_DIR, LIBRARY_SOURCE_DIR, OUTPUT_DIR
from task_store import create_task as db_create_task
from task_store import get_task as db_get_task
from task_store import init_db
from task_store import list_tasks as db_list_tasks
from task_store import update_task as db_update_task
from translation_service import translate_path

app = FastAPI(title="俄语文献翻译系统", version="0.1.0")
LIBRARY_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
init_db()


class TranslateRequest(BaseModel):
    input: str = Field(..., description="输入文件或目录路径")
    output: Optional[str] = Field(default=str(OUTPUT_DIR), description="输出目录或输出文件路径")
    force: bool = Field(default=False, description="是否强制重译")
    include_files: List[str] = Field(default_factory=list, description="仅翻译这些文件名")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _safe_name(name: str) -> str:
    cleaned = Path(name).name.strip().replace("/", "_").replace("\\", "_")
    return cleaned or f"upload_{uuid.uuid4().hex[:8]}.pdf"


def _save_upload_to_library(upload: UploadFile) -> str:
    original = _safe_name(upload.filename or "")
    target = LIBRARY_SOURCE_DIR / original
    if target.exists():
        target = LIBRARY_SOURCE_DIR / f"{target.stem}_{uuid.uuid4().hex[:6]}{target.suffix}"
    data = upload.file.read()
    target.write_bytes(data)
    return target.name


def _update_task(task_id: str, **fields):
    fields["updated_at"] = _now()
    db_update_task(task_id, fields)


def _get_task_snapshot(task_id: str):
    return db_get_task(task_id)


def _compute_progress(task: dict) -> float:
    total_files = max(int(task.get("total_files", 0)), 1)
    finished_files = int(task.get("finished_files", 0))
    done_chunks = int(task.get("done_chunks", 0))
    total_chunks = max(int(task.get("total_chunks", 0)), 1)
    return round(min(100.0, ((finished_files + (done_chunks / total_chunks)) / total_files) * 100), 2)


def _run_async_task(task_id: str, req: TranslateRequest):
    try:
        _update_task(task_id, status="running")

        def on_progress(event: dict):
            kind = event.get("event", "")
            if kind == "task_start":
                _update_task(task_id, total_files=event.get("total_files", 0), message="任务已开始")
                return
            if kind == "file_start":
                _update_task(
                    task_id,
                    current_file=event.get("file", ""),
                    done_chunks=0,
                    total_chunks=1,
                    message=f"正在处理: {event.get('file', '')}",
                )
                task_snapshot = _get_task_snapshot(task_id)
                if task_snapshot:
                    _update_task(task_id, progress=_compute_progress(task_snapshot))
                return
            if kind == "file_chunk_ready":
                _update_task(
                    task_id,
                    total_chunks=max(int(event.get("total_chunks", 1)), 1),
                    done_chunks=0,
                )
                task_snapshot = _get_task_snapshot(task_id)
                if task_snapshot:
                    _update_task(task_id, progress=_compute_progress(task_snapshot))
                return
            if kind == "chunk_progress":
                _update_task(
                    task_id,
                    done_chunks=int(event.get("done_chunks", 0)),
                    total_chunks=max(int(event.get("total_chunks", 1)), 1),
                )
                task_snapshot = _get_task_snapshot(task_id)
                if task_snapshot:
                    _update_task(task_id, progress=_compute_progress(task_snapshot))
                return
            if kind in {"file_done", "file_failed", "file_already_translated"}:
                task_snapshot = _get_task_snapshot(task_id) or {}
                finished = int(task_snapshot.get("finished_files", 0)) + 1
                _update_task(task_id, finished_files=finished, done_chunks=0, total_chunks=1)
                task_snapshot = _get_task_snapshot(task_id)
                if task_snapshot:
                    _update_task(task_id, progress=_compute_progress(task_snapshot))

        summary = translate_path(
            input_path=Path(req.input),
            output_path=Path(req.output) if req.output else OUTPUT_DIR,
            force=req.force,
            progress_callback=on_progress,
            include_files=req.include_files,
        )
        _update_task(task_id, status="success", progress=100.0, summary=summary, message="任务完成")
    except Exception as exc:
        _update_task(task_id, status="failed", error=str(exc), message="任务失败")


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>俄语文献翻译系统</title>
        <style>
          :root {{
            --bg: #f6f7f9;
            --card: #ffffff;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #e5e7eb;
            --brand: #0f766e;
            --brand-soft: #ccfbf1;
          }}
          body {{
            margin: 0;
            background: linear-gradient(135deg, #f8fafc, #eef2ff);
            color: var(--text);
            font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", sans-serif;
          }}
          .wrap {{
            max-width: 900px;
            margin: 30px auto;
            padding: 0 16px;
          }}
          .card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 8px 24px rgba(2, 6, 23, 0.06);
            margin-bottom: 16px;
          }}
          h1 {{
            margin: 0 0 12px;
            font-size: 24px;
          }}
          .row {{
            display: grid;
            grid-template-columns: 130px 1fr;
            gap: 10px;
            align-items: center;
            margin: 10px 0;
          }}
          input[type=text] {{
            width: 100%;
            box-sizing: border-box;
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 10px;
            font-size: 14px;
          }}
          .hint {{
            font-size: 13px;
            color: var(--muted);
            margin-top: 8px;
          }}
          button {{
            border: 0;
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 14px;
            cursor: pointer;
            background: var(--brand);
            color: white;
          }}
          button.secondary {{
            background: #374151;
          }}
          .actions {{
            display: flex;
            gap: 10px;
            margin-top: 12px;
          }}
          .progress {{
            width: 100%;
            height: 12px;
            background: #e5e7eb;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 10px;
          }}
          .bar {{
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #14b8a6, #0f766e);
            transition: width 0.3s;
          }}
          .status {{
            margin-top: 10px;
            padding: 10px;
            background: var(--brand-soft);
            border-radius: 10px;
            font-size: 14px;
          }}
          .task {{
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
            margin-top: 8px;
          }}
          .mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px;
          }}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="card">
            <h1>俄语文献翻译系统</h1>
            <div class="row">
              <div>上传俄语书</div>
              <input id="uploadFiles" type="file" multiple accept=".pdf,.txt">
            </div>
            <div class="actions">
              <button onclick="uploadAndStart()">上传并开始翻译</button>
            </div>
            <div class="hint">支持一次上传多本，会按你上传顺序依次翻译，文件会保存到“我的资料”。</div>
            <div class="row">
              <div>强制重译</div>
              <label><input id="forceFlag" type="checkbox"> 重新翻译已输出文件</label>
            </div>
            <div class="actions">
              <button class="secondary" onclick="refreshTasks()">刷新任务</button>
              <button class="secondary" onclick="loadMyFiles()">刷新我的资料</button>
            </div>
            <div class="hint">你可以做别的事，页面会自动显示进度。</div>
          </div>

          <div class="card">
            <div><strong>当前任务</strong></div>
            <div class="status" id="statusBox">还没有任务</div>
            <div class="progress"><div class="bar" id="progressBar"></div></div>
            <div class="hint mono" id="taskIdBox"></div>
          </div>

          <div class="card">
            <div><strong>最近任务</strong></div>
            <div id="taskList" class="hint">暂无</div>
          </div>

          <div class="card">
            <div><strong>我的资料</strong></div>
            <div class="hint">原文（你上传的俄语书）</div>
            <div id="srcList" class="hint">暂无</div>
            <div class="hint" style="margin-top: 12px;">译文（翻译结果）</div>
            <div id="outList" class="hint">暂无</div>
          </div>

          <div class="hint">
            调试接口仍可用：<a href="/docs">/docs</a>，健康检查：<a href="/health">/health</a>
          </div>
        </div>

        <script>
          let activeTaskId = "";
          let timer = null;

          async function uploadAndStart() {{
            const files = document.getElementById("uploadFiles").files;
            if (!files || files.length === 0) {{
              alert("请先选择要上传的文件");
              return;
            }}
            const form = new FormData();
            for (const f of files) {{
              form.append("files", f);
            }}
            form.append("force", document.getElementById("forceFlag").checked ? "true" : "false");
            try {{
              const res = await fetch("/tasks/upload", {{
                method: "POST",
                body: form
              }});
              const data = await res.json();
              if (!res.ok) {{
                throw new Error(data.detail || "上传失败");
              }}
              activeTaskId = data.task_id;
              document.getElementById("taskIdBox").textContent = "任务ID: " + activeTaskId;
              document.getElementById("statusBox").textContent = "上传成功，任务已开始";
              pollTask();
              loadMyFiles();
            }} catch (err) {{
              alert(err.message || "上传请求失败");
            }}
          }}

          async function pollTask() {{
            if (!activeTaskId) return;
            if (timer) clearInterval(timer);
            await loadTask(activeTaskId);
            timer = setInterval(async () => {{
              const done = await loadTask(activeTaskId);
              if (done && timer) {{
                clearInterval(timer);
                timer = null;
              }}
            }}, 2000);
          }}

          async function loadTask(taskId) {{
            const res = await fetch("/tasks/" + taskId);
            const task = await res.json();
            if (!res.ok) {{
              document.getElementById("statusBox").textContent = "读取任务失败";
              return true;
            }}
            const progress = Number(task.progress || 0);
            document.getElementById("progressBar").style.width = progress + "%";
            const msg = task.message || "";
            const file = task.current_file ? ("，当前文件：" + task.current_file) : "";
            document.getElementById("statusBox").textContent =
              "状态：" + task.status + " | 进度：" + progress + "%" + file + " | " + msg;
            refreshTasks();
            return task.status === "success" || task.status === "failed";
          }}

          async function refreshTasks() {{
            const res = await fetch("/tasks");
            const data = await res.json();
            if (!res.ok) return;
            const tasks = data.tasks || [];
            const list = document.getElementById("taskList");
            if (!tasks.length) {{
              list.textContent = "暂无";
              return;
            }}
            list.innerHTML = tasks.slice(0, 8).map(t =>
              `<div class="task">
                <div><strong>${{t.status}}</strong> | ${{t.progress}}%</div>
                <div class="mono">${{t.id}}</div>
                <div>${{t.current_file || ""}}</div>
                <div class="hint">${{t.updated_at || ""}}</div>
              </div>`
            ).join("");
          }}

          async function loadMyFiles() {{
            const res = await fetch("/my-files");
            const data = await res.json();
            if (!res.ok) return;
            const src = data.source_files || [];
            const out = data.translated_files || [];
            const srcList = document.getElementById("srcList");
            const outList = document.getElementById("outList");
            if (!src.length) {{
              srcList.textContent = "暂无";
            }} else {{
              srcList.innerHTML = src.map(n =>
                `<div class="task"><a href="/my-files/source/${{encodeURIComponent(n)}}" target="_blank">${{n}}</a></div>`
              ).join("");
            }}
            if (!out.length) {{
              outList.textContent = "暂无";
            }} else {{
              outList.innerHTML = out.map(n =>
                `<div class="task"><a href="/my-files/translated/${{encodeURIComponent(n)}}" target="_blank">${{n}}</a></div>`
              ).join("");
            }}
          }}

          refreshTasks();
          loadMyFiles();
        </script>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate")
def translate(req: TranslateRequest):
    try:
        summary = translate_path(
            input_path=Path(req.input),
            output_path=Path(req.output) if req.output else OUTPUT_DIR,
            force=req.force,
            include_files=req.include_files,
        )
        return {"message": "任务执行完成", "summary": summary}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tasks")
def create_task(req: TranslateRequest):
    task_id = uuid.uuid4().hex
    task = {
        "id": task_id,
        "status": "pending",
        "progress": 0.0,
        "message": "任务已创建",
        "error": None,
        "summary": None,
        "created_at": _now(),
        "updated_at": _now(),
        "current_file": "",
        "done_chunks": 0,
        "total_chunks": 1,
        "finished_files": 0,
        "total_files": 0,
        "request": req.model_dump(),
    }
    db_create_task(task)

    t = threading.Thread(target=_run_async_task, args=(task_id, req), daemon=True)
    t.start()
    return {"task_id": task_id, "status": "pending"}


@app.post("/tasks/upload")
def create_task_with_upload(
    files: List[UploadFile] = File(...),
    force: bool = Form(False),
):
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    saved = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            raise HTTPException(status_code=400, detail=f"不支持的文件类型: {f.filename}")
        saved.append(_save_upload_to_library(f))

    req = TranslateRequest(
        input=str(LIBRARY_SOURCE_DIR.resolve()),
        output=str(LIBRARY_OUTPUT_DIR.resolve()),
        force=force,
        include_files=saved,
    )
    resp = create_task(req)
    resp["uploaded_files"] = saved
    return resp


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = db_get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get("/tasks")
def list_tasks():
    tasks = db_list_tasks(limit=100)
    return {"tasks": tasks}


@app.get("/my-files")
def my_files():
    src = sorted(
        [p.name for p in LIBRARY_SOURCE_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}]
    )
    out = sorted(
        [p.name for p in LIBRARY_OUTPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}]
    )
    return {"source_files": src, "translated_files": out}


@app.get("/my-files/source/{name}")
def get_source_file(name: str):
    p = (LIBRARY_SOURCE_DIR / Path(name).name).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=str(p), filename=p.name)


@app.get("/my-files/translated/{name}")
def get_translated_file(name: str):
    p = (LIBRARY_OUTPUT_DIR / Path(name).name).resolve()
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path=str(p), filename=p.name)
