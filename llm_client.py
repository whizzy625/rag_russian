from openai import OpenAI
from config import API_KEY, MODEL_NAMES
import time

client = None
_model_index = 0


def get_client():
    global client

    if not API_KEY:
        raise ValueError(
            "未配置 DASHSCOPE_API_KEY。请在项目根目录创建 .env 文件并填写 DASHSCOPE_API_KEY=你的key，或在环境变量中设置。"
        )

    if client is None:
        client = OpenAI(
            api_key=API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    return client


def _should_switch_model(err: Exception) -> bool:
    msg = str(err).lower()
    switch_signals = [
        "insufficient_quota",
        "quota",
        "token",
        "余额",
        "额度",
        "limit",
        "rate limit",
        "exhaust",
        "429",
        "500",
        "502",
        "503",
        "504",
        "connection error",
        "timeout",
        "timed out",
        "data_inspection_failed",
        "datainspectionfailed",
    ]
    return any(flag in msg for flag in switch_signals)


def _is_transient_error(err: Exception) -> bool:
    msg = str(err).lower()
    transient_signals = [
        "connection error",
        "timeout",
        "timed out",
        "read timeout",
        "connect timeout",
        "temporarily unavailable",
        "server error",
        "peer closed connection",
        "incomplete chunked read",
        "chunked read",
        "remoteprotocolerror",
        "connection reset",
        "broken pipe",
        "502",
        "503",
        "504",
    ]
    return any(flag in msg for flag in transient_signals)


def _is_content_moderation_error(err: Exception) -> bool:
    msg = str(err).lower()
    moderation_signals = [
        "data_inspection_failed",
        "datainspectionfailed",
        "inappropriate-content",
        "inappropriate content",
        "input data may contain inappropriate content",
        "error-code#inappropriate-content",
    ]
    return any(flag in msg for flag in moderation_signals)


def _is_stream_required_error(err: Exception) -> bool:
    msg = str(err).lower()
    stream_required_signals = [
        "only support stream mode",
        "please enable the stream parameter",
        "stream mode",
    ]
    return any(flag in msg for flag in stream_required_signals)


def llm_chat(prompt: str, temperature=0.3):
    global _model_index

    last_error = None
    total = len(MODEL_NAMES)
    if total == 0:
        raise RuntimeError("未配置可用模型。请检查 MODEL_NAME 或 MODEL_NAMES。")

    for step in range(total):
        idx = (_model_index + step) % total
        model = MODEL_NAMES[idx]
        use_stream = False
        for retry in range(3):
            try:
                messages = [
                    {"role": "system", "content": "你是专业俄中学术翻译"},
                    {"role": "user", "content": prompt},
                ]
                if use_stream:
                    stream = get_client().chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=120,
                        stream=True,
                        extra_body={"enable_thinking": False},
                    )
                    parts = []
                    for chunk in stream:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        text = getattr(delta, "content", None)
                        if text:
                            parts.append(text)
                    content = "".join(parts).strip()
                    if not content:
                        raise RuntimeError(f"模型 {model} 流式返回为空")
                else:
                    response = get_client().chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        timeout=120,
                        # DashScope 部分模型在非流式请求下要求显式关闭 thinking
                        extra_body={"enable_thinking": False},
                    )
                    content = response.choices[0].message.content.strip()
                if step > 0:
                    print(f"[llm] 已切换模型: {MODEL_NAMES[_model_index]} -> {model}")
                # 主动轮换：本次成功后，下次从下一个模型开始，均摊免费额度
                _model_index = (idx + 1) % total if total > 1 else idx
                return content
            except Exception as exc:
                last_error = exc
                if (not use_stream) and _is_stream_required_error(exc):
                    print(f"[llm] 模型 {model} 仅支持流式，自动切换到 stream=True 重试...")
                    use_stream = True
                    continue
                # 内容审核错误通常与输入相关，切模型无意义，交给上层做降级处理
                if _is_content_moderation_error(exc):
                    raise
                if _is_transient_error(exc) and retry < 2:
                    wait_sec = 2 ** retry
                    print(f"[llm] 模型 {model} 连接波动，{wait_sec}s 后重试({retry + 1}/3)...")
                    time.sleep(wait_sec)
                    continue
                if not _should_switch_model(exc):
                    raise
                break
        if step < total - 1:
            print(f"[llm] 模型 {model} 不可用，尝试切换到下一个模型...")

    raise RuntimeError(
        f"所有模型均不可用，请检查配额或模型名。当前模型列表: {', '.join(MODEL_NAMES)}"
    ) from last_error
