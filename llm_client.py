from openai import OpenAI
from config import API_KEY, MODEL_NAMES

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
        "data_inspection_failed",
        "datainspectionfailed",
    ]
    return any(flag in msg for flag in switch_signals)


def llm_chat(prompt: str, temperature=0.3):
    global _model_index

    last_error = None
    total = len(MODEL_NAMES)
    if total == 0:
        raise RuntimeError("未配置可用模型。请检查 MODEL_NAME 或 MODEL_NAMES。")

    for step in range(total):
        idx = (_model_index + step) % total
        model = MODEL_NAMES[idx]
        try:
            response = get_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是专业俄中学术翻译"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            if idx != _model_index:
                print(f"[llm] 已切换模型: {MODEL_NAMES[_model_index]} -> {model}")
            _model_index = idx
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if not _should_switch_model(exc):
                raise
            if step < total - 1:
                print(f"[llm] 模型 {model} 不可用，尝试切换到下一个模型...")

    raise RuntimeError(
        f"所有模型均不可用，请检查配额或模型名。当前模型列表: {', '.join(MODEL_NAMES)}"
    ) from last_error
