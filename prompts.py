try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

translation_template = """
你是专业俄语学术翻译。

参考术语：
{terms}

历史翻译参考：
{memory}

上文内容：
{context}

请将以下俄文翻译为中文：

{text}

要求：
1 保留段落结构
2 术语保持一致
3 忠实原文
"""

translation_prompt = PromptTemplate(
    input_variables=["terms", "memory", "context", "text"],
    template=translation_template,
)


summary_template = """
请用中文总结以下内容：

{text}
"""

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=summary_template,
)
