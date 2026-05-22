"""验证 Track A 环境:确认能从 WSL 调用 Windows 端 ollama 的 qwen3:8b。"""
from openai import OpenAI

# WSL2 mirrored networking 下,localhost 直通 Windows ollama
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

resp = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "用一句话介绍《斗罗大陆》主角唐三。/no_think"}],
)
print(resp.choices[0].message.content)
