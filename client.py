"""
加载本地 SFT/RL 模型或 Hugging Face 模型，提供与 OpenAI Chat Completions 相同的请求/响应形式。
model_path 可为：本地目录（如 ./output/sft）或 HF 模型 ID（如 Qwen/Qwen2-0.5B-Instruct）。
运行: uvicorn client:app --host 0.0.0.0 --port 8000
调用: POST /v1/chat/completions，body 同 OpenAI（messages, max_tokens, temperature 等）
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import time
import uuid
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 支持本地目录（SFT/RL 产出）或 Hugging Face 模型 ID（如 Qwen/Qwen2-0.5B-Instruct）
# model_path = "./output/sft"
model_path = "Qwen/Qwen3-14B"
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Local Chat API", description="OpenAI 兼容的本地模型接口")

tokenizer = None
model = None


def load_model(path: str):
    global tokenizer, model
    is_local_dir = os.path.exists(path) and os.path.isdir(path)
    if is_local_dir:
        path = os.path.abspath(path)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_local_dir:
        adapter_config = os.path.join(path, "adapter_config.json")
        if os.path.isfile(adapter_config):
            import json

            with open(adapter_config, "r") as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path", path)
            model = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None,
            )
            model = PeftModel.from_pretrained(model, path)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto" if device == "cuda" else None,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto" if device == "cuda" else None,
        )

    if device == "cpu":
        model = model.to(device)
    model.eval()


@app.on_event("startup")
def startup():
    load_model(model_path)


def messages_to_prompt(messages: list) -> str:
    """将 OpenAI messages 转为模型输入的 prompt 字符串。"""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "") or ""
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply_ids = out[0][prompt_len:]
    text = tokenizer.decode(reply_ids, skip_special_tokens=True)
    completion_tokens = len(reply_ids)
    return text.strip(), prompt_len, completion_tokens


# ---------- OpenAI 兼容的请求/响应体 ----------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "local"
    messages: list = Field(
        ..., description='同 OpenAI: [{"role": "user", "content": "..."}]'
    )
    max_tokens: int = 512
    temperature: float = 0.7
    stream: bool = False


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=501, detail="暂不支持 stream=True")
    prompt = messages_to_prompt(req.messages)
    try:
        content, prompt_tokens, completion_tokens = generate(
            prompt, max_tokens=req.max_tokens, temperature=req.temperature
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
