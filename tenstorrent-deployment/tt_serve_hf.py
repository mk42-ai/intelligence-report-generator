#!/usr/bin/env python3
"""
AIREV Qwen-0.8B-AgentJSON — HuggingFace Inference Server
Deployed on Tenstorrent Wormhole Quiet Box.

Model: Qwen 3.5-0.8B fine-tuned with Progressive Curriculum GRPO
       for OnDemand.io tool calling (2,176 real production plugins)
Hardware: Tenstorrent Wormhole B0 (n300)
Backend: HuggingFace transformers (CPU path with GDN recurrence)

Usage:
    python3 tt_serve_hf.py

Endpoint:
    POST /v1/chat/completions  (OpenAI-compatible)
    GET  /health
    GET  /v1/models
"""
import os, json, time, uuid, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Qwen 0.8B AgentJSON — Tenstorrent Wormhole")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/qwen-0.8b-agentjson"))

print("Loading model with HuggingFace transformers...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
)
model.eval()
PARAM_COUNT = sum(p.numel() for p in model.parameters()) / 1e6
print("Model loaded! Params: %.1fM" % PARAM_COUNT)


def format_chat(messages):
    """Format messages into Qwen chat template with thinking tokens."""
    text = ""
    for msg in messages:
        text += "<|im_start|>%s\n%s<|im_end|>\n" % (msg["role"], msg["content"])
    text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return text


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "qwen-0.8b-tt"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 200
    top_p: Optional[float] = 0.95


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "qwen-0.8b-agentjson",
        "model_version": "grpo-v9-extended",
        "parameters_millions": round(PARAM_COUNT, 1),
        "hardware": "tenstorrent-wormhole-b0",
        "device_type": "n300",
        "backend": "huggingface-transformers",
        "architecture": "Qwen3_5-GDN (Gated Delta Networks)",
        "training": "Progressive Curriculum GRPO (7 phases)",
        "use_case": "OnDemand.io tool calling orchestration",
    }

@app.get("/v1/models")
def models_list():
    return {"data": [{"id": "qwen-0.8b-tt", "object": "model", "owned_by": "airev-ae",
                       "hardware": "tenstorrent-wormhole-b0",
                       "parameters": "752M",
                       "training": "GRPO-v9-extended"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = format_chat(msgs)
    inputs = tokenizer(text, return_tensors="pt")
    input_len = inputs.input_ids.shape[1]

    temp = max(req.temperature or 0.7, 0.01)
    top_p_val = req.top_p or 0.95
    max_new = min(req.max_tokens or 200, 512)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p_val,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    new_tokens = outputs.shape[1] - input_len
    response_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    return JSONResponse({
        "id": "chatcmpl-%s" % uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen-0.8b-tt",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text},
                     "finish_reason": "stop" if outputs[0][-1] == tokenizer.eos_token_id else "length"}],
        "usage": {"prompt_tokens": input_len, "completion_tokens": new_tokens,
                  "total_tokens": input_len + new_tokens},
        "hardware": "tenstorrent-wormhole-b0",
        "performance": {
            "tokens_generated": new_tokens,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
            "prompt_tokens": input_len,
        }
    })


if __name__ == "__main__":
    print("\n=== AIREV Qwen-0.8B AgentJSON ===")
    print("Hardware: Tenstorrent Wormhole B0 (Quiet Box)")
    print("Model: GRPO v9-extended (unified OnDemand + BFCL)")
    print("Port: 8600\n")
    uvicorn.run(app, host="0.0.0.0", port=8600)
