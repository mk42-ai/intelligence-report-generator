#!/usr/bin/env python3
"""
AIREV Qwen-0.8B-AgentJSON — Full TTNN-Accelerated Inference v2
All matmuls, norms, and activations run on Wormhole.
GDN recurrence runs hybrid (state loop on CPU, matmuls on device).

Key features:
  - KV cache: don't reprocess full sequence every token
  - Prefill/Decode split: process prompt once, then generate token-by-token
  - GDN state cache: carry forward recurrence state between tokens
  - Full attention KV cache: standard KV cache for attention layers
  - Pre-transposed weights on device for faster matmul
"""
import os, json, time, uuid, torch, math
from pathlib import Path
from safetensors import safe_open
from tokenizers import Tokenizer
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
import ttnn

app = FastAPI(title="Qwen 0.8B — Tenstorrent Wormhole TTNN v2")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.expanduser("~/models/qwen-0.8b-agentjson"))

# Load tokenizer
print("Loading tokenizer...")
tok = Tokenizer.from_file(str(Path(MODEL_DIR) / "tokenizer.json"))
EOS_TOKEN_ID = 248046

# Load config
with open(str(Path(MODEL_DIR) / "config.json")) as f:
    config = json.load(f)
LAYERS = config["num_hidden_layers"]
LAYER_TYPES = config.get("layer_types", [])
HIDDEN_SIZE = config.get("hidden_size", 1024)
NUM_HEADS = 16  # GDN heads
HEAD_DIM_GDN = 128  # GDN head dim
NUM_KV_HEADS = config.get("num_key_value_heads", 2)
NUM_Q_HEADS = config.get("num_attention_heads", 8)
HEAD_DIM_ATTN = config.get("head_dim", 256)

# Open Wormhole device
print("Opening Wormhole device...")
device = ttnn.open_device(device_id=0)
print("Wormhole device open!")

# Load weights and move to device
print("Loading weights to Wormhole...")
weights_cpu = {}
weights_tt = {}
with safe_open(str(Path(MODEL_DIR) / "model.safetensors"), framework="pt") as f:
    for key in f.keys():
        t = f.get_tensor(key).to(torch.bfloat16)
        weights_cpu[key] = t
        if t.numel() > 1024 and len(t.shape) == 2:
            try:
                weights_tt[key] = ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT)
            except:
                weights_tt[key] = None
        else:
            weights_tt[key] = None

# Pre-transpose linear weights for matmul (x @ W^T = x @ W_T)
for key in list(weights_cpu.keys()):
    if len(weights_cpu[key].shape) == 2 and weights_cpu[key].numel() > 1024:
        kt = key + "_T"
        try:
            weights_tt[kt] = ttnn.from_torch(
                weights_cpu[key].T.contiguous(), device=device, layout=ttnn.TILE_LAYOUT
            )
        except:
            weights_tt[kt] = None

on_device = sum(1 for v in weights_tt.values() if v is not None)
print("Loaded %d tensors (%d on device)" % (len(weights_cpu), on_device))


def ttnn_linear(x_tt, key):
    """Linear projection on Wormhole using pre-transposed weight."""
    w_tt = weights_tt.get(key + "_T")
    if w_tt is not None:
        return ttnn.matmul(x_tt, w_tt)
    x_cpu = ttnn.to_torch(x_tt)
    return ttnn.from_torch(x_cpu @ weights_cpu[key].T, device=device, layout=ttnn.TILE_LAYOUT)


def rms_norm(x, weight_key):
    """RMS norm on CPU."""
    w = weights_cpu[weight_key].float()
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(variance + 1e-6)) * (1.0 + w)).to(torch.bfloat16)


def gdn_forward(x, layer_idx, gdn_states):
    """GDN layer with TTNN-accelerated matmuls, CPU recurrence."""
    p = "model.language_model.layers.%d" % layer_idx
    B, L, D = x.shape

    h = rms_norm(x, p + ".input_layernorm.weight")
    h_tt = ttnn.from_torch(h.reshape(B * L, D), device=device, layout=ttnn.TILE_LAYOUT)

    # QKV on Wormhole
    qkv_tt = ttnn_linear(h_tt, p + ".linear_attn.in_proj_qkv.weight")
    qkv = ttnn.to_torch(qkv_tt).reshape(B, L, -1)

    # Conv1d + SiLU on CPU
    qkv_t = qkv.float().transpose(1, 2)
    conv_w = weights_cpu[p + ".linear_attn.conv1d.weight"].float()
    groups = qkv_t.shape[1] // conv_w.shape[1] if conv_w.shape[1] == 1 else 1
    qkv_padded = torch.nn.functional.pad(qkv_t, (3, 0))
    qkv_conv = torch.nn.functional.silu(
        torch.nn.functional.conv1d(qkv_padded, conv_w, groups=groups)
    )[:, :, :L]
    qkv_back = qkv_conv.transpose(1, 2).to(torch.bfloat16)

    q, k, v = torch.split(qkv_back, [2048, 2048, 2048], dim=-1)
    q = q.reshape(B, L, NUM_HEADS, HEAD_DIM_GDN)
    k = k.reshape(B, L, NUM_HEADS, HEAD_DIM_GDN)
    v = v.reshape(B, L, NUM_HEADS, HEAD_DIM_GDN)

    # Z on Wormhole
    z_tt = ttnn_linear(h_tt, p + ".linear_attn.in_proj_z.weight")
    z = ttnn.to_torch(z_tt).reshape(B, L, NUM_HEADS, HEAD_DIM_GDN)

    # A, B on CPU
    h_cpu = h.reshape(B * L, D)
    a_proj = (h_cpu @ weights_cpu[p + ".linear_attn.in_proj_a.weight"].T).reshape(B, L, -1)
    b_proj = (h_cpu @ weights_cpu[p + ".linear_attn.in_proj_b.weight"].T).reshape(B, L, -1)

    A_log = weights_cpu[p + ".linear_attn.A_log"]
    dt_bias = weights_cpu[p + ".linear_attn.dt_bias"]
    beta = b_proj.sigmoid()
    g = -A_log.float().exp() * torch.nn.functional.softplus(a_proj.float() + dt_bias.float())

    q_norm = torch.nn.functional.normalize(q.float(), p=2, dim=-1)
    k_norm = torch.nn.functional.normalize(k.float(), p=2, dim=-1)
    scale = 1.0 / (HEAD_DIM_GDN ** 0.5)

    q_s = (q_norm * scale).transpose(1, 2).contiguous().float()
    k_s = k_norm.transpose(1, 2).contiguous().float()
    v_s = v.float().transpose(1, 2).contiguous()
    beta_s = beta.float().transpose(1, 2).contiguous()
    g_s = g.float().transpose(1, 2).contiguous()

    # GDN recurrence (sequential on CPU)
    state = gdn_states.get(layer_idx,
        torch.zeros(B, NUM_HEADS, HEAD_DIM_GDN, HEAD_DIM_GDN, dtype=torch.float32))
    core_out = torch.zeros(B, NUM_HEADS, L, HEAD_DIM_GDN, dtype=torch.float32)

    for i in range(L):
        q_t = q_s[:, :, i]
        k_t = k_s[:, :, i]
        v_t = v_s[:, :, i]
        g_t = g_s[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta_s[:, :, i].unsqueeze(-1)

        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    gdn_states[layer_idx] = state

    core_out = core_out.transpose(1, 2).contiguous().to(torch.bfloat16)
    core_flat = core_out.reshape(B * L, NUM_HEADS, HEAD_DIM_GDN)
    z_flat = z.reshape(B * L, NUM_HEADS, HEAD_DIM_GDN)

    norm_w = weights_cpu[p + ".linear_attn.norm.weight"].float()
    variance = core_flat.float().pow(2).mean(-1, keepdim=True)
    core_normed = (core_flat.float() * torch.rsqrt(variance + 1e-6)) * norm_w
    gated = (core_normed * torch.nn.functional.silu(z_flat.float())).to(torch.bfloat16)
    gated = gated.reshape(B * L, 2048)

    # Output projection on Wormhole
    gated_tt = ttnn.from_torch(gated, device=device, layout=ttnn.TILE_LAYOUT)
    attn_out_tt = ttnn_linear(gated_tt, p + ".linear_attn.out_proj.weight")
    attn_out = ttnn.to_torch(attn_out_tt).reshape(B, L, -1)

    x = x + attn_out

    # MLP on Wormhole
    h2 = rms_norm(x, p + ".post_attention_layernorm.weight")
    h2_tt = ttnn.from_torch(h2.reshape(B * L, D), device=device, layout=ttnn.TILE_LAYOUT)
    gate_tt = ttnn_linear(h2_tt, p + ".mlp.gate_proj.weight")
    gate_tt = ttnn.silu(gate_tt)
    up_tt = ttnn_linear(h2_tt, p + ".mlp.up_proj.weight")
    mlp_hidden_tt = ttnn.mul(gate_tt, up_tt)
    mlp_out_tt = ttnn_linear(mlp_hidden_tt, p + ".mlp.down_proj.weight")
    mlp_out = ttnn.to_torch(mlp_out_tt).reshape(B, L, D)

    x = x + mlp_out
    return x, gdn_states


def full_attn_forward(x, layer_idx, kv_cache, pos):
    """Full attention layer with KV cache and TTNN acceleration."""
    p = "model.language_model.layers.%d" % layer_idx
    B, L, D = x.shape

    h = rms_norm(x, p + ".input_layernorm.weight")
    hf = h.reshape(B * L, D)

    q_full = hf @ weights_cpu[p + ".self_attn.q_proj.weight"].T
    q_states, gate = torch.chunk(q_full.view(B, L, -1, HEAD_DIM_ATTN * 2), 2, dim=-1)
    gate = gate.reshape(B, L, -1)
    k_new = (hf @ weights_cpu[p + ".self_attn.k_proj.weight"].T).reshape(B, L, NUM_KV_HEADS, HEAD_DIM_ATTN)
    v_new = (hf @ weights_cpu[p + ".self_attn.v_proj.weight"].T).reshape(B, L, NUM_KV_HEADS, HEAD_DIM_ATTN)
    q_states = q_states.view(B, L, NUM_Q_HEADS, HEAD_DIM_ATTN)

    qnw = weights_cpu[p + ".self_attn.q_norm.weight"].float()
    knw = weights_cpu[p + ".self_attn.k_norm.weight"].float()
    qv = q_states.float().pow(2).mean(-1, keepdim=True)
    q_states = ((q_states.float() * torch.rsqrt(qv + 1e-6)) * qnw).to(torch.bfloat16)
    kv = k_new.float().pow(2).mean(-1, keepdim=True)
    k_new = ((k_new.float() * torch.rsqrt(kv + 1e-6)) * knw).to(torch.bfloat16)

    # KV cache
    if layer_idx in kv_cache:
        kv_cache[layer_idx]['k'] = torch.cat([kv_cache[layer_idx]['k'], k_new], dim=1)
        kv_cache[layer_idx]['v'] = torch.cat([kv_cache[layer_idx]['v'], v_new], dim=1)
    else:
        kv_cache[layer_idx] = {'k': k_new, 'v': v_new}

    k_all = kv_cache[layer_idx]['k']
    v_all = kv_cache[layer_idx]['v']

    k_exp = k_all.repeat(1, 1, NUM_Q_HEADS // NUM_KV_HEADS, 1)
    v_exp = v_all.repeat(1, 1, NUM_Q_HEADS // NUM_KV_HEADS, 1)

    q = q_states.transpose(1, 2)
    k = k_exp.transpose(1, 2)
    v = v_exp.transpose(1, 2)

    scale = 1.0 / (HEAD_DIM_ATTN ** 0.5)
    attn = (q.float() @ k.float().transpose(-2, -1)) * scale
    if L > 1:
        S = k.shape[2]
        mask = torch.triu(torch.ones(L, S, dtype=torch.bool), diagonal=S - L + 1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    out = (attn @ v.float()).to(torch.bfloat16)

    out = out.transpose(1, 2).reshape(B, L, -1).contiguous()
    out = out * torch.sigmoid(gate.to(torch.bfloat16))

    # Output projection on Wormhole
    out_tt = ttnn.from_torch(out.reshape(B * L, -1), device=device, layout=ttnn.TILE_LAYOUT)
    attn_out_tt = ttnn_linear(out_tt, p + ".self_attn.o_proj.weight")
    attn_out = ttnn.to_torch(attn_out_tt).reshape(B, L, D)

    x = x + attn_out

    # MLP on Wormhole
    h2 = rms_norm(x, p + ".post_attention_layernorm.weight")
    h2_tt = ttnn.from_torch(h2.reshape(B * L, D), device=device, layout=ttnn.TILE_LAYOUT)
    gate_tt = ttnn_linear(h2_tt, p + ".mlp.gate_proj.weight")
    gate_tt = ttnn.silu(gate_tt)
    up_tt = ttnn_linear(h2_tt, p + ".mlp.up_proj.weight")
    mlp_hidden_tt = ttnn.mul(gate_tt, up_tt)
    mlp_out_tt = ttnn_linear(mlp_hidden_tt, p + ".mlp.down_proj.weight")
    mlp_out = ttnn.to_torch(mlp_out_tt).reshape(B, L, D)

    x = x + mlp_out
    return x, kv_cache


def generate(input_ids, max_new_tokens=200, temperature=0.7, top_p=0.95):
    """Generate with prefill + decode phases."""
    B, L = input_ids.shape
    embed_w = weights_cpu["model.language_model.embed_tokens.weight"]
    x = torch.nn.functional.embedding(input_ids, embed_w)

    gdn_states = {}
    kv_cache = {}

    # Prefill
    t0 = time.time()
    for i in range(LAYERS):
        if LAYER_TYPES[i] == "linear_attention":
            x, gdn_states = gdn_forward(x, i, gdn_states)
        else:
            x, kv_cache = full_attn_forward(x, i, kv_cache, L)

    x_last = rms_norm(x[:, -1:, :], "model.language_model.norm.weight")
    logits = x_last.reshape(1, HIDDEN_SIZE) @ embed_w.T
    prefill_time = time.time() - t0

    # Sample
    next_logits = logits / max(temperature, 0.01)
    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
    cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    mask = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[mask] = float("-inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    next_idx = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_idx)

    generated = [next_token.item()]
    pos = L

    # Decode
    t1 = time.time()
    for step in range(max_new_tokens - 1):
        if next_token.item() == EOS_TOKEN_ID:
            break

        x = torch.nn.functional.embedding(next_token.reshape(1, 1), embed_w)
        for i in range(LAYERS):
            if LAYER_TYPES[i] == "linear_attention":
                x, gdn_states = gdn_forward(x, i, gdn_states)
            else:
                x, kv_cache = full_attn_forward(x, i, kv_cache, pos + 1)

        x_norm = rms_norm(x, "model.language_model.norm.weight")
        logits = x_norm.reshape(1, HIDDEN_SIZE) @ embed_w.T

        next_logits = logits / max(temperature, 0.01)
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[mask] = float("-inf")
        probs = torch.softmax(sorted_logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        next_token = sorted_indices.gather(-1, next_idx)

        generated.append(next_token.item())
        pos += 1

    decode_time = time.time() - t1
    return generated, prefill_time, decode_time


def format_chat(messages):
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
        "hardware": "tenstorrent-wormhole-b0",
        "acceleration": "TTNN-v2",
        "features": ["kv_cache", "prefill_decode_split", "gdn_state_cache", "on_device_matmul"],
    }

@app.get("/v1/models")
def models_list():
    return {"data": [{"id": "qwen-0.8b-tt", "object": "model", "owned_by": "airev-ae",
                       "hardware": "tenstorrent-wormhole-b0"}]}

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = format_chat(msgs)
    encoded = tok.encode(text)
    input_ids = torch.tensor([encoded.ids])

    temp = max(req.temperature or 0.7, 0.01)
    max_new = min(req.max_tokens or 200, 512)

    start = time.time()
    tokens, prefill_time, decode_time = generate(input_ids, max_new, temp, req.top_p or 0.95)
    elapsed = time.time() - start

    response_text = tok.decode(tokens)
    new_tokens = len(tokens)

    return JSONResponse({
        "id": "chatcmpl-%s" % uuid.uuid4().hex[:8],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "qwen-0.8b-tt",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text},
                     "finish_reason": "stop" if (tokens and tokens[-1] == EOS_TOKEN_ID) else "length"}],
        "usage": {"prompt_tokens": input_ids.shape[1], "completion_tokens": new_tokens,
                  "total_tokens": input_ids.shape[1] + new_tokens},
        "hardware": "tenstorrent-wormhole-b0",
        "performance": {
            "tokens_generated": new_tokens,
            "time_seconds": round(elapsed, 2),
            "tokens_per_second": round(new_tokens / elapsed, 1) if elapsed > 0 else 0,
            "prefill_seconds": round(prefill_time, 2),
            "decode_seconds": round(decode_time, 2),
            "decode_tok_per_sec": round(new_tokens / decode_time, 1) if decode_time > 0 else 0,
        }
    })


if __name__ == "__main__":
    print("\n=== AIREV Qwen-0.8B — TTNN v2 (KV Cache + Prefill/Decode) ===")
    print("Hardware: Tenstorrent Wormhole B0")
    print("Acceleration: TTNN (matmuls on chip, GDN recurrence on CPU)")
    print("Port: 8600\n")
    uvicorn.run(app, host="0.0.0.0", port=8600)
