---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
- ru
datasets:
- A11Sunday/support-json-ru
tags:
- lora
- support
- structured-output
- policy-following
- qwen3.5
---

![Support-JSON](assets/banner.svg)

# Support-JSON · Qwen3.5-4B LoRA

A local Russian SaaS support adapter: customer request, company policies and sourced facts → JSON decision and draft reply. It assists an operator; it does not execute refunds, modify accounts or send messages to customers.

**Локальный помощник поддержки: разбирает обращение по переданным правилам, предлагает действие и готовит черновик ответа на русском.**

[Dataset](https://huggingface.co/datasets/A11Sunday/support-json-ru) · [Animated demo](assets/demo.mp4) · [Evaluation](evaluation/RESULTS.md) · [Capstone package](artifacts/Support_JSON_Capstone-v3.zip)

[![Watch the real application demo](assets/demo-preview.gif)](assets/demo.mp4)

The video is an animated presentation of real local application captures. Transitions and waiting are shortened. A screenshot is not a quality benchmark; all draft replies require review.

## What the adapter predicts

Input: `message`, `history`, policies with `id` / `text`, facts with typed values and `origin`, and available capabilities. Unknown facts are null. Customer reports are distinct from system observations.

Nine output fields: **category**, **priority**, **sentiment**, **action**, **recommended_action**, **reply**, **missing_info**, **supported_rule_ids**, **human_escalation**. Company rules are supplied in context for each request. Preserve the provided prompt and schema rather than sending only a naked customer message.

The included code strictly checks JSON, schema and cross-field invariants. It does not repair invalid output or verify every claim in the reply. Correct JSON does not guarantee a correct policy decision.

## Run locally

Tested on Windows, Python 3.12, RTX 5070 Ti with **16 GB VRAM**, CUDA 12.8. Other hardware is not certified; CPU deployment is not implemented in this example. Base weights are about 9.3 GB on disk; adapter weights are about 130 MB. Allow additional space for downloads and GPU buffers.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install huggingface_hub==1.32.0
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('A11Sunday/support-json-qwen3.5-4b-lora', local_dir='support-json', allow_patterns=['*.json','*.txt','*.py','*.safetensors','app/**','LICENSE','NOTICE'])"
.\.venv\Scripts\python.exe -m pip install -r support-json/app/requirements.inference.txt
.\.venv\Scripts\python.exe -m pip install -e support-json/app
.\.venv\Scripts\python.exe -X utf8 -m support_json.demo --model Qwen/Qwen3.5-4B --adapter support-json
```

Open **http://127.0.0.1:7860**. The first request downloads/loads the pinned base model; subsequent requests use the loaded model. The server is loopback-only. GPU training or a game on the same card competes for memory and computation. Stop the server when finished to release GPU memory.

For a standalone JSON request using the included synthetic sample:

```powershell
.\.venv\Scripts\python.exe -X utf8 support-json/predict.py
# Or supply a context file using --input path/to/context.json.
```

The base revision is pinned to `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; no custom remote model code is required. The model is a **PEFT adapter**, not a merged standalone model or GGUF. `predict.py` forces suggestion-only capabilities. Preserve operator review.

## Training

Qwen3.5-4B, text-only, BF16 LoRA **r=16 / alpha=32**, all-linear, dropout 0, learning rate 1e-4, seed 42, response-only loss, context 2048. Selected run: **512 synthetic train rows × 2 epochs**, micro-batch 2, accumulation 4, 128 updates. Soup 0.75.0 / Transformers 5.17.0 / PEFT 0.21.0 / TRL 0.29.1 / Liger on native Windows.

Three adapters were evaluated on 96 validation examples. The third was selected before opening the 108-example test, with a critical-recall safety constraint and a predeclared ordering favoring action accuracy. The 10,000 published candidates were **not all trained**. The initial train/validation/test corpora and selected training snapshot remain frozen.

## Evaluation: same prompt, same test

108 held-out synthetic SaaS cases; base and LoRA use the same prompt/settings, greedy non-thinking generation, max 700 new tokens, batch 8, both native stop tokens. No JSON repair or constrained decoding.

| Field-only metric | Base | LoRA |
|---|---:|---:|
| Category accuracy | 76.9% | **100.0%** |
| Priority accuracy | 72.2% | **99.1%** |
| Action accuracy | 27.8% | **89.8%** |
| Schema + invariants valid | 54.6% | **95.4%** |

When invalid output fails all task fields, **strict action accuracy is 24.1% → 86.1%**. Field-only scores and strict scores answer different questions. A separate base + four frozen train-example control reaches 40.7% action accuracy with a different prompt; it is not a comparison to a stronger large model.

Content review is separate: a blinded **AI-assisted** audit of 36 paired test replies rates quality **2.78 → 4.31 out of 5** and flags unsupported fact/policy claims in **13/36 → 5/36** replies. The evaluator was an AI assistant, **not independent human reviewers**. Raw predictions, rubric ratings and limitations are included in the evaluation folder.

Additional prototype checks match expected fields plus contract in 8/10 model replies. Known failures include an incorrect denial at an inclusive refund boundary, a failed changed-threshold pair and an invented external refund status. These are validation-derived checks, not a fresh unseen benchmark.

## Limitations

- Small synthetic suite, mostly templated Russian. Training histories are empty; the application accepts history, but robust multi-turn behavior is not established.
- Test company identifiers and authored families differ from train, but semantic templates can overlap. Arbitrary-company generalization is **not proven**.
- Only three emotional variants of one critical test scenario. Critical recall 3/3 is not a broad safety guarantee.
- No independent human annotation or evaluation; no production customer data, strong-large-model comparison or measured operator-time savings.
- Base and LoRA were benchmarked in a shared desktop session. Packet/amortized times are not request latency or isolated speed measurements.
- **5/36 reviewed LoRA replies still contain false claims**. Use as an operator assistant; do not enable autonomous financial or account actions.
- No RAG, live company connectors, verified confidence score or full prompt-injection defense is included.

## License and provenance

Adapter and authored code: **Apache-2.0**, author **A11Sunday**, 2026. Base Qwen license and font notices are retained. Synthetic dataset: **CC BY 4.0**, separately published. The repository contains the adapter and application source; base weights are downloaded from the pinned upstream revision.

[Source app](app/) · [Capstone requirement audit](evaluation/TZ_AUDIT.md) · [Content error analysis](evaluation/error-analysis-final.md) · [Presentation](artifacts/Support_JSON_Capstone-v2.pptx)
