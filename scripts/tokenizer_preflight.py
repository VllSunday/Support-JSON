"""Download only the public tokenizer; check complete SFT sequences before training."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from huggingface_hub import HfApi
from support_json.data import PROFILES, digest, read_jsonl

root = Path(__file__).resolve().parents[1]
revision = HfApi().model_info("Qwen/Qwen3.5-4B").sha
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=revision)
lengths = {}
for split in PROFILES:
    rows = read_jsonl(root / f"data/pilot-v0.2/{split}.chatml.jsonl")
    token_lists = [tokenizer.apply_chat_template(row["messages"], tokenize=True,
                  return_dict=False, add_generation_prompt=False, enable_thinking=False) for row in rows]
    if not all(isinstance(tokens, list) and all(type(token) is int for token in tokens) for tokens in token_lists):
        raise TypeError("Tokenizer must return a flat list of token ids for each conversation")
    counts = [len(tokens) for tokens in token_lists]
    lengths[split] = {"n": len(counts), "max_tokens": max(counts),
                      "mean_tokens": round(sum(counts)/len(counts), 1),
                      "over_4096": sum(n > 4096 for n in counts)}
if any(value["over_4096"] for value in lengths.values()):
    raise ValueError("Training length would truncate examples; adjust config before training")
report = {"model": "Qwen/Qwen3.5-4B", "tokenizer_only": True,
          "model_revision": revision,
          "thinking": False, "limit": 4096, "splits": lengths,
          "chat_template_fingerprint": digest(tokenizer.chat_template),
          "note": "Checks full sequence length. Assistant loss masking and GPU memory still need runtime verification."}
output = root / "reports/tokenizer-preflight.json"
if output.exists():
    raise ValueError("Report already exists; preserve the prior preflight")
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
