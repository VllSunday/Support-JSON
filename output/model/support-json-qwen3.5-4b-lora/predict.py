"""Standalone local inference; no customer text is sent to a model API."""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from contracts import messages, parse_output, validate_output

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--input', default=str(here/'demo_context.json'))
parser.add_argument('--base', default='Qwen/Qwen3.5-4B')
args = parser.parse_args()
revision = json.loads((here/'adapter_config.json').read_text(encoding='utf-8'))['revision']
if not torch.cuda.is_available():
    raise RuntimeError('This tested example requires a CUDA GPU. CPU deployment is not certified.')
context = json.loads(Path(args.input).read_text(encoding='utf-8'))
context.setdefault('history', [])
context.setdefault('facts', [])
context['capabilities'] = {'execution': 'suggest_only', 'tools': []}
tokenizer = AutoTokenizer.from_pretrained(args.base, revision=revision)
base = AutoModelForCausalLM.from_pretrained(args.base, revision=revision, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
model = PeftModel.from_pretrained(base, here).eval()
encoded = tokenizer.apply_chat_template(messages(context), tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors='pt').to('cuda')
if encoded['input_ids'].shape[1] > 2048:
    raise ValueError('Shorten context to the tested pilot limit of 2048 input tokens')
eos = model.generation_config.eos_token_id
stops = sorted(set([tokenizer.eos_token_id] + (eos if isinstance(eos, list) else [eos])))
with torch.inference_mode():
    output = model.generate(**encoded, do_sample=False, max_new_tokens=700,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=stops)
raw = tokenizer.decode(output[0,encoded['input_ids'].shape[1]:], skip_special_tokens=True)
print(raw)
try:
    errors = validate_output(parse_output(raw), context)
except (ValueError, TypeError) as error:
    errors = [str(error)]
if errors:
    print('Operator review required; invalid contract:', json.dumps(errors, ensure_ascii=False))
