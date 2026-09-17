"""Package a portable local LoRA delivery from real, completed experiment outputs."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
selection = json.loads((ROOT / 'reports/final-selection.json').read_text(encoding='utf-8'))
training = json.loads((ROOT / selection['training_report']).read_text(encoding='utf-8'))
comparison = json.loads((ROOT / 'reports/test-comparison-final.json').read_text(encoding='utf-8'))
control = json.loads((ROOT / 'reports/few-shot-control.json').read_text(encoding='utf-8'))
source = ROOT / selection['adapter']
destination = ROOT / 'output/model/support-json-qwen3.5-4b-lora'
if destination.exists():
    raise ValueError('Delivery already exists; use a new version instead of overwriting')
revision = json.loads((ROOT / 'reports/weights-manifest.json').read_text(encoding='utf-8'))['revision']
destination.mkdir(parents=True)
shutil.copyfile(source / 'adapter_model.safetensors', destination / 'adapter_model.safetensors')
shutil.copyfile(ROOT / 'models/qwen3.5-4b/LICENSE', destination / 'LICENSE.base.txt')
resources = ROOT / 'src/support_json/resources'
for name in ('system_prompt.txt', 'output.schema.json'):
    shutil.copyfile(resources / name, destination / name)
contract_source = (ROOT / 'src/support_json/contracts.py').read_text(encoding='utf-8')
contract_source = contract_source.replace('from importlib.resources import files', 'from pathlib import Path')
contract_source = contract_source.replace('RESOURCES = files("support_json").joinpath("resources")', 'RESOURCES = Path(__file__).resolve().parent')
(destination / 'contracts.py').write_text(contract_source, encoding='utf-8')
from support_json.data import read_jsonl
demo_context = read_jsonl(ROOT / 'data/pilot-v0.2/train.jsonl')[0]['input']
(destination / 'demo_context.json').write_text(json.dumps(demo_context, ensure_ascii=False, indent=2), encoding='utf-8')
prediction_source = '''"""Standalone local inference; no customer text is sent to a model API."""
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
'''
(destination / 'predict.py').write_text(prediction_source, encoding='utf-8')
config = json.loads((source / 'adapter_config.json').read_text(encoding='utf-8'))
config['base_model_name_or_path'] = 'Qwen/Qwen3.5-4B'
config['revision'] = revision
(destination / 'adapter_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
accuracy = comparison['accuracy']
lines = [
    '---', 'base_model: Qwen/Qwen3.5-4B', 'library_name: peft', 'language:', '- ru',
    'tags:', '- lora', '- support', '- structured-output', 'pipeline_tag: text-generation', '---', '',
    '# Support-JSON: Russian SaaS support LoRA', '',
    'Local capstone adapter for a support operator: customer request, company policies and sourced facts → JSON decision and draft reply.', '',
    'This folder is a draft local release, not a published Hugging Face repository. Adapter/data publication license is pending the project author’s decision. Base model license: Apache-2.0.', '',
    '## Scope and training', '',
    f"Text-only Qwen3.5-4B, pinned revision `{revision}`. BF16 LoRA r=16 / alpha=32, all-linear, {training['epochs']:g} epochs, seed=42, Soup 0.75.0 / Transformers / PEFT / TRL / Liger. Selected adapter trained on {selection['training_rows']} synthetic examples.", '',
    'A second dataset contains 10,000 candidates. We did not train all 10,000. Numeric variations and emotional variants are not independent real conversations. No external customer dataset is included.', '',
    '## Evaluation', '',
    f"Same {comparison['n']} held-out synthetic examples and prompt for base and LoRA, greedy generation, non-thinking, 700 token limit, batch size {comparison['generation_settings']['batch_size']}. Selection was made on separate validation before test.", '',
    '| Strict task metric | Base | LoRA |', '|---|---:|---:|',
]
for field in ('category', 'priority', 'sentiment', 'action', 'recommended_action', 'human_escalation'):
    lines.append(f"| {field} | {accuracy[field]['base']:.1%} | {accuracy[field]['lora']:.1%} |")
lines += ['', 'Field-only classification metrics (unrelated format errors do not zero a correct field):', '',
          '| Metric | Base | LoRA |', '|---|---:|---:|']
for field in ('category', 'priority', 'action'):
    values = comparison['field_only_accuracy'][field]
    lines.append(f"| {field} | {values['base']:.1%} | {values['lora']:.1%} |")
lines += ['', 'Additional base control with four fixed train examples (different prompt; no test tuning):', '',
          '| Field-only metric | Base + 4 examples | LoRA, ordinary prompt |', '|---|---:|---:|']
for field in ('category', 'priority', 'action'):
    lines.append(f"| {field} | {control['field_only_accuracy'][field]:.1%} | {comparison['field_only_accuracy'][field]['lora']:.1%} |")
lines += ['', 'Invalid JSON/schema/invariants fail all task fields. This strict score includes format failures; it is not classification accuracy among valid outputs only.', '',
          'Response quality and hallucinations require separate content review. Human review is not claimed. Refer to the capstone report for any explicitly labeled AI-assisted audit.', '',
          'The completed project reports include a blinded AI-assisted review of 36 paired test replies with individual explanations. This is not independent human review; model mistakes remain. See RESULTS.md and reports/response-review-summary.json for measured quality and hallucination rates. Additional local prototype checks and their semantic failures are reported separately in reports/demo-checks.json and reports/demo-behavior-review.json.', '',
          '## Limits', '',
          'Seven SaaS categories, a small held-out policy suite, and mostly templated Russian. Not a guarantee for arbitrary company rules. Larger stronger models and production messages have not been compared. Few-shot control reply quality has not been separately reviewed. Do not execute refunds, cancellation or account changes automatically. Keep operator review, especially for conflicts, missing system facts and malformed output.', '',
          '## Local use', '',
          'From the capstone project root with its configured environment:', '', '```powershell',
          '.\\.venv\\Scripts\\python.exe -X utf8 -m support_json.demo --adapter output/model/support-json-qwen3.5-4b-lora', '```', '',
          'Open http://127.0.0.1:7860. The existing pinned base weights must be in models/qwen3.5-4b. Use the exact system prompt and output contract from this project; a naked request without policies is not the trained task.', '',
          'Standalone use without importing the capstone package:', '', '```powershell',
          'python output/model/support-json-qwen3.5-4b-lora/predict.py --base models/qwen3.5-4b',
          '# Replace demo_context.json with your message, policies and sourced facts.',
          '```', '',
          'predict.py downloads the pinned base when --base is omitted. It includes contract checking and never executes operations. User reports are not system confirmations. The sample context is synthetic. Install the tested Torch CUDA, Transformers, PEFT and jsonschema environment first.', '',
          'Portable Python loading (downloads pinned base when missing):', '', '```python',
          'from transformers import AutoTokenizer, AutoModelForCausalLM', 'from peft import PeftModel', 'import torch',
          f'revision = "{revision}"',
          'tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", revision=revision)',
          'base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", revision=revision, dtype=torch.bfloat16, device_map="cuda")',
          'model = PeftModel.from_pretrained(base, "output/model/support-json-qwen3.5-4b-lora").eval()',
          '# Build messages with support_json.contracts.messages(context).',
          '# Apply native chat template with enable_thinking=False.',
          '# Generate greedily; stop on model EOS and tokenizer EOS.', '```', '',
          'Requires the project’s tested Transformers 5.17.0 / PEFT 0.21.0 / PyTorch CUDA environment. BF16 text weights take about 7.83 GiB allocated by PyTorch before inference buffers; total desktop GPU usage is higher. RTX 5070 Ti 16GB was tested; other devices are not certified.', '',
          '## Reproducibility', '',
          'The project includes source code, frozen datasets, SHA256 manifests, training logs, raw predictions, prompt fingerprints and paired metrics. Base weights are not duplicated in this adapter folder. No remote upload is performed by packaging.', '']
(destination / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
manifest = {'source_adapter': selection['adapter'], 'base_revision': revision,
            'selection': selection, 'test_comparison': comparison,
            'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir()}}
(destination / 'delivery-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(str(destination))
