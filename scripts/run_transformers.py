"""Local, unconstrained base/LoRA benchmark; writes each completed prediction."""
import argparse
import json
import importlib.metadata
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from support_json.contracts import messages
from support_json.data import digest, read_jsonl
from support_json.evaluation import score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/qwen3.5-4b')
    parser.add_argument('--adapter')
    parser.add_argument('--few-shot-file')
    parser.add_argument('--dataset', default='data/pilot-v0.2/validation.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-new-tokens', type=int, default=700)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--release-cache-after-batch', action='store_true')
    args = parser.parse_args()
    if Path(args.dataset).name == 'test.jsonl' and Path('reports/pretest-audit-pause.json').exists():
        raise RuntimeError('Pre-test audit pause: review validation before freezing final model')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Use a new output filename; predictions are immutable')
    rows = read_jsonl(args.dataset)
    examples = json.loads(Path(args.few_shot_file).read_text(encoding='utf-8')) if args.few_shot_file else None
    def make_messages(context):
        return messages(context, examples=examples)
    if args.limit:
        rows = rows[:args.limit]
    if not rows or not torch.cuda.is_available():
        raise RuntimeError('Nonempty data and CUDA are required')
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    configured_eos = model.generation_config.eos_token_id
    stop_ids = sorted(set([tokenizer.eos_token_id] + (
        configured_eos if isinstance(configured_eos, list) else [configured_eos])))
    print(json.dumps({'event': 'loaded', 'seconds': time.perf_counter()-started,
                      'model_class': type(model).__name__,
                      'gpu': torch.cuda.get_device_name(),
                      'allocated_gb': torch.cuda.memory_allocated()/2**30}), flush=True)
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start+args.batch_size]
            inputs = tokenizer.apply_chat_template(
                [make_messages(row['input']) for row in batch], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors='pt', padding=True).to('cuda')
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False,
                    max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=stop_ids)
            torch.cuda.synchronize()
            elapsed = round(time.perf_counter()-started, 3)
            for index, row in enumerate(batch):
                tokens = generated[index, inputs['input_ids'].shape[1]:].tolist()
                stops = [i for i, token in enumerate(tokens) if token in stop_ids]
                if stops:
                    tokens = tokens[:stops[0]+1]
                record = {'id': row['id'], 'output': tokenizer.decode(tokens, skip_special_tokens=True),
                          'elapsed_seconds': elapsed, 'batch_index': start//args.batch_size,
                          'batch_size': len(batch), 'amortized_seconds': elapsed/len(batch),
                          'input_tokens': int(inputs['attention_mask'][index].sum()),
                          'output_tokens': len(tokens),
                          'hit_token_limit': len(tokens) == args.max_new_tokens and tokens[-1] not in stop_ids}
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
                records.append(record)
            handle.flush()
            print(json.dumps({'event': 'batch', 'done': start+len(batch), 'total': len(rows),
                              'seconds': elapsed, 'batch_size': len(batch),
                              'allocated_gb': torch.cuda.memory_allocated()/2**30}), flush=True)
            if args.release_cache_after_batch:
                del generated, inputs
                torch.cuda.empty_cache()
    metrics, details = score(rows, records)
    metadata = {'model': args.model, 'adapter': args.adapter,
                'dataset_fingerprint': digest(rows), 'n': len(rows),
                'prompt_fingerprint': digest([make_messages(row['input']) for row in rows]),
                'settings': {'do_sample': False, 'max_new_tokens': args.max_new_tokens,
                             'enable_thinking': False, 'constrained_json': False,
                             'batch_size': args.batch_size, 'eos_token_ids': stop_ids,
                             'pad_token_id': tokenizer.pad_token_id},
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'peak_allocated_gb': torch.cuda.max_memory_allocated()/2**30}
    metadata['release_cache_after_batch'] = args.release_cache_after_batch
    metadata['packages'] = {name: importlib.metadata.version(name) for name in
                            ('transformers', 'peft', 'trl', 'soup-cli', 'flash-linear-attention', 'triton-windows')}
    if examples:
        metadata['few_shot_fingerprint'] = digest(examples)
        metadata['few_shot_file'] = args.few_shot_file
    manifest = Path('reports/weights-manifest.json')
    if manifest.exists():
        metadata['download_manifest'] = json.loads(manifest.read_text(encoding='utf-8'))
    for suffix, value in [('run', metadata), ('metrics', metrics), ('details', details)]:
        output.with_suffix('.'+suffix+'.json').write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
