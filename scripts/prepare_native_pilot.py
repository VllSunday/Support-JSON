"""Pin native Soup configuration and verify its real assistant loss mask."""
import json
import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer
from soup_cli.config.schema import SoupConfig
from soup_cli.data.sft_format import build_format_row
from support_json.data import read_jsonl

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./data/pilot-v0.3/train.chatml.jsonl')
parser.add_argument('--output-config', default='configs/soup-pilot-native.yaml')
parser.add_argument('--experiment', default='support-json-pilot-v0.3-native-liger')
parser.add_argument('--report', default='reports/loss-mask-native.json')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--accumulation', type=int, default=8)
args = parser.parse_args()
config = yaml.safe_load(Path('configs/soup-pilot.yaml').read_text(encoding='utf-8'))
config['base'] = './models/qwen3.5-4b'
config['backend'] = 'transformers'
config['data']['train'] = args.data
config['data']['max_length'] = 2048
config['experiment_name'] = args.experiment
config['output'] = './checkpoints'
config['training']['logging_steps'] = 1
config['training']['epochs'] = args.epochs
config['training']['batch_size'] = args.batch_size
config['training']['gradient_accumulation_steps'] = args.accumulation
config['training']['save_steps'] = 100
config['training']['train_on_eot'] = True
config['training']['use_liger'] = True
tokenizer = AutoTokenizer.from_pretrained(config['base'])
# The shipped template renders an empty thinking span for assistant answers
# without reasoning_content, matching non-thinking inference. Soup's override
# validator rejects native templates with macros; leave the shipped one intact.
cfg = SoupConfig.model_validate(config)
formatter = build_format_row(tokenizer, cfg.data, training_cfg=cfg.training)
counts = []
for row in read_jsonl(cfg.data.train):
    encoded = formatter(row)
    prefix = tokenizer.apply_chat_template(row['messages'][:-1], tokenize=True,
                                          add_generation_prompt=False, return_dict=False)
    assert all(value == -100 for value in encoded['labels'][:len(prefix)])
    selected = [label for label in encoded['labels'] if label != -100]
    assert selected and tokenizer.eos_token_id in selected
    assert row['messages'][-1]['content'] in tokenizer.decode(selected)
    assert len(encoded['input_ids']) < cfg.data.max_length
    counts.append(len(selected))
Path(args.output_config).write_text(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
report = {'n': len(counts), 'assistant_label_tokens_min': min(counts),
          'assistant_label_tokens_max': max(counts), 'prompt_fully_masked': True,
          'all_assistant_answers_and_eos_present': True, 'max_length': cfg.data.max_length,
          'backend': cfg.backend, 'training_not_yet_run': True}
Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
