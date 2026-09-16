"""Record actual checkpoint evidence, distinct from preflight checks."""
import argparse
import hashlib
import json
import math
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--adapter', required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
adapter = Path(args.adapter)
states = list(adapter.glob('checkpoint-*/trainer_state.json'))
if not states:
    raise ValueError('No actual completed trainer checkpoint')
state = json.loads(max(states, key=lambda p: int(p.parent.name.split('-')[-1])).read_text(encoding='utf-8'))
if state['global_step'] != state['max_steps']:
    raise ValueError('Training has not reached its scheduled final step')
logs = [row for row in state['log_history'] if 'loss' in row]
assert logs and all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in logs)
weights = adapter / 'adapter_model.safetensors'
config = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
training_config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
report = {'completed': True, 'adapter': str(adapter), 'global_step': state['global_step'],
          'epochs': state['epoch'], 'first_logged_loss': logs[0]['loss'], 'last_logged_loss': logs[-1]['loss'],
          'micro_batch_size': training_config['training']['batch_size'],
          'gradient_accumulation_steps': training_config['training']['gradient_accumulation_steps'],
          'loss_and_gradient_norm_finite': True, 'log_history': state['log_history'],
          'adapter_bytes': weights.stat().st_size, 'adapter_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
          'lora_r': config['r'], 'lora_alpha': config['lora_alpha'],
          'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
          'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
          'base_revision': json.loads(Path('reports/weights-manifest.json').read_text(encoding='utf-8'))['revision'],
          'note': 'Training loss is not held-out response quality. No remote publication performed.'}
destination = Path(args.output)
if destination.exists():
    raise ValueError('Use a new immutable report path')
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: report[k] for k in ('completed', 'global_step', 'first_logged_loss', 'last_logged_loss', 'adapter_bytes')}))
