"""Verify the delivered adapter, resources and actual standalone smoke response."""
import hashlib
import importlib.util
import json
from pathlib import Path

folder = Path('output/model/support-json-qwen3.5-4b-lora')
manifest = json.loads((folder/'delivery-manifest.json').read_text(encoding='utf-8'))
training = json.loads(Path(manifest['selection']['training_report']).read_text(encoding='utf-8'))
for name, expected in manifest['files_sha256'].items():
    assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == expected, name
assert manifest['files_sha256']['adapter_model.safetensors'] == training['adapter_sha256']
config = json.loads((folder/'adapter_config.json').read_text(encoding='utf-8'))
assert config['base_model_name_or_path'] == 'Qwen/Qwen3.5-4B'
assert config['revision'] == training['base_revision']
spec = importlib.util.spec_from_file_location('delivered_contracts', folder/'contracts.py')
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)
raw = Path('reports/portable-inference.log').read_text(encoding='utf-8').strip()
context = json.loads((folder/'demo_context.json').read_text(encoding='utf-8'))
output = contracts.parse_output(raw)
errors = contracts.validate_output(output, context)
assert not errors, errors
report = {'verified_files': len(manifest['files_sha256']), 'adapter_matches_training': True,
          'base_revision_pinned': True, 'standalone_cuda_inference_completed': True,
          'standalone_response_contract_valid': True, 'actual_output': output,
          'note': 'One training-derived smoke context; this is export verification, not another quality benchmark.'}
Path('reports/export-checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
