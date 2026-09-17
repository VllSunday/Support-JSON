"""Explicit artifact allowlist: no environments, caches, base weights or unrelated files."""
import hashlib
import argparse
import json
import re
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--version', default='')
args = parser.parse_args()
if args.version and not re.fullmatch(r'v[0-9]+', args.version):
    raise ValueError('Version must be v followed by digits')
selected = set()
for name in ('README.md', 'PROJECT_STATUS.md', 'RESULTS.md', 'pyproject.toml', '.gitignore', 'requirements.training.lock.txt',
             'requirements.inference.txt', 'LICENSE', 'LICENSE.dataset.txt', 'NOTICE'):
    selected.add(root / name)
for directory in ('src/support_json', 'scripts', 'tests', 'web', 'docs', 'configs', 'assets',
                  'data/pilot-v0.2', 'data/pilot-v0.3', 'data/expanded-v0.4', 'data/diverse-v0.5',
                  'output/model/support-json-qwen3.5-4b-lora', 'output/presentation'):
    for file in (root / directory).rglob('*'):
        if file.is_file() and '__pycache__' not in file.parts and file.suffix not in {'.pyc', '.tmp'}:
            selected.add(file)
patterns = ('hf-*.json', 'demo-video*.json', 'ui-dark-checks.json', 'base-validation-v3.*', 'lora-validation-v3.*', 'lora-validation-v4.*', 'lora-validation-v5.*', 'base-test-fourshot.*',
            'few-shot-control.json', 'few-shot-manifest.json', 'fourshot-memory-restart.json', 'base-test-fourshot-aborted-batch8.*',
            '*test-final.*', '*comparison-v3.*', '*comparison-v4.*', '*comparison-v5.*', '*comparison-final.*',
            'training-*.json', 'loss-mask-*.json', 'response-review*', 'final-selection.json',
            'business-results.json', 'runtime-contention.json', 'error-analysis-final.md', 'presentation-validation.json', 'presentation-visual-review.json', 'ui-checks.json', 'test-uncertainty.json', 'demo-checks.json', 'demo-behavior-review.json', 'demo-screenshot.png', 'demo-result-screenshot.png',
            'weights-manifest.json', 'model-config-manifest.json', 'export-checks.json', 'portable-inference.log', 'demo-checks.log', 'fla-preflight.json', 'liger-preflight.json',
            'lora-pilot-v3-liger-training.log', 'lora-expanded-1024-training.log', 'lora-diverse-512-training.log', 'tests-v4.log',
            'pretest-audit-pause.resolved.json', 'preliminary-selection-v4.json')
for pattern in patterns:
    selected.update(file for file in (root / 'reports').glob(pattern) if file.is_file())
required = ('RESULTS.md', 'reports/test-comparison-final.json', 'reports/response-review-summary.json',
            'reports/demo-checks.json', 'output/model/support-json-qwen3.5-4b-lora/adapter_model.safetensors')
if any(not (root / name).is_file() for name in required):
    raise ValueError('Complete the actual required artifacts before packaging')
if not list((root/'output/presentation').glob('*.pptx')):
    raise ValueError('Presentation not created')
destination = root / 'output/submission'
destination.mkdir(exist_ok=True)
suffix = f'-{args.version}' if args.version else ''
archive = destination / f'Support_JSON_Capstone{suffix}.zip'
if archive.exists():
    raise ValueError('Preserve prior submission; use a new version')
files = sorted(selected)
manifest = {'excludes': ['base weights', '.venv', 'caches', 'unrelated output/hairstyles'],
            'files_sha256': {file.relative_to(root).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
    for file in files:
        bundle.write(file, file.relative_to(root).as_posix())
    bundle.writestr('SUBMISSION_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
with zipfile.ZipFile(archive) as bundle:
    if bundle.testzip() is not None:
        raise ValueError('Archive integrity test failed')
report = {'archive': str(archive), 'bytes': archive.stat().st_size, 'file_count': len(files)+1,
          'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'integrity_checked': True}
(destination/f'submission-manifest{suffix}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
