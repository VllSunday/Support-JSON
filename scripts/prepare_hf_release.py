"""Stage only explicitly selected public artifacts and verify their provenance."""
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import DatasetCard, ModelCard

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'.build/hf-release'
SOURCE = ROOT/'output/model/support-json-qwen3.5-4b-lora'

def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def tree(source, target):
    for path in source.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc','.tmp'}:
            copy(path, target/path.relative_to(source))

def prepare():
    copy(ROOT/'docs/HF_MODEL_CARD.md', SOURCE/'README.md')
    copy(ROOT/'LICENSE', SOURCE/'LICENSE')
    copy(ROOT/'NOTICE', SOURCE/'NOTICE')
    manifest = json.loads((SOURCE/'delivery-manifest.json').read_text(encoding='utf-8'))
    training = json.loads((ROOT/manifest['selection']['training_report']).read_text(encoding='utf-8'))
    actual = hashlib.sha256((SOURCE/'adapter_model.safetensors').read_bytes()).hexdigest()
    assert actual == training['adapter_sha256']
    names = set(manifest['files_sha256']) | {'LICENSE','NOTICE'}
    manifest['files_sha256'] = {name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in sorted(names)}
    (SOURCE/'delivery-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for name in [*names, 'delivery-manifest.json']:
        copy(SOURCE/name, STAGE/'model'/name)
    for directory in ('src/support_json','web'):
        tree(ROOT/directory, STAGE/'model/app'/directory)
    for name in ('pyproject.toml','requirements.inference.txt','LICENSE','LICENSE.dataset.txt','NOTICE'):
        copy(ROOT/name, STAGE/'model/app'/name)
    copy(ROOT/'data/pilot-v0.2/validation.jsonl', STAGE/'model/app/data/pilot-v0.2/validation.jsonl')
    tree(ROOT/'assets', STAGE/'model/assets')
    for name, source in {
        'RESULTS.md':'RESULTS.md', 'TZ_AUDIT.md':'docs/TZ_AUDIT.md',
        'EVALUATION.md':'docs/EVALUATION.md', 'DEMO_VIDEO.md':'docs/DEMO_VIDEO.md',
        'error-analysis-final.md':'reports/error-analysis-final.md',
        'test-comparison-final.json':'reports/test-comparison-final.json',
        'base-test-final.jsonl':'reports/base-test-final.jsonl',
        'lora-test-final.jsonl':'reports/lora-test-final.jsonl',
        'response-review-summary.json':'reports/response-review-summary.json',
        'response-review-ratings.jsonl':'reports/response-review-ratings.jsonl',
        'response-review-blind.jsonl':'reports/response-review-blind.jsonl',
        'ui-dark-checks.json':'reports/ui-dark-checks.json',
        'demo-checks.json':'reports/demo-checks.json',
        'demo-behavior-review.json':'reports/demo-behavior-review.json',
        'demo-video.json':'reports/demo-video.json',
        'training-diverse-v5.json':'reports/training-diverse-v5.json',
    }.items():
        copy(ROOT/source, STAGE/'model/evaluation'/name)
    copy(ROOT/'output/presentation/Support_JSON_Capstone-v2.pptx', STAGE/'model/artifacts/Support_JSON_Capstone-v2.pptx')
    copy(ROOT/'docs/HF_DATASET_CARD.md', STAGE/'dataset/README.md')
    for path in (STAGE/'dataset/documentation').glob('*.md'):
        copy(ROOT/'docs'/path.name, path)
    for kind, cls in [('model',ModelCard),('dataset',DatasetCard)]:
        card = cls.load(STAGE/kind/'README.md')
        assert card.data.license == ('apache-2.0' if kind == 'model' else 'cc-by-4.0')
    assert not list(STAGE.rglob('*.env'))
    assert not list(STAGE.rglob('*.bin'))
    print(json.dumps({'model_files':len(list((STAGE/'model').rglob('*'))), 'adapter_sha256':actual,
                      'dataset_card_loaded':True,'model_card_loaded':True}))

if __name__ == '__main__':
    prepare()
