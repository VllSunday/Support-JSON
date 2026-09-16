"""Sequential local GPU workflow; select on validation before opening final test."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/Scripts/python.exe'
SOUP = ROOT / '.venv/Scripts/soup.exe'
PILOT = 'checkpoints/support-json-pilot-v0.3-native-liger'
EXPANDED = 'checkpoints/support-json-expanded-v0.4-1024'
STATE = ROOT / 'reports/workflow-v4.json'


def state(stage, **values):
    report = {'stage': stage, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **values}
    STATE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


def command(log, *arguments):
    with (ROOT / log).open('w', encoding='utf-8') as output:
        result = subprocess.run([str(x) for x in arguments], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed stage; inspect {log}')


def rank(comparison):
    # All metrics and tie-breaking chosen before test. Text audit is separate.
    accuracy = comparison['accuracy']
    return (accuracy['action']['lora'], comparison['json']['schema_and_invariants_rate']['lora'],
            accuracy['priority']['lora'], accuracy['category']['lora'])


def main():
    if STATE.exists():
        raise ValueError('Workflow already exists; do not launch duplicate GPU work')
    state('waiting_for_first_validation_comparison')
    started = time.monotonic()
    comparison_path = ROOT / 'reports/validation-comparison-v3.json'
    while not comparison_path.exists():
        if time.monotonic() - started > 1200:
            raise TimeoutError('First comparison did not complete within 20 minutes')
        time.sleep(2)
    first = json.loads(comparison_path.read_text(encoding='utf-8'))
    assert first['n'] == 96
    state('training_expanded_1024', first_validation_rank=rank(first))
    command('reports/lora-expanded-1024-training.log', SOUP, 'train', '--config', 'configs/soup-expanded-1024.yaml', '--yes')
    command('reports/record-expanded-training.log', PYTHON, '-X', 'utf8', 'scripts/record_training.py',
            '--adapter', EXPANDED, '--config', 'configs/soup-expanded-1024.yaml',
            '--dataset', 'data/expanded-v0.4/train-1024.chatml.jsonl', '--output', 'reports/training-expanded-v4.json')
    state('expanded_validation')
    command('reports/lora-validation-v4.log', PYTHON, '-X', 'utf8', 'scripts/run_transformers.py',
            '--batch-size', '4', '--adapter', EXPANDED, '--output', 'reports/lora-validation-v4.jsonl')
    command('reports/validation-comparison-v4.log', PYTHON, '-X', 'utf8', 'scripts/compare_runs.py',
            '--base', 'reports/base-validation-v3.jsonl', '--lora', 'reports/lora-validation-v4.jsonl',
            '--output', 'reports/validation-comparison-v4.json')
    second = json.loads((ROOT / 'reports/validation-comparison-v4.json').read_text(encoding='utf-8'))
    selected = EXPANDED if rank(second) >= rank(first) else PILOT
    decision = {'adapter': selected, 'selected_on': 'validation_only_before_final_test',
                'selection_order': ['action_accuracy', 'schema_and_invariants_rate', 'priority_accuracy', 'category_accuracy'],
                'pilot_validation_rank': rank(first), 'expanded_validation_rank': rank(second),
                'training_rows': 1024 if selected == EXPANDED else 192,
                'all_10000_candidates_trained': False}
    (ROOT / 'reports/final-selection.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    state('final_test_base', **decision)
    command('reports/base-test-final.log', PYTHON, '-X', 'utf8', 'scripts/run_transformers.py',
            '--dataset', 'data/pilot-v0.2/test.jsonl', '--batch-size', '4', '--output', 'reports/base-test-final.jsonl')
    state('final_test_lora', **decision)
    command('reports/lora-test-final.log', PYTHON, '-X', 'utf8', 'scripts/run_transformers.py',
            '--dataset', 'data/pilot-v0.2/test.jsonl', '--batch-size', '4', '--adapter', selected,
            '--output', 'reports/lora-test-final.jsonl')
    command('reports/test-comparison-final.log', PYTHON, '-X', 'utf8', 'scripts/compare_runs.py',
            '--base', 'reports/base-test-final.jsonl', '--lora', 'reports/lora-test-final.jsonl',
            '--dataset', 'data/pilot-v0.2/test.jsonl',
            '--output', 'reports/test-comparison-final.json')
    state('benchmark_complete', **decision)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('failed', error=str(error))
        raise
