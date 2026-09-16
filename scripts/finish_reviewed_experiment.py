"""Resume after validation review. Safety constraint and ranking are fixed before test."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/workflow-v5.json'
DIVERSE='checkpoints/support-json-diverse-v0.5-512'

def state(stage,**values):
    report={'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**values}
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(STATE)
    print(json.dumps(report,ensure_ascii=False),flush=True)

def command(log,*arguments):
    with (ROOT/log).open('w',encoding='utf-8') as output:
        result=subprocess.run([str(x) for x in arguments],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Failed stage; inspect {log}')

def rank(result):
    accuracy=result['accuracy']
    return (accuracy['action']['lora'],result['json']['schema_and_invariants_rate']['lora'],
            accuracy['priority']['lora'],accuracy['category']['lora'])

def main():
    if STATE.exists():
        raise ValueError('Reviewed workflow already started')
    state('diverse_training_complete_validation_pending')
    command('reports/record-diverse-training.log',sys.executable,'-X','utf8','scripts/record_training.py',
        '--adapter',DIVERSE,'--config','configs/soup-diverse-512.yaml','--dataset','data/diverse-v0.5/train.chatml.jsonl',
        '--output','reports/training-diverse-v5.json')
    state('diverse_validation')
    command('reports/lora-validation-v5.log',sys.executable,'-X','utf8','scripts/run_transformers.py',
        '--batch-size','4','--adapter',DIVERSE,'--output','reports/lora-validation-v5.jsonl')
    command('reports/validation-comparison-v5.log',sys.executable,'-X','utf8','scripts/compare_runs.py',
        '--base','reports/base-validation-v3.jsonl','--lora','reports/lora-validation-v5.jsonl',
        '--output','reports/validation-comparison-v5.json')
    candidates=[]
    for version,adapter,n in [('v3','checkpoints/support-json-pilot-v0.3-native-liger',192),
                              ('v4','checkpoints/support-json-expanded-v0.4-1024',1024),('v5',DIVERSE,512)]:
        comparison=json.loads((ROOT/f'reports/validation-comparison-{version}.json').read_text(encoding='utf-8'))
        metrics=json.loads((ROOT/f'reports/lora-validation-{version}.metrics.json').read_text(encoding='utf-8'))
        base_metrics=json.loads((ROOT/'reports/base-validation-v3.metrics.json').read_text(encoding='utf-8'))
        safe=metrics['critical_recall']['value']>=base_metrics['critical_recall']['value']
        candidates.append({'version':version,'adapter':adapter,'training_rows':n,'rank':rank(comparison),
                           'critical_recall':metrics['critical_recall'],'passes_validation_critical_constraint':safe})
    eligible=[candidate for candidate in candidates if candidate['passes_validation_critical_constraint']]
    if not eligible:
        raise ValueError('No adapter preserves baseline critical recall; inspect data before test')
    selected=max(eligible,key=lambda candidate:candidate['rank'])
    decision={**selected,'selected_on':'validation_only_before_final_test',
              'selection_order':['action_accuracy','schema_and_invariants_rate','priority_accuracy','category_accuracy'],
              'constraint':'validation critical recall >= base; chosen before diverse validation/test',
              'all_candidates':candidates,'all_10000_candidates_trained':False,
              'training_report':f"reports/training-{'expanded-v4' if selected['version']=='v4' else 'diverse-v5' if selected['version']=='v5' else 'pilot-v3'}.json"}
    preliminary=ROOT/'reports/final-selection.json'
    if preliminary.exists():
        preliminary.rename(ROOT/'reports/preliminary-selection-v4.json')
    preliminary.write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8')
    marker=ROOT/'reports/pretest-audit-pause.json'
    marker.rename(ROOT/'reports/pretest-audit-pause.resolved.json')
    state('final_test_base',**decision)
    command('reports/base-test-final.log',sys.executable,'-X','utf8','scripts/run_transformers.py',
        '--dataset','data/pilot-v0.2/test.jsonl','--batch-size','8','--output','reports/base-test-final.jsonl')
    state('final_test_lora',**decision)
    command('reports/lora-test-final.log',sys.executable,'-X','utf8','scripts/run_transformers.py',
        '--dataset','data/pilot-v0.2/test.jsonl','--batch-size','8','--adapter',selected['adapter'],
        '--output','reports/lora-test-final.jsonl')
    command('reports/test-comparison-final.log',sys.executable,'-X','utf8','scripts/compare_runs.py',
        '--base','reports/base-test-final.jsonl','--lora','reports/lora-test-final.jsonl',
        '--dataset','data/pilot-v0.2/test.jsonl','--output','reports/test-comparison-final.json')
    state('benchmark_complete',**decision)

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        state('failed',error=str(error))
        raise
