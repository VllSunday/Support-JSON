"""Run one train-only, frozen few-shot baseline after the main GPU workflow exits."""
import json
import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--workflow',default='reports/workflow-v4.json')
parser.add_argument('--state',default='reports/few-shot-workflow.json')
parser.add_argument('--batch-size',type=int,default=4)
args=parser.parse_args()
STATE=ROOT/args.state
def status(stage,**values):
    STATE.write_text(json.dumps({'stage':stage,**values},ensure_ascii=False,indent=2),encoding='utf-8')
    print(stage,flush=True)

def main():
    if STATE.exists():
        raise ValueError('Do not launch duplicate GPU control')
    manifest=json.loads((ROOT/'reports/few-shot-manifest.json').read_text(encoding='utf-8'))
    assert manifest['frozen_before_test']
    status('waiting_for_main_workflow_exit')
    started=time.monotonic()
    while True:
        if time.monotonic()-started>3600:
            raise TimeoutError('Main workflow did not finish in one hour')
        try:
            main_state=json.loads((ROOT/args.workflow).read_text(encoding='utf-8'))
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(3)
            continue
        if main_state['stage']=='failed':
            raise RuntimeError('Main workflow failed; control not started')
        if main_state['stage']=='benchmark_complete':
            break
        time.sleep(3)
    status('running_base_four_shot_test')
    with (ROOT/'reports/base-test-fourshot.log').open('w',encoding='utf-8') as output:
        result=subprocess.run([sys.executable,'-X','utf8','scripts/run_transformers.py',
            '--dataset','data/pilot-v0.2/test.jsonl','--batch-size',str(args.batch_size),'--release-cache-after-batch','--few-shot-file','configs/few-shot-v1.json',
            '--output','reports/base-test-fourshot.jsonl'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Few-shot inference failed')
    from support_json.data import read_jsonl
    metrics=json.loads((ROOT/'reports/base-test-fourshot.metrics.json').read_text(encoding='utf-8'))
    run=json.loads((ROOT/'reports/base-test-fourshot.run.json').read_text(encoding='utf-8'))
    records=read_jsonl(ROOT/'reports/base-test-fourshot.jsonl')
    primary=json.loads((ROOT/'reports/test-comparison-final.json').read_text(encoding='utf-8'))
    assert metrics['dataset_fingerprint']==primary['dataset_fingerprint'] and metrics['missing_predictions']==0
    report={'n':metrics['n'],'dataset_fingerprint':metrics['dataset_fingerprint'],'few_shot_manifest':manifest,
            'field_only_accuracy':metrics['field_only_accuracy'],'strict_accuracy':metrics['accuracy'],
            'schema_and_invariants_rate':metrics['schema_and_invariants_rate'],'critical_recall':metrics['critical_recall'],
            'latency':{'batch_size':args.batch_size,'median_batch_seconds':statistics.median(r['elapsed_seconds'] for r in records),
                       'median_amortized_seconds':statistics.median(r['amortized_seconds'] for r in records),
                       'mean_input_tokens':statistics.mean(r['input_tokens'] for r in records)},
            'prompt_fingerprint':run['prompt_fingerprint'],
            'memory_policy':'Release unused CUDA cache after each batch; partial batch-8 attempt archived due GPU contention with a game. Same examples and generation policy; no quality-driven prompt/selection changes. Primary test base/LoRA both used batch 8; this additional control uses a smaller batch.',
            'note':'Additional control on same test with four predefined train examples. Different prompt from primary base/LoRA comparison. No tuning or model selection from these test results; reply quality/hallucinations not separately reviewed for this control.'}
    (ROOT/'reports/few-shot-control.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    status('complete')

if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error))
        raise
