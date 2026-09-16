"""Freeze four train-only examples before final test: no validation/test retrieval."""
import json
from pathlib import Path
from support_json.data import read_jsonl, digest

source=read_jsonl('data/pilot-v0.2/train.jsonl')
wanted=[('refund_recent_charge','true'),('refund_recent_charge','false'),
        ('cancel_before_renewal','unknown'),('service_mass_incident','true')]
rows=[]
for family,state in wanted:
    rows.append(next(r for r in source if r['metadata']['scenario_family_id']==family
                     and r['metadata']['state']==state and r['target']['sentiment']=='neutral'))
examples=[{'input':r['input'],'result':r['target']} for r in rows]
destination=Path('configs/few-shot-v1.json')
if destination.exists():
    raise ValueError('Few-shot examples already frozen')
destination.write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf-8')
Path('reports/few-shot-manifest.json').write_text(json.dumps({'source':'data/pilot-v0.2/train.jsonl',
    'source_ids':[r['id'] for r in rows],'fingerprint':digest(examples),'n':len(examples),
    'frozen_before_test':not Path('reports/base-test-final.jsonl').exists(),
    'note':'Single predefined control prompt; no tuning on test results.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'n':len(examples),'frozen_before_test':not Path('reports/base-test-final.jsonl').exists()}))
