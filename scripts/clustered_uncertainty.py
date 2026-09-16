"""Paired, family-clustered bootstrap; emotional variants are not independent trials."""
import json
import random
from collections import defaultdict
from pathlib import Path
from support_json.contracts import parse_output, validate_output
from support_json.data import read_jsonl

rows = read_jsonl('data/pilot-v0.2/test.jsonl')
predictions = {label: {row['id']: row['output'] for row in read_jsonl(f'reports/{label}-test-final.jsonl')}
               for label in ('base', 'lora')}
groups = defaultdict(list)
for row in rows:
    scores = {}
    for label in ('base', 'lora'):
        try:
            output = parse_output(predictions[label][row['id']])
            valid = not validate_output(output, row['input'])
        except (ValueError, TypeError):
            output, valid = {}, False
        scores[label] = {'schema_and_invariants': int(valid)}
        for field in ('category', 'priority', 'action'):
            value = output.get(field) if isinstance(output, dict) else None
            scores[label][field] = int(type(value) is type(row['target'][field]) and value == row['target'][field])
    groups[row['metadata']['scenario_family_id']].append(scores)
names = sorted(groups)
rng = random.Random(42)
fields = ('category', 'priority', 'action', 'schema_and_invariants')
samples = {field: [] for field in fields}
for _ in range(5000):
    selected = [item for name in rng.choices(names, k=len(names)) for item in groups[name]]
    for field in fields:
        samples[field].append(sum(s['lora'][field]-s['base'][field] for s in selected)/len(selected))
report = {'clusters': len(names), 'resamples': 5000, 'seed': 42,
          'method': 'paired percentile bootstrap, resampling whole scenario families', 'delta_pp_95_interval': {},
          'note': 'Only nine authored synthetic policy families and one test company. Intervals describe this suite, not arbitrary products/company traffic. Tones and states within a family are kept together.'}
for field in fields:
    values = sorted(samples[field])
    report['delta_pp_95_interval'][field] = [values[124]*100, values[4874]*100]
path = Path('reports/test-uncertainty.json')
if path.exists():
    raise ValueError('Use a new version')
path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
