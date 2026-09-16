"""Blinded qualitative-review package, created only after final selection/test."""
import json
import random
from pathlib import Path
from support_json.data import read_jsonl, write_jsonl, digest

destination = Path('reports/response-review-blind.jsonl')
if destination.exists():
    raise ValueError('Review package already exists')
selection = json.loads(Path('reports/final-selection.json').read_text(encoding='utf-8'))
assert selection['selected_on'] == 'validation_only_before_final_test'
comparison = json.loads(Path('reports/test-comparison-final.json').read_text(encoding='utf-8'))
assert comparison['n'] == 108
rows = read_jsonl('data/pilot-v0.2/test.jsonl')
predictions = {label: {r['id']: r['output'] for r in read_jsonl(f'reports/{label}-test-final.jsonl')}
               for label in ('base', 'lora')}
chosen = {}
for row in rows:
    key = (row['metadata']['scenario_family_id'], row['metadata']['state'])
    chosen.setdefault(key, []).append(row)
rng = random.Random(914)
samples, mapping = [], {}
for index, key in enumerate(sorted(chosen)):
    row = rng.choice(chosen[key])
    labels = ['base', 'lora']
    rng.shuffle(labels)
    sample_id = f'R{index+1:02d}'
    samples.append({'sample_id': sample_id, 'input': row['input'],
                    'A': predictions[labels[0]][row['id']], 'B': predictions[labels[1]][row['id']]})
    mapping[sample_id] = {'record_id': row['id'], 'family': key[0], 'state': key[1],
                           'A': labels[0], 'B': labels[1]}
write_jsonl(destination, samples)
Path('reports/response-review-mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'n_pairs': len(samples), 'test_cases': comparison['n'], 'selection': 'one per family/state, random tone',
                  'test_dataset_fingerprint': comparison['dataset_fingerprint'], 'sample_fingerprint': digest(samples)}))
