"""Aggregate explicitly AI-assisted ratings; never fill human_evaluation."""
import json
from collections import defaultdict
from pathlib import Path
from support_json.data import read_jsonl, digest

samples = read_jsonl('reports/response-review-blind.jsonl')
mapping = json.loads(Path('reports/response-review-mapping.json').read_text(encoding='utf-8'))
ratings = read_jsonl('reports/response-review-ratings.jsonl')
expected = {(sample['sample_id'], label) for sample in samples for label in ('A', 'B')}
seen = set()
by_model = defaultdict(list)
for rating in ratings:
    key = (rating['sample_id'], rating['label'])
    if key not in expected or key in seen:
        raise ValueError('Duplicate or unknown review item')
    seen.add(key)
    if type(rating['response_quality']) is not int or not 1 <= rating['response_quality'] <= 5:
        raise ValueError('Quality must be integer 1–5')
    if rating['hallucination'] is not None and type(rating['hallucination']) is not bool:
        raise ValueError('Hallucination must be boolean or unassessable null')
    if rating['hallucination'] is None and rating['response_quality'] != 1:
        raise ValueError('No usable reply requires quality=1')
    if not isinstance(rating.get('notes'), str) or not rating['notes'].strip():
        raise ValueError('Every rating needs a grounded explanation')
    model = mapping[rating['sample_id']][rating['label']]
    by_model[model].append(rating)
if seen != expected:
    raise ValueError('Finish every item in the frozen paired sample')
summary = {'method': 'blinded paired AI-assisted rubric audit', 'reviewer': 'Codex AI assistant',
           'independent_human_review': False, 'n_pairs': len(samples), 'sample_fingerprint': digest(samples),
           'ratings_fingerprint': digest(ratings), 'models': {},
           'selection': 'one random tone per scenario family/state, seed=914; labels A/B randomly ordered',
           'note': 'Qualitative rubric audit on a small synthetic suite. Not an independent human benchmark. Hallucination denominator excludes unusable reply; quality includes every selected output. No post-test model/prompt tuning.'}
for model in ('base', 'lora'):
    reviews = by_model[model]
    assessable = [r for r in reviews if r['hallucination'] is not None]
    hallucinations = sum(r['hallucination'] for r in assessable)
    summary['models'][model] = {'reviewed': len(reviews), 'assessable': len(assessable),
        'unassessable': len(reviews)-len(assessable), 'hallucination_count': hallucinations,
        'hallucination_rate': hallucinations/len(assessable) if assessable else None,
        'mean_response_quality': sum(r['response_quality'] for r in reviews)/len(reviews)}
path = Path('reports/response-review-summary.json')
if path.exists():
    raise ValueError('Preserve review history; use a new output version')
path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
