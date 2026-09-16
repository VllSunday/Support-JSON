"""Measure model workflow outputs and label business assumptions separately."""
import json
from pathlib import Path
from support_json.contracts import parse_output, validate_output
from support_json.data import read_jsonl

rows = read_jsonl('data/pilot-v0.2/test.jsonl')
comparison = json.loads(Path('reports/test-comparison-final.json').read_text(encoding='utf-8'))
by_id = {row['id']: row for row in rows}
models = {}
for label in ('base', 'lora'):
    predictions = read_jsonl(f'reports/{label}-test-final.jsonl')
    assert len(predictions) == len(rows)
    invalid, declared_escalation, candidates = 0, 0, 0
    for record in predictions:
        row = by_id[record['id']]
        try:
            output = parse_output(record['output'])
            errors = validate_output(output, row['input'])
        except (ValueError, TypeError):
            output, errors = None, ['Invalid JSON']
        if errors:
            invalid += 1
            continue
        declared_escalation += output['human_escalation']
        fields = ('category', 'priority', 'action', 'recommended_action', 'human_escalation')
        all_correct = all(output[key] == row['target'][key] for key in fields)
        all_correct &= set(output['missing_info']) == set(row['target']['missing_info'])
        all_correct &= set(output['supported_rule_ids']) == set(row['target']['supported_rule_ids'])
        candidates += all_correct and not output['human_escalation']
    models[label] = {'n': len(rows), 'invalid_output_count': invalid,
                     'valid_declared_escalation_count': declared_escalation,
                     'complex_case_or_invalid_fraction': (invalid+declared_escalation)/len(rows),
                     'valid_declared_escalation_fraction': declared_escalation/len(rows),
                     'structurally_and_label_correct_non_escalated_fraction': candidates/len(rows)}
report = {'models': models, 'lora_escalation_fraction': models['lora']['valid_declared_escalation_fraction'],
          'latency': comparison['latency'],
          'gold_escalation_fraction': sum(row['target']['human_escalation'] for row in rows)/len(rows),
          'scenario': {'measured': False, 'monthly_tickets': 1000, 'assumed_assistable_fraction': .6,
                       'manual_minutes': 5, 'assisted_operator_minutes': 2,
                       'gross_potential_saved_hours': 1000*.6*(5-2)/60,
                       'excludes': ['exception handling', 'review overhead', 'device/electricity cost']},
          'note': 'Synthetic stress-test rates are not company traffic rates. Correct labels and schema do not prove safe text or autonomous resolution. All draft replies still need operator review. Operator time and electricity cost were not measured.'}
destination = Path('reports/business-results.json')
if destination.exists():
    raise ValueError('Use a new version')
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
