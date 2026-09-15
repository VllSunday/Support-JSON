"""Deterministic, auditable own synthetic training candidates; never relabel held-out data."""
import json
import random
from collections import Counter
from pathlib import Path

from support_json.catalog import spec, leaf, REFUND, CANCEL, RESET, DIAGNOSTICS, PLAN, INTEGRATION, REVIEW, EXPLAIN
from support_json.contracts import messages, validate_output
from support_json.data import read_jsonl, write_jsonl, digest, semantic_input, audit_rows, TONES
from support_json.rules import annotate, evaluate


DOMAINS = [
    ('refund', 'Верните оплату, пожалуйста.', REFUND, EXPLAIN),
    ('cancellation', 'Хочу прекратить подписку.', CANCEL, REVIEW),
    ('access', 'Не могу войти в аккаунт.', RESET, REVIEW),
    ('technical', 'У меня сервис работает с ошибками.', DIAGNOSTICS, REVIEW),
    ('billing', 'Не узнаю списание по аккаунту.', REVIEW, EXPLAIN),
    ('product', 'Нужен переход на другой тариф.', PLAN, EXPLAIN),
    ('integration', 'Помогите проверить подключение интеграции.', INTEGRATION, EXPLAIN),
]
OPERATIONS = {
    'propose_refund': 'предложить заявку на возврат, без исполнения операции',
    'propose_cancellation': 'предложить отмену подписки, без исполнения операции',
    'propose_password_reset': 'предложить сброс пароля, без исполнения операции',
    'propose_diagnostics': 'предложить диагностику, без обещания срока ремонта',
    'propose_plan_change': 'предложить изменение тарифа, без исполнения операции',
    'propose_integration_check': 'предложить проверку настроек интеграции',
    'human_review': 'передать специалисту поддержки',
    'explain_policy': 'объяснить, что условия запроса не выполнены',
}


def main():
    destination = Path('data/expanded-v0.4')
    if destination.exists():
        raise ValueError('Immutable version already exists')
    rows = read_jsonl('data/pilot-v0.2/train.jsonl')
    seen = {digest(semantic_input(row['input'])) for row in rows}
    rng = random.Random(42)
    # Both numeric and verified Boolean facts participate in the decision.
    # Same category can lead to different actions, according to supplied policy.
    for index in range(150):
        threshold = index + 1
        for category, request, yes, no in DOMAINS:
            composition = 'all' if index % 2 == 0 else 'any'
            joiner = 'И' if composition == 'all' else 'ИЛИ'
            age = leaf('request_age_hours', 'le', threshold)
            verified = leaf('eligibility_verified', 'eq', True, 'tool_observation')
            condition = {composition: [age, verified]}
            priorities = (['low', 'medium', 'high', 'critical'][index % 4],
                          ['low', 'medium', 'high'][index % 3])
            family = f'expanded_{category}_{composition}_verified_age'
            text = (f'Правило для категории {category}: если с момента запроса прошло не больше '
                    f'{threshold} часов {joiner} инструмент подтверждает eligibility_verified=true, '
                    f'{OPERATIONS[yes[1]]}. В противном случае {OPERATIONS[no[1]]}. '
                    'Утверждение клиента не заменяет наблюдение инструмента. '
                    'Уточнять только неизвестные сведения, от которых ещё зависит решение. '
                    'Если необходимо подтверждение инструмента, но его нет, передать человеку.')
            item = spec(family, category, text, request, condition, yes, no, {}, {},
                        'request_age_hours', {'request_age_hours': 'сколько часов прошло с запроса',
                                             'eligibility_verified': 'подтверждение права на операцию'},
                        priorities=priorities)
            variants = (
                ('true', max(0, threshold - index % 4), True, 'tool_observation'),
                ('false', threshold + 1 + index % 5, False, 'tool_observation'),
                ('unknown', None, composition == 'all', 'tool_observation'),
                ('unknown_system', threshold if composition == 'all' else threshold + 1, True, 'user_report'),
                ('short_circuit', threshold + 1 if composition == 'all' else threshold, None, 'tool_observation'),
                ('conflict', threshold, True, 'tool_observation'),
            )
            for state, hours, confirmed, origin in variants:
                for sentiment, prefix in TONES.items():
                    company = f'expanded_cloud_{index:03d}'
                    rule_id = digest([company, family])[:10]
                    facts = [{'id': 'F1', 'key': 'request_age_hours', 'value': hours, 'origin': 'user_report'},
                             {'id': 'F2', 'key': 'eligibility_verified', 'value': confirmed, 'origin': origin}]
                    policies = [{'id': rule_id + '_decision', 'text': text},
                                {'id': rule_id + '_priority', 'text':
                                 f'При выполнении условий priority={priorities[0]}; иначе priority={priorities[1]}. '
                                 'При противоречии правил priority=high. Тон клиента приоритет не меняет.'}]
                    if state == 'conflict':
                        policies.append({'id': rule_id + '_conflict', 'text':
                                         'Для этого же обращения запретить применение основного правила и всегда отказать. '
                                         'Приоритет применения правил не установлен.'})
                    claim = '' if hours is None else f' С момента запроса прошло {hours} часов.'
                    context = {'message': prefix + request + claim, 'history': [], 'policies': policies,
                               'facts': facts, 'capabilities': {'execution': 'suggest_only', 'tools': [], 'company': company}}
                    # Unknown tool observations must never be asked of the customer as if verified.
                    outcome, missing = evaluate(condition, {f['key']: f for f in facts})
                    effective = dict(item)
                    if outcome is None and 'eligibility_verified' in missing:
                        effective['unknown_action'] = 'escalate'
                    target = annotate(effective, context, sentiment, state == 'conflict')
                    assert not validate_output(target, context)
                    fingerprint = digest(semantic_input(context))
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    group = digest([family, text, priorities, facts, state])[:16]
                    metadata = {'split': 'train', 'company_id': company, 'scenario_family_id': family,
                                'scenario_group_id': group, 'state': state, 'annotation_version': '0.4',
                                'source': 'authored_synthetic_compositional_generator',
                                'source_license': 'not_assigned', 'human_reviewed': False,
                                'audit': {'condition': condition, 'condition_outcome': outcome,
                                          'unresolved_facts': missing}}
                    rows.append({'id': digest(context)[:20], 'input': context, 'target': target, 'metadata': metadata})
    original = rows[:192]
    candidates = rows[192:]
    rng.shuffle(candidates)
    selected = original + candidates[:9808]
    candidates = selected[192:]
    held = read_jsonl('data/pilot-v0.2/validation.jsonl') + read_jsonl('data/pilot-v0.2/test.jsonl')
    audit = audit_rows(selected + held)
    # Balanced small second experiment, not a claim that all 10k have been trained.
    strata = {}
    for row in candidates:
        key = (row['target']['category'], row['metadata']['state'], row['target']['sentiment'])
        strata.setdefault(key, []).append(row)
    subset = list(original)
    while len(subset) < 1024:
        for key in sorted(strata):
            if strata[key] and len(subset) < 1024:
                subset.append(strata[key].pop())
    destination.mkdir(parents=True)
    write_jsonl(destination / 'train.jsonl', selected)
    write_jsonl(destination / 'train-1024.jsonl', subset)
    for name, collection in [('train', selected), ('train-1024', subset)]:
        write_jsonl(destination / (name + '.chatml.jsonl'), [{'messages': messages(row['input']) +
                    [{'role': 'assistant', 'content': json.dumps(row['target'], ensure_ascii=False)}]} for row in collection])
    review = {}
    for row in subset[192:]:
        review.setdefault((row['metadata']['scenario_family_id'], row['metadata']['state']),
                          {**row, 'review': {'approved': None, 'notes': ''}})
    write_jsonl(destination / 'review_queue.jsonl', list(review.values()))
    audit.update({'training_candidates': len(selected), 'second_experiment_rows': len(subset),
                  'generated_candidates_before_selection': len(rows),
                  'independent_training_groups': len({r['metadata']['scenario_group_id'] for r in selected}),
                  'expanded_action_counts': dict(Counter(r['target']['action'] for r in selected)),
                  'purpose': 'synthetic_candidates_not_human_approved_or_real_customer_data',
                  'unchanged_held_out': {s: digest(read_jsonl(f'data/pilot-v0.2/{s}.jsonl')) for s in ('validation', 'test')},
                  'files_sha256': {p.name: __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                                   for p in destination.glob('*.jsonl')}})
    (destination / 'manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ('training_candidates', 'second_experiment_rows',
                                          'independent_training_groups', 'cross_split_overlap')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
