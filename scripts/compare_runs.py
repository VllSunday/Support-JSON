"""Create a comparison only from completed, matching real model runs."""
import argparse
import json
import statistics
from pathlib import Path

from support_json.data import read_jsonl
from support_json.data import digest
from support_json.evaluation import score


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[round((len(ordered)-1)*probability)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset')
    args = parser.parse_args()
    paths = [Path(args.base), Path(args.lora)]
    metrics = [json.loads(p.with_suffix('.metrics.json').read_text(encoding='utf-8')) for p in paths]
    if metrics[0]['dataset_fingerprint'] != metrics[1]['dataset_fingerprint']:
        raise ValueError('Base and LoRA must be evaluated on the same data')
    runs = [json.loads(p.with_suffix('.run.json').read_text(encoding='utf-8')) for p in paths]
    if runs[0]['settings'] != runs[1]['settings']:
        raise ValueError('Generation settings must be identical')
    if runs[0].get('prompt_fingerprint') != runs[1].get('prompt_fingerprint'):
        raise ValueError('Prompt fingerprints must match')
    records = [read_jsonl(p) for p in paths]
    if args.dataset:
        dataset = read_jsonl(args.dataset)
    else:
        dataset = None
        for split in ('validation', 'test'):
            candidate = read_jsonl(f'data/pilot-v0.2/{split}.jsonl')
            if digest(candidate) == metrics[0]['dataset_fingerprint']:
                dataset = candidate
                break
        if dataset is None:
            raise ValueError('Supply the dataset path for an unfamiliar fingerprint')
    if digest(dataset) != metrics[0]['dataset_fingerprint']:
        raise ValueError('Dataset must match both completed runs')
    rescored = [score(dataset, rows)[0] for rows in records]
    if any(m['missing_predictions'] for m in metrics):
        raise ValueError('Finish both runs before comparison')
    summary = {'n': metrics[0]['n'], 'dataset_fingerprint': metrics[0]['dataset_fingerprint'],
               'accuracy': {}, 'latency': {}, 'json': {}, 'generation_settings': runs[0]['settings']}
    for field in metrics[0]['accuracy']:
        base, lora = (m['accuracy'][field] for m in metrics)
        summary['accuracy'][field] = {'base': base, 'lora': lora, 'delta_pp': (lora-base)*100}
    summary['field_only_accuracy'] = {}
    for field in rescored[0]['field_only_accuracy']:
        base, lora = (m['field_only_accuracy'][field] for m in rescored)
        summary['field_only_accuracy'][field] = {'base': base, 'lora': lora, 'delta_pp': (lora-base)*100}
    summary['scoring_version'] = '0.4'
    summary['critical_recall'] = {label: m['critical_recall'] for label, m in zip(('base', 'lora'), rescored)}
    summary['by_state'] = {label: m['by_state'] for label, m in zip(('base', 'lora'), rescored)}
    summary['macro_f1_strict'] = {label: m['macro_f1'] for label, m in zip(('base', 'lora'), rescored)}
    for label, rows in zip(('base', 'lora'), records):
        latency = [r['elapsed_seconds'] for r in rows]
        summary['latency'][label] = {'median_seconds': statistics.median(latency),
                                   'p95_seconds': percentile(latency, .95),
                                   'total_seconds': sum({r.get('batch_index', i): r['elapsed_seconds']
                                                         for i, r in enumerate(rows)}.values()),
                                   'median_amortized_seconds': statistics.median(
                                       r.get('amortized_seconds', r['elapsed_seconds']) for r in rows),
                                   'token_limit_hits': sum(r.get('hit_token_limit', False) for r in rows)}
    for field in ('json_parse_rate', 'schema_and_invariants_rate', 'missing_info_set_accuracy',
                  'supported_rule_ids_set_accuracy'):
        summary['json'][field] = {label: m[field] for label, m in zip(('base', 'lora'), metrics)}
    summary['human_evaluation'] = {label: m['human_evaluation'] for label, m in zip(('base', 'lora'), metrics)}
    destination = Path(args.output)
    if destination.exists() or destination.with_suffix('.md').exists():
        raise ValueError('Use a new comparison output path')
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Base и LoRA', '', f"Сравнение на {summary['n']} одинаковых обращениях.", '',
             '| Метрика | Base | LoRA | Изменение, п.п. |', '|---|---:|---:|---:|']
    for field, values in summary['accuracy'].items():
        lines.append(f"| {field} | {values['base']:.1%} | {values['lora']:.1%} | {values['delta_pp']:+.1f} |")
    lines += ['', 'Строгие метрики выше включают ошибки контракта. Отдельная точность полей:', '',
              '| Поле | Base | LoRA | Изменение, п.п. |', '|---|---:|---:|---:|']
    for field, values in summary['field_only_accuracy'].items():
        lines.append(f"| {field} | {values['base']:.1%} | {values['lora']:.1%} | {values['delta_pp']:+.1f} |")
    lines += ['', 'Доля корректных схем: ' + ', '.join(
        f"{label} {m['schema_and_invariants_rate']:.1%}" for label, m in zip(('base', 'LoRA'), metrics)),
        '', 'Медиана задержки генерации: ' + ', '.join(
        f"{label} {summary['latency'][label]['median_seconds']:.2f} с" for label in ('base', 'lora')),
        '', 'Загрузка модели исключена из задержки. Данные синтетические и содержат '
        'повторные формулировки одних ситуаций; это ограниченный пилот.',
        '', 'Правильный формат не доказывает качество текста. Если отдельная оценка '
        'ответов не добавлена, качество и галлюцинации остаются неизмеренными.']
    destination.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
