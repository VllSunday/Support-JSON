"""Write a concise final report from completed local evidence, not plans."""
import json
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

test = load('reports/test-comparison-final.json')
control = load('reports/few-shot-control.json')
v3 = load('reports/validation-comparison-v3.json')
v4 = load('reports/validation-comparison-v4.json')
v5 = load('reports/validation-comparison-v5.json')
selection = load('reports/final-selection.json')
review = load('reports/response-review-summary.json')
business = load('reports/business-results.json')
demo = load('reports/demo-checks.json')

def percent(value):
    return 'не оценено' if value is None else f'{value:.1%}'

lines = ['# Support-JSON: результат capstone', '',
         f"Текстовая Qwen3.5-4B дообучена BF16 LoRA через Soup. Выбран адаптер, обученный на {selection['training_rows']} синтетических примерах. Локальная демка помогает оператору, операции не исполняет.", '',
         f"Финальная проверка: {test['n']} одинаковых test-обращений, одна новая компания и отдельные авторские семейства правил. Train/validation/test разделены; адаптер выбран по validation до test. Prompt и параметры генерации совпадают, JSON не исправляется.", '',
         '| Метрика test | Base | LoRA | Изменение, п.п. |', '|---|---:|---:|---:|']
for field, label in [('category','Категория, отдельно'),('priority','Приоритет, отдельно'),('action','Действие, отдельно')]:
    values = test['field_only_accuracy'][field]
    lines.append(f"| {label} | {values['base']:.1%} | {values['lora']:.1%} | {values['delta_pp']:+.1f} |")
base = test['json']['schema_and_invariants_rate']['base']
lora = test['json']['schema_and_invariants_rate']['lora']
lines.append(f'| Полный контракт | {base:.1%} | {lora:.1%} | {(lora-base)*100:+.1f} |')
lines += ['', 'Отдельная точность полей не обнуляет категорию из-за ошибки другого поля. Полный контракт требует всей схемы и согласованности действий, missing_info и эскалации.', '',
          'Дополнительный контроль: base с четырьмя фиксированными примерами из train, выбранными до test. Это другой промпт; настройки по результатам test не менялись.', '',
          f"Категория {control['field_only_accuracy']['category']:.1%}, приоритет {control['field_only_accuracy']['priority']:.1%}, действие {control['field_only_accuracy']['action']:.1%}, полный контракт {control['schema_and_invariants_rate']:.1%}. Качество текста этого контроля отдельно не оценивалось.", '',
          '## Содержательная оценка', '',
          f"AI-assisted аудит {review['n_pairs']} пар, по одному тону каждой семьи/состояния. Это не независимая человеческая разметка. Рубрика и все обоснования сохранены.", '',
          '| AI-assisted | Base | LoRA |', '|---|---:|---:|']
for label, key in [('Quality, 1–5','mean_response_quality'),('Галлюцинации среди оценимых','hallucination_rate'),('Оценимые reply','assessable'),('Нет оценимого reply','unassessable')]:
    a,b = (review['models'][m][key] for m in ('base','lora'))
    if key == 'hallucination_rate':
        a,b = percent(a),percent(b)
    elif key == 'mean_response_quality':
        a,b = f'{a:.2f}',f'{b:.2f}'
    lines.append(f'| {label} | {a} | {b} |')
lines += ['', '## Что дал первый эксперимент', '',
          'Пилот 192: validation-контракт 49% → 90%, действие отдельно 28% → 61%. Однако категория 97% → 88%, critical recall 2/3 → 0/3. Три critical-строки — варианты одного сценария, а не три независимых инцидента. Регрессия раскрыта, оба запуска сохранены.', '',
          f"Расширение: 10 000 собственных синтетических train-кандидатов, 5685 смысловых групп; второй прогон на 1024. Validation-контракт второго: {v4['json']['schema_and_invariants_rate']['lora']:.1%}; действие отдельно: {v4['field_only_accuracy']['action']['lora']:.1%}. Все 10 тысяч не обучались.", '',
          f"Третий прогон: 512 примеров с более разнообразными условиями и ключами фактов, две эпохи. Validation-контракт: {v5['json']['schema_and_invariants_rate']['lora']:.1%}; действие отдельно: {v5['field_only_accuracy']['action']['lora']:.1%}. Итоговый адаптер: {selection['version']}; выбор ограничен critical recall не ниже base на validation. Различия запусков не доказывают отдельный причинный эффект разнообразия: одновременно отличаются объём данных и число эпох.", '',
          '## Бизнес и демо', '',
          f"Медиана генерации выбранной LoRA: {test['latency']['lora']['median_seconds']:.2f} с на пакет из {test['generation_settings']['batch_size']} обращений. Медиана амортизированного времени: {test['latency']['lora']['median_amortized_seconds']:.2f} с на обращение; это не задержка одиночного запроса. Загрузка модели исключена из benchmark latency.", '',
          'Задержки измерены в общей desktop-сессии, не в изолированном тесте скорости. Пользователь сообщил о параллельной игре; точное пересечение с основным benchmark неизвестно. Эти значения не доказывают преимущество скорости LoRA. Наблюдения сохранены в reports/runtime-contention.json.', '',
          f"Доля валидных результатов с предложенной эскалацией на test: {business['lora_escalation_fraction']:.1%}. Доля сложных случаев или ошибок формата: {business['models']['lora']['complex_case_or_invalid_fraction']:.1%}. Эти синтетические доли нельзя переносить на реальный поток.", '',
          'Сценарий: 1000 обращений × 60% пригодных для помощи AI × (5−2) минуты = 30 часов в месяц до затрат на контроль и исключения. Время оператора и стоимость энергии не измерены.', '',
          f"Сквозные проверки демки: {len(demo['checks'])}; все HTTP-статусы верны, ожидаемые поля вместе с контрактом совпали в 8 из 10 модельных ответов. Сырые ответы записаны в reports/demo-checks.json; скриншот сохранён.", '',
          'Отдельное AI-assisted чтение этих 10 demo-ответов отмечает три неподтверждённых/неточных утверждения. В частности, верное использование истории сопровождается выдуманным внешним статусом возврата. Пара с порогами 10 и 9 не пройдена: модель отказала в обоих случаях. Это не часть основного test-аудита и не доказательство надёжного переноса на новые правила; подробности в reports/demo-behavior-review.json.', '',
          '## Передача проекта', '',
          '- Адаптер: output/model/support-json-qwen3.5-4b-lora; базовые веса фиксированной ревизии отдельно в models/qwen3.5-4b.',
          '- Запуск и повторение: README.md, docs/FIRST_RUN.md, configs/*.yaml.',
          '- Данные и происхождение: data/*/manifest.json, docs/DATASET.md, docs/DATASET_EXPANSION.md.',
          '- Оценка: reports/*comparison*.json, сырые предсказания и blinded response review.',
          '- Презентация: output/presentation/Support_JSON_Capstone-v2.pptx; план защиты: docs/DEFENSE_OUTLINE.md.', '',
          '## Практические пределы', '',
          'Большинство русских формулировок шаблонные; test содержит одну компанию и мало независимых критических случаев. Сильные крупные модели не сравнивались. Произвольные регламенты и отсутствие галлюцинаций не гарантируются. Нужны независимая человеческая проверка, естественные переписки и тест на другом устройстве. Публикация не выполнена; лицензию собственного адаптера и данных определяет автор.', '']
destination = Path('RESULTS.md')
if destination.exists():
    raise ValueError('Final summary already exists; preserve evidence')
destination.write_text('\n'.join(lines), encoding='utf-8')
print('RESULTS.md written from completed experiment evidence')
