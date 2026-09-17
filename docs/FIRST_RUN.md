# Первый GPU-эксперимент

Рабочая модель: Qwen/Qwen3.5-4B, ревизия
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Веса находятся в `models/qwen3.5-4b`; режим — только текст.

На этом компьютере установлена RTX 5070 Ti с 16 ГБ VRAM. Используем
PyTorch 2.11.0 с CUDA 12.8 и Soup 0.75.0. BF16-умножение на GPU проверено.
Первый backend — Transformers: полноценной Linux-среды для Unsloth нет.
Это сохраняет обучение через Soup, PEFT и TRL без четырёхбитной квантизации.

## Подготовка

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "soup-cli[train]==0.75.0"
.\.venv\Scripts\python.exe -m pip install "triton-windows==3.6.0.post26" "flash-linear-attention==0.5.2"
.\.venv\Scripts\python.exe -m pip install --no-deps "liger-kernel==0.8.3"
.\.venv\Scripts\python.exe -X utf8 scripts/download_weights.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_fla.py
.\.venv\Scripts\python.exe -X utf8 scripts/check_liger.py
.\.venv\Scripts\python.exe -X utf8 scripts/refresh_training_prompt.py
.\.venv\Scripts\python.exe -X utf8 scripts/prepare_native_pilot.py
```

Загрузчик получает веса, конфигурацию и токенизатор именно фиксированной ревизии.
Два safetensors сверяются по LFS SHA256; маленькие файлы — по LFS SHA256 или
Git blob SHA1 из метаданных этой ревизии. Сохраняются оба manifest. На текущем
компьютере файлы уже скачаны: повторная команда проверит их и пропустит загрузку.
Для одних конфигураций без весов используйте `--metadata-only`.

Последняя команда создаёт `configs/soup-pilot-native.yaml` и проверяет
фактическую разметку loss для всех 192 обучающих записей: контекст полностью
замаскирован, ответ и EOS присутствуют, обрезания нет. Лимит уменьшен до 2048,
поскольку записи исходного пилота и исправленного промпта помещаются в этот лимит. Используется
родной шаблон Qwen; ответы без reasoning_content получают пустой блок think,
как при генерации с enable_thinking=False.

## Сравнение и обучение

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/run_transformers.py --batch-size 4 --output reports/base-validation-v3.jsonl
.\.venv\Scripts\soup.exe train --config configs/soup-pilot-native.yaml --yes
.\.venv\Scripts\python.exe -X utf8 scripts/run_transformers.py --batch-size 4 --adapter checkpoints/support-json-pilot-v0.3-native-liger --output reports/lora-validation-v3.jsonl
```

Эти команды описывают эксперимент; наличие инструкции само по себе не
подтверждает успешный запуск. Реальные результаты появляются только вместе
с файлами предсказаний, метрик и журналами обучения.

Пакеты из четырёх обращений используются одинаково для обеих моделей. Задержка
в предсказании — время целого пакета; amortized_seconds — время на обращение
при последовательной обработке пакетов. Эти величины не взаимозаменяемы.

Генерация жадная, максимум 700 новых токенов, thinking выключен. JSON не
исправляется и не ограничивается схемой при декодировании. Поэтому оценка
измеряет способности модели, включая ошибки формата. Предсказания сохраняются
после каждого завершённого пакета; существующий файл не перезаписывается.

Train: 192; validation: 96; test: 108. На test не выбираем настройки обучения.
Обучение всей совокупности 396 записей нарушило бы это разделение.

Качество ответа и галлюцинации требуют отдельной содержательной оценки.
Автоматический счётчик не выдаёт отсутствие оценок за нулевую долю галлюцинаций.

## Особенности Windows

Первый запуск без Liger остановлен до первого обновления: по счётчикам Windows
обучающий процесс занял около 4,7 ГБ общей GPU-памяти в системной RAM.
В native-конфигурации включён Liger Fused Linear Cross Entropy. Его функция
ошибки и градиент при замороженном классификаторе и маске prompt проверены
относительно PyTorch; поддержка именно Qwen3.5 подтверждена по изменённому
методу forward, а не только сообщению об установке.

`pip check` сообщает, что liger-kernel требует дистрибутив triton. На Windows
используется дистрибутив triton-windows, предоставляющий тот же импорт triton.
Это известное несовпадение имён зависимостей в данном окружении; оно не скрыто
под изменением metadata. Проверки GPU-ядер выполняются отдельно.

У модели и токенизатора различаются EOS. При оценке явно останавливаемся на
обоих штатных токенах завершения, одинаково для base и LoRA. Промпт и параметры
декодирования сохраняются в отчётах и проверяются при сравнении.

## Выполненные следующие эксперименты

После пилота обучены два отдельных адаптера с нуля от той же базы:

- `configs/soup-expanded-1024.yaml`: 1024 примера из `data/expanded-v0.4`, одна эпоха, batch 1 × accumulation 8, 128 обновлений.
- `configs/soup-diverse-512.yaml`: `data/diverse-v0.5`, 512 примеров, две эпохи, batch 2 × accumulation 4, 128 обновлений.

Для этих наборов реальные маски Soup проверены на каждой обучающей записи; отчёты `reports/loss-mask-*.json`. Запуск из подготовленного окружения: `soup train --config <путь-конфигурации> --yes`. Не повторяйте его поверх сохранённого эксперимента: для нового запуска задайте новое имя и директорию checkpoint. Данные и сырые предсказания также сохраняются без перезаписи.

В `reports/final-selection.json` записан выбор по validation до test. Финальные base/LoRA используют одинаковый пакет из восьми test-обращений, тот же промпт, обе штатные остановки EOS и greedy generation. Пакет validation состоял из четырёх обращений. Сравниваются пары с одинаковыми настройками внутри каждого набора; задержки validation и test напрямую не сопоставляются.
