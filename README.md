# Первая модель для Kvasir-Capsule

Это базовый исследовательский pipeline: классификация 14 типов находок на размеченных кадрах. Он нужен как проверяемая отправная точка перед temporal-моделью и объединением кадров в события. Результат модели не является диагнозом.

## Почему такой baseline

- EfficientNet-B0 достаточно лёгкая для первого эксперимента и использует ImageNet pretraining.
- Разбиение выполняется по `video_id`: кадры одного видео не попадают одновременно в обучение и валидацию.
- Сэмплер использует ограниченные веса `sqrt(max_count / class_count)`: редкие классы встречаются чаще, но несколько кадров больше не повторяются сотни раз за эпоху.
- Основная метрика — macro-F1 только по классам, представленным в независимой video-level validation. Для отсутствующих классов отдельно считается частота ложных срабатываний.
- Первые две эпохи обучается только классификатор, затем размораживаются последние 3 блока backbone с пониженным learning rate; early stopping останавливает обучение при отсутствии улучшения.

## Запуск (PowerShell)

Рекомендуется Python 3.11 или 3.12; Python 3.14 может пока не поддерживаться текущими сборками PyTorch.

```powershell
py -3.12 -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --retries 20 --timeout 120 -r requirements-cuda.txt
python prepare_data.py --zip "D:\dataset_quazir\labelled_images-20260816T072545Z-1-001.zip" --output "D:\dataset_quazir\labelled_images"
python train.py --images "D:\dataset_quazir\labelled_images" --metadata "D:\dataset_quazir\metadata.csv" --epochs 15 --batch-size 16 --workers 0
```

Перед обучением можно отдельно проверить данные, разбиения и сэмплер:

```powershell
python train.py --dry-run --output "runs/audit"
```

Пути к текущему датасету уже записаны как значения по умолчанию, поэтому из PyCharm
скрипт можно запускать кнопкой **Run** без аргументов. Эквивалентная команда:

```powershell
.\.venv-win\Scripts\python.exe train.py
```

Если загрузка pretrained-весов запрещена или компьютер без интернета, добавьте `--no-pretrained` (качество будет ниже). Batch size по умолчанию равен 16; при нехватке видеопамяти уменьшите его до 8. На Windows при проблемах с worker-процессами используйте `--workers 0`.

## Защита от переобучения

Новый режим по умолчанию уменьшает влияние почти одинаковых соседних кадров: из каждой
пары `(video_id, class)` берётся не более 500 равномерно расположенных кадров. Для
EfficientNet-B0 первые две эпохи backbone полностью заморожен, затем обучаются только его
последние 3 из 9 блоков с learning rate в 10 раз меньше, чем у классификатора. Также
включены более сильные повороты/отражения/crop, лёгкий blur и random erasing, dropout `0.4`,
label smoothing `0.1`, weight decay `0.01`. Максимальный множитель сэмплера снижен с 20 до
10, поэтому несколько редких кадров повторяются реже.

Все параметры доступны через CLI. Например, ограничение кадров можно отключить флагом
`--max-frames-per-video-class 0`, а количество обучаемых блоков изменить через
`--trainable-backbone-blocks`.

Результаты нового запуска появятся в `runs/regularized_v2`: `best.pt`, таблица истории,
разбиения, отчёт по каждому классу и confusion matrix. Также автоматически создаются:

- `class_distribution.png` — распределение классов в train/validation/diagnostic;
- `class_balance.json` — точные количества, отношение max/min и ожидаемая частота сэмплирования;
- `training_curves.png` — loss и supported-class macro-F1 после каждой эпохи;
- `confusion_matrix.png` — нормализованная матрица ошибок лучшей модели;
- `unsupported_class_false_positives.csv` — ложные срабатывания классов без положительных validation-видео;
- `diagnostic_report.json` и `diagnostic_predictions.csv` — diagnostic-only проверка редких классов на отложенном временном блоке того же видео;
- `data_integrity.json` и `ambiguous_multilabel_rows.csv` — аудит удалённых дублей, пропусков и кадров с конфликтующими single-label целями.
- `training_config.json` — фактические значения регуляризации и learning rate;
- `train_temporal_cap_excluded.csv` — кадры, исключённые только из train как избыточно коррелированные.

TensorBoard и локальный веб-сервер не используются: при 16 ГБ RAM безопаснее сохранять
визуализации непосредственно в PNG. На Windows оставляйте `--workers 0`, чтобы DataLoader
не создавал дополнительные процессы с копиями библиотек и датасета.

## Внешние датасеты

Основным источником независимых видео для `ampulla_of_vater`, `blood_hematin` и `polyp`
выбран Galar. Ссылки, соответствие меток и правила безопасного объединения находятся в
[`EXTERNAL_DATASETS.md`](EXTERNAL_DATASETS.md).

Важно: `ampulla_of_vater`, `blood_hematin` и `polyp` встречаются каждый только в одном исходном видео. Честная video-level validation поэтому не может измерить их sensitivity. Pipeline отрезает для них последний временной блок, удаляет соседние guard-кадры из train и сохраняет результат как `diagnostic_only`; этот результат показывает распознавание только внутри исходного видео и не является доказательством переноса. После добавления независимых видео обычный group split автоматически сможет включить эти классы в полноценную validation.

По умолчанию diagnostic holdout составляет 25% редкого эпизода с промежутком в 3 кадра. Настройки доступны через `--diagnostic-fraction` и `--diagnostic-gap-frames`; полностью отключить probe можно флагом `--no-rare-diagnostic`.

## Следующий этап

После получения честного baseline: извлекать признаки соседних кадров обученной CNN, подавать окна признаков в Temporal CNN/Transformer и объединять последовательные положительные кадры в события по времени. Нельзя оценивать клиническую пригодность только по этому baseline: потребуется внешний датасет и проверка врачами.
