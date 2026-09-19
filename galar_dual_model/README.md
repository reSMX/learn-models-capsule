# Две многометочные модели Galar

Папка содержит изолированную систему обучения и применения двух независимых
моделей `EfficientNet-B0`. Она не меняет поведение корневого `train.py` и не
использует прежнюю 14-классовую постановку.

- **Anatomy** определяет анатомические ориентиры и отделы ЖКТ.
- **Pathology** определяет патологические находки.

Обе задачи многометочные. Каждый выход имеет собственную sigmoid-оценку, поэтому
несколько меток могут быть верны одновременно.

## Состав папки

```text
galar_dual_model/
├── labels.py                  канонические группы меток
├── prepare_metadata.py        разделение исследований и подготовка таблиц
├── prepare_image_cache.py     локальное хранилище уменьшенных изображений
├── dataset.py                 чтение данных
├── train_common.py            общая логика обучения
├── train_anatomy.py           точка входа Anatomy
├── train_pathology.py         точка входа Pathology
├── evaluate_thresholds.py     только диагностическая проверка порогов
├── tools/
│   ├── video_inference.py     обработка видео двумя моделями
│   └── extract_pathology_frames.py
└── runs/                      локальные результаты, не входят в Git
```

## Метки

### Anatomy

Девять меток:

- ориентиры: `z-line`, `pylorus`, `ampulla of vater`, `ileocecal valve`;
- отделы: `mouth`, `esophagus`, `stomach`, `small intestine`, `colon`.

Ориентир и отдел могут быть размечены на одном кадре. Например,
`ampulla of vater` может одновременно относиться к `small intestine`.

### Pathology

Двенадцать меток с положительными примерами:

`ulcer`, `polyp`, `active bleeding`, `blood`, `erythema`, `erosion`,
`angiectasia`, `IBD`, `foreign body`, `hematin`, `cancer`,
`lymphangioectasis`.

Кадры без положительной патологической метки сохраняются как отрицательные
примеры для всех выходов. `blood`, `active bleeding` и `hematin` остаются
разными метками.

`esophagitis`, `varices` и `celiac` присутствуют в схеме исходных таблиц, но не
имеют положительных кадров. Они перечислены в
`ZERO_SUPPORT_PATHOLOGY_LABELS` и не входят в выход модели.

Технические метки `bubbles`, `dirt`, `no view`, `reduced view` и `good view` не
относятся ни к Anatomy, ни к Pathology.

## Подготовка данных

`prepare_metadata.py` читает официальные таблицы разметки по исследованиям и
создаёт одно общее разделение для обеих задач. Кадры одного исследования не
могут попасть одновременно в обучение и проверку.

После разделения обучающая часть прореживается по времени:

- Anatomy сохраняет все кадры с ориентирами и ограничивает длинные
  последовательности только с меткой отдела;
- Pathology сохраняет все кадры редких классов и ограничивает повторяющиеся
  последовательности распространённых сочетаний меток;
- проверочная часть не прореживается и сохраняет все доступные кадры.

Для текущего локального набора доступны 60 исследований: папки 1–40 и 61–80.
Исследования 41–60 отсутствуют локально. При явном
`--missing-study-policy skip` этот факт записывается в итоговый отчёт.

Создаваемые файлы:

```text
galar_dual_model/metadata/
├── anatomy_train.csv
├── anatomy_validation.csv
├── pathology_train.csv
├── pathology_validation.csv
├── anatomy_config.json
├── pathology_config.json
├── split.json
└── summary.json
```

Команда:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\prepare_metadata.py `
  --galar-root "D:\Dataset_galar" `
  --missing-study-policy skip `
  --overwrite
```

`--overwrite` разрешён только для заново создаваемых метаданных. Он не даёт
права перезаписывать результаты обучения.

## Локальное хранилище изображений

`prepare_image_cache.py` собирает общий набор кадров для обеих задач. Один кадр,
нужный двум моделям, хранится один раз. Изображения уменьшаются до `256 x 256`,
кодируются как JPEG и объединяются в последовательные файлы по исследованиям,
чтобы не открывать сотни тысяч отдельных файлов.

Пробная проверка без записи:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\prepare_image_cache.py `
  --galar-root "D:\Dataset_galar" `
  --output-root "C:\Users\Maxx\Galar_256_cache" `
  --workers 0 `
  --dry-run --dry-run-samples 32
```

Полное создание:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\prepare_image_cache.py `
  --galar-root "D:\Dataset_galar" `
  --output-root "C:\Users\Maxx\Galar_256_cache" `
  --image-size 256 `
  --jpeg-quality 95 `
  --jpeg-subsampling "4:4:4" `
  --workers 0 `
  --conversion-batch-size 64 `
  --minimum-free-gb 15
```

Исходные PNG не изменяются. Прерванное создание можно продолжить с `--resume`.
Манифест содержит отпечатки метаданных; обучение отклоняет устаревшее
хранилище.

## Проверка перед обучением

Краткие проверки используют настоящие изображения, выполняют прямой и обратный
проход, но не создают контрольные точки:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_anatomy.py `
  --image-cache-root "C:\Users\Maxx\Galar_256_cache" `
  --workers 0 --device cpu --batch-size 2 `
  --dry-run --dry-run-samples 4

.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_pathology.py `
  --image-cache-root "C:\Users\Maxx\Galar_256_cache" `
  --workers 0 --device cpu --batch-size 2 `
  --dry-run --dry-run-samples 4
```

## Обучение

Не запускайте Anatomy и Pathology одновременно на одной видеокарте. Перед
запуском убедитесь, что другого процесса обучения нет, а выбранный каталог
результата пуст и не существует либо пуст.

Пример Anatomy:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_anatomy.py `
  --image-cache-root "C:\Users\Maxx\Galar_256_cache" `
  --output ".\galar_dual_model\runs\anatomy_new" `
  --workers 0 --batch-size 64 --ram-buffer-gb 2 `
  --read-block-size 512 --device cuda
```

Пример Pathology:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\train_pathology.py `
  --image-cache-root "C:\Users\Maxx\Galar_256_cache" `
  --output ".\galar_dual_model\runs\pathology_new" `
  --workers 0 --batch-size 64 --ram-buffer-gb 2 `
  --read-block-size 512 --device cuda
```

Основные свойства обучения:

- до 20 эпох;
- первые две эпохи основа сети заморожена;
- затем блоки `EfficientNet-B0` размораживаются постепенно от выхода;
- смешанная точность на CUDA;
- ранняя остановка после четырёх эпох без улучшения;
- выбор по макро-F1 при фиксированном пороге на невиданных исследованиях;
- автоматический отказ от записи в непустой каталог.

Точная функция потерь, скорости обучения, дополнение данных и порядок
размораживания определены в `train_common.py` и сохраняются в
`run_config.json`.

Каждый завершённый запуск содержит:

- `best.pt` — лучшую контрольную точку;
- `best_metrics.json` — показатели выбранной эпохи;
- `history.csv` и `history.json` — история эпох;
- `run_config.json` — фактические настройки и происхождение начальных весов.

## Текущие выбранные результаты

### Anatomy

```text
galar_dual_model/runs/anatomy_workers0_20260907/best.pt
```

- лучшая эпоха: 10;
- макро-F1 по поддержанным меткам: `0.399281`;
- micro-F1: `0.820622`.

Редкие ориентиры остаются слабым местом. Высокий micro-F1 в основном отражает
распространённые отделы ЖКТ.

### Pathology

```text
galar_dual_model/runs/pathology/best.pt
```

- лучшая эпоха: 8;
- макро-F1 по поддержанным меткам: `0.144995`;
- micro-F1: `0.171787`.

Вариант `Asymmetric Loss` и вариант со сглаженными весами положительных примеров
не превзошли этот результат при фиксированном пороге `0.5`. Поэтому исходный
вариант `BCEWithLogitsLoss` остаётся выбранным.

Эти числа относятся к одному разделению доступных 60 исследований и не являются
оценкой клинической пригодности.

## Диагностическая проверка порогов

`evaluate_thresholds.py` может проверить общий и отдельные пороги за один проход
по проверочной части:

```powershell
.\.venv-win\Scripts\python.exe -B .\galar_dual_model\evaluate_thresholds.py `
  --checkpoint ".\galar_dual_model\runs\pathology\best.pt" `
  --image-cache-root "C:\Users\Maxx\Galar_256_cache" `
  --workers 0 --device cuda
```

Результат имеет статус `diagnostic_only`: порог настраивается и измеряется на
одних и тех же исследованиях. Его нельзя использовать для выбора контрольной
точки или выдавать за независимую проверку.

## Применение к видео

Обработка видео двумя моделями:

```powershell
.\.venv-win\Scripts\python.exe -m galar_dual_model.tools.video_inference `
  --input "D:\video\study.mp4" `
  --output "D:\video\study_annotated.mp4"
```

Извлечение десяти кадров с наиболее высокими оценками Pathology:

```powershell
.\.venv-win\Scripts\python.exe -m galar_dual_model.tools.extract_pathology_frames `
  --input "D:\video\study.mp4" `
  --count 10
```

Обе команды:

- используют классы и пороги из контрольных точек;
- проверяют соответствие задач;
- применяют проверочное преобразование изображения;
- сохраняют многометочную природу результатов;
- не добавляют скрытое временное сглаживание;
- называют выходы оценками модели, а не вероятностями диагноза.

## Ограничения

- Модели анализируют отдельные кадры без временного и клинического контекста.
- Некоторые метки представлены одним или несколькими исследованиями.
- Pathology имеет низкий макро-F1 и много ложных срабатываний.
- Настройка функции потерь не заменяет добавление независимых исследований.
- Наиболее полезное улучшение данных — получить отсутствующие исследования
  Galar 41–60 и заново построить метаданные и хранилище без нарушения
  разделения по исследованиям.
- Ни одна команда в этой папке не должна использоваться как автоматическая
  медицинская диагностика.
