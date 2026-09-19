# Средства прежней 14-классовой линии

В этой папке находятся вспомогательные программы, которые применялись для
добавления отдельных кадров Galar к одноклассовой задаче Kvasir-Capsule.
Действующая двухмодельная система Anatomy и Pathology находится в
`galar_dual_model/` и не использует эти программы.

Основное обучение прежней линии намеренно осталось в корневом `train.py`, чтобы
сохранить совместимость с уже проведёнными опытами и командами воспроизведения.
Их результаты находятся в корневом `runs/`.

## Состав

- `run_experiment.py` — воспроизводимый запуск двух прежних вариантов
  размораживания основы сети;
- `tools/select_galar_target_videos.py` — выбор исследований Galar для трёх
  недостаточно представленных классов прежней модели;
- `tools/extract_galar_target_frames.py` — извлечение пригодных кадров и
  подготовка таблиц для `train.py`.

## Команды

Показ команды завершённого опыта без его повторного запуска:

```powershell
.\.venv-win\Scripts\python.exe -B -m legacy_14_class.run_experiment `
  head-only --print-command
```

Создание списка подходящих исследований:

```powershell
.\.venv-win\Scripts\python.exe -B -m legacy_14_class.tools.select_galar_target_videos `
  --galar-root "D:\Dataset_galar" `
  --output "D:\Dataset_galar\galar_target_videos.csv"
```

Извлечение кадров из официальных папок:

```powershell
.\.venv-win\Scripts\python.exe -B -m legacy_14_class.tools.extract_galar_target_frames `
  --frames-root "D:\Dataset_galar" `
  --galar-root "D:\Dataset_galar" `
  --skip-missing-studies
```

Эти программы сохраняются для воспроизводимости прежней постановки. Не
используйте их для подготовки многометочных задач Anatomy и Pathology.
