# Диагностика текущей лучшей модели

Дата отчёта: 4 сентября 2026 года.

## Итог

По принятому в проекте критерию выбора — macro-F1 на изолированной по `video_id` validation —
текущим лучшим checkpoint является:

```text
runs/combined_galar_head_only/best.pt
```

- Стратегия: обучается только classifier head, все 9 блоков pretrained backbone заморожены.
- Лучшая эпоха: 6.
- Validation macro-F1: `0.30218095314077376`.
- Balanced accuracy: `0.3107962793`.
- Overall accuracy: `0.8015190996`.
- Weighted F1: `0.8013328351`.
- SHA-256 checkpoint: `E0BC6EE3C58E7F1088F6725615FCF4C4BF0BE0C978B7E0A937846D28C0C1EF1B`.
- Размер checkpoint: 16,382,245 байт.

Это лишь небольшое улучшение относительно `combined_galar_v1`: `+0.006000` macro-F1, или 0.6
процентного пункта. Одновременно overall accuracy снизилась примерно на 0.8 процентного пункта.
Один запуск с одним seed не доказывает, что head-only стратегия стабильно лучше.

## Сравнение завершённых combined-run

| Run | Стратегия | Завершено эпох | Лучшая эпоха | Macro-F1 | Overall accuracy |
|---|---|---:|---:|---:|---:|
| `combined_galar_head_only` | backbone полностью заморожен | 10 | 6 | 0.302181 | 0.801519 |
| `combined_galar_v1` | после 2 эпох последние 3/9 блоков, LR 0.1x | 7 | 3 | 0.296181 | 0.809669 |
| `combined_galar_last_block` | после 2 эпох последний 1/9 блок, LR 0.03x | 6 | 2 | 0.294284 | 0.778733 |

`combined_galar_head_only` остановился после эпохи 10: эпохи 7–10 не превысили лучший результат
эпохи 6. `combined_galar_last_block` остановился после эпохи 6 и не улучшил результат эпохи 2
после разморозки последнего блока.

## Метрики лучшего checkpoint по классам

| Класс | Precision | Recall | F1 | Validation frames |
|---|---:|---:|---:|---:|
| `ampulla_of_vater` | 0.0000 | 0.0000 | 0.0000 | 17 |
| `angiectasia` | 0.0000 | 0.0000 | 0.0000 | 143 |
| `blood_fresh` | 0.0000 | 0.0000 | 0.0000 | 22 |
| `blood_hematin` | 0.9352 | 0.9660 | 0.9503 | 8,211 |
| `erosion` | 0.0000 | 0.0000 | 0.0000 | 191 |
| `erythema` | 0.0000 | 0.0000 | 0.0000 | 9 |
| `foreign_body` | 0.0129 | 0.2353 | 0.0244 | 17 |
| `ileocecal_valve` | 0.4425 | 0.2656 | 0.3320 | 753 |
| `lymphangiectasia` | 0.0160 | 0.0676 | 0.0258 | 74 |
| `normal_clean_mucosa` | 0.7628 | 0.8345 | 0.7971 | 4,424 |
| `polyp` | 0.8201 | 0.6861 | 0.7471 | 3,023 |
| `pylorus` | 0.3224 | 0.3485 | 0.3350 | 198 |
| `reduced_mucosal_view` | 0.7076 | 0.5145 | 0.5958 | 828 |
| `ulcer` | 0.4135 | 0.4331 | 0.4231 | 127 |

Пять классов имеют нулевой recall: `ampulla_of_vater`, `angiectasia`, `blood_fresh`, `erosion` и
`erythema`. Ещё два класса имеют F1 ниже 0.03: `foreign_body` и `lymphangiectasia`. Высокая общая
accuracy в основном поддерживается крупными и лучше распознаваемыми классами
`blood_hematin`, `normal_clean_mucosa` и `polyp`, поэтому она не отражает качество по всем классам.

## Характерные ошибки

- Из 22 кадров `blood_fresh` 17 классифицированы как `blood_hematin`.
- Из 191 кадра `erosion` 140 классифицированы как `normal_clean_mucosa`.
- Все 17 кадров `ampulla_of_vater` пропущены: 9 отнесены к `polyp`, по 4 — к
  `normal_clean_mucosa` и `pylorus`.
- Из 143 кадров `angiectasia` 59 отнесены к `pylorus`, 37 — к `normal_clean_mucosa`, 15 — к
  `polyp`, 14 — к `blood_hematin`.
- Для `polyp` правильно классифицированы 2,074 из 3,023 кадров; основные ошибки —
  `blood_hematin` (507) и `normal_clean_mucosa` (417).

## Валидность оценки

- В combined metadata 71,609 исходных строк и 71,581 пригодная single-label строка после аудита.
- Train содержит 28,117 кадров после temporal cap, validation — 18,037 кадров.
- Train/validation содержат 66/17 независимых source video/study ID.
- Все 14 классов представлены в validation на невиданных целых видео.
- Классов с неоцениваемой validation sensitivity нет.
- Отдельный same-video `diagnostic_only` holdout не создаётся, поэтому штатный
  `diagnostic_report.json` имеет статус `unavailable`. Это ожидаемо и не означает отсутствие
  полноценной validation.

## Ограничения и решение

Checkpoint выбран корректно по основной метрике проекта, однако модель пока не подходит для
клинического применения: macro-F1 остаётся низким, пять классов полностью пропускаются, softmax не
калиброван, temporal-контекст отсутствует, а устойчивость результата на нескольких seed и внешней
клинической выборке не проверена.

Для технического веб-MVP следует использовать `combined_galar_head_only/best.pt` как формально
лучший доступный checkpoint, обязательно показывая top-3 как «оценки модели», а не диагноз. Для
дальнейшего улучшения важнее пересмотреть данные, баланс исследований и постановку задачи, чем
продолжать текущие варианты разморозки backbone без дополнительных изменений.

Исходные артефакты отчёта: `runs/combined_galar_head_only/report.json`, `history.csv`,
`confusion_matrix.csv`, `split_summary.json`, `data_integrity.json` и `training_config.json`.
