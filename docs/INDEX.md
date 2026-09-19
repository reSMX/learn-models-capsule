# Указатель документации

## Действующее направление

- [`../README.md`](../README.md) — входная страница, состояние проекта и
  основные команды.
- [`../galar_dual_model/README.md`](../galar_dual_model/README.md) — подготовка
  данных, обучение и проверка двух Galar-моделей.
- [`../web_demo/AGENTS.md`](../web_demo/AGENTS.md) — требования к будущему
  локальному веб-приложению.
- [`PROJECT_VISION.md`](PROJECT_VISION.md) — назначение системы и этапы развития.

## Прежняя 14-классовая линия

- [`legacy/README.md`](legacy/README.md) — назначение сохранённых материалов.
- [`legacy/BEST_MODEL_DIAGNOSTICS.md`](legacy/BEST_MODEL_DIAGNOSTICS.md) —
  показатели и ограничения выбранной 14-классовой контрольной точки.
- [`legacy/EXTERNAL_DATASETS.md`](legacy/EXTERNAL_DATASETS.md) — прежняя схема
  добавления внешних данных к Kvasir-Capsule.
- [`../legacy_14_class/README.md`](../legacy_14_class/README.md) — сохранённые
  вспомогательные программы этой линии.

## Правила для сопровождающих агентов

- [`AGENT_MAINTENANCE.md`](AGENT_MAINTENANCE.md) — правила обновления
  локального `AGENTS.md`, который намеренно не входит в Git.

Локальные результаты обучения Galar, их веса и наборы данных не входят в Git.
Исключение — уже сохранённые исторические результаты прежней линии в корневом
`runs/`. Точные показатели нового локального запуска следует читать из
`best_metrics.json`, `run_config.json` и его истории.
