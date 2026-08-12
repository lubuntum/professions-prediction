# Изменения Cluster Prediction Service

## Назначение документа

Этот файл объясняет устройство Python-сервиса расчёта профессии. Основная цель
рефакторинга — сделать код линейным и читаемым для Python-разработчика, не
меняя математическую модель, порядок признаков и внешний `/predict` контракт.

## Место сервиса в архитектуре

```text
Frontend
  -> POST Java /api/predictions/predict
Java backend
  -> собирает Pupil + последние PsychTest
  -> POST Python /predict
Python
  -> проверяет/обновляет кластеры
  -> рассчитывает ближайшего Specialist
  -> возвращает Prediction
Java
  -> проверяет Profession/Specialist
  -> сохраняет Prediction
```

Python не хранит пользовательские прогнозы в PostgreSQL и не вызывается
браузером напрямую.

## Новая компактная структура

| Файл | Ответственность |
|---|---|
| `main.py` | FastAPI application, `/predict`, `/health`, HTTP-ошибки |
| `models.py` | Pydantic-модели Pupil, PsychTest, Specialist, Prediction и Health |
| `settings.py` | Чтение env и разрешение путей относительно проекта |
| `mapping.py` | Активные признаки, default value и единый порядок feature vector |
| `specialist.py` | Загрузка Specialist из backend и построение cluster artifacts |
| `clusters.py` | TTL, state, lock, атомарное обновление и stale fallback |
| `prediction.py` | Непосредственный расчёт Prediction |
| `test_mapping.json` | Связь PsychTest → признаки и флаг `is_active` |
| `category.csv` | Диапазоны расстояния и текстовые категории уверенности |
| `tests/` | API, lifecycle, mapping и детерминированность расчёта |

Удалены Java-подобные слои `services/`, вложенные model packages, отдельные
ReferenceService/FeatureService и вспомогательные скрипты старого контура.
Бизнес-поток теперь читается последовательно по указанным семи Python-файлам.

## Внешний API

### `POST /predict`

Вход:

```json
{
  "pupilId": 123,
  "psychTests": {
    "Temperament": {
      "completionTimeSeconds": 17,
      "psychParams": [
        {"name": "extrav_introver_score", "param": 13}
      ],
      "testTypeName": "Temperament",
      "createdAt": null
    }
  }
}
```

Ответ:

```json
{
  "pupilId": 123,
  "cluster": 1,
  "predictedProfession": "Profession A",
  "nearestSpecialistId": 42,
  "distance": 0.82,
  "confidenceCategory": "Маленькое"
}
```

Pydantic использует snake_case внутри кода и aliases для существующего
camelCase JSON. Не менять внешние имена без одновременного изменения Java DTO.

### `GET /health`

Endpoint ничего не обновляет. Он только проверяет состояние файлов и возвращает:

- `fresh` — валидные cluster files младше TTL;
- `stale` — файлы валидны, но требуют обновления;
- `missing` — state или обязательные файлы отсутствуют/повреждены.

Поля `referenceData` и `lastSuccessfulRefresh` оставлены только для
совместимости внешнего health-контракта. Во внутреннем коде используется
термин Cluster.

## Подготовка признаков

`mapping.load_active_features()` читает `test_mapping.json` и включает только
тесты, у которых значение строго равно `is_active=true`.

Текущий активный набор:

- `Temperament`;
- `Group-Roles`;
- `Engineering-Thinking`.

Климов, Холланд и Intellectual Potential присутствуют в mapping, но сейчас
не участвуют в расчёте.

Порядок признаков определяется порядком тестов и `param_names` в JSON. Один и
тот же `select_active_features()` применяется к Specialist и Pupil. Это
защищает от расхождения training/inference vectors.

Если тест, параметр или значение отсутствует, используется `default_value`.
`null` из legacy-базы также считается отсутствующим значением. Сейчас default
равен `0.0`.

Важно: перестановка `param_names` изменяет смысл сохранённых cluster files.
После изменения mapping требуется полная пересборка кластеров.

## Построение cluster files

`specialist.load_specialists()`:

1. Выполняет login в Java backend под отдельным ADMIN service account.
2. Получает защищённый `GET /api/specialists/reference-data`.
3. Проверяет каждый объект через Pydantic `Specialist`.

Название backend endpoint является legacy-совместимым внешним контрактом.

`create_cluster_files()` выполняет неизменённый математический процесс:

1. Формирует таблицу `specialist_id`, `profession`, active features.
2. Проверяет уникальность ID и наличие Profession.
3. Обучает `StandardScaler` на специалистах.
4. Масштабирует feature matrix.
5. Для каждого доступного `k` обучает `KMeans`:
   - `random_state=42`;
   - `n_init=50`.
6. Сохраняет labels и метрики:
   - silhouette score;
   - Davies–Bouldin index;
   - Calinski–Harabasz index.
7. Строит PCA 2D и PCA 3D для анализа/визуализации.
8. Проверяет полный набор файлов до публикации.

Количество рабочих кластеров задаётся `PREDICTION_CLUSTER_COUNT`, по умолчанию
`5`. Допустимые значения ограничены количеством специалистов и уникальных
профессий.

## Артефакты

В активной папке сохраняются как минимум:

```text
specialists.pkl
scaler.pkl
kmeans_<k>.pkl
specialist_features.npy
professions.npy
feature_names.npy
cluster_manifest.json
labels_<k>.npy
cluster_metrics.pkl
pca_2d.pkl
pca_3d.pkl
specialists_pca_2d.npy
specialists_pca_3d.npy
```

`validate_cluster_files()` проверяет обязательные файлы, порядок признаков,
размеры массивов и возможность прочитать scaler/KMeans.

Pickle-файлы никогда не принимаются по HTTP. Они загружаются только из
локальной активной cluster-папки, которой управляет сам сервис.

## Жизненный цикл кластеров

Состояние хранится в `cache/cluster_state.json`:

```json
{
  "updated_at": "2026-08-12T14:47:17Z",
  "active_folder": "2026_08_12_14_47_15_013719",
  "specialists_count": 103,
  "version": 1
}
```

Перед каждым prediction вызывается `ensure_clusters_ready()`.

```text
state + files валидны и моложе TTL
  -> использовать сразу

state отсутствует / files повреждены / TTL истёк
  -> получить filesystem lock
  -> повторно проверить state после lock
  -> загрузить Specialist из backend
  -> создать файлы в .updating-<uuid>
  -> проверить файлы
  -> atomic rename временной папки
  -> atomic replace cluster_state.json

обновление не удалось
  -> использовать предыдущие валидные stale files
  -> если их нет, вернуть HTTP 503 CLUSTERS_UNAVAILABLE
```

Timestamp меняется только после успешного полного формирования и публикации.
Параллельные запросы не запускают несколько пересборок. Зависший lock старше
`CLUSTER_LOCK_STALE_SECONDS` может быть безопасно удалён.

## Алгоритм Prediction

`prediction.predict_pupil()`:

1. Загружает active mapping.
2. Обеспечивает наличие актуальных кластеров.
3. Загружает scaler/KMeans и массивы один раз для активной папки.
4. Собирает Pupil vector в том же порядке, что Specialist vectors.
5. Применяет сохранённый `StandardScaler`.
6. Определяет кластер через `KMeans.predict()`.
7. Оставляет специалистов этого кластера.
8. Рассчитывает евклидово расстояние через `scipy.spatial.distance.cdist`.
9. Выбирает минимальное расстояние через `argmin`.
10. Возвращает profession и ID ближайшего Specialist.
11. Сопоставляет расстояние с первой подходящей строкой `category.csv`.

Изменять scaler, KMeans параметры, метрику расстояния или порядок признаков
можно только как отдельную версию модели с эталонным сравнением результатов.

## Конфигурация

`.env.example` содержит полный набор настроек:

| Переменная | Назначение | Default |
|---|---|---:|
| `BACKEND_BASE_URL` | Базовый URL Java backend | обязательна для refresh |
| `BACKEND_SERVICE_EMAIL` | ADMIN service account | обязательна для refresh |
| `BACKEND_SERVICE_PASSWORD` | Пароль service account | обязательна для refresh |
| `CLUSTER_UPDATE_HOURS` | TTL cluster files | `24` |
| `CLUSTER_DATA_DIR` | Версионированные папки артефактов | `data` |
| `CLUSTER_CACHE_DIR` | State и lock | `cache` |
| `CLUSTER_LOCK_WAIT_SECONDS` | Ожидание конкурентного refresh | `180` |
| `CLUSTER_LOCK_STALE_SECONDS` | Возраст заброшенного lock | `1800` |
| `PREDICTION_CLUSTER_COUNT` | Используемый KMeans k | `5` |
| `BACKEND_CONNECT_TIMEOUT_SECONDS` | HTTP connect timeout | `5` |
| `BACKEND_READ_TIMEOUT_SECONDS` | HTTP read timeout | `60` |

`data/`, `cache/`, `.env`, `.venv` и pytest cache не коммитятся.

## Ошибки и наблюдаемость

- Невалидный JSON/Pydantic input: HTTP 422.
- Нет кластеров и refresh не выполнен: HTTP 503,
  `CLUSTERS_UNAVAILABLE`.
- Ошибка расчёта после получения корректного Pupil: HTTP 500,
  `PREDICTION_FAILED` без внутренних деталей.
- Логи содержат `pupilId`, длительность prediction и состояние refresh, но не
  email, пароль, JWT или полный набор психологических параметров.

Java backend дополнительно преобразует эти ошибки в безопасные продуктовые
коды и не сохраняет частичный Prediction.

## Локальный запуск

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
copy .env.example .env
.venv/Scripts/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Принудительная ручная пересборка:

```text
.venv/Scripts/python specialist.py
```

## Тесты

```text
.venv/Scripts/python -m pytest -q
```

Проверяются:

- `/health` без побочного refresh;
- внешний `/predict` contract и Pydantic validation;
- безопасный 503 при недоступных кластерах;
- fresh/stale/missing lifecycle;
- timestamp только после успешного refresh;
- stale fallback;
- один refresh при конкурентных запросах;
- lock timeout;
- строгий `is_active=true`;
- одинаковый порядок признаков Pupil/Specialist;
- `null` → `default_value`;
- детерминированный результат на фиксированном наборе специалистов.

На момент подготовки документа проходят 16 тестов. Также выполнен сквозной
smoke на восстановленной PostgreSQL БД: backend отдал 103 пригодных профиля,
Python построил кластеры, prediction был сохранён Java backend и прочитан через
`GET /api/predictions/latest`.

## Как безопасно расширять сервис

1. Новый PsychTest сначала добавить в backend DTO и `test_mapping.json`.
2. Включать его в расчёт только явным `is_active=true`.
3. Зафиксировать порядок `param_names` тестом.
4. После изменения mapping удалить/сменить active state и пересобрать кластеры.
5. Не менять математические параметры без эталонной выборки и версии модели.
6. Не публиковать pickle, credentials или реальные PsychTest payloads.
7. Синхронизировать изменение `/predict` с Java DTO и frontend types.
8. Проверить fresh, stale fallback, первый запуск без файлов и concurrency.
9. Запустить pytest и полный Backend → Python → Backend smoke.

