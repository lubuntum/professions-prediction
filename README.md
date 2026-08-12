# Python Prediction Service

FastAPI-сервис рассчитывает подходящую профессию для Pupil по результатам PsychTest и данным Specialist. Frontend обращается только к Java backend; прямой frontend → Python вызов не поддерживается.

## Как работает prediction

1. Backend отправляет в `POST /predict` Pupil и его PsychTest.
2. `mapping.py` читает `test_mapping.json` и оставляет только показатели тестов с `is_active=true`.
3. `clusters.py` проверяет наличие cluster files и время `cache/cluster_state.json`.
4. Если файлов нет или прошло 24 часа, `specialist.py` загружает Specialist из backend и заново формирует кластеры.
5. `prediction.py` подготавливает признаки Pupil в том же порядке, применяет scaler/KMeans и ищет ближайшего Specialist евклидовой дистанцией.
6. Python возвращает Prediction в Java backend, где результат проверяется и сохраняется.

Основные файлы:

| Файл | Назначение |
| --- | --- |
| `main.py` | FastAPI, `/predict`, `/health` и преобразование ошибок в HTTP |
| `models.py` | Pupil, PsychTest, Specialist, Prediction и health-модели |
| `settings.py` | env-настройки путей, TTL, backend и числа кластеров |
| `mapping.py` | выбор активных показателей и единый порядок признаков |
| `specialist.py` | загрузка Specialist и математическое формирование cluster files |
| `clusters.py` | проверка 24 часов, lock, atomic update и stale fallback |
| `prediction.py` | непосредственный расчёт Prediction для Pupil |

## API

`POST /predict`:

```json
{
  "pupilId": 123,
  "psychTests": {
    "Temperament": {
      "completionTimeSeconds": 17,
      "psychParams": [{"name": "extrav_introver_score", "param": 13}],
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

`GET /health` сообщает `fresh`, `stale` или `missing`, но никогда не запускает обновление кластеров. Поля `referenceData` и `lastSuccessfulRefresh` сохранены только как совместимый внешний health-контракт; внутри Python используется термин Cluster.

## Cluster files

`CLUSTER_UPDATE_HOURS` по умолчанию равен 24. Перед prediction выполняется `ensure_clusters_ready()`:

- свежие файлы используются сразу;
- отсутствующие или старые файлы обновляются один раз под filesystem lock;
- новые файлы создаются во временной папке, проверяются и публикуются atomic rename;
- `cache/cluster_state.json` обновляется только после успешного формирования;
- при ошибке допустимо использовать прежние валидные stale-файлы;
- если валидных файлов нет, `/predict` возвращает HTTP 503.

Сгенерированные `data/`, `cache/` и локальный `.env` игнорируются Git. Pickle-файлы не принимаются по HTTP и загружаются только из локальной активной cluster-папки.

## Локальный запуск

Python 3.12:

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
copy .env.example .env
.venv/Scripts/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Для загрузки Specialist нужны `BACKEND_BASE_URL`, `BACKEND_SERVICE_EMAIL` и `BACKEND_SERVICE_PASSWORD`. Текущий backend endpoint требует ADMIN service account. Секреты хранятся только в незакоммиченном `.env`.

Принудительное ручное обновление кластеров:

```text
.venv/Scripts/python specialist.py
```

Тесты:

```text
.venv/Scripts/python -m pytest
```
