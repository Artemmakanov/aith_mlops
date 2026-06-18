# aith_mlops

Учебный MLOps-проект: анализ тональности твитов (sentiment classification) на стеке **ClearML** + **scikit-learn** + **ClearML Serving** + **Streamlit**.

Модель обучается и деплоится через ClearML; UI и тестовые скрипты обращаются к inference **только по HTTP** — модель в них не загружается.

## Стек

| Компонент | Назначение |
|-----------|------------|
| ClearML | датасеты, эксперименты, артефакты, Model Registry |
| scikit-learn | TF-IDF + Logistic Regression pipeline |
| ClearML Serving | HTTP inference endpoint |
| Streamlit | веб-интерфейс для предсказаний |
| Docker | контейнер ClearML Serving |

## Структура репозитория

```
aith_mlops/
├── scripts/
│   ├── create_real_dataset.py   # Этап 1 — загрузка и регистрация датасета
│   ├── train.py                 # Этап 2 — обучение модели на агенте
│   ├── publish_model.py         # Этап 3 — публикация в Model Registry
│   └── test_request.py          # Этап 4 — проверка HTTP endpoint
├── serving/
│   └── preprocess.py            # preprocess/postprocess для ClearML Serving
├── ui/
│   └── app.py                   # Этап 5 — Streamlit UI
├── docker/
│   └── serving.env.example      # пример env для Serving-контейнера
├── tests/
│   └── test_task.py             # Этап 0 — проверка ClearML-агента
└── requirements.txt
```

## Требования

- Python 3.10+
- Развёрнутый ClearML Server (Web UI, API, Files)
- ClearML Agent в очереди `students`
- Docker — для ClearML Serving
- Настроенный `~/clearml.conf` с credentials

## Установка

```bash
git clone https://github.com/Artemmakanov/aith_mlops.git
cd aith_mlops
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Этапы проекта

### Этап 0. Проверка агента

Убедиться, что задачи из репозитория выполняются на удалённом агенте.

```bash
python tests/test_task.py
```

В ClearML UI должна появиться задача `test_task_on_agent` в проекте `students_project`.

---

### Этап 1. Датасет

Скачивает публичный Twitter Sentiment dataset, формирует сбалансированную выборку (1000 строк) и регистрирует её в ClearML Dataset.

```bash
python scripts/create_real_dataset.py
```

На выходе — `dataset_id`. Подставьте его в `scripts/train.py` в параметр `dataset_id`.

---

### Этап 2. Обучение

Обучает sklearn-pipeline (TF-IDF + Logistic Regression), логирует метрики и confusion matrix, сохраняет `model_artifact`.

```bash
python scripts/train.py
```

Скрипт отправляется в очередь `students` и выполняется на агенте. После завершения запомните `task_id` лучшего эксперимента.

---

### Этап 3. Model Registry

Публикует артефакт лучшего эксперимента как версионированную модель в Registry.

1. Укажите `BEST_TASK_ID` в `scripts/publish_model.py`.
2. Запустите:

```bash
python scripts/publish_model.py
```

Модель `sentiment_model_best` появится в Model Registry с тегами `production`, `v1.0`.

---

### Этап 4. Serving

HTTP endpoint для inference. Preprocess-логика — в `serving/preprocess.py`.

**Preprocess** принимает JSON `{"x": ["текст1", "текст2", ...]}` и возвращает список меток.

#### Запуск Serving (Docker)

1. Скопируйте `docker/serving.env.example` → `docker/serving.env`.
2. Заполните ClearML credentials и `CLEARML_SERVING_TASK_ID`.
3. Поднимите контейнер ClearML Serving (порт inference зависит от конфигурации, по умолчанию в примере — `9090`; в тестах используется `9008`).

Endpoint:

```
POST http://localhost:<PORT>/serve/sentiment
Content-Type: application/json

{"x": ["I love this product!"]}
```

#### Проверка endpoint

```bash
python scripts/test_request.py
```

Ожидаемый ответ — JSON-массив меток, например `["positive"]`.

---

### Этап 5. UI (Streamlit)

Веб-интерфейс с полем ввода, кнопкой **Predict**, отображением label и latency. Работает через HTTP, модель не загружает.

```bash
streamlit run ui/app.py
```

По умолчанию UI обращается к `http://localhost:9008/serve/sentiment`. URL можно изменить:

- в sidebar приложения;
- через переменную окружения:

```bash
SERVING_URL=http://localhost:9008/serve/sentiment streamlit run ui/app.py
```

## Переменные окружения

| Переменная | Где используется | Описание |
|------------|------------------|----------|
| `SERVING_URL` | `ui/app.py` | URL inference endpoint |
| `CLEARML_*` | Serving-контейнер | Подключение к ClearML Server (см. `docker/serving.env.example`) |
| `SERVING_PORT` | Serving-контейнер | Порт HTTP inference |

## Порты (типичная конфигурация)

| Сервис | Порт |
|--------|------|
| ClearML Web UI | 8080 |
| ClearML API | 8008 |
| ClearML Files | 8081 |
| Inference (Serving) | 9008 / 9090 |

> Порт inference зависит от настройки Serving. Убедитесь, что URL в `test_request.py`, `ui/app.py` и фактический порт контейнера совпадают.

## Troubleshooting

**UI / test_request: endpoint unavailable**
- Проверьте, что ClearML Serving запущен.
- Убедитесь, что endpoint `sentiment` создан и привязан к опубликованной модели.
- Сверьте порт и URL.

**Задача зависает в очереди**
- Проверьте, что агент слушает очередь `students` и подключён к серверу.

**Ошибка credentials**
- Проверьте `~/clearml.conf` и переменные в `docker/serving.env`.
