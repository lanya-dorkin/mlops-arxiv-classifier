# ArXiv Paper Category Classifier

Автоматическая классификация научных статей ArXiv по категориям с использованием DistilBERT.

## Описание

Проект реализует классификацию научных статей из ArXiv по их категориям (например, `cs.LG`, `math.OC`, `physics.optics`). Поддерживаются две модели:

* **Baseline**: TF-IDF + Logistic Regression
* **DistilBERT**: Fine-tuned DistilBERT

Проект включает полный MLOps пайплайн: версионирование данных через DVC, трекинг экспериментов через MLflow, REST API для инференса и Docker-контейнеризацию.

## Архитектура

* **Модель**: DistilBERT (distilbert-base-uncased) с замороженным backbone и обучаемым классификатором поверх эмбеддингов
* **Функция потерь**: CrossEntropyLoss
* **Оптимизатор**: AdamW с линейным warmup decay scheduler
* **Разделение данных**: по дате публикации (train < 2023-01-01, val: 2023-2024, test >= 2024-01-01)
* **Входные данные**: конкатенация title + summary, токенизация через DistilBertTokenizer (max_length=512)

## Техническая часть

### Setup

1. **Prerequisites:**
   * Python 3.12 или выше
   * `uv` для управления зависимостями
   * Git
   * Kaggle API ключ (для скачивания данных)
   * MLflow (опционально, для трекинга экспериментов)
   * Docker (опционально, для контейнеризации API)

2. **Установка зависимостей:**
   ```bash
   uv sync
   ```

3. **Настройка Kaggle API:**
   Поместите файл `kaggle.json` с вашими credentials в `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   # Скопируйте kaggle.json в ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Инициализация DVC (опционально):**
   ```bash
   bash scripts/setup_dvc.sh
   ```
   Для использования облачного хранилища:
   ```bash
   dvc remote add -d myremote gdrive://YOUR_FOLDER_ID
   # или
   dvc remote add -d myremote s3://your-bucket/path
   ```

5. **Проверка pre-commit хуков (опционально):**
   ```bash
   uv run pre-commit run -a
   ```

### Train

Все команды запускаются из корневой директории проекта.

1. **Запуск сервера MLflow (опционально):**
   Если сервер уже запущен, можно пропустить этот шаг:
   ```bash
   uv run python -m mlflow ui --host 127.0.0.1 --port 8080
   ```
   Если MLflow не запущен, обучение продолжится без трекинга (с предупреждением в логах).

2. **Настройка конфигов (опционально):**
   Конфигурирование проекта осуществляется через Hydra. Конфигурационные файлы находятся в директории `configs/`:
   * `configs/config.yaml` - основной конфиг
   * `configs/model/` - настройки моделей (distilbert, baseline)
   * `configs/training/` - параметры обучения
   * `configs/data/` - настройки датасета
   * `configs/mlflow/` - настройки MLflow

3. **Скачивание данных:**
   ```bash
   uv run bash scripts/download_artifacts.sh
   # или напрямую
   uv run python -m arxiv_classifier.commands download
   ```
   Данные будут скачаны из Kaggle в директорию `data/`.

4. **Обучение baseline модели:**
   ```bash
   uv run python -m arxiv_classifier.commands baseline
   # или через DVC
   uv run dvc repro train_baseline
   ```
   Модель сохранится в `train_artifacts/baseline_model.pkl`, метрики в `train_artifacts/baseline_metrics.json`.

5. **Обучение DistilBERT модели:**
   ```bash
   uv run python -m arxiv_classifier.commands train
   # или через DVC
   uv run dvc repro train
   ```
   Модель сохранится в `train_artifacts/model.pt`, label encoder в `train_artifacts/label_encoder.pkl`.
   
   Во время обучения можно следить за процессом в MLflow UI (по умолчанию http://127.0.0.1:8080). Сохраняются следующие метрики:
   * loss (train и val)
   * accuracy (train и val)
   * f1-score (train и val)
   * epoch

6. **Тестирование модели:**
   ```bash
   uv run python -m arxiv_classifier.commands test \
       --checkpoint_path="train_artifacts/model.pt"
   ```

### Production preparation

После обучения модель сохраняется в формате PyTorch Lightning checkpoint (`train_artifacts/model.pt`). Для продакшена можно экспортировать модель в другие форматы:

1. **Экспорт в ONNX:**
   ```bash
   bash scripts/convert_to_onnx.sh
   # или
   uv run python -m arxiv_classifier.commands export_onnx \
       --checkpoint_path="train_artifacts/model.pt" \
       --output_path="train_artifacts/model.onnx"
   ```

2. **Экспорт в TorchScript:**
   ```bash
   uv run python -m arxiv_classifier.commands export_torchscript \
       --checkpoint_path="train_artifacts/model.pt" \
       --output_path="train_artifacts/model.pt.script"
   ```

### Inference

Инференс модели может быть осуществлён несколькими способами:

1. **CLI инференс (одиночный пример):**
   ```bash
   uv run python -m arxiv_classifier.commands infer \
       --checkpoint_path="train_artifacts/model.pt" \
       --title="Machine Learning" \
       --summary="This paper presents a novel approach..."
   ```

2. **CLI инференс (batch из JSON):**
   Создайте файл `samples.json`:
   ```json
   [
     {
       "title": "Deep Learning",
       "summary": "We present a new architecture..."
     },
     {
       "title": "Quantum Computing",
       "summary": "This work explores..."
     }
   ]
   ```
   Затем:
   ```bash
   uv run python -m arxiv_classifier.commands infer \
       --checkpoint_path="train_artifacts/model.pt" \
       --json_file="samples.json"
   ```

3. **REST API сервер:**
   ```bash
   bash scripts/start_api.sh
   # или напрямую
   uv run python -m arxiv_classifier.commands serve \
       --checkpoint_path="train_artifacts/model.pt" \
       --host="0.0.0.0" \
       --port=8000
   ```
   
   API документация доступна по адресу http://localhost:8000/docs
   
   Пример запроса:
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Deep Learning",
       "summary": "We present a novel neural network architecture..."
     }'
   ```
   
   Ответ:
   ```json
   {
     "category": "cs.LG",
     "confidence": 0.95
   }
   ```

4. **Docker deployment:**
   ```bash
   # Сборка образа
   docker build -t arxiv-classifier .
   
   # Запуск контейнера
   docker run -p 8000:8000 \
     -v $(pwd)/train_artifacts:/app/train_artifacts \
     arxiv-classifier
   ```

## Структура проекта

```
mlops-arxiv-classifier/
├── arxiv_classifier/          # Основной код приложения
│   ├── commands.py            # CLI команды (train, infer, serve, etc.)
│   ├── api.py                 # FastAPI приложение
│   ├── data/                  # Модули для работы с данными
│   │   ├── downloader.py     # Скачивание данных с Kaggle
│   │   ├── preprocessing.py  # Предобработка данных
│   │   └── dataset.py        # PyTorch Dataset и DataModule
│   ├── models/                # Модели
│   │   ├── baseline.py       # TF-IDF baseline
│   │   └── distilbert_classifier.py  # DistilBERT модель
│   └── utils/                 # Утилиты
│       ├── logger.py         # Логирование
│       ├── mlflow.py         # MLflow интеграция
│       ├── git.py            # Git утилиты
│       └── export.py         # Экспорт моделей
├── configs/                   # Конфигурационные файлы (Hydra)
│   ├── config.yaml           # Главный конфиг
│   ├── model/                # Конфиги моделей
│   ├── training/              # Параметры обучения
│   ├── data/                 # Настройки данных
│   └── mlflow/               # Настройки MLflow
├── scripts/                   # Вспомогательные скрипты
│   ├── setup_dvc.sh          # Инициализация DVC
│   ├── download_artifacts.sh # Скачивание данных
│   ├── start_api.sh          # Запуск API
│   └── convert_to_onnx.sh    # Экспорт в ONNX
├── data/                      # Данные (игнорируется git, отслеживается DVC)
├── train_artifacts/           # Обученные модели и артефакты
├── dvc.yaml                   # DVC pipeline определение
├── dvc.lock                   # DVC lock файл (версии данных)
├── Dockerfile                 # Docker образ для API
└── pyproject.toml            # Зависимости проекта
```

## DVC Pipeline

Проект использует DVC для версионирования данных и воспроизводимости экспериментов:

```bash
# Запуск всего pipeline
uv run dvc repro train

# Запуск отдельных стадий
uv run dvc repro download
uv run dvc repro train_baseline
uv run dvc repro train

# Просмотр статуса
uv run dvc status

# Просмотр метрик
uv run dvc metrics show
```

## Команды

Основные CLI команды:

* `download` - скачать данные с Kaggle
* `train` - обучить DistilBERT модель
* `baseline` - обучить baseline модель
* `test` - протестировать обученную модель
* `infer` - выполнить инференс (CLI)
* `serve` - запустить REST API сервер
* `export_onnx` - экспортировать модель в ONNX
* `export_torchscript` - экспортировать модель в TorchScript

Все команды доступны через:
```bash
uv run python -m arxiv_classifier.commands <command>
```
