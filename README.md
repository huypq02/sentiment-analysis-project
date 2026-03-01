# Sentiment Analysis of Product Reviews

A working ML pipeline for sentiment classification. Loads reviews, cleans text, extracts features with TF-IDF or Bag-of-Words, trains a classifier (Logistic Regression or Naive Bayes), and serves predictions via FastAPI. Swappable components throughout so you can try different feature extractors and models without touching the core pipeline logic.

Built to be testable and actually runnable, not a tutorial.

## Table of Contents

1.  [Quick Start](#quick-start)
2.  [Why This Structure](#why-this-structure)
3.  [What's Inside](#whats-inside)
4.  [How It Works](#how-it-works)
5.  [Running Tests](#running-tests)
6.  [API Endpoints](#api-endpoints)
7.  [Configuration](#configuration)
8.  [Notes](#notes)
9.  [Troubleshooting](#troubleshooting)
10. [License](#license)

## Quick Start

### 1) Install dependencies

```bash
pip install -e .
```

This installs the package in editable mode. Alternatively, if you prefer not to install the package:

```bash
pip install -r requirements.txt
export PYTHONPATH=src  # On Windows: set PYTHONPATH=src
```

### 2) Train the model

```bash
python -m sentimentanalysis.pipeline.training
```

### 3) Start the API

```bash
uvicorn sentimentanalysis.app.main:app --host 0.0.0.0 --port 8080
```

### 4) Make a prediction

PowerShell:

```powershell
curl -X POST "http://localhost:8080/v1/predictions" `
  -H "Content-Type: application/json" `
  -d '{"text":"The movie was terrible and I hated it"}'
```

The training script loads the movie reviews CSV, preprocesses the text, extracts features, trains on the data, and saves the model + extractor to `models/`. The API loads those artifacts on startup and serves predictions via versioned routes (e.g., `/v1/predictions`).

---

## Why This Structure

The main goal: I want to swap TF-IDF for Bag-of-Words, or swap Logistic Regression for Naive Bayes, without touching the actual pipeline orchestration code. Everything that can vary lives in one of four layers:

- **Data layer**: Loads CSVs, handles columns
- **Feature layer**: TF-IDF, Bag-of-Words (add more)
- **Model layer**: LogReg, Naive Bayes (add more)
- **Pipeline layer**: Orchestrates everything

Configuration tells the system which components to use. Unit tests guard against breakage when I refactor. The API is thin and just calls the pipeline.

---

## What's Inside

```
sentiment-analysis-project/
├── src/sentimentanalysis/
│   ├── app/                      # FastAPI application
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── __init__.py
│   ├── config/                   # Configuration management
│   │   ├── constants.py
│   │   ├── dataclasses.py
│   │   └── __init__.py
│   ├── data/                     # Data loading & preprocessing
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── __init__.py
│   ├── features/                 # Feature extraction (TF-IDF, BoW)
│   │   ├── base_feature_extractor.py
│   │   ├── bow_extractor.py
│   │   ├── tfidf_extractor.py
│   │   ├── factory.py
│   │   └── __init__.py
│   ├── models/                   # ML models (LogReg, Naive Bayes)
│   │   ├── model_interface.py
│   │   ├── logreg_model.py
│   │   ├── naive_bayes_model.py
│   │   ├── factory.py
│   │   └── __init__.py
│   ├── pipeline/                 # Training & prediction pipelines
│   │   ├── training.py
│   │   ├── prediction.py
│   │   ├── sentiment_pipeline.py
│   │   ├── evaluation.py
│   │   └── __init__.py
│   ├── utils/                    # Utilities
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── __init__.py
│   └── __init__.py
├── tests/                        # Unit tests
│   ├── test_preprocessor.py
│   ├── test_feature_extractor.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   └── __init__.py
├── config/                       # Configuration files
│   └── config.yaml
├── data/                         # Data directory
│   ├── raw/                      # Raw data (CSVs, etc.)
│   └── processed/                # Processed data
├── models/                       # Trained model artifacts
├── notebooks/                    # Jupyter notebooks (optional)
├── experiments/                  # Experiment scripts
├── mlops/                        # MLOps utilities
├── Dockerfile                    # Docker configuration
├── Makefile                      # Build automation
├── pyproject.toml                # Python package config
├── requirements.txt              # Dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## How It Works

```
Raw CSV → Preprocess → Feature Extract → Train → Evaluate → Serve
  ↓          ↓              ↓              ↓       ↓         ↓
Reviews  Tokenize       TF-IDF or      LogReg  Metrics  REST API
         + stopwords    Bag-of-Words   or NB   (F1, etc)
         remove negs
```

**Negation handling**: I keep words like "not", "never", "don't" in the text because sentiment depends on them. A preprocessing step removes normal English stopwords _except_ negations.

**Feature extraction**: Both TF-IDF and Bag-of-Words return sparse matrices (mostly zeros). Naive Bayes uses `StandardScaler(with_mean=False)` to handle sparse data without trying to center it.

**Training flow**:

1. Load the CSV
2. Preprocess text (tokenize, remove stopwords except negations)
3. 80/20 train-test split
4. Fit the feature extractor on train data
5. Transform both train and test
6. Train the model
7. Evaluate on test set
8. Save model + extractor as pickle files

The API loads those pickle files on startup and uses them for inference.

---

## Running Tests

Basic unit tests using unittest. They check that preprocessing works, feature extraction returns the right shapes, and models can be trained/evaluated. All use dummy data and are quick.

```bash
python -m unittest discover tests -v
python -m unittest tests.test_model -v

# With coverage
coverage run -m unittest discover tests -v
coverage report
coverage html
```

---

## API Endpoints

### Health Check

```bash
curl http://localhost:8080/v1/health
```

Response:

```json
{
  "status": "healthy",
  "service": "Sentiment Analysis API",
  "version": "1.2.0"
}
```

### Predict

```bash
curl -X POST http://localhost:8080/v1/predictions \
    -H "Content-Type: application/json" \
    -d '{"text": "The movie are too bad!"}'
```

Response:

```json
{
  "text": "The movie are too bad!",
  "sentiment": "negative"
}
```

---

## Configuration

Edit `config/config.yaml` to specify which dataset to load and where to save artifacts:

```yaml
dataset:
  raw_dir: "data/raw"
  file: "movie_reviews_imdb.csv"

models:
  dir: "models"
  model: "models/sentiment_logreg.pkl"
  extractor: "models/tfidf_extractor.pkl"
```

The `training.py` script reads this config, loads the CSV, trains, and saves. To swap between TF-IDF and Bag-of-Words, or between LogReg and Naive Bayes, you'd modify the factory calls in `training.py` or refactor to make it config-driven.

---

## Notes

- Prefer module execution (`python -m sentimentanalysis...`) over direct file paths.
- API import path is `sentimentanalysis.app.main:app`.
- Model and extractor artifact locations are controlled by `config/config.yaml`.
- For clean imports, install in editable mode: `pip install -e .`. Alternatively, set `PYTHONPATH=src` and run from the project root.

---

## Troubleshooting

- **Port already in use (`[Errno 10048]`)**
  - Another process is using port `8080`.
  - Start API on a different port:

    ```bash
    uvicorn sentimentanalysis.app.main:app --host 0.0.0.0 --port 8081
    ```

- **`ModuleNotFoundError: No module named 'sentimentanalysis'`**
  - Install the package in editable mode:

    ```bash
    pip install -e .
    ```

  - Or, run from the project root and set `PYTHONPATH`:

    ```bash
    export PYTHONPATH=src  # On Windows: set PYTHONPATH=src
    ```

- **404 on prediction endpoint**
  - Use `POST /v1/predictions` (not `/predict` or `/predictions`).

---

## License

MIT License. See LICENSE for details.
