# Sentiment Analysis of Product Reviews

A working ML pipeline for sentiment classification. Loads reviews, cleans text, extracts features with TF-IDF or Bag-of-Words, trains a classifier (Logistic Regression or Naive Bayes), and serves predictions via FastAPI. Swappable components throughout so you can try different feature extractors and models without touching the core pipeline logic.

Built to be testable and actually runnable, not a tutorial.

## Quick Start

```bash
pip install -r requirements.txt
python src/pipeline/training.py

# Start the API
uvicorn src.app.main:app --host 0.0.0.0 --port 8080

# Make a prediction
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text":"The book was terrible and I hated it"}'
```

The training script loads the book reviews CSV, preprocesses the text, extracts features, trains on the data, and saves the model + extractor to `models/`. The API loads those artifacts on startup and serves predictions.

## Table of Contents

1. [Why This Structure](#why-this-structure)
2. [What's Inside](#whats-inside)
3. [How It Works](#how-it-works)
4. [Running Tests](#running-tests)
5. [API Endpoints](#api-endpoints)
6. [Configuration](#configuration)
7. [Extending It](#extending-it)

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
│
├── src/
│   ├── data/
│   │   ├── data_loader.py         # Reads CSV
│   │   └── preprocessor.py        # Lowercase, remove punctuation, tokenize, drop stopwords
│   ├── features/
│   │   ├── base_feature_extractor.py
│   │   ├── bow_extractor.py       # Bag-of-Words
│   │   └── tfidf_extractor.py     # TF-IDF (currently used)
│   ├── models/
│   │   ├── model_interface.py     # Abstract base
│   │   ├── naive_bayes_model.py   # MultinomialNB
│   │   └── logreg_model.py        # LogisticRegression
│   ├── pipeline/
│   │   ├── training.py            # Load → Preprocess → Extract → Train → Save
│   │   ├── prediction.py          # Load model → Preprocess → Extract → Predict
│   │   └── evaluation.py          # Compute metrics
│   ├── app/
│   │   ├── main.py                # FastAPI server
│   │   └── schemas.py             # Pydantic models
│   └── utils/
│       ├── logger.py
│       └── load_config.py
│
├── tests/
│   ├── test_preprocessor.py
│   ├── test_feature_extractor.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── config/
│   └── config.yaml                # Points to data, model paths, selects components
│
├── data/
│   ├── raw/                       # CSVs go here
│   └── processed/                 # Cleaned data (if needed)
│
├── models/                        # Trained artifacts
├── requirements.txt
├── Dockerfile
└── README.md
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
curl http://localhost:8080/health
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
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "These books are too bad!"}'
```

Response:

```json
{
  "text": "These books are too bad!",
  "rating": 1,
  "sentiment": "Negative"
}
```

Ratings come from the model output (1-5 range). Sentiment is assigned by `rating_to_sentiment()` which maps ratings to "Positive", "Neutral", or "Negative" based on thresholds.

---

## Configuration

Edit `config/config.yaml` to specify which dataset to load and where to save artifacts:

```yaml
dataset:
  raw_dir: "data/raw"
  file: "book_reviews_sample.csv"

models:
  dir: "models"
  model: "models/sentiment_logreg.pkl"
  extractor: "models/tfidf_extractor.pkl"
```

The `training.py` script reads this config, loads the CSV, trains, and saves. To swap between TF-IDF and Bag-of-Words, or between LogReg and Naive Bayes, you'd modify the factory calls in `training.py` or refactor to make it config-driven.

---

## Extending It

### Add a New Feature Extractor

1. Create `src/features/word2vec_extractor.py`
2. Subclass `BaseFeatureExtractor` and implement `fit()` and `transform()`
3. Add it to the extractor factory
4. Update training script to use it

```python
from src.features.base_feature_extractor import BaseFeatureExtractor

class Word2VecExtractor(BaseFeatureExtractor):
    def fit(self, sentences):
        # Train Word2Vec on sentences
        pass

    def transform(self, sentences):
        # Return vectors
        pass
```

### Add a New Model

1. Create `src/models/svm_model.py`
2. Subclass `SentimentModel` and implement `train()`, `predict()`, `evaluate()`
3. Add it to the model factory
4. Update training script to use it

The pipeline stays the same. Both training and prediction flow through the same orchestration—no changes needed.

---

## Notes

- **Sparse matrices**: Feature extractors return sparse matrices. Make sure models can handle them. Naive Bayes uses `StandardScaler(with_mean=False)` for this reason.
- **Stopwords**: Negation words are kept intentionally because they flip sentiment. The preprocessor removes English stopwords _except_ negations.
- **Persistence**: Models and extractors are saved as pickle files. If you change code, old pickles won't deserialize correctly.
- **Logging**: Each module has a logger. Output goes to console; can be redirected to files if needed.

---

## Performance

On the sample data, typical results:

- TF-IDF + LogReg: ~85–90% accuracy
- Bag-of-Words + Naive Bayes: ~80–85% accuracy

Metrics are logged to `models/metrics.txt` after each training run.

---

## Deployment

### Local

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t sentiment-analyzer .
docker run -p 8080:8080 sentiment-analyzer
```

---

That's it. No hidden magic, no sprawling config files. The code does what it says it does.
