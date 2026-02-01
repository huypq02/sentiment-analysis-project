# Sentiment Analysis of Product Reviews

> **A production-grade ML system built end-to-end to show how I’d ship a real sentiment classifier.**

## Overview

I built this as a full pipeline, not a notebook demo. It loads real review data, cleans it, extracts features, trains a model, and serves predictions over an API. The core idea is simple: every piece is a swap-in component (feature extractors, models, and pipelines), so I can compare approaches without rewriting the system.

It’s structured the way I build production ML: configuration-driven, testable, and easy to extend. I wanted this to feel like a repo you could hand to another engineer and they’d be productive immediately.

## Quick Start

```bash
# Clone, install, and train in ~5 minutes
git clone <repo>
cd sentiment-analysis-project
pip install -r requirements.txt
python src/pipeline/train_pipeline.py

# Start the API
uvicorn src.app.main:app --host 0.0.0.0 --port 8080

# Try a prediction
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text":"This product exceeded my expectations!"}'
```

## Table of Contents

1. [Why It’s Built This Way](#why-its-built-this-way)
2. [Project Structure](#project-structure)
3. [Architecture Notes](#architecture-notes)
4. [How the Pipeline Runs](#how-the-pipeline-runs)
5. [Testing](#testing)
6. [API Usage](#api-usage)
7. [Getting Started](#getting-started)
8. [Design Decisions](#design-decisions)
9. [Performance & Metrics](#performance--metrics)
10. [Deployment](#deployment)

---

## Why It’s Built This Way

- **Strategy pattern everywhere**: I don’t want to touch pipeline code when I swap TF‑IDF for BoW or Logistic Regression for Naive Bayes. That swap happens in config and the rest of the system stays put.
- **Real structure, not a tutorial**: There’s a clean separation between data, features, models, pipelines, and API. It keeps the codebase stable as it grows.
- **Tests that actually help**: I added unit tests early because I refactor a lot when experimenting. The tests keep the surface area honest.

---

## Project Structure

```
sentiment-analysis-project/
│
├── src/                           # Core application code
│   ├── data/
│   │   ├── data_loader.py        # Load & parse review datasets
│   │   └── preprocessor.py       # Text cleaning, tokenization
│   ├── features/
│   │   ├── base_feature_extractor.py  # Abstract interface
│   │   ├── bow_extractor.py           # Bag-of-Words strategy
│   │   └── tfidf_extractor.py         # TF-IDF strategy
│   ├── models/
│   │   ├── model_interface.py         # Abstract model interface
│   │   ├── naive_bayes_model.py       # Bernoulli NB classifier
│   │   └── logreg_model.py            # Logistic Regression classifier
│   ├── pipeline/
│   │   ├── sentiment_classifier.py    # Main orchestrator
│   │   ├── train_pipeline.py          # Training workflow
│   │   ├── predict_pipeline.py        # Inference workflow
│   │   └── evaluation.py              # Metrics & validation
│   ├── utils/
│   │   ├── logger.py                  # Centralized logging
│   │   ├── load_config.py             # Config file handling
│   │   └── metrics.py                 # Evaluation metrics
│   └── app/
│       ├── main.py                    # FastAPI application
│       └── schemas.py                 # Request/response models
│
├── tests/                         # Unit & integration tests
│   ├── test_preprocessor.py
│   ├── test_feature_extractor.py
│   ├── test_model.py
│   └── test_pipeline.py
│
├── config/                        # Configuration files
│   └── config.yaml               # Model, feature, data settings
│
├── data/
│   ├── raw/                      # Original review datasets
│   └── processed/                # Cleaned, ready-to-train data
│
├── models/                        # Trained artifacts
├── notebooks/                     # EDA & prototyping (coming soon)
├── Dockerfile                     # Container image
├── requirements.txt
├── Makefile
└── README.md
```

---

## Architecture Notes

I went with a Strategy pattern because I want to compare ideas quickly:

```python
# Strategy-style wiring
feature_strategy = TFIDFExtractor()  # or BOWExtractor()
model_strategy = LogisticRegression()  # or NaiveBayes()
classifier = SentimentClassifier(feature_strategy, model_strategy)
```

It’s intentionally boring. The goal is for each layer to do one thing well and stay replaceable.

### Tech Stack

- **Python 3.8+**
- **scikit-learn**
- **FastAPI**
- **unittest**
- **YAML**
- **Docker**

---

## How the Pipeline Runs

```
[Raw Reviews] → [Preprocess] → [Feature Extract] → [Model] → [Predict]
        ↓              ↓              ↓               ↓        ↓
     CSV         Tokenize,      TF-IDF or        Train    REST API
                            Remove Stop      Bag-of-Words   Classifier  Output
                            Words
```

### Stages

1. **Load** reviews from CSV.
2. **Clean** and tokenize text.
3. **Extract features** (TF‑IDF or BoW).
4. **Train** the model (LogReg or NB).
5. **Evaluate** with accuracy/F1/confusion matrix.
6. **Serve** predictions via FastAPI.

---

## Testing

I keep tests close to the code because I refactor frequently. Coverage is tracked and the pipeline has integration tests.

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific module tests
python -m unittest tests.test_model -v
python -m unittest tests.test_feature_extractor -v

# Run with coverage report
pip install coverage
coverage run -m unittest discover tests -v
coverage report
coverage html  # Open htmlcov/index.html in browser
```

---

## API Usage

### Start the Service

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8080
```

### Health Check

```bash
curl http://localhost:8080/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "Sentiment Analysis API",
  "version": "1.2.0"
}
```

### Make a Prediction

```bash
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "This product exceeded my expectations!"}'
```

**Response:**

```json
{
  "text": "This product exceeded my expectations!",
  "rating": 4.8,
  "sentiment": "Positive"
}
```

### Sentiment Thresholds

- **Positive**: confidence score ≥ 0.7
- **Neutral**: 0.4 ≤ score < 0.7
- **Negative**: score < 0.4

### Schemas

Defined in `src/app/schemas.py`:

```python
class ReviewRequest(BaseModel):
        text: str

class ReviewResponse(BaseModel):
        text: str
        rating: float
        sentiment: str
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
git clone https://github.com/yourusername/sentiment-analysis-project
cd sentiment-analysis-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training a Model

```bash
# Configure your model in config/config.yaml
python src/pipeline/train_pipeline.py
```

This will:

1. Load raw reviews from `data/raw/`
2. Preprocess text
3. Extract features (TF‑IDF or BoW)
4. Train selected model
5. Save model artifact to `models/`
6. Log metrics (accuracy, F1, confusion matrix)

### Making Predictions

```python
from src.pipeline.predict_pipeline import PredictPipeline

pipeline = PredictPipeline(config_path="config/config.yaml")
result = pipeline.predict("This product is fantastic!")
print(result)  # {"sentiment": "Positive", "confidence": 0.95}
```

### Experimenting with Different Strategies

Edit `config/config.yaml`:

```yaml
feature_extractor: "tfidf" # or "bow"
model: "logistic_regression" # or "naive_bayes"
```

---

## Design Decisions

### 1. Strategy Pattern

I don’t want feature or model swaps to cascade into pipeline edits. Using a strategy interface keeps the orchestration stable.

### 2. Centralized Configuration

All hyperparameters live in `config/config.yaml`. I avoid “hidden defaults” buried in code.

### 3. Tests from Day One

I refactor aggressively, so tests are my safety net. It keeps experiments honest.

### 4. Separation of Concerns

- **Data layer** loads and cleans
- **Feature layer** extracts signals
- **Model layer** trains/predicts
- **Pipeline layer** orchestrates
- **API layer** serves

---

## Performance & Metrics

### Benchmark Results

| Model               | Feature Extractor | Accuracy | Precision | Recall | F1-Score |
| ------------------- | ----------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | TF-IDF            | 0.89     | 0.87      | 0.91   | 0.89     |
| Naive Bayes         | Bag-of-Words      | 0.84     | 0.82      | 0.86   | 0.84     |
| Logistic Regression | Bag-of-Words      | 0.86     | 0.85      | 0.88   | 0.86     |

### Evaluation Metrics

```bash
# Generate detailed report
python -c "from src.utils.metrics import print_metrics; print_metrics('models/trained_model.pkl')"
```

Output includes:

- **Confusion Matrix**
- **Classification Report**
- **ROC-AUC**

---

## Deployment

### Local Development

```bash
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8080
```

### Docker

```bash
docker build -t sentiment-analyzer .
docker run -p 8080:8080 sentiment-analyzer
```

### Production (CI/CD)

GitHub Actions runs tests, builds the image, and publishes artifacts on push. See `.github/workflows/` for the details.

---

## Extending the Project

### Add a New Feature Extractor

```python
# src/features/word2vec_extractor.py
from src.features.base_feature_extractor import BaseFeatureExtractor

class Word2VecExtractor(BaseFeatureExtractor):
        def extract(self, text):
                # Your implementation here
                return vectors
```

Then update `config/config.yaml`:

```yaml
feature_extractor: "word2vec"
```

### Add a New Model

```python
# src/models/svm_model.py
from src.models.model_interface import ModelInterface

class SVMClassifier(ModelInterface):
        def train(self, X, y):
                # Your SVM training logic
                pass
```

Update the config and the pipeline stays unchanged.

---

## About This Project

I built this as a practical reference: how I’d structure a real ML sentiment system so it’s testable, swappable, and actually deployable. If you want to extend it, open a PR or issue.

---

### Project Status

- ✅ **Core pipeline**: Fully functional, tested, production-ready
- ✅ **API layer**: FastAPI with health checks and predictions
- ✅ **Testing**: >80% code coverage with CI/CD
- 🚧 **Notebooks**: EDA notebooks coming soon
- 🚀 **Next**: Advanced extractors (BERT, Word2Vec), ensemble models

---

**Pull requests welcome.** This is meant to be learned from and iterated on.
