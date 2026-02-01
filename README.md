# Sentiment Analysis of Product Reviews

> **A production-grade ML system demonstrating scalable architecture, clean code, and real-world ML engineering practices.**

## Overview

Built a sentiment classifier for product reviews from the ground up-handling the full ML lifecycle from data ingestion to API deployment. The system uses a **modular, strategy-based architecture** that lets you swap models and feature extractors without touching the pipeline. Think of it as building Lego blocks instead of a monolith: each component is independent, testable, and replaceable.

This isn't a toy notebook project-it's structured for a real team to collaborate on, extend, and deploy confidently.

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

1. [What Makes This Different](#what-makes-this-different)
2. [Project Structure](#project-structure)
3. [The Architecture](#the-architecture)
4. [How It Works](#how-it-works)
5. [Testing & Quality](#testing--quality)
6. [API Usage](#api-usage)
7. [Getting Started](#getting-started)
8. [Key Design Decisions](#key-design-decisions)
9. [Performance & Metrics](#performance--metrics)
10. [Deployment](#deployment)

---

## What Makes This Different

**Strategy Pattern in Action**: Instead of hardcoding models and feature extractors, this project treats them as pluggable strategies. Want to compare Naive Bayes vs. Logistic Regression? Swap one config line. Need to upgrade from TF-IDF to word embeddings? Add a new extractor class without touching existing code.

**Production-Grade Structure**: Unlike tutorial projects, this follows real ML engineering practices:

- Centralized configuration management
- Comprehensive unit tests (>80% coverage)
- Separated concerns: data, features, models, pipelines
- CI/CD pipeline with automated testing
- API-ready with request/response schemas

**Built for Teams**: Clear separation of responsibilities means data scientists, ML engineers, and backend engineers can work in parallel without stepping on each other's toes.

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
│   │   ├── sentiment_classifier.py    # Main orchestrator (context)
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

## The Architecture

### Design Pattern: Strategy

The **Strategy Pattern** is the backbone of this project. Here's why it matters:

```python
# Before: Monolithic, rigid
if model_type == "naive_bayes":
    model = NaiveBayesClassifier()
elif model_type == "logistic_regression":
    model = LogisticRegressionClassifier()
# ... painful to extend

# After: Plug-and-play strategies
feature_strategy = TFIDFExtractor()  # or BOWExtractor()
model_strategy = LogisticRegression()  # or NaiveBayes()
classifier = SentimentClassifier(feature_strategy, model_strategy)
# Change one line, not the whole pipeline
```

### Key Components

| Component              | Purpose                                                  | Extensibility                                           |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| **Feature Extractors** | Convert text → numerical vectors                         | Add new extractors by inheriting `BaseFeatureExtractor` |
| **Models**             | Classify sentiments                                      | Add new classifiers by inheriting `ModelInterface`      |
| **Pipeline**           | Orchestrate: load → preprocess → extract → train/predict | Modify workflow logic without changing strategies       |
| **API Layer**          | Expose predictions via HTTP                              | FastAPI handles requests, schema validation             |

### Tech Stack

- **Python 3.8+** for core ML
- **scikit-learn** for models & feature engineering
- **FastAPI** for REST API
- **unittest** for testing, coverage tool for metrics
- **YAML** for configuration management
- **Docker** for containerization & deployment

---

## How It Works

The pipeline flows through distinct, testable stages:

```
[Raw Reviews] → [Preprocess] → [Feature Extract] → [Model] → [Predict]
    ↓              ↓              ↓               ↓        ↓
   CSV         Tokenize,      TF-IDF or        Train    REST API
              Remove Stop      Bag-of-Words   Classifier  Output
              Words
```

### In Detail

1. **Data Ingestion** (`src/data/data_loader.py`): Load labeled reviews from CSV.
2. **Text Preprocessing** (`src/data/preprocessor.py`): Clean (lowercase, remove special chars), tokenize, strip stopwords.
3. **Feature Extraction** (pluggable): Convert text to vectors:
   - **TF-IDF**: Weighted term frequencies (fast, interpretable)
   - **Bag-of-Words**: Simple word counts (baseline)
   - _(Add more: Word2Vec, BERT embeddings, etc.)_
4. **Model Training** (pluggable):
   - **Logistic Regression**: Fast, linear, highly interpretable
   - **Naive Bayes**: Probabilistic baseline
   - _(Add more: SVM, Random Forest, Neural Nets)_
5. **Evaluation**: Compute accuracy, precision, recall, F1, confusion matrix.
6. **Inference**: Load trained model + apply same preprocessing → predict on new reviews.
7. **API Serving**: FastAPI exposes `/predict` endpoint for real-time classification.

---

## Testing & Quality

### Why This Matters

Quality code is maintainable code. This project treats testing as a first-class citizen:

- **Unit tests** for each module (preprocessor, extractors, models)
- **Integration tests** for the full pipeline
- **Coverage tracking** (aiming for >80%)
- **Continuous Integration** (GitHub Actions runs tests on every push)

### Running Tests

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

### Test Coverage

```
Name                                    Stmts   Miss  Cover
────────────────────────────────────────────────────────────
src/data/preprocessor.py                  15      2    87%
src/features/tfidf_extractor.py           18      1    94%
src/models/logreg_model.py                22      3    86%
src/pipeline/sentiment_classifier.py      25      2    92%
────────────────────────────────────────────────────────────
TOTAL                                    120     12    90%
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

Defined in `src/app/schemas.py` with Pydantic validation:

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
- Basic understanding of ML (optional, but helpful)

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
3. Extract features (TF-IDF or BoW)
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

That's it. No code changes needed. This is the power of the Strategy Pattern.

---

## Key Design Decisions

### 1. Why Strategy Pattern?

The Strategy Pattern eliminates the "feature-selection-gets-messy" problem. Instead of:

```python
if use_tfidf:
    vectors = tfidf(text)
else:
    vectors = bow(text)
```

You get:

```python
feature_extractor = TFIDFExtractor()  # swap to BOWExtractor() in one line
classifier.extract_features(text, feature_extractor)
```

This scales: Add a new extractor without modifying the pipeline.

### 2. Centralized Configuration

All hyperparameters live in `config/config.yaml`, not scattered through code. Change model in config, not in `main.py`.

### 3. Testing from Day One

Every module has corresponding tests. This makes refactoring safe and catches regressions early.

### 4. Separation of Concerns

- **Data layer** handles loading/preprocessing
- **Feature layer** extracts signals
- **Model layer** makes predictions
- **Pipeline layer** orchestrates everything
- **API layer** exposes to the world

Each can evolve independently.

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

- **Confusion Matrix**: Visual breakdown of predictions
- **Classification Report**: Precision, Recall, F1 per class
- **ROC-AUC**: Model discrimination ability

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

GitHub Actions automatically:

1. Runs tests on every push
2. Generates coverage reports
3. Builds Docker image
4. Deploys to cloud (Render, AWS, GCP, etc.)

See `.github/workflows/` for configuration.

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

Again, just update the config. The pipeline doesn't change.

---

## Learning Resources

- [Design Patterns in Python](https://refactoring.guru/design-patterns/strategy/python/example)
- [ML Project Structure](https://drivendata.github.io/cookiecutter-data-science/)
- [Andrew Ng: Structuring ML Projects](https://www.deeplearning.ai/short-courses/structuring-machine-learning-projects/)
- [Text Classification Best Practices](https://nlp.stanford.edu/pubs/crosslingual_embeddings.pdf)

---

## About This Project

Built as a professional learning resource to demonstrate how real ML systems are structured, tested, and deployed. It's not a toy—it's how teams actually build this stuff.

**Questions?** Open an issue, start a discussion, or reach out. This is a living project; contributions and feedback are welcome.

---

### Project Status

- ✅ **Core pipeline**: Fully functional, tested, production-ready
- ✅ **API layer**: FastAPI with health checks and predictions
- ✅ **Testing**: >80% code coverage with CI/CD
- 🚧 **Notebooks**: EDA notebooks coming soon
- 🚀 **Next**: Advanced extractors (BERT, Word2Vec), ensemble models

---

**Pull requests welcome.** This is meant to be learned from and iterated on.
