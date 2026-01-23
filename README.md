# Sentiment Analysis of Product Reviews — Modular ML/NLP Project

## Project Description

This project builds a robust sentiment classification system for product reviews using Python, ML, and NLP. It features a modular, extensible architecture based on the **Strategy Design Pattern**, enabling rapid experimentation and easy deployment of multiple models and feature extractors. The project is designed to be production-ready for large organizations and is an excellent learning resource for ML engineers.

---

## Table of Contents

1. [Objectives](#objectives)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [How It Works](#how-it-works)
5. [Testing](#testing)
6. [API Usage (FastAPI)](#api-usage-fastapi)
7. [Hands-On Roadmap](#hands-on-roadmap)
8. [Theory & Best Practices](#theory--best-practices)
9. [Evaluation & Metrics](#evaluation--metrics)
10. [Deployment & MLOps](#deployment--mlops)
11. [References & Further Study](#references--further-study)

---

## Objectives

- **Build** a sentiment classifier for product reviews (positive/negative).
- **Apply** ML and NLP techniques for real-world text data.
- **Use** the Strategy Pattern to easily swap models and feature extraction methods.
- **Structure** code professionally for team use, maintenance, and deployment.
- **Learn** end-to-end ML engineering: theory → code → evaluation → deployment.

---

## Project Structure

```
sentiment_analysis_project/
│
├── config/            # YAML/JSON for model, feature, data configs
├── data/              # raw/processed/external datasets
├── models/            # trained model artifacts (.pkl, .pt, etc.)
├── notebooks/         # ⚠️ IN PREPARATION: Jupyter notebooks for EDA, prototyping (not yet implemented)
├── src/
│   ├── data/          # loaders, preprocessors
│   ├── features/      # base & concrete feature extractors (Strategy Pattern)
│   ├── models/        # base & concrete models (Strategy Pattern)
│   ├── pipeline/      # context (SentimentPipeline), train/inference pipelines
│   ├── utils/         # metrics, visualization, logging
│   └── app/           # FastAPI/Streamlit serving
├── tests/             # unit/integration tests
├── requirements.txt   # dependencies
├── Makefile           # build/test/run commands
├── README.md          # this documentation
└── .gitignore
```

---

## Requirements

### Technical

- **Python 3.8+**
- **Libraries:** numpy, pandas, scikit-learn, matplotlib, (optional: PyTorch/TensorFlow for advanced models), nltk/spacy for NLP
- **OOP:** Use abstract base classes for all strategies (models, feature extraction)
- **Config:** Centralize all parameters in config files
- **Data:** Product reviews, labeled positive/negative (csv, json, etc.)

### ML/NLP

- **Text Preprocessing:** Cleaning, tokenization, stopword removal
- **Feature Extraction:** TF-IDF, Bag-of-Words, Word2Vec, BERT (extensible)
- **Models:** Naive Bayes, SVM, Logistic Regression, Random Forest, (optional: Deep Learning)
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
- **Strategy Pattern:** Easily swap algorithms and feature extractors

### Engineering

- **Testing:** Unit tests for every module
- **Artifact Management:** Save/load trained models
- **API/UI:** Serve predictions via FastAPI or Streamlit
- **Extensibility:** Add new strategies with minimal code changes
- **Versioning:** Use .gitignore for data/models; code tracked in git

---

## How It Works

1. **Data Loading:** Load product reviews from `data/raw/`.
2. **Preprocessing:** Clean and tokenize text (see `src/data/preprocessor.py`).
3. **Feature Extraction:** Select and apply feature extraction strategy (`src/features/`).
4. **Model Training:** Choose and train classifier strategy (`src/models/`).
5. **Evaluation:** Compute metrics on test set (`src/pipeline/evaluation.py`).
6. **Inference:** Predict sentiment for new reviews using trained model.
7. **Serving:** Expose model via API/UI (`src/app/`).

---

## Testing

The project includes comprehensive unit tests to ensure code quality and reliability. All tests are located in the `tests/` directory and use Python's built-in `unittest` framework.

### Available Test Modules

- **`test_preprocessor.py`** - Tests for text preprocessing and cleaning
- **`test_feature_extractor.py`** - Tests for feature extraction strategies (TF-IDF, BoW, etc.)
- **`test_model.py`** - Tests for model implementations and training
- **`test_pipeline.py`** - Tests for end-to-end pipeline workflows

### Running Tests

#### Run All Tests

```bash
# Using unittest discovery (from project root)
python -m unittest discover -s tests -p "test_*.py" -v

# Or simply
python -m unittest discover tests -v
```

#### Run Specific Test Files

```bash
# Test only the preprocessor
python -m unittest tests.test_preprocessor -v

# Test only feature extractors
python -m unittest tests.test_feature_extractor -v

# Test only models
python -m unittest tests.test_model -v

# Test only pipeline
python -m unittest tests.test_pipeline -v
```

#### Run Specific Test Classes or Methods

```bash
# Run a specific test class
python -m unittest tests.test_preprocessor.TestPreprocessor -v

# Run a specific test method
python -m unittest tests.test_preprocessor.TestPreprocessor.test_remove_punctuation -v
```

### Running Tests with Coverage

```bash
# Install coverage tool
pip install coverage

# Run tests with coverage
coverage run -m unittest discover tests -v

# View coverage report in terminal
coverage report

# Generate HTML coverage report
coverage html
```

Open `htmlcov/index.html` in your browser to view the detailed coverage report.

### Testing Best Practices

1. **Run tests before committing:** Always run the full test suite before pushing code
2. **Write tests for new features:** Add corresponding tests in `tests/` when adding new functionality
3. **Check coverage:** Aim for >80% code coverage using the coverage tool
4. **Use setUp/tearDown:** Leverage unittest's setUp and tearDown methods for reusable test fixtures
5. **Test edge cases:** Include tests for invalid inputs, empty data, and boundary conditions
6. **Follow naming conventions:** Test files should start with `test_`, test methods should start with `test_`

### Continuous Integration

Tests are automatically run on every push/PR via GitHub Actions. Check the `.github/workflows/` directory for CI configuration.

---

## API Usage (FastAPI)

The FastAPI service is implemented in `src/app/main.py` and runs on port **8080**.

### Run locally

- Run via Uvicorn:
  - `uvicorn src.app.main:app --host 0.0.0.0 --port 8080`
- Or run the module directly:
  - `python -m src.app.main`

### Endpoints

#### `GET /health`

- Example:
  - `curl http://localhost:8080/health`
- Response:
  - `{"status":"healthy","service":"Sentiment Analysis API","version":"1.1.2"}`

#### `POST /predict`

- Request schema:
  - `{"text": "review text"}`
- Example:
  - `curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"text\":\"This product is amazing!\"}"`
- Response schema:
  - `{"text": "review text", "rating": <score>, "sentiment": "<Positive|Neutral|Negative>"}`

Sentiment thresholds:

- Positive: rating ≥ 4
- Neutral: rating = 3
- Negative: rating ≤ 2

### Schemas

Request/response schemas are defined in `src/app/schemas.py`:

- `ReviewRequest(text: str)`
- `ReviewResponse(text: str, rating: float, sentiment: str)`

---

## Hands-On Roadmap

> **📝 NOTE:** The notebooks referenced below are currently **IN PREPARATION** and have not been implemented yet. The exploratory data analysis and prototyping notebooks will be added in future iterations.

| Step | Concept             | Action                                    | Reference                                                                                                  |
| ---- | ------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1    | **Theory**          | Understand sentiment analysis & ML basics | [SuperDataScience](https://www.superdatascience.com/blogs/the-ultimate-guide-to-regression-classification) |
| 2    | **Data**            | Explore/clean sample reviews in notebooks | `notebooks/01_data_exploration.ipynb`                                                                      |
| 3    | **OOP Design**      | Implement base & concrete strategies      | `src/features/base_feature_extractor.py`, `src/models/model_interface.py`                                  |
| 4    | **Modeling**        | Train/evaluate multiple classifiers       | `src/pipeline/train_pipeline.py`                                                                           |
| 5    | **Testing**         | Run unit tests to verify implementation   | `python -m unittest discover tests -v` (see [Testing](#testing) section)                                   |
| 6    | **Metrics**         | Visualize confusion matrix, F1, etc.      | `src/utils/metrics.py`                                                                                     |
| 7    | **Deployment**      | Serve model via API/UI                    | `src/app/`                                                                                                 |
| 8    | **Experimentation** | Swap strategies, tune hyperparameters     | `config/`                                                                                                  |

---

## Theory & Best Practices

- **Strategy Pattern:** Define abstract base classes (interfaces) for feature extraction and models; implement concrete strategies; context (pipeline) manages strategy selection and execution.
- **ML Pipeline:** Data loading → preprocessing → feature engineering → modeling → evaluation → deployment.
- **Trade-offs:** Model complexity vs. interpretability, training time vs. accuracy, data requirements vs. overfitting.
- **Extensibility:** Add new strategies in `src/features/` or `src/models/` without changing the pipeline.

---

## Evaluation & Metrics

- **Accuracy:** % of correct predictions.
- **Precision/Recall/F1:** Especially important for imbalanced datasets.
- **Confusion Matrix:** Visual breakdown of true/false positives/negatives.
- **Custom Metrics:** (Optional) ROC-AUC, PR curves.

---

## Deployment & MLOps

- **Artifact Management:** Save/load models in `models/`; use config for reproducible runs.
- **API Serving:** FastAPI app available in `src/app/` with `/health` and `/predict`.
- **CI/CD:** GitHub Actions workflows for CI (lint + tests + optional Docker build) and CD (deploy on Render after CI success).
- **Testing:** Run unit/integration tests in `tests/` before deployment.
- **Versioning:** Track code in git, data/models in .gitignore.

---

## References & Further Study

- [Strategy Pattern in Python](https://refactoring.guru/design-patterns/strategy/python/example)
- [Cookiecutter Data Science Project Structure](https://drivendata.github.io/cookiecutter-data-science/)
- [Structuring ML Projects (Andrew Ng)](https://www.deeplearning.ai/short-courses/structuring-machine-learning-projects/)
- [SuperDataScience: Regression vs Classification](https://www.superdatascience.com/blogs/the-ultimate-guide-to-regression-classification)
- [Neural Networks & Deep Learning (Michael Nielsen)](http://neuralnetworksanddeeplearning.com/chap1)
- [SVM Kernel Functions](https://data-flair.training/blogs/svm-kernel-functions/)

---

## Author & Mentorship

This project is guided by a mentor with 10+ years of experience in ML, NLP, and production AI, following a logical, scientific, intuitive, and hands-on teaching style.  
For questions, guidance, or code reviews, open an issue or discussion.

---

> **⚠️ CURRENT PROJECT STATUS:**
>
> - ✅ Core ML pipeline architecture implemented
> - 🚧 Notebooks for EDA and prototyping: **IN PREPARATION**
> - ✅ CI/CD pipeline (GitHub Actions + Render deploy hook)
> - ✅ API serving layer (FastAPI)

**Ready to get started? Clone this repo and dive into `src/` for the modular ML engineering foundation. Notebooks and deployment features coming soon!**
