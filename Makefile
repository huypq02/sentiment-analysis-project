.PHONY: help install train predict evaluate serve test clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install package in development mode"
	@echo "  make train         - Train the sentiment analysis model"
	@echo "  make evaluate      - Evaluate the trained model on test data"
	@echo "  make predict TEXT='your text' - Make a prediction"
	@echo "  make serve         - Start the FastAPI server"
	@echo "  make test          - Run tests with coverage"
	@echo "  make clean         - Remove build artifacts and cache"

install:
	pip install -e .

train:
	python -m sentimentanalysis.cli train

evaluate:
	python -m sentimentanalysis.cli evaluate

predict:
	python -m sentimentanalysis.cli predict "$(TEXT)"

serve:
	uvicorn sentimentanalysis.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest --cov=sentimentanalysis --cov-report=html --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
