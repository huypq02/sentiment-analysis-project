.PHONY: help install train predict evaluate serve test clean

help:
	@echo Available commands:
	@echo   make install                                 - Install package in development mode
	@echo   make train ARGS='--debug'                    - Train model with optional CLI args
	@echo   make evaluate ARGS='--debug --test-size 0.3' - Evaluate with optional CLI args
	@echo   make predict TEXT='your text' ARGS='--debug' - Predict with optional CLI args
	@echo   make serve                                   - Start the FastAPI server
	@echo   make test                                    - Run tests with coverage
	@echo   make clean                                   - Remove build artifacts and cache

install:
	pip install -e .

train:
	python -m sentimentanalysis.cli train $(ARGS)

evaluate:
	python -m sentimentanalysis.cli evaluate $(ARGS)

predict:
	python -m sentimentanalysis.cli predict "$(TEXT)" $(ARGS)

serve:
	uvicorn sentimentanalysis.app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest --cov=sentimentanalysis --cov-report=html --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
