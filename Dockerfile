FROM python:3.13
WORKDIR /usr/local/app/sentimentanalysis

# Install package in editable mode
COPY pyproject.toml ./
COPY README.md ./
RUN pip install -e .

# Copy in the source code
COPY src ./src
COPY config ./config
COPY data ./data

EXPOSE 8080

# Setup an app user so the container doesn't run as the root user
RUN useradd -m app && chown -R app:app /usr/local/app/sentimentanalysis

USER app

CMD [ "uvicorn", "sentimentanalysis.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
