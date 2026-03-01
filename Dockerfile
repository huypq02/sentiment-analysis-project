FROM python:3.13
WORKDIR /usr/local/app/sentimentanalysis
ENV PORT=8080
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy source code and config first (required by pyproject.toml)
COPY src ./src
COPY config ./config
COPY data ./data
COPY models ./models
COPY README.md ./

# Install package and dependencies in editable mode
COPY pyproject.toml ./
RUN pip install -e . && \
    python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

EXPOSE 8080

# Setup an app user so the container doesn't run as the root user
RUN useradd -m app && chown -R app:app /usr/local/app/sentimentanalysis /usr/local/share/nltk_data

USER app

CMD ["sh", "-c", "uvicorn sentimentanalysis.app.main:app --host 0.0.0.0 --port ${PORT}"]
