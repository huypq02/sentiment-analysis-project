FROM python:3.13
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; \
    nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('punkt_tab', download_dir='/usr/local/share/nltk_data'); \
    nltk.download('stopwords', download_dir='/usr/local/share/nltk_data')"

# Copy in the source code
COPY src ./src
COPY config ./config
COPY data ./data

EXPOSE 8080

# Setup an app user so the container doesn't run as the root user
RUN useradd app
USER app

# Set NLTK data path so it finds the downloaded data
ENV NLTK_DATA=/usr/local/share/nltk_data

CMD [ "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
