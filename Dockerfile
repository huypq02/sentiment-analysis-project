FROM python:3.13
WORKDIR /usr/local/app

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY src ./src
COPY config ./config
COPY data ./data

EXPOSE 8080

# Setup an app user so the container doesn't run as the root user
RUN useradd app && chown -R app:app /usr/local/app

USER app

CMD [ "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
