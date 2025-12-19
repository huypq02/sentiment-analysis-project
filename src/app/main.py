from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "1.0.0"
    }