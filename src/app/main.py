from fastapi import FastAPI
import uvicorn
from src.pipeline import predict
from src.utils import rating_to_sentiment
from .schemas import ReviewRequest, ReviewResponse

app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "1.2.0",
    }


@app.post("/predict")
async def prediction(request: ReviewRequest):
    text = request.text

    rating = predict(text=text)
    sentiment = rating_to_sentiment(rating)

    return ReviewResponse(text=text, rating=rating, sentiment=sentiment)


if __name__ == "__main__":
    config = uvicorn.Config("src.app.main:app", port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()
