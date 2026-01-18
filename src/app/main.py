from fastapi import FastAPI
import uvicorn
from src.pipeline import predict_main
from .schemas import ReviewRequest, ReviewResponse

app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "1.1.1",
    }


@app.post("/predict")
async def prediction(request: ReviewRequest):
    text = request.text

    rating = predict_main(text=text)

    # TODO: consider adjusting the type of sentiment based on the rating's result
    if rating >= 4:
        sentiment = "Positive"
    elif rating <= 2:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return ReviewResponse(text=text, rating=rating, sentiment=sentiment)


if __name__ == "__main__":
    config = uvicorn.Config("src.app.main:app", port=8080, log_level="info")
    server = uvicorn.Server(config)
    server.run()
