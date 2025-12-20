from fastapi import FastAPI
from src.pipeline.predict_pipeline import predict_main
from .schemas import ReviewRequest, ReviewResponse

app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "1.0.0"
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

    return ReviewResponse(text=text, 
                          rating=rating, 
                          sentiment=sentiment)